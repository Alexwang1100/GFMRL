import numpy as np
import random
import math
import os, sys, random, time
import logging
sys.path.append("../")

from objects import *
from utilities import *


class MultiAgentEnv:
    def __init__(self, num_agents, num_orders):
        self.num_agents = num_agents
        self.num_orders = num_orders

        # Order states: location (lat, long), floor, deadline, volume, weight, time
        self.order_states = np.zeros((num_orders, 6))
        # Agent states: location (lat, long), remaining tasks, fatigue, familiarity, time
        self.agent_states = np.zeros((num_agents, 5))
        self.next_order_count = np.zeros(num_orders)

        # Random initial prices for each order
        self.order_prices = np.random.uniform(0.5, 2, self.num_orders)
        # Each agent's price for the order they chose
        self.agent_prices = np.random.uniform(0.5, 2, (
        self.num_agents, self.num_orders))  # 2D array for storing prices for nearby orders

        # New class attribute to store indices of successful agents
        self.successful_agents = []
        self.complaint_orders = []

        self.alpha = 0.5  # Weight for profit
        self.beta = 0.3  # Weight for completion rate
        self.gamma = 0.2  # Weight for customer satisfaction
        self.N = 2.0  # N Kilometers

    def reset(self):
        # Reset the list during environment reset
        self.order_states = np.zeros((self.num_orders, 6))
        self.agent_states = np.zeros((self.num_agents, 5))
        self.order_prices = np.random.uniform(0.5, 2, self.num_orders)
        self.agent_prices = np.random.uniform(0.5, 2, (self.num_agents, self.num_orders))
        self.next_order_count = np.zeros(self.num_orders)
        self.successful_agents.clear()
        self.complaint_orders.clear()
        return self.order_states, self.agent_states

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        successful_agents = []

        for i, action in enumerate(actions):
            nearby_orders = self.get_nearby_orders(i)
            if not nearby_orders:
                continue

            self.agent_prices[i, nearby_orders] = action  # Setting the price

            for order_idx in nearby_orders:
                successful_agent_for_this_order = np.argmin(self.agent_prices[:, order_idx])
                if successful_agent_for_this_order == i:
                    successful_agents.append(i)
                    reward = self.calculate_reward(i, order_idx)
                    rewards[i] += reward  # Accumulate rewards for multiple nearby orders

        return rewards

    def get_nearby_orders(self, agent_idx):
        agent_loc = self.agent_states[agent_idx, :2]
        nearby_orders = []
        for i, order in enumerate(self.order_states):
            order_loc = order[:2]
            if self.is_within_NKM(agent_loc, order_loc):
                nearby_orders.append(i)
        return nearby_orders

    def is_within_NKM(self, loc1, loc2):
        # loc1/loc2: (lat, lon)
        lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
        lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
        R = 6371.0
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance <= self.N

    def calculate_reward(self, agent_idx, order_idx):
        profit = self.calculate_profit(agent_idx, order_idx)
        completion_rate = self.calculate_completion_rate(agent_idx, order_idx)
        customer_satisfaction = self.calculate_customer_satisfaction(agent_idx, order_idx)

        reward = self.alpha * profit + self.beta * completion_rate + self.gamma * customer_satisfaction
        return reward

    def update_states(self, agent_idx, order_idx):
        self.order_states[order_idx] = 1
        self.agent_states[agent_idx] = order_idx

    def calculate_profit(self, agent_idx, order_idx):
        if len(self.successful_agents) == 0:
            return 0

        successful_agents = np.array(self.successful_agents)
        total_profit = np.sum(self.agent_prices[successful_agents == agent_idx, order_idx])
        num_successful_agents = np.sum(successful_agents == agent_idx)

        avg_profit_per_successful_agent = total_profit / num_successful_agents if num_successful_agents > 0 else 0
        return avg_profit_per_successful_agent

    def calculate_completion_rate(self, agent_idx, order_idx):
        total_tasks = self.num_orders
        completed_tasks = np.sum(self.order_states)
        completion_rate = completed_tasks / total_tasks
        return completion_rate

    def calculate_customer_satisfaction(self, agent_idx, order_idx):
        # calculating customer satisfaction
        num_complaints = len(self.complaint_orders)
        num_total_orders = self.num_orders
        complaint_rate = num_complaints / num_total_orders if num_total_orders > 0 else 0
        customer_satisfaction = 1 - complaint_rate
        return customer_satisfaction


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


# Historical data: list of (state, action, reward, next_state)
def load_historical_data(replay_buffer, historical_data):
    for data in historical_data:
        state, action, reward, next_state = data
        replay_buffer.push(state, action, reward, next_state)


def pretrain(model, replay_buffer, optimizer, batch_size=64):
    # 10000 experiences
    replay_buffer = ReplayBuffer(10000)
    state, action, reward, next_state = replay_buffer.sample(batch_size)
    # Implement forward pass and loss calculation using state, action, reward
    loss = model(state, action) - reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Load historical data into replay buffer
# historical_data = [...] # This would be your preprocessed data
# load_historical_data(replay_buffer, historical_data)


# class Courier:
#     def __init__(self, id):
#         self.id = id
#         self.location = (random.randint(0, 10), random.randint(0, 10))
#         self.tasks = []
#         self.available_time = random.randint(1, 8)
#         self.success_rate = random.uniform(0.5, 1.0)
#
#
# class Task:
#     def __init__(self, id):
#         self.id = id
#         self.type = random.choice(["pickup", "delivery"])
#         self.location = (random.randint(0, 10), random.randint(0, 10))
#         self.eta = random.randint(1, 5)
#
#
# class AOI:
#     def __init__(self, id):
#         self.id = id
#         self.tasks = []
#         self.available_couriers = []
#         self.weather = random.choice(["sunny", "rainy", "cloudy"])
#         self.traffic = random.choice(["light", "moderate", "heavy"])
#
#
# class MultiAgentEnvironment:
#     def __init__(self, num_couriers, num_tasks, num_aois):
#         self.couriers = [Courier(i) for i in range(num_couriers)]
#         self.tasks = [Task(i) for i in range(num_tasks)]
#         self.aois = [AOI(i) for i in range(num_aois)]
#
#         # Assign tasks to AOIs
#         for task in self.tasks:
#             aoi = random.choice(self.aois)
#             aoi.tasks.append(task)
#
#         # Assign couriers to AOIs
#         for courier in self.couriers:
#             aoi = random.choice(self.aois)
#             aoi.available_couriers.append(courier)
#
#     def reward(self, profit, completion_rate, satisfaction):
#         alpha = 0.5
#         beta = 0.3
#         gamma = 0.2
#         return alpha * profit + beta * completion_rate + gamma * satisfaction
#
#     def step(self):
#         total_profit = 0
#         total_tasks = len(self.tasks)
#         completed_tasks = 0
#         total_satisfaction = 0
#
#         for aoi in self.aois:
#             for task in aoi.tasks:
#                 if len(aoi.available_couriers) > 0:
#                     courier = random.choice(aoi.available_couriers)
#                     courier.tasks.append(task)
#                     aoi.tasks.remove(task)
#                     completed_tasks += 1
#                     total_profit += 10  # Assume a fixed profit for each completed task
#                     total_satisfaction += courier.success_rate
#
#         completion_rate = completed_tasks / total_tasks
#         avg_satisfaction = total_satisfaction / completed_tasks if completed_tasks > 0 else 0
#         return self.reward(total_profit, completion_rate, avg_satisfaction)
#
#
# # Create the environment with 5 couriers, 10 tasks, and 3 AOIs
# env = MultiAgentEnvironment(5, 10, 3)
# reward = env.step()
# print(f"Reward: {reward}")

class CityReal:
    '''A real city is consists of M*N grids '''

    def __init__(self, mapped_matrix_int, order_num_dist, idle_courier_dist_time, idle_courier_location_mat,
                 order_time_dist, order_price_dist,
                 l_max, M, N, n_side, probability=1.0 / 28, real_orders="", onoff_courier_location_mat="",
                 global_flag="global", time_interval=10):
        """
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        :param order_num_dist: 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                            node_id1 is node the index in self.nodes
        :param idle_courier_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle couriers in
        the city at each time
        :param idle_courier_location_mat: 144 x num_valid_grids matrix.
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]
        :param onoff_courier_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online courier number - offline courier number
        onoff_courier_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :return:
        """
        # City.__init__(self, M, N, n_side, time_interval)
        self.M = M  # row numbers
        self.N = N  # column numbers
        self.nodes = [Node(i) for i in xrange(M * N)]  # a list of nodes: node id start from 0
        self.couriers = {}  # courier[courier_id] = courier_instance  , courier_id start from 0
        self.n_couriers = 0  # total idle number of couriers. online and not on service.
        self.n_offline_couriers = 0  # total number of offline couriers.
        self.construct_map_simulation(M, N, n_side)
        self.city_time = 0
        # self.idle_courier_distribution = np.zeros((M, N))
        self.n_intervals = 1440 / time_interval
        self.n_nodes = self.M * self.N
        self.n_side = n_side
        self.order_response_rate = 0

        self.RANDOM_SEED = 0

        self.l_max = l_max  # Start from 1. The max number of layers an order can across.
        assert l_max <= M - 1 and l_max <= N - 1
        assert 1 <= l_max <= 9  # Ignore orders less than 10 minutes and larger than 1.5 hours

        self.target_grids = []
        self.n_valid_grids = 0  # num of valid grid
        self.nodes = [None for _ in np.arange(self.M * self.N)]
        self.construct_node_real(mapped_matrix_int)
        self.mapped_matrix_int = mapped_matrix_int

        self.construct_map_real(n_side)
        self.order_num_dist = order_num_dist
        self.distribution_name = "Poisson"
        self.idle_courier_dist_time = idle_courier_dist_time
        self.idle_courier_location_mat = idle_courier_location_mat

        self.order_time_dist = order_time_dist[:l_max] / np.sum(order_time_dist[:l_max])
        self.order_price_dist = order_price_dist

        target_node_ids = []
        target_grids_sorted = np.sort(mapped_matrix_int[np.where(mapped_matrix_int > 0)])
        for item in target_grids_sorted:
            x, y = np.where(mapped_matrix_int == item)
            target_node_ids.append(ids_2dto1d(x, y, M, N)[0])
        self.target_node_ids = target_node_ids
        # store valid note id. Sort by number of orders emerged. descending.

        self.node_mapping = {}
        self.construct_mapping()

        self.real_orders = real_orders  # 4 weeks' data
        # [[92, 300, 143, 2, 13.2],...] origin grid, destination grid, start time, end time, price.

        self.p = probability  # sample probability
        self.time_keys = [int(dt.strftime('%H%M')) for dt in
                          datetime_range(datetime(2022, 8, 1, 0), datetime(2022, 9, 2, 0),
                                         timedelta(minutes=time_interval))]
        self.day_orders = []  # one day's order.

        self.onoff_courier_location_mat = onoff_courier_location_mat

        # Stats
        self.all_grids_on_number = 0  # current online # couriers.
        self.all_grids_off_number = 0

        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))
        self.global_flag = global_flag
        self.weights_layers_neighbors = [1.0, np.exp(-1), np.exp(-2)]

    def construct_map_simulation(self, M, N, n):
        """Connect node to its neighbors based on a simulated M by N map
            :param M: M row index matrix
            :param N: N column index matrix
            :param n: n - sided polygon
        """
        for idx, current_node in enumerate(self.nodes):
            if current_node is not None:
                i, j = ids_1dto2d(idx, M, N)
                current_node.set_neighbors(get_neighbor_list(i, j, M, N, n, self.nodes))

    def construct_mapping(self):
        """
        :return:
        """
        target_grid_id = self.mapped_matrix_int[np.where(self.mapped_matrix_int > 0)]
        for g_id, n_id in zip(target_grid_id, self.target_grids):
            self.node_mapping[g_id] = n_id

    def construct_node_real(self, mapped_matrix_int):
        """ Initialize node, only valid node in mapped_matrix_in will be initialized.
        """
        row_inds, col_inds = np.where(mapped_matrix_int >= 0)

        target_ids = []  # start from 0. 
        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id] = Node(node_id)  # node id start from 0.
            target_ids.append(node_id)

        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id].get_layers_neighbors(self.l_max, self.M, self.N, self)

        self.target_grids = target_ids
        self.n_valid_grids = len(target_ids)

    def construct_map_real(self, n_side):
        """Build node connection. 
        """
        for idx, current_node in enumerate(self.nodes):
            i, j = ids_1dto2d(idx, self.M, self.N)
            if current_node is not None:
                current_node.set_neighbors(get_neighbor_list(i, j, self.M, self.N, n_side, self.nodes))

    def initial_order_random(self, distribution_all, dis_paras_all):
        """ Initialize order distribution
        :param distribution: 'Poisson', 'Gaussian'
        :param dis_paras:     lambda,    mu, sigma
        """
        for idx, node in enumerate(self.nodes):
            if node is not None:
                node.order_distribution(distribution_all[idx], dis_paras_all[idx])

    def get_observation(self):
        next_state = np.zeros((2, self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[0, row_id, column_id] = _node.idle_courier_num
                next_state[1, row_id, column_id] = _node.order_num

        return next_state

    def get_num_idle_couriers(self):
        """ Compute idle couriers
        :return:
        """
        temp_n_idle_couriers = 0
        for _node in self.nodes:
            if _node is not None:
                temp_n_idle_couriers += _node.idle_courier_num
        return temp_n_idle_couriers

    def get_observation_courier_state(self):
        """ Get idle courier distribution, computing #couriers from node.
        :return:
        """
        next_state = np.zeros((self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[row_id, column_id] = _node.get_idle_courier_numbers_loop()

        return next_state

    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def reset(self):
        """ Return initial observation: get order distribution and idle courier distribution

        """

        _M = self.M
        _N = self.N
        assert self.city_time == 0
        # initialization couriers according to the distribution at time 0
        num_idle_courier = self.utility_get_n_idle_couriers_real()
        self.step_courier_online_offline_control(num_idle_courier)

        # generate orders at first time step
        distribution_name = [self.distribution_name] * (_M * _N)
        distribution_param_dictionary = self.order_num_dist[self.city_time]
        distribution_param = [0] * (_M * _N)
        for key, value in distribution_param_dictionary.iteritems():
            if self.distribution_name == 'Gaussian':
                mu, sigma = value
                distribution_param[key] = mu, sigma
            elif self.distribution_name == 'Poisson':
                mu = value[0]
                distribution_param[key] = mu
            else:
                print("Wrong distribution")

        self.initial_order_random(distribution_name, distribution_param)
        self.step_generate_order_real()

        return self.get_observation()

    def reset_clean(self, generate_order=1, ratio=1, city_time=""):
        """ 1. bootstrap oneday's order data.
            2. clean current couriers and orders, regenerate new orders and couriers.
            can reset anytime
        :return:
        """
        if city_time != "":
            self.city_time = city_time

        # clean orders and couriers
        self.couriers = {}  # courier[courier_id] = courier_instance  , courier_id start from 0
        self.n_couriers = 0  # total idle number of couriers. online and not on service.
        self.n_offline_couriers = 0  # total number of offline couriers.
        for node in self.nodes:
            if node is not None:
                node.clean_node()

        # Generate one day's order.
        if generate_order == 1:
            self.utility_bootstrap_oneday_order()

        # Init orders of current time step
        moment = self.city_time % self.n_intervals
        self.step_bootstrap_order_real(self.day_orders[moment])

        # Init current courier distribution
        if self.global_flag == "global":
            num_idle_courier = self.utility_get_n_idle_couriers_real()
            num_idle_courier = int(num_idle_courier * ratio)
        else:
            num_idle_courier = self.utility_get_n_idle_couriers_nodewise()
        self.step_courier_online_offline_control_new(num_idle_courier)
        return self.get_observation()

    def utility_collect_offline_couriers_id(self):
        """count how many couriers are offline
        :return: offline_couriers: a list of offline courier id
        """
        count = 0  # offline courier num
        offline_couriers = []  # record offline courier id
        for key, _courier in self.couriers.iteritems():
            if _courier.online is False:
                count += 1
                offline_couriers.append(_courier.get_courier_id())
        return offline_couriers

    def utility_get_n_idle_couriers_nodewise(self):
        """ compute idle couriers.
        :return:
        """
        time = self.city_time % self.n_intervals
        idle_courier_num = np.sum(self.idle_courier_location_mat[time])
        return int(idle_courier_num)

    def utility_add_courier_real_new(self, num_added_courier):
        curr_idle_courier_distribution = self.get_observation()[0]
        curr_idle_courier_distribution_resort = np.array(
            [int(curr_idle_courier_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_courier_distribution = self.idle_courier_location_mat[self.city_time % self.n_intervals, :]

        idle_diff = idle_courier_distribution.astype(int) - curr_idle_courier_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        node_ids = np.random.choice(self.target_node_ids, size=[num_added_courier],
                                    p=idle_diff / float(np.sum(idle_diff)))

        n_total_couriers = len(self.couriers.keys())
        for ii, node_id in enumerate(node_ids):
            added_courier_id = n_total_couriers + ii
            self.couriers[added_courier_id] = courier(added_courier_id)
            self.couriers[added_courier_id].set_position(self.nodes[node_id])
            self.nodes[node_id].add_courier(added_courier_id, self.couriers[added_courier_id])

        self.n_couriers += num_added_courier

    def utility_add_courier_real_new_offlinefirst(self, num_added_courier):

        # curr_idle_courier_distribution = self.get_observation()[0][np.where(self.mapped_matrix_int > 0)]
        curr_idle_courier_distribution = self.get_observation()[0]
        curr_idle_courier_distribution_resort = np.array(
            [int(curr_idle_courier_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_courier_distribution = self.idle_courier_location_mat[self.city_time % self.n_intervals, :]

        idle_diff = idle_courier_distribution.astype(int) - curr_idle_courier_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        if float(np.sum(idle_diff)) == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_courier],
                                    p=idle_diff / float(np.sum(idle_diff)))

        for ii, node_id in enumerate(node_ids):

            if self.nodes[node_id].offline_courier_num > 0:
                self.nodes[node_id].set_offline_courier_online()
                self.n_couriers += 1
                self.n_offline_couriers -= 1
            else:

                n_total_couriers = len(self.couriers.keys())
                added_courier_id = n_total_couriers
                self.couriers[added_courier_id] = courier(added_courier_id)
                self.couriers[added_courier_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_courier(added_courier_id, self.couriers[added_courier_id])
                self.n_couriers += 1

    def utility_add_courier_real_nodewise(self, node_id, num_added_courier):

        while num_added_courier > 0:
            if self.nodes[node_id].offline_courier_num > 0:
                self.nodes[node_id].set_offline_courier_online()
                self.n_couriers += 1
                self.n_offline_couriers -= 1
            else:

                n_total_couriers = len(self.couriers.keys())
                added_courier_id = n_total_couriers
                self.couriers[added_courier_id] = courier(added_courier_id)
                self.couriers[added_courier_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_courier(added_courier_id, self.couriers[added_courier_id])
                self.n_couriers += 1
            num_added_courier -= 1

    def utility_set_couriers_offline_real_nodewise(self, node_id, n_couriers_to_off):

        while n_couriers_to_off > 0:
            if self.nodes[node_id].idle_courier_num > 0:
                self.nodes[node_id].set_idle_courier_offline_random()
                self.n_couriers -= 1
                self.n_offline_couriers += 1
                n_couriers_to_off -= 1
                self.all_grids_off_number += 1
            else:
                break

    def utility_set_couriers_offline_real_new(self, n_couriers_to_off):

        curr_idle_courier_distribution = self.get_observation()[0]
        curr_idle_courier_distribution_resort = np.array([int(curr_idle_courier_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle courier distribution
        idle_courier_distribution = self.idle_courier_location_mat[self.city_time % self.n_intervals, :]

        # diff of curr idle courier distribution and history
        idle_diff = curr_idle_courier_distribution_resort - idle_courier_distribution.astype(int)
        idle_diff[np.where(idle_diff <= 0)] = 0

        n_couriers_can_be_off = int(np.sum(curr_idle_courier_distribution_resort[np.where(idle_diff >= 0)]))
        if n_couriers_to_off > n_couriers_can_be_off:
            n_couriers_to_off = n_couriers_can_be_off

        sum_idle_diff = np.sum(idle_diff)
        if sum_idle_diff == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[n_couriers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        for ii, node_id in enumerate(node_ids):
            if self.nodes[node_id].idle_courier_num > 0:
                self.nodes[node_id].set_idle_courier_offline_random()
                self.n_couriers -= 1
                self.n_offline_couriers += 1
                n_couriers_to_off -= 1

    def utility_bootstrap_oneday_order(self):

        num_all_orders = len(self.real_orders)
        index_sampled_orders = np.where(np.random.binomial(1, self.p, num_all_orders) == 1)
        one_day_orders = self.real_orders[index_sampled_orders]

        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))

        day_orders = [[] for _ in np.arange(self.n_intervals)]
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]
            start_time = int(iorder[2])
            if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys():
                continue
            start_node = self.node_mapping.get(iorder[0], -100)
            end_node = self.node_mapping.get(iorder[1], -100)
            duration = int(iorder[3])
            price = iorder[4]

            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1
                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        self.day_orders = day_orders

    def step_courier_status_control(self):
        # Deal with orders finished at time T=1, check courier status. finish order, set back to off service
        for key, _courier in self.couriers.iteritems():
            _courier.status_control_eachtime(self)
        moment = self.city_time % self.n_intervals
        orders_to_on_couriers = self.out_grid_in_orders[moment, :]
        for idx, item in enumerate(orders_to_on_couriers):
            if item != 0:
                node_id = self.target_grids[idx]
                self.utility_add_courier_real_nodewise(node_id, int(item))

    def step_courier_online_offline_nodewise(self):
        """ node wise control courier online offline
        :return:
        """
        moment = self.city_time % self.n_intervals
        curr_onoff_distribution = self.onoff_courier_location_mat[moment]

        self.all_grids_on_number = 0
        self.all_grids_off_number = 0
        for idx, target_node_id in enumerate(self.target_node_ids):
            curr_mu = curr_onoff_distribution[idx, 0]
            curr_sigma = curr_onoff_distribution[idx, 1]
            on_off_number = np.round(np.random.normal(curr_mu, curr_sigma, 1)[0]).astype(int)

            if on_off_number > 0:
                self.utility_add_courier_real_nodewise(target_node_id, on_off_number)
                self.all_grids_on_number += on_off_number
            elif on_off_number < 0:
                self.utility_set_couriers_offline_real_nodewise(target_node_id, abs(on_off_number))
            else:
                pass

    def step_courier_online_offline_control_new(self, n_idle_couriers):
        """ control the online offline status of couriers

        :param n_idle_couriers: the number of idle couriers expected at current moment
        :return:
        """

        offline_couriers = self.utility_collect_offline_couriers_id()
        self.n_offline_couriers = len(offline_couriers)

        if n_idle_couriers > self.n_couriers:

            self.utility_add_courier_real_new_offlinefirst(n_idle_couriers - self.n_couriers)

        elif n_idle_couriers < self.n_couriers:
            self.utility_set_couriers_offline_real_new(self.n_couriers - n_idle_couriers)
        else:
            pass

    def step_courier_online_offline_control(self, n_idle_couriers):
        """ control the online offline status of couriers

        :param n_idle_couriers: the number of idle couriers expected at current moment
        :return:
        """

        offline_couriers = self.utility_collect_offline_couriers_id()
        self.n_offline_couriers = len(offline_couriers)
        if n_idle_couriers > self.n_couriers:
            # bring couriers online.
            while self.n_couriers < n_idle_couriers:
                if self.n_offline_couriers > 0:
                    for ii in np.arange(self.n_offline_couriers):
                        self.couriers[offline_couriers[ii]].set_online()
                        self.n_couriers += 1
                        self.n_offline_couriers -= 1
                        if self.n_couriers == n_idle_couriers:
                            break

                self.utility_add_courier_real_new(n_idle_couriers - self.n_couriers)

        elif n_idle_couriers < self.n_couriers:
            self.utility_set_couriers_offline_real_new(self.n_couriers - n_idle_couriers)
        else:
            pass

    def utility_get_n_idle_couriers_real(self):
        """ control the number of idle couriers in simulator;
        :return:
        """
        time = self.city_time % self.n_intervals
        mean, std = self.idle_courier_dist_time[time]
        np.random.seed(self.city_time)
        return np.round(np.random.normal(mean, std, 1)[0]).astype(int)

    def utility_set_neighbor_weight(self, weights):
        self.weights_layers_neighbors = weights

    def step_generate_order_real(self):
        # generate order at t + 1
        for node in self.nodes:
            if node is not None:
                node_id = node.get_node_index()
                # generate orders start from each node
                random_seed = node.get_node_index() + self.city_time
                node.generate_order_real(self.l_max, self.order_time_dist, self.order_price_dist,
                                         self.city_time, self.nodes, random_seed)

    def step_bootstrap_order_real(self, day_orders_t):
        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_node = self.nodes[start_node_id]

            if end_node_id in self.target_grids:
                end_node = self.nodes[end_node_id]
            else:
                end_node = None
            start_node.add_order_real(self.city_time, end_node, iorder[3], iorder[4])

    def step_assign_order(self):

        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time,
                                                                                                         self)
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1
        return reward

    def step_assign_order_broadcast_neighbor_reward_update(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        for node in self.nodes:
            if node is not None:
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time,
                                                                                                         self)
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
                node_reward[node.get_node_index()] += reward_node
        # Second round broadcast
        for node in self.nodes:
            if node is not None:
                if node.order_num != 0:
                    reward_node_broadcast, finished_order_num_node_broadcast \
                        = node.simple_order_assign_broadcast_update(self, neighbor_reward)
                    reward += reward_node_broadcast
                    finished_order_num += finished_order_num_node_broadcast

        node_reward = node_reward + neighbor_reward
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1

        return reward, [node_reward, neighbor_reward]

    def step_remove_unfinished_orders(self):
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)

    def step_pre_order_assigin(self, next_state):

        remain_couriers = next_state[0] - next_state[1]
        remain_couriers[remain_couriers < 0] = 0

        remain_orders = next_state[1] - next_state[0]
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_couriers) == 0:
            context = np.array([remain_couriers, remain_orders])
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_couriers_1d = remain_couriers.flatten()

        for node in self.nodes:
            if node is not None:
                curr_node_id = node.get_node_index()
                if remain_orders_1d[curr_node_id] != 0:
                    for neighbor_node in node.neighbors:
                        if neighbor_node is not None:
                            neighbor_id = neighbor_node.get_node_index()
                            a = remain_orders_1d[curr_node_id]
                            b = remain_couriers_1d[neighbor_id]
                            remain_orders_1d[curr_node_id] = max(a - b, 0)
                            remain_couriers_1d[neighbor_id] = max(b - a, 0)
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        context = np.array([remain_couriers_1d.reshape(self.M, self.N),
                            remain_orders_1d.reshape(self.M, self.N)])
        return context

    def step_dispatch_invalid(self, dispatch_actions):
        """
        :param dispatch_actions:
        :return:
        """
        save_remove_id = []
        for action in dispatch_actions:

            start_node_id, end_node_id, num_of_couriers = action
            if self.nodes[start_node_id] is None or num_of_couriers == 0:
                continue  # not a feasible action

            if self.nodes[start_node_id].get_courier_numbers() < num_of_couriers:
                num_of_couriers = self.nodes[start_node_id].get_courier_numbers()

            if end_node_id < 0:
                for _ in np.arange(num_of_couriers):
                    self.nodes[start_node_id].set_idle_courier_offline_random()
                    self.n_couriers -= 1
                    self.n_offline_couriers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] is None:
                for _ in np.arange(num_of_couriers):
                    self.nodes[start_node_id].set_idle_courier_offline_random()
                    self.n_couriers -= 1
                    self.n_offline_couriers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] not in self.nodes[start_node_id].neighbors:
                raise ValueError('City:step(): not a feasible dispatch')

            for _ in np.arange(num_of_couriers):
                # t = 1 dispatch start, idle courier decrease
                remove_courier_id = self.nodes[start_node_id].remove_idle_courier_random()
                save_remove_id.append((end_node_id, remove_courier_id))
                self.couriers[remove_courier_id].set_position(None)
                self.couriers[remove_courier_id].set_offline_for_start_dispatch()
                self.n_couriers -= 1

        return save_remove_id

    def step_add_dispatched_couriers(self, save_remove_id):
        # couriers dispatched at t, arrived at t + 1
        for destination_node_id, arrive_courier_id in save_remove_id:
            self.couriers[arrive_courier_id].set_position(self.nodes[destination_node_id])
            self.couriers[arrive_courier_id].set_online_for_finish_dispatch()
            self.nodes[destination_node_id].add_courier(arrive_courier_id, self.couriers[arrive_courier_id])
            self.n_couriers += 1

    def step_increase_city_time(self):
        self.city_time += 1
        # set city time of couriers
        for courier_id, courier in self.couriers.iteritems():
            courier.set_city_time(self.city_time)

    def step(self, dispatch_actions, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the courier distribution
        save_remove_id = self.step_dispatch_invalid(dispatch_actions)
        # When the couriers go to invalid grid, set them offline.

        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        self.step_increase_city_time()
        self.step_courier_status_control()  # couriers finish order become available again.

        # couriers dispatched at t, arrived at t + 1, become available at t+1
        self.step_add_dispatched_couriers(save_remove_id)

        # generate order at t + 1
        if generate_order == 1:
            self.step_generate_order_real()
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        self.step_courier_online_offline_nodewise()

        self.step_remove_unfinished_orders()
        # get states S_{t+1}  [courier_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info





