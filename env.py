"""
-*- coding: utf-8 -*-
@Time : 2023/1/12 11:33
@Author : wanghai11
"""

import random
import numpy as np



class MultiAgentEnv:
    def __init__(self, num_agents, num_orders):
        self.num_agents = num_agents
        self.num_orders = num_orders

        self.order_states = np.zeros(num_orders)  # Each order's state
        self.agent_states = np.zeros(num_agents)  # Each agent's state

        self.order_prices = np.random.rand(num_orders)  # Random initial prices for each order
        self.agent_prices = np.zeros(num_agents)  # Each agent's price for the order they chose

        self.alpha = 0.5  # Weight for profit in reward function
        self.beta = 0.3  # Weight for completion rate in reward function
        self.gamma = 0.2  # Weight for customer satisfaction in reward function

    def reset(self):
        self.order_states = np.zeros(self.num_orders)
        self.agent_states = np.zeros(self.num_agents)
        self.order_prices = np.random.rand(self.num_orders)
        self.agent_prices = np.zeros(self.num_agents)
        return self.order_states, self.agent_states

    def step(self, actions):
        rewards = np.zeros(self.num_agents)

        for i, action in enumerate(actions):
            order_idx = action
            self.agent_prices[i] = self.order_prices[order_idx]
            reward = self.calculate_reward(i, order_idx)
            rewards[i] = reward

            self.update_states(i, order_idx)

        return rewards

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
        profit = self.agent_prices[agent_idx] - self.order_prices[order_idx]
        return profit

    def calculate_completion_rate(self, agent_idx, order_idx):
        total_tasks = self.num_orders  # 假设总任务数等于揽件任务数
        completed_tasks = np.sum(self.order_states)
        completion_rate = completed_tasks / total_tasks
        return completion_rate

    def calculate_customer_satisfaction(self, agent_idx, order_idx):
        # 假设有一个投诉列表，其中包含所有投诉订单的索引
        complaint_orders = []  # 假设投诉列表为空
        complaint_rate = len(complaint_orders) / self.num_orders
        customer_satisfaction = 1 - complaint_rate  # 客户满意度等于1减去投诉率
        return customer_satisfaction


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






