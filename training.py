import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from env import MultiAgentEnv
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# Define the GRU feature analysis unit
class GRUUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUUnit, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim)

    def forward(self, x, hidden=None):
        # x should be of shape (seq_len, batch, input_size)
        out, h_n = self.gru(x, hidden)
        return out, h_n


# Define the GAT network for agent interactions
class GATNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GATNetwork, self).__init__()
        self.gat_conv = GATConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        # x is the node features, edge_index defines the connections between nodes
        return self.gat_conv(x, edge_index)


# Define the Actor network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.tanh(self.fc2(x))
        return x


# Define the Critic network
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, gru_hidden_dim, gat_output_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(gru_hidden_dim + gat_output_dim, output_dim)

    def forward(self, gru_output, gat_output):
        # Concatenate the outputs of the GRU and GAT
        concatenated = torch.cat((gru_output, gat_output), dim=1)
        return F.softmax(self.fc(concatenated), dim=1)


class TransformerCritic(nn.Module):
    def __init__(self, input_dim, num_agents, hidden_dim, nhead, num_layers):
        super(TransformerCritic, self).__init__()
        self.num_agents = num_agents

        # Define an embedding layer for processing input data
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Define a transformer layer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead),
            num_layers=num_layers
        )

        # Define a linear layer for output
        self.output_layer = nn.Linear(hidden_dim * num_agents, 1)

    def forward(self, x):
        # x should be a tensor of shape (batch_size, num_agents, input_dim)
        embedded = self.embedding(x)

        # Transformer requires input of shape (seq_length, batch_size, hidden_dim)
        transformed = self.transformer(embedded.permute(1, 0, 2))

        # Flatten and pass through output layer
        flattened = transformed.permute(1, 0, 2).reshape(x.size(0), -1)
        output = self.output_layer(flattened)
        return output


# # Define the input dimensions, number of agents, and other parameters
# input_dim = 10
# num_agents = 3
# hidden_dim = 64
# nhead = 4
# num_layers = 2
#
# # Create an instance of the TransformerCritic
# critic = TransformerCritic(input_dim, num_agents, hidden_dim, nhead, num_layers)
#
# # Example input tensor of shape (batch_size, num_agents, input_dim)
# x = torch.rand((5, num_agents, input_dim))
#
# # Forward pass
# output = critic(x)
# print(output)



# Define the MADDPG agent
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim + action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)


# GFMRL
class GFMRLAgent:
    def __init__(self, gru_hidden_dim, gat_output_dim, output_dim, input_dim, num_agents, hidden_dim, nhead, num_layers, learning_rate=0.001):
        self.actor = PolicyNetwork(gru_hidden_dim, gat_output_dim, output_dim)
        self.critic = TransformerCritic(input_dim, num_agents, hidden_dim, nhead, num_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)


# Define the training process
def train(agents, environment, n_epochs=1000):
    for epoch in range(n_epochs):
        # Reset the environment and get the initial state
        states = environment.reset()
        done = False

        while not done:
            # Collect actions from all agents
            actions = [agent.actor(torch.tensor(state, dtype=torch.float32)).detach().numpy() for agent, state in zip(agents, states)]
            actions = np.array(actions)

            # Take a step in the environment
            next_states, rewards, dones = environment.step(actions)

            # Update each agent
            for i, agent in enumerate(agents):
                state = torch.tensor(states[i], dtype=torch.float32)
                action = torch.tensor(actions[i], dtype=torch.float32)
                reward = torch.tensor(rewards[i], dtype=torch.float32)
                next_state = torch.tensor(next_states[i], dtype=torch.float32)
                done = torch.tensor(dones[i], dtype=torch.float32)

                # Update the critic
                Q_expected = agent.critic(state, action)
                Q_target = reward + 0.99 * agent.critic(next_state, agent.actor(next_state))
                critic_loss = nn.MSELoss()(Q_expected, Q_target.detach())
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                agent.critic_optimizer.step()

                # Update the actor
                policy_loss = -agent.critic(state, agent.actor(state)).mean()
                agent.actor_optimizer.zero_grad()
                policy_loss.backward()
                agent.actor_optimizer.step()

            # Update the states
            states = next_states

        # Print the progress
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{n_epochs} completed')

# Create the agents and the environment
n_agents = 100
state_dim = 100
action_dim = 20
# gru_hidden_dim, gat_output_dim, output_dim, input_dim, num_agents, hidden_dim, nhead, num_layers =
# agents = [MADDPGAgent(state_dim, action_dim) for _ in range(n_agents)]
agents = [GFMRLAgent(gru_hidden_dim, gat_output_dim, output_dim, input_dim, num_agents, hidden_dim, nhead, num_layers) for _ in range(n_agents)]
environment = MultiAgentEnv(100, 100000)

# Train the agents
train(agents, environment)
