import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np


#############
# Q-Network #
#############
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action = nn.Linear(64, action_dim)
        # self.duration_head = nn.Linear(64, duration_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_q = self.action(x)  # shape: [batch_size, phase_dim]
        return action_q #, duration_q


#############
# DQN Agent #
#############
class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon_decay=0.995,lr=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim  # number of phase options
        # self.duration_n = duration_n  # number of discrete duration options
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=5000)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
            # duration_action = random.randrange(self.duration_n)
            return action

        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_q = self.q_network(state_tensor)
        action = int(torch.argmax(action_q, dim=1).item())
        # duration_action = int(torch.argmax(duration_q, dim=1).item())
        return action

    def store_transition(self, transition):
        # transition is a tuple: (state, (phase_action, duration_action), reward, next_state, done)
        self.memory.append(transition)

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.sample_memory(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(states).to(self.device)
        # actions is a tuple of (phase, duration) per sample
        actions = torch.LongTensor([a for a in actions]).unsqueeze(1).to(self.device)
        # duration_actions = torch.LongTensor([a[1] for a in actions]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_action_q= self.q_network(states)
        current_action_q = current_action_q.gather(1, actions)
        # current_duration_q = current_duration_q.gather(1, duration_actions)
        current_q = current_action_q #+ current_duration_q  # combine the two Q-values

        with torch.no_grad():
            next_action_q = self.target_network(next_states)
            max_next_action_q = next_action_q.max(1)[0].unsqueeze(1)
            # max_next_duration_q = next_duration_q.max(1)[0].unsqueeze(1)
            max_next_q = max_next_action_q #+ max_next_duration_q
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

