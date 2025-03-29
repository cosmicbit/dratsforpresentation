import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


#####################
# Q-Network & Agent #
#####################
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    """
    Centralized DQN Agent for controlling two intersections.

    state_dim: dimension of the state vector (e.g., 10)
    action_dim: size of the joint action space (e.g., 4)
    """

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.991, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim  # Joint action space for two intersections.
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
        """
        Returns a joint action for both intersections.
        With probability epsilon, chooses a random joint action.
        Otherwise, selects the joint action with highest Q-value.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def store_transition(self, transition):
        """
        Transition is a tuple: (state, joint_action, reward, next_state, done)
        """
        self.memory.append(transition)

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.sample_memory(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # Update epsilon (exploration decay)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
