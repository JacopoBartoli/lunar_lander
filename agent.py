import random

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from model import QNetwork
from replay_memory import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """
    A simple DQN agent with target network.
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an Agent object.
        :param state_size: dimension of the state space
        :param action_size: dimension of the action space
        :param seed: seed for experiment reproducibility.
        """
        super(Agent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        self.policy = QNetwork(state_size, action_size, seed=seed).to(device)
        self.target_net = QNetwork(state_size, action_size, seed=seed).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Method that perform single step of the Agent.
        Step phases:
        - add the experience in the Replay Memory
        - extract a random batch of experiences from the Replay Memory
        - learn from the extracted experiences
        :param state: environment state
        :param action: action taken by the agent
        :param reward: reward obtained after the action
        :param next_state: state of the environment after the action
        :param done: boolean that specify if the game is solved after the action
        :return:
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Given a state return the action chosen by the Agent
        :param state: state of the environment
        :param eps: epsilon for e-greedy strategy
        :return: Agent's action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy.eval()
        with torch.no_grad():
            action_values = self.policy(state)
        self.policy.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Method that make the Agent learn from a sample of the Replay Memory.
        :param experiences: batch of experiences
        :param gamma: gamma for QValues computing
        :return:
        """
        # Extract the random minibatch.
        states, actions, rewards, next_states, dones = experiences

        # Extract next maximum estimated value from target network.
        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute target value using Bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        # Compute expected value from the policy network.
        q_expected = self.policy(states).gather(1, actions)

        # Loss computing.
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network.
        self.soft_update(TAU)

    def soft_update(self, tau):
        """
        Soft update of the model's parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param tau: interpolation parameter
        :return:
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)