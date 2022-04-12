import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Simple QNetwork implementation.
    """

    def __init__(self, state_size, action_size, hidden_dim=64, seed=0):
        """
        Initialize the network parameter and build the model
        :param state_size: dimension of the state space
        :param action_size: dimension of the action space
        :param hidden_dim: dimension of the hidden representation
        :param seed: seed used for experiment reproducibility.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        """
        Build a network that map the state in to an action
        :param state: representation of environment state
        :return: actions QValues.
        """
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
