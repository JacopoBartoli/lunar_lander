import random
import torch
import numpy as np
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    """
    Class che implement a fixed size buffer.
    """

    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize the ReplayBuffer.
        :param buffer_size: the length of the buffer
        :param batch_size: the batch size that need to be sampled
        :param seed: seed used for experiment reproducibility.
        """
        super(ReplayBuffer, self).__init__()

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = np.empty(buffer_size, dtype=self.experience)

        self.batch_size = batch_size
        self.push_count = 0
        self.buffer_size = buffer_size
        self.full = False

        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience in the buffer.
        :param state: actual state of the environment
        :param action: action taken at the current timestep
        :param reward: reward obtained after the action
        :param next_state: environment state after the action
        :param done: boolean that specify if the game is solved after the action
        :return:
        """
        # Cast all the input in tensor.
        # In this way we cast them only one time, instead of casting each time they will get sampled from the buffer.
        state = torch.from_numpy(state).float()
        action = torch.tensor(action).long().unsqueeze(-1)
        reward = torch.tensor(reward).float().unsqueeze(-1)
        next_state = torch.from_numpy(next_state).float()
        done = torch.tensor(done).float().unsqueeze(-1)

        e = self.experience(state, action, reward, next_state, done)

        # Deal with the numpy array like it is a que.
        if self.full:
            self.memory[0:-1] = self.memory[1:]
            self.memory[-1] = e
        else:
            self.memory[self.push_count] = e
        self.push_count = (self.push_count + 1) % self.buffer_size

        # Check if the numpy array is completely filled.
        if not self.full and self.push_count == 0:
            self.full = True

    def sample(self):
        """
        Return a random sample of the buffer. The number of sample is equals to the batch size.
        :return: random sampled batch of experiences.
        """
        if self.full:
            samples = random.sample(range(self.buffer_size), self.batch_size)
        else:
            samples = random.sample(range(self.push_count), self.batch_size)

        experiences = self.memory[samples]

        batch = self.experience(*zip(*experiences))

        states = torch.stack(batch.state, dim=0).to(device)
        actions = torch.stack(batch.action).to(device)
        rewards = torch.stack(batch.reward).to(device)
        next_states = torch.stack(batch.next_state, dim=0).to(device)
        dones = torch.stack(batch.done).to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Override of the len method.
        :return: return the number of element actually inside the numpy array.
        """
        if not self.full:
            return self.push_count
        else:
            return self.buffer_size
