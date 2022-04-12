import os

import numpy as np

import torch

import gym

from tqdm import tqdm

from model.agent import Agent

from collections import namedtuple

ENV_NAME = 'LunarLander-v2'
MAX_TRAJECTORIES = 2000
CHECKPOINT_PATH = './checkpoints/checkpoint_600.pth'
OUT_PATH = 'LunarLander_600'

MAX_TIMESTEP = 1000

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "done"])


def run_episode(env, a, max_t):
    """
    Run a simgle game episode.
    :param env: the game environment
    :param a: the agent that need to make the decision for the next actions
    :param max_t: maximum number of action that an agent can take in the environment
    :return: a list of collected experiences
    """

    data = np.empty(max_t, dtype=Experience)
    index = 0

    state = env.reset()

    for step in range(max_t):

        action = a.act(state)

        next_state, reward, done, _ = env.step(action)

        data[index] = Experience(state, action, reward, done)
        index += 1

        state = next_state

        if done:
            break
    return data[:index]


def store_dataset(data, file_path):
    """
    Save the dataset on disk.
    :param data: a list of data
    :param file_path: name of the file that contains the dataset
    :return:
    """
    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, 'data')

    try:
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
    except OSError as error:
        print(error)

    dataset_path = os.path.join(dataset_dir, file_path)
    np.savez_compressed(dataset_path, data)


if __name__ == '__main__':

    environment = gym.make(ENV_NAME)
    state_size = environment.observation_space.shape[0]
    action_size = environment.action_space.n

    agent = Agent(state_size=state_size, action_size=action_size, seed=1)
    agent.policy.load_state_dict(torch.load(CHECKPOINT_PATH))

    datalist = np.empty(MAX_TRAJECTORIES * MAX_TIMESTEP, dtype=Experience)
    num_elem = 0

    for i in tqdm(range(MAX_TRAJECTORIES)):
        experiences = run_episode(environment, agent, MAX_TIMESTEP)

        idx = experiences.shape[0]

        datalist[num_elem: num_elem + idx] = experiences
        num_elem += idx

    out = datalist[:num_elem]
    store_dataset(out, OUT_PATH)


