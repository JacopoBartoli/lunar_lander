import math

import gym

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

from agent import Agent


def run_episode(environment, a, max_t, epsilon):
    state = environment.reset()
    scr = 0
    for t in range(max_t):
        # environment.render()
        action = a.act(state, epsilon)
        next_state, reward, done, _ = environment.step(action)
        a.step(state, action, reward, next_state, done)
        state = next_state
        scr += reward
        if done:
            break
    return scr


MAX_EPISODES = 2000
MAX_TIMESTEP = 1000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
DIM_WINDOW = 100

THRESHOLDS = [-200, -100, -50, 0, 50, 100, 200, math.inf]

if __name__ == '__main__':
    writer = SummaryWriter()

    scores_window = np.zeros(DIM_WINDOW)
    scores = np.empty(MAX_EPISODES)
    episode_count = 0
    thresh_idx = 0

    env = gym.make('LunarLander-v2')
    env.seed(0)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size=state_size, action_size=action_size, seed=1)

    eps = EPS_START
    for episode in tqdm(range(MAX_EPISODES)):

        score = run_episode(env, agent, MAX_TIMESTEP, eps)

        # Update epsilon
        eps = max(EPS_END, EPS_DECAY * eps)
        scores[episode] = score
        scores_window[episode % DIM_WINDOW] = score

        if episode < DIM_WINDOW:
            mean_reward = np.sum(scores_window) / (episode + 1)
        else:
            mean_reward = np.mean(scores_window)

        episode_count += 1

        writer.add_scalar("Score/reward", score, episode)
        writer.add_scalar("Score/mean reward", mean_reward, episode)
        writer.add_scalar("Parameters/epsilon", eps, episode)

        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.policy.state_dict(), './checkpoints/checkpoint_200.pth')
            break

        if THRESHOLDS[thresh_idx] <= np.mean(scores_window):
            torch.save(agent.policy.state_dict(), './checkpoints/checkpoint_{}.pth'.format(THRESHOLDS[thresh_idx]))
            thresh_idx += 1

    env.close()
    writer.close()
