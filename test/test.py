import os
import datetime

import torch

from tqdm import tqdm

import gym
from gym.wrappers.monitoring import video_recorder

from model.agent import Agent


def create_video_path(env_name):
    """
    Create the path hierarchy.
    :param env_name: the name of the environment.
    :return: the recording path
    """
    time = datetime.datetime.now()
    time = time.strftime("%b%m_%H-%M-%S")

    video_path = "./video/{}_{}".format(env_name, time)
    cwd = os.getcwd()
    video_path = os.path.join(cwd, video_path)
    try:
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
    except OSError as error:
        print(error)

    recorder_path = os.path.join(video_path, "{}".format(env_name))

    return recorder_path


def run_episode(env_name, checkpoint_path, recordings=False):
    """
    Load the weights of the policy for the DQNAgent, and then save a recording of a single game.
    :param env_name: the name of the environment
    :param checkpoint_path: the path to the policy checkpoint
    :param recordings: flag that enable the video of the test
    :return:
    """

    # Create the agent and get the action space and the action space sizes.
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size=state_size, action_size=action_size, seed=1)

    agent.policy.load_state_dict(torch.load(checkpoint_path))
    state = env.reset()
    score = 0
    done = False

    if recordings:
        recorder_path = create_video_path(env_name)

        vid = video_recorder.VideoRecorder(env, base_path=recorder_path + '')
        while not done:
            vid.capture_frame()

            action = agent.act(state)

            state, reward, done, _ = env.step(action)

            score += reward

        vid.close()
    else:
        while not done:

            action = agent.act(state)

            state, reward, done, _ = env.step(action)

            score += reward

    env.close()

    return score


if __name__ == '__main__':

    ENVIRONMENT_NAME = 'LunarLander-v2'
    NUM_EPISODE = 100
    scr = 0
    for episode in tqdm(range(NUM_EPISODE)):
        scr += run_episode(ENVIRONMENT_NAME, checkpoint_path='./checkpoints/checkpoint_end.pth', recordings=False)

    scr = scr / NUM_EPISODE

    print("Mean score: {}".format(scr))


