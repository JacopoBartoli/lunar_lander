import gym
import torch
from gym.wrappers.monitoring import video_recorder

from agent import Agent
import datetime
import os


def create_paths(env_name):
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


def save_video(env_name, checkpoint_path='./checkpoint.pth'):
    """
    Load the weights of the policy for the DQNAgent, and then save a recording of a single game.
    :param env_name: the name of the environment
    :param checkpoint_path: the path to the policy checkpoint
    :return:
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size=state_size, action_size=action_size, seed=1)

    recorder_path = create_paths(env_name)

    vid = video_recorder.VideoRecorder(env, base_path=recorder_path)
    agent.policy.load_state_dict(torch.load(checkpoint_path))
    state = env.reset()
    done = False
    while not done:
        vid.capture_frame()

        action = agent.act(state)

        state, reward, done, _ = env.step(action)
    env.close()
    vid.close()


if __name__ == '__main__':
    environment_name = 'LunarLander-v2'
    save_video(environment_name)
