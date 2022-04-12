# Introduction
This repository contains an implementation of a DQN agent that solves the [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) game and the code needed for the generation of datasets of experiences.

The codebase's goals were the creation of an ε-greedy agent that can solve a simple Gym environment and the creation of a dataset of experiences that can be used by Off-Line algorithms. 

## Overview
The model folder contains the code for the agent, the policy network and the replay buffer.

In the train directory can be found the code needed to train the agent.
Simple training is performed with [train.py](./train/train.py) while a training that can save the model during the evolution is performed by [checkpoint_training.py](./train/checkpoint_training.py).

The last directory contains the code needed for testing the trained model. The [test.py](./test/test.py) allows to enable the visual recording of the game played by the agent.

The [dataset_generator.py](./dataset_generator.py) allows to collect and store the experience of the agent on disk.

### Directories tree
```
.
├── README.md
├── conda_env.yml
├── dataset_generator.py
├── model
│   ├── agent.py
│   ├── deep_model.py
│   └── replay_memory.py
├── test
│   └── test.py
├── train
│   ├── checkpoint_training.py
│   └── train.py
```

### Installation
Dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```
