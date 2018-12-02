# Project 2: Collaboration and Competition

This repository contains solution to Project 3 of the Deep Reinforcement Learning course at Udacity.

## Project details

The objective of this project is to solve a "Tennis" environment. Essentially, we have two tennis
players (or basically two racquets) which try to play a game of tennis.
For each time step one agent gets the ball over the net, a reward of +.1 is given to that agent, whereas a
reward of -0.01 is given is one agent drops the ball or it goes out of bounds.
Consequently, the longer the rallies, the higher the reward.

### State and agent space

The state space has 8 dimensions (locally observable for each agent), corresponding to position and  velocity of the ball and the racquet.

In each time step, the agents must decide two actions. These two numbers represent the moving towards
(or away) from the net and whether to jump or not. Each of these numbers should be between $[-1,1]$.

The task is episodic, and the problem is considered solved when the highest scoring agent for each
episode receives an average score of +0.5 over 100 consecutive episodes.

## Getting started

### Github

Please start by cloning the following repo [here](https://github.com/udacity/deep-reinforcement-learning)

### Setup Python + Jupyter

To use the code, please follow the excellent recipe given [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).
However, note that if you are on a Windows OS, in Step 3 of the recipe, remove `torch==0.4.0` from `requirements.txt`, and run the following lines instead

* `pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl`
* `pip3 install torchvision`

### Download the environment

After successfully installing dependencies, we must build the environment. Please download the version suitable to your OS

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then place the file in the `/p3_continuous_control` catalogue, and also overwrite `/p3_navigation/Tennis.ipynb`
from the `https://github.com/udacity/deep-reinforcement-learning` repo, with the file in this repo, and you are set!

## Instructions

To train the agent, please look at `Tennis.ipynb`. This file will take you though the process, from importing the necessary packages to training the model. Essentially, everything regarding the training of the model is given in Part 4 of the file.
The underlying algorithm is written in `ddpg_agent.py`, wheareas the neural networks used by the agent is written in `model.py`.
