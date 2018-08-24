[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Project 1 - Navigation

This repository contains solution to Project 1 of the Deep Reinforcement Learning course at Udacity.

## Project details

The objective of this project is to solve a "Banana collector game". Essentially, it is a square world filled with blue and yellow bananas.
For each yellow banana picked up, a reward of +1 is given, whereas a blue banana yields a reward of -1.
Consequently, we want a agent which picks up the yellow bananas while avoiding the blue ones.


### State and agent space

The state space has 37 dimensions, of which the first parameter is the agent's velocity, and the remaining 36 are ray-based perception of the forward facing environment of the agent.

In each time step, the agent has four choices. He can

* `0` - Walk forwards,
* `1` - Walk backwards,
* `2` - Turn left,
* `3` - Turn right.

The task is episodic, and the problem is considered solved when the agent received an average score of +13 over 100 consecutive episodes.

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

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then place the file in the `/p1_navigation` catalogue, and also overwrite `/p1_navigation/Navigation.ipynb`
from the `https://github.com/udacity/deep-reinforcement-learning` repo, with the file in this repo, and you are set!

## Instructions

To train the agent, please look at `Navigation.ipynb`. This file will take you though the process, from importing the necessary packages to training the model. Essentially, everything regarding the training of the model is given in Part 4 of the file.
The underlying algorithm is written in `dqn_agent.py`, wheareas the neural networks used by the agent is written in `model.py`.
