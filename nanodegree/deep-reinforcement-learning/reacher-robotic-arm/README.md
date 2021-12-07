# Project 2: Continuous Control

## Introduction

For this project, you I worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of my agent was to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.

## Distributed Training

For this project, Udacity provided with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

To solve the problem, I used the second version, i.e., the one with 20 agents.

## Solving the Environment

The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).

## Instructions

### Installation

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create an Ubuntu 18.04 virtual machine using VirtualBox.

    You may use this [tutorial](https://www.youtube.com/watch?v=44Se48TNOtI).

2. Install conda.
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p
source $HOME/miniconda3/bin/activate
conda --help
```

3. Create (and activate) a new environment with Python 3.6.
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
	
4. Clone the repository below, and navigate to the python/ folder. Then, install several dependencies.
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Clone this repository and install dependencies.
```bash
cd ..
git clone https://github.com/rnmourao/udacity.git
cd udacity/nanodegree/deep-reinforcement-learning/reacher-robotic-arm
pip install -r requirements.txt
```

5. Download the environment.

    - Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.

6. Place the file into reacher-robotic-arm folder, and unzip (or decompress) the file. 

7. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

### Executing the Notebook


1. Start Jupyter.

```bash
jupyter notebook
```

2. Open the Continuous_Control.ipynb file.


3. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)


4. Execute the notebook.