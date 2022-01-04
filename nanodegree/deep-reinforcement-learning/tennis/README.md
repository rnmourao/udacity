[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

## Introduction

For this project, I worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, my agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, I added up the rewards that each agent received (without discounting), to get a score for each agent. This yielded 2 (potentially different) scores. I then took the maximum of these 2 scores.
- This yielded a single **score** for each episode.

## Solving the Environment

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

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
cd udacity/nanodegree/deep-reinforcement-learning/tennis
pip install -r requirements.txt
```

5. Download the environment.

    - Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.

6. Place the file into tennis folder, and unzip (or decompress) the file. 

7. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

### Executing the Notebook


1. Start Jupyter.

```bash
jupyter notebook
```

2. Open the Tennis.ipynb file.


3. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)


4. Execute the notebook.