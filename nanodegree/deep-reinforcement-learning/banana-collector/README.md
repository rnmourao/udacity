[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction


This is the first of three required projects of Udacity's Deep Reinforcement Learning course. It uses Unity's Reinforcement Learning framework.

For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 was provided for collecting a yellow banana, and a reward of -1 was provided for collecting a blue banana.  Thus, the agent's goal was to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space had 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent had to learn how to best select actions.  Four discrete actions were available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task was episodic, and the environment was solved with the agent getting an average score of +13 over 100 consecutive episodes.

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
git clone https://github.com/rnmourao/udacity-bananas.git
cd udacity-bananas
pip install -r requirements.txt
```

5. Download the environment.

    Please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

6. Place the file into udacity-bananas folder, and unzip (or decompress) the file. 

7. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

### Executing the Notebook


1. Start Jupyter.

```bash
jupyter notebook
```

2. Open the Navigation.ipynb file.


3. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)


4. Execute the notebook.