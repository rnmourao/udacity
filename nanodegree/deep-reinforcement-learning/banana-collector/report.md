# Implementation Report


## Introduction


This document provides a brief explanation of this project. 

It describes the learning algorithm, Deep Q-Network, and the details of the neural network architecture. 

Finally, it suggests some ideas to improve this project, that may be used in future works.


## Deep Q-Networks

According to [1], Deep reinforcement learning (DRL) is a machine learning approach to artificial intelligence concerned with creating computer programs that can solve problems requiring intelligence.

Deep Q-Network (DQN) is a popular DRL algorithm that, as explained by [2], approximates a state-value function in a Q-Learning framework with a neural network. In the Atari Games case, they take in several frames of the game as an input and output state values for each action as an output.

To improve the learning, this project added to the DQN two techniques: **Experience Replay** and **Q-Target**. The first one saves episodes steps in memory, and later uses a sample of these steps to train the model, similar to a supervised learning algorithm. On the other hand, Q-Target uses an auxiliary model to stabilize the agent's actions, instead of change the network in every step.

The Experience Replay used a memory size of 10,000 steps, and a sample of 64 steps were used on each learning cycle. The Q-Target model was update every 4 steps.

## Neural Network Architecture

The Neural Network was developed using **Pytorch** framework, and is described as:

- First layer: 37 nodes, according to the number of input features
- Second layer (hidden): 64 nodes
- Third layer (hidden): 64 nodes
- Last layer: 4 nodes, according to the number of actions

The hidden layers used Rectified Linear (RELU) activation function. The optimizer used was **Adam**, at a Learning Rate of 0.0005.

The Reinforcement Learning **hyperparameters** are:

- Number of episodes: 5000
- Maximum number of time steps per episode: 1000
- Maximum Exploration Rate: 1.0
- Exploration Rate Decay: 0.995
- Minimum Exploration Rate: 0.1
- Discount Factor (GAMMA): 0.99
- Soft update rate for the target model (TAU): 0.001
- Stop criteria was a averaged sum of rewards above +13

## Results

The agent achieved the minimum score of +13 after 482 episodes.

## Futures Works

There are other techniques that could be used to improve the result, such as:

- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)

## References

1. MORALES, M., **Grokking Deep Reinforcement Learning**, Simon and Schuster, 2020.

2. [Deep Q-Network](https://paperswithcode.com/method/dqn), Papers with Code, accessed on Oct 16th, 2021.