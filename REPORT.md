# Project 2: Continuous Control

* [Introduction](#introduction)
* [Learning Algorithm](#learning-algorithm)   
* [Project Structure](#project-structure)   
* [Implementation](#implementation)   
* [Results](#results)   
* [Future work](#ideas-for-future-work)

## Introduction



## Learning Algorithm

> DDPG combines the actor-critic approach with the insights from the Deep Q Network. DQN learn value functions 
> using deep neural networks in a stalbe and robust way. They utilize a replay buffer to minimize correlations between samples
> and a target Q network. DDPG is model free, off policy actor-critic algorithm that can learn high-dimensional, continuous action spaces.  
> The authors of the [the paper](https://arxiv.org/pdf/1509.02971.pdf) highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.


* **Policy-based**: Unlike its value-based counterparts (like DQN), this method tries to
  learn the policy that the agent should use to maximize its objective directly. Recall
  that value-based methods (like Q-learning) try to learn an action-value function 
  to then recover the implict policy (greedy policy).

* **Actor-critic**: Actor-critic methods leverage the strengths of both policy-based and value-based methods.
  Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent.     
  Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action 
  pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based 
  agents, while requiring fewer training samples than policy-based agents.

* **Model-free**: We do not need access to the dynamics of the environment. This algorithm
  learns the policy using samples taken from the environment. We learn the action-value function
  (critic) by using *Q-learning* over samples taken from the world, and the policy by
  using the *Deterministic Policy Gradients* theorem over those samples.

* **Off-policy**: The sample experiences that we use for learning do not necessarily come
  from the actual policy we are learning, but instead come from a different policy (exploratory
  policy). As in DQN, we store these experiences in a replay buffer and learn from
  samples of this buffer, which might come from different timesteps (and potentially from
  different versions of the exploratory policy).
  
In DDPG the actor is the policy based part and the critic is the Q-learning part. 
To learn the actor function <a href="https://www.codecogs.com/eqnedit.php?latex=\mu(s|\theta^Q)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu(s|\theta^Q)" title="\mu(s|\theta^Q)" /></a> the DDPG algorithm performs gradient ascend w.r.t parameters to solve 

![](images/policy.png)



The critic Q(s,a) can learned by considering the Bellman equation

![](images/bellman.png)

which describes the optimal action value function. 
The loss function to learn the optimal action value function is 

![](images/qloss.png)

This mean squared loss function provides us with the information, how close the critic Q(s,a) comes to fullfill the Bellman equation. 

To explore the continous action space, a exploration policy is used:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu'&space;=&space;\mu(s|\theta^Q)&space;&plus;&space;\mathcal{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu'&space;=&space;\mu(s|\theta^Q)&space;&plus;&space;\mathcal{N}" title="\mu' = \mu(s|\theta^Q) + \mathcal{N}" /></a>

This policy adds to action from the actor function a noise sampled from a noise process. 

After we have covered the ideas behind the DDPG here is the full algorithm 
![](images/ddpgalgorithm.png)
  
  
## Project Structure

The code is written in PyTorch and Python3, executed in Jupyter Notebook

- Continuous_Control.ipynb	: Training and evaluation of the agent
- ddpgagent.py	: An agent that implement the DDPG algorithm
- models.py	: DNN models for the actor and the critic
- replaybuffer.py : Implementation of experience replay buffer
- checkpoint.pt : parameters for actor/critic network



## Implementation

### Models


#### Actor
![](images/actormodel.png)

#### Critic

![](images/crtiticmodel.png)

### Hyperparameters


```python
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
tau = 0.05
Actor learning rate = 1e-3
Critic learning rate = 1e-3
Gamma = 0.99
learn_interval = 1
learn_experiences = 10
Epsilon = 1
Epsilon decay = 0.99
batch_size = 128
BUFFER_SIZE = 1e6
```

### Results



![](images/score.png)

## Ideas for Future Work


1. Implement the D4PG, A3C algorithm and compare with this DDPG performance
2. Implement a prioritised Replay buffer

## References

* [1] [Sutton, Richard & Barto, Andrew. *Reinforcement Learning: An introduction.*](http://incompleteideas.net/book/RLbook2018.pdf)
* [2] [*Continuous control through deep reinforcement learning* paper by Lillicrap et. al.](https://arxiv.org/pdf/1509.02971.pdf)
* [3] [*Deterministic Policy Gradients Algorithms* paper by Silver et. al.](http://proceedings.mlr.press/v32/silver14.pdf)
* [4] [Post on *Deep Deterministic Policy Gradients* from OpenAI's **Spinning Up in RL**](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
* [5] [Post on *Policy Gradient Algorithms* by **Lilian Weng**](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
* [6] [A Gentle Introduction to Exploding Gradients in Neural Networks](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)
