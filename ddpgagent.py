import copy
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from replaybuffer import ReplayBuffer
import models

BATCH_SIZE = 1024
BUFFER_SIZE = int(1e6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, seed=0, tau=0.01,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, learn_interval=20, learn_experiences=10):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.tau = tau
        self.gamma = gamma
        
        
        # actor and critic
        self.actor = models.ActorModel(state_size, action_size).to(device)
        self.actor_target = models.ActorModel(state_size, action_size).to(device)
        
        self.critic = models.CriticModel(state_size, action_size).to(device)      
        self.critic_target = models.CriticModel(state_size, action_size).to(device)
        
        # optimizers 
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.noise_process = OUNoise(action_size, 0)
        self.noise_decay = 0.99
        self.epsilon = 1
        
        self.learn_interval = learn_interval
        self.learn_experiences = learn_experiences
        
    def reset(self):
        self.noise_process.reset()
    
    def step(self, t, states, actions, rewards, next_states, dones):
        # Save experience in replay memory       
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):       
            self.memory.add(state, action, reward, next_state, done)
               
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > 2*BATCH_SIZE and (t % self.learn_interval) == 0:
            for _ in range(self.learn_experiences):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
          
    def act(self, state, train=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()

        self.actor.train()
        if train:
            actions += self.noise_process.sample() * self.epsilon
            self.epsilon *= self.noise_decay

        actions = np.clip(actions, -1, 1)

        return actions

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        self.update_critic(states, actions, next_states, rewards, dones)  
        self.update_actor(states)
        
        self.soft_update(self.actor, self.actor_target, self.tau)    
        self.soft_update(self.critic, self.critic_target, self.tau)  
        
        
    def update_critic(self, states, actions, next_states, rewards, dones):
        """update the critic network using the loss calculated as 
           Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        """
        
        next_actions = self.actor_target(next_states)
        next_state_action_values = self.critic_target(next_states, next_actions)
                
        expected_values = rewards + (1.0 - dones) * self.gamma * next_state_action_values
        state_action_values = self.critic(states, actions)
        
        critic_loss = F.mse_loss(expected_values, state_action_values)
        self.critic_optim.zero_grad()                
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()

    
    def update_actor(self,states):
        """update the actor network using the critics Qvalue for (state,action) pair """
        
        actions = self.actor(states)
        loss = -self.critic(states, actions).mean()
        
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for local_params, target_params in zip(local_model.parameters(),target_model.parameters()):            
            target_params.data.copy_(local_params*tau + (1-tau)*target_params)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        
        Params:
        =======
             size (int or tuple): sample space 
             mu (float): mean
             theta (float):optimal parameter
             sigma (float) :variance
        """       
        
        self.size=size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        np.random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        
        return self.state
        
        