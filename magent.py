import torch
import numpy as np
import random
import ddpgagent
from replaybuffer import ReplayBuffer

class MultipleAgents:
    def __init__(self, state_size, action_size, num_agents, batch_size=1024, buffer_size=1e6, seed=0,
                 gamma=0.99, learn_interval=1, learn_experiences=5, tau=0.01):
        
        self.agents = [ddpgagent.DDPGAgent(state_size, action_size, seed, tau) for _ in range(num_agents)]
        
        self.num_agents = num_agents
        self.gamma = gamma
        self.learn_interval = learn_interval
        self.learn_experiences = learn_experiences
        self.batch_size = batch_size
        
       
        self.memory = ReplayBuffer(action_size, int(buffer_size), batch_size, seed)

        
    def act(self, states, train=True):
        
        actions = [agent.act(state, train) for agent, state in zip(self.agents,states)]
        
        return np.array(actions)
    
    
    def step(self, t, states, actions, rewards, next_states, dones):
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):       
            self.memory.add(state, action, reward, next_state, done)
               
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > (2 * self.batch_size) and (t % self.learn_interval) == 0:
            for _ in range(self.learn_experiences):
                experiences = self.memory.sample()
                for agent in self.agents:
                    agent.learn(experiences, self.gamma)
                    
    def reset(self):
        for agent in self.agents:
            agent.reset()
    
