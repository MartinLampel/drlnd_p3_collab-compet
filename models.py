import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def lecunishUniformInitializer(layer):
    fan_in = layer.weight.data.size()[0]
    limit = np.sqrt(1./fan_in)
    
    return (-limit, limit)

class ActorModel(nn.Module):
    """Actor (Policy) Model"""
    def __init__(self, input_size, action_size, hidden_units=[256,128]):
        super(ActorModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_units[0])
        self.fc2 = nn.Linear( hidden_units[0],  hidden_units[1])
        self.fc3 = nn.Linear( hidden_units[1], action_size)
        
        self.reset_parameters()
        
    def reset_parameters(self) :
        self.fc1.weight.data.uniform_(*lecunishUniformInitializer(self.fc1))
        self.fc2.weight.data.uniform_(*lecunishUniformInitializer(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3 )
        
    def forward(self, states):
        
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        return x
        
        

class CriticModel(nn.Module):
    """Critic (Value) Model"""
    def __init__(self, input_size, action_size, hidden_units=[256,128]):
        super(CriticModel, self).__init__()   
        
        self.fc1 = nn.Linear(input_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + action_size, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        
        self.reset_parameters()
        
    def reset_parameters(self) :
        self.fc1.weight.data.uniform_(*lecunishUniformInitializer(self.fc1))
        self.fc2.weight.data.uniform_(*lecunishUniformInitializer(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3 )
        
    def forward(self, states, actions):
        
        x = F.relu(self.fc1(states))    
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
                   
        return x

        
        