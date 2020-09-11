""" Actor-Critic on CartPole-v0 environment """

import gym
import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dims, hidden_dims=128, num_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_actions)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        return self.net(observation)