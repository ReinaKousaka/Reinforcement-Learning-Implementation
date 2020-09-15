import torch
import torch.nn as nn
import numpy as np
import gym
import random
from matplotlib import pyplot as plt

"""
Soft updates: theta_prime = tau * theta + (1 - tau) * theta_prime
"""

class Ornstein_Uhlenbeck_process:
    def __init__(self, tau, mu, sigma, dt, x0):
        """
        tau: time constant
        sigma: standard deviation
        mu: mean
        dt: time step
        """
        self.tau = tau
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_pre = None
        self.reset()

        self.sigma_bis = self.sigma * np.sqrt(2. / tau)

    def __call__(self):
        x = self.x_pre + self.dt * (self.mu - self.x_pre) / self.tau + \
                self.sigma_bis * np.sqrt(self.dt) * np.random.randn()
        self.x_pre = x
        return x
    
    def reset(self):
        self.x_pre = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer:
	def __init__(self, buffer_size=10000):
		self.buffer_size = buffer_size
		self.buffer = []
		self.index_pointer = 0
	
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.buffer_size:
			self.buffer.append((state, action, reward, next_state, done))
		else:
			self.buffer[self.index_pointer] = (state, action, reward, next_state, done)
			self.index_pointer = (self.index_pointer + 1) % self.buffer_size

	def sample(self, batch_size):
		samples = random.sample(self.buffer, batch_size)
		return samples
	
	def __len__(self):
		return len(self.buffer)

class Critic_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=256):
        super(Critic_Network, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden1)
        self._initialize(self.linear1)
    
        self.linear2 = nn.Linear(hidden1 + action_dim, hidden2)
        self._initialize(self.linear2)

        self.linear3 = nn.Linear(hidden2, 1)
        nn.init.uniform_(self.net[2].weight.data, -0.003, 0.003)  # the number from the paper
        nn.init.uniform_(self.net[2].bias.data, -0.003, 0.003)
    
    def _initialize(self, m):
        # Initialization and batch norm ideas from: https://www.youtube.com/watch?v=6Yd5WnYls_Y&t=1414s
        if type(m) == nn.Linear:
            f = 1 / np.sqrt(m.weight.data.size()[0])
            nn.init.uniform_(m.weight.data, -f, f)
            nn.init.uniform_(m.bias.data, -f, f)
    
    def forward(self, s, a):
        # inputs: state & action; output: Q(s, a)
        x = self.linear1(x)  # to Tensor ?
        x = nn.ReLU(x)
        x = self.linear2(torch.cat((x, action), dim=1))
        x = nn.ReLU(x)
        x = self.linear3(x)        
        return x

class Actor_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=256):
        super(Actor_Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU()
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim)
            nn.Tanh()
        )
        self.net.apply(self._initialize)
        nn.init.uniform_(self.net[2].weight.data, -0.003, 0.003)  # the number from the paper
        nn.init.uniform_(self.net[2].bias.data, -0.003, 0.003)
    
    def _initialize(self, m):
        if type(m) == nn.Linear:
            f = 1 / np.sqrt(m.weight.data.size()[0])
            nn.init.uniform_(m.weight.data, -f, f)
            nn.init.uniform_(m.bias.data, -f, f)
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().numpy()[0]   # ??
    

class Agent:
    def __init__(self, env ):
        self.env = env
        self.memory = ReplayBuffer()
        self.noise = Ornstein_Uhlenbeck_process()
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.critic_target = Critic_Network(state_dim=state_dim, action_dim=action_dim)
        self.critic_net = Critic_Network(state_dim=state_dim, action_dim=action_dim)
        self.actor_target = Actor_Network(state_dim=state_dim, action_dim=action_dim)
        self.actor_net = Actor_Network(state_dim=state_dim, action_dim=action_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def train(self):
        pass


if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')

    # N = Ornstein_Uhlenbeck_process(sigma=1, mu=10, tau=.05, x0=0, dt=0.001)
    # y = []
    # for _ in range(200):
    #     y.append(N())
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(np.arange(200), y)
    # plt.show()
    