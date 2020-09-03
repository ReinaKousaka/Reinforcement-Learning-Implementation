import numpy as np
import torch
from matplotlib import pyplot as plt
import sys

LEFT, RIGHT = -1, 1

class Env():
	""" S = {0, 1, 2, 3}, where 3 is the terminal state
		A = {LEFT, RIGHT}
	"""
	def __init__(self):
		self.state = 0

	def reset(self):
		self.state = 0

	def step(self, action):
		factor = -1 if self.state == 1 else 1
		self.state += factor * action
		done = True if self.state == 3 else False
		return -1, done 

class Agent():
	def __init__(self, env, alpha):
		self.env = env
		self.alpha = alpha
		self.x = torch.tensor([[0, 1], [1, 0]], dtype=torch.double)
		self.theta = torch.tensor([0., 0.], dtype=torch.double, requires_grad=True)
			

	def get_episode(self):
		self.env.reset()
		trajectory = []
		while True:
			pre_state = self.env.state
			h = torch.matmul(self.theta, self.x)
			temp = torch.exp(h[0]) / (torch.sum(torch.exp(h)))
			action = LEFT if np.random.random() < temp else RIGHT
			reward, done = self.env.step(action)
			trajectory.append((pre_state, action, reward))
			if done:
				return trajectory
	
	def MC_policy_gradient(self, num_episodes, gamma=1):
		self.theta = torch.tensor([0., 0.], dtype=torch.double, requires_grad=True)
		total_reward = np.zeros(num_episodes)
		for _ in range(num_episodes):
			print('\r{}/{}.'.format(_, num_episodes), end=' ')
			print(self.theta)
			sys.stdout.flush()  # for debugging

			trajectory = self.get_episode()
			rewards = list(zip(*trajectory))[2]
			T = len(trajectory)
			for t, (state, action, reward) in enumerate(trajectory):
				discount_factor = gamma ** np.arange(T - t)
				G = np.dot(discount_factor, rewards[t:])
				
				h = torch.matmul(self.theta, self.x)
				if action == LEFT:
					pi = torch.exp(h[0]) / (torch.sum(torch.exp(h)))
				else:
					pi = torch.exp(h[1]) / (torch.sum(torch.exp(h)))
				pi = torch.log(pi)
				pi.backward() 
				self.theta.data += self.alpha * (gamma ** t) * G * self.theta.grad
				self.theta.grad.zero_()
			total_reward[_] = np.sum(rewards)
		return total_reward

env = Env()
num_episodes = 1000
num_runs = 100
agent = Agent(env, 1e-9)
# records = []
# for run in range(num_runs):
# 	records.append(agent.MC_policy_gradient(num_episodes))
# y = np.mean(records, axis=0)
y = agent.MC_policy_gradient(num_episodes)
x = np.arange(num_episodes) + 1
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, label='1e-6')
ax.legend()
plt.show()