import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

LEFT =0 
RIGHT = 1
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)

class Env:
	def __init__(self):
		self.state = 0

	def reset(self):
		self.state = 0
		return 0

	def step(self, action):
		if self.state == 1:
			self.state = 2 if action == LEFT else 0
			return self.state, -1, False
		self.state = self.state - 1 if action == LEFT else self.state + 1
		self.state = max(0, self.state)
		if self.state == 3:
			return self.state, -1, True
		else:
			return self.state, -1, False


class REINFORCE(nn.Module):
	def __init__(self, lr):
		super(REINFORCE, self).__init__()
		# policy network
		self.linear1 = nn.Linear(2, 10)
		self.relu = nn.ReLU()
		self.linear2 = nn.Linear(10, 1)

		self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

	def forward(self, x):
		out = self.linear1(x)
		out = self.relu(out)
		out = self.linear2(out)
		out = torch.exp(out)
		out = torch.softmax(out, dim=0)
		return out


class Agent:
	def __init__(self, lr, env, gamma=1):
		self.model = REINFORCE(lr)
		self.env = env
		self.gamma = gamma
	
	def get_action(self, state):
		# The state does not matter in this question.
		if random.random() < self.model(X)[0].item():
			return 0
		else:
			return 1
	
	def generate_episode(self):
		done = False
		trajectory = []
		pre_state = self.env.reset()
		while not done:
			action = self.get_action(pre_state)
			state, reward, done = self.env.step(action)
			trajectory.append([pre_state, action, reward])
			if done:
				return trajectory
			pre_state = state
	
	def learn(self):
		n_episodes = 1000
		y = []
		for _ in range(n_episodes):
			trajectory = self.generate_episode()
			G = 0
			total_reward = 0
			for state, action, reward in reversed(trajectory):
				total_reward += reward
				G = G * self.gamma + reward
				pi = self.model(X)[action] 
				loss = -1.0 * G * torch.log(pi)
				self.model.optimizer.zero_grad()
				loss.backward()
				self.model.optimizer.step()
			y.append(total_reward)
		return y	
		



if __name__ == "__main__":
	env = Env()
	agent1 = Agent(1/(2**13), env)
	agent2 = Agent(1/(2**12), env)
	y1 = agent1.learn()
	y2 = agent2.learn()
	fig, ax = plt.subplots(1, 1)
	ax.plot(np.arange(1000) + 1, y1, label='13')
	ax.plot(np.arange(1000) + 1, y2, label='12')
	ax.legend(loc='best')
	plt.show()