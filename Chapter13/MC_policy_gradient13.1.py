import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from tqdm import tqdm

LEFT = 0
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

class Agent:
	def __init__(self, env, lr, gamma=1):
		self.env = env
		self.lr = lr
		self.theta = torch.tensor([-1.47, 1.47], requires_grad=True)
		self.gamma = gamma
	
	def reset(self):
		self.theta = torch.tensor([-1.47, 1.47], requires_grad=True)

	def choose_action(self, state):
		# the state makes no difference in our case
		with torch.no_grad():
			h = torch.matmul(X, self.theta)
			pi = torch.softmax(h, dim=0)
			prob_left = pi[0].item()
		return LEFT if random.random() < prob_left else RIGHT
	
	def get_episode(self):
		trajectory = []
		pre_state = self.env.reset()
		while True:
			action = self.choose_action(pre_state)
			state, reward, done = self.env.step(action)
			trajectory.append((pre_state, action, reward))
			if done:
				return trajectory
			pre_state = state
	
	def learn(self):
		n_epochs = 5
		n_episodes = 1000
		y = np.zeros(n_episodes)
		for epoch in tqdm(range(n_epochs)):
			self.reset()
			for _ in range(n_episodes):
				trajectory = self.get_episode()
				G = 0.
				total_reward = 0
				for state, action, reward in reversed(trajectory):
					G = G * self.gamma + reward
					total_reward += reward
					h = torch.matmul(X, self.theta)
					pi = torch.softmax(h, dim=0)
					pi = torch.log(pi)
					pi_action = pi[action]
					pi_action.backward()		# gradient!
					with torch.no_grad():
						self.theta += self.lr * G * self.theta.grad
					self.theta.grad.zero_()
				y[_] += total_reward
		y = y / n_epochs
		return y
		


if __name__ == '__main__':
	env = Env()
	agent1 = Agent(env, 2e-4)
	y1 = agent1.learn()
	agent2 = Agent(env, 2e-5)
	y2 = agent2.learn()
	agent3 = Agent(env, 2e-3)
	y3 = agent1.learn()
	x = np.arange(1000) + 1
	fig, ax = plt.subplots(1, 1)
	ax.plot(x, y1, label='2e-4')
	ax.plot(x, y2, label='2e-5')
	ax.plot(x, y3, label='2e-3')
	ax.set_xlabel('Episode')
	ax.set_ylabel('Total reward on episode')
	ax.legend()
	plt.show()

