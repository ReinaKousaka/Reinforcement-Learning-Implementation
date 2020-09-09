import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from tqdm import tqdm

LEFT = 0
RIGHT = 1
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
MAX_EPISODE_LENGTH = 100

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

class REINFORCE_Agent:
	def __init__(self, env, lr, gamma=1):
		self.env = env
		self.lr = lr
		self.theta = torch.tensor([-1.47, 1.47])
		self.gamma = gamma
	
	def reset(self):
		self.theta = torch.tensor([-1.47, 1.47])

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
		cnt = 0
		while True:
			cnt += 1
			action = self.choose_action(pre_state)
			state, reward, done = self.env.step(action)
			trajectory.append((pre_state, action, reward))
			if done or cnt == MAX_EPISODE_LENGTH:
				return trajectory
			pre_state = state
	
	def learn(self):
		n_epochs = 50
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
					grad_ln_action = X[action] - (pi[0].item() * X[0] + pi[1].item() * X[1])
					self.theta += self.lr * G * grad_ln_action
				y[_] += total_reward
		y = y / n_epochs
		return y

class REINFORCE_Basline_Agent(REINFORCE_Agent):
	def __init__(self, env, alpha_theta, alpha_w, gamma=1):
		super(REINFORCE_Basline_Agent, self).__init__(env, alpha_theta, gamma)
		self.w = 0  # The state-value weight vector is dim-1 in this question!
		self.alpha_w = alpha_w
	
	def reset(self):
		self.theta = torch.tensor([-1.47, 1.47])
		self.w = 0

	def learn(self):
		n_episodes = 1000
		n_epochs = 50
		y = np.zeros(n_episodes)
		for epoch in tqdm(range(n_epochs)):
			self.reset()
			for _ in range(n_episodes):
				trajectory = self.get_episode()
				G = 0
				total_reward = 0
				for state, action, reward in reversed(trajectory):
					G = G * self.gamma + reward
					total_reward += reward
					delta = G - self.w
					self.w = self.w + self.alpha_w * delta * 1  # the grad of the baseline is 1
					
					h = torch.matmul(X, self.theta)
					pi = torch.softmax(h, dim=0)
					grad_ln_action = X[action] - (pi[0].item() * X[0] + pi[1].item() * X[1])
					self.theta += self.lr * delta * grad_ln_action
					
				y[_] += total_reward
		return y / n_epochs
				
				

if __name__ == '__main__':
	env = Env()
	agent1 = REINFORCE_Agent(env, 2e-4)
	y1 = agent1.learn()
	agent2 = REINFORCE_Basline_Agent(env, 2e-3, alpha_w=2e-2)
	y2 = agent2.learn()
	
	x = np.arange(1000) + 1
	fig, ax = plt.subplots(1, 1)
	ax.plot(x, y1, label='REINFORCE')
	ax.plot(x, y2, label='REINFORCE with basline')

	ax.set_xlabel('Episode')
	ax.set_ylabel('Total reward on episode')
	ax.legend()
	plt.show()

