""" DQN on CartPole-v0 environment """

import gym
import torch
import torch.nn as nn
import time
import random
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt

class Net(nn.Module):
	def __init__(self, input_dims, num_actions):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dims, 128),
			nn.ReLU(),
			nn.Linear(128, num_actions),
		)
	
	def forward(self, x):
		return self.net(x)


class DQNAgent:
	def __init__(self, env, gamma=0.99, epsilon=0.1, lr=0.01, batch_size=16):
		self.gamma = gamma
		self.epsilon = epsilon
		self.lr = lr
		self.env = env
		self.memory = ReplayBuffer()
		self.batch_size = batch_size
		self.num_actions = env.action_space.n

		self.Q_eval = Net(num_actions=env.action_space.n, input_dims=env.observation_space.shape[0])
		self.optimizer = torch.optim.Adam(self.Q_eval.parameters(), lr=lr)  # optimizer
		self.Q_target = Net(num_actions=env.action_space.n, input_dims=env.observation_space.shape[0])
		self.Q_target.load_state_dict(self.Q_eval.state_dict())
		self.Q_target.eval()

	def choose_action(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, self.num_actions - 1)
		else:
			res = self.Q_eval(torch.FloatTensor(state)).argmax()
			res = res.detach().numpy()
			return res
	
	def update_model(self):
		samples = self.memory.sample(self.batch_size)
		loss = self._compute_loss(self, samples)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()
	
	def _compute_loss(self, samples):
		states, actions, rewards, next_states, dones = samples
		states = torch.FloatTensor(states)
		actions = torch.FloatTensor(actions)
		rewards = torch.FloatTensor(rewards)
		next_states = torch.FloatTensor(next_states)
		dones = torch.FloatTensor(dones)

		curr_q_value = self.Q_eval(states).gather(1, action)
		next_q_value = self.Q_target(next_states).max(dim=1, keepdim=True)[0].detach()
		mask = 1 - done
		target = (rewards + self.gamma * next_q_value * mask)
		loss = torch.nn.MSELoss(target, curr_q_value)
		return loss
	
	def train(self):
		num_episodes = 100
		state = self.env.reset()
		score = 0
		scores_lst = []
		losses_lst = []
		for _ in range(num_episodes):
			self.env.render()

			action = self.choose_action(state)
			next_state, reward, done, info = self.env.step(action)
			self.memory.push(state, action, reward, next_state, done)

			score += reward
			state = next_state
			if done:
				scores_lst.append(score)
				score = 0
				state = self.env.reset()
				continue
			if len(self.memory) >= self.batch_size:  # if we have enough replays
				loss = self.update_model()
				losses_lst.append(loss)
		self.env.close()
		return scores_lst, losses_lst




class ReplayBuffer:
	def __init__(self, buffer_size=100000):
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
		assert batch_size <= len(self.buffer)
		states, actions, rewards, next_states, dones = [], [], [], [], []
		for _ in range(batch_size):
			i = random.randint(0, len(self.buffer_size) - 1)
			state, action, reward, next_state, done = self.buffer[i]
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			next_states.append(next_state)
			dones.append(done)
		return (states, actions, rewards, next_states, dones)
	
	def __len__(self):
		return len(self.buffer)


if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	agent = DQNAgent(env=env, )
	score, loss = agent.train()
	fig, ax = plt.subplots(1, 1)
	ax.plot(np.arange(len(loss)), loss)
	plt.show()