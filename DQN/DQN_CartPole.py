""" DQN on CartPole-v0 environment """

import gym
import torch
import torch.nn as nn
import random
import numpy as np
from matplotlib import pyplot as plt
import sys

sys.setrecursionlimit(3000)


class Net(nn.Module):
	def __init__(self, input_dims, num_actions):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dims, 128),
			nn.ReLU(),
			nn.Linear(128, num_actions),
		)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)
	
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

	def choose_action(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, self.num_actions - 1)
		else:
			res = self.Q_eval(torch.FloatTensor(state)).argmax()
			res = res.detach().numpy()
			return res
	
	def update_model(self):
		samples = self.memory.sample(self.batch_size)
		loss = self._compute_loss(samples)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()
	
	def _compute_loss(self, samples):
		target_list = torch.zeros(len(samples))
		pred_list = torch.zeros(len(samples))
		for i, (state, action, reward, next_state, done) in enumerate(samples):
			if done:
				target_list[i] = reward
			else:
				self.Q_target.load_state_dict(self.Q_eval.state_dict())
				temp = self.Q_target(torch.from_numpy(next_state).float()).detach()
				target_list[i] = reward + self.gamma * torch.max(temp)
			pred_list[i] = self.Q_eval(torch.from_numpy(state).float())[action] 
		loss = torch.nn.MSELoss()(target_list, pred_list)  
		return loss
	
	def train(self, time_step=2000):
		state = self.env.reset()
		score = 0
		scores_lst = []
		losses_lst = []
		for _ in range(time_step):
			#self.env.render()
			action = self.choose_action(state)
			next_state, reward, done, info = self.env.step(action)
			self.memory.push(state, action, reward, next_state, done)

			if len(self.memory) >= self.batch_size:  # if we have enough replays
				loss = self.update_model()
				losses_lst.append(loss)
			score += reward
			state = next_state
			if done:
				scores_lst.append(score)
				score = 0
				state = self.env.reset()
			
		self.env.close()
		return scores_lst, losses_lst
	
	def test(self, render):
		state = self.env.reset()
		done = False
		score = 0
		while not done:
			if render:
				self.env.render()
			action = self.choose_action(state)
			next_state, reward, done, info = self.env.step(action)
			state = next_state
			score += reward
		self.env.close()
		return score



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


if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	agent = DQNAgent(env=env)
	print('The score before training: {}'.format(agent.test(render=False)))
	score, loss = agent.train()
	print('The score after training: {}'.format(agent.test(render=True)))
	

	# Draw
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.plot(np.arange(len(loss)), loss)
	ax1.set_ylabel('Loss')
	ax2.plot(np.arange(len(score)), score)
	ax2.set_ylabel('Score')
	fig.tight_layout()
	plt.show()