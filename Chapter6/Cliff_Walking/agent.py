import numpy as np
import random
from matplotlib import pyplot as plt

NUM_STATES = 48  # the 4*12 gridworld
NUM_ACTIONS = 4

class Agent:
	def __init__(self, env, gamma, alpha, epsilon):
		self.Q = np.zeros((NUM_STATES, NUM_ACTIONS))
		self.gamma = gamma
		self.alpha = alpha
		self.epsilon = epsilon
		self.env = env

	def _choose_action(self, state):
		""" return the action based on the state"""
		if random.random() < self.epsilon:
			return random.choice(self.env.A)
		else:
			return np.argmax(self.Q[state])

	def learning(self, episode):  
		raise NotImplementedError

	def draw(self, episode, lst, name_label, average_range=10):
		"""name_label: the name of the label
		lst: the data list
		average_range: the number to average over during the episodes
		"""
		smoothed = np.copy(lst)
		for i in range(average_range, episode):
			smoothed[i] = np.mean(lst[i - average_range: i + 1])
		plt.plot(np.arange(episode), smoothed, label=name_label)

	

class AgentQlearning(Agent):
	# This is the agent that trained by Q-learning
	def learning(self, episode):
		reward_lst = np.zeros(episode)
		for i in range(episode):
			self.env.reset()  # Initialize S
			while True:
				pre_state = self.env.state
				action = self._choose_action(self.env.state)  # Choose action A from S
				reward, done = self.env.update(action)  # Take the action, observe S', R
				next_state = self.env.state
				reward_lst[i] += reward
				if done:
					self.Q[pre_state, action] += self.alpha * (reward - self.Q[pre_state, action])
					break
				self.Q[pre_state, action] += self.alpha * \
					(reward + self.gamma * np.max(self.Q[next_state]) - self.Q[pre_state, action])
		self.draw(episode, reward_lst, 'Q-learning', 10)
		


class AgentSarsa(Agent):
	# This is the agent that trained by SARSA
	def learning(self, episode):
		reward_lst = np.zeros(episode)
		for i in range(episode):
			self.env.reset()  # Initialize S
			action = self._choose_action(self.env.state)  # Choose action A from S (epsilon-greedy)
			while True:
				pre_state = self.env.state
				reward, done = self.env.update(action)  # Take action A, observe S', R
				reward_lst[i] += reward
				if done:
					self.Q[pre_state, action] += self.alpha * (reward - self.Q[pre_state, action])
					break
				else:
					action_prime = self._choose_action(self.env.state)  # Choose A' from S' (epsilon-greedy)
					next_state = self.env.state
					self.Q[pre_state, action] += self.alpha * \
						(reward + self.gamma * self.Q[next_state, action_prime] - self.Q[pre_state, action])
				action = action_prime
		self.draw(episode, reward_lst, 'Sarsa', 10)