import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

TRUE_VALUE = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]

class Env:
	def __init__(self):
		self.nS = 7  # 7 states in total
		self.state = 3

	def reset(self):
		self.state = 3  # start at C

	def step(self):
		""" return (state, reward, done) """
		self.state = random.choice((self.state - 1, self.state + 1))
		if self.state == 0:
			return self.state, 0, True
		elif self.state == 6:
			return self.state, 1, True
		else:
			return self.state, 0, False

def generate_episode(env):
	env.reset()
	episode = []
	while True:
		pre_state = env.state
		state, reward, done = env.step()
		episode.append((pre_state, reward))
		if done:
			return episode

def compute_RMS(v):
	accumulator = 0
	for i in range(1, 6):
		accumulator += (v[i] - TRUE_VALUE[i - 1]) ** 2
	return np.sqrt(accumulator)

def MC_prediction(env, num_episode=100):
	returns = defaultdict(list)
	error = []
	v = np.zeros(env.nS)
	for _ in range(num_episode):
		episode = generate_episode(env)
		G = episode[-1][1]  # since the discount factor is 1, all the return should be the same
		for state, reward in episode:
			returns[state].append(G)
			v[state] = np.mean(returns[state])
		error.append(compute_RMS(v))
	for key, value in returns.items():
		v[key] = np.mean(value)
	return v, error

def TD_0_prediction(env, alpha=0.1, num_episode=100):
	""" we dont have policy in this question"""
	v = np.zeros(env.nS)
	error = []
	for _ in range(num_episode):
		env.reset()
		while True:
			pre_state = env.state
			state, reward, done = env.step()
			v[pre_state] += alpha * (reward + v[state] - v[pre_state])
			if done:
				break
		error.append(compute_RMS(v))
	return v, error

def draw(env, v, error, label, num_episode=100):
	x = ['A', 'B', 'C', 'D', 'E']
	plt.subplot(1, 2, 1)
	plt.plot(x, v[1: env.nS - 1], label=label)

	plt.subplot(1, 2, 2)
	plt.plot(np.arange(num_episode), error, label=label)
	

if __name__ == '__main__':
	env = Env()
	v, error = MC_prediction(env)
	draw(env, v, error, 'MC')
	v, error = TD_0_prediction(env)
	draw(env, v, error, 'TD')


	plt.legend()
	plt.show()


