import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

TRUE_VALUE = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0]
INITIAL_VALUE = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]

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
		accumulator += ((v[i] - TRUE_VALUE[i]) ** 2) / 5
	return np.sqrt(accumulator)

def MC_prediction(env, alpha=0.05, num_episode=100):
	v = INITIAL_VALUE.copy()
	error, total_error = [], 0
	for _ in range(num_episode):
		episode = generate_episode(env)
		G = episode[-1][1]
		for state, reward in episode:
			v[state] += alpha * (G - v[state])
		total_error += compute_RMS(v)
		error.append(total_error / (_ + 1))
	return v, error

def TD_0_prediction(env, alpha=0.05, num_episode=100):
	""" we dont have policy in this question"""
	v = INITIAL_VALUE.copy()
	error, total_error = [], 0
	for _ in range(num_episode):
		env.reset()
		while True:
			pre_state = env.state
			state, reward, done = env.step()
			v[pre_state] += alpha * (reward + v[state] - v[pre_state])
			if done:
				break
		total_error += compute_RMS(v)
		error.append(total_error / (_ + 1))
	return v, error

def draw(env, num_episode=100):
	x = ['A', 'B', 'C', 'D', 'E']
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

	v, error = TD_0_prediction(env)
	ax1.plot(x, v[1: 6], label='TD', color='blue')
	ax2.plot(np.arange(num_episode), error, label='TD', color='blue')
	v, error = MC_prediction(env)
	ax1.plot(x, v[1: 6], label='MC', color='red')
	ax2.plot(np.arange(num_episode), error, label='MC', color='red')
	ax1.plot(x, TRUE_VALUE[1: 6], label='True Values', color='green')
	
	ax1.set(title='Estimated Value')
	ax2.set(title='Averaged Empirical RMS Error')
	ax1.legend()
	ax2.legend()
	fig.tight_layout()
	plt.show()

if __name__ == '__main__':
	env = Env()
	draw(env)





