import numpy as np
import random
import matplotlib.pyplot as plt
import sys

NUM_STATES = 1000

class Env:
	def __init__(self):
		self.S = np.arange(1, NUM_STATES + 1)  # states from 1 to 1000
		self.state = 500  # We start from 500
	
	def reset(self):
		self.state = 500

	def step(self):
		""" Return (state, reward, done) """
		self.state = random.randint(self.state - 100, self.state + 100)
		self.state = max(self.state, 1)
		self.state = min(self.state, NUM_STATES)
		if self.state == NUM_STATES:
			return self.state, 1, True
		elif self.state == 1:
			return self.state, -1, True
		else:
			return self.state, 0, False

class ValueFunction:
	"""The state aggregation value function"""
	def __init__(self, num_group=10):
		self.num_group =  num_group
		self.group_size = NUM_STATES // num_group
		self.w = np.zeros(num_group)  # The weight vector

	def update(self, state, x):
		# Due to State aggregation, only the state's group has gradient 1, all others' component are 0,
		# We skip the vector dot product here.
		self.w[(state - 1) // self.group_size] += x

	def get_value(self, state):
		""" return the approximation of v(state, weight)"""
		return self.w[(state - 1) // self.group_size]


def get_episode(env):
	env.reset()
	episode = []
	while True:
		state, reward, done = env.step()
		episode.append((state, reward))
		if done:
			return episode

def draw(env, value_func):
	y = np.zeros(NUM_STATES)
	for i in range(1, NUM_STATES + 1):
		y[i - 1] = value_func.get_value(i)
	plt.plot(env.S, y, label='Approximation MC value V')
	

def draw_true_value(env):
	y = np.zeros(NUM_STATES)
	for i in range(1, NUM_STATES + 1):
		y[i - 1] = (2 / NUM_STATES) * (i - (NUM_STATES // 2))
	plt.plot(env.S, y, label='True value V')

def Gradient_MC_Approximation(env, num_episodes=100000, alpha=2e-5):
	# The environment has only 1 action, so there's no input for policy
	value_func = ValueFunction(10)  # Create the 10 groups
	for _ in range(num_episodes):
		if _ % 1000 == 0:
			print('\rEpisode {}/{}.'.format(_, num_episodes), end='')  # the progress bar
			sys.stdout.flush()

		episode = get_episode(env)
		# Since gamma==1, the G for all states in this episode are the same
		G = 1 if episode[-1][1] == 1 else -1
		for state, reward in episode:
			x = alpha * (G - value_func.get_value(state))
			value_func.update(state, x)
	draw(env, value_func)


if __name__ == '__main__':
	env = Env()
	draw_true_value(env)
	Gradient_MC_Approximation(env)

	plt.xlabel('State')
	plt.ylabel('Value scale')
	plt.legend()
	plt.show()