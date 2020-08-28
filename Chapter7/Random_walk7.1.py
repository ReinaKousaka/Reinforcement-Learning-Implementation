import numpy as np
import random
import matplotlib.pyplot as plt
import sys

TRUE_VALUE = np.arange(-20, 22, 2) / 20
TRUE_VALUE[0] = 0
TRUE_VALUE[20] = 0

class Env:
	def __init__(self):
		self.nS = 21  # 21 states in total
		self.state = 10

	def reset(self):
		self.state = 10  # start at C

	def step(self):
		""" return (state, reward, done) """
		self.state = random.choice((self.state - 1, self.state + 1))
		if self.state == 0:
			return self.state, 0, True
		elif self.state == 20:
			return self.state, 1, True
		else:
			return self.state, 0, False


def compute_RMS(v):
	return np.sqrt(np.sum((v - TRUE_VALUE) ** 2 / 19))


def n_step_TD_prediction(env, n, alpha=0.05, gamma=1, num_episode=10, num_repetition=100):
	""" we dont have policy in this question"""
	error_lst = np.zeros(num_repetition)
	for run in range(num_repetition):
		v = np.zeros(21)
		total_error = 0
		for _ in range(num_episode):
			episode = []
			env.reset()
			T = float('inf')
			t = 0
			done = False
			while True:
				if not done:
					pre_state = env.state
					state, reward, done = env.step()
					episode.append((pre_state, reward))
					if done:
						T = t + 1
				update_time = t - n
				if update_time >= 0:
					target, discount_factor = 0, 1
					for i in range(update_time, min(T, update_time + n)):
						target += discount_factor * episode[i][1]
						discount_factor *= gamma
					if update_time + n < T:
						target += discount_factor * v[(episode[update_time + n][0])]
					v[episode[update_time][0]] += alpha * (target - v[episode[update_time][0]])
				t += 1
				if update_time == T - 1:
					break
			total_error += compute_RMS(v)
		error_lst[run] = total_error / num_episode
	return np.mean(error_lst)

def draw(env):
	alphas = np.arange(0, 1.1, 0.1)
	steps = np.power(2, np.arange(10))
	fig, ax = plt.subplots(nrows=1, ncols=1)
	i = 0
	for n in steps:
		print('\rProgress: {}%'.format((i / len(steps)) * 100), end='')
		sys.stdout.flush()
		i += 1
		errors = []
		for alpha in alphas:
			errors.append(n_step_TD_prediction(env, n, alpha=alpha))
		ax.plot(alphas, errors, label='n={}'.format(n))
	ax.legend(loc='best')
	ax.set(title='Performance of n-step TD methods as a function of alpha')
	ax.set_xlabel('Alpha')
	ax.set_ylabel('Average RMS error')
	plt.show()


if __name__ == '__main__':
	env = Env()
	draw(env)