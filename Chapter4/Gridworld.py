import numpy as np

SIZE = 4	# the 4*4 gridworld

class Env:
	def step(self, s, action):
		"""action: 0-up, 1-right, 2-down, 3-left"""
		if s == 0 or s == 15:	# The terminal state
			return s, 0
		if action == 0 and s > 3:
			s = s - SIZE
		elif action == 1 and s != 3 and s != 7 and s != 11:
			s = s + 1
		elif action == 2 and s < 12:
			s = s + SIZE
		elif action == 3 and s != 4 and s != 8 and s != 12:
			s = s - 1
		return s, -1


def policy_evaluation(env, policy, gamma, theta):
	"""
	env: the environment
	policy: pi to be evaluated
	gamma: the discount factor
	theta: the thershold determining the accuracy
	return value: the value function v
	"""
	v = np.zeros(SIZE * SIZE)
	delta = theta
	while delta >= theta:
		delta = 0
		temp = v.copy()
		v.fill(0)
		for s in range(SIZE * SIZE):
			for a in range(4):
				sp, reward = env.step(s, a)
				v[s] += policy[s][a] * (reward + gamma * temp[sp])	# the trans is deterministic
			delta = max(delta, np.abs(v[s] - temp[s]))
	return v

if __name__ == '__main__':
	env = Env()
	v = np.zeros(SIZE * SIZE)
	policy = np.zeros((SIZE * SIZE, 4))	# |S| * |A|, the entry is the prob
	
	# Note the answer is same as Figure 4.1 in the book
	policy.fill(0.25)
	result = np.round(policy_evaluation(env, policy, 1, 0.001), decimals=0)
	for i in range(SIZE * SIZE):
		if i % SIZE == SIZE - 1:
			print(result[i])
		else:
			print(result[i], end='  ')

