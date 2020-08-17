import numpy as np

NUM_STATES = 48  # the 4*12 gridworld
NUM_ACTIONS = 4

class Env:
	def __init__(self):
		self.state = 36
		self.A = np.arange(NUM_ACTIONS)

	def reset(self):
		self.state = 36  # i.e. the left down corner

	def update(self, action):
		"""
		Take the given action 
		update the state inplace
		:return: (reward, done)
		"""
		if action == 0:  # UP
			if self.state > 11:
				self.state -= 12
		elif action == 1:  # RIGHT
			if self.state != 11 and self.state != 23 and self.state != 34 and self.state != 47:
				self.state += 1
		elif action == 2:  # DOWN
			if self.state < 36:
				self.state += 12
		else:  # LEFT
			if self.state != 0 and self.state != 12 and self.state != 24 and self.state != 36:
				self.state -= 1
		if self.state < 47 and self.state > 36:
			return -100, True
		elif self.state == 47:
			return 0, True
		else:
			return -1, False