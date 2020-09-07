import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm


LEFT =0 
RIGHT = 1
MAX_EPISODES = 1000
MAX_TIMESTEPS = 100

ALPHA = 1/(2**13)
GAMMA = 1
class Env():
	""" S = {0, 1, 2, 3}, where 3 is the terminal state
		A = {LEFT, RIGHT}
	"""
	def __init__(self):
		self.state = 0

	def reset(self):
		self.state = 0

	def step(self, action):
		if self.state == 1 and action == LEFT:
			self.state += 1
		elif self.state == 1 and action == RIGHT:
			self.state -= 1
		elif action == LEFT:
			self.state -= 1
		else:
			self.state += 1
		done = True if self.state == 3 else False
		return -1, done 


class reinforce(nn.Module):
	def __init__(self):
		super(reinforce, self).__init__()
		# policy network
		self.linear = nn.Linear(2, 1)

	def forward(self, x):
		out = self.linear(x)
		out = torch.exp(out)
		out = torch.softmax(out, dim=0)
		return out

	def pi(self, s, a):
		probs = self.forward(X)
		return probs[a]

	def update_weight(self, states, actions, rewards, optimizer):
		G = Variable(torch.Tensor([0]))
		# for each step of the episode t = T - 1, ..., 0
		# r_tt represents r_{t+1}
		for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
			G = Variable(torch.Tensor([r_tt])) + GAMMA * G
			loss = (-1.0) * G * torch.log(self.pi(s_t, a_t))
			# update policy parameter \theta
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

def get_action(agent):
	if np.random.random() < agent(X)[0].item():
		return 0
	else:
		return 1


X = torch.tensor([[0, 1],
					[1, 0]], dtype=torch.float32)


if __name__ == "__main__":
	env = Env()
	total_reward = []
	num_epochs = 10
	for epoch in tqdm(range(num_epochs)):
		agent = reinforce()
		optimizer = optim.Adam(agent.parameters(), lr=ALPHA)
		for i_episode in range(MAX_EPISODES):
			state = env.reset()
			states = []
			actions = []
			rewards = [0]   # no reward at t = 0
			for timesteps in range(MAX_TIMESTEPS):
				action = get_action(agent)
				states.append(state)
				actions.append(action)
				reward, done= env.step(action)
				rewards.append(reward)
				if done:
					break
			agent.update_weight(states, actions, rewards, optimizer)
			if epoch == 0:
				total_reward.append([np.sum(rewards)])
			else:
				total_reward[i_episode].append(np.sum(rewards))
			
	for i in range(MAX_EPISODES):
		total_reward[i] = np.mean(total_reward[i])
	fig, ax = plt.subplots(1, 1)
	ax.plot(np.arange(1000) + 1, total_reward)
	plt.show()


