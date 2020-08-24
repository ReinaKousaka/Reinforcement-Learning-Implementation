import numpy as np
import gym
from collections import defaultdict
from matplotlib import pyplot as plt
from plot_utils import plot_blackjack_values

STICK = 0
HIT = 1

def generate_episode(env):
	episode = []  # A list of tuples (state, action, reward)
	state = env.reset()
	while True:
		action = STICK if state[0] >= 20 else HIT  # We stick only if the sum is 20 or 21
		next_state, reward, done, info = env.step(action)
		episode.append((state, action, reward))
		state = next_state
		if done:
			break
	return episode

def on_policy_MC_prediction_first(env, num_episode, gamma=1):
	""" The first-visit MC prediction """
	returns = defaultdict(list)
	for _ in range(num_episode):
		episode = generate_episode(env)
		G = np.zeros(len(episode))
		for i, t in enumerate(reversed(episode)):
			state, action, reward = t
			G[i] = gamma * G[i] + reward
		state_appear = {}
		for i, t in enumerate(episode):
			state, action, reward = t
			if state not in state_appear:
				state_appear[state] = 1
				returns[state].append(G[i])
	# Take the average
	V = {}
	for key, value in returns.items():
		V[key] = np.mean(value)
	return V

def on_policy_MC_prediction_every(env, num_episode, gamma=1):
	""" The every-visit version MC prediction"""
	returns = defaultdict(list)
	for _ in range(num_episode):
		episode = generate_episode(env)
		states, actions, rewards = zip(*episode)
		discount_vector = np.array([gamma ** i for i in range(len(rewards) + 1)])
		for i, state, in enumerate(states):
			returns[state].append(np.dot(rewards[i:], discount_vector[:len(rewards) - i]))
	V = {}
	for key, value in returns.items():
		V[key] = np.mean(value)
	return V

if __name__ == '__main__':
	env = gym.make('Blackjack-v0')
	# print(env.observation_space)  # (current_sum 0-31, deal's face up 1-10, usable_ace 0-1), 32*10*2
	# print(env.action_space)  # stick, hit
	V = on_policy_MC_prediction_every(env, 50000)
	plot_blackjack_values(V)