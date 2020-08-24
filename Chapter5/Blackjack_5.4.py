import numpy as np
from collections import defaultdict
import gym
from plot_utils import plot_policy, plot_blackjack_values


def generate_episode(env, policy):
    episode = []  # A list of tuples (state, action, reward)
    state = env.reset()
    while True:
        if state not in policy:
            policy[state] = (1 / env.action_space.n) * np.ones(env.action_space.n)  # initialize with same prob
        action = np.random.choice(np.arange(env.action_space.n), p=policy[state])  # under the given policy
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def on_policy_MC_control(env, num_episode, epsilon, gamma=1):
    policy = {}  # A map from state to numpy array of prob-index
    returns = defaultdict(list)
    Q = defaultdict(float)  # A map from (state, action) to its value function
    for _ in range(num_episode):
        episode = generate_episode(env, policy)
        G = 0
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            returns[(state, action)].append(G)
            Q[(state, action)] = np.mean(returns[(state, action)])

            lst, max_q = [], -10  # Get the greedy action
            for a in range(env.action_space.n):
                if Q[(state, a)] > max_q:
                    max_q = Q[(state, a)]
                    lst = [a]
                elif Q[(state, a)] == max_q:
                    lst.append(a)
            greedy_action = np.random.choice(lst)  # with ties broken arbitrarily
            for a in range(env.action_space.n):
                policy[state][a] = 1 - epsilon + epsilon / env.action_space.n if a == greedy_action \
                    else epsilon / env.action_space.n
    return policy, Q


def draw_policy(policy):
    Pi = {}
    for key, value in policy.items():
        max_prob, greedy_action = value[0], 0
        for a in range(1, len(value)):
            if value[a] > max_prob:
                max_prob = value[a]
                greedy_action = a
        Pi[key] = greedy_action
    plot_policy(Pi)


def draw_value(Q):
    V = {}
    for key, value in Q.items():
        state, action = key
        if state not in V or value > V[state]:
            V[state] = value
    plot_blackjack_values(V)


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    policy, Q = on_policy_MC_control(env, 50000, 0.1)
    draw_policy(policy)
    draw_value(Q)