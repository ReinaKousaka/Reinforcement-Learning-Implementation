import numpy as np
import random
import matplotlib.pyplot as plt
import sys

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


def Batch_MC_prediction(env, alpha=0.05, num_episode=100):
    v = INITIAL_VALUE.copy()
    error, total_error = [], 0
    episodes = []
    updates = np.zeros(7)
    for _ in range(num_episode):
        print('\rEpisode {}/{}.'.format(_ + 1, num_episode), end='')
        sys.stdout.flush()
        episode = generate_episode(env)
        episodes.append(episode)
        time = 0
        while True:  # iterate until the value converges
            time += 1
            print('time={}'.format(time))
            updates.fill(0)
            for episode in episodes:
                G = episode[-1][1]
                # for state, reward in episode:
                #     updates[state] += alpha * (G - v[state])
                for i in range(len(episode) - 1):
                    updates[(episode[i][0])] += G - v[(episode[i][0])]
            #print(updates)

            if np.sum(np.abs(updates)) < 1e-3:
                break
            v += updates
            #print(v)
            #exit(0)
        total_error += compute_RMS(v)
        error.append(total_error / (_ + 1))
    return v, error


def Batch_TD_0_prediction(env, alpha=0.05, num_episode=100):
    v = INITIAL_VALUE.copy()
    error, total_error = [], 0
    episodes = []
    updates = np.zeros(7)
    for _ in range(num_episode):
        print('\rEpisode {}/{}.'.format(_ + 1, num_episode), end='')
        sys.stdout.flush()
        episode = generate_episode(env)
        episodes.append(episode)
        while True:  # iterate until the value converges
            updates.fill(0)
            for episode in episodes:
                G = episode[-1][1]
                for i, (state, reward) in enumerate(episode):
                    updates[state] += alpha * (G + v[episode[min(i + 1, len(episode) - 1)][0]] - v[state])
            if np.sum(np.abs(updates)) < 1e-3:
                break
            print(v)
            v += updates
        total_error += compute_RMS(v)
        error.append(total_error / (_ + 1))
    return v, error


def draw(env, num_episode=100):
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    #v, error = Batch_TD_0_prediction(env)
    #ax1.plot(np.arange(num_episode), v[1: 6], label='TD', color='blue')
    v, error = Batch_MC_prediction(env)
    ax1.plot(np.arange(num_episode), v[1: 6], label='MC', color='red')
    ax1.xlabel('Episodes')
    ax1.ylabel('Averaged Empirical RMS Error')
    ax1.set(title='Batch Training')
    ax1.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    env = Env()
    draw(env)





