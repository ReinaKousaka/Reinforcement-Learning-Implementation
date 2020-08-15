import numpy as np
import matplotlib
import matplotlib.pyplot as plt
NUM_STATES = 101
class Env(object):
    """the environment for Gambler's Problem"""
    def __init__(self, ph):
        """
        ph: the probabilty of the coin coming up heads
        S: the states
        P: the transitions 
        """
        self.ph = ph
        self.S = np.arange(NUM_STATES)
        self.P = {}
        for s in range(NUM_STATES):
            dic = {}
            for a in range(self.get_num_actions(s)):
                reward = 1 if s + a == 100 else 0
                win = True if s + a == 100 else False
                lose = True if s - a == 0 else False
                dic[a] = [(ph, s + a, reward, win), (1 - ph, s - a, 0, lose)]
            self.P[s] = dic

    def get_num_actions(self, s):
        """return number of actions in state s"""
        return 1 + min(s, 100 - s)

    def A(self, s):
        """return the list of actions in state s"""
        return np.arange(self.get_num_actions(s), dtype=int)

def value_iteration(env, gamma, theta):
    """ 
    The Value Iteration
    theta: a small threshold determining accuary of estimation
    return value: The value function and An optimal deterministic policy
    """
    v = np.zeros(NUM_STATES)
    delta = theta
    sweeps_history = []     # For the figure in the question, we store the sweep
    while delta >= theta:
        delta = 0
        temp = v.copy()
        v.fill(0)
        for s in env.S:     # iterate over all states
            if s == 0 or s == 100:  # the dummy states
                continue
            for a in env.A(s):
                accumulator = 0
                for prob, next_s, reward, done in env.P[s][a]:
                    accumulator += prob * (reward + gamma * temp[next_s])
                v[s] = max(accumulator, v[s])
            delta = max(delta, np.abs(v[s] - temp[s]))
        sweeps_history.append(v.copy())

    pi = np.zeros(NUM_STATES, dtype=int)
    for s in range(1, NUM_STATES - 1):
        q_values = []
        for a in env.A(s):
            accumulator = 0
            for prob, next_s, reward, done in env.P[s][a]:
                accumulator+= prob * (reward + gamma * v[next_s])
            q_values.append(accumulator)
        pi[s] = np.argmax(np.round(q_values[1:], 5)) + 1    # The np.round is necessary here!!!
   
    draw(sweeps_history, pi)    
    return v, pi

def draw(sweeps_history, pi):
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.scatter(np.arange(NUM_STATES, dtype=int), pi)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.savefig('./figure_1.png')   # save the figure
    plt.close()

if __name__ == '__main__':
    env1 = Env(0.4)     # the example from the book
    value_iteration(env1, 1, 1e-6)
    #env2 = Env(0.25)   # Exercise 4.9 from the book
    #value_iteration(env2, 1, 1e-6)
    #env3 = Env(0.55)
    #value_iteration(env3, 1, 1e-6)    