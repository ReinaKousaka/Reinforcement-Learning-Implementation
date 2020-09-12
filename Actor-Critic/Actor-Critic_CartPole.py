""" Actor-Critic on CartPole-v1 environment """

import gym
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, input_dims, num_actions, lr, hidden_dims=128):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_actions)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        return self.net(observation)

class Agent:
    def __init__(self, env, alpha_theta, alpha_w, gamma=0.99):
        """ alpha_theta: the step size of policy parameter
            alpha_w: the step size of state-value
        """
        self.env = env
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.log_probs = None  # The attribute that stores the most recent log(Prob of action selected), i.e. ln(pi(A|S, theta))

        self.actor = Net(input_dims=self.env.observation_space.shape[0], num_actions=self.env.action_space.n, lr=alpha_theta)
        self.critic = Net(input_dims=self.env.observation_space.shape[0], num_actions=1, lr=alpha_w)

    def choose_action(self, observation):
        # given the state, choose action under actor
        probs = torch.softmax(self.actor(torch.FloatTensor(observation)), dim=0)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        return action.item()
    
    def train(self, num_episodes=100):
        y = []
        for _ in tqdm(range(num_episodes)):
            state = self.env.reset()
            I = 1
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:  # calculate the TD error
                    delta = reward - self.critic(torch.FloatTensor(state))
                else:
                    delta = reward + self.gamma * self.critic(torch.FloatTensor(next_state)) - self.critic(torch.FloatTensor(state))
                loss_critic = delta ** 2
                loss_actor = -self.log_probs * delta * I
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                (loss_actor + loss_critic).backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                I *= self.gamma
                state = next_state
            y.append(total_reward)
        return y

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(env=env, alpha_theta=0.00001, alpha_w=0.0005)
    num_episodes=3000
    y = agent.train(num_episodes)

    # draw
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(num_episodes) + 1, y)
    plt.show()

