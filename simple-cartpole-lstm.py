import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Categorical
from torch.autograd import Variable


GAMMA = 0.99
HIDDEN_SIZE = 64
NUM_ACTIONS = 2

SavedAction = namedtuple('SaveAction', ['log_prob','value'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.lstm = nn.LSTMCell(4, HIDDEN_SIZE)
        self.action_head = nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)
        self.saved_actions = []
        self.rewards = []
        self.reset()

    def reset(self):
        self.hidden = Variable(torch.zeros(1, HIDDEN_SIZE)), Variable(torch.zeros(1, HIDDEN_SIZE))

    def detach_weights(self):
        self.hidden = self.hidden[0].detach(), self.hidden[1].detach()

    def forward(self, x):
        x = x.unsqueeze(0)
        self.hidden = self.lstm(x, self.hidden)
        x = self.hidden[0]
        x = x.squeeze(0)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


class DiscreteAgent():
    def __init__(self):
        self.model = Policy()
        self.optimizer = opt.Adam(self.model.parameters(), lr=2.5e-4)
        self.eps = np.finfo(np.float32).eps.item()

    def select_action(self, state):
        probs, state_value = self.model( torch.from_numpy(state).float() )
        m = Categorical(probs) # I don't know what this does
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def finish_episode(self):
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []
        for r in self.model.rewards[::-1]:
            R = r + GAMMA * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        del self.model.rewards[:] #WHY IS THIS HERE???
        del self.model.saved_actions[:]



def main(env, args):
    agent = DiscreteAgent()

    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            agent.model.rewards.append(reward)
            if done:
                agent.model.reset()
                break
            else:
                agent.model.detach_weights()

        running_reward = running_reward * GAMMA + t * 0.01 #why 0.01?
        agent.finish_episode()
        if i_episode % args.log_freq ==0:
            print('Episode {}\t Last length: {:5d}\t Average length: {:.2f}'.format(i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            agent.model.reset()
            print('Finished. Running reward: {}, last episode time steps: {}'.format(running_reward, t))
            for t in range(100000):
                action = agent.select_action(state)
                state, reward, done, _ = env.step(action)
                env.render()
                agent.model.rewards.append(reward)
            break



parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--seed', type=int, default=13, help='random seed for both torch and gym')

if __name__ == '__main__':
    args = parser.parse_args()
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    main(env, args)
    env.env.close()

