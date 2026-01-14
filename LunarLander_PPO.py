import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.affine = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )
        self.action_layer = nn.Linear(128,action_dim)
        self.value_layer = nn.Linear(128,1)

    def forward(self,state):
        state = self.affine(state)

        action_probs = F.softmax(self.action_layer(state), dim=1)

        state_values = self.value_layer(state)

        return action_probs, state_values

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.lr = 0.0003

        self.police = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.police.parameters(), lr= self.lr)
        self.police_old = ActorCritic(state_dim, action_dim)
        self.police_old.load_state_dict(self.police.state_dict())

        self.MseLos = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.police_old(state)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action).item()
    
    def update(self, memory):
        old_states = torch.FloatTensor(np.array(memory.states))
        old_actions = torch.LongTensor(np.array(memory.actions))
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs))

        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean())/ (rewards.std() + 1e-7)

        for _ in range(self.K_epochs):
            probs, state_values = self.police(old_states)
            state_values = torch.squeeze(state_values)

            dist = Categorical(probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 *self.MseLos(state_values, rewards)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.police_old.load_state_dict(self.police.state_dict())

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:], self.states[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

def train():
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    memory = Memory()

    max_episodes = 1000
    update_timestep = 2000
    time_step = 0

    for episode in range(1, max_episodes+1):
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1,1000):
            time_step+=1
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done or truncated)

            state = next_state
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update(memory)
                memory.clear()
                time_step = 0
            if done or truncated:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode} \t Reward: {current_ep_reward:.2f}")
    return agent

train()