import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import os
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.affine = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.action_layer = nn.Linear(256,action_dim)
        self.value_layer = nn.Linear(256,1)

    def forward(self,state):
        state = self.affine(state)

        action_probs = F.softmax(self.action_layer(state), dim=1)

        state_values = self.value_layer(state)

        return action_probs, state_values

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.975
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.lr = 0.0003

        self.police = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.police.parameters(), lr= self.lr)
        self.police_old = ActorCritic(state_dim, action_dim)
        self.police_old.load_state_dict(self.police.state_dict())

        self.MseLos = nn.MSELoss()

    def select_action(self, state, train = True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.police_old(state)
        dist = Categorical(probs)
        if train:
            action = dist.sample()
        else:
            return torch.argmax(probs).item()
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
        rewards = (rewards - rewards.mean())/ (rewards.std() + 1e-8)

        for _ in range(self.K_epochs):
            probs, state_values = self.police(old_states)
            state_values = torch.squeeze(state_values)

            dist = Categorical(probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 *self.MseLos(state_values, rewards) - 0.01 * dist_entropy
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

def train(start_episode, episodes, continuation = False):
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    reward_history = []
    if continuation:
        reward_history = load_checkpoint(file_name = f"save_ep_{start_episode}.pth")

        
    memory = Memory()
    update_timestep = 2000
    time_step = 0

    for episode in range(start_episode, episodes+1):
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
            if (done or truncated) or ((t == 200) and (episode >3000)):
                break
        reward_history.append(current_ep_reward)
        if episode % 100 == 0:
            print(f"Episode {episode} \t Reward: {current_ep_reward:.2f}")
        if episode % 1000 == 0:
            save_checkpoint(agent, episode, reward_history)
    return agent, reward_history

def test_PPO(agent):
    test_episodes = 10
    env = gym.make("LunarLander-v3", render_mode = "human")
    reward_history = []
    for episode in range(1,test_episodes+1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state, train = False)
            next_state, reward, trum, tern, _ = env.step(action)
            total_reward += reward
            state = next_state
            done = trum or tern
        reward_history.append(total_reward)
        print(f"Test episode:{episode} \t reward:{total_reward:.2f}")
    env.close()

def save_checkpoint(agent, episodes, reward_history):
    name_folder = "LunarLander_PPO"
    name_file = f"save_ep_{episodes}.pth"
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)
    path = os.path.join(name_folder, name_file)
    checkpoint = {'police': agent.police.state_dict(),
                  'reward_history' : reward_history}
    torch.save(checkpoint, path)
    print("Save is done")
def load_checkpoint(file_name):
    name_folder = "LunarLander_PPO"
    path = os.path.join(name_folder, file_name)
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        agent.police.load_state_dict(checkpoint['police'])
        reward_history = checkpoint[checkpoint['reward_history']]
    return reward_history
def matplot(reward_hitory):
    a = [i for i in range(len(reward_hitory))]
    plt.plot(a, reward_history)
    plt.show() 


agent, reward_history = train(start_episode=1, episodes=4000)
matplot(reward_history)
test_PPO(agent)