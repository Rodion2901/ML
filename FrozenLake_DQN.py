import gymnasium as gym
import numpy as np
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import deque
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self,state_size, action_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state, action):
        self.state_size = state
        self.action_size = action

        self.q_network = DQN(state, action)
        self.target_network = DQN(state, action)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.memory = ReplayBuffer(10000)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.993
        self.update_count = 0
        self.taget_update = 100

        self.gamma = 0.99
        self.batch_size = 32


    def act(self, state, train=True):
        if train and (np.random.rand() < self.epsilon):
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_network(next_states)
            target = next_q.max(1)[0]
            target_q = rewards + (self.gamma * target * (1 - dones))

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_count += 1
        
        if self.update_count % self.taget_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def encode_one_hot(state, state_size):
    one_hot = np.zeros(state_size)
    one_hot[state] = 1
    return one_hot

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states),
                np.array(actions, dtype=int),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)

def train_dqn(name="FrozenLake-v1", episodes=1000):
    env = gym.make(name, map_name="4x4", is_slippery=False)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    reward_history = []

    for episode in range(episodes):
        done = False
        total_reward = 0
        state, _ = env.reset()
        state = encode_one_hot(state, state_size)
        
        while not done:
            action = agent.act(state, train=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = encode_one_hot(next_state, state_size)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        
        if (episode+1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            success_rate = np.mean([1 if r > 0 else 0 for r in reward_history[-100:]])

            print(
                f"Episode: {episode+1}, "
                f"Avg Reward: {avg_reward:.3f}, "
                f"Success Rate: {success_rate:.2%}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )
    
    return agent

def test_dqn(agent, test_episodes=10):
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    state_size = env.observation_space.n
    reward_history = []
    
    for episode in range(test_episodes):
        done = False
        total_reward = 0
        state, _ = env.reset()
        state = encode_one_hot(state, state_size)
        
        while not done:
            action = agent.act(state, train=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = encode_one_hot(next_state, state_size)
            state = next_state
            total_reward += reward
        
        reward_history.append(total_reward)
        print(f"Test Episode {episode+1}: Reward = {total_reward}")
    
    success_rate = np.mean([1 if r > 0 else 0 for r in reward_history])
    print(f"\nTest Results:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward: {np.mean(reward_history):.3f}")
    
    env.close()

agent = train_dqn(episodes=1000)
test_dqn(agent, test_episodes=10)