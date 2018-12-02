import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # how often to update the network
TIMES_UPDATE = 1        # how many times to train each agent
EPSILON = 1             # starting point for noise decline
EPSILON_DECAY = 1e-4    # noise decay for each episode
ALPHA = 0.5             # randomness in priority when using prioritized experience
BETA_START = 0.3        # starting point for importance sampling tuning
BETA_INCREASE = 1e-4    # increase of beta for importance sampling


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#--------------------------DISCLAIMER---------------------------------------
# This script is based upon the ddpg_agent.py from the repo
# https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum

class MultiAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed, PRB=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            ReplayBuffer (obj): ReplayBuffer object for the Agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.PRB = PRB

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process (individual noise process for each agent)
        self.noise = [OUNoise(action_size, random_seed+i) for i in range(self.num_agents)]

        # Replay memory
        if self.PRB:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE,
                BATCH_SIZE, random_seed, ALPHA, BETA_START, BETA_INCREASE)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Decay of the noise
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Check whether it is time to learn
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for _ in range(TIMES_UPDATE):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            for i in range(self.num_agents):
                actions[i] += self.epsilon * self.noise[i].sample()
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def reset(self):
        """Reset the noise process and decline the noise impact on next iteration."""
        for n in self.noise:
            n.reset()
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, sample_idx, sample_weights = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        # Compute critic loss based on Prioritized Replay Buffer or not
        if self.PRB:
            errors = (Q_targets - Q_expected).detach().squeeze().data.cpu().numpy()
            self.memory.update_priorities(sample_idx, errors)
            critic_loss = torch.sum(sample_weights * (Q_expected - Q_targets)**2)
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        if self.PRB:
            errors = (Q_targets - Q_expected).detach().squeeze().data.cpu().numpy()
            self.memory.update_priorities(sample_idx, errors)
            loss = torch.sum(sample_weights * (Q_expected - Q_targets)**2)
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, None, None)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Fixed-side buffer to store prioritized experience tuples. This is
    a modified version of PrioritizedReplayMemory from
    https://github.com/qfettes/DeepRL-Tutorials."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha,
    beta_start=0.4, beta_increase=1e-4):
        """Initialize a Prioritized ReplayBuffer.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): control randomness in experience selection
            beta_start (float): control importance sampling (starting point)
            beta_increase (float): how much to increase beta for each sampling
        """

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = []
        self.priorities = np.zeros(self.buffer_size,)
        self.pos = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increase = beta_increase

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_prio = self.priorities.max() if self.memory else 1.0

        e = self.experience(state, action, reward, next_state, done)

        # Check if there is still room in the replay buffer, and append
        # the experience if there is
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)

        # If not, add the experience at its proper slot, pushing out the
        # oldest experience
        else:
            self.memory[self.pos] = e

        # Add the new experience with maximum priority to make sure it is
        # learned from at least once
        self.priorities[self.pos] = max_prio

        # Increment pos with one, starting over when reaching buffer_size
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self):
        """Sample a batch of experiences from memory with probabilities
        given according to https://arxiv.org/pdf/1511.05952.pdf"""

        N = len(self.memory)
        prios = self.priorities[:N]

        # Normalize probabilities
        probs = prios / prios.sum()

        # Draw a sample according to the probabilities and compute the
        # sample weights
        idx = np.random.choice(N, self.batch_size, p=probs)
        experiences = [self.memory[i] for i in idx]
        sample_probs = probs[idx]

        # To find maximum weight, we must compute the error with smallest prob
        min_prob = probs.min()
        max_weight = (N * min_prob)**(-self.beta)
        weights = (N * sample_probs) ** (-self.beta) / max_weight

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(weights).float().to(device)

        self.beta = min(self.beta+self.beta_increase, 1)

        return (states, actions, rewards, next_states, dones, idx, weights)

    def update_priorities(self, idx, errors):
        """Update priorities according to their TD-errors."""

        for i, error in zip(idx, errors):
            prio = np.abs(error) + 1e-4
            self.priorities[i] = prio**self.alpha

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
