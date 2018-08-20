import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.5             # randomness in priority when using prioritized experience
BETA_START = 0.3        # starting point for importance sampling tuning
BETA_INCREASE = 1e-4    # increase of beta for importance sampling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, DDQN=False, PRB=False,
        Dueling=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            DDQN (bool): apply Double DDQN algorithm
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.DDQN = DDQN
        self.PRB = PRB

        # Q-Network
        if Dueling:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if self.PRB:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE,
                BATCH_SIZE, seed, ALPHA, BETA_START, BETA_INCREASE)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Put network in eval mode to evaluate states
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # Put network back into training mode for further training
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int32)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, idx, probs) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, sample_idx, sample_weights = experiences

        # Get max predicted Q values (for next states) for target model
        if self.DDQN:
            Q_actions = self.qnetwork_local(next_states).detach(
            ).argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach(
            ).gather(1, Q_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach(
                ).max(1)[0].unsqueeze(1)
        # Compute Q-targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        # Get expected Q-values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if self.PRB:
            errors = (Q_targets - Q_expected).detach().squeeze().data.cpu().numpy()
            self.memory.update_priorities(sample_idx, errors)
            loss = torch.sum(sample_weights * (Q_expected - Q_targets)**2)
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ----------------------- update target network -------------------#
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # Return sample with two additional None's at end. There are for prioritized replay
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
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
