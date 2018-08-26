"""
DQN agent
Code implemented from starter code provided by UDACITY RL NANODEGREE Program
URL for original code :

About DQN agent :

	File contains the Agent and ReplayBuffer class
	The Agent is the actor taking action and learning from experiences.
	The Agent possesses two types of memory : long term and short term.
	The long term memory is represented by a QNetwork mapping states to actions values. 
	By learning from the experiences  the QNetwork is updated and the network weights can be saved
	to represent the cumulated knwoledge (decide best action for given state)
	The short term memory is represented by a buffer that stores all results for exploration.
	When the buffer is full older memories are over written (hence short term)
	At some point the agent pause its exploration and transform the experiences gathered in the short term
	memory as long term memory.

	This mechanism is similar to the human brain activity. Short term memories are stored in Hippocampus 
        equivalent to Hippocampus where model is equivalent to neocortex.
	It is believed that some memories are reinforced and transferred in long term memory when 
	human beings go to sleep (pause from their exploration)
	See  https://qbi.uq.edu.au/brain-basics/memory/where-are-memories-stored
	
Modifications:
- Comments

"""

import numpy as np
import random
from collections import namedtuple, deque

from dqn_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

#If cuda is installed then use GPU else use CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment.
        Agent is made of several functions:
            init : create an agent
			step : save experience in memory and decide if it is time to learn
            act : decide the most relevant action in function of the environment state and current knowledge
            learn : learn for past experiences
					Use the soft_update function to update the target Qnetwork
    
    """
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Function triggering the learning phase if enough exploration has been done
		At each step, record the experience in the memory
		After collecting experiences, learn from them
        Params
        ======
            experience = state, action, reward, next_state, done
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
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
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
		# switch network into evaluation mode when not training
        self.qnetwork_local.eval()
		# deactivate gradient tracking as we do not to back propagate 
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
		# switch network into training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss using mean square error
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
		# update network weights
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
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
    """Fixed-size buffer to store experience tuples.
        Agent is made of several functions:
            init : create memory
			add : save experience in memory
            sample : create mini batch
            len : return size of memory buffer    
    """
	
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
		# get experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)

		# build mini batch
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
