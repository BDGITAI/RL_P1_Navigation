import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
		# call to mother class nn.Module constructor
        super(QNetwork, self).__init__()
		# set a seed to allow reproducing results
        self.seed = torch.manual_seed(seed)
		# define first neural network layer. Input size shall accept environment observations
		# hence first parameters is state_size. Use linear layer
        self.fc1 = nn.Linear(state_size, fc1_units)
		# second layer for neural net. Input size shall be equal to previous layer output
        self.fc2 = nn.Linear(fc1_units, fc2_units)
		# third and last layer . Output size shall be sized according to action_size as the 
		# model provide the value for each action based on the observed state
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
		# feed forward the NN with RELU activation function
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
