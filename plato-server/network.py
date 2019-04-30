import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class QNetwork(nn.Module):
  def __init__(self, state_dims=5, action_dims=6, hidden_dims=128):
    super(QNetwork, self).__init__()

    self.fc1 = nn.Linear(state_dims, hidden_dims)
    self.fc2 = nn.Linear(hidden_dims, hidden_dims)
    self.out = nn.Linear(hidden_dims, action_dims)

    # NOTE: If you need to `.share_memory()` the network, this field will not be
    # shared. Instead, make it a `torch.nn.Parameter` (and make sure to update
    # it with `self.updates[0] += 1`)
    self.updates = 0
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.out(x)