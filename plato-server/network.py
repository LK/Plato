import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class BaseNetwork(nn.Module):
  def __init__(self, state_dims=4):
    super(BaseNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dims, 128)
    self.fc2 = nn.Linear(128, 128)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x

class ValueNetwork(nn.Module):
  def __init__(self, base):
    super(ValueNetwork, self).__init__()
    self.base = base
    self.value = nn.Linear(128, 1)

  def forward(self, x):
    x = self.base(x)
    x = self.value(x)
    return x

class PolicyNetwork(nn.Module):
  def __init__(self, base, actions=6, hidden_size=128):
    super(PolicyNetwork, self).__init__()
    self.base = base
    self.policy = nn.Linear(hidden_size, actions)

  def forward(self, x):
    x = self.base(x)
    x = self.policy(x)
    return F.softmax(x)

class JointNetwork(nn.Module):
  def __init__(self, value, policy):
    super(JointNetwork, self).__init__()
    self.value = value
    self.policy = policy
    self.updates = torch.nn.Parameter(torch.Tensor([0]))

  def forward(self, x):
    value = self.value(x)
    policy = self.policy(x)
    return torch.cat((policy, value), 1)