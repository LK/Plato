import random
import torch

class ExperienceMemory(object):
  def __init__(self, capacity=20000):
    self.capacity = capacity
    self.memory = []
    self.pos = 0
    
  def record_transition(self, transition):
    if len(self.memory) < self.capacity:
      self.memory.append(transition.squeeze())
    else:
      self.memory[self.pos] = transition.squeeze()
      self.pos = (self.pos + 1) % self.capacity
  
  def get_batch(self, batch_size=32):
    return torch.stack(random.sample(self.memory, batch_size))
    # return torch.stack([self.memory[-1]])
  
  def __len__(self):
    return len(self.memory)