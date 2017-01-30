import copy
import logging
import struct
import torch

GAMMA = 0.99

# Weighting of entropy term in policy loss
BETA = 0.01

def start_learner(pipe, joint_network, gradient_queue, packet_fmt='<Bfffff'):
  logging.info('Started learner thread, waiting for data.')

  # Stores the history of state-action-reward tuples for the entire duration of
  # the episode.
  history = []

  # Make a local copy of the model
  joint_network_ = copy.deepcopy(joint_network)

  joint_network_.train()

  # Train loop
  while True:
    data = pipe.recv()

    if data == 'STOP':
      logging.info('Received STOP packet, will perform weight update and quit')

      R = 0
      for state, action, reward in reversed(history):
        R = reward + GAMMA*R
        out = joint_network_(state)
        value_out = out[-1]
        policy_out = out[action]

        entropy = out[:-1] * torch.log(out[:-1])
        policy_loss = torch.log(policy_out) * (R - value_out) + BETA * entropy
        value_loss = (R - value_out) ** 2
        
        policy_loss.backward(retain_variables=True)
        value_loss.backward(retain_variables=True)
      
      gradient_queue.put(joint_network_.parameters())
      return
    try:
      packet = struct.unpack(packet_fmt, data)
    except struct.error as err:
      logging.error('Bad packet, skipping: %s', str(err))
      continue

    # Append (state, action, reward) tuple to history (see server.py for packet
    # format).
    history.append((packet[2:], packet[0], packet[1]))

    logging.debug('Successfully unpacked packet: %s', str(packet))