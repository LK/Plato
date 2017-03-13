import copy
import logging
import math
import struct
import torch

GAMMA = 0.99

# Weighting of entropy term in policy loss
BETA = 0.01

def start_learner(pipe, joint_network, gradient_queue, writer, packet_fmt='<Bfffff'):
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
    if data == b'STOP':
      logging.info('Received STOP packet, will perform weight update and quit')
      logging.debug(len(history))
      logging.debug(history[-1][2])
      writer.log_episode(len(history), history[-1][2])

      R = 0
      for state, action, reward in reversed(history):
        R = reward + GAMMA*R
        state_var = torch.autograd.Variable(torch.Tensor(state)).view(1, 4)
        out = joint_network_(state_var)
        value_out = out[0][-1]
        policy_out = out[0][action]

        entropy = - torch.sum(out[0][:-1] * torch.log(out[0][:-1]))
        policy_loss = torch.log(policy_out) * (R - value_out) + BETA * entropy
        value_loss = (R - value_out) ** 2
        
        policy_loss.backward(retain_variables=True)
        value_loss.backward(retain_variables=True)
        totalnorm = 0
        for p in joint_network_.parameters():
          modulenorm = p.grad.data.norm()
          totalnorm += modulenorm ** 2
        totalnorm = math.sqrt(totalnorm)
        writer.log_update(value_loss.data.numpy()[0], policy_loss.data.numpy()[0], totalnorm, out[0][:-1].data.numpy())
        # print(out[0][:-1])
      gradient_queue.put(list(joint_network_.parameters()))
      return
    try:
      packet = struct.unpack(packet_fmt, data)
    except struct.error as err:
      logging.error('Bad packet, skipping: %s', str(err))
      continue

    # Append (state, action, reward) tuple to history (see server.py for packet
    # format).
    history.append((packet[2:], packet[0], packet[1]))
    logging.debug('Successfully unpacked packet with reward %f', packet[1])