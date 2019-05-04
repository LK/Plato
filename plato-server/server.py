import h5py
from http.server import BaseHTTPRequestHandler, HTTPServer
import learner
import logging
from metrics_writer import MetricsWriter
from network import *
import socket
import struct
import threading
import time
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
from experience_memory import ExperienceMemory
import math

'''
The EnvironmentServer is responsible for fielding connections from agents in the
environment, spinning up a process for each and recording transitions they send
to the ExperienceMemory.

Packet format:
 - Client ID [int32]:           a random unique identifier to identify an
                                individual client.
 - Start State [below]:         a description of the start state for this 
                                transition.
 - Action [byte]:               the action that was selected after the state in
                                this packet.
 - Reward [float32]:            the reward received after the above action.
 - End State [below]:           a description of the end state for this
                                transition.
 - Terminal [byte]:             1 if this transition ends the episode, 0
                                otherwise.

 ------------------------------------------------------------------------------
  Below is the default configuration of state variables. It is possible to use
  a different configuration, as long as the robot is updated accordingly and
  the --action command-line argument is passed. Each state variables must be
  passed as a float32.
 ------------------------------------------------------------------------------

 - Agent heading [float32]:     the heading of the agent before the action.
 - Agent energy [float32]:      the energy of the agent before the action.
 - Agent gun heat [float32]:    the agent's gun heat
 - Agent X position [float32]:  the agent's X position
 - Agent Y position [float32]:  the agent's Y position
 - Opponent bearing [float32]:  the opponent's bearing before the action.
 - Opponent energy [float32]:   the opponent's energy before the action.
 - Distance [float32]:          the distance to the other robot.
'''

GAMMA = 0.99

class EnvironmentServer(object):
  def __init__(self, state_dims, action_dims, ip, port, filename, lock):
    self.state_dims = state_dims
    self.ip = ip
    self.port = port
    self.lock = lock

    self.episodes = dict()
    self.writer = None

    # Build network
    self.network = QNetwork(state_dims, action_dims, 32)
    self.optimizer = torch.optim.Adam(self.network.parameters())

    # Create experience replay
    self.memory = ExperienceMemory()

    # Read or create HDF5 file
    self.lock.acquire()
    self.file = h5py.File(filename, driver='sec2')
    if len(self.file.keys()) > 0:
      logging.info('Restoring weights from %s...', filename)
      self.network.fc1.weight = torch.nn.Parameter(torch.from_numpy(self.file['fc1']['w'][:])).type(torch.float)
      self.network.fc1.bias = torch.nn.Parameter(torch.from_numpy(self.file['fc1']['b'][:])).type(torch.float)
      self.network.fc2.weight = torch.nn.Parameter(torch.from_numpy(self.file['fc2']['w'][:])).type(torch.float)
      self.network.fc2.bias = torch.nn.Parameter(torch.from_numpy(self.file['fc2']['b'][:])).type(torch.float)
      self.network.out.weight = torch.nn.Parameter(torch.from_numpy(self.file['out']['w'][:])).type(torch.float)
      self.network.out.bias = torch.nn.Parameter(torch.from_numpy(self.file['out']['b'][:])).type(torch.float)
      self.network.updates = self.file.attrs['updates']
      logging.info('Restored network with %d updates', self.network.updates)
    else:
      logging.debug('Saving initial weights to %s', filename)
      fc1 = self.file.create_group('fc1')
      fc2 = self.file.create_group('fc2')
      out = self.file.create_group('out')

      fc1.create_dataset('w', data=self.network.fc1.weight.data.numpy())
      fc1.create_dataset('b', data=self.network.fc1.bias.data.numpy())
      fc2.create_dataset('w', data=self.network.fc2.weight.data.numpy())
      fc2.create_dataset('b', data=self.network.fc2.bias.data.numpy())
      out.create_dataset('w', data=self.network.out.weight.data.numpy())
      out.create_dataset('b', data=self.network.out.bias.data.numpy())
      
      self.file.attrs['updates'] = 0

      self.file.flush()
    self.lock.release()

  def start(self):
    """ Start the server asynchronously. """
    # t = threading.Thread(target=self._run)
    # t.daemon = True
    # t.start()
    self._run()

  def _run(self):
    logging.debug('Starting learning server.')

    self.writer = MetricsWriter('/tmp/plato')
    self.writer.start_listening()

    # Packet format (omitting the client ID, which is stripped)
    state_struct = ('f' * self.state_dims) 
    packet_fmt = '>' + state_struct + 'Bf' + state_struct + '?'

    # Start listening for packets
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((self.ip, self.port))

    logging.info('Listening for client packets on %s:%d', self.ip, self.port)

    while True:
      buf = sock.recv(65535)

      # Extract and strip the client ID from the received packet
      client_id = struct.unpack('>i', buf[:4])[0]
      buf = buf[4:]

      if client_id not in self.episodes:
        self.episodes[client_id] = { 'reward': 0, 'length': 0, 'q_forward': [], 'q_backward': [], 'q_left': [], 'q_right': [], 'q_fire': [], 'q_nothing': [] }

      logging.debug('Received client packet from %d' % client_id)

      # Add experience to memory
      packet = struct.unpack(packet_fmt, buf)
      transition = torch.Tensor([packet])
      self.memory.record_transition(transition)

      est_q = self.network(transition[0, self.state_dims + 2:self.state_dims + 2 + self.state_dims].squeeze()).detach()

      self.episodes[client_id]['reward'] += transition[0, self.state_dims + 1]
      self.episodes[client_id]['length'] += 10
      self.episodes[client_id]['q_forward'].append(est_q[0])
      self.episodes[client_id]['q_backward'].append(est_q[1])
      self.episodes[client_id]['q_left'].append(est_q[2])
      self.episodes[client_id]['q_right'].append(est_q[3])
      self.episodes[client_id]['q_fire'].append(est_q[4])
      self.episodes[client_id]['q_nothing'].append(est_q[5])

      if transition[0, -1] == 1:
        self.writer.log_episode(
          self.episodes[client_id]['length'],
          self.episodes[client_id]['reward'],
          self.episodes[client_id]['q_forward'],
          self.episodes[client_id]['q_backward'],
          self.episodes[client_id]['q_left'],
          self.episodes[client_id]['q_right'],
          self.episodes[client_id]['q_fire'],
          self.episodes[client_id]['q_nothing']
        )
        del self.episodes[client_id]

      if len(self.memory) >= 32:
        self.perform_update()
      
  def perform_update(self):
    sample = self.memory.get_batch()
    sample_start_state = sample[:, :self.state_dims]
    sample_action = sample[:, self.state_dims].squeeze()
    sample_reward = sample[:, self.state_dims + 1].squeeze()
    sample_end_state = sample[:, self.state_dims + 2:self.state_dims + 2 + self.state_dims]
    sample_terminal = sample[:, -1].squeeze()

    assert(self.state_dims + 2 + self.state_dims == sample.shape[1] - 1)

    # print("sample_start_state", sample_start_state)
    # print("sample_action", sample_action)
    # print("sample_reward", sample_reward)
    # print("sample_end_state", sample_end_state)
    # print("sample_terminal", sample_terminal)

    y = torch.zeros((sample.shape[0],))
    for i in range(sample.shape[0]):
      if sample_terminal[i] == 1:
        y[i] = sample_reward[i]
      else:
        m = torch.max(self.network(sample_end_state[i]))
        y[i] = sample_reward[i] + GAMMA * m

    # print("y", y)

    # print("self.network(sample_start_state)", self.network(sample_start_state))
    # print("self.network(sample_start_state).gather(1, sample_action.type(torch.LongTensor).unsqueeze(1)).squeeze()", self.network(sample_start_state).gather(1, sample_action.type(torch.LongTensor).unsqueeze(1)).squeeze())

    est = self.network(sample_start_state).gather(1, sample_action.type(torch.LongTensor).unsqueeze(1)).squeeze()
    loss = torch.sum((y - est)**2).squeeze()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.network.updates += 1

    self.lock.acquire()
    self.file['fc1']['w'][...] = self.network.fc1.weight.data.numpy()
    self.file['fc1']['b'][...] = self.network.fc1.bias.data.numpy()
    self.file['fc2']['w'][...] = self.network.fc2.weight.data.numpy()
    self.file['fc2']['b'][...] = self.network.fc2.bias.data.numpy()
    self.file['out']['w'][...] = self.network.out.weight.data.numpy()
    self.file['out']['b'][...] = self.network.out.bias.data.numpy()
    self.file.attrs['updates'] = self.network.updates
    self.file.flush()
    self.lock.release()

    totalnorm = 0
    for p in self.network.parameters():
      modulenorm = p.grad.data.norm()
      totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)

    self.writer.log_update(loss.item(), totalnorm, np.average(self.network(sample_start_state).detach().numpy(), axis=0))

    # print("wrote network with", self.network.updates, "updates", "avg reward:", np.average(sample_reward), "avg terminal:", np.average(sample_terminal))
    

  # def weight_serializer(self):
  #   while True:
  #     time.sleep(30)
  #     logging.info('Serializing weights...')
  #     self.lock.acquire()
  #     self.file['fc1']['w'][...] = self.base_network.fc1.weight.data.numpy()
  #     self.file['fc1']['b'][...] = self.base_network.fc1.bias.data.numpy()
  #     self.file['fc2']['w'][...] = self.base_network.fc2.weight.data.numpy()
  #     self.file['fc2']['b'][...] = self.base_network.fc2.bias.data.numpy()
  #     self.file['v']['w'][...] = self.value_network.value.weight.data.numpy()
  #     self.file['v']['b'][...] = self.value_network.value.bias.data.numpy()
  #     self.file['p']['w'][...] = self.policy_network.policy.weight.data.numpy()
  #     self.file['p']['b'][...] = self.policy_network.policy.bias.data.numpy()
  #     self.file.attrs['updates'] = self.joint_network.updates
  #     self.file.flush()
  #     self.lock.release()
  #     logging.info('Saved network with %d updates', self.joint_network.updates)


  # def gradient_applier(self, global_model, gradient_queue, optimizer):
  #   logging.debug('Starting gradient applier process')
  #   optimizer.zero_grad()
  #   while True:
  #     logging.info('Waiting for gradient...')
  #     local_params = gradient_queue.get()
  #     print(len(local_params))
  #     print(len(list(global_model.parameters())))
  #     logging.info('Applying gradients')
  #     for (local_param, global_param) in zip(local_params, 
  #                                            global_model.parameters()):
  #       global_param.grad.data = local_param.grad.data.clamp(-100, 100)

  #     optimizer.step()
  #     global_model.updates += 1

  #     self.lock.acquire()
  #     self.file['fc1']['w'][...] = self.base_network.fc1.weight.data.numpy()
  #     self.file['fc1']['b'][...] = self.base_network.fc1.bias.data.numpy()
  #     self.file['fc2']['w'][...] = self.base_network.fc2.weight.data.numpy()
  #     self.file['fc2']['b'][...] = self.base_network.fc2.bias.data.numpy()
  #     self.file['v']['w'][...] = self.value_network.value.weight.data.numpy()
  #     self.file['v']['b'][...] = self.value_network.value.bias.data.numpy()
  #     self.file['p']['w'][...] = self.policy_network.policy.weight.data.numpy()
  #     self.file['p']['b'][...] = self.policy_network.policy.bias.data.numpy()
  #     self.file.attrs['updates'] = global_model.updates

  #     self.file.flush()
  #     self.lock.release()

  #     logging.debug('Updated saved weights')

  #     optimizer.zero_grad()
  
class WeightServer(object):
  class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      self.send_response(200)
      self.send_header('Content-Type', 'binary/octet-stream')
      self.end_headers()
      self.send_weights()

    def log_request(self, code='-', size='-'):
      logging.debug('Weight server sent %s response', code)

  def __init__(self, ip, port, filename, lock):
    self.ip = ip
    self.port = port
    self.filename = filename
    self.lock = lock

  def start(self):
    """ Start the server asynchronously. """
    t = threading.Thread(target=self._run)
    t.daemon = True
    t.start()

  def _run(self):
    logging.debug('Starting weight server.')

    def send_weights(handler):
      self.lock.acquire()
      f = open(self.filename, 'rb')
      handler.wfile.write(f.read())
      f.close()
      self.lock.release()

    self.Handler.send_weights = send_weights
    httpd = HTTPServer((self.ip, self.port), self.Handler)

    logging.info('Listening for weight requests on %s:%d', self.ip, self.port)
    
    httpd.serve_forever()