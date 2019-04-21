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

'''
Packet format:
 - Client ID [int32]:           a random unique identifier to identify an
                                individual client.
 - Action [byte]:               the action that was selected after the state in
                                this packet.
 - Reward [float32]:            the reward received after the above action.

 ------------------------------------------------------------------------------
  Below is the default configuration of state variables. It is possible to use
  a different configuration, as long as the robot is updated accordingly and
  the --action command-line argument is passed. Each state variables must be
  passed as a float32.
 ------------------------------------------------------------------------------

 - Agent heading [float32]:     the heading of the agent before the action.
 - Agent energy [float32]:      the energy of the agent before the action.
 - Opponent bearing [float32]:  the opponent's bearing before the action.
 - Opponent energy [float32]:   the opponent's energy before the action.
'''

class LearningServer(object):
  def __init__(self, state_dims, ip, port, filename, lock):
    self.state_dims = state_dims
    self.ip = ip
    self.port = port
    self.lock = lock

    # Build networks
    self.base_network = BaseNetwork(state_dims)
    self.value_network = ValueNetwork(self.base_network)
    self.policy_network = PolicyNetwork(self.base_network)
    self.joint_network = JointNetwork(self.value_network, self.policy_network)
    self.joint_network.share_memory()

    # Read or create HDF5 file
    self.lock.acquire()
    self.file = h5py.File(filename, driver='sec2')
    if len(self.file.keys()) > 0:
      logging.info('Restoring weights from %s...', filename)
      self.base_network.fc1.weight = torch.nn.Parameter(torch.from_numpy(self.file['fc1']['w'][:])).type(torch.float)
      self.base_network.fc1.bias = torch.nn.Parameter(torch.from_numpy(self.file['fc1']['b'][:])).type(torch.float)
      self.base_network.fc2.weight = torch.nn.Parameter(torch.from_numpy(self.file['fc2']['w'][:])).type(torch.float)
      self.base_network.fc2.bias = torch.nn.Parameter(torch.from_numpy(self.file['fc2']['b'][:])).type(torch.float)
      self.value_network.value.weight = torch.nn.Parameter(torch.from_numpy(self.file['v']['w'][:])).type(torch.float)
      self.value_network.value.bias = torch.nn.Parameter(torch.from_numpy(self.file['v']['b'][:])).type(torch.float)
      self.policy_network.policy.weight = torch.nn.Parameter(torch.from_numpy(self.file['p']['w'][:])).type(torch.float)
      self.policy_network.policy.bias = torch.nn.Parameter(torch.from_numpy(self.file['p']['b'][:])).type(torch.float)
      self.joint_network.updates = self.file.attrs['updates']
    else:
      logging.debug('Saving initial weights to %s', filename)
      fc1 = self.file.create_group('fc1')
      fc2 = self.file.create_group('fc2')
      v = self.file.create_group('v')
      p = self.file.create_group('p')

      fc1.create_dataset('w', data=self.base_network.fc1.weight.data.numpy())
      fc1.create_dataset('b', data=self.base_network.fc1.bias.data.numpy())
      fc2.create_dataset('w', data=self.base_network.fc2.weight.data.numpy())
      fc2.create_dataset('b', data=self.base_network.fc2.bias.data.numpy())
      v.create_dataset('w', data=self.value_network.value.weight.data.numpy())
      v.create_dataset('b', data=self.value_network.value.bias.data.numpy())
      p.create_dataset('w', data=self.policy_network.policy.weight.data.numpy())
      p.create_dataset('b', data=self.policy_network.policy.bias.data.numpy())
      
      self.file.attrs['updates'] = 0

      self.file.flush()
    self.lock.release()

    self.joint_network.share_memory()

  def start(self):
    """ Start the server asynchronously. """
    t = threading.Thread(target=self._run)
    t.daemon = True
    t.start()

  def _run(self):
    logging.debug('Starting learning server.')

    writer = MetricsWriter('/tmp/plato')
    writer.start_listening()

    # Client ID -> (learner process, pipe)
    client_pool = dict()

    # Client ID -> time of last received packet
    last_received = dict()

    # Packet format (omitting the client ID, which is stripped)
    packet_fmt = '>Bf' + ('f'* self.state_dims)

    # Start the gradient applier process
    gradient_queue = mp.SimpleQueue()
    optimizer = optim.Adam(self.joint_network.parameters())
    gradient_applier_process = mp.Process(target=self.gradient_applier, 
      args=(self.joint_network, gradient_queue, optimizer))
    gradient_applier_process.start()

    # Start listening for packets
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((self.ip, self.port))

    logging.info('Listening for client packets on %s:%d', self.ip, self.port)

    while True:
      # Scan for "dead" clients (no packets for >20 seconds)
      # TODO: Do this sparingly, instead of after every packet
      removed_clients = []
      for client, timestamp in last_received.items():
        if time.time() - timestamp > 20:
          # Remove dead clients from pool

          logging.info('Client %s has become inactive, stopping learner (%d ' \
                       'remaining)', client, len(client_pool)-1)

          client_pool[client][1].send(b'STOP')
          removed_clients.append(client)
      for client in removed_clients:
        del client_pool[client]
        del last_received[client]

      buf = sock.recv(65535)

      # Extract and strip the client ID from the received packet
      client_id = struct.unpack('<i', buf[:4])[0]
      buf = buf[4:]

      logging.debug('Received client packet from %d' % client_id)

      last_received[client_id] = time.time()

      # If it's a new client, create a client and add to pool
      if client_id not in client_pool:
        logging.info('Creating process for new client %d (%d total)', client_id,
                     len(client_pool)+1)

        client_pool[client_id] = self.create_learner_process(self.joint_network,
                                                    gradient_queue, packet_fmt, writer)

      # Check if the learner process died and recreate
      if not client_pool[client_id][0].is_alive():
        logging.warning('Learner %d died - recreating', client_id)

        client_pool[client_id] = self.create_learner_process(self.joint_network,
                                                    gradient_queue, packet_fmt, writer)

      # Forward the packet to the learner process
      try:
        client_pool[client_id][1].send(buf)
      except:
        logging.error('Failed to send a packet to %d learner', client_id)

  def create_learner_process(self, joint_network, gradient_queue, packet_fmt, writer):
    logging.debug('Creating learner process')

    parent_conn, child_conn = mp.Pipe()

    p = mp.Process(target=learner.start_learner, 
                 args=(child_conn, joint_network, gradient_queue, writer, packet_fmt))
    p.start()
    return (p, parent_conn)

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
  #       print(local_param)
  #       print(local_param.grad)
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