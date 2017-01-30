import argparse
import learner
import logging
from network import *
import socket
import struct
import time
import torch.multiprocessing as mp
import torch.optim as optim

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

def main():
  logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

  parser = argparse.ArgumentParser(
      description='Start the learning server for Plato.')

  parser.add_argument(
      '--ip', default='127.0.0.1',
      help='The IP address to listen to connections on.')
  parser.add_argument(
      '--port', type=int, default=8000,
      help='The port to listen to connections on.')
  parser.add_argument(
      '--state-dims', type=int, default=4, dest='state_dims',
      help='The number of dimensions in the state space.')
  parser.add_argument(
      '--actions', type=int, default=4,
      help='The number of possible actions.')

  args = parser.parse_args()

  # Build the networks
  base   = BaseNetwork(args.state_dims)
  value  = ValueNetwork(base)
  policy = PolicyNetwork(base)
  joint  = JointNetwork(value, policy)
  joint.share_memory()

  # Queue of gradients coming in from the workers
  gradient_queue = mp.Queue()

  # The format for client packets
  packet_fmt = '>Bf' + ('f'*args.state_dims)

  # Set up a mapping of client ID -> (process, pipe)
  pool = {}

  # Set up a mapping of client ID -> timestamp of last received packet
  last_received = {}

  # Set up the load balancer to forward requests to individual learner processes
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind((args.ip, args.port))

  # Start the gradient applier process
  optimizer = optim.Adam(joint.parameters())
  gradient_applier_process = mp.Process(target=gradient_applier, 
                                        args=(joint, gradient_queue, optimizer))
  gradient_applier_process.start()

  logging.info('Listening for client packets on %s:%d', args.ip, args.port)

  while True:
    # Scan for "dead" clients (no packets for >20 seconds)
    # TODO: Do this sparingly, instead of after every packet
    for client, timestamp in last_received.items():
      if time.time() - timestamp > 20:
        # Remove dead clients from pool

        logging.info('Client %s has become inactive, stopping learner (%d ' \
                     'remaining)', client, len(pool)-1)

        pool[client][0].terminate()
        del pool[client]
        del last_received[client]

    (buf, addr) = sock.recvfrom(65535)
    logging.debug('Received client packet')

    # Extract and strip the client ID from the received packet
    client_id = struct.unpack('<i', buf[:4])[0]
    buf = buf[4:]

    last_received[client_id] = time.time()

    # If it's a new client, create a client and add to pool
    if client_id not in pool:
      logging.info('Creating process for new client %d (%d total)', client_id,
                   len(pool))

      pool[client_id] = create_learner_process(joint, gradient_queue,
                                               packet_fmt)

    # Check if the learner process died and recreate
    if not pool[client_id][0].is_alive():
      logging.warning('Learner %d died - recreating', client_id)

      pool[client_id] = create_learner_process(joint, gradient_queue, 
                                               packet_fmt)

    # Forward the packet to the learner process
    try:
      pool[client_id][1].send(buf)
    except:
      logging.error('Failed to send a packet to %d learner', client_id)

def create_learner_process(joint_network, gradient_queue, packet_fmt):
  logging.debug('Creating learner process')

  parent_conn, child_conn = mp.Pipe()

  p = mp.Process(target=learner.start_learner, 
                 args=(child_conn, joint_network, gradient_queue, packet_fmt))
  p.start()
  return (p, parent_conn)

def gradient_applier(global_model, gradient_queue, optimizer):
  logging.debug('Starting gradient applier process')
  while True:
    optimizer.zero_grad()
    local_params = gradient_queue.get()
    logging.debug('Applying gradients')
    for (local_param, global_param) in zip(local_params, 
                                           global_model.parameters()):
      global_param.grad.data = local_param.grad.data

    optimizer.step()

if __name__ == '__main__':
  main()
