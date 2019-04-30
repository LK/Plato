import argparse
import logging
from server import EnvironmentServer, WeightServer
import signal
import sys
from torch.multiprocessing import Lock

def main():
  logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

  parser = argparse.ArgumentParser(
      description='Start the learning server for Plato.')

  parser.add_argument(
      '--ip', default='127.0.0.1',
      help='The IP address to listen to connections on.')
  parser.add_argument(
      '--learn-port', type=int, default=8000, dest='learn_port',
      help='The port to start the learning server on.')
  parser.add_argument(
      '--weight-port', type=int, default=8001, dest='weight_port',
      help='The port to start the weight server on.')
  parser.add_argument(
      '--state-dims', type=int, default=8, dest='state_dims',
      help='The number of dimensions in the state space.')
  parser.add_argument(
      '--actions', type=int, default=6,
      help='The number of possible actions.')

  args = parser.parse_args()

  lock = Lock()

  weight_server = WeightServer(args.ip, args.weight_port, 'network.hdf5', lock)
  weight_server.start()

  learning_server = EnvironmentServer(args.state_dims, args.actions, args.ip, args.learn_port, 
    'network.hdf5', lock)
  learning_server.start()

  def signal_handler(signal, frame):
    logging.info('Stopping...')
    sys.exit()

  signal.signal(signal.SIGINT, signal_handler)
  signal.pause()

if __name__ == '__main__':
	main()