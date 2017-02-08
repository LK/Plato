import argparse
import logging
from server import LearningServer, WeightServer
import signal
import sys
import threading

def main():
  logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

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
      '--state-dims', type=int, default=4, dest='state_dims',
      help='The number of dimensions in the state space.')
  parser.add_argument(
      '--actions', type=int, default=4,
      help='The number of possible actions.')

  args = parser.parse_args()

  learning_server = LearningServer(args.state_dims, args.ip, args.learn_port, 
    'network.hdf5')
  learning_server.start()

  weight_server = WeightServer(args.ip, args.weight_port, 'network.hdf5')
  weight_server.start()

  def signal_handler(signal, frame):
    logging.info('Stopping...')
    sys.exit(0)

  signal.signal(signal.SIGINT, signal_handler)
  signal.pause()

if __name__ == '__main__':
	main()