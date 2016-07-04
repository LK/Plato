import argparse
import learner
import logging
from multiprocessing import Process, Pipe
import socket

def main():
	logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s',
											datefmt='%m/%d/%Y %I:%M:%S %p')

	parser = argparse.ArgumentParser(
			description='Start the learning server for Plato.')

	parser.add_argument(
			'learners', type=int, default=1,
			help='The number of learner processes to start')
	parser.add_argument(
			'ip', type=string, default='127.0.0.1',
			help='The IP address to listen to connections on.')
	parser.add_argument(
			'port', type=int, default='8080',
			help='The port to listen to connections on.')

	args = parser.parse_args()

	# Set up a pool of learner processes (and corresponding pipes)
	pool = []
	conn = []

	for i in range(args.learners):
		process, parent_conn = create_learner_process()

		pool.append(process)
		conn.append(conn)

	logging.info('Created learner pool with %d processes', len(pool))

	# Set up the load balancer to forward requests to individual learner processes
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((args.ip, args.port))

	logging.info('Listening for client packets on %s:%d'. args.ip, args.port)

	pool_index = 0

	while True:
		buf = sock.recv(65535)
		logging.debug('Received client packet')

		# Check if the learner process died
		if not pool[pool_index].is_alive():
			logging.warning('Learner %d died - recreating', pool_index)

			# Remove from the pool
			del pool[pool_index]
			del conn[pool_index]

			# Create another learner process
			process, parent_conn = create_learner_process()
			pool.append(process)
			conn.append(conn)

		try:
			conn[pool_index].send(buf)
		except:
			logging.error('Failed to send a packet to learner %d', pool_index)

		pool_index = (pool_index + 1) % len(pool)

def create_learner_process():
	logging.debug('Creating learner process')

		parent_conn, child_conn = Pipe()

		# TODO: Don't assume learner initialization succeeded
		p = Process(target=learner.start_learner, args=(child_conn,))
		return (p, parent_conn)

def __name__ == '__main__':
	main()
