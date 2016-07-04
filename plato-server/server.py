import argparse
import learner
import logging
from multiprocessing import Process, Pipe
import socket
import time

'''
Packet format:
 - Client ID [int32]:						a random unique identifier to identify an
 																individual client.
 - Action [unsigned char]:			the action that was selected after the state in
																this packet.
 - Reward [float16]:						the reward received after the above action.
 - Agent heading [float16]:			the heading of the agent before the action.
 - Agent energy [float16]:			the energy of the agent before the action.
 - Opponent bearing [float16]:	the opponent's bearing before the action.
 - Opponent energy [float16]:		the opponent's energy before the action.
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
			'--port', type=int, default='8000',
			help='The port to listen to connections on.')

	args = parser.parse_args()

	# Set up a mapping of client ID -> (process, pipe)
	pool = {}

	# Set up a mapping of client ID -> timestamp of last received packet
	last_received = {}

	# Set up the load balancer to forward requests to individual learner processes
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((args.ip, args.port))

	logging.info('Listening for client packets on %s:%d', args.ip, args.port)

	while True:
		# Scan for "dead" clients (no packets for >20 seconds)
		for client, timestamp in last_received.iteritems():
			if time.time() - timestamp > 20:
				# Send STOP packet and remove from pool

				logging.info('Client %s has become inactive, stopping learner (%d ' \
										 'remaining)', client, len(pool)-1)

				pool[client][1].send('STOP')
				del pool[client]

		buf = sock.recv(65535)
		logging.debug('Received client packet')

		# Extract and strip the client ID and exit flag from the received packet
		client_id = struct.unpack('<i', buf)
		buf = buf[4:]

		last_received[client_id] = time.time()

		# If it's a new client, create a client and add to pool
		if client_id not in pool:
			logging.info('Creating process for new client %d (%d total)', client_id,
									 len(pool))

			pool[client_id] = create_learner_process()

		# Check if the learner process died and recreate
		if not pool[client_id][0].is_alive():
			logging.warning('Learner %d died - recreating', pool_index)

			pool[client_id] = create_learner_process()

		try:
			pool[client_id][1].send(buf)
		except:
			logging.error('Failed to send a packet to %d learner', client_id)

def create_learner_process():
	logging.debug('Creating learner process')

	parent_conn, child_conn = Pipe()

	# TODO: Don't assume learner initialization succeeded
	p = Process(target=learner.start_learner, args=(child_conn,))
	return (p, parent_conn)

if __name__ == '__main__':
	main()
