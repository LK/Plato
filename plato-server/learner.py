import logging
import struct
import sys
import tensorflow as tf

GAMMA = 0.99

def start_learner(pipe, packet_fmt='<Bfffff'):
	if pipe == None:
		logging.error('Attempted to start learner without server pipe - exiting')
		sys.exit(1)

	# Stores the history of state-action-reward tuples for the entire duration of
	# the episode.
	history = []

	# Train loop
	while True:
		logging.debug('Started learner thread, waiting for data')

		data = pipe.recv()

		logging.debug('Received packet')

		if data == 'STOP':
			logging.info('Received STOP packet, will terminate after last update')

			R = 0
			for state, action, reward in reversed(history):
				R = reward + GAMMA*R


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
