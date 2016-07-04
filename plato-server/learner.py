import logging
import struct
import sys
import tensorflow as tf

PACKET_FMT = '<Bfffff'

def start_learner(pipe):
	if pipe == None:
		logging.error('Attempted to start learner without server pipe - exiting')
		sys.exit(1)

	# Train loop
	while True:
		data = pipe.recv()

		logging.debug('Received packet')

		if data == 'STOP':
			logging.info('Received STOP packet, will terminate after last update')

			return

		try:
			packet = struct.unpack(PACKET_FMT, data)
		except struct.error as err:
			logging.error('Bad packet, skipping: %s', str(err))
			continue

		action = packet[0]
		reward = packet[1]
		state  = packet[2:]
