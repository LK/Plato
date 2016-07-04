import logging
from multiprocessing import Pipe
import struct
import sys
import tensorflow as tf

'''
Packet format:
 - Action [unsigned char]:			the action that was selected after the state in
																this packet.
 - Reward [float16]:						the reward received after the above action.
 - Agent heading [float16]:			the heading of the agent before the action.
 - Agent energy [float16]:			the energy of the agent before the action.
 - Opponent bearing [float16]:	the opponent's bearing before the action.
 - Opponent energy [float16]:		the opponent's energy before the action.
'''

PACKET_FMT = '<Bfffff'

def start_learner(pipe):
	if pipe == None:
		logging.error('Attempted to start learner without server pipe - exiting')
		sys.exit(1)

	# Train loop
	while True:
		data = pipe.recv()
		logging.debug('Received packet')

		try:
			packet = struct.unpack(PACKET_FMT, data)
		except struct.error as err:
			logging.error('Bad packet, skipping: %s', str(err))
			continue

		action = packet[0]
		reward = packet[1]
		state  = packet[2:]
