import tensorflow as tf
import socket
import struct

SERVER_IP = '127.0.0.1'
SERVER_PORT = 8000

'''
Packet format:
 - Agent heading [float16]: the heading of the agent.
 - Agent energy [float16]: the energy of the agent.
 - Opponent bearing [float16]: the opponent's bearing.
 - Opponent energy [float16]: the opponent's energy.
 - Action [unsigned char]: the action that was selected.
 - Reward [float16]: the reward after the action was taken.
'''
PACKET_FMT = '<ffffBf'

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

while True:
    data = sock.recvfrom(struct.calcsize(PACKET_FMT))
