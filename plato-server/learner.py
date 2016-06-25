import socket
import struct
import tensorflow as tf

'''
Packet format:
 - Action [unsigned char]:     the action that was selected after the state in
                               this packet.
 - Reward [float16]:           the reward received after the above action.
 - Agent heading [float16]:    the heading of the agent before the action.
 - Agent energy [float16]:     the energy of the agent before the action.
 - Opponent bearing [float16]: the opponent's bearing before the action.
 - Opponent energy [float16]:  the opponent's energy before the action.
'''
PACKET_FMT = '<ffffBf'
SERVER_IP = '127.0.0.1'
SERVER_PORT = 8000

def main():
    # Start the UDP server
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((SERVER_IP, SERVER_PORT))

    # Train loop
    while True:
        data = sock.recvfrom(struct.calcsize(PACKET_FMT))

        try:
            packet = struct.unpack(PACKET_FMT, data)
        except struct.error as err:
            print '[!] Received bad packet, skipping: ' + str(err)
            continue

        action = packet[0]
        reward = packet[1]
        state  = packet[2:]

def __name__ == '__main__':
    main()
