package lk;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.Random;

public class StateReporter {

	DatagramSocket s;
	InetAddress host;
	int port;
	int ID;
	State lastState;

	int packetsSent;

	public StateReporter(String host, int port) {
		try {
			Random rand = new Random();
			this.ID = rand.nextInt();
			this.host = InetAddress.getByName(host);
			this.port = port;
			this.s = new DatagramSocket();
			this.packetsSent = 0;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void recordTransition(int action, float reward, State newState, boolean isTerminal) {
		if (this.lastState == null) {
			this.lastState = newState;
			return;
		}

		try {
			ByteBuffer buf = ByteBuffer.allocate(State.size() * 2 + 6);
			this.lastState.writeToBuffer(buf);
			buf.put((byte) action);
			buf.putFloat(reward);
			newState.writeToBuffer(buf);
			buf.put((byte) (isTerminal ? 1 : 0));

			this.lastState = newState;

			this.send(buf.array());
		} catch (Exception exception) {
			exception.printStackTrace();
		}
	}

	private void send(byte[] bytes) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(bytes.length + 4);
		buf.putInt(this.ID);
		buf.put(bytes);
		DatagramPacket packet = new DatagramPacket(buf.array(), buf.capacity(), this.host, this.port);
		this.s.send(packet);
		this.packetsSent += 1;
	}

	public void close() {
		try {
			this.s.close();
			System.out.println("Closing socket with " + this.packetsSent + " packets sent");
			// this.send("STOP".getBytes());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
