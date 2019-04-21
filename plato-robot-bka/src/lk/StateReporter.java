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
	
	public StateReporter(String host, int port) {
		try {
			Random rand = new Random();
			this.ID = rand.nextInt();
			this.host = InetAddress.getByName(host);
			this.port = port;
			this.s = new DatagramSocket();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void report(float agentHeading, float agentEnergy, float oppBearing, float oppEnergy, float reward, int action) {
		try {
			ByteBuffer buf = ByteBuffer.allocate(21);
			buf.put((byte)action);
			buf.putFloat(reward);
			buf.putFloat(agentHeading);
			buf.putFloat(agentEnergy);
			buf.putFloat(oppBearing);
			buf.putFloat(oppEnergy);
			
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
	}
	
	public void close() {
		try {
//			this.s.close();
			this.send("STOP".getBytes());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}
