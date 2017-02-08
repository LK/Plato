package lk;

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
	
	public void report(float agentHeading, float agentEnergy, float oppBearing, float oppEnergy, float reward) {
		try {
			ByteBuffer buf = ByteBuffer.allocate(25);
			buf.putInt(this.ID);
			buf.put((byte)0);
			buf.putFloat(reward);
			buf.putFloat(agentHeading);
			buf.putFloat(agentEnergy);
			buf.putFloat(oppBearing);
			buf.putFloat(oppEnergy);
			
			DatagramPacket packet = new DatagramPacket(buf.array(), buf.capacity(), this.host, this.port);
			this.s.send(packet);
		} catch (Exception exception) {
			exception.printStackTrace();
		}
	}
	
	public void close() {
		try {
			this.s.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}
