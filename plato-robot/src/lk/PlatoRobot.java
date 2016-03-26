package lk;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import robocode.AdvancedRobot;
import robocode.DeathEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;
import ch.systemsx.cisd.hdf5.*;

import org.apache.commons.io.FileUtils;
import org.neuroph.core.*;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.MultiLayerPerceptron;

public class PlatoRobot extends AdvancedRobot {

	private final boolean LEARN = true;
	private BufferedWriter output;
	private NeuralNetwork network;
	private int consecutiveWins = 0;
	private double epsilon = 0.0f;
	
	private enum Action {
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT,
		FIRE,
		NOTHING
	}
	
	public void run() {
		if (this.consecutiveWins % 10 == 0) buildNetwork();
		
		if (LEARN) {
			try {
				File file = this.getDataFile(this.getCurrentTimeStamp() + ".txt");
				System.out.println(file.getAbsolutePath());
				file.createNewFile();
				FileOutputStream outputStream = new FileOutputStream(file);
				this.output = new BufferedWriter(new OutputStreamWriter(outputStream));
			} catch (Exception e) {
				System.out.println("COULD NOT CREATE FILE");
			}
		}
		
		while (true) {
			this.setTurnRadarRight(360);
			this.execute();
		}
	}
	
	// When we scan a robot, record a new state.
	public void onScannedRobot(ScannedRobotEvent event) {
		Action a;
		Random rand = new Random();
		double r = rand.nextDouble();
		if (r < this.epsilon) {
			r = rand.nextDouble();
			if (r < 1.0/6.0) {
				a = Action.FORWARD;
			} else if (r < 2.0/6.0) {
				a = Action.BACKWARD;
			} else if (r < 3.0/6.0) {
				a = Action.LEFT;
			} else if (r < 4.0/6.0) {
				a = Action.RIGHT;
			} else if (r < 5.0/6.0) {
				a = Action.FIRE;
			} else {
				a = Action.NOTHING;
			}
		} else {
			double maxQ = 0.0;
			Action maxA = Action.NOTHING;
			for (Action A : Action.values()) {
				this.network.setInput(this.getHeading(), this.getEnergy(), event.getBearing(), event.getEnergy(), A.ordinal());
				this.network.calculate();
				double Q = this.network.getOutput()[0];
				if (Q > maxQ) {
					maxQ = Q;
					maxA = A;
				}
			}
			a = maxA;
		}
		
		switch (a) {
		case FORWARD:
			this.setAhead(10);
			break;
		case BACKWARD:
			this.setBack(10);
			break;
		case LEFT:
			this.setTurnLeft(10);
			break;
		case RIGHT:
			this.setTurnRight(10);
			break;
		case FIRE:
			this.setFire(1);
			break;
		default:
			break;
		}
		
		this.write(this.getHeading() + " " + this.getEnergy() + " " + event.getBearing() + " " + event.getEnergy() + " " + a.ordinal());
	}
	
	public void onWin(WinEvent event) {
		this.write("W");
		this.consecutiveWins += 1;
		
		if (LEARN) {
			try {
				this.output.close();
			} catch (IOException e) {
				System.out.println("COULDN'T WRITE/UPLOAD FILE");
			}
		}
	}
	
	public void onDeath(DeathEvent event) {
		this.write("L");
		this.consecutiveWins = 0;
		
		if (LEARN) {
			try {
				this.output.close();
			} catch (IOException e) {
				System.out.println("COULDN'T WRITE TO FILE");
			}
		}
	}
	
	public String getCurrentTimeStamp() {
	    return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
	}
	
	public void write(String s) {
		if (!LEARN) return;
		
		try {
			this.output.write(s);
			this.output.newLine();
		} catch (IOException e) {
			System.out.println("COULDN'T WRITE TO FILE");
		}
	}
	
	public void buildNetwork() {
		try {
			IHDF5SimpleReader reader;
			
			if (LEARN) {
				File dataFile = this.getDataFile("network.h5");
				FileUtils.copyURLToFile(new URL("http://localhost:8080/download"), dataFile);
				reader = HDF5Factory.openForReading(dataFile);
			} else {
				reader = HDF5Factory.openForReading(this.getDataFile("saved_network.h5"));
			}
			
			this.epsilon = reader.readDouble("/epsilon");
			System.out.println(this.epsilon);
			double[] weights = reader.readDoubleArray("/network");
			MultiLayerPerceptron net = new MultiLayerPerceptron(5, 256, 256, 256, 1, 1);			
			int idx = 0;
			
			ReLUTransferFunction transferFunction = new ReLUTransferFunction();
			
			// This is excruciatingly stupid (and slow!), but Torch and Neuroph flatten their weights differently.
			for (int i = 1; i < net.getLayersCount() - 1; i++) {
				Layer layer = net.getLayerAt(i);
				for (int j = 0; j < layer.getNeuronsCount() - 1; j++) {
					layer.getNeuronAt(j).setTransferFunction(transferFunction);
					Connection[] connections = layer.getNeuronAt(j).getInputConnections();
					for (int k = 0; k < connections.length - 1; k++) {
						connections[k].setWeight(new Weight(weights[idx++]));
					}
				}
				for (int j = 0; j < layer.getNeuronsCount() - 1; j++) {
					Connection[] connections = layer.getNeuronAt(j).getInputConnections();
					connections[connections.length - 1].setWeight(new Weight(weights[idx++]));
				}
			}
			
			net.getLayerAt(5).getNeuronAt(0).getInputConnections()[0].setWeight(new Weight(1.0));
			net.getLayerAt(5).getNeuronAt(0).getInputConnections()[1].setWeight(new Weight(0.0));
			
			for (int i = 0; i < net.getLayerAt(4).getNeuronsCount(); i++) {
				net.getLayerAt(4).getNeuronAt(i).setTransferFunction(new Linear());
			}
			for (int i = 0; i < net.getLayerAt(5).getNeuronsCount(); i++) {
				net.getLayerAt(5).getNeuronAt(i).setTransferFunction(new Linear());
			}
			
			this.network = net;
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
}
