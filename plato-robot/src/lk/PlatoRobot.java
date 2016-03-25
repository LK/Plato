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
	
	public void run() {
		buildNetwork();
		try {
			File file = this.getDataFile(this.getCurrentTimeStamp() + ".txt");
			file.createNewFile();
			FileOutputStream outputStream = new FileOutputStream(file);
			this.output = new BufferedWriter(new OutputStreamWriter(outputStream));
		} catch (Exception e) {
			System.out.println("COULD NOT CREATE FILE");
		}
		
		while (true) {
			this.setTurnRadarRight(360);
			this.execute();
		}
	}
	
	// When we scan a robot, record a new state.
	public void onScannedRobot(ScannedRobotEvent event) {
		this.write(this.getHeading() + " " + this.getEnergy() + " " + event.getBearing() + " " + event.getEnergy());
	}
	
	public void onWin(WinEvent event) {
		this.write("W");
		try {
			this.output.close();
		} catch (IOException e) {
			System.out.println("COULDN'T WRITE/UPLOAD FILE");
		}
	}
	
	public void onDeath(DeathEvent event) {
		this.write("L");
		try {
			this.output.close();
		} catch (IOException e) {
			System.out.println("COULDN'T WRITE TO FILE");
		}
	}
	
	public String getCurrentTimeStamp() {
	    return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
	}
	
	public void write(String s) {
		try {
			this.output.write(s);
			this.output.newLine();
		} catch (IOException e) {
			System.out.println("COULDN'T WRITE TO FILE");
		}
	}
	
	public void buildNetwork() {
		try {
			File dataFile = this.getDataFile("network.h5");
			FileUtils.copyURLToFile(new URL("http://localhost:8080/download"), dataFile);
			IHDF5SimpleReader reader = HDF5Factory.openForReading(dataFile);
			
			double[] weights = reader.readDoubleArray("/network");
			NeuralNetwork net = new MultiLayerPerceptron(5, 10, 3, 3, 1, 1);

			System.out.println(weights.length);
			
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
