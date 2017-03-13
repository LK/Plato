package lk;

import java.io.File;
import java.net.URL;

import org.apache.commons.io.FileUtils;
import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.transfer.Linear;
import org.neuroph.core.transfer.RectifiedLinear;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.comp.neuron.BiasNeuron;

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;

public class Network {

	private NeuralNetwork<?> baseNetwork;
	private NeuralNetwork<?> valueNetwork;
	private NeuralNetwork<?> policyNetwork;
	
	public double[] base(double[] input) {
		baseNetwork.setInput(input);
		baseNetwork.calculate();
		return baseNetwork.getOutput();
	}
	
	public double value(double[] input) {
		valueNetwork.setInput(base(input));
		valueNetwork.calculate();
		return valueNetwork.getOutput()[0];
	}
	
	public double[] policy(double[] input) {
		policyNetwork.setInput(base(input));
		policyNetwork.calculate();
		double[] outputs = policyNetwork.getOutput();
		
		double[] exps = new double[outputs.length];
		double sum = 0;
		for (int i = 0; i < outputs.length; i++) {
			exps[i] = Math.exp(outputs[i]);
			sum += Math.exp(outputs[i]);
		}
		for (int i = 0; i < exps.length; i++) {
			exps[i] /= sum;
		}
		
		return exps;
	}
	
	public void downloadNetwork(String address, File dataFile) {
		try {
			FileUtils.copyURLToFile(new URL(address), dataFile);
			IHDF5Reader reader = HDF5Factory.openForReading(dataFile);
			
			System.out.format("Loaded network %d\n", reader.getIntAttribute("/", "updates"));
			
			this.baseNetwork = new MultiLayerPerceptron(4, 128, 128);
			setupLayer(this.baseNetwork.getLayerAt(1), reader.readFloatMatrix("/fc1/w"), reader.readFloatArray("/fc1/b"), new RectifiedLinear());
			setupLayer(this.baseNetwork.getLayerAt(2), reader.readFloatMatrix("/fc2/w"), reader.readFloatArray("/fc2/b"), new RectifiedLinear());
			
			this.valueNetwork = new MultiLayerPerceptron(128, 1);
			setupLayer(this.valueNetwork.getLayerAt(1), reader.readFloatMatrix("/v/w"), reader.readFloatArray("/v/b"), new Linear());
			
			this.policyNetwork = new MultiLayerPerceptron(128, 4);
			this.policyNetwork.getLayerAt(0).removeNeuronAt(128);
			setupLayer(this.policyNetwork.getLayerAt(1), reader.readFloatMatrix("/p/w"), reader.readFloatArray("/p/b"), new Linear());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void setupLayer(Layer layer, float[][] weight, float[] bias, TransferFunction function) {
		int w_i = 0;
		int w_j = 0;
		int b_i = 0;

		for (Neuron neuron : layer.getNeurons()) {
			for (Connection conn : neuron.getInputConnections()) {
				neuron.setTransferFunction(function);
				if (conn.getFromNeuron() instanceof BiasNeuron) {
					conn.setWeight(new Weight((double)bias[b_i++]));
				} else {
					conn.setWeight(new Weight((double)weight[w_i][w_j++]));
				}
			}
			w_i++;
			w_j = 0;
		}
	}
	
//	public static void main(String[] args) {
//		long startTime = System.nanoTime();
//		Network n = new Network();
//		n.downloadNetwork("http://localhost:8001", new File("net.hdf5"));
//		long endTime = System.nanoTime();
//
//		long duration = (endTime - startTime);
//		System.out.println(duration/1000000);
//		
//		double[] a = {-1, 1, 2, 3};
//		double[] b = {24, -123, 31, -31.3};
//		double[] c = {-22.123, 123.3, 312.3, 3100};
//		double[] d = {.2233, .141414, -.003, -.223};
//		
//		System.out.format("%f %f %f %f\n", n.policy(a)[0], n.policy(a)[1], n.policy(a)[2], n.policy(a)[3]);
//		System.out.format("%f %f %f %f\n", n.policy(b)[0], n.policy(b)[1], n.policy(b)[2], n.policy(b)[3]);
//		System.out.format("%f %f %f %f\n", n.policy(c)[0], n.policy(c)[1], n.policy(c)[2], n.policy(c)[3]);
//		System.out.format("%f %f %f %f\n", n.policy(d)[0], n.policy(d)[1], n.policy(d)[2], n.policy(d)[3]);
//		
//		System.out.println(n.value(a));
//		System.out.println(n.value(b));
//		System.out.println(n.value(c));
//		System.out.println(n.value(d));
//	}
}