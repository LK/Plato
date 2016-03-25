package lk;

import org.neuroph.core.transfer.TransferFunction;

public class ReLUTransferFunction extends TransferFunction {

	@Override
	public double getOutput(double net) {
		return Math.max(net, 0.0);
	}
	
	@Override
	public double getDerivative(double net) {
		return net >= 0 ? 1.0 : 0.0;
	}

}
