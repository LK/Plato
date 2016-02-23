import numpy as np
import random

class ReLULayer(object):
    def __init__(self, size_in, size_out):
        ''' Initialize a new ReLU layer. '''
        self.size_in = size_in
        self.size_out = size_out

        # Initialize the weights as described in http://cs231n.github.io/neural-networks-2/.
        # Number of rows must equal size_in + 1 (for bias term), number of
        # columns must equal number of outputs.
        self.W = np.random.randn(self.size_in+1, self.size_out) * np.sqrt(2.0/self.size_in+1)

        # Because ReLUs that don't activate cut off the gradient, set the bias
        # weights to .5 to speed up training early on.
        self.W[:, 0] = .5

    def forward(self, in_act):
        ''' Perform a forward pass through this layer. '''
        # Add column of 1s to input activations for bias
        if len(in_act.shape) == 1:
            self.in_act = np.insert(in_act, 0, 1)
        else:
            self.in_act = np.insert(in_act, 0, np.ones(in_act.shape[0]), axis=1)

        # Compute the activations, capping at 0 (ReLU)
        self.out_act = np.dot(self.in_act, self.W)
        self.out_act[self.out_act < 0] = 0

        return self.out_act

    def backward(self, grad):
        # The derivative of the output wrt to the weights is just the inputs.
        # However, because the ReLU function has a derivative of 0 when x < 0,
        # we need to keep in mind that the derivative of the output wrt to the
        # weights becomes 0 when the activation is 0.
        output_wrt_weights = np.ones(self.W.shape) * self.in_act[:, None]
        output_wrt_weights[:, self.out_act < 0] = 0

        # Because the output of this neuron has "influence" on every neuron in
        # the following layer, we need to sum over the gradients.
        cost_wrt_output = np.sum(np.atleast_2d(grad), axis=1)

        # Chain rule!
        cost_wrt_weights = cost_wrt_output * output_wrt_weights

        self.dW = cost_wrt_weights

        # Now compute the derivative of the output wrt to the inputs (which is
        # just the weights). Once again, we must remember that when the
        # activation is < 0, the output does not change with the inputs.
        output_wrt_inputs = self.W
        output_wrt_inputs[:, self.out_act < 0] = 0

        # Chain rule!
        cost_wrt_inputs = cost_wrt_output * output_wrt_inputs

        # Return the derivative of the cost wrt to the inputs to continue on in
        # the next layer. Remember to take out the bias term derivatives - the
        # next layer doesn't care about that.
        return cost_wrt_inputs[1:, :]

class LinearLayer(object):
    def __init__(self, size_in, size_out):
        ''' Initialize a new linear layer. '''
        self.size_in = size_in
        self.size_out = size_out

        # Initialize the weights as described in http://cs231n.github.io/neural-networks-2/.
        # Number of rows must equal size_in + 1 (for bias term), number of
        # columns must equal number of outputs.
        self.W = np.random.randn(self.size_in+1, self.size_out) * np.sqrt(2.0/self.size_in+1)

    def forward(self, in_act):
        ''' Perform a forward pass through this layer. '''
        # Add column of 1s to input activations for bias
        if len(in_act.shape) == 1:
            self.in_act = np.insert(in_act, 0, 1)
        else:
            self.in_act = np.insert(in_act, 0, np.ones(in_act.shape[0]), axis=1)

        # Compute the activations
        self.out_act = np.dot(self.in_act, self.W)

        return self.out_act

    def backward(self, grad):
        ''' Backpropagate through this layer. '''
        output_wrt_weights = np.ones(self.W.shape) * self.in_act[:, None]
        cost_wrt_output = np.sum(np.atleast_2d(grad), axis=1)
        cost_wrt_weights = cost_wrt_output * output_wrt_weights
        self.dW = cost_wrt_weights
        output_wrt_inputs = self.W
        cost_wrt_inputs = cost_wrt_output * output_wrt_inputs
        return cost_wrt_inputs[1:, :]

class NeuralNetwork(object):
    def __init__(self, layers, alpha=.001):
        self.layers = layers
        self.alpha = alpha

    def forward(self, inputs):
        ''' Perform a forward pass through the entire network. '''
        prev_act = inputs
        for layer in self.layers:
            prev_act = layer.forward(prev_act)

        return prev_act

    def backward(self, dJ):
        ''' Perform a backward pass through the entire network. '''
        prev_grad = dJ
        for layer in reversed(self.layers):
            prev_grad = layer.backward(prev_grad)

    def update(self):
        for layer in self.layers:
            before = layer.W
            layer.W = layer.W - layer.dW * self.alpha

class Trainer(object):
    def __init__(self, net, X, y):
        self.net = net
        self.X = X
        self.y = np.matrix(y).T

    def train(self):
        # Perform one training epoch (using each training sample once, in random
        # order)
        # TODO: Extract cost function computation, add regularization term
        order = list(range(self.X.shape[0]))
        random.shuffle(order)
        for i in order:
            out = self.net.forward(self.X[i,:])
            dJ = out - self.y[i]
            self.net.backward(dJ[0,0])
            self.net.update()

        out_ = self.net.forward(self.X)
        cost = np.linalg.norm(out_ - self.y)
        return cost

# Perform numerical gradient computation to validate backward() implementation
def grad_check():
    net = NeuralNetwork([ReLULayer(2, 10), LinearLayer(10, 5), ReLULayer(5, 1)])

    testin = np.array([0, 0])
    for layer in net.layers:
        out = net.forward(testin)
        net.backward(out)

        computed_dW = layer.dW
        numerical_dW = np.ones(layer.dW.shape)
        for (x,y), val in np.ndenumerate(layer.W):
            layer.W[x,y] = val + 1e-10
            out1 = net.forward(testin)
            layer.W[x,y] = val
            numerical_dW[x, y] = (.5 * (out1**2) - .5 * (out**2))/1e-10

        print computed_dW - numerical_dW

def main():
    # Uncomment to run gradient check and exit
    #grad_check()
    #return

    i = 0
    net = NeuralNetwork([ReLULayer(2, 10), ReLULayer(10, 1)], alpha=.01)
    trainer = Trainer(net, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]).T)
    try:
        while True:
            print trainer.train()
            i += 1
    except KeyboardInterrupt:
        print str(i) + ' epochs'
        print net.forward(np.array([0, 0]))
        print net.forward(np.array([0, 1]))
        print net.forward(np.array([1, 0]))
        print net.forward(np.array([1, 1]))
        sys.exit(0)

if __name__ == '__main__':
    main()
