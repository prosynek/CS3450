# Paige Rosynek
# CS 3450 021
# Lab 06 - Implementing Forward Propagation
# 04.23.2023

import torch
from numpy import newaxis as np_newaxis

# TODO: Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.


class Layer:
    def __init__(self, output_shape):
        """
        Initializes the output of the given layer and gradient to tensors of given shape.

        :param output_shape: tuple or int that represents the shape of the output of this layer
        """
        self.output = torch.zeros(output_shape)
        self.grad = torch.zeros(output_shape)

    def accumulate_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """

    def clear_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = self.grad * 0.0

    def step(self):
        """
        TODO: Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass


class Input(Layer):
    def __init__(self, output_shape, train):
        """
        Initializes Input layer instance.

        :param output_shape: tuple or int that represents the shape of the output of this layer
        :param train: boolean of whether or not to update the layer parameters during backpropagation
        """
        Layer.__init__(self, output_shape) 
        self.to_train = train
        

    def set(self, output):
        """
        Sets the output of this input layer.

        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        
        # check size of output to be set is == to output shape initialized
        assert self.output.shape == output.shape, 'Shape of parameter '+ str(output.shape) + ' does not match defined shape ' + str(self.output.shape)
        
        self.output = output

    def randomize(self):
        """
        Sets the output of this input layer to random values sampled from the standard normal
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.randn(self.output.shape)    # need assert?

    def forward(self):
        """
        Performs forward propagation for this layer. Input layer has no forward propagation steps.
        """
        pass

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self):
        """
        TODO: Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """


class Linear(Layer):
    def __init__(self, x_layer, W_layer, b_layer):
        """
        Initializes an instance of a Linear layer (Wx + b).

        :param x_layer: Input layer that represents x in Wx + b
        :param W_layer: Input layer that represents the weight matrix, W, in Wx + b
        :param b_layer: Input layer that represents the biases, b, in Wx + b
        """
        # check dimensions match
        assert W_layer.output.shape[1] == x_layer.output.shape[0], 'Cannot multiply tensors with shapes: ' + str(W_layer.output.shape) + '\t' + str(x_layer.output.shape)
        assert W_layer.output.shape[0] == b_layer.output.shape[0], 'Rows of weights must match rows of bias. Got ' + str(W_layer.output.shape) + '\t' + str(b_layer.output.shape)

        Layer.__init__(self, b_layer.output.shape) # TODO: Pass along any arguments to the parent's initializer here.
        self.x = x_layer
        self.W = W_layer
        self.b = b_layer

    def forward(self):
        """
        Performs forward propagation for this layer. output = Wx + b
        """
        self.output = torch.matmul(self.W.output, self.x.output) + self.b.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """


class ReLU(Layer):
    def __init__(self, x_layer):
        """
        Initializes ReLU activation layer instance.

        :param x_layer: Layer to perform ReLU activation on
        """
        Layer.__init__(self, x_layer.output.shape) 
        self.x = x_layer

    def forward(self):
        """
        Performs forward propagation for this layer. output = ReLU(x)
        """
        self.output = self.x.output * (self.x.output > 0)  

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, true_layer, pred_layer):
        """
        Initializes Mean-Squared Error Loss Layer instance.

        :param true_layer: Layer that represents the true values 
        :param pred_layer: Layer that contains the predicted values
        """
        # check dimensions match
        assert true_layer.output.shape == pred_layer.output.shape, 'Shape mismatch : '+ str(true_layer.output.shape) + ' must match ' + str(pred_layer.output.shape)

        Layer.__init__(self, 1) 
        self.y_true = true_layer
        self.y_pred = pred_layer

    def forward(self):
        """
        Performs forward propagation for this layer. output = L = MSE(y_true, y_pred)
        """
        self.output = torch.mean((self.y_true.output - self.y_pred.output) ** 2)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """


class Regularization(Layer):
    def __init__(self, coef, W_layer):
        """
        Initializes Regularization Layer instance.

        :param coef: scalar (int) that represents the regularization coefficient - lambda
        :param W_layer: Layer that represents the weight matrix to perform regularization on
        """
        Layer.__init__(self, 1) 
        self.coef = coef
        self.W = W_layer

    def forward(self):
        """
        Performs forward propagation for this layer. Calculates the squared frobenius norm of W * coef/2. 
        """
        self.output = (0.5 * self.coef) * torch.sum(self.W.output**2) 

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """


class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    TODO: Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, x_layer, y_true):
        """
        Initializes Softmax (+ cross-entropy) Layer instance.

        :param x_layer: Layer to perform softmax activation on 
        :param y_true: Layer that represents the true values for classification
        """
        # check dimensions match
        assert x_layer.output.shape == y_true.output.shape, 'Shape mismatch : '+ str(x_layer.output.shape) + ' must match ' + str(y_true.output.shape)

        Layer.__init__(self, 1) 
        self.x = x_layer
        self.classifications = torch.zeros(x_layer.output.shape)  # to store softmax output (o)
        self.y_true = y_true

    def forward(self):
        """
        Performs forward propagation for this layer. Calculates classifications = o = softmax(x) and output = CE(o, y_true)
        """
        # softmax
        exp_x = torch.exp(self.x.output - torch.max(self.x.output, dim=0).values)
        self.classifications = exp_x / torch.sum(exp_x, dim=0)

        # cross-entropy loss
        k = torch.sum(-1 * self.y_true.output * torch.log(self.classifications + 1e-8), dim=1)
        self.output = torch.mean(k)

        # print(f'CE loss = {self.output}')
        # print(f'CE loss shape = {self.output.shape}\n\n')

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """


class Sum(Layer):
    def __init__(self, s1_layer, s2_layer):
        """
        Initializes Sum Layer instance.

        :param s1_layer: Layer to add
        :param s2_true: Layer to add
        """
        Layer.__init__(self, 1) 
        self.s1 = s1_layer
        self.s2 = s2_layer

    def forward(self):
        """
        Performs forward propagation for this layer. output = s1 + s2
        """
        self.output = self.s1.output + self.s2.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """


