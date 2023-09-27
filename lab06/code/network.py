# Paige Rosynek
# CS 3450 021
# Lab 06 - Implementing Forward Propagation
# 04.23.2023

class Network:
    def __init__(self):
        """
        Initializes Network instance.

        TODO: Initialize a `layers` attribute to hold all the layers in the gradient tape.
        """
        self.layers = []    # 'gradient tape'
        self.input = None
        self.output = None

    def add(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        # check that an input layer has been set before adding additional layers
        assert self.input != None, 'Must set input layer (set_input) before adding layers'

        self.layers.append(layer)

    def set_input(self, input_layer):
        """
        Sets the input layer for the network.

        :param input_layer: The sublayer that represents the signal input (e.g., the image to be classified)
        """
        self.input = input_layer

        # set input layer as first layer
        self.layers.append(input_layer)

    def forward(self):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward.

        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        # TODO: Implement this method
        # TODO: Either remove the input option and output options, or if you keep them, assign the
        #  input to the input layer's output before performing the forward evaluation of the network.
        #
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.

        for layer in self.layers:
            layer.forward()

        # check for softmax (if can access classifications, then softmax layer)
        try:
            self.output = self.layers[-1].classifications
        except:
            self.output = self.layers[-1].output

    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation backward through the 
        gradient tape.

        """

    def step(self):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 

        """
