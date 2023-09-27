# Paige Rosynek
# CS 3450 021
# Lab 07 - Completing From-Scratch Library
# 05.04.2023

import torch


class Network:
    def __init__(self):
        """
        Initializes Network instance.
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
        self.layers.append(input_layer)     # set input layer as first layer

    def forward(self):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward.

        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
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
        for layer in self.layers:
            layer.clear_grad()

        # set the last gradient to start ( dJ/dJ = 1 or dJ/dL = 1)
        self.layers[-1].set_grad(torch.ones(self.layers[-1].output.shape, dtype=torch.float64))

        for layer in list(reversed(self.layers)):
            layer.backward()

    def step(self, step_size):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 
        
        :param step_size: learning rate for SGD
        """
        for layer in self.layers:
            layer.step(step_size)
