# Paige Rosynek
# CS 3450 021
# Lab 07 - Completing From-Scratch Library
# 05.04.2023

from unittest import TestCase
import layers
import network
import numpy as np
import torch
import unittest


class TestNetwork(TestCase):
    """
    Tests Network class.
    """
    def setUp(self):
        # init network
        self.network = network.Network()

        # input layer - x
        x = layers.Input(output_shape=(2,1), train=False)
        x.set(torch.tensor([[10], [5]], dtype=torch.float64))

        # set network input
        self.network.set_input(x)


        #---------input -> hidden--------
        # W - weights
        W = layers.Input((3,2), train=True)
        W.set(torch.tensor([[1, 2],
                            [2, 5],
                            [7, 3]], dtype=torch.float64))

        # b - bias 
        b = layers.Input((3,1), train=True)
        b.set(torch.tensor([[4], [2], [1]], dtype=torch.float64))
        
        # hidden layer
        z = layers.Linear(x, W, b)
        h = layers.ReLU(z)                    # relu activation

        
        #---------hidden -> output--------
        # M - weights
        M = layers.Input((2,3), train=True)
        M.set(torch.tensor([[4, 1, 2],
                            [3, 6, 1]], dtype=torch.float64))

        # c - bias
        c = layers.Input((2,1), train=True)
        c.set(torch.tensor([[5], [1]], dtype=torch.float64))


        # output layer (no loss, regularization, or activation)
        o = layers.Linear(h, M, c)


        # add layers to network
        self.network.add(W)
        self.network.add(b)
        self.network.add(z)
        self.network.add(h)
        self.network.add(M)
        self.network.add(c)
        self.network.add(o)


    def test_forward(self):
        self.network.forward()
        np.testing.assert_allclose(self.network.output.numpy(), np.array([[320], [441]]))

    def test_backward(self):
        # NOT REQUIRED
        pass

    def test_step(self):
        # NOT REQUIRED
        pass


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


