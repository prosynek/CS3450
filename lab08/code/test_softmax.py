# Paige Rosynek
# CS 3450 021
# Lab 07 - Completing From-Scratch Library
# 05.04.2023

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest


class TestSoftmax(TestCase):
    """
    Tests Softmax Layer.
    """
    def setUp(self):
        self.v = layers.Input((2,1), train=True)
        self.v.set(torch.tensor([[10], [5]], dtype=torch.float64))

        self.y_true = layers.Input((2,1), train=False)
        self.y_true.set(torch.tensor([[1], [0]], dtype=torch.float64))

        self.softmax = layers.Softmax(self.v, self.y_true)

    def test_forward(self):
        self.softmax.forward()
        np.testing.assert_allclose(self.softmax.classifications.numpy(), np.array([[0.99330715], [0.00669285]]), rtol=0.001)      # probabilities
        np.testing.assert_allclose(self.softmax.output.numpy(), np.array([0.003358]), rtol=0.001)                                 # cross-entropy loss

    def test_backward(self):
        self.softmax.forward()
        dj_dout = torch.ones(1)     # dJ/dL = 1
        self.softmax.set_grad(dj_dout)
        self.softmax.backward()
        np.testing.assert_allclose(self.v.grad.detach().numpy(), np.array([[-0.007], [0.007]]), atol=0.001)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


