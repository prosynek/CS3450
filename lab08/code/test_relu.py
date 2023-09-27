# Paige Rosynek
# CS 3450 021
# Lab 07 - Completing From-Scratch Library
# 05.04.2023

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest


class TestReLU(TestCase):
    """
    Tests ReLU Layer.
    """
    def setUp(self):
        # input - x
        self.x = layers.Input(output_shape=(3,1), train=True)
        self.x.set(torch.tensor([[-4], [10], [5]], dtype=torch.float64))

        self.relu = layers.ReLU(self.x)

    def test_forward(self):
        self.relu.forward()
        np.testing.assert_allclose(self.relu.output.detach().numpy(), np.array([[0], [10], [5]]))

    def test_backward(self):
        self.relu.set_grad(torch.ones(self.relu.output.shape))  # assumes dJ/dout = 1s
        self.relu.backward()
        np.testing.assert_allclose(self.x.grad.detach().numpy(), np.array([[0], [1], [1]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


