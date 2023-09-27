# Paige Rosynek
# CS 3450 021
# Lab 06 - Implementing Forward Propagation
# 04.23.2023

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
        self.x.set(torch.tensor([[-10], [4], [0]], dtype=torch.float64))

        self.relu = layers.ReLU(self.x)

    def test_forward(self):
        self.relu.forward()
        np.testing.assert_allclose(self.relu.output.numpy(), np.array([[0], [4], [0]]))

    def test_backward(self):
        pass

    def test_step(self):
        pass


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


