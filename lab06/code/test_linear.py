# Paige Rosynek
# CS 3450 021
# Lab 06 - Implementing Forward Propagation
# 04.23.2023

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest


class TestLinear(TestCase):
    """
    Tests Linear Layer.
    """
    def setUp(self):
        # input - x
        self.x = layers.Input(output_shape=(3,1), train=False)
        self.x.set(torch.tensor([[2], [10], [5]], dtype=torch.float64))

        # bias - b
        self.b = layers.Input((2,1), train=True)
        self.b.set(torch.tensor([[4], [8]], dtype=torch.float64))

        # weights - W
        self.W = layers.Input((2,3), train=True)
        self.W.set(torch.tensor([[1, 5, 7],
                                 [3, 2, 4]], dtype=torch.float64))

        self.linear = layers.Linear(self.x, self.W, self.b)

    def test_forward(self):
        self.linear.forward()
        np.testing.assert_allclose(self.linear.output.numpy(), np.array([[91], [54]]))

    def test_backward(self):
        pass

    def test_step(self):
        pass


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


