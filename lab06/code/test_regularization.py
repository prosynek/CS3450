# Paige Rosynek
# CS 3450 021
# Lab 06 - Implementing Forward Propagation
# 04.23.2023

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest


class TestRegularization(TestCase):
    """
    Tests Regularization Layer.
    """
    def setUp(self):
        # weights - W
        self.W = layers.Input((2,3), train=True)
        self.W.set(torch.tensor([[1, 5, 7],
                                 [3, 2, 4]], dtype=torch.float64))

        # lambda
        self.coef = 0.1

        self.regularize = layers.Regularization(self.coef, self.W)

    def test_forward(self):
        self.regularize.forward()
        np.testing.assert_allclose(self.regularize.output.numpy(), np.array([[5.2]]))

    def test_backward(self):
        pass

    def test_step(self):
        pass


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


