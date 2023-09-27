# Paige Rosynek
# CS 3450 021
# Lab 07 - Completing From-Scratch Library
# 05.04.2023

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
        dj_dout = torch.ones(self.regularize.output.shape, dtype=torch.float64)
        self.regularize.set_grad(dj_dout)
        self.regularize.backward()
        np.testing.assert_allclose(self.W.grad.detach().numpy(), np.array([[0.1, 0.5, 0.7],
                                                                           [0.3, 0.2, 0.4]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


