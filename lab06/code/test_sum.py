# Paige Rosynek
# CS 3450 021
# Lab 06 - Implementing Forward Propagation
# 04.23.2023

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest


class TestSum(TestCase):
    """
    Tests Sum Layer.
    """
    def setUp(self):
        self.s1 = layers.Input((1,1), train=True)
        self.s1.set(torch.tensor([[50]], dtype=torch.float64))

        self.s2 = layers.Input((1,1), train=True)
        self.s2.set(torch.tensor([[5]], dtype=torch.float64))

        self.sum = layers.Sum(self.s1, self.s2)

    def test_forward(self):
        self.sum.forward()
        np.testing.assert_allclose(self.sum.output.numpy(), np.array([[55]]))

    def test_backward(self):
        pass

    def test_step(self):
        pass

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


