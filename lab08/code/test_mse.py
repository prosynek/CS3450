# Paige Rosynek
# CS 3450 021
# Lab 07 - Completing From-Scratch Library
# 05.04.2023

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest


class TestMSE(TestCase):
    """
    Tests MSE Layer.
    """
    def setUp(self):
        self.o = layers.Input((2,1), train=True)
        self.o.set(torch.tensor([[10], [5]], dtype=torch.float64))

        self.y = layers.Input((2,1), train=True)
        self.y.set(torch.tensor([[11], [5]], dtype=torch.float64))

        self.mse_loss = layers.MSELoss(self.y, self.o)

    def test_forward(self):
        self.mse_loss.forward()
        np.testing.assert_allclose(self.mse_loss.output.numpy(), np.array([[0.5]]))

    def test_backward(self):
        dj_dout = torch.ones(self.mse_loss.output.shape, dtype=torch.float64)
        self.mse_loss.set_grad(dj_dout)
        self.mse_loss.backward()
        np.testing.assert_allclose(self.o.grad.detach().numpy(), np.array([[-1], [0]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


