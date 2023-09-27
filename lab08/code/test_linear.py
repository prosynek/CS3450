# Paige Rosynek
# CS 3450 021
# Lab 07 - Completing From-Scratch Library
# 05.04.2023

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
        self.x = layers.Input(output_shape=(3,1), train=True)
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
        dj_dout = torch.ones(self.linear.output.shape, dtype=torch.float64)  # assumes dj/dout = 1s
        self.linear.set_grad(dj_dout)

        self.linear.backward()
        np.testing.assert_allclose(self.x.grad.detach().numpy(), np.array([[4], [7], [11]]))    # not necessary - dont usually train input
        np.testing.assert_allclose(self.W.grad.detach().numpy(), np.array([[2, 10, 5],
                                                                           [2, 10, 5]]))
        np.testing.assert_allclose(self.b.grad.detach().numpy(), dj_dout.numpy())

    def test_step(self):
        # assuming same gradients as backward() test
        dj_dout = torch.ones(self.linear.output.shape, dtype=torch.float64)  # assumes dj/dout = 1s
        self.linear.set_grad(dj_dout)
        self.b.set_grad(dj_dout)
        self.W.set_grad(torch.tensor([[2, 10, 5],
                                      [2, 10, 5]],  dtype=torch.float64))
        
        step_size = 0.1
        self.W.step(step_size)
        self.b.step(step_size)

        np.testing.assert_allclose(self.W.output.detach().numpy(), np.array([[0.8, 4.0, 6.5],
                                                                             [2.8, 1.0, 3.5]]))
        np.testing.assert_allclose(self.b.output.detach().numpy(), np.array([[3.9], [7.9]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


