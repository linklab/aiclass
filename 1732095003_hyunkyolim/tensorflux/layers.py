# -*- coding:utf-8 -*-
import tensorflux.graph as tfg
import sys
import numpy as np
import tensorflux.functions as tff
from numba import jit, float64, uint8, void, cuda


class Affine(tfg.Operation):
    """Returns w * x + b.
    """
    def __init__(self, w, x, b, name=None, graph=None):
        """Construct Affine

        Args:
          x: Weight node, y: Input node, b: Bias node
        """
        self.w_value = None
        self.x_value = None
        self.b_value = None
        self.dw = None
        self.db = None
        super().__init__([w, x, b], name, graph)

    def forward(self, w_value, x_value, b_value, is_numba):
        """Compute the output of the affine operation

        Args:
          x_value: Weight value, y_value: Input value, b_value: Bias value
        """
        self.w_value = w_value
        self.x_value = x_value
        self.b_value = b_value
        if is_numba:
            return self._forward(w_value, x_value, b_value)
        else:
            return np.dot(x_value, w_value) + b_value


    @staticmethod
    @jit(nopython=True)
    def _forward(w_value, x_value, b_value):
        return np.dot(x_value, w_value) + b_value  # [Note] Matmul Order

    def backward(self, din, is_numba):
        if is_numba:
            dx, dw, db = self._backward(din, self.w_value, self.x_value)
            self.dw = dw
            self.db = db
        else:
            dx = np.dot(din, self.w_value.T)
            self.dw = np.dot(self.x_value.T, din)
            self.db = np.sum(din, axis=0)

        # if type(din) == np.ndarray and self.x_value.size == 2 and din.size == 1:
        #     self.dw = np.dot(self.x_value.T, np.asscalar(din))
        # else:
        #     self.dw = np.dot(self.x_value.T, din)
        #
        # dx = np.dot(din, self.w_value.T)
        #
        # if din.ndim > 1:
        #     self.db = np.sum(din, axis=0)
        # else:
        #     self.db = din
        #
        # self.dw = np.reshape(self.dw, self.w_value.shape)
        # dx = np.reshape(dx, self.x_value.shape)
        # self.db = np.reshape(self.db, self.b_value.shape)

        return dx

    @staticmethod
    @jit(nopython=True)
    def _backward(din, w_value, x_value):
        dx = np.dot(din, w_value.T)
        dw = np.dot(x_value.T, din)
        db = np.sum(din, axis=0)
        return dx, dw, db

    def __str__(self):
        return "Affine: " + self.name


class ReLU(tfg.Operation):
    def __init__(self, u, name=None, graph=None):
        """Construct ReLU

        Args:
          u: affine node
        """
        self.u_value = None

        self.mask = None
        super().__init__([u], name, graph)

    def forward(self, u_value, is_numba):
        self.u_value = u_value
        self.mask = (u_value <= 0.0)
        out = u_value.copy()
        out[self.mask] = 0.0
        return out

    def backward(self, din, is_numba):
        dx = din.copy()
        dx[self.mask] = 0.0
        return dx

    # def forward(self, u_value):
    #     self.u_value = u_value
    #
    #     if type(u_value) == np.ndarray:
    #         self.mask = (u_value <= 0.0)
    #         out = u_value.copy()
    #         out[self.mask] = 0.0
    #     else:
    #         if u_value <= 0:
    #             out = 0.0
    #         else:
    #             out = u_value
    #     return out

    # def backward(self, din):
    #     if type(din) == np.ndarray:
    #         dx = din.copy()
    #         dx[self.mask] = 0.0
    #     else:
    #         if self.u_value <= 0.0:
    #             dx = 0.0
    #         else:
    #             dx = din
    #     return dx

    def __str__(self):
        return "ReLU: " + self.name


class Sigmoid(tfg.Operation):
    def __init__(self, u, name=None, graph=None):
        """Construct ReLU

        Args:
          u: affine node
        """
        self.u_value = None

        self.out = None
        super().__init__([u], name, graph)

    def forward(self, u_value, is_numba):
        self.u_value = u_value
        self.out = tff.sigmoid(u_value, is_numba=is_numba)
        return self.out

    def backward(self, din, is_numba):
        if is_numba:
            return self._backward(din, self.out)
        else:
            dx = din * self.out * (1.0 - self.out)
            return dx

    @staticmethod
    @jit(nopython=True)
    def _backward(din, out):
        dx = din * out * (1.0 - out)
        return dx

    def __str__(self):
        return "Sigmoid: " + self.name


class SquaredError(tfg.Operation):
    def __init__(self, forward_final_output, target, name=None, graph=None):
        """Construct SquaredError

        Args:
          output: output node
        """
        self.forward_final_output_value = None  # forward_final_output_value
        self.target_value = None  # target_value
        super().__init__([forward_final_output, target], name, graph)

    def forward(self, forward_final_output_value, target_value, is_numba):
        self.forward_final_output_value = forward_final_output_value
        self.target_value = target_value
        return tff.squared_error(forward_final_output_value, target_value, is_numba=is_numba)

    def backward(self, din, is_numba):
        if is_numba:
            return self._backward(din, self.forward_final_output_value, self.target_value)
        else:
            dx = (self.forward_final_output_value - self.target_value) * din
            return dx

    @staticmethod
    @jit(nopython=True)
    def _backward(din, forward_final_output_value, target_value):
        dx = (forward_final_output_value - target_value) * din
        return dx

    def __str__(self):
        return "SquaredError: " + self.name


class SoftmaxWithCrossEntropyLoss(tfg.Operation):
    def __init__(self, forward_final_output, target, name=None, graph=None):
        """Construct SquaredError

        Args:
          output: output node
        """
        self.target_value = None  # target_value
        self.y = None
        super().__init__([forward_final_output, target], name, graph)

    def forward(self, forward_final_output_value, target_value, is_numba):
        self.target_value = target_value
        self.y = tff.softmax(forward_final_output_value, is_numba=is_numba)
        loss = tff.cross_entropy_error(self.y, self.target_value, is_numba=is_numba)
        return loss

    def backward(self, din, is_numba):
        batch_size = self.target_value.shape[0]
        if is_numba:
            return self._backward(din, self.y, self.target_value, batch_size)
        else:
            dx = (self.y - self.target_value) / float(batch_size)
            return dx

    @staticmethod
    @jit(nopython=True)
    def _backward(din, y, target_value, batch_size):
        dx = (y - target_value) / float(batch_size)
        return dx

    def __str__(self):
        return "SoftmaxWithCrossEntropyLoss: " + self.name


class Affine2(tfg.Operation):
    """Returns w * (x1, x2) + b.
    """
    def __init__(self, w, x1, x2, b, name=None, graph=None):
        """Construct Affine

        Args:
          x: Weight node, y: Input node, b: Bias node
        """
        self.w_value = None
        self.x_value = None
        self.b_value = None

        self.dw = None
        self.db = None
        super().__init__([w, x1, x2, b], name, graph)

    def forward(self, w_value, x1_value, x2_value, b_value):
        """Compute the output of the add operation

        Args:
          x_value: Weight value, y_value: Input value, b_value: Bias value
        """

        self.w_value = w_value
        self.x_value = np.asarray([x1_value, x2_value]).T
        self.b_value = b_value

        # return np.matmul(x_value, w_value) + b_value # [Note] Matmul Order
        return np.dot(self.x_value, self.w_value) + self.b_value  # [Note] Matmul Order

    def backward(self, din):
        self.dw = np.dot(self.x_value, np.asscalar(din))
        dx = np.dot(din, self.w_value.T)
        self.db = din

        self.dw = np.reshape(self.dw, self.w_value.shape)
        dx = np.reshape(dx, self.x_value.shape)
        self.db = np.reshape(self.db, self.b_value.shape)

        return dx

    def __str__(self):
        return "Affine2: " + self.name


class Convolution(tfg.Operation):
    def __init__(self, w, x, b, pad, stride, name=None, graph=None):
        """Construct Convolution

        Args:
          x: Filter node, y: Input node, b: Bias node
        """
        self.w_value = None
        self.x_value = None
        self.b_value = None

        self.pad = pad
        self.stride = stride

        self.col = None
        self.col_w = None
        self.dw = None
        self.db = None
        super().__init__([w, x, b], name, graph)

    def forward(self, w_value, x_value, b_value, is_numba):
        self.w_value = w_value
        self.x_value = x_value
        self.b_value = b_value

        if is_numba:
            x_value, col, col_w, out = self._forward(w_value, x_value, b_value, self.pad, self.stride)
            self.x_value = x_value
            self.col = col
            self.col_w = col_w
            return out
        else:
            FN, C, FH, FW = self.w_value.shape
            N, C, H, W = x_value.shape
            out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
            out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

            col = tff.im2col(x_value, FH, FW, self.stride, self.pad)
            col_w = self.w_value.reshape(FN, -1).T

            out = np.dot(col, col_w) + self.b_value
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

            self.x_value = x_value
            self.col = col
            self.col_w = col_w

            return out

    @staticmethod
    @jit(nopython=True)
    def _forward(w_value, x_value, b_value, pad, stride):
        FN, C, FH, FW = w_value.shape
        N, C, H, W = x_value.shape
        out_h = 1 + int((H + 2 * pad - FH) / stride)
        out_w = 1 + int((W + 2 * pad - FW) / stride)

        col = tff.im2col(x_value, FH, FW, stride, pad)
        col_w = w_value.reshape(FN, -1).T

        out = np.dot(col, col_w) + b_value
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return x_value, col, col_w, out

    def backward(self, din, is_numba):
        if is_numba:
            dw, dx, db = self._backward(din, self.w_value, self.x_value, self.col, self.col_w, self.pad, self.stride)
            self.dw = dw
            self.db = db
            return dx
        else:
            FN, C, FH, FW = self.w_value.shape
            din = din.transpose(0, 2, 3, 1).reshape(-1, FN)

            self.db = np.sum(din, axis=0)
            self.dw = np.dot(self.col.T, din)
            self.dw = self.dw.transpose(1, 0).reshape(FN, C, FH, FW)

            dcol = np.dot(din, self.col_w.T)
            dx = tff.col2im(dcol, self.x_value.shape, FH, FW, self.stride, self.pad)
            return dx

    @staticmethod
    @jit(nopython=True)
    def _backward(din, w_value, x_value, col, col_w, pad, stride):
        FN, C, FH, FW = w_value.shape
        din = din.transpose(0, 2, 3, 1).reshape(-1, FN)

        db = np.sum(din, axis=0)
        dw = np.dot(col.T, din)
        dw = dw.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(din, col_w.T)
        dx = tff.col2im(dcol, x_value.shape, FH, FW, stride, pad)

        return dw, dx, db


class Pooling(tfg.Operation):
    def __init__(self, w, x, pad, stride, name=None, graph=None):
        """Construct Pooling

        Args:
          x: Filter node, y: Input node, b: Bias node
        """
        self.w_value = None
        self.x_value = None
        self.stride = stride
        self.pad = pad

        self.arg_max
        super().__init__([w, x], name, graph)

    def forward(self, w_value, x_value, is_numba):
        self.w_value = w_value
        self.x_value = x_value

        if is_numba:
            arg_max, out = self._forward(self.w_value, self.x_value, self.pad, self.stride)
            self.arg_max = arg_max
            return out
        else:
            FN, C, FH, FW = self.w_value.shape
            N, C, H, W = self.x_value.shape
            out_h = int(1 + (H - FH) / self.stride)
            out_w = int(1 + (W - FW) / self.stride)

            col = tff.im2col(self.x_value, FH, FW, self.stride, self.pad)
            col = col.reshape(-1, FH * FW)

            arg_max = np.argmax(col, axis=1)
            out = np.max(col, axis=1)
            out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

            self.arg_max = arg_max
            return out

    @staticmethod
    @jit(nopython=True)
    def _forward(w_value, x_value, pad, stride):
        FN, C, FH, FW = w_value.shape
        N, C, H, W = x_value.shape
        out_h = int(1 + (H - FH) / stride)
        out_w = int(1 + (W - FW) / stride)

        col = tff.im2col(x_value, FH, FW, stride, pad)
        col = col.reshape(-1, FH * FW)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return arg_max, out

    def backward(self, din, is_numba):
        if is_numba:
            self._backward(din, self.w_value, self.x_value, self.pad, self.stride, self.arg_max)
        else:
            FN, C, FH, FW = self.w_value.shape
            din = din.transpose(0, 2, 3, 1)

            pool_size = FH * FW
            dmax = np.zeros((din.size, pool_size))
            dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = din.flatten()
            dmax = dmax.reshape(din.shape + (pool_size,))

            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            dx = tff.col2im(dcol, self.x_value.shape, FH, FW, self.stride, self.pad)

            return dx

    @staticmethod
    @jit(nopython=True)
    def _backward(din, w_value, x_value, pad, stride, arg_max):
        FN, C, FH, FW = w_value.shape
        din = din.transpose(0, 2, 3, 1)

        pool_size = FH * FW
        dmax = np.zeros((din.size, pool_size))
        dmax[np.arange(arg_max.size), arg_max.flatten()] = din.flatten()
        dmax = dmax.reshape(din.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = tff.col2im(dcol, x_value.shape, FH, FW, stride, pad)

        return dx

class Reshape(tfg.Operation):
    def __init__(self, u, p_shape, n_shape, name=None, graph=None):
        self.u_value = None
        self.p_shape = p_shape
        self.n_shape = n_shape
        self.batch_size = None
        super().__init__([u], name, graph)

    def forward(self, u_value, is_numba=False):
        self.batch_size = u_value.shape[0]
        out = np.reshape(u_value, (self.batch_size, self.n_shape))
        return out

    def backward(self, din, is_numba=False):
        dx = np.reshape(din, (self.batch_size, *self.p_shape))
        return dx