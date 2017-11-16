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
            return self._forward(w_value, x_value, b_value) # _forward함수를 한 번 거치게 한다
        else:
            return np.dot(x_value, w_value) + b_value


    @staticmethod #static moethod이어야지만 적용된다) # numba가 적용이 안되면 에러가 나게 해준다
    @jit(nopython=True) # numba가 적용이 안되면 에러가 안 난다.
    def _forward(w_value, x_value, b_value):
        return np.dot(x_value, w_value) + b_value  # [Note] Matmul Order
    # jit이 적용된 연산이다

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