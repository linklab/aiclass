# -*- coding:utf-8 -*-
import tensorflux.graph as tfg
import math
import numpy as np
import tensorflux.functions as tff
import random


class AffineFL(tfg.Operation):
    """Returns w * x + b.
    """
    # def __init__(self, w, x, b, name=None): # w : 가중치 노드, x : 값 노드, b : bias 노드
    #     """Construct Affine
    #
    #     Args:
    #       x: Weight node, y: Input node, b: Bias node
    #     """
    #     self.inputs = None
    #     super().__init__([w, x, b], name)

    def __init__(self, w, x, b, name=None):
        self.inputs = None
        super().__init__([w, x, b], name)


    def forward(self, w, x, b):
        """Compute the output of the add operation

        Args:
          x_value: Weight value, y_value: Input value, b_value: Bias value
        """
        self.inputs = [w, x, b]
        # return np.matmul(x_value, w_value) + b_value # [Note] Matmax Order
        return x.dot(w) + b  # [Note] Matmax Order
    def backward(self):
        pass

    def __str__(self):
        return "Affine: " + self.name

class AffineSL(tfg.Operation):
    """Returns w * x + b.
    """
    def __init__(self, w, x1, x2, b, name=None):
        """Construct Affine
        Args:
          x: Weight node, y: Input node, b: Bias node
        """
        self.inputs = None
        super().__init__([w, x1, x2, b], name)

    def forward(self, w, x1, x2, b):

        val_x1 = np.dot(x1, w) + b
        val_x2 = np.dot(x2, w) + b

        val_Result = np.greater(val_x1, val_x2)
        if val_Result == True:
            return val_x1
        elif val_Result == False:
            return val_x2

    def backward(self):
        pass

    def __str__(self):
        return "Affine2: " + self.name

class ReLU(tfg.Operation):
    def __init__(self, u, name=None):   # u : Affine 노드
        """Construct ReLU

        Args:
          u: affine node
        """
        self.inputs = None
        self.mask = None
        super().__init__([u], name)

    def forward(self, u_value):
        if type(u_value) == np.ndarray:
            self.mask = (u_value <= 0)
            out = u_value.copy()
            out[self.mask] = 0
        else:
            if u_value <= 0:
                out = 0.0
            else:
                out = u_value
        return out

    def backward(self, din):
        pass

    def __str__(self):
        return "ReLU: " + self.name

class Sigmoid(tfg.Operation):
    def __init__(self, u, name=None):
        """Construct ReLU

        Args:
          u: affine node
        """
        self.inputs = None
        self.out = None
        super().__init__([u], name)

    def forward(self, u_value):
        self.inputs = [u_value]
        self.out = tff.sigmoid(u_value)
        return self.out

    def backward(self, din):
        pass

    def __str__(self):
        return "Sigmoid: " + self.name


class SquaredError(tfg.Operation):
    def __init__(self, forward_final_output, target, name=None):
        """Construct SquaredError

        Args:
          output: output node
        """
        self.inputs = None
        super().__init__([forward_final_output, target], name)  # 노드, 노드

    def forward(self, forward_final_output_value, target_value):    # 값, 값
        self.inputs = [forward_final_output_value, target_value]
        return tff.squared_error(forward_final_output_value, target_value)

    def backward(self, din):
        pass

    def __str__(self):
        return "SquaredError:" + self.name