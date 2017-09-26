# -*- coding:utf-8 -*-
import tensorflux.graph as tfg
import math
import numpy as np
import tensorflux.functions as tff
import random


class Affine(tfg.Operation):
    """Returns w * x + b.
    """
    def __init__(self, w, x, b, name=None):
        """Construct Affine

        Args:
          x: Weight node, y: Input node, b: Bias node
        """
        self.inputs = None
        super().__init__([w, x, b], name)

    def forward(self, w_value, x_value, b_value):
        """Compute the output of the add operation

        Args:
          x_value: Weight value, y_value: Input value, b_value: Bias value
        """
        self.inputs = [w_value, x_value, b_value]
        # return np.matmul(x_value, w_value) + b_value # [Note] Matmax Order
        return x_value.dot(w_value) + b_value  # [Note] Matmax Order

    def backward(self):
        pass

    def __str__(self):
        return "Affine: " + self.name


class Affine2(tfg.Operation):
    def __init__(self, w, x1, x2, b, name=None):
        self.inputs = None
        super().__init__([w,x1,x2,b], name)

    def forward(self, w_value, x1_value, x2_value, b_value):
        self.inputs = [w_value,x1_value,x2_value,b_value]
        x_input = np.asarray([x1_value, x2_value,]).T

        # return (w_value[0]*x1_value+w_value[1]*x2_value)+b_value
        return x_input.dot(w_value) + b_value

    def backward(self, w_value, x1_value, x2_value, b_value,error):
        pass

    def __str__(self):
        return "Affine2: " + self.name


class ReLU(tfg.Operation):

    def __init__(self, u, name=None):
        """Construct ReLU

        Args:
          u: affine node
        """
        self.inputs = None
        self.mask = None
        self.out = None
        super().__init__([u], name)

    def forward(self, u_value):
        self.inputs = [u_value]

        if type(u_value) == np.ndarray:
            self.mask = (u_value <= 0)  # 각 엘리먼트를 계산해줌
            out = u_value.copy()        # 밸류값
            out[self.mask] = 0          # 마스크가 트루면 0 대입
        else:
            if u_value <= 0:
                out = 0.0
            else:
                out = u_value
        self.out = out
        return out

    def backward(self, din):
        pass

    def __str__(self):
        return "ReLU: " + self.name

    def value(self):
        return self.out


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
        super().__init__([forward_final_output, target], name)

    def forward(self, forward_final_output_value, target_value):
        self.inputs = [forward_final_output_value, target_value]
        return tff.squared_error(forward_final_output_value, target_value)

    def backward(self, din):
        pass

    def __str__(self):
        return "SquaredError:" + self.name