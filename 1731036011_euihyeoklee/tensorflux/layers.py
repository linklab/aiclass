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

class ReLU(tfg.Operation):
    def __init__(self, u, name=None):
        """Construct ReLU

        Args:
          u: affine node
        """
        self.inputs = None
        self.mask = None
        super().__init__([u], name)

    def forward(self, u_value):
        self.inputs = [u_value]

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
        super().__init__([forward_final_output, target], name)

    def forward(self, forward_final_output_value, target_value):
        self.inputs = [forward_final_output_value, target_value]
        return tff.squared_error(forward_final_output_value, target_value)

    def backward(self, din):
        pass

    def __str__(self):
        return "SquaredError:" + self.name


class AffineGather(tfg.Operation):
    """Returns w * x + b.
     """

    def __init__(self, w, x1, x2, b, name=None):
        """Construct Affine

        Args:
        x: Weight node, y: Input node, b: Bias node
        """
        self.inputs = None
        super().__init__([w, x1, x2, b], name)

    def forward(self, w_value, x1, x2, b_value):
        """Compute the output of the add operation

        Args:
        x_value: Weight value, y_value: Input value, b_value: Bias value
        """
        # print("--x1" + str(x1))
        # print("--x2" + str(x2))
        # self.inputs = [w_value, x1, x2, b_value]
        # print(self.inputs[0])

        # output1 = np.dot(x1, w_value) + b_value
        # output2 = np.dot(x2, w_value) + b_value

        # print(x1[0])
        output = np.array([[x1[0], x2[0]]])


        # output2 = x2.dot(w_value) + b_value
        # print(output1)
        # output_list = [x1.dot(w_value) + b_value, x2.dot(w_value) + b_value]
        # output = np.array(output_list)
        #
        return output.dot(w_value) + b_value

    def backward(self):
        pass

    def __str__(self):
        return "AffineGather: " + self.name



