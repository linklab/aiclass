# -*- coding:utf-8 -*-
import tensorflux_HW1.graph as tfg
import math
import numpy as np
import tensorflux_HW1.functions as tff
import random

#추가 affine
class AffineForThreeNeuron(tfg.Operation):
    """Returns w * x + b.
    """
    def __init__(self, w, z1, z2, b, name=None):
        """Construct Affine

        Args:
          w: Weight node, x: Input node, b: Bias node
        """
        self.inputs = None
        super().__init__([w, z1, z2, b], name)

    def forward(self, w_value, z1, z2, b_value):

        #z1 * weight + bias
        #print(z1)
        #print(z2)

        x_input = np.asarray([z1, z2]).T
        #value = [z1, z2]
        #print(value)
    #        val_z1 = z1.dot(w_value) + b_value
     #   val_z2 = z2.dot(w_value) + b_value
        #new_list = [float(val_z1), float(val_z2)]
        #new_value = np.array(new_list)
        #print(new_value)
        #set_value = val_z1 + val_z2
        #print(set_value)
        #print("x : " + str(z1))
        #print("output : " + str(z2))
     #   new_value = np.array([z1, z2])
        #print(new_value) #[[ 0.1  0.1]]
        # return np.matmul(x_value, w_value) + b_value # [Note] Matmax Order
        return x_input.dot(w_value) + b_value

    def backward(self):
        pass

    def __str__(self):
        return "AffineForThreeNeuron: " + self.name


class Affine(tfg.Operation):
    """Returns w * x + b.
    """
    def __init__(self, w, x, b, name=None):
        """Construct Affine

        Args:
          w: Weight node, x: Input node, b: Bias node
        """
        self.inputs = None
        super().__init__([w, x, b], name)

    def forward(self, w_value, x_value, b_value):
        """Compute the output of the add operation

        Args:
          x_value: Weight value, y_value: Input value, b_value: Bias value
        """
        self.inputs = [w_value, x_value, b_value]

        #print(x_value)
        #print(x_value.dot(w_value))  #[0.00   0.2231]
        #print(w_value)
        #[[0.10085925  0.10085925]
        #[0.10085925  0.10085925]]
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
        #print(out)
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