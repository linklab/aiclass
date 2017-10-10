# -*- coding:utf-8 -*-
import tensorflux.graph as tfg
import math
import numpy as np
import tensorflux.functions as tff
import random


class Affine(tfg.Operation):
    """Returns w * x + b.
    """
    def __init__(self, w, x, b, name=None, graph=None): # graph=None 추가됨
        """Construct Affine

        Args:
          x: Weight node, y: Input node, b: Bias node
        """
        self.inputs = None
        # graph.add_edge(w, self) # networks.py에서 networkx 그래프 그려주는 대신에 이렇게 함
        # graph.add_edge(x, self)
        # graph.add_edge(b, self)
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
    """Returns w * (x1, x2) + b.
    """
    def __init__(self, w, x1, x2, b, name=None, graph=None):
        """Construct Affine

        Args:
          x: Weight node, y: Input node, b: Bias node
        """
        self.inputs = None
        # graph.add_edge(w, self) #self:Affine객체에 노드 연결
        # graph.add_edge(x1, self)
        # graph.add_edge(x2, self)
        # graph.add_edge(b, self)
        super().__init__([w, x1, x2, b], name)

    def forward(self, w_value, x1_value, x2_value, b_value):
        """Compute the output of the add operation

        Args:
          x_value: Weight value, y_value: Input value, b_value: Bias value
        """
        self.inputs = [w_value, x1_value, x2_value, b_value]

        x_input = np.asarray([x1_value, x2_value]).T # scalar값을 묶어서 array화 시킨다. T를 붙인 이유는
        # (2,1):x_input과 (2,1):input_node이 계산되지 않으므로..
        # return np.matmul(x_value, w_value) + b_value # [Note] Matmul Order
        return x_input.dot(w_value) + b_value  # [Note] Matmul Order

    def backward(self):
        pass

    def __str__(self):
        return "Affine: " + self.name


class ReLU(tfg.Operation):
    def __init__(self, u, name=None):
        """Construct ReLU

        Args:
          u: affine node. RuLU 다음에 튀어나는 값이 u 노드가 됨.
        """
        self.inputs = None
        self.mask = None
        super().__init__([u], name)

    def forward(self, u_value):
        self.inputs = [u_value]

        if type(u_value) == np.ndarray:
            self.mask = (u_value <= 0)
            # [-1.0, 2.0, -2.0] --> [True, False, True]
            out = u_value.copy()
            out[self.mask] = 0
        else:
            if u_value <= 0:
                out = 0.0
            else:
                out = u_value
        return out # scalar, array 모두 커버 중

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
        self.out = tff.sigmoid(u_value) # function.py
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
          forward_final_output: feedforward의 결과물
        """
        self.inputs = None
        super().__init__([forward_final_output, target], name)

    def forward(self, forward_final_output_value, target_value):
        self.inputs = [forward_final_output_value, target_value]
        return tff.squared_error(forward_final_output_value, target_value) # function.py

    def backward(self, din):
        pass

    def __str__(self):
        return "SquaredError:" + self.name