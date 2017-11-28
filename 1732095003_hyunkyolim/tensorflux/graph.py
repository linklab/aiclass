# -*- coding:utf-8 -*-

# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import networkx as nx
import numpy as np
from numba import jit, float64, uint8, void, cuda

class Graph(nx.Graph):
    """Represents a computational graph (a neural network)
    """
    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []
        super().__init__()


class Placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """
    def __init__(self, name=None):
        """Construct placeholder
        """
        self.output = None
        self.consumers = []
        self.name = name

    def __str__(self):
        return self.name


class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None, name=None):
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        self.value = initial_value
        self.output = None

        self.consumers = []
        self.name = name

    def __str__(self):
        return self.name


class Constant:
    """Represents a constant.
    """

    def __init__(self, value=None, name=None):
        """Construct Constant

        Args:
          value: this constant's value
        """
        self.value = value
        self.output = None

        self.consumers = []
        self.name = name

    def __str__(self):
        return self.name


class Operation:
    """Represents a graph node that performs a computation (forwaring operation).

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """

    def __init__(self, input_nodes=[], name=None, graph=None):
        """Construct Forwarding Operation
        """
        self.input_nodes = input_nodes
        self.output = None

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []
        self.name = name

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            if input_node is not None:
                input_node.consumers.append(self)
                graph.add_edge(input_node, self)

    def forward(self, is_train=True, is_numba=False):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass

    def backward(self):
        pass

    def __str__(self):
        return "O: " + self.name


class Add(Operation):
    def __init__(self, x, y, name=None):
        super().__init__([x, y], name)

    def forward(self, x_value, y_value, is_train=True, is_numba=False):
        if is_numba:
            return self._forward(x_value, y_value)
        else:
            return x_value + y_value

    def backward(self, d_in):
        d_x_value = d_in * 1
        d_y_value = d_in * 1
        return d_x_value, d_y_value

    @staticmethod
    @jit(nopython=True)
    def _forward(x_value, y_value):
        return x_value + y_value


class Mul(Operation):
    def __init__(self, x, y, name=None):
        self.x_value = None
        self.y_value = None
        super().__init__([x, y], name)

    def forward(self, x_value, y_value, is_train=True, is_numba=False):
        self.x_value = x_value
        self.y_value = y_value
        if is_numba:
            return self._forward(x_value, y_value)
        else:
            return x_value * y_value

    def backward(self, d_in):
        d_x_value = d_in * self.y_value
        d_y_value = d_in * self.x_value
        return d_x_value, d_y_value

    @staticmethod
    @jit(nopython=True)
    def _forward(x_value, y_value):
        return x_value * y_value

class Matmul(Operation):
    def __init__(self, x, y, name=None):
        self.x_value = None
        self.y_value = None
        super().__init__([x, y], name)

    def forward(self, x_value, y_value, is_train=True, is_numba=False):
        self.x_value = x_value
        self.y_value = y_value
        if is_numba:
            return self._forward(x_value, y_value)
        else:
            return x_value.dot(y_value)

    def backward(self, d_in):
        d_x_value = np.dot(self.y_value.T, d_in)
        d_y_value = np.dot(d_in, self.x_value.T)
        return d_x_value, d_y_value

    @staticmethod
    @jit(nopython=True)
    def _forward(x_value, y_value):
        return x_value.dot(y_value)
