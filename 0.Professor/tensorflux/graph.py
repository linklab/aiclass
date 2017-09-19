# -*- coding:utf-8 -*-

# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import networkx as nx


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

    def get_shape(self):
        return self.value.shape

    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.name


class Operation:
    """Represents a graph node that performs a computation (forwaring operation).

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """

    def __init__(self, input_nodes=[], name=None):
        """Construct Forwarding Operation
        """
        self.input_nodes = input_nodes
        self.output = None

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []
        self.name = name

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

    def forward(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass

    def __str__(self):
        return "O: " + self.name


class Add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y, name=None):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        self.inputs = None
        super().__init__([x, y], name)

    def forward(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        self.inputs = [x_value, y_value]
        return x_value + y_value


class Mul(Operation):
    """Returns x * y.
    """

    def __init__(self, x, y, name=None):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        self.inputs = None
        super().__init__([x, y], name)

    def forward(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        self.inputs = [x_value, y_value]
        return x_value * y_value


class Matmul(Operation):
    """Multiplies matrix x by matrix y, producing x * y.
    """

    def __init__(self, x, y, name=None):
        """Construct matmul

        Args:
          x: First matrix
          y: Second matrix
        """
        self.inputs = None
        super().__init__([x, y], name)

    def forward(self, x_value, y_value):
        """Compute the output of the matmul operation

        Args:
          x_value: First matrix value
          y_value: Second matrix value
        """
        self.inputs = [x_value, y_value]
        return x_value.dot(y_value)