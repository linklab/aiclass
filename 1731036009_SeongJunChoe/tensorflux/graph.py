# # -*- coding:utf-8 -*-
#
# # Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
# import networkx as nx
#
# _default_graph = None
#
#
# class Graph(nx.Graph):
#     """Represents a computational graph (a neural network)
#     """
#
#     def __init__(self):
#         """Construct Graph"""
#         self.operations = []
#         self.placeholders = []
#         self.variables = []
#         super().__init__()
#
#     # def initialize(self):
#     #    global _default_graph
#     #    _default_graph = self
#     # default 그래프를 수행하면, 서로 다른 그래프를 생성할 수 없음 (오로지 하나의 그래프가 생성되는 문제가 발생)
#
# class Placeholder:
#     """Represents a placeholder node that has to be provided with a value
#        when computing the output of a computational graph
#     """
#     def __init__(self, name=None):
#         """Construct placeholder
#         """
#         self.consumers = []
#         self.name = name
#
#     def __str__(self):
#         return "P: " + self.name
#
#
# class Variable:
#     """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
#     """
#
#     def __init__(self, initial_value=None, name=None):
#         """Construct Variable
#
#         Args:
#           initial_value: The initial value of this variable
#         """
#         self.value = initial_value
#         self.output = None
#
#         self.consumers = []
#         self.name = name
#
#     # 없어도 무관하나 있어도 무관? (객체지향 코딩 방법)
#     # 지워도 문제 없음
#     def get_shape(self):
#         return self.value.shape
#
#     def set_value(self, value):
#         self.value = value
#
#     def __str__(self):
#         return "V: " + self.name
#
#
# class Operation:
#     """Represents a graph node that performs a computation (forwaring operation).
#
#     An `Operation` is a node in a `Graph` that takes zero or
#     more objects as input, and produces zero or more objects
#     as output.
#     """
#
#     def __init__(self, input_nodes=[], name=None):
#         """Construct Forwarding Operation
#         """
#         self.input_nodes = input_nodes
#
#         # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
#         self.consumers = []
#         self.name = name
#
#         # Append this operation to the list of consumers of all input nodes
#         for input_node in input_nodes:
#             input_node.consumers.append(self)
#
#     def forward(self):
#         """Computes the output of this operation.
#         "" Must be implemented by the particular operation.
#         """
#         pass
#
#     def extended(self, data):
#         self.input_nodes.append(data)
#
#     def __str__(self):
#         return "O: " + self.name
#
#
# class Add(Operation):
#     """Returns x + y element-wise.
#     """
#
#     def __init__(self, x, y, name=None):
#         """Construct add
#
#         Args:
#           x: First summand node
#           y: Second summand node
#         """
#         self.inputs = None  # 수치 값이 들어감
#         super().__init__([x, y], name)
#
#     def forward(self, x_value, y_value):  # 부모 클래스를 오버라이드?
#         """Compute the output of the add operation
#
#         Args:
#           x_value: First summand value
#           y_value: Second summand value
#         """
#         self.inputs = [x_value, y_value]
#         return x_value + y_value  ##numpy의 배열은 각각 원소끼리 더하기를 수행하므로 가능
#
#
# class Mul(Operation):
#     """Returns x * y.
#     """
#
#     def __init__(self, x, y, name=None):
#         """Construct add
#
#         Args:
#           x: First summand node
#           y: Second summand node
#         """
#         self.inputs = None
#         super().__init__([x, y], name)
#
#     def forward(self, x_value, y_value):
#         """Compute the output of the add operation
#
#         Args:
#           x_value: First summand value
#           y_value: Second summand value
#         """
#         self.inputs = [x_value, y_value]
#         return x_value * y_value
#
#
# class Matmul(Operation):
#     """Multiplies matrix x by matrix y, producing x * y.
#     """
#
#     def __init__(self, x, y, name=None):
#         """Construct matmul
#
#         Args:
#           x: First matrix
#           y: Second matrix
#         """
#         self.inputs = None
#         super().__init__([x, y], name)
#
#     def forward(self, x_value, y_value):
#         """Compute the output of the matmul operation
#
#         Args:
#           x_value: First matrix value
#           y_value: Second matrix value
#         """
#         self.inputs = [x_value, y_value]
#         return x_value.dot(y_value)
# -*- coding:utf-8 -*-

# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import networkx as nx
import numpy as np


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
        # for input_node in input_nodes:
        #     input_node.consumers.append(self)
        #     graph.add_edge(input_node, self)
        #     graph.add_edge(input_node, self)
        #     graph.add_edge(input_node, self)

    def forward(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass

    def backward(self):
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

    def backward(self, d_in):
        d_x_value = d_in * 1
        d_y_value = d_in * 1
        return d_x_value, d_y_value


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

    def backward(self, d_in):
        d_x_value = d_in * self.inputs[1]
        d_y_value = d_in * self.inputs[0]
        return d_x_value, d_y_value


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

    def backward(self, d_in):
        d_x_value = np.dot(self.inputs[1].T, d_in)
        d_y_value = np.dot(d_in, self.inputs[0].T)
        return d_x_value, d_y_value
