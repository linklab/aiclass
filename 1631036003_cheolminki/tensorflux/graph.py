# -*- coding: utf-8 -*-
# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/

import networkx as nx

#_default_graph = None


class Graph(nx.Graph):      #nx.Graph를 상속받는다.
    """Represents a computational graph (a neural network)
    """

    def __init__(self):                 #생성자
        """Construct Graph"""
        self.operations = []            # 리스트 3개, operations, placeholders, variables
        self.placeholders = []          # 그래프의 vertex(node)
        self.variables = []
        super().__init__()

    def initialize(self):
        global _default_graph
        _default_graph = self           # 객체를 만들고 initialize하면 객체 자신이 선언

class Placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """
    def __init__(self, name=None):      #placeholder의 이름 설정 가능
        """Construct placeholder
        """
        self.consumers = [] #placeholder에 값을 넣을건데 placeholder를 소비하는 것 (ex) 다음 operator)
        self.name = name    #placeholder의 이름, 초기 None
        self.output = None
        if self.name is None:
            self.name = 'p' + str(len(_default_graph.placeholders) + 1)

        # Append this placeholder to the list of placeholders in the currently active default graph
        _default_graph.placeholders.append(self)    #placeholders라는 리스트로 initialize
        _default_graph.add_node(self)

    def __str__(self):      #자바의 tostring과 같은 효과
        return self.name

class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None, name=None):      #variable은 초기 value값을 넣어줌
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        self.value = initial_value
        self.consumers = []
        self.name = name
        self.output = None
        if self.name is None:
            self.name = 'v' + str(len(_default_graph.variables) + 1)

        # Append this variable to the list of variables in the currently active default graph
        _default_graph.variables.append(self)
        _default_graph.add_node(self)

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

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []
        self.name = name
        self.output = None
        if self.name is None:
            self.name = 'o' + str(len(_default_graph.operations) + 1)

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)
            _default_graph.add_edge(input_node, self)

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)
        _default_graph.add_node(self)

    def forward(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass        #forward는 자식클래스에서 정의할거라서 pass

    def __str__(self):
        return self.name


class Add(Operation):       #operation class를 상속받음.
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
        return x_value.dot(y_value) #matrix 연산, numpy의 dot product