import networkx as nx

_default_graph = None


class Graph(nx.Graph):
    """Represents a computational graph (a neural network)
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []
        super().__init__()

    def initialize(self):
        global _default_graph
        _default_graph = self


class Placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """
    def __init__(self, name=None): #이름 줄 수 있음
        """Construct placeholder
        """
        self.output = None
        self.consumers = [] #consumers는 placeholder를 소비하는 것 즉 operation
        self.name = name    #자기 멤버변수 네임에 디폴트는 none
        if self.name is None:
            self.name = 'p' + str(len(_default_graph.placeholders) + 1)

        # Append this placeholder to the list of placeholders in the currently active default graph
        _default_graph.placeholders.append(self)
        _default_graph.add_node(self)

    def __str__(self):  #to string과 같음
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
        if self.name is None:
            self.name = 'v' + str(len(_default_graph.variables) + 1)

        # Append this variable to the list of variables in the currently active default graph
        _default_graph.variables.append(self)
        _default_graph.add_node(self)

    def __str__(self):
        return "V: " + self.name


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
        if self.name is None:
            self.name = 'o' + str(len(_default_graph.operations) + 1)

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)
            _default_graph.add_edge(input_node, self)

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)


    def forward(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass

    def __str__(self):
        return self.name


class Add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y, name=None):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        self.inputs = None  #실제 값을 넣어줌 포워드할때
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