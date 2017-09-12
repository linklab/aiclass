# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
_default_graph = None


class Graph():# 클래스를 만들 때는 두 줄을 new line
    """Represents a computational graph (a neural network)
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def initialize(self):
        global _default_graph
        _default_graph = self


class Placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """
    def __init__(self, name=None):
        """Construct placeholder
        """
        self.output = None
        self.consumers = [] # placeholder를 소비하는 변수인 operation노드를 의미한다.
        self.name = name
        if self.name is None:
            self.name = 'p' + str(len(_default_graph.placeholders) + 1)

        # Append this placeholder to the list of placeholders in the currently active default graph
        _default_graph.placeholders.append(self)

    def __str__(self): # 이 객체를 print할 때 찍히는 내용
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
        if self.name is None:
            self.name = 'o' + str(len(_default_graph.operations) + 1)

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self) # 각각의 변수가 갖고 있는 컨슈머를 지금 오퍼레이션(자기자신)을 등록

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)

    def forward(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass

    def __str__(self):
        return self.name


class Add(Operation):#상속
    """Returns x + y element-wise.
    """

    def __init__(self, x, y, name=None):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        self.inputs = None #이 자식클래스에서 관리하는 값(실제값)
        super().__init__([x, y], name) #부모클래스의 생성자를 부름

    def forward(self, x_value, y_value):#상속받은 것을 오버라이드
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        self.inputs = [x_value, y_value]
        return x_value + y_value # plus of numpy array


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

    def forward(self, x_value, y_value):# numpy 객체가 들어옴
        """Compute the output of the matmul operation

        Args:
          x_value: First matrix value
          y_value: Second matrix value
        """
        self.inputs = [x_value, y_value]
        return x_value.dot(y_value)