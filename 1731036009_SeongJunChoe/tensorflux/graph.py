_default_graph = None

class Graph():

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def initialize(self):
        global _default_graph
        _default_graph = self