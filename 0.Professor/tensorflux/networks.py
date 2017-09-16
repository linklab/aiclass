from collections import OrderedDict
import tensorflux.graph as tfg
import tensorflux.enums as tfe
import tensorflux.layers as tfl
import networkx as nx


class Single_Neuron_Net(tfg.Graph):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input_node = None
        self.target_node = None

        self.activator = None
        self.initializer = None
        self.optimizer = None

        self.params = {}

        self.output = None
        self.error = None
        super().__init__()

    def set_data(self, input_node, target_node):
        self.input_node = input_node
        self.target_node = target_node

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        params_size_list = [self.input_size, self.output_size]
        self.initializer = initializer(self.params, params_size_list)

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator
        idx = 0
        u = tfl.Affine(self.params['W' + str(idx)], self.input_node, self.params['b' + str(idx)], name="A")
        self.output = activator(u, name="R")
        self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
        if isinstance(self, nx.Graph):
            self.add_edge(self.params['W' + str(idx)], u)
            self.add_edge(self.input_node, u)
            self.add_edge(self.params['b' + str(idx)], u)
            self.add_edge(u, self.output)
            self.add_edge(self.output, self.error)
            self.add_edge(self.error, self.target_node)
