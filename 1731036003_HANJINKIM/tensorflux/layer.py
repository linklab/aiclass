import tensorflux.graph as tfg

class Affine:
    def __init__(self, weight_node, input_node, bias_node):
        y = tfg.Mul(weight_node, input_node)
        z = tfg.Add(y, bias_node)