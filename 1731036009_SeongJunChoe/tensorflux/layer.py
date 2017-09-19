import numpy as np
import tensorflux.graph as tfg

class Affine:
    def __init__(self, weight_node, input_node, bias_node):
        # Create hidden node y
        y = tfg.Mul(weight_node, input_node) #곱 연산 클래스를 생성
        # Create output node z
        z = tfg.Add(y, bias_node) #합 연산 클래스를 생성