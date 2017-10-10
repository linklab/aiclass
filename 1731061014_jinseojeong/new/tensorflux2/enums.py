from enum import Enum
import tensorflux2.initializers as tfi
import tensorflux2.optimizers as tfo
import tensorflux2.layers as tfl


class Initializer(Enum):
    Zero                = tfi.Zero_Initializer
    One                 = tfi.One_Initializer
    Point_One           = tfi.Point_One_Initializer
    Truncated_Normal    = tfi.Truncated_Normal_Initializer
    Value_Assignment    = tfi.Value_Assignment_Initializer
    Random_Initializer  = tfi.Random_Initializer


class Optimizer(Enum):
    SGD = tfo.SGD


class Activator(Enum):
    Sigmoid = tfl.Sigmoid,
    ReLU = tfl.ReLU
