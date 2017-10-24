from enum import Enum
import HW2.initializers as tfi
import HW2.optimizers as tfo
import HW2.layers as tfl


class Initializer(Enum):
    Zero                = tfi.Zero_Initializer
    One                 = tfi.One_Initializer
    Point_One           = tfi.Point_One_Initializer
    Truncated_Normal    = tfi.Truncated_Normal_Initializer
    Value_Assignment    = tfi.Value_Assignment_Initializer
    Normal              = tfi.Random_Normal_Initializer
    Uniform             = tfi.Random_Uniform_Initializer


class Optimizer(Enum):
    SGD         = tfo.SGD
    Momentum    = tfo.Momentum
    NAG         = tfo.NAG
    AdaGrad     = tfo.AdaGrad
    Adam        = tfo.Adam

class Activator(Enum):
    Sigmoid = tfl.Sigmoid
    ReLU = tfl.ReLU
