from enum import Enum
import tensorflux.initializers as tfi
import tensorflux.optimizers as tfo
import tensorflux.layers as tfl


class Initializer(Enum):
    Zero                = tfi.Zero_Initializer
    One                 = tfi.One_Initializer
    Point_One           = tfi.Point_One_Initializer
    Value_Assignment    = tfi.Value_Assignment_Initializer
    Uniform             = tfi.Random_Uniform_Initializer
    Normal              = tfi.Random_Normal_Initializer
    Truncated_Normal    = tfi.Truncated_Normal_Initializer
    Xavier              = tfi.Xavier
    He                  = tfi.He


class Optimizer(Enum):
    SGD         = tfo.SGD
    Momentum    = tfo.Momentum
    NAG         = tfo.NAG
    AdaGrad     = tfo.AdaGrad
    Adam        = tfo.Adam


class Activator(Enum):
    Sigmoid = tfl.Sigmoid
    ReLU = tfl.ReLU
