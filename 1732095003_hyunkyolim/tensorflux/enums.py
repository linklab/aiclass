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
    Lecun_Normal        = tfi.Lecun_Normal
    Lecun_Uniform       = tfi.Lecun_Uniform
    Xavier_Normal       = tfi.Xavier_Normal
    Xavier_Uniform      = tfi.Xavier_Uniform
    He_Normal           = tfi.He_Normal
    He_Uniform          = tfi.He_Uniform
    Conv_Xavier_Normal  = tfi.Conv_Xavier_Normal
    Conv_Xavier_Uniform = tfi.Conv_Xavier_Uniform
    Conv_He_Normal      = tfi.Conv_He_Normal
    Conv_He_Uniform     = tfi.Conv_He_Uniform


class Optimizer(Enum):
    SGD         = tfo.SGD
    Momentum    = tfo.Momentum
    NAG         = tfo.NAG
    AdaGrad     = tfo.AdaGrad
    Adam        = tfo.Adam


class Activator(Enum):
    Sigmoid = tfl.Sigmoid
    ReLU = tfl.ReLU
