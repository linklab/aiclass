from enum import Enum
import Tensorflux.initializers as tfi
import Tensorflux.optimizers as tfo
import Tensorflux.layers as tfl


class Initializer(Enum):
    #class가 담긴다, 파이썬에서 클래스도 객체
    Zero = tfi.Zero_Initializer
    One = tfi.One_Initializer
    Point_One = tfi.Point_One_Initializer
    Truncated_Normal = tfi.Truncated_Normal_Initializer
    Value_Assignment = tfi.Value_Assignment_Initializer


class Optimizer(Enum):
    SGD = tfo.SGD


class Activator(Enum):
    Sigmoid = tfl.Sigmoid,
    ReLU = tfl.ReLU
