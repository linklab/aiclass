from enum import Enum
import tensorflux.initializers as tfi
import tensorflux.optimizers as tfo
import tensorflux.layers as tfl


class Initializer(Enum):
    Zero = tfi.Zero_Initializer
    One = tfi.One_Initializer
    Point_One = tfi.Point_One_Initializer
    Random = tfi.Random_Initializer
    Truncated_Normal=tfi.Truncated_Normal_Initializer
    Value_Assignment=tfi.Value_Assignment_Initializer


class Optimizer(Enum):
    SGD = tfo.SGD


class Activator(Enum):
    Sigmoid = tfl.Sigmoid,
    ReLU = tfl.ReLU
