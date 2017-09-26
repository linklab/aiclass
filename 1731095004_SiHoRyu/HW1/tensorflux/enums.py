from enum import Enum
import HW1.tensorflux.initializers as tfi
import HW1.tensorflux.optimizers as tfo
import HW1.tensorflux.layers as tfl


class Initializer(Enum):
    Zero = tfi.Zero_Initializer
    One = tfi.One_Initializer
    Point_One = tfi.Point_One_Initializer
    Truncated_Normal=tfi.Truncated_Normal_Initializer
    Value_Assignment=tfi.Value_Assignment_Initializer


class Optimizer(Enum):
    SGD = tfo.SGD


class Activator(Enum):
    Sigmoid = tfl.Sigmoid
    ReLU = tfl.ReLU
