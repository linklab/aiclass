from enum import Enum
import tensorflux_HW1.initializers as tfi
import tensorflux_HW1.optimizers as tfo
import tensorflux_HW1.layers as tfl

#Enum class를 상속 받아서 각 initializer classes 를 정의함
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
    Sigmoid = tfl.Sigmoid #시그모이드 함수
    ReLU = tfl.ReLU       #ReLU 함수
