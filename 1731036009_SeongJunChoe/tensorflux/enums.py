# from enum import Enum # 파이선에 있는 기본 모듈
# import tensorflux.initializers as tfi
# import tensorflux.optimizers as tfo
# import tensorflux.layers as tfl
#
# # Enum 클래스를 상속 받음
# class Initializer(Enum):
#     Zero                    = tfi.Zero_Initializer # tfi의 Zero_Initializer 클래스가 할당됨, 파이선은 클래스도 변수
#     One                     = tfi.One_Initializer
#     Point_One               = tfi.Point_One_Initializer
#     Random                  = tfi.Random_Initializer
#     Truncated_Normal        = tfi.Truncated_Normal_Initializer
#     Value_Assignment        = tfi.Value_Assignment_Initializer
#
#     # Zero = tfi.Zero_Initializer
#     # One = tfi.One_Initializer
#     # Point_One = tfi.Point_One_Initializer
#     # Truncated_Normal=tfi.Truncated_Normal_Initializer
#     # Value_Assignment=tfi.Value_Assignment_Initializer
#
#
# class Optimizer(Enum):
#     SGD = tfo.SGD
#
#
# class Activator(Enum):
#     Sigmoid = tfl.Sigmoid,
#     ReLU = tfl.ReLU
from enum import Enum
import tensorflux.initializers as tfi
import tensorflux.optimizers as tfo
import tensorflux.layers as tfl


class Initializer(Enum):
    Zero                = tfi.Zero_Initializer
    One                 = tfi.One_Initializer
    Point_One           = tfi.Point_One_Initializer
    Truncated_Normal    = tfi.Truncated_Normal_Initializer
    Value_Assignment    = tfi.Value_Assignment_Initializer
    Normal              = tfi.Random_Normal_Initializer
    Uniform             = tfi.Random_Uniform_Initializer


class Optimizer(Enum):
    SGD = tfo.SGD


class Activator(Enum):
    Sigmoid = tfl.Sigmoid
    ReLU = tfl.ReLU
