import tensorflux.graph as tfg
import tensorflux.enums as tfe
import tensorflux.session as tfs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from networkx.drawing.nx_agraph import graphviz_layout
import random
import string


class Deep_Neural_Network(tfg.Graph):
    def __init__(self):
        super().__init__()