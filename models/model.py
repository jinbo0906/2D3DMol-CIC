from numbers import Number
from torch import nn
import torch

from .model_utils import model_decorator


@model_decorator
class Mol_ClCNetwork(nn.Module):

    def __init__(self, conf):
        super(Mol_ClCNetwork, self).__init__()
        pass

