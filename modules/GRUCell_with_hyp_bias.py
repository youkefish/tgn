import math
from torch.nn import Module
import torch
from torch import Tensor
from typing import Optional
from torch import _VF
from torch.nn import init
from torch.nn.parameter import Parameter
import manifolds

class RNNCellBase_with_hyp_bias(Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self, manifold: str, c: float, input_size: int, hidden_size: int, bias: bool, num_chunks: int) -> None:
        super(RNNCellBase_with_hyp_bias, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        self.manifold = getattr(manifolds, manifold)()
        self.c = c
        if bias:
            #bias_ih and bias_hh are hyperbolic ones!!!!!!!!!!
            bias_i = Parameter(torch.Tensor(num_chunks * hidden_size))
            bias_i = self.manifold.proj_tan0(bias_i.view(1, -1), self.c)
            hyp_bias_i = self.manifold.expmap0(bias_i, self.c)
            hyp_bias_i = self.manifold.proj(hyp_bias_i, self.c)
            self.bias_ih = hyp_bias_i
            bias_h = Parameter(torch.Tensor(num_chunks * hidden_size))
            bias_h = self.manifold.proj_tan0(bias_h.view(1, -1), self.c)
            hyp_bias_h = self.manifold.expmap0(bias_h, self.c)
            hyp_bias_h = self.manifold.proj(hyp_bias_h, self.c)
            self.bias_hh = hyp_bias_h
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)



class GRUCell_with_hyp_bias(RNNCellBase_with_hyp_bias):
    r"""A gated recurrent unit (GRU) cell with hyperbolic bias
    """

    def __init__(self, manifold,c, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(GRUCell_with_hyp_bias, self).__init__(manifold, c, input_size, hidden_size, bias, num_chunks=3)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        return _VF.gru_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
