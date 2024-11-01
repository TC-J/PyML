from typing import Callable, TypeAlias

import torch
from torch import dtype, Tensor
from torch.nn import Module, ModuleList, ReLU, Linear, Dropout, Threshold

from numpy import asarray, ndarray, savetxt

LayerIndex: TypeAlias = int

Probability: TypeAlias = float

ThresholdValue: TypeAlias = float

ReplacementValue: TypeAlias = float|int

class MLP(Module):
    """
        A multi-layer-perceptron, as a PyTorch `Module`.

        Multiple, ordered, PyTorch `Linear` layers with its hidden-layers' activation
        function.
        
        Optionally, with dropout- and/or threshold- training-layers.
    """
    def __init__(
        self,
        D_in: int,
        D_out: int,
        H: int|None = None,
        Hn: int = 0,
        activation: Callable = ReLU(),
        dropouts: list[Probability|None] | float | None = None,
        thresholds: list[tuple[float|float]|None] | tuple[float|float] | None = None,
        dtype: dtype = torch.float32
    ):
        """
             
        """
        super().__init__()

        self.dtype = dtype

        if not H or H == 0:
            H = int(round(D_in * 2/3 + D_out))

        layers = [D_in, *([H] * Hn), D_out]

        if not dropouts:
            dropouts = [None] * Hn

        elif len(dropouts) < Hn:
            dropouts = [
                *[Dropout(x) if x else None for x in dropouts],
                *[None for _ in range(Hn - len(dropouts))]
            ]

        else:
            dropouts = [
                Dropout(dropouts) if dropouts else None for _ in range(Hn)
            ]
        
        if not thresholds:
            thresholds = [None] * Hn
        
        elif isinstance(thresholds, tuple):
            thresholds = [Threshold(thresholds[0], thresholds[1]) for _ in range(Hn)]
        
        elif len(thresholds) < Hn:
            thresholds = [
                *[Threshold(t[0], t[1]) if t else None for t in thresholds],
                *([None] * (Hn - len(thresholds)))
            ]

        else:
            thresholds = [Threshold(t[0], t[1]) if t else None for t in thresholds]

        self.layers = ModuleList()

        for layer in range(len(layers) - 1):
            linear = Linear(
                in_features=layers[layer],
                out_features=layers[layer+1],
                dtype=dtype
            )

            if layer < (len(layers) - 2):
                self.layers.extend(modules=[linear, activation])

                if dropouts[layer]:
                    self.layers.append(dropouts[layer])

                if thresholds[layer]:
                    self.layers.append(thresholds[layer])

            else:
                self.layers.append(module=linear)
        

    def forward(
        self, 
        x: Tensor|ndarray|list|float|int
    ) -> Tensor:
        if isinstance(x, ndarray|list):
            x = torch.from_numpy(asarray(x))

        elif isinstance(x, int|float):
            x = torch.tensor([x], dtype=self.dtype)

        for layer in self.layers:
            x = layer(x)
        
        return x