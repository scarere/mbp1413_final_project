import torch
import torch.nn as nn
from torch import Tensor
from kornia.augmentation import *

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            RandomGaussianBlur((3, 3),(0.1, 2.0),p=0.5),
            RandomChannelShuffle(p=0.5),
            RandomEqualize(p=0.5),
            RandomInvert(p=0.5),
            RandomPosterize(3, p=0.5),
            RandomSolarize(0.1, 0.1, p=0.5)
        )

        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    # TODO: for geometric transformations we have to transform y
    
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, y: Tensor) -> Tensor: 
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out, y