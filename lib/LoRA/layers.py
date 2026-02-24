import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

import math
from typing import Optional, List

class LoRALinear(nn.Linear):
    """
    Implements a Linear layer with Low-Rank Adaptation (LoRA).

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        r (int): Rank of the adaptation matrices (default: 2).
        alpha (float): Scaling hyperparameter for LoRA weights (default: 1).
    """
    def __init__(
            self,
            in_features,
            out_features, 
            r = 2, 
            alpha=1,
    ):
        nn.Linear.__init__(self, in_features, out_features)
        self.alpha = alpha
        self.r = r
        self.lora_A = nn.Parameter(self.weight.new_zeros((in_features, r)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((r, out_features)))
        self.scaling = self.alpha / r
        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def merge(self):
        """
        Permanently fuses the LoRA weights into the main weight matrix to eliminate inference latency.
        """
        if self.merged:
            return
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.weight += delta_w
            self.merged = True
    
    def unmerge(self):
        """
        Subtracts the LoRA weights from the main weight matrix to restore the original state.
        """
        if not self.merged:
            return
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.weight -= delta_w
            self.merged = False

    def forward(self, x):
        """
        Performs the forward pass by adding the low-rank adaptation to the standard linear projection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output of the linear layer plus the scaled LoRA path.
        """
        if self.merged:
            return F.linear(x, self.weight, bias=self.bias)
        result = F.linear(x, self.weight, bias=self.bias)
        result += (x @ self.lora_A @ self.lora_B) * self.scaling
        return result
    
class LoRAConv(nn.Module):
    """
    Wraps a 1D Convolutional layer (typically from Conv1D/Transformers) with Low-Rank Adaptation.

    Args:
        conv1d (nn.Module): The original 1D convolution module to be adapted.
        r (int): Rank of the adaptation matrices (default: 4).
        alpha (float): Scaling hyperparameter for LoRA weights (default: 1).
    """
    def __init__(
            self,
            conv1d,
            r=4,
            alpha=1,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.conv = conv1d
        in_channels = self.conv.nx
        out_channels = self.conv.nf
        self.lora_A = nn.Parameter(
            torch.zeros((in_channels, r))
        )
        self.lora_B = nn.Parameter(
            torch.zeros((r, out_channels))
        )
        self.scaling = self.alpha / r
        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        if self.merged:
            return
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.conv.weight += delta_w
            self.merged = True

    def unmerge(self):
        if not self.merged:
            return
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.conv.weight -= delta_w
            self.merged = False

    def forward(self, x):
        """
        Computes the forward pass using both the original convolution and the low-rank bottleneck path.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The sum of the original convolution output and the scaled LoRA output.
        """
        if self.merged:
            return self.conv(x)
        out_conv = self.conv(x)
        out_lora = x @ self.lora_A @ self.lora_B
        return out_conv + (out_lora * self.scaling)
        