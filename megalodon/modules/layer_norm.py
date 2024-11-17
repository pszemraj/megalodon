from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numbers
import logging

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    Standard PyTorch LayerNorm implementation to replace LayerNorm.
    Maintains the same interface for easy swapping.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight if self.elementwise_affine else None,
            self.bias if self.elementwise_affine else None,
            self.eps,
        )

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class RMSNorm(nn.Module):
    """
    Standard RMSNorm implementation in PyTorch to replace RMSNorm.
    RMSNorm is a variant of LayerNorm that only normalizes by root mean square,
    without centering the mean.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        # Calculate RMS
        norm_dims = tuple(range(-(len(self.normalized_shape)), 0))
        rms = torch.rsqrt(torch.mean(x * x, dim=norm_dims, keepdim=True) + self.eps)

        # Normalize
        x_normed = x * rms

        # Apply weight if using elementwise affine
        if self.elementwise_affine:
            x_normed = x_normed * self.weight

        return x_normed

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


def layernorm_fwd(
    x: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], dim: int, eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Standard forward pass for layer normalization"""
    dims = tuple(range(-dim, 0))
    mean = x.mean(dims, keepdim=True)
    var = x.var(dims, unbiased=False, keepdim=True)
    invvar = 1 / torch.sqrt(var + eps)

    # Normalize
    x_normed = (x - mean) * invvar

    # Apply weight and bias if provided
    if weight is not None:
        x_normed = x_normed * (weight + 1.0)
    if bias is not None:
        x_normed = x_normed + bias

    return x_normed, mean.squeeze(dims), invvar.squeeze(dims)


def layernorm_bwd(
    grad_output: Tensor,
    x: Tensor,
    mean: Tensor,
    invvar: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    dim: int,
    eps: float,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Standard backward pass for layer normalization"""
    dims = tuple(range(-dim, 0))

    if weight is None:
        # Gradient without affine transformation
        grad_input = grad_output
    else:
        # Gradient with affine transformation
        grad_input = grad_output * (weight + 1.0)

    # Compute gradients
    grad_input = grad_input * invvar.unsqueeze(dims)

    # Compute weight and bias gradients if needed
    grad_weight = None
    grad_bias = None
    if weight is not None:
        grad_weight = torch.sum(grad_output * x_normed, dims)
    if bias is not None:
        grad_bias = torch.sum(grad_output, dims)

    return grad_input, grad_weight, grad_bias


def rmsnorm_fwd(
    x: Tensor, weight: Optional[Tensor], dim: int, eps: float
) -> Tuple[Tensor, Tensor]:
    """Standard forward pass for RMS normalization"""
    dims = tuple(range(-dim, 0))
    rms = torch.rsqrt(torch.mean(x * x, dim=dims, keepdim=True) + eps)
    x_normed = x * rms

    if weight is not None:
        x_normed = x_normed * (weight + 1.0)

    return x_normed, rms.squeeze(dims)


def rmsnorm_bwd(
    grad_output: Tensor,
    x: Tensor,
    rms: Tensor,
    weight: Optional[Tensor],
    dim: int,
    eps: float,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Standard backward pass for RMS normalization"""
    dims = tuple(range(-dim, 0))

    if weight is None:
        grad_input = grad_output
    else:
        grad_input = grad_output * (weight + 1.0)

    grad_input = grad_input * rms.unsqueeze(dims)

    grad_weight = None
    if weight is not None:
        grad_weight = torch.sum(grad_output * x_normed, dims)

    return grad_input, grad_weight
