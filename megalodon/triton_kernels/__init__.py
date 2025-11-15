"""
Triton kernel implementations for Megalodon operations.

These kernels provide Python-based alternatives to the CUDA C++ kernels,
offering easier modification and experimentation while maintaining performance.

All kernels are designed to be drop-in replacements for the CUDA versions.
"""

from .timestep_norm import timestep_norm_triton, group_timestep_norm_triton
from .ema_hidden import ema_hidden_triton
from .ema_parameters import ema_parameters_triton
from .fftconv import fftconv_triton
from .attention import swift_attention_triton

__all__ = [
    'timestep_norm_triton',
    'group_timestep_norm_triton',
    'ema_hidden_triton',
    'ema_parameters_triton',
    'fftconv_triton',
    'swift_attention_triton',
]
