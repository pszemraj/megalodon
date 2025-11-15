"""
Triton implementation of TimestepNorm.

This kernel computes running mean and variance across sequence timesteps
using Welford's online algorithm for numerical stability.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _timestep_norm_fwd_kernel(
    # Input pointers
    x_ptr, prev_count_ptr, prev_mean_ptr, prev_var_ptr,
    gamma_ptr, beta_ptr, padding_mask_ptr,
    # Output pointers
    y_ptr, count_ptr, mean_ptr, var_ptr,
    cummean_ptr, cumrstd_ptr,
    # Dimensions
    B, L, D, G,
    eps: tl.constexpr,
    has_mask: tl.constexpr,
    # Strides
    stride_xb, stride_xl, stride_xd,
    stride_yb, stride_yl, stride_yd,
    stride_cb, stride_cl, stride_cg,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Welford's algorithm for online mean/variance computation.

    Each program handles one batch element and one group.
    """
    # Program ID
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Features in this group
    group_size = D // G
    feat_start = pid_g * group_size
    feat_offs = feat_start + tl.arange(0, BLOCK_SIZE)
    feat_mask = feat_offs < (feat_start + group_size)

    # Load previous statistics
    prev_count = tl.load(prev_count_ptr + pid_b)
    prev_mean_val = tl.load(prev_mean_ptr + pid_b * G + pid_g)
    prev_var_val = tl.load(prev_var_ptr + pid_b * G + pid_g)

    # Initialize Welford accumulators
    count = prev_count
    mean = prev_mean_val
    M2 = prev_var_val * prev_count

    # Load gamma and beta
    gamma = tl.load(gamma_ptr + feat_offs, mask=feat_mask, other=0.0)
    beta = tl.load(beta_ptr + feat_offs, mask=feat_mask, other=0.0)

    # Process each timestep
    for t in range(L):
        # Load input
        x_ptrs = x_ptr + pid_b * stride_xb + t * stride_xl + feat_offs * stride_xd
        x = tl.load(x_ptrs, mask=feat_mask, other=0.0)

        # Check padding mask
        valid = 1.0
        if has_mask:
            mask_val = tl.load(padding_mask_ptr + pid_b * L + t)
            valid = tl.where(mask_val > 0, 1.0, 0.0)

        # Welford update (only if not masked)
        if valid > 0.5:
            count = count + 1

            # Compute mean over group features
            x_sum = tl.sum(x, axis=0)
            x_mean = x_sum / group_size

            delta1 = x_mean - mean
            mean = mean + delta1 / count
            delta2 = x_mean - mean
            M2 = M2 + delta1 * delta2

        # Compute variance and reciprocal std
        var = M2 / tl.maximum(count, 1.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        # Normalize
        y = (x - mean) * rstd * gamma + beta

        # Store output
        y_ptrs = y_ptr + pid_b * stride_yb + t * stride_yl + feat_offs * stride_yd
        tl.store(y_ptrs, y, mask=feat_mask)

        # Store cumulative stats for backward
        cummean_ptrs = cummean_ptr + pid_b * stride_cb + t * stride_cl + pid_g * stride_cg
        cumrstd_ptrs = cumrstd_ptr + pid_b * stride_cb + t * stride_cl + pid_g * stride_cg
        if tl.program_id(1) == 0:  # Only first group stores
            tl.store(cummean_ptrs, mean)
            tl.store(cumrstd_ptrs, rstd)

    # Store final statistics
    tl.store(count_ptr + pid_b, count)
    tl.store(mean_ptr + pid_b * G + pid_g, mean)
    tl.store(var_ptr + pid_b * G + pid_g, var)


@triton.jit
def _timestep_norm_bwd_kernel(
    # Input pointers
    dy_ptr, x_ptr, cummean_ptr, cumrstd_ptr,
    gamma_ptr, dmean_ptr, dvar_ptr,
    # Output pointers
    dx_ptr, dprev_mean_ptr, dprev_var_ptr,
    dgamma_ptr, dbeta_ptr,
    # Dimensions
    B, L, D, G,
    # Strides
    stride_xb, stride_xl, stride_xd,
    stride_cb, stride_cl, stride_cg,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass for TimestepNorm."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = D // G
    feat_start = pid_g * group_size
    feat_offs = feat_start + tl.arange(0, BLOCK_SIZE)
    feat_mask = feat_offs < (feat_start + group_size)

    # Load gamma
    gamma = tl.load(gamma_ptr + feat_offs, mask=feat_mask, other=0.0)

    # Accumulators for dgamma and dbeta
    dgamma_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dbeta_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Process timesteps in reverse
    for t in range(L - 1, -1, -1):
        # Load cumulative stats
        cummean = tl.load(cummean_ptr + pid_b * stride_cb + t * stride_cl + pid_g * stride_cg)
        cumrstd = tl.load(cumrstd_ptr + pid_b * stride_cb + t * stride_cl + pid_g * stride_cg)

        # Load input and grad output
        x_ptrs = x_ptr + pid_b * stride_xb + t * stride_xl + feat_offs * stride_xd
        dy_ptrs = dy_ptr + pid_b * stride_xb + t * stride_xl + feat_offs * stride_xd

        x = tl.load(x_ptrs, mask=feat_mask, other=0.0)
        dy = tl.load(dy_ptrs, mask=feat_mask, other=0.0)

        # Normalized value
        xnorm = (x - cummean) * cumrstd

        # Gradients
        dbeta_acc += dy
        dgamma_acc += dy * xnorm

        # dx
        dx = dy * gamma * cumrstd

        # Store dx
        dx_ptrs = dx_ptr + pid_b * stride_xb + t * stride_xl + feat_offs * stride_xd
        tl.store(dx_ptrs, dx, mask=feat_mask)

    # Store accumulated gradients
    tl.atomic_add(dgamma_ptr + feat_offs, dgamma_acc, mask=feat_mask)
    tl.atomic_add(dbeta_ptr + feat_offs, dbeta_acc, mask=feat_mask)


class TimestepNormTriton(torch.autograd.Function):
    """
    Triton implementation of TimestepNorm with Welford's algorithm.

    This provides a numerically stable way to compute running statistics
    across sequence timesteps.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # [B, L, D]
        prev_count: torch.Tensor,  # [B]
        prev_mean: torch.Tensor,  # [B, G]
        prev_var: torch.Tensor,  # [B, G]
        gamma: torch.Tensor,  # [D]
        beta: torch.Tensor,  # [D]
        padding_mask: Optional[torch.Tensor] = None,  # [B, L]
        num_groups: int = 1,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, L, D = x.shape
        G = num_groups
        assert D % G == 0, f"D ({D}) must be divisible by num_groups ({G})"

        # Outputs
        y = torch.empty_like(x)
        count = torch.empty_like(prev_count)
        mean = torch.empty_like(prev_mean)
        var = torch.empty_like(prev_var)
        cummean = torch.empty(B, L, G, dtype=x.dtype, device=x.device)
        cumrstd = torch.empty(B, L, G, dtype=x.dtype, device=x.device)

        # Launch kernel
        BLOCK_SIZE = triton.next_power_of_2(D // G)
        grid = (B, G)

        _timestep_norm_fwd_kernel[grid](
            x, prev_count, prev_mean, prev_var, gamma, beta,
            padding_mask if padding_mask is not None else x,  # dummy
            y, count, mean, var, cummean, cumrstd,
            B, L, D, G, eps,
            padding_mask is not None,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            cummean.stride(0), cummean.stride(1), cummean.stride(2),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x, cummean, cumrstd, gamma)
        ctx.B = B
        ctx.L = L
        ctx.D = D
        ctx.G = G

        return y, count, mean, var

    @staticmethod
    def backward(ctx, dy, dcount, dmean, dvar):
        x, cummean, cumrstd, gamma = ctx.saved_tensors
        B, L, D, G = ctx.B, ctx.L, ctx.D, ctx.G

        dx = torch.empty_like(x)
        dgamma = torch.zeros_like(gamma)
        dbeta = torch.zeros_like(gamma)
        dprev_mean = torch.zeros(B, G, dtype=x.dtype, device=x.device)
        dprev_var = torch.zeros(B, G, dtype=x.dtype, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(D // G)
        grid = (B, G)

        _timestep_norm_bwd_kernel[grid](
            dy, x, cummean, cumrstd, gamma, dmean, dvar,
            dx, dprev_mean, dprev_var, dgamma, dbeta,
            B, L, D, G,
            x.stride(0), x.stride(1), x.stride(2),
            cummean.stride(0), cummean.stride(1), cummean.stride(2),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return dx, None, dprev_mean, dprev_var, dgamma, dbeta, None, None, None


def timestep_norm_triton(
    x: torch.Tensor,
    prev_count: Optional[torch.Tensor] = None,
    prev_mean: Optional[torch.Tensor] = None,
    prev_var: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    num_groups: int = 1,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton-based TimestepNorm.

    Args:
        x: Input tensor [B, L, D]
        prev_count: Previous timestep count [B]
        prev_mean: Previous mean [B, G]
        prev_var: Previous variance [B, G]
        gamma: Affine weight [D]
        beta: Affine bias [D]
        num_groups: Number of groups for group normalization
        eps: Epsilon for numerical stability

    Returns:
        y: Normalized output [B, L, D]
        count: Final count [B]
        mean: Final mean [B, G]
        var: Final variance [B, G]
    """
    B, L, D = x.shape
    G = num_groups

    if prev_count is None:
        prev_count = torch.zeros(B, dtype=torch.int64, device=x.device)
    if prev_mean is None:
        prev_mean = torch.zeros(B, G, dtype=x.dtype, device=x.device)
    if prev_var is None:
        prev_var = torch.zeros(B, G, dtype=x.dtype, device=x.device)
    if gamma is None:
        gamma = torch.ones(D, dtype=x.dtype, device=x.device)
    if beta is None:
        beta = torch.zeros(D, dtype=x.dtype, device=x.device)

    return TimestepNormTriton.apply(
        x, prev_count, prev_mean, prev_var, gamma, beta, None, num_groups, eps
    )


def group_timestep_norm_triton(
    x: torch.Tensor,
    prev_count: torch.Tensor,
    prev_mean: torch.Tensor,
    prev_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    num_groups: int = 32,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Group TimestepNorm with padding mask support.

    This is the main API matching the CUDA implementation.
    """
    return TimestepNormTriton.apply(
        x, prev_count, prev_mean, prev_var, gamma, beta,
        padding_mask, num_groups, eps
    )
