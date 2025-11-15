"""
Triton implementation of EMA Parameters computation.

Computes the convolution kernel and bias from EMA coefficients by
building a Vandermonde matrix and applying projections.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _vandermonde_kernel(
    # Input pointers
    q_real_ptr, q_imag_ptr,
    # Output pointers
    vander_real_ptr, vander_imag_ptr,
    # Dimensions
    D, N, L,
    # Strides
    stride_qd, stride_qn,
    stride_vd, stride_vn, stride_vl,
    BLOCK_N: tl.constexpr,
):
    """
    Build Vandermonde matrix: vander[d, n, t] = q[d, n]^t

    Each program handles one dimension d.
    """
    pid_d = tl.program_id(0)

    n_offs = tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load q coefficients
    q_real = tl.load(q_real_ptr + pid_d * stride_qd + n_offs * stride_qn,
                    mask=n_mask, other=0.0)
    q_imag = tl.load(q_imag_ptr + pid_d * stride_qd + n_offs * stride_qn,
                    mask=n_mask, other=0.0)

    # Initialize: vander[t=0] = 1
    v_real = tl.full([BLOCK_N], 1.0, dtype=tl.float32)
    v_imag = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(L):
        # Store current power
        tl.store(vander_real_ptr + pid_d * stride_vd + n_offs * stride_vn + t * stride_vl,
                v_real, mask=n_mask)
        tl.store(vander_imag_ptr + pid_d * stride_vd + n_offs * stride_vn + t * stride_vl,
                v_imag, mask=n_mask)

        # Multiply by q: v[t+1] = v[t] * q
        # (v_r + i*v_i) * (q_r + i*q_i) = (v_r*q_r - v_i*q_i) + i*(v_r*q_i + v_i*q_r)
        v_real_new = v_real * q_real - v_imag * q_imag
        v_imag_new = v_real * q_imag + v_imag * q_real
        v_real = v_real_new
        v_imag = v_imag_new


@triton.jit
def _ema_kernel_kernel(
    # Input pointers
    vander_real_ptr, vander_imag_ptr,
    gamma_real_ptr, gamma_imag_ptr, p_ptr,
    # Output pointers
    kernel_real_ptr, kernel_imag_ptr,
    # Dimensions
    D, N, L,
    # Strides
    stride_vd, stride_vn, stride_vl,
    stride_gd, stride_gn,
    stride_kd, stride_kl,
    BLOCK_N: tl.constexpr,
):
    """
    Compute convolution kernel: k[d, t] = sum_n gamma[d, n] * p[d, n] * vander[d, n, t]

    Each program handles one dimension and one timestep.
    """
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)

    n_offs = tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load vander at this timestep
    v_real = tl.load(vander_real_ptr + pid_d * stride_vd + n_offs * stride_vn + pid_t * stride_vl,
                    mask=n_mask, other=0.0)
    v_imag = tl.load(vander_imag_ptr + pid_d * stride_vd + n_offs * stride_vn + pid_t * stride_vl,
                    mask=n_mask, other=0.0)

    # Load gamma and p
    gamma_real = tl.load(gamma_real_ptr + pid_d * stride_gd + n_offs * stride_gn,
                        mask=n_mask, other=0.0)
    gamma_imag = tl.load(gamma_imag_ptr + pid_d * stride_gd + n_offs * stride_gn,
                        mask=n_mask, other=0.0)
    p = tl.load(p_ptr + pid_d * stride_gd + n_offs * stride_gn,
               mask=n_mask, other=0.0)

    # Compute gamma * vander (complex multiplication)
    gv_real = gamma_real * v_real - gamma_imag * v_imag
    gv_imag = gamma_real * v_imag + gamma_imag * v_real

    # Multiply by p (real)
    gvp_real = p * gv_real
    gvp_imag = p * gv_imag

    # Sum over N dimension
    k_real = tl.sum(gvp_real, axis=0)
    k_imag = tl.sum(gvp_imag, axis=0)

    # Store kernel
    tl.store(kernel_real_ptr + pid_d * stride_kd + pid_t * stride_kl, k_real)
    tl.store(kernel_imag_ptr + pid_d * stride_kd + pid_t * stride_kl, k_imag)


@triton.jit
def _ema_bias_kernel(
    # Input pointers
    hx_real_ptr, hx_imag_ptr, gamma_real_ptr, gamma_imag_ptr,
    # Output pointers
    bias_real_ptr, bias_imag_ptr,
    # Dimensions
    B, D, N,
    # Strides
    stride_hb, stride_hd, stride_hn,
    stride_gd, stride_gn,
    stride_bb, stride_bd,
    BLOCK_N: tl.constexpr,
):
    """
    Compute bias from initial state: bias[b, d] = sum_n gamma[d, n] * hx[b, d, n]

    Each program handles one batch and one dimension.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    n_offs = tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load hx
    hx_real = tl.load(hx_real_ptr + pid_b * stride_hb + pid_d * stride_hd + n_offs * stride_hn,
                     mask=n_mask, other=0.0)
    hx_imag = tl.load(hx_imag_ptr + pid_b * stride_hb + pid_d * stride_hd + n_offs * stride_hn,
                     mask=n_mask, other=0.0)

    # Load gamma
    gamma_real = tl.load(gamma_real_ptr + pid_d * stride_gd + n_offs * stride_gn,
                        mask=n_mask, other=0.0)
    gamma_imag = tl.load(gamma_imag_ptr + pid_d * stride_gd + n_offs * stride_gn,
                        mask=n_mask, other=0.0)

    # Compute gamma * hx (complex multiplication)
    gh_real = gamma_real * hx_real - gamma_imag * hx_imag
    gh_imag = gamma_real * hx_imag + gamma_imag * hx_real

    # Sum over N dimension
    bias_real = tl.sum(gh_real, axis=0)
    bias_imag = tl.sum(gh_imag, axis=0)

    # Store bias
    tl.store(bias_real_ptr + pid_b * stride_bb + pid_d * stride_bd, bias_real)
    tl.store(bias_imag_ptr + pid_b * stride_bb + pid_d * stride_bd, bias_imag)


class EMAParametersTriton(torch.autograd.Function):
    """
    Triton implementation of EMA parameters computation.

    Builds convolution kernel via Vandermonde matrix construction.
    """

    @staticmethod
    def forward(
        ctx,
        p: torch.Tensor,  # [D, N, 1] real
        q: torch.Tensor,  # [D, N, 1] complex
        gamma: torch.Tensor,  # [D, N] complex
        hx: Optional[torch.Tensor],  # [B, D, N] complex
        length: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Compute EMA parameters.

        Args:
            p: Input coefficient (real)
            q: Decay coefficient (complex)
            gamma: Projection weights (complex)
            hx: Initial hidden state (complex, optional)
            length: Sequence length

        Returns:
            kernel: Convolution kernel [D, L] (real - takes real part)
            bias: Bias from initial state [B, D] (real, optional)
            vander: Vandermonde matrix [D, N, L] (complex, for backward)
        """
        D, N, _ = p.shape
        L = length
        device = p.device
        dtype = p.dtype

        # Split complex inputs
        q = q.squeeze(-1)  # [D, N]
        q_real = q.real if torch.is_complex(q) else q[..., 0]
        q_imag = q.imag if torch.is_complex(q) else q[..., 1]

        gamma_real = gamma.real if torch.is_complex(gamma) else gamma[..., 0]
        gamma_imag = gamma.imag if torch.is_complex(gamma) else gamma[..., 1]

        p = p.squeeze(-1)  # [D, N]

        # Build Vandermonde matrix
        vander = torch.zeros(D, N, L, 2, dtype=dtype, device=device)
        vander_real = vander[..., 0]
        vander_imag = vander[..., 1]

        BLOCK_N = triton.next_power_of_2(N)
        grid = (D,)

        _vandermonde_kernel[grid](
            q_real, q_imag, vander_real, vander_imag,
            D, N, L,
            q_real.stride(0), q_real.stride(1),
            vander.stride(0), vander.stride(1), vander.stride(2),
            BLOCK_N=BLOCK_N,
        )

        # Compute kernel: k = gamma @ vander * p
        kernel = torch.zeros(D, L, 2, dtype=dtype, device=device)
        kernel_real = kernel[..., 0]
        kernel_imag = kernel[..., 1]

        grid = (D, L)

        _ema_kernel_kernel[grid](
            vander_real, vander_imag, gamma_real, gamma_imag, p,
            kernel_real, kernel_imag,
            D, N, L,
            vander.stride(0), vander.stride(1), vander.stride(2),
            gamma.stride(0), gamma.stride(1) if gamma.ndim > 1 else 1,
            kernel.stride(0), kernel.stride(1),
            BLOCK_N=BLOCK_N,
        )

        # Take real part of kernel (imaginary should be ~0)
        kernel_out = kernel[..., 0]

        # Compute bias if initial state provided
        bias = None
        if hx is not None:
            B = hx.shape[0]
            hx_real = hx.real if torch.is_complex(hx) else hx[..., 0]
            hx_imag = hx.imag if torch.is_complex(hx) else hx[..., 1]

            bias = torch.zeros(B, D, 2, dtype=dtype, device=device)
            bias_real = bias[..., 0]
            bias_imag = bias[..., 1]

            grid = (B, D)

            _ema_bias_kernel[grid](
                hx_real, hx_imag, gamma_real, gamma_imag,
                bias_real, bias_imag,
                B, D, N,
                hx.stride(0) if hx.ndim > 2 else 0,
                hx.stride(1) if hx.ndim > 2 else 0,
                hx.stride(2) if hx.ndim > 2 else 0,
                gamma.stride(0), gamma.stride(1) if gamma.ndim > 1 else 1,
                bias.stride(0), bias.stride(1),
                BLOCK_N=BLOCK_N,
            )

            bias = bias[..., 0]  # Take real part

        vander_complex = torch.view_as_complex(vander.contiguous())
        ctx.save_for_backward(p, q, gamma, vander_complex)

        return kernel_out, bias, vander_complex

    @staticmethod
    def backward(ctx, dkernel, dbias, dvander):
        # Simplified backward - full implementation would compute gradients
        # w.r.t. p, q, gamma
        p, q, gamma, vander = ctx.saved_tensors

        # Return None gradients for simplicity
        # Full implementation would backprop through Vandermonde construction
        return None, None, None, None, None


def ema_parameters_triton(
    p: torch.Tensor,
    log_q: torch.Tensor,
    gamma: torch.Tensor,
    hx: Optional[torch.Tensor] = None,
    length: int = 2048,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute EMA parameters using Triton.

    Args:
        p: Input coefficient [D, N, 1] (real)
        log_q: Log of decay coefficient [D, N, 1] (complex)
        gamma: Projection weights [D, N] (complex)
        hx: Initial hidden state [B, D, N] (complex, optional)
        length: Sequence length

    Returns:
        kernel: Convolution kernel [D, L] (real)
        bias: Bias from initial state [B, D] (real, optional)
    """
    # Convert log_q to q
    q = torch.exp(log_q)

    kernel, bias, _ = EMAParametersTriton.apply(p, q, gamma, hx, length)
    return kernel, bias
