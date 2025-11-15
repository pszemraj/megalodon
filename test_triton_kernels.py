#!/usr/bin/env python3
"""
Quick validation script for Triton kernels.
Run this to check if kernels can be imported and execute basic operations.
"""
import sys

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if not torch.cuda.is_available():
            print("✗ CUDA not available - Triton kernels require GPU")
            return False
        print(f"✓ CUDA {torch.version.cuda}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        import triton
        print(f"✓ Triton {triton.__version__}")
    except ImportError:
        print("✗ Triton not installed (pip install triton)")
        return False

    return True

def test_imports():
    """Test that all Triton kernels can be imported."""
    try:
        from megalodon.triton_kernels import (
            timestep_norm_triton,
            ema_hidden_triton,
            ema_parameters_triton,
            fftconv_triton,
            swift_attention_triton,
        )
        print("✓ All Triton kernels imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_execution():
    """Test basic execution of each kernel."""
    import torch

    tests_passed = 0
    tests_total = 5

    # Test 1: TimestepNorm
    try:
        from megalodon.triton_kernels import timestep_norm_triton
        x = torch.randn(2, 128, 64, device='cuda')
        prev_count = torch.zeros(2, 1, device='cuda')
        prev_mean = torch.zeros(2, 1, 64, device='cuda')
        prev_var = torch.ones(2, 1, 64, device='cuda')
        gamma = torch.ones(64, device='cuda')
        beta = torch.zeros(64, device='cuda')
        y, _, _, _ = timestep_norm_triton(x, prev_count, prev_mean, prev_var, gamma, beta)
        assert y.shape == x.shape
        print("✓ TimestepNorm executes")
        tests_passed += 1
    except Exception as e:
        print(f"✗ TimestepNorm failed: {e}")

    # Test 2: EMA Hidden
    try:
        from megalodon.triton_kernels import ema_hidden_triton
        x = torch.randn(2, 128, 32, 2, device='cuda')
        p = torch.randn(2, 128, 32, 2, device='cuda')
        q = torch.randn(2, 128, 32, 2, device='cuda')
        h = ema_hidden_triton(x, p, q)
        assert h.shape == x.shape
        print("✓ EMA Hidden executes")
        tests_passed += 1
    except Exception as e:
        print(f"✗ EMA Hidden failed: {e}")

    # Test 3: EMA Parameters
    try:
        from megalodon.triton_kernels import ema_parameters_triton
        p = torch.randn(32, 8, 2, device='cuda')
        q = torch.randn(32, 8, 2, device='cuda')
        gamma = torch.randn(32, 8, 2, device='cuda')
        kernel, bias = ema_parameters_triton(p, q, gamma, None, L=128)
        assert kernel.shape == (32, 128, 2)
        print("✓ EMA Parameters executes")
        tests_passed += 1
    except Exception as e:
        print(f"✗ EMA Parameters failed: {e}")

    # Test 4: FFTConv
    try:
        from megalodon.triton_kernels import fftconv_triton
        x = torch.randn(2, 32, 128, device='cuda')
        k = torch.randn(32, 128, device='cuda')
        y = fftconv_triton(x, k)
        assert y.shape == x.shape
        print("✓ FFTConv executes")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FFTConv failed: {e}")

    # Test 5: Flash Attention
    try:
        from megalodon.triton_kernels import swift_attention_triton
        q = torch.randn(2, 512, 4, 64, device='cuda')
        k = torch.randn(2, 512, 4, 64, device='cuda')
        v = torch.randn(2, 512, 4, 64, device='cuda')
        out = swift_attention_triton(q, k, v, scale=0.125)
        assert out.shape == q.shape
        print("✓ Flash Attention executes")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Flash Attention failed: {e}")

    print(f"\nPassed {tests_passed}/{tests_total} execution tests")
    return tests_passed == tests_total

if __name__ == "__main__":
    print("=== Triton Kernel Validation ===\n")

    if not check_requirements():
        sys.exit(1)

    print()
    if not test_imports():
        sys.exit(1)

    print()
    if not test_basic_execution():
        print("\n⚠️  Some kernels failed - review errors above")
        sys.exit(1)

    print("\n✓ All validation tests passed!")
    print("Note: This validates basic execution only.")
    print("For correctness, compare outputs against CUDA kernels.")
