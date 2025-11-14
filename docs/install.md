# Installation

This document covers environment setup and installation of Megalodon.

## System Requirements

### Hardware

**Minimum**:
- 1 NVIDIA GPU with CUDA support (for GPU version)
- 16GB RAM
- 50GB disk space

**Recommended**:
- Multiple NVIDIA GPUs (A100, H100, or V100 for large models)
- 64GB+ RAM
- NVMe SSD for fast checkpoint loading

**CPU-Only**:
- Supported, but custom CUDA kernels will not be available
- Model will fall back to PyTorch implementations (slower)
- Useful for development or testing on small models

### Software

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
  - macOS and Windows are not officially supported (CUDA kernel compilation may fail)
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.7 or later (for GPU version)
- **GCC**: 7.0 or later (for C++ extension compilation)

## Installation Methods

### Option 1: CPU-Only Installation (No CUDA)

If you don't have a CUDA-capable GPU or want to test on CPU:

```bash
# Install ninja (build tool)
pip install ninja

# Install PyTorch CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Clone the repository
git clone https://github.com/XuezheMax/megalodon.git
cd megalodon

# Install Python dependencies
pip install -r requirements.txt

# Install megalodon (will skip CUDA extension build)
pip install -e .
```

**Note**: Custom CUDA kernels will not be available. The model will use PyTorch fallback implementations, which are significantly slower. This is **only recommended for development or testing on small models**.

### Option 2: GPU Installation (CUDA 11.7)

This is the **recommended** installation for actual use. Follows the original README instructions.

#### Step 1: Install PyTorch with CUDA

Using conda (recommended):

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Or using pip:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

Verify installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

#### Step 2: Install NVIDIA Apex

Apex provides optimized layer norm and other ops used by Megalodon.

```bash
# Clone Apex repository
git clone https://github.com/NVIDIA/apex.git
cd apex

# Checkout specific version (important for compatibility)
git checkout 23.08

# Install packaging dependency
pip install packaging

# Build and install with C++ and CUDA extensions
# Choose one of the following based on your pip version:

# If pip >= 23.1:
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./

# If pip < 23.1:
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ..
```

**Troubleshooting**:
- If compilation fails with "cuda_runtime.h not found", ensure CUDA toolkit is installed: `sudo apt install nvidia-cuda-toolkit`
- If you get version mismatch errors, ensure PyTorch was installed with CUDA 11.7 (not 11.8 or 12.x)

#### Step 3: Install Fairscale (BF16 Branch)

Fairscale provides FSDP (Fully Sharded Data Parallel) support.

```bash
# Clone Fairscale repository
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale

# Checkout BF16-compatible branch (required for bfloat16 training)
git checkout ngoyal_bf16_changes

# Install
pip install .

cd ..
```

**Note**: The `ngoyal_bf16_changes` branch is unmaintained. If it's no longer available, you may need to use the main branch, but BF16 support may be limited.

#### Step 4: Install Megalodon

```bash
# Clone the repository
git clone https://github.com/XuezheMax/megalodon.git
cd megalodon

# Install Python dependencies
pip install -r requirements.txt

# Install megalodon with custom CUDA extensions
pip install -e .
```

The `pip install -e .` command will:
1. Compile custom CUDA kernels (timestep_norm, ema_hidden, ema_parameters, fftconv, attention)
2. Build the `megalodon_extension` module
3. Install megalodon in editable mode

**Compilation time**: 5-10 minutes depending on your system.

**Verify installation**:

```bash
python -c "import megalodon; from megalodon.model.mega import Mega; print('Success')"
# Should print: Success
```

If you see import errors about `megalodon_extension`, the CUDA extension build failed. Check build logs.

### Option 3: Docker (Alternative)

If you want a containerized environment (not officially provided, but here's a sketch):

```dockerfile
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3.9 python3-pip git

# Install PyTorch
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install Apex
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && git checkout 23.08 && \
    pip install packaging && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install Fairscale
RUN git clone https://github.com/facebookresearch/fairscale.git && \
    cd fairscale && git checkout ngoyal_bf16_changes && \
    pip install .

# Install Megalodon
RUN git clone https://github.com/XuezheMax/megalodon.git && \
    cd megalodon && \
    pip install -r requirements.txt && \
    pip install -e .

WORKDIR /workspace/megalodon
```

Build and run:

```bash
docker build -t megalodon .
docker run --gpus all -it megalodon bash
```

## Python Dependencies

From `requirements.txt`:

- **fire**: Command-line interface for `eval.py`
- **pathlib**: Path manipulation (standard library in Python 3.4+, may be redundant)
- **sentencepiece**: Tokenization (SentencePiece models)
- **tqdm**: Progress bars
- **typeguard**: Runtime type checking (likely unused in practice)
- **wandb**: Weights & Biases logging (not used in provided scripts, but may be referenced)

Additional implicit dependencies (installed with PyTorch):
- **numpy**: Array operations
- **torch.distributed**: Multi-GPU training

## Environment Variables

Optional environment variables that may affect behavior:

### CUDA Settings

```bash
# Limit visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set CUDA device order (by bus ID for multi-node consistency)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Enable TF32 on Ampere GPUs (done automatically in eval.py)
export NVIDIA_TF32_OVERRIDE=1
```

### Distributed Training

```bash
# Master address for multi-node training
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500

# SLURM environment (if using SLURM, these are set automatically)
export SLURM_PROCID=0
export SLURM_NTASKS=8
```

### Python Settings

```bash
# Disable Python bytecode caching (useful for development)
export PYTHONDONTWRITEBYTECODE=1

# Unbuffered Python output (see logs immediately)
export PYTHONUNBUFFERED=1
```

## Verifying Installation

### Check GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU count: 4
GPU name: NVIDIA A100-SXM4-40GB
```

### Check Custom Extensions

```bash
python -c "import megalodon_extension.ops as ops; print('Custom kernels available')"
```

If this fails, CUDA extensions were not built correctly.

### Check Dependencies

```bash
python -c "import apex; print(f'Apex version: {apex.__version__}')"
python -c "import fairscale; print('Fairscale installed')"
python -c "import sentencepiece; print('SentencePiece installed')"
```

### Run a Small Model

```python
import torch
from megalodon.model.mega import ModelStore, build_model
from megalodon.distributed import initialize_model_parallel

# Initialize (single GPU, no parallelism)
torch.distributed.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)
initialize_model_parallel(model_parallel_size=1, chunk_parallel_size=1)

# Build a small model
cfg = ModelStore["mega200M"]
cfg.vocab_size = 10000
model = build_model(cfg, dtype="bf16", fp32_reduce_scatter=False, reshard_after_forward=False)

# Test forward pass
x = torch.randint(0, 10000, (1, 2048)).cuda()
y, _ = model(x)
print(f"Output shape: {y.shape}")  # Should be [1, 2048, 10000]
print("Success!")
```

Expected output:
```
Output shape: torch.Size([1, 2048, 10000])
Success!
```

## Troubleshooting

### Build Errors

**"nvcc not found"**:
- Install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
- Or add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

**"torch/extension.h not found"**:
- Ensure PyTorch is installed: `pip install torch`
- Check PyTorch version: `python -c "import torch; print(torch.__version__)"`

**"undefined symbol: _ZN2at..."**:
- Version mismatch between PyTorch and CUDA extensions
- Rebuild extensions: `pip install -e . --force-reinstall --no-cache-dir`

**"CUDA_HOME not set"**:
- Set explicitly: `export CUDA_HOME=/usr/local/cuda`
- Or install CUDA toolkit if missing

### Runtime Errors

**"CUDA out of memory"**:
- Reduce batch size
- Use smaller model (e.g., mega200M instead of mega7.3B)
- Enable gradient checkpointing (set `layerwise_ckpt=True` in config)

**"distributed package not available"**:
- PyTorch was built without distributed support
- Reinstall PyTorch with conda (includes distributed by default)

**"No module named 'megalodon_extension'"**:
- CUDA extensions didn't compile
- Check build logs: `pip install -e . 2>&1 | tee build.log`
- Try CPU-only installation if GPU not available

**"fairscale module not found"**:
- Install fairscale: `pip install fairscale` or follow Step 3 above
- If BF16 branch doesn't exist, use main branch: `git checkout main`

**"ImportError: cannot import name 'Tokenizer'"**:
- SentencePiece not installed: `pip install sentencepiece`

### Performance Issues

**Slow training/inference**:
- Check CUDA extensions are being used: `python -c "import megalodon_extension; print('OK')"`
- Enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True` (done in eval.py)
- Use efficient attention: `cfg.efficient_attn = "swift"`

**High memory usage**:
- Use BF16 instead of FP32: `dtype="bf16"`
- Enable FSDP sharding: `reshard_after_forward=True`
- Reduce batch size

## Uninstalling

To remove Megalodon:

```bash
cd megalodon
pip uninstall megalodon

# Optionally remove dependencies
pip uninstall apex fairscale
```

To do a clean reinstall:

```bash
pip uninstall megalodon
rm -rf build/ dist/ *.egg-info
pip install -e . --force-reinstall --no-cache-dir
```

## Platform-Specific Notes

### Ubuntu / Debian

Standard installation works as documented. Ensure CUDA drivers are installed:

```bash
nvidia-smi  # Should show GPU info
```

### CentOS / RHEL

May need to install development tools:

```bash
sudo yum groupinstall "Development Tools"
sudo yum install cuda-toolkit-11-7
```

### macOS

**Not officially supported**. CUDA is not available on macOS. CPU-only installation may work but is untested.

### Windows

**Not officially supported**. WSL2 with Ubuntu is recommended:

1. Install WSL2 with Ubuntu 20.04
2. Install CUDA on WSL2 (see NVIDIA WSL-CUDA documentation)
3. Follow Linux installation steps inside WSL2

## Advanced: Building from Source

If you want to modify CUDA kernels:

```bash
cd megalodon
# Edit csrc/*.cu or csrc/ops/*.cu

# Rebuild extensions
python setup.py build_ext --inplace

# Test
python -c "import megalodon_extension; print('OK')"
```

To clean build artifacts:

```bash
rm -rf build/ dist/ megalodon_extension*.so
```

## Hardware-Specific Optimizations

### Ampere GPUs (A100, A30)

- Enable TF32 (done automatically in eval.py):
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```
- Use BF16 for best performance: `dtype="bf16"`

### Volta / Turing GPUs (V100, T4)

- BF16 not supported, use FP16: `dtype="fp16"`
- May need mixed precision apex: `from apex import amp`

### Older GPUs (Pascal, Maxwell)

- FP16 may have poor performance
- Use FP32: `dtype="fp32"`
- Consider CPU-only for development

## Multi-Node Setup

For distributed training across multiple machines:

1. **Ensure network connectivity**:
   ```bash
   # On all nodes, check connectivity
   ping <other-node-ip>
   ```

2. **Setup passwordless SSH** (if using SLURM)

3. **Launch with torchrun**:
   ```bash
   # On master node:
   torchrun \
       --nnodes=2 \
       --nproc_per_node=8 \
       --node_rank=0 \
       --master_addr=192.168.1.1 \
       --master_port=29500 \
       eval.py ...

   # On worker node:
   torchrun \
       --nnodes=2 \
       --nproc_per_node=8 \
       --node_rank=1 \
       --master_addr=192.168.1.1 \
       --master_port=29500 \
       eval.py ...
   ```

4. **Or use SLURM** (if available):
   ```bash
   srun --nodes=2 --ntasks-per-node=8 python eval.py ...
   ```

**Requirements**:
- All nodes must have same GPU count and model
- Fast interconnect (InfiniBand recommended for >1 node)
- Synchronized checkpoints (shared filesystem or manual sync)
