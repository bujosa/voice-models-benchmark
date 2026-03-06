# Venv Issues on Jetson Thor

## The Main Problem

Jetson Thor uses JetPack 7.0 with CUDA 13.0 on aarch64. PyTorch comes from the special index `pypi.jetson-ai-lab.io/sbsa/cu130`. The wheels include native NVIDIA libs that are not on the system.

## Issue 1: Venv Copied with `cp -a`

When copying a working venv:
```bash
cp -a ~/workspace/project-a/venv ~/workspace/project-b/venv
```

**Problem:** The shebangs of `bin/pip`, `bin/pip3`, etc. still point to the original path:
```
#!/home/<your-user>/workspace/project-a/venv/bin/python3
```

**Consequence:** `pip install` installs packages in the original venv, not the new one.

**Fix:**
```bash
grep -rl "project-a/venv/bin/python" ~/workspace/project-b/venv/bin/ | \
  xargs -I{} sed -i "1s|project-a/venv|project-b/venv|" {}
```

## Issue 2: Missing Native Libs

When creating a fresh venv with `python3 -m venv` and installing torch from the Jetson index, libs are missing such as:
- `libnvpl_lapack_lp64_gomp.so.0`
- `libnvpl_blas_lp64_gomp.so.0`
- `libarm_compute.so`
- `libgfortran.so.5`
- `libcudss.so.0`

**Cause:** The PyTorch wheel on the Jetson index does not include all libs. The original venv has them because additional nvidia-* packages were installed.

**Quick fix:** Copy the entire venv (with `cp -a`) and then correct the shebangs.

**Proper fix:** Install the nvidia-* packages explicitly:
```bash
pip install nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx
```

## Issue 3: cuBLAS Symlinks

```
pip install nvidia-cublas
```
This breaks cuBLAS on Jetson because it overwrites the system JetPack libs with incompatible versions.

**Fix:** Re-symlink to the JetPack libs:
```bash
VENV_NVIDIA=$VENV/lib/python3.12/site-packages/nvidia
SYSTEM_CUBLAS=/usr/local/cuda/lib64

ln -sf $SYSTEM_CUBLAS/libcublas.so.13 $VENV_NVIDIA/cublas/lib/libcublas.so.13
ln -sf $SYSTEM_CUBLAS/libcublasLt.so.13 $VENV_NVIDIA/cublas/lib/libcublasLt.so.13
```

## Recommended Recipe

```bash
# 1. Copy a working venv
cp -a ~/workspace/working-project/venv ~/workspace/new-project/venv

# 2. Fix shebangs
grep -rl "working-project/venv/bin/python" ~/workspace/new-project/venv/bin/ | \
  xargs -I{} sed -i "1s|working-project/venv|new-project/venv|" {}

# 3. Verify
~/workspace/new-project/venv/bin/pip show torch | grep Location
# Should show: /home/<your-user>/workspace/new-project/venv/lib/...

# 4. Install additional dependencies
~/workspace/new-project/venv/bin/pip install <new-package>

# 5. Verify it did not install in the wrong venv
~/workspace/new-project/venv/bin/pip show <new-package> | grep Location
```
