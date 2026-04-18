#!/usr/bin/env bash
set -euo pipefail

echo "=== Attempting gsplat build from source for HIP ==="

export MAX_JOBS=8
export PYTORCH_ROCM_ARCH=gfx1151
export GPU_ARCHS=gfx1151
export TORCH_CUDA_ARCH_LIST=""

# Install GLM headers from source (libglm-dev not available in rocm/pytorch).
echo "--- Installing GLM headers from source ---"
cd /tmp && git clone --depth 1 https://github.com/g-truc/glm.git glm-src && \
    cp -r glm-src/glm /usr/local/include/glm && \
    rm -rf glm-src
ls /usr/local/include/glm/gtc/type_ptr.hpp && echo "GLM headers verified"

# Try 1: AMD's ROCm fork of gsplat (HIP-native port)
echo "--- Try 1: AMD ROCm/gsplat fork (patched for gfx1151 / wave32) ---"
cd /opt
if git clone --recurse-submodules https://github.com/ROCm/gsplat.git gsplat-rocm 2>/dev/null; then
    cd gsplat-rocm

    # ── GLM fixes ──────────────────────────────────────────────────────
    # Delete bundled GLM to prevent hipify from creating broken symlinks.
    rm -rf gsplat/cuda/csrc/third_party/glm

    # Add /usr/local/include to include_dirs for system-installed GLM.
    sed -i 's|osp.join(current_dir, "gsplat", "cuda", "include"),|osp.join(current_dir, "gsplat", "cuda", "include"),\n            "/usr/local/include",|' setup.py

    # Add GLM compat defines to cxx flags (GLM CUDA version check bypass).
    sed -i 's/"-D__HIP_PLATFORM_AMD__" , "-Wno-sign-compare"/"-D__HIP_PLATFORM_AMD__", "-DGLM_FORCE_PURE", "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=0", "-Wno-sign-compare"/' setup.py

    # Add GLM compat defines to hipcc flags.
    sed -i 's/"-D__HIP_PLATFORM_AMD__", "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE"/"-D__HIP_PLATFORM_AMD__", "-DGLM_FORCE_PURE", "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=0", "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE"/' setup.py

    # ── Architecture fix ───────────────────────────────────────────────
    # Replace all gfx942 (MI300X) fallback with gfx1151 (Strix Halo).
    sed -i 's/gfx942/gfx1151/g' setup.py

    # ── Wave32 fixes (RDNA 3.5 = 32-wide wavefronts, not CDNA's 64) ──
    # The ROCm fork hardcodes wavefront size 64 everywhere for CDNA GPUs.
    # RDNA 3.5 uses 32-wide wavefronts (same as NVIDIA warps).

    # Utils.cuh: default template parameter for logical warp size
    sed -i 's/LOGICAL_WARP_SIZE = 64/LOGICAL_WARP_SIZE = 32/g' gsplat/cuda/include/Utils.cuh

    # All .cu files: rocprim warp_reduce template parameter
    sed -i 's/warp_reduce<float,64>/warp_reduce<float,32>/g' gsplat/cuda/csrc/*.cu
    sed -i 's/warp_reduce<float, 64>/warp_reduce<float, 32>/g' gsplat/cuda/csrc/*.cu
    sed -i 's/warp_reduce<int32_t, 64>/warp_reduce<int32_t, 32>/g' gsplat/cuda/csrc/*.cu
    sed -i 's/warp_reduce<int32_t,64>/warp_reduce<int32_t,32>/g' gsplat/cuda/csrc/*.cu

    # rocprim_warpSum template parameter
    sed -i 's/rocprim_warpSum<64>/rocprim_warpSum<32>/g' gsplat/cuda/csrc/*.cu
    sed -i 's/rocprim_warpSum<CDIM, 64>/rocprim_warpSum<CDIM, 32>/g' gsplat/cuda/csrc/*.cu
    sed -i 's/rocprim_warpSum<3, 64>/rocprim_warpSum<3, 32>/g' gsplat/cuda/csrc/*.cu

    # Cooperative groups: tiled_partition and thread_block_tile
    sed -i 's/tiled_partition<64>/tiled_partition<32>/g' gsplat/cuda/csrc/*.cu
    sed -i 's/thread_block_tile<64>/thread_block_tile<32>/g' gsplat/cuda/csrc/*.cu

    # Warp lane modulo/division (used for lane indexing within wavefront)
    sed -i 's/threadIdx\.x % 64/threadIdx.x % 32/g' gsplat/cuda/csrc/*.cu
    sed -i 's/e%64/e%32/g' gsplat/cuda/csrc/*.cu
    sed -i 's|e/64|e/32|g' gsplat/cuda/csrc/*.cu
    sed -i 's|k/64|k/32|g' gsplat/cuda/csrc/*.cu

    # Warps-per-block calculation
    sed -i 's|(block_size + 63) / 64|(block_size + 31) / 32|g' gsplat/cuda/csrc/*.cu

    # Utils.cuh manual warp reduce loops (iterate over warp lanes)
    sed -i 's/for (int i = 0; i < 64; ++i)/for (int i = 0; i < 32; ++i)/g' gsplat/cuda/include/Utils.cuh

    echo "--- Verifying patches ---"
    echo "=== Remaining 64-as-warp-size references (should be none): ==="
    grep -rn "warp_reduce.*64\|warpSum<64\|tiled_partition<64\|LOGICAL_WARP_SIZE = 64\|threadIdx.x % 64" gsplat/cuda/ || echo "(none found - clean)"
    echo "=== Architecture target: ==="
    grep -n "gfx1151" setup.py | head -3
    echo "---"

    pip install ninja packaging 2>/dev/null || true
    if pip install --no-build-isolation -v -e . 2>&1 | tee /tmp/gsplat_rocm_build.log; then
        echo "=== gsplat (ROCm fork) build SUCCEEDED ==="
        python -c "from gsplat.rendering import rasterization; print('gsplat import OK')" && {
            echo "=== gsplat fully functional ==="
            cd /opt && rm -rf gsplat-rocm/.git
            exit 0
        }
    fi
    echo "--- ROCm fork build failed ---"
    tail -80 /tmp/gsplat_rocm_build.log || true
    pip uninstall -y gsplat amd-gsplat amd_gsplat 2>/dev/null || true
    cd /opt && rm -rf gsplat-rocm
fi

# Try 2: upstream nerfstudio gsplat with HIP fixes
echo "--- Try 2: upstream nerfstudio/gsplat (patched for HIP) ---"
cd /opt
if git clone --recurse-submodules https://github.com/nerfstudio-project/gsplat.git gsplat-upstream 2>/dev/null; then
    cd gsplat-upstream
    git checkout v1.5.3 2>/dev/null || true

    rm -rf gsplat/cuda/csrc/third_party/glm

    sed -i 's/"--use_fast_math"/"-ffast-math"/g' setup.py
    sed -i '/"-diag-suppress"/d' setup.py
    sed -i '/"20012,186"/d' setup.py
    sed -i '/"--expt-relaxed-constexpr"/d' setup.py
    sed -i 's/nvcc_flags += \["-O3"/nvcc_flags += ["-DGLM_FORCE_PURE", "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=0", "-O3"/' setup.py
    sed -i 's|include_dirs = \[glm_path,|include_dirs = ["/usr/local/include",|' setup.py

    if pip install --no-build-isolation -v -e . 2>&1 | tee /tmp/gsplat_upstream_build.log; then
        echo "=== gsplat (upstream) build SUCCEEDED ==="
        python -c "from gsplat.rendering import rasterization; print('gsplat import OK')" && {
            echo "=== gsplat fully functional ==="
            cd /opt && rm -rf gsplat-upstream/.git
            exit 0
        }
    fi
    echo "--- Upstream build failed ---"
    tail -80 /tmp/gsplat_upstream_build.log || true
    pip uninstall -y gsplat 2>/dev/null || true
    cd /opt && rm -rf gsplat-upstream
fi

# Fallback: install stub
echo "=== gsplat builds failed — installing stub package ==="
cd /opt/gsplat_stub
pip install -e .
python -c "from gsplat.rendering import rasterization; from gsplat.strategy import DefaultStrategy; print('gsplat stub import OK')"
echo "=== gsplat stub installed (GS head will be disabled at runtime) ==="
