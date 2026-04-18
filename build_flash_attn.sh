#!/usr/bin/env bash
set -euo pipefail

echo "=== Building flash-attention from source (CK RDNA3 backend) ==="

cd /opt
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

export MAX_JOBS=8
export PYTORCH_ROCM_ARCH=gfx1151
export GPU_ARCHS=gfx1151

pip install ninja packaging 2>/dev/null || true
if pip install --no-build-isolation -e . 2>&1 | tee /tmp/flash_attn_build.log; then
    echo "=== flash-attention build SUCCEEDED ==="
else
    echo "=== flash-attention build FAILED — will use SDPA fallback ==="
    cat /tmp/flash_attn_build.log | tail -40
    pip uninstall -y flash-attn 2>/dev/null || true
fi

cd /opt
rm -rf flash-attention/.git
