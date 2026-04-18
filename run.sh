#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${SCRIPT_DIR}/input}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/output}"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# Detect available capabilities from the built image
CAPS=$(docker run --rm --entrypoint python hyworld2:latest -c "
has_fa = False
has_gs = False
try:
    from flash_attn.flash_attn_interface import flash_attn_func; has_fa = True
except: pass
try:
    from gsplat.rendering import rasterization
    rasterization(test=True)
    has_gs = True
except NotImplementedError: pass
except TypeError: has_gs = True
except: pass
print(f'FLASH_ATTN={int(has_fa)}')
print(f'GSPLAT={int(has_gs)}')
" 2>/dev/null || echo "FLASH_ATTN=0
GSPLAT=0")

eval "$CAPS"

EXTRA_ARGS=()

if [ "$FLASH_ATTN" = "1" ]; then
    echo "[hyworld2] flash-attention available → enabling bf16"
    EXTRA_ARGS+=(--enable_bf16)
else
    echo "[hyworld2] flash-attention unavailable → fp32 SDPA fallback"
fi

if [ "$GSPLAT" = "0" ]; then
    echo "[hyworld2] gsplat unavailable → disabling GS head"
    EXTRA_ARGS+=(--disable_heads gs --no_save_gs)
else
    echo "[hyworld2] gsplat available → full output (including 3DGS)"
fi

echo "[hyworld2] Input:  $INPUT_DIR"
echo "[hyworld2] Output: $OUTPUT_DIR"
echo "[hyworld2] Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e HF_HOME=/cache/huggingface \
    -v "${HOME}/.cache/huggingface:/cache/huggingface" \
    -v "${INPUT_DIR}:/input" \
    -v "${OUTPUT_DIR}:/output" \
    hyworld2:latest \
    --input_path /input \
    --output_path /output \
    --no_interactive \
    "${EXTRA_ARGS[@]}"
