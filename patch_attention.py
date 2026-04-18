"""
Patch HY-World 2.0 attention.py to gracefully handle missing flash-attention.

The original code has a try/except that catches FA3 ImportError and falls back to FA2,
but if FA2 is also missing the module fails to load. The _apply_attention method already
has an SDPA fallback for fp32 dtypes, so we just need the import to not crash.
"""

import re
from pathlib import Path

target = Path("/opt/hyworld2/hyworld2/worldrecon/hyworldmirror/models/layers/attention.py")

if not target.exists():
    print(f"WARN: {target} not found, skipping patch")
    exit(0)

# Check if flash_attn is actually importable
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    print("flash-attention is available — no patch needed")
    exit(0)
except ImportError:
    pass

print("flash-attention not available — patching attention.py for SDPA fallback")

src = target.read_text()

old_import = """try:
 from flash_attn_interface import flash_attn_func as flash_attn_func_v3
 _USE_FLASH_ATTN_V3 = True
except ImportError:
 from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
 _USE_FLASH_ATTN_V3 = False"""

new_import = """_USE_FLASH_ATTN_V3 = False
_FLASH_ATTN_AVAILABLE = False
try:
 from flash_attn_interface import flash_attn_func as flash_attn_func_v3
 _USE_FLASH_ATTN_V3 = True
 _FLASH_ATTN_AVAILABLE = True
except ImportError:
 try:
  from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
  _FLASH_ATTN_AVAILABLE = True
 except ImportError:
  flash_attn_func_v2 = None
  flash_attn_func_v3 = None
  import warnings
  warnings.warn("flash-attention not found; falling back to PyTorch SDPA for all dtypes")"""

if old_import in src:
    src = src.replace(old_import, new_import)
else:
    # Try a more flexible match
    pattern = r"try:\s*\n\s*from flash_attn_interface.*?_USE_FLASH_ATTN_V3 = False"
    match = re.search(pattern, src, re.DOTALL)
    if match:
        src = src[:match.start()] + new_import + src[match.end():]
    else:
        print("WARN: Could not find flash-attn import block to patch")
        exit(0)

# Also patch _apply_attention to check _FLASH_ATTN_AVAILABLE
old_apply = "if q.dtype==torch.bfloat16 or q.dtype==torch.float16:"
new_apply = "if _FLASH_ATTN_AVAILABLE and (q.dtype==torch.bfloat16 or q.dtype==torch.float16):"

src = src.replace(old_apply, new_apply)

target.write_text(src)
print("Patch applied successfully — SDPA fallback active for all dtypes")
