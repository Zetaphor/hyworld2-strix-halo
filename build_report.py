import sys

has_fa = False
has_gs = False

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    has_fa = True
except Exception:
    pass

try:
    from gsplat.rendering import rasterization
    rasterization(test=True)
    has_gs = True
except NotImplementedError:
    pass
except TypeError:
    has_gs = True
except Exception:
    pass

import torch

print("=== HY-World 2.0 Build Report ===")
print(f"  flash-attention: {'YES' if has_fa else 'NO (SDPA fallback)'}")
print(f"  gsplat:          {'YES' if has_gs else 'NO (stub, GS head disabled)'}")
print(f"  Python:          {sys.version}")
print(f"  PyTorch:         {torch.__version__}")
print(f"  ROCm:            {torch.version.hip}")
