FROM rocm/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_ROCM_ARCH=gfx1151
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV HIP_VISIBLE_DEVICES=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ── Clone HY-World 2.0 ──────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/HY-World-2.0.git /opt/hyworld2

WORKDIR /opt/hyworld2

# ── Install Python deps (strip CUDA gsplat wheel, relax pinned versions) ─
RUN sed -i '/^gsplat/d' requirements.txt && \
    sed -i 's/open3d==0.18.0/open3d/' requirements.txt && \
    sed -i 's/^torch$/# torch/' requirements.txt && \
    sed -i 's/^torchvision$/# torchvision/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# ── Build flash-attention from source (CK RDNA3 backend) ────────────
COPY build_flash_attn.sh /opt/build_flash_attn.sh
RUN chmod +x /opt/build_flash_attn.sh && /opt/build_flash_attn.sh

# ── Build gsplat from source for HIP ────────────────────────────────
COPY build_gsplat.sh /opt/build_gsplat.sh
COPY gsplat_stub /opt/gsplat_stub
RUN chmod +x /opt/build_gsplat.sh && /opt/build_gsplat.sh

# ── Patch attention.py if flash-attn not available ───────────────────
COPY patch_attention.py /opt/patch_attention.py
RUN python /opt/patch_attention.py

# ── Write capability report on build ─────────────────────────────────
COPY build_report.py /opt/build_report.py
RUN python /opt/build_report.py

WORKDIR /opt/hyworld2
ENTRYPOINT ["python", "-m", "hyworld2.worldrecon.pipeline"]
