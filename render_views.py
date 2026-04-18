"""Render Gaussian splat viewpoints from a .ply file using gsplat."""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from gsplat.rendering import rasterization


def load_ply(path: str):
    """Load a 3DGS .ply file, returning tensors for means, quats, scales, opacities, colors."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

        header_str = header.decode()
        n_verts = 0
        for line in header_str.split("\n"):
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])

        props = []
        for line in header_str.split("\n"):
            if line.startswith("property float"):
                props.append(line.split()[-1])

        n_props = len(props)
        data = np.frombuffer(f.read(n_verts * n_props * 4), dtype=np.float32)
        data = data.reshape(n_verts, n_props)

    prop_idx = {name: i for i, name in enumerate(props)}

    means = data[:, [prop_idx["x"], prop_idx["y"], prop_idx["z"]]]
    quats = data[:, [prop_idx["rot_0"], prop_idx["rot_1"], prop_idx["rot_2"], prop_idx["rot_3"]]]
    scales = data[:, [prop_idx["scale_0"], prop_idx["scale_1"], prop_idx["scale_2"]]]
    opacities = data[:, prop_idx["opacity"]]

    sh_r = data[:, prop_idx["f_dc_0"]]
    sh_g = data[:, prop_idx["f_dc_1"]]
    sh_b = data[:, prop_idx["f_dc_2"]]

    C0 = 0.28209479177387814
    colors = np.stack([
        0.5 + C0 * sh_r,
        0.5 + C0 * sh_g,
        0.5 + C0 * sh_b,
    ], axis=-1)
    colors = np.clip(colors, 0.0, 1.0)

    return (
        torch.tensor(means, dtype=torch.float32),
        torch.tensor(quats, dtype=torch.float32),
        torch.tensor(scales, dtype=torch.float32),
        torch.tensor(opacities, dtype=torch.float32),
        torch.tensor(colors, dtype=torch.float32),
    )


def make_perturbed_cameras(orig_viewmat, angles_deg):
    """Generate cameras with small angular perturbations from the original viewpoint."""
    viewmats = []
    orig = np.array(orig_viewmat, dtype=np.float64)
    R_orig = orig[:3, :3]
    t_orig = orig[:3, 3]
    cam_pos = -R_orig.T @ t_orig

    for az_deg, el_deg in angles_deg:
        az = math.radians(az_deg)
        el = math.radians(el_deg)

        Ry = np.array([
            [math.cos(az), 0, math.sin(az)],
            [0, 1, 0],
            [-math.sin(az), 0, math.cos(az)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(el), -math.sin(el)],
            [0, math.sin(el), math.cos(el)],
        ])

        R_new = (Rx @ Ry @ R_orig.T).T
        t_new = -R_new @ cam_pos

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R_new.astype(np.float32)
        mat[:3, 3] = t_new.astype(np.float32)
        viewmats.append(mat)

    return viewmats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--camera_params", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument("--width", type=int, default=952)
    parser.add_argument("--height", type=int, default=560)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    means, quats, scales, opacities, colors = load_ply(args.ply)
    means = means.to(device)
    quats = quats.to(device)
    scales = torch.exp(scales.to(device))
    opacities = opacities.to(device)  # already sigmoid-activated in the PLY
    colors = colors.to(device)

    with open(args.camera_params) as f:
        cam = json.load(f)
    K_raw = cam["intrinsics"][0]["matrix"]
    fx, fy = K_raw[0][0], K_raw[1][1]
    orig_cx, orig_cy = K_raw[0][2], K_raw[1][2]
    orig_w = int(orig_cx * 2)
    orig_h = int(orig_cy * 2)

    scale_x = args.width / orig_w
    scale_y = args.height / orig_h
    K = torch.tensor([[
        [fx * scale_x, 0.0, args.width / 2.0],
        [0.0, fy * scale_y, args.height / 2.0],
        [0.0, 0.0, 1.0],
    ]], dtype=torch.float32, device=device)

    orig_ext = np.array(cam["extrinsics"][0]["matrix"], dtype=np.float32)

    perturbations = [
        (10, 0),    # slight right
        (-10, 0),   # slight left
        (0, 8),     # slight up
        (15, 5),    # right + up
        (-15, 5),   # left + up
    ]
    perturbed = make_perturbed_cameras(orig_ext, perturbations[:args.n_views])

    all_views = [("original", orig_ext)] + [
        (f"view_{i:02d}", vm) for i, vm in enumerate(perturbed)
    ]

    for name, vm in all_views:
        viewmat = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            rendered, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K,
                width=args.width,
                height=args.height,
                near_plane=0.01,
                far_plane=1000.0,
                render_mode="RGB",
                rasterize_mode="antialiased",
            )

        rgb = rendered[0].clamp(0, 1).cpu().numpy()
        alpha = alphas[0].cpu().numpy()
        # composite over white background
        img = rgb * alpha + (1.0 - alpha)
        img = (img.clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(out / f"{name}.png")
        print(f"Saved {name}.png ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
