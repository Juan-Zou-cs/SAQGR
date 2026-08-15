from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import nibabel as nib
import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(".")
sys.path.append("./..")

from dataset_2d import Dataset_2D, parse_fixed_indices
from model import CNN_2D


TASK_CONFIGS = OrderedDict({
    "6to90": {
        "b1000_task": "6to90_b1000",
        "b2000_task": "6to90_b2000",
        "data_b1000": "/media/sit/ST1/dataset/dwi_dataset/6to90_b1000",
        "data_b2000": "/media/sit/ST1/dataset/dwi_dataset/6to90_b2000",
    },
    "10to90": {
        "b1000_task": "10to90_b1000",
        "b2000_task": "10to90_b2000",
        "data_b1000": "/media/sit/ST1/dataset/dwi_dataset/10to90_b1000",
        "data_b2000": "/media/sit/ST1/dataset/dwi_dataset/10to90_b2000",
    },
})

NODDI_MAP_NAMES = ("NDI", "ODI", "FWF")
PAPER_ORDER = (("Vic", "NDI"), ("Viso", "FWF"), ("OD", "ODI"))
METRIC_NAMES = ("nrmse", "ssim", "psnr")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_logger(path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger(os.path.abspath(path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s:\t %(message)s")
        fh = logging.FileHandler(path, mode="a+", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def get_num_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def finite_mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std, int(arr.size)


def squeeze_extra_dim(t):
    if torch.is_tensor(t) and t.ndim == 5 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def unpack_batch(batch):
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    if len(batch) == 3:
        x, y, mask = batch
        bvec_out = None
    elif len(batch) == 4:
        x, y, mask, bvec_out = batch
    elif len(batch) == 5:
        x, y, mask, _, bvec_out = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return squeeze_extra_dim(x), squeeze_extra_dim(y), squeeze_extra_dim(mask), bvec_out


def normalize_bvec_np(bvec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(bvec, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (3, 90):
        arr = arr.T
    if arr.shape != (90, 3):
        raise ValueError(f"Expected bvecs_90 [90,3], got {arr.shape}")
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(n[:, 0] < eps):
        raise ValueError("Found zero-length vector in bvecs_90.")
    return (arr / np.maximum(n, eps)).astype(np.float32)


def ensure_bvec_batch(bvec, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if bvec is None:
        raise ValueError("Dataset_2D must return bvec_out; use return_bvec=True.")
    if not torch.is_tensor(bvec):
        bvec = torch.as_tensor(bvec)
    bvec = bvec.to(device=device, dtype=dtype, non_blocking=True)
    if bvec.ndim == 4 and bvec.shape[1] == 1:
        bvec = bvec.squeeze(1)
    if bvec.ndim == 2:
        bvec = bvec.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    elif bvec.ndim == 3:
        if bvec.shape[0] == 1 and batch_size > 1:
            bvec = bvec.expand(batch_size, -1, -1).contiguous()
        elif bvec.shape[0] != batch_size:
            raise ValueError(f"bvec batch mismatch: {tuple(bvec.shape)}, expected B={batch_size}")
    else:
        raise ValueError(f"Unexpected bvec shape: {tuple(bvec.shape)}")
    if bvec.shape[1:] != (90, 3):
        raise ValueError(f"Expected [B,90,3], got {tuple(bvec.shape)}")
    return bvec / torch.linalg.norm(bvec, dim=-1, keepdim=True).clamp_min(1e-8)


def read_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "network"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in ckpt.items()
    }


def infer_graph_config_from_checkpoint(state_dict):
    node_dim = None
    for key, value in state_dict.items():
        if key.endswith("out.readout_weight") and getattr(value, "ndim", 0) == 2:
            node_dim = int(value.shape[1])
            break
    if node_dim is None:
        for key, value in state_dict.items():
            if key.endswith("out.bvec_scale.bias") and getattr(value, "ndim", 0) == 1:
                node_dim = int(value.shape[0])
                break
    layer_ids = set()
    pattern = re.compile(r"(?:^|\.)out\.graph_fuse\.(\d+)\.")
    for key in state_dict:
        m = pattern.search(key)
        if m:
            layer_ids.add(int(m.group(1)))
    graph_layers = max(layer_ids) + 1 if layer_ids else None
    return node_dim, graph_layers


def build_model(checkpoint_path: str, in_channels: int, fixed_indices, device: torch.device, args):
    state = read_checkpoint_state_dict(checkpoint_path)
    detected_node_dim, detected_layers = infer_graph_config_from_checkpoint(state)

    node_dim = int(args.graph_node_dim) if int(args.graph_node_dim) > 0 else detected_node_dim
    layers = int(args.graph_layers) if int(args.graph_layers) > 0 else detected_layers
    if node_dim is None:
        raise ValueError("Could not infer graph_node_dim; pass --graph_node_dim.")
    if layers is None or layers <= 0:
        layers = 1

    model = CNN_2D(
        n_channels=int(in_channels),
        n_out=90,
        res_hiddens=(128, 256),
        mamba_scans=args.mamba_scans,
        d_state=int(args.mamba_d_state),
        d_conv=int(args.mamba_d_conv),
        expand=int(args.mamba_expand),
        dropout=float(args.dropout),
        graph_node_dim=int(node_dim),
        graph_layers=int(layers),
        graph_topk=int(args.graph_topk),
        graph_tau=float(args.graph_tau),
        fixed_indices=fixed_indices,
    ).to(device)

    incompatible = model.load_state_dict(state, strict=bool(args.strict_load))
    if not bool(args.strict_load):
        print("Missing keys   :", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)
    model.eval()

    config = {
        "in_channels": int(in_channels),
        "graph_node_dim": int(node_dim),
        "graph_layers": int(layers),
        "graph_topk": int(args.graph_topk),
        "graph_tau": float(args.graph_tau),
        "parameters": int(get_num_parameters(model)),
    }
    return model, config


def get_subject_id(file_path: str) -> str:
    try:
        with h5py.File(file_path, "r") as f:
            if "subject_id" in f.attrs:
                v = f.attrs["subject_id"]
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                return str(v)
    except Exception:
        pass
    stem = Path(file_path).stem
    for token in ("_b1000", "_b2000", "_b3000"):
        if token in stem:
            stem = stem.split(token)[0]
    return stem


def build_subject_groups(examples):
    groups = OrderedDict()
    for row_index, example in enumerate(examples):
        fp, sid = str(example[0]), int(example[1])
        if fp not in groups:
            groups[fp] = {
                "file_path": fp,
                "subject_id": get_subject_id(fp),
                "items": [],
            }
        groups[fp]["items"].append((sid, int(row_index)))
    output = []
    for group in groups.values():
        items = sorted(group["items"], key=lambda x: x[0])
        output.append({
            "file_path": group["file_path"],
            "subject_id": group["subject_id"],
            "slice_ids": [int(x[0]) for x in items],
            "row_indices": [int(x[1]) for x in items],
        })
    return output


def map_groups_by_subject(groups):
    out = {}
    for g in groups:
        sid = str(g["subject_id"])
        if sid in out:
            raise ValueError(f"Duplicate subject_id={sid}")
        out[sid] = g
    return out


def infer_one_subject(
    model, dataset, row_indices, fixed_indices, device, batch_size, num_workers, subject_name
):
    subset = Subset(dataset, list(row_indices))
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    refill_batches, gt_batches, mask_batches, bvec_batches = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader), desc=f"Infer {subject_name}", ncols=108, leave=False):
            x, y, mask, bvec_out = unpack_batch(batch)
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            if x.ndim != 4 or x.shape[1] != len(fixed_indices):
                raise ValueError(f"Expected x [B,{len(fixed_indices)},H,W], got {tuple(x.shape)}")

            bvec_forward = ensure_bvec_batch(bvec_out, x.shape[0], device, x.dtype)
            # Model returns both raw and measurement-consistent outputs.
            # NODDI fitting uses the final ODDC/refilled output by default.
            pred_raw, pred_refill = model(x, bvec_out=bvec_forward)
            if pred_raw.shape[1] != 90 or pred_refill.shape[1] != 90:
                raise ValueError(
                    f"Expected model outputs [B,90,H,W], got raw={tuple(pred_raw.shape)}, "
                    f"refill={tuple(pred_refill.shape)}"
                )

            refill_batches.append(pred_refill.cpu().numpy().astype(np.float32, copy=False))
            gt_batches.append(y.cpu().numpy().astype(np.float32, copy=False))

            mask_np = mask.cpu().numpy().astype(np.float32, copy=False)
            if mask_np.ndim == 4 and mask_np.shape[1] == 1:
                mask_np = mask_np[:, 0]
            elif mask_np.ndim != 3:
                raise ValueError(f"Unexpected mask shape {mask_np.shape}")
            mask_batches.append(mask_np)
            bvec_batches.append(bvec_forward.cpu().numpy().astype(np.float32, copy=False))

    refill = np.concatenate(refill_batches, axis=0)
    gt = np.concatenate(gt_batches, axis=0)
    mask = np.concatenate(mask_batches, axis=0) > 0
    bvec_all = np.concatenate(bvec_batches, axis=0)
    bvecs = normalize_bvec_np(bvec_all[0])

    return refill, gt, mask, bvecs


def check_shell_pair(subject_id, shell1000, shell2000):
    refill1, gt1, mask1, _ = shell1000
    refill2, gt2, mask2, _ = shell2000
    expected = gt1.shape
    for name, arr in (
        ("refill1000", refill1),
        ("refill2000", refill2),
        ("gt2000", gt2),
    ):
        if arr.shape != expected:
            raise ValueError(f"{subject_id}: shell mismatch gt1000={expected}, {name}={arr.shape}")
    if mask1.shape != mask2.shape:
        raise ValueError(f"{subject_id}: mask mismatch {mask1.shape} vs {mask2.shape}")


def build_multishell_dwi(shell1000, shell2000, mask_shw, min_signal, clip_max):
    b1 = np.asarray(shell1000, dtype=np.float32).copy()
    b2 = np.asarray(shell2000, dtype=np.float32).copy()
    mask = np.asarray(mask_shw, dtype=bool)
    if b1.shape != b2.shape or b1.ndim != 4 or b1.shape[1] != 90:
        raise ValueError(f"Unexpected shell shapes: {b1.shape}, {b2.shape}")
    b1 = np.nan_to_num(b1, nan=min_signal, posinf=clip_max, neginf=min_signal)
    b2 = np.nan_to_num(b2, nan=min_signal, posinf=clip_max, neginf=min_signal)
    np.clip(b1, min_signal, clip_max, out=b1)
    np.clip(b2, min_signal, clip_max, out=b2)
    b1 = np.transpose(b1, (2, 3, 0, 1))
    b2 = np.transpose(b2, (2, 3, 0, 1))
    mask_hws = np.transpose(mask, (1, 2, 0))
    b0 = np.ones(b1.shape[:3] + (1,), dtype=np.float32)
    dwi = np.concatenate([b0, b1, b2], axis=-1)
    dwi[~mask_hws] = 0.0
    return dwi.astype(np.float32, copy=False)


def build_multishell_protocol(bvec1000, bvec2000):
    b1 = normalize_bvec_np(bvec1000)
    b2 = normalize_bvec_np(bvec2000)
    bvals = np.concatenate([
        np.zeros(1, dtype=np.float64),
        np.full(90, 1000.0, dtype=np.float64),
        np.full(90, 2000.0, dtype=np.float64),
    ])
    bvecs = np.concatenate([
        np.zeros((1, 3), dtype=np.float64),
        b1.astype(np.float64),
        b2.astype(np.float64),
    ], axis=0)
    return bvals, bvecs


def import_amico():
    try:
        import amico
    except ImportError as e:
        raise ImportError("Install AMICO with: pip install dmri-amico") from e
    return amico


def write_amico_subject_files(subject_dir, dwi_hwsv, mask_shw, bvals, bvecs):
    subject_dir.mkdir(parents=True, exist_ok=True)
    mask_hws = np.transpose(np.asarray(mask_shw, dtype=np.uint8), (1, 2, 0))
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(np.asarray(dwi_hwsv, dtype=np.float32), affine), str(subject_dir / "DWI.nii.gz"))
    nib.save(nib.Nifti1Image(mask_hws.astype(np.uint8), affine), str(subject_dir / "mask.nii.gz"))
    np.savetxt(subject_dir / "DWI.bval", bvals.reshape(1, -1), fmt="%.8f")
    np.savetxt(subject_dir / "DWI.bvec", bvecs.T, fmt="%.10f")


def fit_amico_noddi(
    amico_module, study_root, subject_relpath, dwi_hwsv, mask_shw,
    bvals, bvecs, nthreads, blas_nthreads, dti_fit_method,
    regenerate_kernels, save_amico_results
):
    subject_dir = study_root / subject_relpath
    write_amico_subject_files(subject_dir, dwi_hwsv, mask_shw, bvals, bvecs)
    scheme_path = amico_module.util.fsl2scheme(
        str(subject_dir / "DWI.bval"),
        str(subject_dir / "DWI.bvec"),
    )
    scheme_name = Path(scheme_path).name
    if not (subject_dir / scheme_name).is_file():
        candidates = list(subject_dir.glob("*.scheme"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cannot resolve scheme in {subject_dir}")
        scheme_name = candidates[0].name

    ae = amico_module.Evaluation(study_path=str(study_root), subject=str(subject_relpath))
    ae.load_data(
        dwi_filename="DWI.nii.gz",
        scheme_filename=scheme_name,
        mask_filename="mask.nii.gz",
        b0_thr=0,
        replace_bad_voxels=0.0,
    )
    ae.set_model("NODDI")
    ae.set_config("nthreads", int(nthreads))
    ae.set_config("BLAS_nthreads", int(blas_nthreads))
    ae.set_config("DTI_fit_method", str(dti_fit_method).upper())
    ae.generate_kernels(regenerate=bool(regenerate_kernels))
    ae.load_kernels()
    ae.fit()

    maps_hwsk = np.asarray(ae.RESULTS["MAPs"], dtype=np.float32)
    names = list(ae.model.maps_name)
    if names[:3] != ["NDI", "ODI", "FWF"]:
        raise RuntimeError(f"Unexpected AMICO map order: {names}")

    maps = {}
    for i, name in enumerate(NODDI_MAP_NAMES):
        m = np.transpose(maps_hwsk[..., i], (2, 0, 1))
        m = np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
        m[~np.asarray(mask_shw, dtype=bool)] = 0.0
        maps[name] = m

    if bool(save_amico_results):
        ae.save_results()
    del ae, maps_hwsk
    gc.collect()
    return maps


def nrmse_fullfov(pred, gt):
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    g = np.asarray(gt, dtype=np.float64).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(g)
    p = p[valid]
    g = g[valid]
    if g.size == 0:
        return float("nan")
    denominator = np.sum(g * g, dtype=np.float64)
    if denominator <= 0:
        return float("nan")
    numerator = np.sum((p - g) ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / denominator))


def psnr_fullfov(pred, gt):
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    g = np.asarray(gt, dtype=np.float64).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(g)
    p = p[valid]
    g = g[valid]
    if g.size == 0:
        return float("nan")
    mse = float(np.mean((p - g) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim_subject_fullfov(pred, gt, win_size=7):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected [S,H,W], got {pred.shape}")

    values = []

    for sid in range(gt.shape[0]):
        g = gt[sid]
        p = pred[sid]

        actual_win = min(int(win_size), int(min(g.shape)))
        if actual_win % 2 == 0:
            actual_win -= 1
        if actual_win < 3:
            continue

        score = SSIM(
            g,
            p,
            data_range=1.0,
            win_size=actual_win,
            full=False,
        )

        if np.isfinite(score):
            values.append(float(score))

    return float(np.mean(values)) if values else float("nan")


def evaluate_maps(pred_maps, gt_maps, ssim_win_size):
    out = OrderedDict()

    for name in NODDI_MAP_NAMES:
        out[name] = {
            "nrmse": nrmse_fullfov(
                pred_maps[name],
                gt_maps[name],
            ),
            "ssim": ssim_subject_fullfov(
                pred_maps[name],
                gt_maps[name],
                ssim_win_size,
            ),
            "psnr": psnr_fullfov(
                pred_maps[name],
                gt_maps[name],
            ),
        }

    return out


def save_maps(output_dir, prefix, maps):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in NODDI_MAP_NAMES:
        np.save(output_dir / f"{prefix}_{name}.npy", maps[name].astype(np.float32))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--sampling", choices=["6to90", "10to90", "all"], default="all")

    p.add_argument("--data_6_b1000", default=TASK_CONFIGS["6to90"]["data_b1000"])
    p.add_argument("--data_6_b2000", default=TASK_CONFIGS["6to90"]["data_b2000"])
    p.add_argument("--data_10_b1000", default=TASK_CONFIGS["10to90"]["data_b1000"])
    p.add_argument("--data_10_b2000", default=TASK_CONFIGS["10to90"]["data_b2000"])

    p.add_argument("--model_6_b1000", default="")
    p.add_argument("--model_6_b2000", default="")
    p.add_argument("--model_10_b1000", default="")
    p.add_argument("--model_10_b2000", default="")

    p.add_argument("--result_root", default="/home/sit/project/2D_CNN/test_results/noddi_ours_fullfov_refill")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--strict_load", type=int, default=1, choices=[0,1])

    p.add_argument("--mamba_scans", default="hv", choices=["h","hv","hv_flip"])
    p.add_argument("--mamba_d_state", type=int, default=16)
    p.add_argument("--mamba_d_conv", type=int, default=4)
    p.add_argument("--mamba_expand", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--graph_node_dim", type=int, default=-1)
    p.add_argument("--graph_layers", type=int, default=-1)
    p.add_argument("--graph_topk", type=int, default=6)
    p.add_argument("--graph_tau", type=float, default=0.15)

    p.add_argument("--mode", choices=["refill"], default="refill")
    p.add_argument("--min_signal", type=float, default=1e-6)
    p.add_argument("--clip_max", type=float, default=1.0)
    p.add_argument("--ssim_win_size", type=int, default=7)

    p.add_argument("--amico_nthreads", type=int, default=32)
    p.add_argument("--amico_blas_nthreads", type=int, default=1)
    p.add_argument("--amico_dti_fit", default="OLS", choices=["OLS","LS","WLS"])
    p.add_argument("--regenerate_kernels", type=int, default=0, choices=[0,1])
    p.add_argument("--save_amico_results", type=int, default=0, choices=[0,1])
    p.add_argument("--keep_amico_work", type=int, default=0, choices=[0,1])
    p.add_argument("--save_dwi", type=int, default=0, choices=[0,1])
    p.add_argument("--save_maps", type=int, default=1, choices=[0,1])
    return p.parse_args()


def run_one_sampling(sampling: str, args, amico, device):
    cfg = TASK_CONFIGS[sampling]
    k = 6 if sampling == "6to90" else 10

    if sampling == "6to90":
        data_b1000, data_b2000 = args.data_6_b1000, args.data_6_b2000
        model_b1000, model_b2000 = args.model_6_b1000, args.model_6_b2000
    else:
        data_b1000, data_b2000 = args.data_10_b1000, args.data_10_b2000
        model_b1000, model_b2000 = args.model_10_b1000, args.model_10_b2000

    if not model_b1000 or not Path(model_b1000).is_file():
        raise FileNotFoundError(f"{sampling} b1000 model not found: {model_b1000}")
    if not model_b2000 or not Path(model_b2000).is_file():
        raise FileNotFoundError(f"{sampling} b2000 model not found: {model_b2000}")

    task1000 = cfg["b1000_task"]
    task2000 = cfg["b2000_task"]
    fixed1000 = list(parse_fixed_indices(task1000))
    fixed2000 = list(parse_fixed_indices(task2000))

    result_root = Path(args.result_root) / sampling
    result_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(result_root / "noddi_ours_fullfov_refill_logger.txt"))

    ds1000 = Dataset_2D(
        data_b1000, data_type="test", return_bvec=True,
        direction_task=task1000, fixed_indices=fixed1000,
        skip_empty_slices=False,
    )
    ds2000 = Dataset_2D(
        data_b2000, data_type="test", return_bvec=True,
        direction_task=task2000, fixed_indices=fixed2000,
        skip_empty_slices=False,
    )

    m1000 = map_groups_by_subject(build_subject_groups(ds1000.examples))
    m2000 = map_groups_by_subject(build_subject_groups(ds2000.examples))
    subject_ids = sorted(set(m1000) & set(m2000))
    if not subject_ids:
        raise RuntimeError(f"{sampling}: no common subjects")

    for sid in subject_ids:
        if m1000[sid]["slice_ids"] != m2000[sid]["slice_ids"]:
            raise ValueError(f"{sampling}/{sid}: b1000 and b2000 slice IDs do not align")

    model1000, conf1000 = build_model(model_b1000, k, fixed1000, device, args)
    model2000, conf2000 = build_model(model_b2000, k, fixed2000, device, args)

    logger.info("=" * 88)
    logger.info(f"Sampling             : {sampling}")
    logger.info(f"Input directions K   : {k}")
    logger.info(f"Subjects             : {len(subject_ids)}")
    logger.info(f"b1000 fixed indices  : {fixed1000}")
    logger.info(f"b2000 fixed indices  : {fixed2000}")
    logger.info("Mode                 : refill only")
    logger.info("Metric region        : FULL FOV")
    logger.info("NODDI                : synthetic b0=1 + 90@1000 + 90@2000")
    logger.info("Paper maps           : Vic / Viso / OD")
    logger.info("=" * 88)

    study_root = result_root / "amico_study"
    study_root.mkdir(parents=True, exist_ok=True)

    modes = ["refill"]
    metrics_by_mode = {
        mode: np.full((len(subject_ids), 3, 3), np.nan, dtype=np.float64)
        for mode in modes
    }
    records = []

    for sidx, sid in enumerate(subject_ids):
        logger.info(f"[{sidx+1}/{len(subject_ids)}] {sid}")

        shell1 = infer_one_subject(
            model1000, ds1000, m1000[sid]["row_indices"], fixed1000,
            device, args.batch_size, args.num_workers, f"{sid}/b1000"
        )
        shell2 = infer_one_subject(
            model2000, ds2000, m2000[sid]["row_indices"], fixed2000,
            device, args.batch_size, args.num_workers, f"{sid}/b2000"
        )
        check_shell_pair(sid, shell1, shell2)

        refill1, gt1, mask1, bvec1 = shell1
        refill2, gt2, mask2, bvec2 = shell2
        mask = np.asarray(mask1, bool) & np.asarray(mask2, bool)
        bvals, bvecs = build_multishell_protocol(bvec1, bvec2)

        gt_dwi = build_multishell_dwi(gt1, gt2, mask, args.min_signal, args.clip_max)
        gt_maps = fit_amico_noddi(
            amico, study_root, f"{sid}/gt", gt_dwi, mask, bvals, bvecs,
            args.amico_nthreads, args.amico_blas_nthreads, args.amico_dti_fit,
            bool(args.regenerate_kernels and sidx == 0),
            bool(args.save_amico_results),
        )

        subject_dir = result_root / "subjects" / sid
        subject_dir.mkdir(parents=True, exist_ok=True)
        np.save(subject_dir / "mask.npy", mask.astype(np.uint8))
        if args.save_maps:
            save_maps(subject_dir, "GT", gt_maps)

        mode = "refill"
        pred_dwi = build_multishell_dwi(refill1, refill2, mask, args.min_signal, args.clip_max)
        pred_maps = fit_amico_noddi(
            amico, study_root, f"{sid}/{mode}", pred_dwi, mask, bvals, bvecs,
            args.amico_nthreads, args.amico_blas_nthreads, args.amico_dti_fit,
            False, bool(args.save_amico_results),
        )
        metric_dict = evaluate_maps(pred_maps, gt_maps, args.ssim_win_size)

        for mi, map_name in enumerate(NODDI_MAP_NAMES):
            row = {
                "subject_id": sid,
                "sampling": sampling,
                "mode": mode,
                "map": map_name,
                "nrmse": float(metric_dict[map_name]["nrmse"]),
                "ssim": float(metric_dict[map_name]["ssim"]),
                "psnr": float(metric_dict[map_name]["psnr"]),
            }
            records.append(row)
            for qi, qname in enumerate(METRIC_NAMES):
                metrics_by_mode[mode][sidx, mi, qi] = row[qname]

        if args.save_maps:
            save_maps(subject_dir, "Pred_refill", pred_maps)
        if args.save_dwi:
            np.save(subject_dir / "Pred_refill_multishell_181.npy", pred_dwi.astype(np.float32))
        if args.save_dwi:
            np.save(subject_dir / "GT_multishell_181.npy", gt_dwi.astype(np.float32))

        if not args.keep_amico_work:
            work = study_root / sid
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = OrderedDict()
    for mode in modes:
        arr = metrics_by_mode[mode]
        np.save(result_root / f"metrics_fullfov_{mode}.npy", arr)
        np.savez(
            result_root / f"metrics_fullfov_{mode}.npz",
            metrics=arr,
            subjects=np.asarray(subject_ids),
            map_names=np.asarray(NODDI_MAP_NAMES),
            paper_map_names=np.asarray(["Vic", "OD", "Viso"]),
            metric_names=np.asarray(METRIC_NAMES),
        )

        summary[mode] = OrderedDict()
        sum_arr = np.full((3, 3, 2), np.nan, dtype=np.float64)
        for mi, map_name in enumerate(NODDI_MAP_NAMES):
            summary[mode][map_name] = OrderedDict()
            for qi, qname in enumerate(METRIC_NAMES):
                mean, std, n = finite_mean_std(arr[:, mi, qi])
                summary[mode][map_name][qname] = {"mean": mean, "std": std, "n": n}
                sum_arr[mi, qi, 0] = mean
                sum_arr[mi, qi, 1] = std
        np.save(result_root / f"summary_fullfov_{mode}.npy", sum_arr)

    with (result_root / "noddi_subject_metrics_fullfov_refill.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id","sampling","mode","map","nrmse","ssim","psnr"])
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "method": "Ours",
        "sampling": sampling,
        "input_directions_per_shell": k,
        "evaluation_modes": ["refill"],
        "metric_region": "full_fov",
        "input_shells": [1000, 2000],
        "protocol": "synthetic normalized b0=1 + 90@b1000 + 90@b2000 -> AMICO NODDI",
        "fixed_indices": {task1000: fixed1000, task2000: fixed2000},
        "model_paths": {"b1000": model_b1000, "b2000": model_b2000},
        "model_config": {"b1000": conf1000, "b2000": conf2000},
        "subject_order": subject_ids,
        "map_aliases": {"Vic": "NDI", "Viso": "FWF", "OD": "ODI"},
        "metric_protocol": {
            "nrmse": "sqrt(sum((pred-gt)^2)/sum(gt^2)) over the entire FOV",
            "ssim": "one SSIM value per full 2D slice, data_range=1, then average across slices",
            "psnr": "full-FOV MSE over all pixels/voxels, data_range=1",
            "aggregation": "subject-level first, then mean +/- sample std (ddof=1)",
        },
        "summary": summary,
        "records": records,
    }
    with (result_root / "noddi_summary_fullfov_refill.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    latex_lines = []
    for mode in modes:
        latex_lines.append(f"% ===== {sampling} / {mode} =====")
        for paper_name, internal in PAPER_ORDER:
            one = summary[mode][internal]
            latex_lines.append(
                f"{paper_name} & Ours & "
                f"${one['nrmse']['mean']:.4f} \\pm {one['nrmse']['std']:.4f}$ & "
                f"${one['ssim']['mean']:.4f} \\pm {one['ssim']['std']:.4f}$ & "
                f"${one['psnr']['mean']:.4f} \\pm {one['psnr']['std']:.4f}$ \\\\"
            )
    (result_root / "latex_ours_noddi_fullfov_refill.txt").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    print(f"\n===== Ours {sampling} / FULL-FOV =====")
    for mode in modes:
        print(f"[{mode}]")
        for paper_name, internal in PAPER_ORDER:
            one = summary[mode][internal]
            print(
                f"{paper_name}: "
                f"nRMSE={one['nrmse']['mean']:.4f} +/- {one['nrmse']['std']:.4f}, "
                f"SSIM={one['ssim']['mean']:.4f} +/- {one['ssim']['std']:.4f}, "
                f"PSNR={one['psnr']['mean']:.4f} +/- {one['psnr']['std']:.4f}"
            )


def main():
    start = time.time()
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    amico = import_amico()
    amico.setup()

    samplings = ["6to90", "10to90"] if args.sampling == "all" else [args.sampling]
    for sampling in samplings:
        run_one_sampling(sampling, args, amico, device)

    print(f"\nAll requested Ours NODDI tasks completed in {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
