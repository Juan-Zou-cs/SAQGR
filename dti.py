import argparse
import csv
import gc
import json
import logging
import os
import random
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

sys.path.append(".")
sys.path.append("./..")

from dataset_2d import Dataset_2D, DIRECTION_TASKS, parse_fixed_indices
from model import CNN_2D


SUPPORTED_TASKS = ("10to90_b1000", "10to90_b2000")
MAP_NAMES = ("FA", "MD", "AD")


# ============================================================
# General helpers
# ============================================================

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
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s:\t %(message)s"
        )
        fh = logging.FileHandler(path, mode="a+", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def get_num_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def squeeze_extra_dim(t):
    if torch.is_tensor(t) and t.ndim == 5 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def unpack_batch(batch):
    """Support Dataset_2D batches returning 3, 4, or 5 objects."""
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if len(batch) == 3:
        x, y_tg, mask = batch
        bvec_out = None
    elif len(batch) == 4:
        x, y_tg, mask, bvec_out = batch
    elif len(batch) == 5:
        x, y_tg, mask, _, bvec_out = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    return (
        squeeze_extra_dim(x),
        squeeze_extra_dim(y_tg),
        squeeze_extra_dim(mask),
        bvec_out,
    )


def ensure_bvec_batch(
    bvec,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bvec is None:
        raise ValueError(
            "bvec_out is required. Make sure HDF5 contains bvecs_90 and "
            "Dataset_2D(return_bvec=True) is used."
        )

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
            raise ValueError(
                f"bvec batch mismatch: got {tuple(bvec.shape)}, expected B={batch_size}"
            )
    else:
        raise ValueError(
            f"bvec_out must be [90,3] or [B,90,3], got {tuple(bvec.shape)}"
        )

    if bvec.shape[1] != 90 or bvec.shape[-1] != 3:
        raise ValueError(f"Expected bvec_out [B,90,3], got {tuple(bvec.shape)}")

    norm = torch.linalg.norm(bvec, dim=-1, keepdim=True).clamp_min(1e-8)
    return (bvec / norm).contiguous()


def normalize_bvec_np(bvec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(bvec, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (3, 90):
        arr = arr.T
    if arr.shape != (90, 3):
        raise ValueError(f"Expected bvecs_90 [90,3], got {arr.shape}")

    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norm[:, 0] < eps):
        raise ValueError("Found zero-length vector in bvecs_90")
    return arr / np.maximum(norm, eps)


def normalize_mask_np(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    while arr.ndim > 2 and 1 in (arr.shape[0], arr.shape[-1]):
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            break
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got {arr.shape}")
    return arr > 0


def read_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Read a checkpoint and return a plain, non-DataParallel state_dict.

    This helper is also used BEFORE model construction to infer the graph-node
    dimension stored in the checkpoint.  In your training script the QGSR/SGR
    defaults were:
        graph_node_dim=8
        graph_layers=1
        graph_topk=8
        graph_tau=0.10
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    state_dict = {}
    for key, value in checkpoint.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        state_dict[new_key] = value

    return state_dict


def infer_graph_config_from_checkpoint(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer graph_node_dim and graph_layers when possible.

    graph_node_dim is encoded directly in the checkpoint:
        out.readout_weight: [90, node_dim]
        out.node_proj.weight: [90*node_dim, 256, 1, 1]

    graph_topk and graph_tau do NOT change parameter tensor shapes and therefore
    cannot be recovered from a plain state_dict. Their CLI defaults below are
    set to the values used by the supplied training script: topk=8, tau=0.10.
    """
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

    graph_layer_ids = set()
    pattern = re.compile(r"(?:^|\.)out\.graph_fuse\.(\d+)\.")
    for key in state_dict.keys():
        match = pattern.search(key)
        if match:
            graph_layer_ids.add(int(match.group(1)))

    graph_layers = None
    if graph_layer_ids:
        graph_layers = max(graph_layer_ids) + 1

    return node_dim, graph_layers


def load_state_dict_compatible(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = True,
) -> None:
    state_dict = read_checkpoint_state_dict(checkpoint_path)

    model_has_module = any(
        key.startswith("module.") for key in model.state_dict().keys()
    )
    if model_has_module:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=bool(strict))

    if not strict:
        print(f"Missing keys   : {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")


def get_subject_id(file_path: str) -> str:
    try:
        with h5py.File(file_path, "r") as f:
            if "subject_id" in f.attrs:
                return str(f.attrs["subject_id"])
    except Exception:
        pass

    stem = Path(file_path).stem
    for token in ("_b1000", "_b2000", "_b3000"):
        if token in stem:
            stem = stem.split(token)[0]
    return stem


def build_subject_groups(examples: Sequence[Tuple[str, int]]) -> List[Dict]:
    """Group Dataset_2D row indices by original HDF5 file."""
    groups: "OrderedDict[str, Dict]" = OrderedDict()

    for row_index, example in enumerate(examples):
        file_path = str(example[0])
        slice_id = int(example[1])
        if file_path not in groups:
            groups[file_path] = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "subject_id": get_subject_id(file_path),
                "items": [],
            }
        groups[file_path]["items"].append((slice_id, row_index))

    output = []
    for g in groups.values():
        items = sorted(g["items"], key=lambda x: x[0])
        output.append(
            {
                "file_path": g["file_path"],
                "file_name": g["file_name"],
                "subject_id": g["subject_id"],
                "slice_ids": [int(x[0]) for x in items],
                "row_indices": [int(x[1]) for x in items],
            }
        )
    return output



def load_gt_one_subject(
    dataset: Dataset_2D,
    row_indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    subject_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load only GT 90-DWI, mask and bvecs for one subject.

    Returns:
        gt_90   : [S,90,H,W]
        mask    : [S,H,W] bool
        bvecs90 : [90,3]
    """
    subset = Subset(dataset, list(row_indices))
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        persistent_workers=False,
    )

    gt_batches = []
    mask_batches = []
    bvec_tables = []

    for batch in tqdm(
        loader,
        total=len(loader),
        desc=f"GT {subject_name}",
        ncols=105,
        leave=False,
    ):
        _, y_tg, mask, bvec_out = unpack_batch(batch)

        gt_batches.append(
            y_tg.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)
        if mask_np.ndim == 4 and mask_np.shape[1] == 1:
            mask_np = mask_np[:, 0]
        elif mask_np.ndim != 3:
            raise ValueError(
                f"Expected mask [B,H,W] or [B,1,H,W], got {mask_np.shape}"
            )
        mask_batches.append(mask_np)

        if bvec_out is None:
            raise ValueError(
                "Dataset_2D must return bvec_out for DTI fitting."
            )
        if not torch.is_tensor(bvec_out):
            bvec_out = torch.as_tensor(bvec_out)
        bvec_np = bvec_out.detach().cpu().numpy().astype(np.float32)

        if bvec_np.ndim == 2:
            bvec_np = bvec_np[None]
        if bvec_np.ndim == 4 and bvec_np.shape[1] == 1:
            bvec_np = bvec_np[:, 0]
        if bvec_np.ndim != 3:
            raise ValueError(f"Unexpected bvec batch shape={bvec_np.shape}")
        bvec_tables.append(bvec_np)

    gt_90 = np.concatenate(gt_batches, axis=0)
    mask = np.concatenate(mask_batches, axis=0) > 0
    bvec_all = np.concatenate(bvec_tables, axis=0)

    bvecs = normalize_bvec_np(bvec_all[0])

    return gt_90, mask, bvecs.astype(np.float32)


# ============================================================
# Inference
# ============================================================

def infer_one_subject(
    model: torch.nn.Module,
    dataset: Dataset_2D,
    row_indices: Sequence[int],
    fixed_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    subject_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One forward pass per slice batch, then keep TWO outputs.

    Returns:
        pred_raw_90    : [S,90,H,W], raw model prediction, NO refill
        pred_refill_90 : [S,90,H,W], observed directions copied from input x
        gt_90          : [S,90,H,W]
        mask           : [S,H,W]
        bvecs          : [90,3]

    The refill/ODDC operation is performed inside CNN_2D and only restores
    the physically acquired input channels at their fixed target indices.
    No GT value from an unobserved direction is inserted.
    """
    subset = Subset(dataset, list(row_indices))
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    raw_batches = []
    refill_batches = []
    gt_batches = []
    mask_batches = []
    bvec_tables = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            total=len(loader),
            desc=f"Infer {subject_name}",
            ncols=105,
            leave=False,
        ):
            x, y_tg, mask, bvec_out = unpack_batch(batch)

            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y_tg = y_tg.to(dtype=torch.float32)

            if x.ndim != 4:
                raise ValueError(f"Expected x [B,K,H,W], got {tuple(x.shape)}")
            if len(fixed_indices) != x.shape[1]:
                raise ValueError(
                    f"fixed_indices K={len(fixed_indices)} but "
                    f"input has K={x.shape[1]} channels"
                )

            bvec_forward = ensure_bvec_batch(
                bvec_out,
                batch_size=x.shape[0],
                device=device,
                dtype=x.dtype,
            )

            # The model now contains ODDC and directly returns both outputs:
            #   pred_raw    : learned 90-direction prediction before ODDC
            #   pred_refill : final measurement-consistent prediction
            pred_raw, pred_refill = model(x, bvec_out=bvec_forward)

            if (
                pred_raw.ndim != 4
                or pred_raw.shape[1] != 90
                or pred_refill.ndim != 4
                or pred_refill.shape[1] != 90
            ):
                raise RuntimeError(
                    f"Expected both outputs [B,90,H,W], got "
                    f"raw={tuple(pred_raw.shape)}, refill={tuple(pred_refill.shape)}"
                )

            raw_batches.append(
                pred_raw.detach().cpu().numpy().astype(np.float32)
            )
            refill_batches.append(
                pred_refill.detach().cpu().numpy().astype(np.float32)
            )
            gt_batches.append(
                y_tg.detach().cpu().numpy().astype(np.float32)
            )

            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            if mask_np.ndim == 4 and mask_np.shape[1] == 1:
                mask_np = mask_np[:, 0]
            elif mask_np.ndim != 3:
                raise ValueError(
                    f"Expected mask [B,H,W] or [B,1,H,W], got {mask_np.shape}"
                )
            mask_batches.append(mask_np)

            bvec_tables.append(
                bvec_forward.detach().cpu().numpy().astype(np.float32)
            )

    pred_raw_90 = np.concatenate(raw_batches, axis=0)
    pred_refill_90 = np.concatenate(refill_batches, axis=0)
    gt_90 = np.concatenate(gt_batches, axis=0)
    mask = np.concatenate(mask_batches, axis=0) > 0
    bvec_all = np.concatenate(bvec_tables, axis=0)

    bvecs = normalize_bvec_np(bvec_all[0])
    for i in range(1, bvec_all.shape[0]):
        cur = normalize_bvec_np(bvec_all[i])
        dots = np.abs(np.sum(cur * bvecs, axis=1))
        dots = np.clip(dots, 0.0, 1.0)
        max_angle = float(np.max(np.degrees(np.arccos(dots))))
        if max_angle > 1e-3:
            print(
                f"[WARNING] {subject_name}: bvec table differs across slices, "
                f"max same-index angle={max_angle:.6f} deg; using first table."
            )
            break

    return (
        pred_raw_90,
        pred_refill_90,
        gt_90,
        mask,
        bvecs.astype(np.float32),
    )


# ============================================================
# DTI fitting
# ============================================================

def make_gradient_table(shell: int, bvecs_90: np.ndarray):
    bvecs_90 = normalize_bvec_np(bvecs_90)

    bvals = np.concatenate(
        [
            np.zeros(1, dtype=np.float64),
            np.full(90, float(shell), dtype=np.float64),
        ]
    )
    bvecs = np.concatenate(
        [
            np.zeros((1, 3), dtype=np.float64),
            bvecs_90.astype(np.float64),
        ],
        axis=0,
    )
    return gradient_table(bvals, bvecs=bvecs)


def fit_dti_maps(
    signal_schw: np.ndarray,
    mask_shw: np.ndarray,
    bvecs_90: np.ndarray,
    shell: int,
    fit_method: str,
    min_signal: float,
    clip_max: float,
) -> Dict[str, np.ndarray]:
    """
    signal_schw: [S,90,H,W], normalized S/S0.
    mask_shw    : [S,H,W].

    Output FA/MD/AD: [S,H,W].
    """
    signal = np.asarray(signal_schw, dtype=np.float32)
    mask = np.asarray(mask_shw, dtype=bool)

    if signal.ndim != 4 or signal.shape[1] != 90:
        raise ValueError(f"Expected [S,90,H,W], got {signal.shape}")
    if mask.shape != (signal.shape[0], signal.shape[2], signal.shape[3]):
        raise ValueError(f"Mask {mask.shape} incompatible with signal {signal.shape}")

    # [S,90,H,W] -> [S,H,W,90]
    dwi = np.transpose(signal, (0, 2, 3, 1)).copy()
    dwi = np.nan_to_num(
        dwi,
        copy=False,
        nan=float(min_signal),
        posinf=float(clip_max),
        neginf=float(min_signal),
    )
    np.clip(dwi, float(min_signal), float(clip_max), out=dwi)

    # y_90 is S/S0, hence normalized b0 = 1.
    s0 = np.ones(dwi.shape[:-1] + (1,), dtype=np.float32)
    data = np.concatenate([s0, dwi], axis=-1)

    gtab = make_gradient_table(shell, bvecs_90)
    tenmodel = dti.TensorModel(
        gtab,
        fit_method=str(fit_method).upper(),
        min_signal=float(min_signal),
    )
    tenfit = tenmodel.fit(data, mask=mask)

    fa = np.asarray(tenfit.fa, dtype=np.float32)
    md = np.asarray(tenfit.md, dtype=np.float32)
    ad = np.asarray(tenfit.ad, dtype=np.float32)

    fa = np.nan_to_num(fa, nan=0.0, posinf=1.0, neginf=0.0)
    fa = np.clip(fa, 0.0, 1.0)
    md = np.nan_to_num(md, nan=0.0, posinf=0.0, neginf=0.0)
    ad = np.nan_to_num(ad, nan=0.0, posinf=0.0, neginf=0.0)

    fa[~mask] = 0.0
    md[~mask] = 0.0
    ad[~mask] = 0.0

    return {"FA": fa, "MD": md, "AD": ad}


# ============================================================
# Parameter-map normalization + metrics
# ============================================================

def nrmse_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """
    sqrt(sum((pred-gt)^2) / sum(gt^2)) inside the brain mask.
    """
    m = np.asarray(mask, dtype=bool)
    p = np.asarray(pred, dtype=np.float64)[m]
    g = np.asarray(gt, dtype=np.float64)[m]

    valid = np.isfinite(p) & np.isfinite(g)
    p = p[valid]
    g = g[valid]

    if g.size == 0:
        return float("nan")

    denom = np.sum(g * g, dtype=np.float64)
    if denom <= 0:
        return float("nan")

    return float(
        np.sqrt(np.sum((p - g) ** 2, dtype=np.float64) / denom)
    )


def psnr_masked_unit_range(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    PSNR after FA/MD/AD have all been mapped to [0,1].

    Therefore data_range is FIXED to 1.0 for every map, subject and method.
    """
    m = np.asarray(mask, dtype=bool)
    p = np.asarray(pred, dtype=np.float64)[m]
    g = np.asarray(gt, dtype=np.float64)[m]

    valid = np.isfinite(p) & np.isfinite(g)
    p = p[valid]
    g = g[valid]

    if g.size == 0:
        return float("nan")

    mse = float(np.mean((p - g) ** 2))
    if mse <= 0:
        return float("inf")

    return float(10.0 * np.log10(1.0 / mse))


def bbox_from_mask(mask_2d: np.ndarray):
    coords = np.argwhere(mask_2d)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(r1), int(c0), int(c1)


def ssim_subject_masked_unit_range(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    win_size: int,
) -> float:
    """
    Slice-wise SSIM with data_range=1.0.

    Important difference from the old evaluator:
    skimage returns the full SSIM map and we average ONLY its brain-mask
    pixels.  Zero background is therefore not counted as perfect agreement.
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    values = []

    for sid in range(gt.shape[0]):
        m = mask[sid]
        bbox = bbox_from_mask(m)
        if bbox is None:
            continue

        r0, r1, c0, c1 = bbox
        pad = max(0, int(win_size) // 2)

        r0 = max(0, r0 - pad)
        r1 = min(gt.shape[1], r1 + pad)
        c0 = max(0, c0 - pad)
        c1 = min(gt.shape[2], c1 + pad)

        g = gt[sid, r0:r1, c0:c1].copy()
        p = pred[sid, r0:r1, c0:c1].copy()
        mm = m[r0:r1, c0:c1]

        min_hw = min(g.shape)
        if min_hw < 3:
            continue

        w = min(int(win_size), int(min_hw))
        if w % 2 == 0:
            w -= 1
        if w < 3:
            continue

        # Outside the brain we set both images to zero only to provide a valid
        # local neighborhood for SSIM near the brain boundary. Those pixels
        # are NOT included in the final average.
        g[~mm] = 0.0
        p[~mm] = 0.0

        _, ssim_map = SSIM(
            g,
            p,
            data_range=1.0,
            win_size=w,
            full=True,
        )

        brain_scores = np.asarray(ssim_map, dtype=np.float64)[mm]
        brain_scores = brain_scores[np.isfinite(brain_scores)]

        if brain_scores.size > 0:
            values.append(float(np.mean(brain_scores)))

    return float(np.mean(values)) if values else float("nan")


def normalize_parameter_map(
    map_name: str,
    array: np.ndarray,
    md_upper: float,
    ad_upper: float,
) -> np.ndarray:
    """
    Map FA/MD/AD to a common [0,1] comparison domain.

    FA:
        already dimensionless; clip to [0,1].

    MD / AD:
        x_norm = clip(x / upper, 0, 1)

    Crucially, the SAME md_upper/ad_upper is used for:
        GT
        no-refill prediction
        refill prediction
        every subject

    Pred and GT are never normalized independently.
    """
    x = np.asarray(array, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if map_name == "FA":
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    if map_name == "MD":
        upper = float(md_upper)
    elif map_name == "AD":
        upper = float(ad_upper)
    else:
        raise ValueError(f"Unsupported parameter map={map_name}")

    if not np.isfinite(upper) or upper <= 0:
        raise ValueError(f"Invalid {map_name} upper normalization value={upper}")

    return np.clip(x / upper, 0.0, 1.0).astype(np.float32)


def evaluate_normalized_maps(
    pred_maps_norm: Dict[str, np.ndarray],
    gt_maps_norm: Dict[str, np.ndarray],
    mask: np.ndarray,
    ssim_win_size: int,
) -> Dict[str, Dict[str, float]]:
    result = OrderedDict()

    for name in MAP_NAMES:
        result[name] = {
            "nrmse": nrmse_masked(
                pred_maps_norm[name],
                gt_maps_norm[name],
                mask,
            ),
            "ssim": ssim_subject_masked_unit_range(
                pred_maps_norm[name],
                gt_maps_norm[name],
                mask,
                win_size=ssim_win_size,
            ),
            "psnr": psnr_masked_unit_range(
                pred_maps_norm[name],
                gt_maps_norm[name],
                mask,
            ),
        }

    return result


def resolve_upper_from_values(
    values: Sequence[np.ndarray],
    percentile: float,
    explicit_upper: float,
    name: str,
) -> float:
    """
    Resolve one GLOBAL normalization upper bound.

    If explicit_upper > 0, it is used exactly.
    Otherwise the specified percentile is computed from all masked GT values
    in the test set. This scale is then frozen and shared by both prediction
    modes and all subjects.
    """
    if float(explicit_upper) > 0:
        return float(explicit_upper)

    if not values:
        raise ValueError(f"No GT values collected for {name}")

    x = np.concatenate(
        [np.asarray(v, dtype=np.float32).reshape(-1) for v in values],
        axis=0,
    )
    x = x[np.isfinite(x)]
    x = x[x >= 0]

    if x.size == 0:
        raise ValueError(f"No finite nonnegative GT values found for {name}")

    upper = float(np.percentile(x, float(percentile)))
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError(
            f"Invalid {name} normalization upper={upper} "
            f"from percentile={percentile}"
        )

    return upper


def load_normalization_json(path: str) -> Optional[Dict[str, float]]:
    path = str(path or "").strip()
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Normalization JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    for key in ("md_upper", "ad_upper"):
        if key not in payload:
            raise KeyError(f"{p}: missing key {key}")

    return {
        "md_upper": float(payload["md_upper"]),
        "ad_upper": float(payload["ad_upper"]),
        "norm_percentile": float(payload.get("norm_percentile", float("nan"))),
    }


def finite_mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan"), float("nan"), 0

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return mean, std, int(x.size)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="HCP 10->90 Ours: NO-REFILL + ODDC/REFILL DTI evaluation",
    )

    p.add_argument(
        "--direction_task",
        type=str,
        required=True,
        choices=SUPPORTED_TASKS,
    )
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--result_path", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--strict_load", type=int, default=1, choices=[0, 1])

    # Must match training/model checkpoint.
    p.add_argument("--mamba_scans", type=str, default="hv", choices=["h", "hv", "hv_flip"])
    p.add_argument("--mamba_d_state", type=int, default=16)
    p.add_argument("--mamba_d_conv", type=int, default=4)
    p.add_argument("--mamba_expand", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--graph_node_dim", type=int, default=-1, help="<=0: auto-infer from checkpoint; supplied training script uses 8")
    p.add_argument("--graph_layers", type=int, default=-1, help="<=0: auto-infer from checkpoint when possible; training default is 1")
    p.add_argument("--graph_topk", type=int, default=6, help="Must match training; current HCP QGSR test log uses 6")
    p.add_argument("--graph_tau", type=float, default=0.15, help="Must match training; current HCP QGSR test log uses 0.15")

    p.add_argument("--fit_method", type=str, default="WLS", choices=["WLS", "OLS", "LS"])
    p.add_argument("--min_signal", type=float, default=1e-6)
    p.add_argument("--clip_max", type=float, default=1.0)
    p.add_argument("--ssim_win_size", type=int, default=7)

    # --------------------------------------------------------
    # DTI parameter normalization.
    #
    # Recommended workflow:
    #   1) Run Ours once without --norm_json.
    #      The script estimates one robust GT scale and writes
    #      dti_normalization.json.
    #   2) Reuse that SAME JSON for every baseline:
    #      --norm_json /.../dti_normalization.json
    #
    # This guarantees identical [0,1] scaling across methods.
    # --------------------------------------------------------
    p.add_argument(
        "--norm_json",
        type=str,
        default="",
        help="Existing normalization JSON to reuse across methods.",
    )
    p.add_argument(
        "--norm_percentile",
        type=float,
        default=99.5,
        help="Global GT percentile used when md/ad upper values are not supplied.",
    )
    p.add_argument(
        "--md_upper",
        type=float,
        default=-1.0,
        help="Fixed physical MD upper bound; <=0 means estimate from global GT.",
    )
    p.add_argument(
        "--ad_upper",
        type=float,
        default=-1.0,
        help="Fixed physical AD upper bound; <=0 means estimate from global GT.",
    )

    p.add_argument("--save_dwi", type=int, default=0, choices=[0, 1])
    p.add_argument("--save_maps", type=int, default=1, choices=[0, 1])
    p.add_argument("--log_name", type=str, default="dti_eval_logger.txt")

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    start_time = time.time()
    opt = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    seed_everything(opt.seed)

    result_root = Path(opt.result_path)
    result_root.mkdir(parents=True, exist_ok=True)

    logger = get_logger(str(result_root / opt.log_name))
    logger.info(opt)

    shell = int(opt.direction_task.rsplit("_b", 1)[1])
    fixed_indices = parse_fixed_indices(opt.direction_task)
    n_channels = len(fixed_indices)

    if opt.direction_task not in DIRECTION_TASKS:
        raise ValueError(f"Unknown direction task {opt.direction_task}")

    if n_channels != 10:
        raise ValueError(
            f"This Table-II evaluator is for 10->90 only, got K={n_channels}"
        )

    # ========================================================
    # Resolve model structural configuration from checkpoint.
    # graph_node_dim changes tensor shapes and therefore can be
    # detected reliably. topk/tau must still match the training run.
    # ========================================================
    checkpoint_state = read_checkpoint_state_dict(opt.model_path)
    detected_node_dim, detected_graph_layers = (
        infer_graph_config_from_checkpoint(checkpoint_state)
    )

    if int(opt.graph_node_dim) <= 0:
        if detected_node_dim is None:
            raise RuntimeError(
                "Could not infer graph_node_dim from checkpoint. "
                "Please pass --graph_node_dim explicitly."
            )
        graph_node_dim = int(detected_node_dim)
    else:
        graph_node_dim = int(opt.graph_node_dim)

    if int(opt.graph_layers) <= 0:
        graph_layers = (
            int(detected_graph_layers)
            if detected_graph_layers is not None
            else 1
        )
    else:
        graph_layers = int(opt.graph_layers)

    if (
        detected_node_dim is not None
        and graph_node_dim != int(detected_node_dim)
    ):
        raise ValueError(
            f"graph_node_dim mismatch: requested={graph_node_dim}, "
            f"checkpoint={detected_node_dim}. "
            f"Use --graph_node_dim -1 for auto."
        )

    if (
        detected_graph_layers is not None
        and graph_layers != int(detected_graph_layers)
    ):
        raise ValueError(
            f"graph_layers mismatch: requested={graph_layers}, "
            f"checkpoint={detected_graph_layers}. "
            f"Use --graph_layers -1 for auto."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("=" * 86)
    logger.info("HCP Ours -> 90 DWI -> DTI FA/MD/AD")
    logger.info(f"direction_task     : {opt.direction_task}")
    logger.info(f"shell              : b={shell}")
    logger.info(f"fixed_indices      : {fixed_indices}")
    logger.info(f"data_path          : {opt.data_path}")
    logger.info(f"model_path         : {opt.model_path}")
    logger.info(f"device             : {device}")
    logger.info(f"DTI fit            : {opt.fit_method}")
    logger.info("DTI evaluation     : BOTH raw and model-integrated ODDC/refill; refill is the final downstream output")
    logger.info("FA normalization   : native [0,1]")
    logger.info(
        "MD/AD normalization: shared global mapping -> [0,1]; "
        "same scale for GT/no-refill/refill"
    )
    logger.info("PSNR data_range    : fixed 1.0")
    logger.info("SSIM data_range    : fixed 1.0")
    logger.info("SSIM aggregation   : brain-mask pixels of full SSIM map")
    logger.info(
        f"checkpoint detected: graph_node_dim={detected_node_dim}, "
        f"graph_layers={detected_graph_layers}"
    )
    logger.info(
        "graph config used  : "
        f"node_dim={graph_node_dim}, layers={graph_layers}, "
        f"topk={opt.graph_topk}, tau={opt.graph_tau}"
    )
    logger.info("=" * 86)

    dataset = Dataset_2D(
        opt.data_path,
        data_type="test",
        return_bvec=True,
        direction_task=opt.direction_task,
        fixed_indices=fixed_indices,
        skip_empty_slices=True,
    )
    subject_groups = build_subject_groups(dataset.examples)

    if not subject_groups:
        raise RuntimeError("No test subjects found.")

    logger.info(f"test slices        : {len(dataset)}")
    logger.info(f"test subjects      : {len(subject_groups)}")

    # ========================================================
    # PASS 1: GT DTI maps only.
    #
    # We first establish ONE normalization scale for MD/AD.
    # This avoids subject-specific gt.max() and avoids prediction-specific
    # min-max normalization.  The generated JSON should be reused unchanged
    # for every baseline in Table II.
    # ========================================================
    normalization_from_file = load_normalization_json(opt.norm_json)

    md_gt_values: List[np.ndarray] = []
    ad_gt_values: List[np.ndarray] = []

    logger.info("")
    logger.info("PASS 1/2: fitting GT DTI maps and resolving normalization.")

    for subject_no, group in enumerate(subject_groups, start=1):
        subject_id = group["subject_id"]

        gt_90, mask, bvecs_90 = load_gt_one_subject(
            dataset=dataset,
            row_indices=group["row_indices"],
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            subject_name=subject_id,
        )

        gt_maps_raw = fit_dti_maps(
            signal_schw=gt_90,
            mask_shw=mask,
            bvecs_90=bvecs_90,
            shell=shell,
            fit_method=opt.fit_method,
            min_signal=opt.min_signal,
            clip_max=opt.clip_max,
        )

        subject_dir = result_root / "subjects" / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            subject_dir / "slice_ids.npy",
            np.asarray(group["slice_ids"], dtype=np.int32),
        )
        np.save(subject_dir / "mask.npy", mask.astype(np.uint8))
        np.save(
            subject_dir / "bvecs_90.npy",
            bvecs_90.astype(np.float32),
        )

        # Save raw GT maps. They retain DIPY physical diffusivity units.
        for map_name in MAP_NAMES:
            np.save(
                subject_dir / f"GT_{map_name}_raw.npy",
                gt_maps_raw[map_name].astype(np.float32),
            )

        if normalization_from_file is None:
            md_vals = gt_maps_raw["MD"][mask]
            ad_vals = gt_maps_raw["AD"][mask]

            md_vals = md_vals[
                np.isfinite(md_vals) & (md_vals >= 0)
            ].astype(np.float32, copy=False)
            ad_vals = ad_vals[
                np.isfinite(ad_vals) & (ad_vals >= 0)
            ].astype(np.float32, copy=False)

            md_gt_values.append(md_vals)
            ad_gt_values.append(ad_vals)

        logger.info(
            f"GT [{subject_no}/{len(subject_groups)}] {subject_id}: "
            f"slices={gt_90.shape[0]}"
        )

        del gt_90, mask, bvecs_90, gt_maps_raw
        gc.collect()

    if normalization_from_file is not None:
        md_upper = float(normalization_from_file["md_upper"])
        ad_upper = float(normalization_from_file["ad_upper"])
        norm_source = f"loaded from {opt.norm_json}"
    else:
        md_upper = resolve_upper_from_values(
            md_gt_values,
            percentile=opt.norm_percentile,
            explicit_upper=opt.md_upper,
            name="MD",
        )
        ad_upper = resolve_upper_from_values(
            ad_gt_values,
            percentile=opt.norm_percentile,
            explicit_upper=opt.ad_upper,
            name="AD",
        )
        norm_source = (
            f"global GT percentile={opt.norm_percentile} "
            f"(unless explicit upper supplied)"
        )

    del md_gt_values, ad_gt_values
    gc.collect()

    normalization_payload = {
        "direction_task": opt.direction_task,
        "shell": int(shell),
        "map_range": {
            "FA": [0.0, 1.0],
            "MD": [0.0, float(md_upper)],
            "AD": [0.0, float(ad_upper)],
        },
        "md_upper": float(md_upper),
        "ad_upper": float(ad_upper),
        "norm_percentile": float(opt.norm_percentile),
        "source": norm_source,
        "mapping": {
            "FA": "clip(x,0,1)",
            "MD": "clip(x / md_upper,0,1)",
            "AD": "clip(x / ad_upper,0,1)",
        },
        "important": (
            "Use this exact JSON for all compared methods. "
            "GT and predictions must share the same normalization."
        ),
    }

    norm_json_path = result_root / "dti_normalization.json"
    with norm_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            normalization_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("")
    logger.info("Resolved DTI normalization:")
    logger.info(f"  FA : [0, 1]")
    logger.info(f"  MD : [0, {md_upper:.10g}] -> [0,1]")
    logger.info(f"  AD : [0, {ad_upper:.10g}] -> [0,1]")
    logger.info(f"  saved: {norm_json_path}")

    # ========================================================
    # Build/load model only after normalization is fixed.
    # ========================================================
    model = CNN_2D(
        n_channels=n_channels,
        n_out=90,
        res_hiddens=(128, 256),
        mamba_scans=opt.mamba_scans,
        d_state=opt.mamba_d_state,
        d_conv=opt.mamba_d_conv,
        expand=opt.mamba_expand,
        dropout=opt.dropout,
        graph_node_dim=graph_node_dim,
        graph_layers=graph_layers,
        graph_topk=opt.graph_topk,
        graph_tau=opt.graph_tau,
        fixed_indices=fixed_indices,
    ).to(device)

    logger.info(f"Model parameters    : {get_num_parameters(model)}")

    load_state_dict_compatible(
        model,
        checkpoint_path=opt.model_path,
        strict=bool(opt.strict_load),
    )
    model.eval()
    logger.info("Checkpoint loaded successfully.")

    # ========================================================
    # PASS 2: one network forward, then split into:
    #   1) no_refill = raw network output
    #   2) refill    = ODDC, pred[:, observed_idx] <- x
    #
    # Both are fitted independently with DIPY.
    # ========================================================
    logger.info("")
    logger.info("PASS 2/2: model inference + NO-REFILL/REFILL DTI evaluation.")

    records: List[Dict] = []

    subject_names = [g["subject_id"] for g in subject_groups]
    map_names = list(MAP_NAMES)
    metric_names = ["nrmse", "ssim", "psnr"]

    # Two requested arrays:
    #   [subject, parameter_map, metric]
    metrics_no_refill = np.full(
        (len(subject_groups), len(MAP_NAMES), len(metric_names)),
        np.nan,
        dtype=np.float64,
    )
    metrics_refill = np.full_like(metrics_no_refill, np.nan)

    for subject_index, group in enumerate(subject_groups):
        subject_no = subject_index + 1
        subject_id = group["subject_id"]
        subject_dir = result_root / "subjects" / subject_id

        (
            pred_raw_90,
            pred_refill_90,
            gt_90_unused,
            mask,
            bvecs_90,
        ) = infer_one_subject(
            model=model,
            dataset=dataset,
            row_indices=group["row_indices"],
            fixed_indices=fixed_indices,
            device=device,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            subject_name=subject_id,
        )

        # Load GT maps fitted in PASS 1.
        gt_maps_raw = {
            map_name: np.load(
                subject_dir / f"GT_{map_name}_raw.npy"
            ).astype(np.float32)
            for map_name in MAP_NAMES
        }

        # The ODDC/refilled branch is the default final output used for the
        # paper's downstream DTI analysis. The raw branch is retained only
        # for the script's original diagnostic no-refill comparison.
        pred_maps_raw = fit_dti_maps(
            signal_schw=pred_raw_90,
            mask_shw=mask,
            bvecs_90=bvecs_90,
            shell=shell,
            fit_method=opt.fit_method,
            min_signal=opt.min_signal,
            clip_max=opt.clip_max,
        )

        pred_maps_refill = fit_dti_maps(
            signal_schw=pred_refill_90,
            mask_shw=mask,
            bvecs_90=bvecs_90,
            shell=shell,
            fit_method=opt.fit_method,
            min_signal=opt.min_signal,
            clip_max=opt.clip_max,
        )

        # ----------------------------------------------------
        # Shared normalization: same GT-derived scale for BOTH modes.
        # ----------------------------------------------------
        gt_maps_norm = {
            name: normalize_parameter_map(
                name,
                gt_maps_raw[name],
                md_upper=md_upper,
                ad_upper=ad_upper,
            )
            for name in MAP_NAMES
        }

        pred_maps_norm = {
            name: normalize_parameter_map(
                name,
                pred_maps_raw[name],
                md_upper=md_upper,
                ad_upper=ad_upper,
            )
            for name in MAP_NAMES
        }

        pred_maps_refill_norm = {
            name: normalize_parameter_map(
                name,
                pred_maps_refill[name],
                md_upper=md_upper,
                ad_upper=ad_upper,
            )
            for name in MAP_NAMES
        }

        no_refill_metrics = evaluate_normalized_maps(
            pred_maps_norm=pred_maps_norm,
            gt_maps_norm=gt_maps_norm,
            mask=mask,
            ssim_win_size=opt.ssim_win_size,
        )

        refill_metrics = evaluate_normalized_maps(
            pred_maps_norm=pred_maps_refill_norm,
            gt_maps_norm=gt_maps_norm,
            mask=mask,
            ssim_win_size=opt.ssim_win_size,
        )

        logger.info(
            f"[{subject_no}/{len(subject_groups)}] {subject_id}: "
            f"raw={pred_raw_90.shape}, refill={pred_refill_90.shape}"
        )

        for mode_name, metric_dict, target_array in (
            ("no_refill", no_refill_metrics, metrics_no_refill),
            ("refill", refill_metrics, metrics_refill),
        ):
            logger.info(f"    [{mode_name.upper()}]")

            for map_index, map_name in enumerate(MAP_NAMES):
                row = {
                    "subject_id": subject_id,
                    "file_name": group["file_name"],
                    "file_path": group["file_path"],
                    "shell": int(shell),
                    "direction_task": opt.direction_task,
                    "mode": mode_name,
                    "map": map_name,
                    "nrmse": float(metric_dict[map_name]["nrmse"]),
                    "ssim": float(metric_dict[map_name]["ssim"]),
                    "psnr": float(metric_dict[map_name]["psnr"]),
                }
                records.append(row)

                for metric_index, metric_name in enumerate(metric_names):
                    target_array[
                        subject_index,
                        map_index,
                        metric_index,
                    ] = row[metric_name]

                logger.info(
                    f"        {map_name}: "
                    f"nRMSE={row['nrmse']:.6f}, "
                    f"SSIM={row['ssim']:.6f}, "
                    f"PSNR={row['psnr']:.6f} dB"
                )

        # ----------------------------------------------------
        # Save two DWI groups if requested.
        # ----------------------------------------------------
        if bool(opt.save_dwi):
            np.save(
                subject_dir / "pred_90_no_refill.npy",
                pred_raw_90.astype(np.float32, copy=False),
            )
            np.save(
                subject_dir / "pred_90_refill.npy",
                pred_refill_90.astype(np.float32, copy=False),
            )
            np.save(
                subject_dir / "gt_90.npy",
                gt_90_unused.astype(np.float32, copy=False),
            )

        # ----------------------------------------------------
        # Save parameter maps:
        #   raw physical maps + normalized maps.
        # This makes later qualitative FA/MD/AD figures reproducible.
        # ----------------------------------------------------
        if bool(opt.save_maps):
            for map_name in MAP_NAMES:
                np.save(
                    subject_dir / f"Pred_no_refill_{map_name}_raw.npy",
                    pred_maps_raw[map_name].astype(np.float32),
                )
                np.save(
                    subject_dir / f"Pred_refill_{map_name}_raw.npy",
                    pred_maps_refill[map_name].astype(np.float32),
                )

                np.save(
                    subject_dir / f"GT_{map_name}.npy",
                    gt_maps_norm[map_name].astype(np.float32),
                )
                np.save(
                    subject_dir / f"Pred_no_refill_{map_name}.npy",
                    pred_maps_norm[map_name].astype(np.float32),
                )
                np.save(
                    subject_dir / f"Pred_refill_{map_name}.npy",
                    pred_maps_refill_norm[map_name].astype(np.float32),
                )

        del (
            pred_raw_90,
            pred_refill_90,
            gt_90_unused,
            mask,
            bvecs_90,
            gt_maps_raw,
            pred_maps_raw,
            pred_maps_refill,
            gt_maps_norm,
            pred_maps_norm,
            pred_maps_refill_norm,
            no_refill_metrics,
            refill_metrics,
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========================================================
    # Save exactly TWO metric arrays requested by the user.
    #
    # Shape:
    #   [N_subjects, 3 maps, 3 metrics]
    #
    # map order:
    #   FA, MD, AD
    #
    # metric order:
    #   nRMSE, SSIM, PSNR
    # ========================================================
    no_refill_npy = result_root / "metrics_no_refill.npy"
    refill_npy = result_root / "metrics_refill.npy"

    np.save(no_refill_npy, metrics_no_refill)
    np.save(refill_npy, metrics_refill)

    np.savez(
        result_root / "metrics_both.npz",
        no_refill=metrics_no_refill,
        refill=metrics_refill,
        subjects=np.asarray(subject_names),
        map_names=np.asarray(map_names),
        metric_names=np.asarray(metric_names),
    )

    # ========================================================
    # CSV
    # ========================================================
    csv_path = result_root / "dti_subject_metrics_both.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "file_name",
                "file_path",
                "shell",
                "direction_task",
                "mode",
                "map",
                "nrmse",
                "ssim",
                "psnr",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    # ========================================================
    # Summary: mean +/- sample std across subjects.
    # ========================================================
    summary = OrderedDict()

    summary_arrays = {}

    for mode_name, metric_array in (
        ("no_refill", metrics_no_refill),
        ("refill", metrics_refill),
    ):
        summary[mode_name] = OrderedDict()

        # [map, metric, mean/std]
        summary_array = np.full(
            (len(MAP_NAMES), len(metric_names), 2),
            np.nan,
            dtype=np.float64,
        )

        for map_index, map_name in enumerate(MAP_NAMES):
            summary[mode_name][map_name] = OrderedDict()

            for metric_index, metric_name in enumerate(metric_names):
                mean, std, n = finite_mean_std(
                    metric_array[:, map_index, metric_index]
                )

                summary[mode_name][map_name][metric_name] = {
                    "mean": mean,
                    "std": std,
                    "n": n,
                }

                summary_array[map_index, metric_index, 0] = mean
                summary_array[map_index, metric_index, 1] = std

        summary_arrays[mode_name] = summary_array
        np.save(
            result_root / f"summary_{mode_name}.npy",
            summary_array,
        )

    # ========================================================
    # JSON metadata
    # ========================================================
    payload = {
        "method": "Ours",
        "direction_task": opt.direction_task,
        "shell": int(shell),
        "fixed_indices": [int(v) for v in fixed_indices],
        "num_input_directions": 10,
        "num_output_directions": 90,
        "mode_definition": {
            "no_refill": (
                "Raw network 90-direction output. "
                "No observed direction is overwritten."
            ),
            "refill": (
                "ODDC/data consistency: for each observed fixed direction, "
                "the model output is replaced by the acquired input signal "
                "before DTI fitting."
            ),
        },
        "array_layout": {
            "metrics_no_refill.npy": "[subject, map, metric]",
            "metrics_refill.npy": "[subject, map, metric]",
            "map_order": map_names,
            "metric_order": metric_names,
            "subject_order": subject_names,
            "summary_shape": "[map, metric, mean_or_std]",
            "summary_last_axis": ["mean", "std"],
        },
        "dti_fit_method": opt.fit_method,
        "signal_assumption": "normalized S/S0 with synthetic normalized b0=1",
        "normalization": normalization_payload,
        "metric_protocol": {
            "nrmse": (
                "sqrt(sum((pred-gt)^2)/sum(gt^2)) inside brain mask "
                "after shared normalization"
            ),
            "ssim": (
                "slice-wise SSIM with data_range=1.0; "
                "average only brain-mask pixels of the full SSIM map"
            ),
            "psnr": (
                "masked MSE with fixed data_range=1.0 "
                "after shared normalization"
            ),
            "aggregation": "subject-level metrics, then mean +/- sample std",
        },
        "checkpoint_graph_config": {
            "detected_graph_node_dim": detected_node_dim,
            "detected_graph_layers": detected_graph_layers,
            "used_graph_node_dim": int(graph_node_dim),
            "used_graph_layers": int(graph_layers),
            "used_graph_topk": int(opt.graph_topk),
            "used_graph_tau": float(opt.graph_tau),
        },
        "summary": summary,
        "records": records,
    }

    json_path = result_root / "dti_summary_both.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Two LaTeX versions.
    # ========================================================
    combined_lines = []

    for mode_name, mode_title in (
        ("no_refill", "NO REFILL"),
        ("refill", "REFILL / ODDC"),
    ):
        logger.info("=" * 86)
        logger.info(
            f"FINAL TABLE-II VALUES — {opt.direction_task} — {mode_title}"
        )
        logger.info("=" * 86)

        latex_lines = []

        for map_name in MAP_NAMES:
            one = summary[mode_name][map_name]

            nrmse_m = one["nrmse"]["mean"]
            nrmse_s = one["nrmse"]["std"]
            ssim_m = one["ssim"]["mean"]
            ssim_s = one["ssim"]["std"]
            psnr_m = one["psnr"]["mean"]
            psnr_s = one["psnr"]["std"]

            logger.info(
                f"{map_name}: "
                f"nRMSE={nrmse_m:.4f} +/- {nrmse_s:.4f}; "
                f"SSIM={ssim_m:.4f} +/- {ssim_s:.4f}; "
                f"PSNR={psnr_m:.4f} +/- {psnr_s:.4f}"
            )

            latex_lines.append(
                f"{map_name} & Ours & "
                f"${nrmse_m:.4f} \\pm {nrmse_s:.4f}$ & "
                f"${ssim_m:.4f} \\pm {ssim_s:.4f}$ & "
                f"${psnr_m:.4f} \\pm {psnr_s:.4f}$ \\\\"
            )

        latex_path = (
            result_root
            / f"latex_ours_{opt.direction_task}_{mode_name}.txt"
        )
        latex_path.write_text(
            "\n".join(latex_lines) + "\n",
            encoding="utf-8",
        )

        combined_lines.append(f"% ===== {mode_title} =====")
        combined_lines.extend(latex_lines)
        combined_lines.append("")

    combined_path = (
        result_root / f"latex_ours_{opt.direction_task}_both.txt"
    )
    combined_path.write_text(
        "\n".join(combined_lines),
        encoding="utf-8",
    )

    elapsed = time.time() - start_time

    logger.info("")
    logger.info(f"Normalization JSON : {norm_json_path}")
    logger.info(f"No-refill array    : {no_refill_npy}")
    logger.info(f"Refill array       : {refill_npy}")
    logger.info(f"CSV                : {csv_path}")
    logger.info(f"JSON               : {json_path}")
    logger.info(f"LaTeX              : {combined_path}")
    logger.info(f"Finished in        : {elapsed:.2f} s")

    print("\nDTI evaluation completed.")
    print("Two independent result arrays were produced:")
    print(f"  NO REFILL : {no_refill_npy}")
    print(f"  REFILL    : {refill_npy}")
    print("Array shape = [subject, map, metric]")
    print("Map order   = [FA, MD, AD]")
    print("Metric order= [nRMSE, SSIM, PSNR]")
    print(f"Normalization: {norm_json_path}")
    print(f"Summary JSON : {json_path}")
    print(f"LaTeX        : {combined_path}")


if __name__ == "__main__":
    main()
