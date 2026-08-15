import argparse
import csv
import gc
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import normalized_root_mse as NRMSE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(".")
sys.path.append("./..")

from dataset_2d import Dataset_2D, DIRECTION_TASKS, parse_fixed_indices
from model import CNN_2D


def parse_int_list(text: str) -> Optional[List[int]]:
    text = str(text or "").strip()
    if not text:
        return None
    text = text.replace("[", "").replace("]", "").replace(";", ",")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def get_logger(path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger(os.path.abspath(path))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s:\t %(message)s"
        )

        file_handler = logging.FileHandler(path, mode="a+", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_num_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def squeeze_extra_dim(tensor):
    if torch.is_tensor(tensor) and tensor.ndim == 5 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor


def unpack_batch(batch):
    """Support Dataset_2D batches with 3, 4 or 5 returned objects."""
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


def normalize_bvec(bvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return bvec / torch.clamp(
        torch.linalg.norm(bvec, dim=-1, keepdim=True), min=eps
    )


def ensure_bvec_batch(
    bvec,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bvec is None:
        raise ValueError("bvec_out is required for the SGR head.")

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
                f"bvec batch mismatch: got {tuple(bvec.shape)}, "
                f"expected B={batch_size}"
            )
    else:
        raise ValueError(
            f"bvec_out must be [90,3] or [B,90,3], got {tuple(bvec.shape)}"
        )

    if bvec.shape[1] != 90 or bvec.shape[-1] != 3:
        raise ValueError(f"Expected bvec_out [B,90,3], got {tuple(bvec.shape)}")

    return normalize_bvec(bvec)


def load_external_bvec(path: str) -> Optional[torch.Tensor]:
    path = str(path or "").strip()
    if not path:
        return None

    if path.endswith(".npy"):
        array = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        key = "bvecs_90" if "bvecs_90" in data else list(data.keys())[0]
        array = data[key]
    else:
        raise ValueError("--bvec_path only supports .npy or .npz files.")

    array = np.asarray(array)
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(f"Loaded bvec must be [O,3], got {array.shape}")

    if array.shape[0] == 91:
        array = array[1:91]
    elif array.shape[0] > 90:
        array = array[:90]

    if array.shape[0] != 90:
        raise ValueError(f"Expected 90 b-vectors, got {array.shape[0]}")

    return torch.as_tensor(array, dtype=torch.float32)


def broadcast_mask(mask: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Convert a dataset mask to [B,1,H,W]."""
    mask = squeeze_extra_dim(mask)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    if mask.shape[-2:] != output.shape[-2:]:
        mask = torch.nn.functional.interpolate(
            mask.float(), size=output.shape[-2:], mode="nearest"
        )

    if mask.shape[1] == 1:
        return mask
    return mask.max(dim=1, keepdim=True).values



def load_state_dict_compatible(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool,
) -> torch.nn.Module:
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

    model_keys = model.state_dict().keys()
    model_has_module = any(k.startswith("module.") for k in model_keys)
    ckpt_has_module = any(k.startswith("module.") for k in checkpoint.keys())

    if ckpt_has_module and not model_has_module:
        checkpoint = {
            k[len("module."):]: v
            for k, v in checkpoint.items()
            if k.startswith("module.")
        }
    elif not ckpt_has_module and model_has_module:
        checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}

    incompatible = model.load_state_dict(checkpoint, strict=bool(strict))
    if not strict:
        print(f"Missing keys   : {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")

    return model


def build_subject_groups(examples: Sequence[Tuple[str, int]]) -> List[Dict]:
    """Group dataset row indices by HDF5 file and sort by slice_id."""
    groups: "OrderedDict[str, Dict]" = OrderedDict()

    for row_index, example in enumerate(examples):
        if len(example) < 2:
            raise ValueError(
                "Dataset_2D.examples must contain (file_path, slice_id)."
            )

        file_path = str(example[0])
        slice_id = int(example[1])

        if file_path not in groups:
            groups[file_path] = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "items": [],
            }

        groups[file_path]["items"].append((slice_id, int(row_index)))

    output: List[Dict] = []
    for group in groups.values():
        items = sorted(group["items"], key=lambda item: item[0])
        output.append(
            {
                "file_path": group["file_path"],
                "file_name": group["file_name"],
                "slice_ids": [int(item[0]) for item in items],
                "row_indices": [int(item[1]) for item in items],
            }
        )

    return output


def one_subject_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    data_range: float,
    ssim_win_size: int,
) -> Tuple[float, float, float, float]:
    """
    Reproduce the original script's metric convention.

    pred/gt: [H,W,S,90]

    MSE, nRMSE and PSNR are computed on the entire subject volume.
    SSIM is computed for each [H,W,90] slice and then averaged over S.
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 4 or pred.shape[-1] != 90:
        raise ValueError(f"Expected [H,W,S,90], got {pred.shape}")

    mse = float(MSE(pred, gt))
    nrmse = float(NRMSE(pred, gt))
    psnr = float(PSNR(pred, gt, data_range=float(data_range)))

    ssim_values: List[float] = []
    for slice_index in range(gt.shape[2]):
        pred_slice = pred[:, :, slice_index, :]
        gt_slice = gt[:, :, slice_index, :]

        actual_win = min(int(ssim_win_size), int(min(pred_slice.shape)))
        if actual_win % 2 == 0:
            actual_win -= 1
        if actual_win < 3:
            continue

        value = SSIM(
            pred_slice,
            gt_slice,
            data_range=float(data_range),
            win_size=actual_win,
            channel_axis=None,
        )
        if np.isfinite(value):
            ssim_values.append(float(value))

    ssim = float(np.mean(ssim_values)) if ssim_values else float("nan")
    return mse, nrmse, psnr, ssim


def infer_one_subject(
    model: torch.nn.Module,
    dataset: Dataset_2D,
    row_indices: Sequence[int],
    fixed_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    external_bvec: Optional[torch.Tensor],
    subject_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Infer one subject once and return both outputs:

        raw_pred_slices    : [S,90,H,W], without observed-direction refill
        refill_pred_slices : [S,90,H,W], with observed directions replaced by x
        gt_slices          : [S,90,H,W]
        mask_slices        : [S,H,W]

    Both raw and refilled predictions are later evaluated over all 90 directions.
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

    raw_pred_batches: List[np.ndarray] = []
    refill_pred_batches: List[np.ndarray] = []
    gt_batches: List[np.ndarray] = []
    mask_batches: List[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        pbar = tqdm(
            loader,
            total=len(loader),
            desc=f"Subject {subject_id}",
            ncols=110,
            leave=False,
        )

        for batch in pbar:
            x, y_tg, mask, bvec_out = unpack_batch(batch)

            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y_tg = y_tg.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)

            if bvec_out is None:
                if external_bvec is None:
                    raise ValueError(
                        "Dataset_2D did not return bvec_out and --bvec_path is empty."
                    )
                bvec_out = external_bvec

            bvec_out = ensure_bvec_batch(
                bvec_out,
                batch_size=x.shape[0],
                device=device,
                dtype=x.dtype,
            )

            # One network forward pass. The data-consistency layer is now
            # part of the model, which directly returns both outputs.
            raw_pred, refill_pred = model(x, bvec_out=bvec_out)

            mask_4d = broadcast_mask(mask, raw_pred)

            raw_pred_batches.append(
                raw_pred.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            refill_pred_batches.append(
                refill_pred.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            gt_batches.append(
                y_tg.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            mask_batches.append(
                mask_4d[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            )

            del (
                x,
                y_tg,
                mask,
                bvec_out,
                raw_pred,
                refill_pred,
                mask_4d,
            )

    raw_pred_slices = np.concatenate(raw_pred_batches, axis=0)
    refill_pred_slices = np.concatenate(refill_pred_batches, axis=0)
    gt_slices = np.concatenate(gt_batches, axis=0)
    mask_slices = np.concatenate(mask_batches, axis=0)

    del raw_pred_batches, refill_pred_batches, gt_batches, mask_batches
    gc.collect()

    return raw_pred_slices, refill_pred_slices, gt_slices, mask_slices

def to_subject_layout(
    pred_slices: np.ndarray,
    gt_slices: np.ndarray,
    mask_slices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [S,90,H,W] -> [H,W,S,90]
    [S,H,W]    -> [H,W,S]
    """
    if pred_slices.shape != gt_slices.shape:
        raise ValueError(
            f"pred/gt slices mismatch: {pred_slices.shape} vs {gt_slices.shape}"
        )
    if pred_slices.ndim != 4 or pred_slices.shape[1] != 90:
        raise ValueError(f"Expected [S,90,H,W], got {pred_slices.shape}")
    if mask_slices.ndim != 3:
        raise ValueError(f"Expected mask [S,H,W], got {mask_slices.shape}")

    pred_subject = np.transpose(pred_slices, (2, 3, 0, 1))
    gt_subject = np.transpose(gt_slices, (2, 3, 0, 1))
    mask_subject = np.transpose(mask_slices, (1, 2, 0))
    return pred_subject, gt_subject, mask_subject


def finite_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")

    mean_value = float(np.mean(array))
    std_value = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return mean_value, std_value


parser = argparse.ArgumentParser(
    description="Test Deep LGM-SGR-v2 with original subject-level metrics."
)
parser.add_argument(
    "--direction_task",
    type=str,
    default="6to90_b1000",
    choices=list(DIRECTION_TASKS.keys()),
)
parser.add_argument("--fixed_indices", type=str, default="")
parser.add_argument(
    "--data_path",
    type=str,
    default="/media/sit/ST1/dataset/dwi_dataset/6to90_b1000",
)
parser.add_argument(
    "--data_type",
    type=str,
    default="test",
    choices=["test", "valid", "val", "train"],
)
parser.add_argument(
    "--model_path",
    type=str,
    default=(
        "/home/sit/project/2D_CNN/final_model/"
        "lgm_sgr_6to90_b1000/LGM_SGR_v2.pth"
    ),
)
parser.add_argument(
    "--result_path",
    type=str,
    default=(
        "/home/sit/project/2D_CNN/test_results/"
        "lgm_sgr_6to90_b1000"
    ),
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_of_out", type=int, default=90)
parser.add_argument("--crop_size", type=int, nargs=2, default=[144, 144])
parser.add_argument("--split", type=float, default=0.8)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--strict_load", type=int, default=1, choices=[0, 1])
parser.add_argument("--bvec_path", type=str, default="")

# Same current LGM configuration.
parser.add_argument(
    "--mamba_scans",
    type=str,
    default="hv",
    choices=["h", "hv", "hv_flip"],
)
parser.add_argument("--mamba_d_state", type=int, default=16)
parser.add_argument("--mamba_d_conv", type=int, default=4)
parser.add_argument("--mamba_expand", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.0)

# Same current SGR-v2 configuration.
parser.add_argument("--graph_node_dim", type=int, default=24)
parser.add_argument("--graph_layers", type=int, default=1)
parser.add_argument("--graph_topk", type=int, default=6)
parser.add_argument("--graph_tau", type=float, default=0.15)

parser.add_argument(
    "--use_data_consistency",
    type=int,
    default=1,
    choices=[0, 1],
    help="Retained for command compatibility. This script always reports both no-refill and refill metrics.",
)
parser.add_argument("--data_range", type=float, default=1.0)
parser.add_argument("--ssim_win_size", type=int, default=7)
parser.add_argument(
    "--save_subject_npy", type=int, default=0, choices=[0, 1]
)
parser.add_argument("--log_name", type=str, default="test_logger.txt")


def main() -> None:
    start_time = time.time()
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    seed_everything(opt.seed)
    os.makedirs(opt.result_path, exist_ok=True)
    logger = get_logger(os.path.join(opt.result_path, opt.log_name))
    logger.info(opt)

    fixed_indices = parse_fixed_indices(
        opt.direction_task,
        parse_int_list(opt.fixed_indices),
    )
    num_in = len(fixed_indices)

    if int(opt.num_of_out) != 90:
        raise ValueError("This script requires --num_of_out 90.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("================ Direction task ================")
    logger.info(f"direction_task : {opt.direction_task}")
    logger.info(f"fixed_indices  : {fixed_indices}")
    logger.info(f"input channels : {num_in}")
    logger.info("metric scope   : all 90 directions together")
    logger.info("outputs        : no-refill 90 directions + refill 90 directions")
    logger.info("known/unknown  : not reported separately")
    logger.info("================================================")

    print("Loading testing dataset ...")
    dataset_test = Dataset_2D(
        opt.data_path,
        data_type=opt.data_type,
        split=opt.split,
        crop_size=tuple(opt.crop_size),
        return_bvec=True,
        direction_task=opt.direction_task,
        fixed_indices=fixed_indices,
    )
    subject_groups = build_subject_groups(dataset_test.examples)

    print(f"{len(dataset_test)} testing slices")
    print(f"{len(subject_groups)} testing subjects/files")

    external_bvec = load_external_bvec(opt.bvec_path)

    model = CNN_2D(
        n_channels=num_in,
        n_out=opt.num_of_out,
        res_hiddens=(128, 256),
        mamba_scans=opt.mamba_scans,
        d_state=opt.mamba_d_state,
        d_conv=opt.mamba_d_conv,
        expand=opt.mamba_expand,
        dropout=opt.dropout,
        graph_node_dim=opt.graph_node_dim,
        graph_layers=opt.graph_layers,
        graph_topk=opt.graph_topk,
        graph_tau=opt.graph_tau,
        fixed_indices=fixed_indices,
    ).to(device)

    logger.info(f"Model parameters: {get_num_parameters(model)}")
    logger.info(
        "Model config: "
        f"mamba_scans={opt.mamba_scans}, "
        f"d_state={opt.mamba_d_state}, "
        f"d_conv={opt.mamba_d_conv}, "
        f"expand={opt.mamba_expand}, "
        f"dropout={opt.dropout}, "
        f"graph_node_dim={opt.graph_node_dim}, "
        f"graph_layers={opt.graph_layers}, "
        f"graph_topk={opt.graph_topk}, "
        f"graph_tau={opt.graph_tau}"
    )

    load_state_dict_compatible(
        model,
        checkpoint_path=opt.model_path,
        strict=bool(opt.strict_load),
    )
    model.eval()

    logger.info(f"Loaded model: {opt.model_path}")
    logger.info(
        "Observed-direction data consistency: integrated in model; both raw and ODDC outputs will be reported"
    )
    logger.info("Testing!")

    records: List[Dict] = []

    for subject_id, group in enumerate(subject_groups):
        (
            raw_pred_slices,
            refill_pred_slices,
            gt_slices,
            mask_slices,
        ) = infer_one_subject(
            model=model,
            dataset=dataset_test,
            row_indices=group["row_indices"],
            fixed_indices=fixed_indices,
            device=device,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            external_bvec=external_bvec,
            subject_id=subject_id,
        )

        raw_pred_subject, gt_subject, mask_subject = to_subject_layout(
            raw_pred_slices,
            gt_slices,
            mask_slices,
        )
        refill_pred_subject, _, _ = to_subject_layout(
            refill_pred_slices,
            gt_slices,
            mask_slices,
        )

        # Match the original convention: apply mask first, then evaluate.
        # Both outputs retain all 90 directions.
        mask_4d = np.expand_dims(mask_subject, axis=-1)
        raw_pred_masked = raw_pred_subject * mask_4d
        refill_pred_masked = refill_pred_subject * mask_4d
        gt_masked = gt_subject * mask_4d

        raw_mse, raw_nrmse, raw_psnr, raw_ssim = one_subject_metrics(
            raw_pred_masked,
            gt_masked,
            data_range=float(opt.data_range),
            ssim_win_size=int(opt.ssim_win_size),
        )
        refill_mse, refill_nrmse, refill_psnr, refill_ssim = one_subject_metrics(
            refill_pred_masked,
            gt_masked,
            data_range=float(opt.data_range),
            ssim_win_size=int(opt.ssim_win_size),
        )

        logger.info(
            f"{subject_id} subject [NO REFILL, all 90]: "
            f"mse={raw_mse}\t "
            f"nrmse={raw_nrmse}\t "
            f"psnr={raw_psnr}\t "
            f"ssim={raw_ssim}\t "
            f"slices={raw_pred_subject.shape[2]}\t "
            f"file={group['file_name']}"
        )
        logger.info(
            f"{subject_id} subject [REFILL, all 90]: "
            f"mse={refill_mse}\t "
            f"nrmse={refill_nrmse}\t "
            f"psnr={refill_psnr}\t "
            f"ssim={refill_ssim}\t "
            f"slices={refill_pred_subject.shape[2]}\t "
            f"file={group['file_name']}"
        )
        logger.info("")

        record = {
            "subject_id": int(subject_id),
            "file_name": group["file_name"],
            "file_path": group["file_path"],
            "num_slices": int(raw_pred_subject.shape[2]),
            "num_directions": 90,
            "no_refill_mse": float(raw_mse),
            "no_refill_nrmse": float(raw_nrmse),
            "no_refill_psnr": float(raw_psnr),
            "no_refill_ssim": float(raw_ssim),
            "refill_mse": float(refill_mse),
            "refill_nrmse": float(refill_nrmse),
            "refill_psnr": float(refill_psnr),
            "refill_ssim": float(refill_ssim),
        }
        records.append(record)

        if bool(opt.save_subject_npy):
            subject_dir = os.path.join(
                opt.result_path,
                f"subject_{subject_id:03d}",
            )
            os.makedirs(subject_dir, exist_ok=True)
            np.save(
                os.path.join(subject_dir, "pred_no_refill_masked.npy"),
                raw_pred_masked.astype(np.float32, copy=False),
            )
            np.save(
                os.path.join(subject_dir, "pred_refill_masked.npy"),
                refill_pred_masked.astype(np.float32, copy=False),
            )
            np.save(
                os.path.join(subject_dir, "gt_masked.npy"),
                gt_masked.astype(np.float32, copy=False),
            )
            np.save(
                os.path.join(subject_dir, "mask.npy"),
                mask_subject.astype(np.float32, copy=False),
            )

        del (
            raw_pred_slices,
            refill_pred_slices,
            gt_slices,
            mask_slices,
            raw_pred_subject,
            refill_pred_subject,
            gt_subject,
            mask_subject,
            mask_4d,
            raw_pred_masked,
            refill_pred_masked,
            gt_masked,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries: Dict[str, Dict[str, Dict[str, float]]] = {
        "no_refill": {},
        "refill": {},
    }

    for mode_name, field_prefix in (
        ("no_refill", "no_refill"),
        ("refill", "refill"),
    ):
        for metric_name in ("mse", "nrmse", "psnr", "ssim"):
            field_name = f"{field_prefix}_{metric_name}"
            mean_value, std_value = finite_mean_std(
                [record[field_name] for record in records]
            )
            summaries[mode_name][metric_name] = {
                "mean": float(mean_value),
                "std": float(std_value),
            }

    for mode_name, mode_label in (
        ("no_refill", "NO REFILL, all 90 directions"),
        ("refill", "REFILL, all 90 directions"),
    ):
        summary = summaries[mode_name]
        logger.info(
            f"All Subjects Metrics [{mode_label}]: "
            f"mse={summary['mse']['mean']}\t "
            f"nrmse={summary['nrmse']['mean']}\t "
            f"psnr={summary['psnr']['mean']}\t "
            f"ssim={summary['ssim']['mean']}"
        )

        for metric_name, label in (
            ("mse", "MSE"),
            ("nrmse", "nRMSE"),
            ("ssim", "SSIM"),
            ("psnr", "PSNR"),
        ):
            logger.info(
                f"{mode_label} {label}: "
                f"mean={summary[metric_name]['mean']}\t "
                f"std={summary[metric_name]['std']}"
            )
            logger.info(
                [record[f"{mode_name}_{metric_name}"] for record in records]
            )

    csv_path = os.path.join(
        opt.result_path,
        "per_subject_metrics_no_refill_and_refill.csv",
    )
    csv_fields = [
        "subject_id",
        "file_name",
        "file_path",
        "num_slices",
        "num_directions",
        "no_refill_mse",
        "no_refill_nrmse",
        "no_refill_psnr",
        "no_refill_ssim",
        "refill_mse",
        "refill_nrmse",
        "refill_psnr",
        "refill_ssim",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(records)

    json_path = os.path.join(
        opt.result_path,
        "per_subject_metrics_no_refill_and_refill.json",
    )
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "direction_task": opt.direction_task,
                "fixed_indices": [int(v) for v in fixed_indices],
                "num_input_directions": int(num_in),
                "num_evaluated_directions": 90,
                "metric_scope": {
                    "no_refill": (
                        "Raw network output is evaluated against GT over all "
                        "90 directions."
                    ),
                    "refill": (
                        "Observed input directions are copied into the raw "
                        "network output, then the resulting complete output is "
                        "evaluated against GT over all 90 directions."
                    ),
                    "ssim": (
                        "Each subject is [H,W,S,90]. SSIM compares each "
                        "[H,W,90] slice and is averaged over slices."
                    ),
                },
                "data_range": float(opt.data_range),
                "model_configuration": {
                    "res_hiddens": [128, 256],
                    "mamba_scans": opt.mamba_scans,
                    "mamba_d_state": int(opt.mamba_d_state),
                    "mamba_d_conv": int(opt.mamba_d_conv),
                    "mamba_expand": int(opt.mamba_expand),
                    "dropout": float(opt.dropout),
                    "graph_node_dim": int(opt.graph_node_dim),
                    "graph_layers": int(opt.graph_layers),
                    "graph_topk": int(opt.graph_topk),
                    "graph_tau": float(opt.graph_tau),
                },
                "records": records,
                "summary": summaries,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    elapsed = time.time() - start_time
    logger.info(f"Saved CSV : {csv_path}")
    logger.info(f"Saved JSON: {json_path}")
    logger.info(f"Finished in {elapsed:.2f} seconds")

    print("\nTesting completed.")
    print("Metric scope: all 90 directions")
    print("Outputs: no-refill metrics + refill metrics")
    print(f"Log : {os.path.join(opt.result_path, opt.log_name)}")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
