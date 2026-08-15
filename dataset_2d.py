import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from torch.utils.data import Dataset


DIRECTION_TASKS: Dict[str, List[int]] = {
    "10to90_b1000": [12, 13, 14, 16, 17, 28, 30, 32, 33, 86],
    "10to90_b2000": [20, 21, 22, 33, 46, 52, 53, 58, 75, 88],
    "6to90_b1000": [7, 37, 39, 68, 69, 81],
    "6to90_b2000": [1, 4, 25, 30, 72, 76],
}

DIRECTION_TASKS_AD = {
    "10to90_b1000": [9, 15, 34, 41, 43, 44, 52, 53, 68, 69],
    "6to90_b1000": [28, 47, 54, 61, 67, 85],

    "10to90_b2000": [0, 9, 35, 37, 45, 47, 54, 55, 79, 84],
    "6to90_b2000": [15, 39, 41, 53, 72, 86],
}

def parse_fixed_indices(direction_task: str = "10to90_b1000", fixed_indices: Optional[Sequence[int]] = None) -> List[int]:
    """Return Python 0-based fixed indices for a direction task."""
    if fixed_indices is not None:
        idx = [int(i) for i in fixed_indices]
    else:
        if direction_task not in DIRECTION_TASKS:
            raise ValueError(
                f"Unsupported direction_task={direction_task}. "
                f"Choose one of {list(DIRECTION_TASKS.keys())}, or pass fixed_indices explicitly."
            )
        idx = list(DIRECTION_TASKS[direction_task])

    if len(idx) == 0:
        raise ValueError("fixed_indices cannot be empty")
    if len(set(idx)) != len(idx):
        raise ValueError(f"fixed_indices contains duplicates: {idx}")
    if min(idx) < 0 or max(idx) >= 90:
        raise ValueError(f"fixed_indices must be in [0, 89], got {idx}")
    return idx


def parse_ad_fixed_indices(
    direction_task: str = "10to90_b1000",
    fixed_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Return the FINAL AD-native fixed indices.

    Dataset_AD_2D always uses DIRECTION_TASKS_AD as the source of truth.
    This is intentional so that existing AD training scripts do not need
    to be modified even if they still pass the old HCP fixed_indices.

    If fixed_indices is supplied and differs from DIRECTION_TASKS_AD, it is
    ignored and a warning is printed.
    """
    if direction_task not in DIRECTION_TASKS_AD:
        raise ValueError(
            f"Unsupported AD direction_task={direction_task}. "
            f"Choose one of {list(DIRECTION_TASKS_AD.keys())}."
        )

    idx = [int(i) for i in DIRECTION_TASKS_AD[direction_task]]

    if fixed_indices is not None:
        supplied = [int(i) for i in fixed_indices]
        if supplied != idx:
            print(
                "[Dataset_AD_2D] WARNING: supplied fixed_indices differ from "
                "the finalized AD-native indices and will be ignored.\n"
                f"  supplied : {supplied}\n"
                f"  AD final : {idx}"
            )

    if len(idx) == 0:
        raise ValueError("AD fixed_indices cannot be empty")
    if len(set(idx)) != len(idx):
        raise ValueError(f"AD fixed_indices contains duplicates: {idx}")
    if min(idx) < 0 or max(idx) >= 90:
        raise ValueError(f"AD fixed_indices must be in [0, 89], got {idx}")

    expected_k = 10 if direction_task.startswith("10to90_") else 6
    if len(idx) != expected_k:
        raise ValueError(
            f"AD task/index mismatch: task={direction_task}, "
            f"expected K={expected_k}, got K={len(idx)}"
        )

    return idx


def get_unknown_indices(fixed_indices: Sequence[int], n_out: int = 90) -> List[int]:
    fixed_set = set(int(i) for i in fixed_indices)
    return [i for i in range(int(n_out)) if i not in fixed_set]


def center_crop_or_pad_hw(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Center crop or zero-pad the last two dimensions to size=(H,W)."""
    h1, w1 = int(size[0]), int(size[1])
    h, w = image.shape[-2:]

    pad_h = max(0, h1 - h)
    pad_w = max(0, w1 - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_width = [(0, 0)] * (image.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
        image = np.pad(image, pad_width=pad_width, mode="constant")

    h2, w2 = image.shape[-2:]
    up = (h2 - h1) // 2
    left = (w2 - w1) // 2
    slicer = [slice(None)] * (image.ndim - 2) + [slice(up, up + h1), slice(left, left + w1)]
    return image[tuple(slicer)]


def _normalize_mask(mask_2d: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask_2d is None:
        return None
    arr = np.asarray(mask_2d)
    while arr.ndim > 2 and 1 in (arr.shape[0], arr.shape[-1]):
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            break
    if arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape {mask_2d.shape}; expected 2D after squeeze")
    return arr


def _to_chw(arr: np.ndarray, expected_c: Optional[int] = None) -> np.ndarray:
    """Convert [C,H,W] or [H,W,C] or [1,C,H,W] or [1,H,W,C] to [C,H,W]."""
    arr = np.asarray(arr)
    while arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"_to_chw expects 3D after squeeze, got shape={arr.shape}")

    if expected_c is not None:
        if arr.shape[0] == expected_c:
            return arr
        if arr.shape[-1] == expected_c:
            return np.transpose(arr, (2, 0, 1))
        raise ValueError(f"Cannot infer channel axis for shape={arr.shape}, expected_c={expected_c}")

    return arr


def _get_stage_dir(data_path: str, data_type: str) -> pathlib.Path:
    root = pathlib.Path(data_path)
    key = str(data_type).lower()
    if key in ("train", "fit"):
        candidates = [root / "train", root / "Train"]
    elif key in ("val", "valid", "validate"):
        candidates = [root / "valid", root / "val", root / "Valid", root / "Val"]
    elif key == "test":
        candidates = [root / "test", root / "Test"]
    else:
        candidates = [root]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _find_h5_files(data_path: str, data_type: str, split: float = 0.8) -> List[pathlib.Path]:
    stage_dir = _get_stage_dir(data_path, data_type)
    suffixes = (".h5", ".hdf5", ".h5df")
    files = [p for p in sorted(stage_dir.iterdir()) if p.is_file() and p.suffix.lower() in suffixes] if stage_dir.exists() else []

    if files:
        return files

    # Fallback: files are directly under root, split deterministically.
    root = pathlib.Path(data_path)
    files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in suffixes] if root.exists() else []
    if not files:
        raise FileNotFoundError(f"No h5/hdf5/h5df files found under {data_path}")

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(files)).tolist()
    files = [files[i] for i in perm]
    n_train = int(round(len(files) * float(split)))
    key = str(data_type).lower()
    if key in ("train", "fit"):
        return files[:n_train]
    if key in ("val", "valid", "validate"):
        return files[n_train:]
    return files


def _find_first_key(h5_file: h5py.File, candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key in h5_file:
            return key
    return None


def _read_y90_from_h5(f: h5py.File, slice_id: int, y_channels: int = 90) -> np.ndarray:
    key = _find_first_key(f, ["y_90", "y90", "target", "label", "dwi_90"])
    if key is None:
        raise KeyError("No y_90/y90/target/label/dwi_90 key found in h5")

    arr = np.asarray(f[key][slice_id])

    # Accept y with b0 + 90 diffusion channels.
    try:
        y = _to_chw(arr, expected_c=y_channels)
    except ValueError:
        y = _to_chw(arr, expected_c=y_channels + 1)
        y = y[1:y_channels + 1]

    if y.shape[0] > y_channels:
        y = y[:y_channels]
    if y.shape[0] != y_channels:
        raise ValueError(f"Target y should have {y_channels} channels after processing, got shape={y.shape}")
    return y.astype(np.float32)


def _read_mask_from_h5(f: h5py.File, slice_id: int, crop_size: Tuple[int, int]) -> np.ndarray:
    key = _find_first_key(f, ["mask", "brain_mask", "x_mask", "nodif_brain_mask"])
    if key is None:
        return np.ones(crop_size, dtype=np.float32)
    mask = _normalize_mask(np.asarray(f[key][slice_id]))
    if mask is None:
        return np.ones(crop_size, dtype=np.float32)
    mask = center_crop_or_pad_hw(mask, crop_size)
    return (mask > 0).astype(np.float32)


def _as_bvec_array(arr: np.ndarray) -> np.ndarray:
    """Convert bvec/bevc array to [N,3]. Supports [N,3], [3,N], [1,N,3], [1,3,N]."""
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"bvec must be 2D after squeeze, got shape={arr.shape}")
    if arr.shape[-1] == 3:
        return arr.astype(np.float32)
    if arr.shape[0] == 3:
        return arr.T.astype(np.float32)
    raise ValueError(f"Cannot convert bvec/bevc to [N,3], got shape={arr.shape}")


def _normalize_bvecs(bvecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return (bvecs / norm).astype(np.float32)


def _read_target_bvecs_from_h5(f: h5py.File, y_channels: int = 90) -> Optional[np.ndarray]:
    key = _find_first_key(
        f,
        [
            "bvecs_90",
            "bvec_90",
            "target_bvecs",
            "bvecs_out",
            "bvec_out",
            "bvecs",
            "bvec",
            "bevc",  # typo-compatible
        ],
    )
    if key is None:
        return None

    bvec_out = _as_bvec_array(f[key][()])
    if bvec_out.shape[0] == y_channels + 1:
        bvec_out = bvec_out[1:y_channels + 1]
    elif bvec_out.shape[0] > y_channels:
        bvec_out = bvec_out[:y_channels]

    if bvec_out.shape[0] != y_channels:
        raise ValueError(f"bvec_out should contain {y_channels} directions, got shape={bvec_out.shape} from key={key}")
    return _normalize_bvecs(bvec_out)


class Dataset_2D(Dataset):
    """
    Direction-task-aware 2D dataset.

    Unlike the previous fixed x_10-only loader, this loader always constructs x_in
    from y_90 according to fixed_indices. Therefore the input channel order is
    guaranteed to match the fixed_indices passed into the model for refill.
    """

    def __init__(
        self,
        data_path: str,
        data_type: str,
        split: float = 0.8,
        crop_size: Tuple[int, int] = (144, 144),
        return_bvec: bool = True,
        direction_task: str = "10to90_b1000",
        fixed_indices: Optional[Sequence[int]] = None,
        skip_empty_slices: bool = True,
    ):
        super().__init__()
        self.data_path = str(data_path)
        self.data_type = str(data_type)
        self.crop_size = tuple(crop_size)
        self.return_bvec = bool(return_bvec)
        self.direction_task = str(direction_task)
        self.fixed_indices = parse_fixed_indices(self.direction_task, fixed_indices)
        self.unknown_indices = get_unknown_indices(self.fixed_indices, n_out=90)
        self.num_input_channels = len(self.fixed_indices)
        self.skip_empty_slices = bool(skip_empty_slices)

        self.files = [str(p) for p in _find_h5_files(self.data_path, self.data_type, split=split)]
        self.examples: List[Tuple[str, int]] = []

        for file in self.files:
            with h5py.File(file, "r") as f:
                y_key = _find_first_key(f, ["y_90", "y90", "target", "label", "dwi_90"])
                if y_key is None:
                    continue
                slices = int(f[y_key].shape[0])

                if slices > 113:
                    slice_ids = list(range(60, slices - 53))
                else:
                    slice_ids = list(range(slices))

                if self.skip_empty_slices:
                    filtered: List[int] = []
                    for sid in slice_ids:
                        try:
                            if "mask" in f:
                                m = _normalize_mask(np.asarray(f["mask"][sid]))
                                m = center_crop_or_pad_hw(m, self.crop_size)
                                if float(np.sum(m > 0)) > 0:
                                    filtered.append(sid)
                            else:
                                y = _read_y90_from_h5(f, sid)
                                x = y[np.asarray(self.fixed_indices, dtype=np.int64)]
                                x = center_crop_or_pad_hw(x, self.crop_size)
                                if float(np.max(np.abs(x))) > 0:
                                    filtered.append(sid)
                        except Exception:
                            # If a single slice is malformed, skip it instead of killing dataset construction.
                            continue
                    if filtered:
                        slice_ids = filtered

                self.examples.extend((file, int(sid)) for sid in slice_ids)

        if not self.examples:
            raise ValueError(f"Dataset_2D: no valid examples found under {self.data_path}")

        print(
            f"Dataset_2D: data_type={self.data_type}, files={len(self.files)}, slices={len(self.examples)}, "
            f"direction_task={self.direction_task}, K={self.num_input_channels}, fixed_indices={self.fixed_indices}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        file, slice_id = self.examples[idx]
        fixed = np.asarray(self.fixed_indices, dtype=np.int64)

        with h5py.File(file, "r") as f:
            y90 = _read_y90_from_h5(f, slice_id, y_channels=90)
            x_in = y90[fixed]

            x_in = center_crop_or_pad_hw(x_in, self.crop_size).astype(np.float32)
            y90 = center_crop_or_pad_hw(y90, self.crop_size).astype(np.float32)
            mask = _read_mask_from_h5(f, slice_id, self.crop_size).astype(np.float32)

            if not self.return_bvec:
                return x_in, y90, mask

            bvec_out = _read_target_bvecs_from_h5(f, y_channels=90)
            if bvec_out is None:
                return x_in, y90, mask
            bvec_in = bvec_out[fixed].astype(np.float32)
            return x_in, y90, mask, bvec_in, bvec_out.astype(np.float32)


