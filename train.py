import os
import sys
import argparse
import tempfile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
sys.path.append("./..")
from dataset_2d import Dataset_2D
from model import CNN_2D


DIRECTION_TASKS = {
    "10to90_b1000": [12, 13, 14, 16, 17, 28, 30, 32, 33, 86],
    "10to90_b2000": [20, 21, 22, 33, 46, 52, 53, 58, 75, 88],
    "6to90_b1000": [7, 37, 39, 68, 69, 81],
    "6to90_b2000": [1, 4, 25, 30, 72, 76],
}


def _resolve_direction_task(data_path, num_of_channels):
    task_name = os.path.basename(os.path.normpath(data_path))

    if task_name not in DIRECTION_TASKS:
        raise ValueError(
            f"Cannot identify direction task from data_path={data_path}. "
            f"Supported tasks: {list(DIRECTION_TASKS.keys())}"
        )

    fixed_indices = DIRECTION_TASKS[task_name]

    if len(fixed_indices) != int(num_of_channels):
        raise ValueError(
            f"Task/channel mismatch: task={task_name}, "
            f"fixed directions={len(fixed_indices)}, "
            f"--num_of_channels={num_of_channels}"
        )

    return task_name, fixed_indices


def get_num_param(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def RMSE(a1, a2, mask):
    return np.sqrt(np.sum(((a1 - a2) * mask) ** 2))


def nRMSE(a1, a_ref, a_mask):
    """
    a1:    [B, C, H, W]
    a_ref: [B, C, H, W]
    a_mask:[B, 1, H, W]
    return:[1, C]
    """
    n_batch = a_ref.shape[0]
    n_ch = a_ref.shape[1]

    b1 = np.reshape(a1, (n_batch, n_ch, -1))
    b1 = b1.transpose(0, 2, 1).reshape(-1, n_ch)

    b_ref = np.reshape(a_ref, (n_batch, n_ch, -1))
    b_ref = b_ref.transpose(0, 2, 1).reshape(-1, n_ch)

    b_mask = np.reshape(a_mask, (n_batch, 1, -1))
    b_mask = b_mask.transpose(0, 2, 1).reshape(-1, 1)

    nrmse = np.zeros([1, n_ch], dtype=np.float64)

    for n_c in range(n_ch):
        numerator = RMSE(b_ref[:, n_c], b1[:, n_c], b_mask[:, 0])
        denominator = RMSE(b_ref[:, n_c], -b_ref[:, n_c], b_mask[:, 0])
        nrmse[0, n_c] = 2.0 * numerator / (denominator + 1e-12)

    return nrmse


def _squeeze_extra_dim(t):
    if torch.is_tensor(t) and t.ndim == 5 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def _broadcast_mask_to_output(mask, output):
    mask = _squeeze_extra_dim(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"mask must be 2D/3D/4D tensor, got {tuple(mask.shape)}")
    if mask.shape[-2:] != output.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask.float(), size=output.shape[-2:], mode="nearest")
    if mask.shape[1] == 1:
        return mask
    if mask.shape[1] == output.shape[1]:
        return mask
    return mask.max(dim=1, keepdim=True).values


def _normalize_bvec(bvec, eps=1e-8):
    return bvec / torch.clamp(torch.linalg.norm(bvec, dim=-1, keepdim=True), min=eps)


def _ensure_bvec_batch(bvec, batch_size, device, dtype, name="bvec_out"):
    if bvec is None:
        raise ValueError(f"{name} is required for SGR head, but got None.")
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
            raise ValueError(f"{name} batch mismatch: got {tuple(bvec.shape)}, expected B={batch_size}")
    else:
        raise ValueError(f"{name} must be [O,3] or [B,O,3], got {tuple(bvec.shape)}")

    if bvec.shape[-1] != 3:
        raise ValueError(f"{name} last dim must be 3, got {bvec.shape[-1]}")
    return _normalize_bvec(bvec)


def _load_external_bvec(path):
    if path is None or str(path).strip() == "":
        return None
    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        key = "bvecs_90" if "bvecs_90" in data else list(data.keys())[0]
        arr = data[key]
    else:
        raise ValueError("Only .npy/.npz bvec_path is supported.")
    if arr.shape[-1] != 3:
        raise ValueError(f"Loaded bvec should have last dim 3, got shape {arr.shape}")
    return torch.as_tensor(arr, dtype=torch.float32)


def _unpack_batch(batch):
    """
    Compatible with different Dataset_2D versions:
        old baseline dataset:      (x, y_tg, x_mask)
        h5 bvec dataset:           (x, y_tg, x_mask, bvec_out)
        bvec fixed-index dataset:  (x, y_tg, x_mask, bvec_in, bvec_out)

    SGR head uses bvec_out. If the dataset returns only 3 items, provide --bvec_path.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            x, y_tg, x_mask = batch
            bvec_out = None
        elif len(batch) == 4:
            x, y_tg, x_mask, bvec_out = batch
        elif len(batch) == 5:
            x, y_tg, x_mask, _, bvec_out = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        return _squeeze_extra_dim(x), _squeeze_extra_dim(y_tg), _squeeze_extra_dim(x_mask), bvec_out
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _save_state_dict_atomic(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path), suffix=".tmp") as tmp:
        tmp_path = tmp.name
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def get_score_model(model, data_loader, device, external_bvec=None):
    nrmse_list = []
    model.eval()

    with torch.no_grad():
        val_pbar = tqdm(
            data_loader,
            total=len(data_loader),
            desc="Validation",
            ncols=100,
            leave=False,
        )

        for batch in val_pbar:
            x, y_tg, x_mask, bvec_out = _unpack_batch(batch)
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y_tg = y_tg.to(device=device, dtype=torch.float32, non_blocking=True)
            x_mask = x_mask.to(device=device, dtype=torch.float32, non_blocking=True)

            if bvec_out is None:
                if external_bvec is None:
                    raise ValueError("Validation needs bvec_out from Dataset_2D or --bvec_path for SGR head.")
                bvec_out = external_bvec
            bvec_out = _ensure_bvec_batch(bvec_out, x.shape[0], device, x.dtype, name="bvec_out")

            # Model returns (raw_prediction, ODDC_prediction).
            # Validation/model selection follows the training objective and
            # therefore uses the RAW 90-direction prediction before ODDC.
            y_raw, _ = model(x, bvec_out=bvec_out)
            x_mask = _broadcast_mask_to_output(x_mask, y_raw)
            y_raw = y_raw * x_mask

            this_nrmse = nRMSE(
                y_raw.detach().cpu().numpy(),
                y_tg.detach().cpu().numpy(),
                x_mask.detach().cpu().numpy(),
            )
            nrmse_list.append(this_nrmse)

    per_channel_nrmse = np.concatenate(nrmse_list, axis=0).mean(axis=0)
    mean_nrmse = float(np.mean(per_channel_nrmse))
    return per_channel_nrmse, mean_nrmse


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Training LGM-SGR with best nRMSE saving")
parser.add_argument("--sampling", type=str, default="10_DWI", help="Q-space undersampling pattern name")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
parser.add_argument("--num_of_channels", type=int, default=10, help="Number of CNN input channels")
parser.add_argument("--num_of_out", type=int, default=90, help="Number of CNN output channels")
parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--data_path", type=str, default="/media/sit/ST1/dataset/dwi_dataset/10to90_b1000", help="Path to the data")
parser.add_argument("--model_dir", type=str, default="/home/sit/project/2D_CNN/new_model/two_lgm_sgr_10to90_b1000", help="Path to save best model")
parser.add_argument("--model_name", type=str, default="2D_CNN_LGM_SGR_best_nrmse.pth", help="Best model filename")
parser.add_argument("--bvec_path", type=str, default="", help="Optional .npy/.npz bvecs_90 path if Dataset_2D does not return bvec_out")

# LGM-Net parameters
parser.add_argument("--mamba_scans", type=str, default="hv", choices=["h", "hv", "hv_flip"], help="Spatial scan directions. h is fastest, hv is default, hv_flip is strongest but slowest.")
parser.add_argument("--mamba_d_state", type=int, default=16, help="Mamba d_state")
parser.add_argument("--mamba_d_conv", type=int, default=4, help="Mamba local conv width")
parser.add_argument("--mamba_expand", type=int, default=2, help="Mamba expansion ratio")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout inside spatial Mamba branch")

# SGR output head parameters
parser.add_argument("--graph_node_dim", type=int, default=8, help="Node feature dimension for each output direction in SGR head")
parser.add_argument("--graph_layers", type=int, default=1, help="Number of spherical graph propagation layers in SGR head")
parser.add_argument("--graph_topk", type=int, default=8, help="Top-k angular neighbors per output direction")
parser.add_argument("--graph_tau", type=float, default=0.10, help="Temperature for angular graph weights")

opt = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    task_name, fixed_indices = _resolve_direction_task(opt.data_path, opt.num_of_channels)

    print(f"Direction task: {task_name}")
    print(f"Fixed direction indices: {fixed_indices}")
    print(f"Model input channels: {opt.num_of_channels}")
    print("Loading training dataset ...\n")

    dataset_train = Dataset_2D(
        opt.data_path,
        data_type="train",
        split=0.8,
        direction_task=task_name,
        fixed_indices=fixed_indices,
    )
    dataset_val = Dataset_2D(
        opt.data_path,
        data_type="valid",
        split=0.8,
        direction_task=task_name,
        fixed_indices=fixed_indices,
    )

    train_loader = DataLoader(
        dataset=dataset_train,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(opt.num_workers > 0),
    )
    val_loader = DataLoader(
        dataset=dataset_val,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(opt.num_workers > 0),
    )

    print(f"# of training samples: {len(dataset_train)}")
    print(f"# of training batches: {len(train_loader)}")
    print(f"# of validation samples: {len(dataset_val)}")
    print(f"# of validation batches: {len(val_loader)}")

    external_bvec = _load_external_bvec(opt.bvec_path)
    if external_bvec is not None:
        print(f"Loaded external bvecs_90 from {opt.bvec_path}, shape={tuple(external_bvec.shape)}")

    model = CNN_2D(
        n_channels=opt.num_of_channels,
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

    criterion = nn.L1Loss(reduction="mean").to(device)
    print(f"# of parameters: {get_num_param(model)}")
    print("LGM-SGR structure: Res_Block(10,128) -> LG_Mamba_Block(128,256) -> LG_Mamba_Block(256,256) -> SphericalGraphReadoutHead(256,90)")
    print(f"mamba_scans={opt.mamba_scans}, d_state={opt.mamba_d_state}, d_conv={opt.mamba_d_conv}, expand={opt.mamba_expand}")
    print(f"SGR head: node_dim={opt.graph_node_dim}, layers={opt.graph_layers}, topk={opt.graph_topk}, tau={opt.graph_tau}")
    print("Loss is unchanged: plain L1 between RAW predicted 90 DWI (before ODDC) and GT 90 DWI.")
    print("Dataset_2D should return bvec_out, or provide --bvec_path.\n")

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
    )

    os.makedirs(opt.model_dir, exist_ok=True)
    best_weights_path = os.path.join(opt.model_dir, opt.model_name)
    best_nrmse = float("inf")
    best_epoch = -1

    printed_first_batch = False

    for epoch in range(opt.epochs):
        model.train()
        loss_list = []

        train_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch [{epoch + 1}/{opt.epochs}]",
            ncols=100,
            leave=True,
        )

        for batch in train_pbar:
            x, y_tg, x_mask, bvec_out = _unpack_batch(batch)
            optimizer.zero_grad(set_to_none=True)

            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y_tg = y_tg.to(device=device, dtype=torch.float32, non_blocking=True)
            x_mask = x_mask.to(device=device, dtype=torch.float32, non_blocking=True)

            if bvec_out is None:
                if external_bvec is None:
                    raise ValueError("Training needs bvec_out from Dataset_2D or --bvec_path for SGR head.")
                bvec_out = external_bvec
            bvec_out = _ensure_bvec_batch(bvec_out, x.shape[0], device, x.dtype, name="bvec_out")

            # Model returns two outputs:
            #   y_raw   : learned 90-direction prediction before ODDC
            #   y_final : measurement-consistent prediction after ODDC
            # Training loss is computed ONLY from y_raw, so all 90 directional
            # nodes receive direct supervision.
            y_raw, y_final = model(x, bvec_out=bvec_out)
            x_mask = _broadcast_mask_to_output(x_mask, y_raw)
            y_raw = y_raw * x_mask

            loss = criterion(y_raw, y_tg)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            loss_list.append(loss_value)

            if not printed_first_batch:
                print("\n================ First training batch check ================")
                print(f"x        shape: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
                print(f"y_tg     shape: {tuple(y_tg.shape)}, dtype={y_tg.dtype}, device={y_tg.device}")
                print(f"mask     shape: {tuple(x_mask.shape)}, dtype={x_mask.dtype}, device={x_mask.device}")
                print(f"bvec_out shape: {tuple(bvec_out.shape)}, dtype={bvec_out.dtype}, device={bvec_out.device}")
                print(f"pred_raw shape: {tuple(y_raw.shape)}, dtype={y_raw.dtype}, device={y_raw.device}")
                print(f"pred_dc  shape: {tuple(y_final.shape)}, dtype={y_final.dtype}, device={y_final.device}")
                print(f"loss     : {loss_value:.8f}")
                print("============================================================\n")
                printed_first_batch = True

            train_pbar.set_postfix(loss=f"{loss_value:.6f}")

        avg_loss = float(np.mean(loss_list))
        print(f"Epoch: {epoch + 1} out of {opt.epochs}")
        print(f"Loss: {avg_loss:.6f}")

        nrmse_val, mean_nrmse = get_score_model(model, val_loader, device, external_bvec=external_bvec)
        print("Validation Accuracy in nRMSE: " + str(nrmse_val))
        print(f"Validation mean nRMSE over 90 directions: {mean_nrmse:.8f}")

        scheduler.step(mean_nrmse)

        if mean_nrmse < best_nrmse:
            best_nrmse = mean_nrmse
            best_epoch = epoch + 1
            _save_state_dict_atomic(model.state_dict(), best_weights_path)
            print(f"New best model saved to: {best_weights_path}")
            print(f"Best epoch: {best_epoch}, best mean nRMSE: {best_nrmse:.8f}")
        else:
            print(f"No improvement. Best epoch: {best_epoch}, best mean nRMSE: {best_nrmse:.8f}")

        print()

    print(f"Training finished. Best epoch: {best_epoch}, best mean nRMSE: {best_nrmse:.8f}")
    print(f"Best model path: {best_weights_path}")


if __name__ == "__main__":
    main()
