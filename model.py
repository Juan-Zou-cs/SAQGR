

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


class Res_Block(nn.Module):
    """Original baseline residual block. Kept unchanged for the first stem layer."""
    def __init__(self, C_in, C_out):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.relu1(self.conv1(x))
        y2 = self.relu2(self.conv2(y1))
        y3 = self.relu3(self.conv3(y2))
        return y3 + y1


class Out_Block(nn.Module):
    """Original output block, kept only for optional ablation."""
    def __init__(self, C_in, C_out):
        super(Out_Block, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(x)


class SpatialMamba2D(nn.Module):
    """
    Spatial Mamba module.

    Mamba is applied on spatial sequences, not on diffusion-direction sequences.
    This avoids imposing an artificial 1D causal order on bvec directions.

    scans:
        "h"       : horizontal scan only
        "hv"      : horizontal + vertical scans
        "hv_flip" : horizontal + vertical + reverse horizontal + reverse vertical
    """
    def __init__(
        self,
        channels,
        scans="hv",
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
    ):
        super(SpatialMamba2D, self).__init__()
        self.channels = int(channels)
        self.scans = str(scans)
        if self.scans not in ("h", "hv", "hv_flip"):
            raise ValueError("scans must be one of: 'h', 'hv', 'hv_flip'")

        self.norm = nn.LayerNorm(channels)
        self.use_mamba = HAS_MAMBA

        if self.use_mamba:
            self.seq = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback is only for debugging and shape checking.
            # For formal experiments, install mamba_ssm.
            self.seq = nn.Sequential(
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )

        self.proj_norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _run_seq(self, seq):
        # seq: [B, L, C]
        seq = self.norm(seq)
        out = self.seq(seq)
        out = self.proj(self.proj_norm(out))
        out = self.drop(out)
        return out

    def _scan_horizontal(self, x, reverse=False):
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        if reverse:
            seq = torch.flip(seq, dims=[1])
        out = self._run_seq(seq)
        if reverse:
            out = torch.flip(out, dims=[1])
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

    def _scan_vertical(self, x, reverse=False):
        B, C, H, W = x.shape
        xt = x.transpose(2, 3).contiguous()  # [B, C, W, H]
        seq = xt.permute(0, 2, 3, 1).contiguous().view(B, W * H, C)
        if reverse:
            seq = torch.flip(seq, dims=[1])
        out = self._run_seq(seq)
        if reverse:
            out = torch.flip(out, dims=[1])
        out = out.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        return out

    def forward(self, x):
        outs = [self._scan_horizontal(x, reverse=False)]

        if self.scans in ("hv", "hv_flip"):
            outs.append(self._scan_vertical(x, reverse=False))

        if self.scans == "hv_flip":
            outs.append(self._scan_horizontal(x, reverse=True))
            outs.append(self._scan_vertical(x, reverse=True))

        return torch.stack(outs, dim=0).mean(dim=0)


class LG_Mamba_Block(nn.Module):
    """
    Local-Global Mamba Block.

    Layout:
        1x1 Conv projection
        local 3x3 Conv branch
        spatial Mamba global branch
        channel-wise gated fusion
        3x3 Conv output
        residual add
    """
    def __init__(
        self,
        C_in,
        C_out,
        scans="hv",
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
    ):
        super(LG_Mamba_Block, self).__init__()

        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        self.local_branch = nn.Sequential(
            nn.Conv2d(C_out, C_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.global_branch = SpatialMamba2D(
            channels=C_out,
            scans=scans,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.global_act = nn.ReLU(inplace=True)

        self.gate = nn.Sequential(
            nn.Conv2d(C_out * 2, C_out, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        self.conv3 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.relu1(self.conv1(x))

        y_local = self.local_branch(y1)
        y_global = self.global_act(self.global_branch(y1))

        gate = self.gate(torch.cat([y_local, y_global], dim=1))
        y_mix = gate * y_global + (1.0 - gate) * y_local

        y3 = self.relu3(self.conv3(y_mix))
        return y3 + y1


def normalize_bvec(bvec, eps=1e-8):
    return bvec / torch.clamp(torch.linalg.norm(bvec, dim=-1, keepdim=True), min=eps)


def bvec_6d_antipodal_features(bvec):
    """
    Antipodal-safe b-vector encoding for diffusion directions.

    bvec and -bvec represent the same diffusion direction, so only even-order
    polynomial terms are used.

    Input:
        bvec: [..., 3]
    Output:
        feat: [..., 6]
    """
    bvec = normalize_bvec(bvec)
    gx = bvec[..., 0]
    gy = bvec[..., 1]
    gz = bvec[..., 2]
    s2 = math.sqrt(2.0)
    return torch.stack(
        [
            gx * gx,
            gy * gy,
            gz * gz,
            s2 * gx * gy,
            s2 * gx * gz,
            s2 * gy * gz,
        ],
        dim=-1,
    )


class SelfPreservedSphericalGraphReadoutHead(nn.Module):
    """
    Self-preserved Spherical Graph Readout Head (SGR-v2).

    It replaces the plain Out_Block(256,90) and the weaker SGR-v1 head.

    Difference from SGR-v1:
      - SGR-v1 uses only neighbor aggregation A @ node, which can over-smooth
        directional differences.
      - SGR-v2 keeps self node, spherical-neighbor node, and their difference:
            fused_j = Fuse([self_j, neighbor_j, self_j - neighbor_j])
      - bvec_out is injected into node features through antipodal-safe FiLM,
        instead of being used only to build the graph.
      - direction-wise grouped readout gives each output direction its own
        final D-to-1 readout weights.

    This is still a main output head, not a residual correction to a raw output.
    """
    def __init__(
        self,
        C_in,
        n_out=90,
        node_dim=16,
        graph_layers=1,
        topk=8,
        tau=0.15,
        bvec_hidden=32,
        fixed_bvec_out=None,
    ):
        super(SelfPreservedSphericalGraphReadoutHead, self).__init__()
        self.C_in = int(C_in)
        self.n_out = int(n_out)
        self.node_dim = int(node_dim)
        self.graph_layers = int(graph_layers)
        self.topk = int(topk)
        self.tau = float(tau)
        self.bvec_hidden = int(bvec_hidden)

        if self.node_dim <= 0:
            raise ValueError("node_dim must be positive")
        if self.graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if self.topk <= 0:
            raise ValueError("topk must be positive")

        # Per-direction node embedding generated from the same spatial feature map.
        # Output shape after reshape: [B, O, D, H, W]
        self.node_proj = nn.Conv2d(C_in, self.n_out * self.node_dim, kernel_size=1, padding=0)

        # bvec FiLM uses antipodal-safe 6D features.
        self.bvec_mlp = nn.Sequential(
            nn.Linear(6, self.bvec_hidden),
            nn.SiLU(),
            nn.Linear(self.bvec_hidden, self.bvec_hidden),
            nn.SiLU(),
        )
        self.bvec_scale = nn.Linear(self.bvec_hidden, self.node_dim)
        self.bvec_shift = nn.Linear(self.bvec_hidden, self.node_dim)

        # Fuse self / neighbor / difference for each graph layer.
        self.graph_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.node_dim * 3, self.node_dim * 2, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.node_dim * 2, self.node_dim, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.graph_layers)
        ])

        # Direction-wise grouped readout. Each output direction owns a D-to-1
        # linear readout. This is more expressive than a single shared readout,
        # but still much more structured than a plain Conv2d(256,90).
        self.readout_weight = nn.Parameter(torch.empty(self.n_out, self.node_dim))
        self.readout_bias = nn.Parameter(torch.zeros(self.n_out))
        nn.init.xavier_uniform_(self.readout_weight)

        self.register_buffer("fixed_bvec_out", torch.empty(0), persistent=False)
        if fixed_bvec_out is not None:
            self.set_fixed_bvec_out(fixed_bvec_out)

    def set_fixed_bvec_out(self, fixed_bvec_out):
        if not torch.is_tensor(fixed_bvec_out):
            fixed_bvec_out = torch.as_tensor(fixed_bvec_out, dtype=torch.float32)
        if fixed_bvec_out.ndim != 2 or fixed_bvec_out.shape[-1] != 3:
            raise ValueError(f"fixed_bvec_out must be [O,3], got {tuple(fixed_bvec_out.shape)}")
        if fixed_bvec_out.shape[0] != self.n_out:
            raise ValueError(f"fixed_bvec_out has O={fixed_bvec_out.shape[0]}, but n_out={self.n_out}")
        self.fixed_bvec_out = normalize_bvec(fixed_bvec_out.detach().float())

    def _prepare_bvec_out(self, bvec_out, batch_size, device, dtype):
        if bvec_out is None:
            if self.fixed_bvec_out.numel() > 0:
                bvec_out = self.fixed_bvec_out.to(device=device, dtype=dtype)
            else:
                raise ValueError(
                    "SelfPreservedSphericalGraphReadoutHead requires bvec_out. "
                    "Call model(x, bvec_out=bvec_out), or initialize/set fixed_bvec_out."
                )

        if not torch.is_tensor(bvec_out):
            bvec_out = torch.as_tensor(bvec_out)
        bvec_out = bvec_out.to(device=device, dtype=dtype)

        if bvec_out.ndim == 4 and bvec_out.shape[1] == 1:
            bvec_out = bvec_out.squeeze(1)

        if bvec_out.ndim == 2:
            bvec_out = bvec_out.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        elif bvec_out.ndim == 3:
            if bvec_out.shape[0] == 1 and batch_size > 1:
                bvec_out = bvec_out.expand(batch_size, -1, -1).contiguous()
            elif bvec_out.shape[0] != batch_size:
                raise ValueError(f"bvec_out batch mismatch: got {tuple(bvec_out.shape)}, expected B={batch_size}")
        else:
            raise ValueError(f"bvec_out must be [O,3] or [B,O,3], got {tuple(bvec_out.shape)}")

        if bvec_out.shape[1] != self.n_out:
            raise ValueError(f"bvec_out direction number mismatch: got {bvec_out.shape[1]}, expected {self.n_out}")
        if bvec_out.shape[-1] != 3:
            raise ValueError(f"bvec_out last dim must be 3, got {bvec_out.shape[-1]}")
        return normalize_bvec(bvec_out)

    def _build_graph(self, bvec_out):
        """
        bvec_out: [B,O,3]
        return A: [B,O,O], row-normalized. A[o,j] means source node j contributes to target node o.
        """
        B, O, _ = bvec_out.shape
        k = min(max(1, self.topk), O)

        # Antipodal-aware similarity.
        sim = torch.abs(torch.matmul(bvec_out, bvec_out.transpose(1, 2))).clamp(0.0, 1.0)
        top_val, top_idx = torch.topk(sim, k=k, dim=-1, largest=True, sorted=False)

        top_w = torch.exp((top_val - 1.0) / max(self.tau, 1e-6))
        A = torch.zeros_like(sim)
        A.scatter_(dim=-1, index=top_idx, src=top_w)
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return A

    def _apply_bvec_film(self, node, bvec_out):
        """
        node    : [B,O,D,H,W]
        bvec_out: [B,O,3]
        return  : [B,O,D,H,W]
        """
        B, O, D, H, W = node.shape
        bfeat = bvec_6d_antipodal_features(bvec_out)  # [B,O,6]
        h = self.bvec_mlp(bfeat)
        scale = torch.tanh(self.bvec_scale(h)).view(B, O, D, 1, 1)
        shift = self.bvec_shift(h).view(B, O, D, 1, 1)
        return node * (1.0 + scale) + shift

    def forward(self, x, bvec_out=None):
        """
        x        : [B,C,H,W]
        bvec_out : [B,O,3] or [O,3]
        return   : [B,O,H,W]
        """
        if x.ndim != 4:
            raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}")
        B, _, H, W = x.shape
        bvec_out = self._prepare_bvec_out(bvec_out, B, x.device, x.dtype)
        A = self._build_graph(bvec_out)  # [B,O,O]

        node = self.node_proj(x).view(B, self.n_out, self.node_dim, H, W)
        node = self._apply_bvec_film(node, bvec_out)

        for fuse in self.graph_fuse:
            node_self = node
            node_neigh = torch.einsum("boj,bjdhw->bodhw", A, node_self)
            node_diff = node_self - node_neigh
            node_cat = torch.cat([node_self, node_neigh, node_diff], dim=2)
            node_cat = node_cat.reshape(B * self.n_out, self.node_dim * 3, H, W)
            node = fuse(node_cat).view(B, self.n_out, self.node_dim, H, W)
            node = self._apply_bvec_film(node, bvec_out)

        # Direction-wise readout: [B,O,D,H,W] * [O,D] -> [B,O,H,W]
        y = (node * self.readout_weight.view(1, self.n_out, self.node_dim, 1, 1)).sum(dim=2)
        y = y + self.readout_bias.view(1, self.n_out, 1, 1)
        return y


# Backward-compatible alias. The training script can keep using graph_node_dim,
# graph_layers, graph_topk and graph_tau without modification.
SphericalGraphReadoutHead = SelfPreservedSphericalGraphReadoutHead


class ObservedDirectionDataConsistency(nn.Module):
    """
    Parameter-free observed-direction data consistency (ODDC).

    Given a raw 90-direction prediction and the physically acquired sparse
    input channels, the observed directions are restored exactly at their
    corresponding target indices.
    """
    def __init__(self, fixed_indices=None):
        super(ObservedDirectionDataConsistency, self).__init__()
        self.register_buffer("fixed_indices", torch.empty(0, dtype=torch.long), persistent=False)
        if fixed_indices is not None:
            self.set_fixed_indices(fixed_indices)

    def set_fixed_indices(self, fixed_indices):
        idx = torch.as_tensor(list(fixed_indices), dtype=torch.long)
        if idx.ndim != 1 or idx.numel() <= 0:
            raise ValueError("fixed_indices must be a non-empty 1D sequence.")
        if torch.unique(idx).numel() != idx.numel():
            raise ValueError(f"fixed_indices contains duplicates: {idx.tolist()}")
        self.fixed_indices = idx

    def forward(self, raw_pred, x_obs):
        if self.fixed_indices.numel() == 0:
            raise ValueError(
                "ODDC fixed_indices are not set. Pass fixed_indices when constructing CNN_2D "
                "or call model.set_fixed_indices(...)."
            )
        if raw_pred.ndim != 4 or x_obs.ndim != 4:
            raise ValueError(
                f"Expected raw_pred/x_obs as [B,C,H,W], got {tuple(raw_pred.shape)} and {tuple(x_obs.shape)}"
            )
        if raw_pred.shape[0] != x_obs.shape[0] or raw_pred.shape[-2:] != x_obs.shape[-2:]:
            raise ValueError(
                f"raw_pred/x_obs shape mismatch: {tuple(raw_pred.shape)} vs {tuple(x_obs.shape)}"
            )
        if self.fixed_indices.numel() != x_obs.shape[1]:
            raise ValueError(
                f"fixed_indices K={self.fixed_indices.numel()} but input has K={x_obs.shape[1]} channels"
            )
        if int(self.fixed_indices.max()) >= raw_pred.shape[1] or int(self.fixed_indices.min()) < 0:
            raise ValueError(
                f"fixed_indices={self.fixed_indices.tolist()} are incompatible with output channels={raw_pred.shape[1]}"
            )

        idx = self.fixed_indices.to(device=raw_pred.device)
        final_pred = raw_pred.clone()
        final_pred[:, idx, :, :] = x_obs.to(device=raw_pred.device, dtype=raw_pred.dtype)
        return final_pred


class CNN_2D(nn.Module):
    """
    Deep LGM-Net with two LG-Mamba blocks and SGR-v2 output head.

    Backbone is unchanged from model_2d_deep_lgm.py. Only the output head is
    changed from Out_Block(256,90) / SGR-v1 to SelfPreservedSphericalGraphReadoutHead.
    """
    def __init__(
        self,
        n_channels=10,
        n_out=90,
        res_hiddens=(128, 256),
        mamba_scans="hv",
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        graph_node_dim=16,
        graph_layers=1,
        graph_topk=8,
        graph_tau=0.15,
        fixed_bvec_out=None,
        fixed_indices=None,
    ):
        super(CNN_2D, self).__init__()

        if len(res_hiddens) != 2:
            raise ValueError("Deep LGM-Net expects res_hiddens=(128, 256).")

        c1, c2 = int(res_hiddens[0]), int(res_hiddens[1])

        self.block1 = Res_Block(n_channels, c1)
        self.block2 = LG_Mamba_Block(
            c1,
            c2,
            scans=mamba_scans,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.block3 = LG_Mamba_Block(
            c2,
            c2,
            scans=mamba_scans,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.out = SelfPreservedSphericalGraphReadoutHead(
            C_in=c2,
            n_out=n_out,
            node_dim=graph_node_dim,
            graph_layers=graph_layers,
            topk=graph_topk,
            tau=graph_tau,
            fixed_bvec_out=fixed_bvec_out,
        )
        self.oddc = ObservedDirectionDataConsistency(fixed_indices=fixed_indices)

        if not HAS_MAMBA:
            print(
                "[Warning] mamba_ssm is not installed. SpatialMamba2D is using an MLP fallback "
                "only for debugging. For formal LG-Mamba experiments, install mamba_ssm."
            )

    def set_fixed_bvec_out(self, fixed_bvec_out):
        self.out.set_fixed_bvec_out(fixed_bvec_out)

    def set_fixed_indices(self, fixed_indices):
        self.oddc.set_fixed_indices(fixed_indices)

    def forward(self, x, bvec_out=None):
        # Keep the physically acquired sparse measurements for ODDC.
        x_obs = x

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Raw 90-direction prediction generated by the learnable network.
        raw_pred = self.out(x, bvec_out=bvec_out)

        # Final measurement-consistent prediction used for inference/downstream fitting.
        final_pred = self.oddc(raw_pred, x_obs)

        # Return BOTH outputs:
        #   raw_pred   -> training loss / raw validation
        #   final_pred -> final reconstruction / downstream DTI-NODDI fitting
        return raw_pred, final_pred


def main():
    num_channels = 10
    num_out = 90
    batch_size = 1
    fixed_indices = list(range(num_channels))
    model = CNN_2D(
        num_channels,
        num_out,
        mamba_scans="h",
        graph_node_dim=16,
        graph_layers=1,
        fixed_indices=fixed_indices,
    )
    dummy_input = torch.rand(batch_size, num_channels, 32, 32)
    dummy_bvec_out = torch.randn(batch_size, num_out, 3)
    dummy_bvec_out = normalize_bvec(dummy_bvec_out)
    with torch.no_grad():
        dummy_raw, dummy_final = model(dummy_input, bvec_out=dummy_bvec_out)
    print(f"For Deep LGM-SGR-v2 with input of size {dummy_input.shape}:")
    print(f"The bvec_out size is {dummy_bvec_out.shape}")
    print(f"The raw output size is {dummy_raw.shape}")
    print(f"The final ODDC output size is {dummy_final.shape}")
    print(f"HAS_MAMBA={HAS_MAMBA}")


if __name__ == "__main__":
    main()

