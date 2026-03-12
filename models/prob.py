# =============================================================================
# probe_encoder.py
#   • Wraps your frozen LadderSymEncoder
#   • Adds four optional heads:
#       - ColumnProbe for a1 and a2  (local correspondence)
#       - GlobalProbe  for a1 and a2  (global correspondence)
#   • Toggle behaviour with probe_local / probe_global flags.
# =============================================================================
import torch
import torch.nn as nn
from typing import Optional
# ----------------------------------------------------------------------------- 
# 1.  Small probe heads
# -----------------------------------------------------------------------------
class ColumnProbe(nn.Module):
    """Predict spectrogram column index (0 … n_cols-1) from a single token."""
    def __init__(self, embed_dim: int, n_cols: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_cols)

    def forward(self, token):               # token: [B, D]
        return self.head(token)             # [B, n_cols]


class GlobalProbe(nn.Module):
    """Predict a global sequence label (C classes) from mean‑pooled tokens."""
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, z):                   # z: [B, N, D]
        pooled = z.mean(dim=1)              # simple mean‑pool across time
        return self.head(pooled)            # [B, n_classes]

# ----------------------------------------------------------------------------- 
# 2.  Wrapper that adds probes to your frozen LadderSym encoder
# -----------------------------------------------------------------------------
class LocalGlobalTester(nn.Module):
    """
    Wraps a frozen LadderSymEncoder and, depending on flags, trains:
      • local column‑index probes  (one for each stream)
      • global statistic probes    (one for each stream)
    """
    def __init__(self,
                 encoder: nn.Module,
                 n_cols: int,
                 n_global: int,
                 probe_local: bool = False,
                 probe_global: bool = False):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():   # freeze encoder weights
            p.requires_grad_(False)

        self.probe_local  = probe_local
        self.probe_global = probe_global
        d_out = encoder.proj.out_features     # projected embed dim

        if probe_local:
            self.col_a1 = ColumnProbe(d_out, n_cols)
            self.col_a2 = ColumnProbe(d_out, n_cols)
        if probe_global:
            self.glob_a1 = GlobalProbe(d_out, n_global)
            self.glob_a2 = GlobalProbe(d_out, n_global)

    def forward(self, a1, a2):
        """
        Returns a dict with logits (and internal idx) depending on flags.
        Caller is responsible for building the loss with proper targets.
        """
        z = self.encoder(a1, a2)              # [B, N_total, D]
        n_cols = z.shape[1] // 2
        z_a1, z_a2 = z[:, :n_cols], z[:, n_cols:]

        out = {}
        if self.probe_local:
            idx = torch.randint(0, n_cols, (z.size(0),), device=z.device)
            tok_a1 = z_a1[torch.arange(z.size(0)), idx]   # [B, D]
            tok_a2 = z_a2[torch.arange(z.size(0)), idx]
            out["logits_a1_local"] = self.col_a1(tok_a1)
            out["logits_a2_local"] = self.col_a2(tok_a2)
            out["targets_local"]   = idx                  # use for CE loss

        if self.probe_global:
            out["logits_a1_global"] = self.glob_a1(z_a1)
            out["logits_a2_global"] = self.glob_a2(z_a2)
            # caller must supply `targets_global`

        return out

# ----------------------------------------------------------------------------- 
# 3.  Example training loop snippet (stand‑alone test)
#     Replace `DummyEncoder` with your actual LadderSymEncoder instance.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # -------- dummy encoder with same interface ----------
    class DummyEncoder(nn.Module):
        def __init__(self, proj_dim=512):
            super().__init__()
            self.proj = nn.Linear(32, proj_dim, bias=False)
        def forward(self, a1, a2):
            B, T, F = a1.shape
            z = torch.randn(B, T, 32, device=a1.device)
            return self.proj(z)  # fake patch sequence

    encoder = DummyEncoder()
    tester = LocalGlobalTester(
        encoder      = encoder,
        n_cols       = 16,      # number of patch columns per stream
        n_global     = 3,       # e.g., low / mid / high peak energy
        probe_local  = True,
        probe_global = True,
    )

    # fake batch
    B, T, F = 4, 32, 32
    a1 = torch.randn(B, T, F)
    a2 = torch.randn(B, T, F)
    global_target = torch.randint(0, 3, (B,))

    # forward
    outs = tester(a1, a2)

    # build simple probe loss
    loss = 0
    if "logits_a1_local" in outs:
        ce = nn.CrossEntropyLoss()
        loss += ce(outs["logits_a1_local"], outs["targets_local"])
        loss += ce(outs["logits_a2_local"], outs["targets_local"])
    if "logits_a1_global" in outs:
        ce = nn.CrossEntropyLoss()
        loss += ce(outs["logits_a1_global"], global_target)
        loss += ce(outs["logits_a2_global"], global_target)

    print("loss =", loss.item())