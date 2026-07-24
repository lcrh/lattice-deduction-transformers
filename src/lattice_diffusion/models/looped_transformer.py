"""Looped transformer with stable iteration mechanics.

Architecture: `h = LayerNorm(h + backbone(h))` with input re-injection each
loop. Stable to 128+ loops with improving NLL — the plain residual
`h = h + backbone(h)` diverges around ~96 loops, so the LayerNorm wrap on
the outer residual is load-bearing.

Default forward returns mean-across-iterations predictions (more stable,
higher soundness). Use return_all=True for per-iteration outputs (training).

Independent implementation. Related prior work on iterated / recurrent
transformer backbones: Dehghani et al. 2019 (Universal Transformer);
Wang et al. 2025 (HRM, arXiv:2506.21734); Jolicoeur-Martineau et al. 2025
(TRM).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from lattice_diffusion.models.transformer2d import Transformer2D, Transformer2DConfig


@dataclass
class LoopedTransformerConfig:
    dim: int = 128
    seq_len: int = 81
    n_channels: int = 9
    grid_rows: int = 9
    grid_cols: int = 9
    num_layers: int = 4
    n_heads: int = 4
    n_loops: int = 16
    # Outer solve-step carry architecture. "off" is the legacy one-stream
    # model; "h" reuses the final head-feeding hidden; "z" uses a separate
    # TRM-style scratchpad and answer stream.
    carry_mode: str = "off"

    dropout: float = 0.1
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    attention_bias: bool = True
    ffn_mult: float = 4.0
    ffn_bias: bool = True

    cls_token: bool = False
    pre_norm: bool = True

    use_rope: bool = False
    rope_base: float = 10000.0


class PowersetModel(nn.Module):
    """Looped transformer with normalized residual and input re-injection.

    Each loop:
        h = h + h0                        # re-inject input
        h = LayerNorm(h + backbone(h))     # normalized outer residual

    Heads:
    - bce_head: per-candidate keep/eliminate logits [B, S, C]
    - softmax_head: per-cell digit prediction logits [B, S, C]
    - conflict_head: grid-level SAT/UNSAT logit [B, 1] (requires cls_token=True)

    Default forward returns mean-across-iterations (averaged sigmoid/logits).
    return_all=True returns per-iteration outputs for training loss.
    """

    def __init__(self, cfg: LoopedTransformerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.carry_mode not in {"off", "h", "z"}:
            raise ValueError(
                f"carry_mode must be one of off|h|z, got {cfg.carry_mode!r}"
            )
        if cfg.carry_mode == "z" and cfg.n_loops % 2:
            raise ValueError(
                "carry_mode='z' uses one scratchpad and one answer backbone "
                f"application per cycle, so n_loops must be even (got {cfg.n_loops})"
            )

        self.input_proj = nn.Linear(cfg.n_channels, cfg.dim, bias=True)
        backbone_cfg = Transformer2DConfig(
            dim=cfg.dim,
            seq_len=cfg.seq_len,
            grid_rows=cfg.grid_rows,
            grid_cols=cfg.grid_cols,
            num_layers=cfg.num_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            ffn_dropout=cfg.ffn_dropout,
            attention_bias=cfg.attention_bias,
            ffn_mult=cfg.ffn_mult,
            ffn_bias=cfg.ffn_bias,
            cls_token=cfg.cls_token,
            pre_norm=cfg.pre_norm,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
        )
        self.backbone = Transformer2D(backbone_cfg)
        self.loop_norm = nn.LayerNorm(cfg.dim)
        self.bce_head = nn.Linear(cfg.dim, cfg.n_channels, bias=True)
        self.softmax_head = nn.Linear(cfg.dim, cfg.n_channels, bias=True)
        if cfg.cls_token:
            self.conflict_head = nn.Linear(cfg.dim, 1, bias=True)
        # Instantiate carry-only parameters after every legacy parameter, so
        # adding the projection cannot perturb shared-weight initialization
        # under the same seed. Off checkpoints retain their exact schema.
        if cfg.carry_mode in {"h", "z"}:
            self.carry_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
            nn.init.zeros_(self.carry_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        use_final: bool = False,
        *,
        orig_x: torch.Tensor | None = None,
        carry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass.

        Default (return_all=False, use_final=False): returns mean-across-iterations.
        use_final=True: returns only the last iteration's predictions.
        return_all=True: returns list of per-iteration predictions (for training).

        Returns dict with keys:
        - "bce": [B, S, C] mean logits, or list of per-loop [B, S, C]
        - "softmax": [B, S, C] mean logits, or list of per-loop [B, S, C]
        - "cls": [B, dim] mean hidden, or list of per-loop [B, dim] (when cls_token=True)
        """
        all_bce = []
        all_softmax = []
        all_cls = []
        all_conflict = []

        if self.cfg.carry_mode in {"off", "h"}:
            # Keep the legacy path literal when carry_mode="off": same
            # operations, output ordering, RNG consumption, and state_dict.
            h0 = self.input_proj(x)  # [B, S, dim]
            h = h0
            if self.cfg.carry_mode == "h" and carry is not None:
                h = h + self.carry_proj(carry)

            for _ in range(self.cfg.n_loops):
                h = h + h0  # re-inject input

                out = self.backbone(h)
                if self.cfg.cls_token:
                    cell_out = out[:, :self.cfg.seq_len, :]
                    cls_h = out[:, self.cfg.seq_len, :]
                    h = self.loop_norm(h + cell_out)
                else:
                    h = self.loop_norm(h + out)

                bce = self.bce_head(h)
                softmax = self.softmax_head(h)
                all_bce.append(bce)
                all_softmax.append(softmax)
                if self.cfg.cls_token:
                    all_cls.append(cls_h)
                    all_conflict.append(self.conflict_head(cls_h))
            carry_out = h if self.cfg.carry_mode == "h" else None
        else:
            if orig_x is None:
                raise ValueError("carry_mode='z' requires orig_x (the immutable puzzle)")
            puzzle_h = self.input_proj(orig_x)
            y = self.input_proj(x)
            z = (
                torch.zeros_like(y)
                if carry is None
                else self.carry_proj(carry)
            )

            # Compute-match the legacy n_loops backbone applications: each
            # cycle spends one application updating the opaque scratchpad and
            # one refining the answer stream. Only answer refinements are read
            # by heads, giving n_loops/2 deeply-supervised readouts.
            for _ in range(self.cfg.n_loops // 2):
                z_input = puzzle_h + y + z
                z_out = self.backbone(z_input)
                z_cells = z_out[:, :self.cfg.seq_len, :] if self.cfg.cls_token else z_out
                z = self.loop_norm(z_input + z_cells)

                y_input = y + z
                y_out = self.backbone(y_input)
                if self.cfg.cls_token:
                    y_cells = y_out[:, :self.cfg.seq_len, :]
                    cls_h = y_out[:, self.cfg.seq_len, :]
                    y = self.loop_norm(y_input + y_cells)
                else:
                    y = self.loop_norm(y_input + y_out)

                all_bce.append(self.bce_head(y))
                all_softmax.append(self.softmax_head(y))
                if self.cfg.cls_token:
                    all_cls.append(cls_h)
                    all_conflict.append(self.conflict_head(cls_h))
            carry_out = z

        result = {}
        if return_all:
            result["bce"] = all_bce
            result["softmax"] = all_softmax
            if self.cfg.cls_token:
                result["cls"] = all_cls
                result["conflict"] = all_conflict
        elif use_final:
            result["bce"] = all_bce[-1]
            result["softmax"] = all_softmax[-1]
            if self.cfg.cls_token:
                result["cls"] = all_cls[-1]
                result["conflict"] = all_conflict[-1]
        else:
            # Mean across iterations
            result["bce"] = torch.stack(all_bce, dim=0).mean(dim=0)
            result["softmax"] = torch.stack(all_softmax, dim=0).mean(dim=0)
            if self.cfg.cls_token:
                result["cls"] = torch.stack(all_cls, dim=0).mean(dim=0)
                result["conflict"] = self.conflict_head(result["cls"])

        if carry_out is not None:
            result["carry_out"] = carry_out
        return result
