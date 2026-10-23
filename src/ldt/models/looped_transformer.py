"""Looped transformer: `h = LayerNorm(h + h0 + backbone(h + h0))` per loop.

The LayerNorm on the outer residual keeps it stable past 128 loops; the plain
residual diverges around 96.

Related work on recurrent transformer backbones: Universal Transformer
(Dehghani et al. 2019), HRM (Wang et al. 2025), TRM (Jolicoeur-Martineau et al. 2025).
"""

from dataclasses import dataclass
from typing import Literal, get_args

import torch
import torch.nn as nn

from ldt.models.transformer2d import Transformer2D, Transformer2DConfig

# Latent state carried from one solve step to the next (TRM-style):
#   off    - none; each step sees only the lattice state
#   h      - the final hidden state, fed back through `carry_proj`
#   z      - a scratchpad stream: each of the n_loops/2 cycles spends one backbone
#            pass updating the scratchpad and one refining the answer stream
#   zero_z - the z architecture with the scratchpad zeroed every step (control)
CarryMode = Literal["off", "h", "z", "zero_z"]
CARRY_MODES = get_args(CarryMode)


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

    carry: CarryMode = "off"

    def __post_init__(self):
        if self.carry not in CARRY_MODES:
            raise ValueError(f"carry must be one of {CARRY_MODES}, got {self.carry!r}")
        if self.carry in ("z", "zero_z") and self.n_loops % 2:
            raise ValueError(f"carry={self.carry!r} alternates scratchpad and answer passes, "
                             f"so n_loops must be even (got {self.n_loops})")

    @property
    def carries_state(self) -> bool:
        """Whether solve steps pass a latent (`carry_out` -> `carry`) along."""
        return self.carry in ("h", "z")

    @property
    def needs_puzzle(self) -> bool:
        """Whether forward needs the original puzzle (`orig_x`)."""
        return self.carry in ("z", "zero_z")


class PowersetModel(nn.Module):
    """Looped transformer with keep/eliminate, digit and (optional) conflict heads."""

    def __init__(self, cfg: LoopedTransformerConfig):
        super().__init__()
        self.cfg = cfg

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
        # Created after every other parameter, so enabling carry does not
        # change their initialization under the same seed. zero_z keeps the
        # (unused) projection so it is initialized exactly like z.
        if cfg.carry != "off":
            self.carry_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
            nn.init.zeros_(self.carry_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        use_final: bool = False,
        n_loops: int | None = None,
        *,
        orig_x: torch.Tensor | None = None,
        carry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run the loop and read out the heads.

        Args:
            x: Lattice state [B, S, C].
            return_all: Return per-loop lists (training).
            use_final: Return only the last loop's outputs.
            n_loops: Override cfg.n_loops (weights are shared across loops).
            orig_x: Original puzzle; required for carry z / zero_z.
            carry: Previous step's `carry_out` [B, S, dim]; None means zeros.

        Returns:
            "bce", "softmax" [B, S, C]; "cls" [B, dim] and "conflict" [B, 1] with
            cls_token; "carry_out" [B, S, dim] for carry h / z. Values are averaged
            over loops unless return_all / use_final.
        """
        n_loops = self.cfg.n_loops if n_loops is None else n_loops
        all_bce = []
        all_softmax = []
        all_cls = []
        all_conflict = []

        if self.cfg.carry in ("off", "h"):
            h0 = self.input_proj(x)  # [B, S, dim]
            h = h0
            if self.cfg.carry == "h" and carry is not None:
                h = h + self.carry_proj(carry)

            for _ in range(n_loops):
                h = h + h0

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
            carry_out = h if self.cfg.carry == "h" else None
        else:
            if orig_x is None:
                raise ValueError(f"carry={self.cfg.carry!r} needs orig_x (the original puzzle)")
            puzzle_h = self.input_proj(orig_x)
            y = self.input_proj(x)
            z = (self.carry_proj(carry) if self.cfg.carry == "z" and carry is not None
                 else torch.zeros_like(y))
            # Same number of backbone passes as the plain model: each cycle
            # updates the scratchpad z, then refines the answer stream y;
            # only y is read out, giving n_loops / 2 supervised readouts.
            for _ in range(n_loops // 2):
                z_input = puzzle_h + y + z
                z_out = self.backbone(z_input)
                z_cells = z_out[:, :self.cfg.seq_len, :] if self.cfg.cls_token else z_out
                z = self.loop_norm(z_input + z_cells)

                y_input = y + z
                y_out = self.backbone(y_input)
                if self.cfg.cls_token:
                    cls_h = y_out[:, self.cfg.seq_len, :]
                    y = self.loop_norm(y_input + y_out[:, :self.cfg.seq_len, :])
                else:
                    y = self.loop_norm(y_input + y_out)

                all_bce.append(self.bce_head(y))
                all_softmax.append(self.softmax_head(y))
                if self.cfg.cls_token:
                    all_cls.append(cls_h)
                    all_conflict.append(self.conflict_head(cls_h))
            carry_out = z if self.cfg.carry == "z" else None

        per_loop = {"bce": all_bce, "softmax": all_softmax}
        if self.cfg.cls_token:
            per_loop |= {"cls": all_cls, "conflict": all_conflict}
        if return_all:
            result = per_loop
        elif use_final:
            result = {k: v[-1] for k, v in per_loop.items()}
        else:
            result = {k: torch.stack(v, dim=0).mean(dim=0)
                      for k, v in per_loop.items() if k != "conflict"}
            if self.cfg.cls_token:
                result["conflict"] = self.conflict_head(result["cls"])

        if carry_out is not None:
            result["carry_out"] = carry_out
        return result
