from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Transformer2DConfig:
    dim: int = 128
    seq_len: int = 81
    num_layers: int = 4
    n_heads: int = 4

    pre_norm: bool = True
    norm_eps: float = 1e-5
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    attention_bias: bool = True

    ffn_mult: float = 4.0
    ffn_bias: bool = True

    grid_rows: int = 9
    grid_cols: int = 9

    final_norm: bool = False
    cls_token: bool = False  # append a learnable CLS token to the sequence

    # Position encoding choice. Default = learned 2D row+col embeddings.
    # `use_rope=True` swaps to 2D RoPE applied to Q/K inside attention;
    # learned row/col embeddings are skipped in that case. Useful for
    # large grids where learned absolute embeddings can struggle to
    # extrapolate.
    use_rope: bool = False
    rope_base: float = 10000.0


def _precompute_2d_rope(
    head_dim: int, grid_rows: int, grid_cols: int, base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape [seq_len, head_dim] for 2D RoPE.

    Splits head_dim in half: first half rotates by ROW position, second half
    by COL position. Within each half, dim pairs (2i, 2i+1) get inverse-frequency
    `base ** (-2i / half)`. Requires head_dim % 4 == 0.
    """
    assert head_dim % 4 == 0, f"2D RoPE needs head_dim % 4 == 0, got {head_dim}"
    half = head_dim // 2
    n_pairs = half // 2
    inv_freq = base ** (-torch.arange(n_pairs, dtype=torch.float32) * 2 / half)
    rows = torch.arange(grid_rows, dtype=torch.float32)
    cols = torch.arange(grid_cols, dtype=torch.float32)

    row_angles = rows[:, None] * inv_freq[None, :]   # [grid_rows, n_pairs]
    col_angles = cols[:, None] * inv_freq[None, :]   # [grid_cols, n_pairs]
    seq_len = grid_rows * grid_cols
    row_per_tok = row_angles.unsqueeze(1).expand(grid_rows, grid_cols, n_pairs).reshape(seq_len, n_pairs)
    col_per_tok = col_angles.unsqueeze(0).expand(grid_rows, grid_cols, n_pairs).reshape(seq_len, n_pairs)

    cos = torch.zeros(seq_len, head_dim)
    sin = torch.zeros(seq_len, head_dim)
    # Row half: pairs (0,1), (2,3), ... — both elements share cos/sin.
    cos[:, 0:half:2] = torch.cos(row_per_tok)
    cos[:, 1:half:2] = torch.cos(row_per_tok)
    sin[:, 0:half:2] = torch.sin(row_per_tok)
    sin[:, 1:half:2] = torch.sin(row_per_tok)
    # Col half: same pattern but for col angles.
    cos[:, half::2] = torch.cos(col_per_tok)
    cos[:, half + 1::2] = torch.cos(col_per_tok)
    sin[:, half::2] = torch.sin(col_per_tok)
    sin[:, half + 1::2] = torch.sin(col_per_tok)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (2i, 2i+1) pairs of x [..., L, head_dim] by angles with cos/sin [L, head_dim]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_pair = cos[..., 0::2]   # one entry per pair
    sin_pair = sin[..., 0::2]
    new_even = x_even * cos_pair - x_odd * sin_pair
    new_odd = x_even * sin_pair + x_odd * cos_pair
    return torch.stack([new_even, new_odd], dim=-1).flatten(-2, -1)


class _RoPEAttention(nn.Module):
    """Multi-head attention with 2D RoPE on the grid tokens' Q/K (the CLS token is not rotated)."""

    def __init__(self, cfg: "Transformer2DConfig"):
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.cls_token = cfg.cls_token
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.attention_bias)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.attention_bias)
        self.attn_dropout_p = cfg.attn_dropout

        cos, sin = _precompute_2d_rope(
            self.head_dim, cfg.grid_rows, cfg.grid_cols, base=cfg.rope_base,
        )
        if cfg.cls_token:
            # CLS at position seq_len: pad cos/sin with (1, 0) → no rotation.
            pad_cos = torch.ones(1, self.head_dim)
            pad_sin = torch.zeros(1, self.head_dim)
            cos = torch.cat([cos, pad_cos], dim=0)
            sin = torch.cat([sin, pad_sin], dim=0)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, hd]
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # Apply RoPE to Q, K (broadcasts cos/sin across batch and heads).
        cos = self.rope_cos[:L]  # [L, hd]
        sin = self.rope_sin[:L]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )  # [B, H, L, hd]
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn)


class FeedForward(nn.Module):
    def __init__(self, cfg: Transformer2DConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_mult)
        self.fc1 = nn.Linear(cfg.dim, hidden, bias=cfg.ffn_bias)
        self.fc2 = nn.Linear(hidden, cfg.dim, bias=cfg.ffn_bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Transformer2DConfig):
        super().__init__()
        self.pre_norm = cfg.pre_norm
        self.use_rope = cfg.use_rope
        self.norm1 = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps)
        self.norm2 = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps)
        if cfg.use_rope:
            self.attn = _RoPEAttention(cfg)
        else:
            self.attn = nn.MultiheadAttention(
                cfg.dim,
                cfg.n_heads,
                dropout=cfg.attn_dropout,
                bias=cfg.attention_bias,
                batch_first=True,
            )
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.ffn = FeedForward(cfg)

    def _attn(self, h: torch.Tensor) -> torch.Tensor:
        if self.use_rope:
            return self.attn(h)
        return self.attn(h, h, h, need_weights=False)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            h = self.norm1(x)
            x = x + self.attn_dropout(self._attn(h))
            x = x + self.ffn(self.norm2(x))
            return x

        h = x + self.attn_dropout(self._attn(x))
        x = self.norm1(h)
        h = x + self.ffn(x)
        x = self.norm2(h)
        return x


class Transformer2D(nn.Module):
    """Transformer encoder with learned 2D positional embeddings."""

    def __init__(self, cfg: Transformer2DConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = Transformer2DConfig()
        self.cfg = cfg

        if cfg.dim % cfg.n_heads != 0:
            raise ValueError(f"dim ({cfg.dim}) must be divisible by n_heads ({cfg.n_heads}).")
        if cfg.grid_rows * cfg.grid_cols != cfg.seq_len:
            raise ValueError(
                f"grid_rows * grid_cols must equal seq_len. "
                f"Got {cfg.grid_rows}*{cfg.grid_cols} != {cfg.seq_len}."
            )

        # Learned 2D position embeddings — disabled when RoPE is enabled
        # (RoPE encodes position relatively inside attention, no need for
        # absolute embedding addition).
        self.use_rope = cfg.use_rope
        if not cfg.use_rope:
            self.row_embed = nn.Embedding(cfg.grid_rows, cfg.dim)
            self.col_embed = nn.Embedding(cfg.grid_cols, cfg.dim)
            rows = torch.arange(cfg.grid_rows).unsqueeze(1).expand(cfg.grid_rows, cfg.grid_cols).reshape(cfg.seq_len)
            cols = torch.arange(cfg.grid_cols).unsqueeze(0).expand(cfg.grid_rows, cfg.grid_cols).reshape(cfg.seq_len)
            self.register_buffer("row_idx", rows)
            self.register_buffer("col_idx", cols)

        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps) if cfg.final_norm else nn.Identity()

        if cfg.cls_token:
            self.cls_embed = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
            self.cls_pos = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, L, D], got {tuple(x.shape)}")

        # Add 2D positional embeddings to grid tokens (only when not using RoPE).
        if not self.use_rope:
            x = x + self.row_embed(self.row_idx).unsqueeze(0) + self.col_embed(self.col_idx).unsqueeze(0)

        # Append CLS token with its own positional embedding
        if self.cfg.cls_token:
            cls = (self.cls_embed + self.cls_pos).expand(x.shape[0], -1, -1)
            x = torch.cat([x, cls], dim=1)  # [B, seq_len+1, dim]

        for layer in self.layers:
            x = layer(x)

        return self.final_norm(x)
