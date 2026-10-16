"""Exponential moving average of the trainable parameters.

`swap_in` puts the shadow weights into the live model and `swap_out` restores them.
"""

from __future__ import annotations

import torch


class ModelEMA:
    """Simple EMA wrapper that shadows trainable params on the unwrapped model."""

    def __init__(self, base: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in base.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, base: torch.nn.Module) -> None:
        for name, param in base.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, base: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, param in base.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            backup[name] = param.detach().clone()
            param.data.copy_(shadow)
        return backup

    @torch.no_grad()
    def swap_out(self, base: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in base.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)


@torch.no_grad()
def swap_in_ema_if_present(model: torch.nn.Module, ckpt: dict) -> None:
    """At eval time, copy EMA weights from the loaded checkpoint into the
    live model in-place. No-op if the checkpoint has no EMA state. The
    caller doesn't need to swap back; the model is being used for eval only.

    Two name-mangling cases handled here:
    1. `save_checkpoint` does `data.update(extra)` so `ema_state_dict` is
       at the top level of the loaded dict — pass the whole `ckpt`, not
       `ckpt["extra"]`.
    2. EMA was constructed against a `torch.compile`-wrapped model whose
       `named_parameters()` uses `_orig_mod.<name>` keys. The live model
       at eval time has unprefixed names. Strip `_orig_mod.` from EMA
       keys before matching.
    """
    ema_state = ckpt.get("ema_state_dict") if isinstance(ckpt, dict) else None
    if not ema_state:
        return
    # Strip `_orig_mod.` prefix from EMA shadow keys (added by torch.compile).
    stripped: dict[str, torch.Tensor] = {
        k.replace("_orig_mod.", ""): v for k, v in ema_state.items()
    }
    n_copied = 0
    for name, param in model.named_parameters():
        shadow = stripped.get(name)
        if shadow is None:
            continue
        param.data.copy_(shadow)
        n_copied += 1
    print(f"  Loaded EMA weights into model: {n_copied} param tensors", flush=True)
