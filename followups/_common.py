"""Shared helpers for followup experiments (E1 arch_ablation, E4 ood_snowflake, ...).

Small, dependency-light module holding the two pieces every followup's
`configs.py` / `collect.py` needs and would otherwise duplicate:

  * `flag(name)`            underscore -> `--dashed` CLI flag form.
  * `launch_command(...)`   build a `uv run modal run --detach ... -- <flags>`
                            command from an ordered effective-flag dict.
  * `volume_done_set(...)`  which `<config>_seed<N>` runs already have a landed
                            `.eval.json` on the Modal checkpoints volume
                            (the "is this (config, seed) done?" query).
  * `read_volume_text(...)` read one volume file to text, None if missing.

`modal` is imported lazily inside the volume helpers so that `configs.py list`
(which needs none of it) runs with no modal dependency. Both E1 and E4 import
this module — keep it experiment-agnostic (no hard-coded config matrix here).

Followup scripts run from the repo root (`uv run python
followups/<exp>/collect.py`), so absolute package imports resolve via cwd —
no sys.path manipulation:

    from followups import _common
"""

from __future__ import annotations

# Default Modal volume that training writes checkpoints to (see
# src/lattice_diffusion/modal/image.py: checkpoint_volume).
VOLUME_NAME = "lattice-diffusion-checkpoints"

RUN_ENTRYPOINT = "experiments/sudoku/run.py"


def flag(name: str) -> str:
    """`some_name` -> `--some-name` (Modal CLI flag form)."""
    return "--" + name.replace("_", "-")


def launch_command(
    config_name: str,
    seed: int,
    eff_flags: dict,
    *,
    ckpt_subdir: str,
    entrypoint: str = RUN_ENTRYPOINT,
    flag_order: tuple[str, ...] = (),
    skip_if_done: bool = True,
) -> str:
    """Build a single `uv run modal run --detach <entrypoint> -- <flags>` command.

    `eff_flags` is the effective (BASE + overrides) flag dict for this config;
    only keys present are emitted. `flag_order` fixes the emission order for
    readability (keys not listed are appended in dict order). The deterministic
    exchange path is set via `--ckpt-subdir` / `--ckpt-name <config>_seed<N>`.
    """
    parts = [f"uv run modal run --detach {entrypoint} --"]
    parts.append(f"{flag('ckpt_subdir')} {ckpt_subdir}")
    parts.append(f"{flag('ckpt_name')} {config_name}_seed{seed}")
    parts.append(f"{flag('seed')} {seed}")
    emitted = {"ckpt_subdir", "ckpt_name", "seed"}
    ordered = list(flag_order) + [k for k in eff_flags if k not in flag_order]
    for name in ordered:
        if name in emitted or name not in eff_flags:
            continue
        parts.append(f"{flag(name)} {eff_flags[name]}")
        emitted.add(name)
    if skip_if_done:
        parts.append("--skip-if-done")
    return " \\\n    ".join(parts)


def open_volume(volume_name: str = VOLUME_NAME):
    """Return a Modal Volume handle (lazy modal import). Raises on modal error."""
    from modal import Volume
    return Volume.from_name(volume_name)


def volume_done_set(subdir: str, *, volume_name: str = VOLUME_NAME,
                    suffix: str = ".eval.json") -> set[str]:
    """Set of `<config>_seed<N>` basenames that have a landed `<suffix>` file.

    Queries `/<subdir>/` on the volume. Raises on any modal/volume/auth error;
    callers decide how to degrade (status/remaining print a message + exit 1).
    """
    vol = open_volume(volume_name)
    try:
        vol.reload()
    except Exception:  # noqa: BLE001 — reload is best-effort metadata refresh
        pass
    done: set[str] = set()
    for entry in vol.iterdir(f"/{subdir}"):
        base = entry.path.rsplit("/", 1)[-1]
        if base.endswith(suffix):
            done.add(base[: -len(suffix)])
    return done


def read_volume_text(vol, path: str) -> str | None:
    """Read one volume file to a decoded str; None if it does not exist.

    Non-missing errors (auth, network) propagate so the caller can abort.
    """
    try:
        chunks = [c for c in vol.read_file(path)]
        return b"".join(chunks).decode("utf-8")
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 — some modal versions raise GRPCError
        msg = str(exc).lower()
        if "not found" in msg or "no such" in msg or "missing" in msg:
            return None
        raise
