"""Typed experiment configs: YAML files plus dotted command-line overrides.

A config is a tree of dataclasses. A YAML file and `key.path=value`
overrides are merged over the dataclass defaults, strictly: an unknown key
is an error, not a silently ignored typo. Values are parsed as YAML, so
`train.steps=2000`, `model.use_rope=true` and `data.n_puzzles=null` work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
import types
import typing
from pathlib import Path
from typing import Any

import yaml


def to_dict(obj) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(obj)))


def _dataclass_type(hint) -> type | None:
    """The dataclass inside `hint` (plain or Optional), else None."""
    if dataclasses.is_dataclass(hint):
        return hint
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        found = [a for a in typing.get_args(hint) if dataclasses.is_dataclass(a)]
        return found[0] if len(found) == 1 else None
    return None


def from_dict(cls, data: dict, where: str = ""):
    """Build dataclass `cls` from a plain dict, recursing into nested
    dataclass fields and turning lists back into tuples where typed so."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} in {where or cls.__name__}; "
                       f"valid: {sorted(names)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        sub = _dataclass_type(hint)
        if sub is not None and isinstance(value, dict):
            value = from_dict(sub, value, f"{where}{name}.")
        elif isinstance(value, list) and typing.get_origin(hint) is tuple:
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def merge(base: dict, update: dict, where: str = "") -> dict:
    """Recursive merge of `update` into a copy of `base`; keys must exist."""
    out = dict(base)
    for key, value in update.items():
        if key not in base:
            raise KeyError(f"unknown config key {where}{key}; valid: {sorted(base)}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = merge(base[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def parse_overrides(text: str | list[str]) -> dict:
    """'a.b=1 c=x' (or a list of such tokens) -> nested dict."""
    tokens = shlex.split(text) if isinstance(text, str) else list(text)
    out: dict[str, Any] = {}
    for token in tokens:
        key, sep, raw = token.partition("=")
        if not sep or not key:
            raise ValueError(f"override {token!r} is not of the form key.path=value")
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        else:
            out[path] = value
    return out


def load(cls, path: str | Path | None = None, overrides: str | list[str] = ""):
    """Defaults of `cls`, then the YAML file, then the overrides.

    Keys listed in `cls.DERIVED` are computed from other fields (e.g. model
    grid size from the dataset) and may not be set by hand.
    """
    data = to_dict(cls())
    derived = set(getattr(cls, "DERIVED", ()))
    for update in (yaml.safe_load(Path(path).read_text()) if path else None,
                   parse_overrides(overrides) if overrides else None):
        if update:
            clash = derived & set(flatten(update))
            if clash:
                raise KeyError(f"{sorted(clash)} are derived from other fields; do not set them")
            data = merge(data, update)
    return from_dict(cls, data)


def stable_hash(d: dict, length: int = 12) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:length]
