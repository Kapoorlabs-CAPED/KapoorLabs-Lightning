"""Shared helper: read the ``training_config.json`` Hydra dump that ships
next to every Lightning checkpoint.

Each ``CareInception``-trained model writes a ``training_config.json``
into ``log_path`` containing the full Hydra config (``train_data_paths``
+ ``parameters``). Every ``predict-*.py`` script in this folder consults
this file *first* — values from the JSON win over the Hydra parameter
yaml defaults, so models trained with a different arch than the
current yaml still predict correctly without anyone editing config.

If the JSON is missing the script falls back to the Hydra defaults.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_arch_from_training_config(log_path: str) -> dict:
    """Return the ``parameters`` block from ``training_config.json``.

    Returns an empty dict when the file is missing — callers should
    then fall back to the Hydra parameter yaml. The whole ``parameters``
    sub-dict is returned (not a curated subset) so each script can
    pluck out only the arch / hyperparam keys it needs via ``.get``.
    """
    p = Path(log_path) / "training_config.json"
    if not p.is_file():
        return {}
    with p.open() as fh:
        blob = json.load(fh)
    return blob.get("parameters", blob)
