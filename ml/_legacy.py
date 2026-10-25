"""Loads the existing competition code (code/business_entity_resolution/src) without
adding a generic top-level `src` package to sys.path. The competition code stays the
single source of truth for normalization, features and the trained model.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ER_DIR = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
MODEL_PATH = ER_DIR / "models" / "entity_match_model.joblib"
THRESHOLD_PATH = ER_DIR / "models" / "threshold.txt"


def _load(name: str) -> ModuleType:
    full = f"ml._er_{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, ER_DIR / "src" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


preprocess = _load("preprocess")
features = _load("features")
