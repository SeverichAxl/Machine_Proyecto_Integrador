import os, random
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    path = os.environ.get("ML_CONFIG", ROOT / "config" / "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def path(cfg, key):
    d = ROOT / cfg["paths"][key]
    d.mkdir(parents=True, exist_ok=True)
    return d

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)

def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))

def load_splits(cfg):
    d = path(cfg, "processed")
    return [pd.read_parquet(d / f"{s}.parquet") for s in ("train", "val", "test")]

def load_train_val(cfg):
    """Solo train/val: lo usan los módulos que deciden o ajustan modelos (nunca test)."""
    d = path(cfg, "processed")
    return [pd.read_parquet(d / f"{s}.parquet") for s in ("train", "val")]

def load_movies(cfg):
    return pd.read_parquet(path(cfg, "processed") / "movies.parquet")
