"""Partición temporal por usuario: primeros 70% train, siguientes 15% val, últimos 15% test."""
import json
import pandas as pd
from .common import load_config, path

def main():
    cfg = load_config(); out = path(cfg, "processed"); sp = cfg["split"]
    df = pd.read_parquet(out / "ratings_subset.parquet")
    df = df.sort_values(["userId", "timestamp", "movieId"], kind="mergesort").reset_index(drop=True)
    n = df.groupby("userId").userId.transform("size"); pos = df.groupby("userId").cumcount()
    frac = pos / n
    part = pd.Series("test", index=df.index)
    part[frac < sp["train"] + sp["val"]] = "val"
    part[frac < sp["train"]] = "train"
    rep = {}
    for name in ("train", "val", "test"):
        d = df[part == name].reset_index(drop=True)
        d.to_parquet(out / f"{name}.parquet")
        rep[name] = {"ratings": len(d), "users": int(d.userId.nunique()), "movies": int(d.movieId.nunique()),
                     "t_min": str(pd.to_datetime(d.timestamp.min(), unit="s")), "t_max": str(pd.to_datetime(d.timestamp.max(), unit="s"))}
        if name != "train":
            tr = pd.read_parquet(out / "train.parquet")
            rep[name]["pct_ratings_cold_movie"] = float((~d.movieId.isin(tr.movieId)).mean())
            rep[name]["pct_ratings_cold_user"] = float((~d.userId.isin(tr.userId)).mean())
    (out / "split_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
