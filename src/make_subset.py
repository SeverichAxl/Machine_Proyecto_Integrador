"""Filtra por actividad mínima y muestrea usuarios completos. Registra cuántos datos se pierden."""
import json
import numpy as np, pandas as pd
from .common import load_config, path, ROOT, set_seed

def main():
    cfg = load_config(); set_seed(cfg["seed"]); s = cfg["subset"]
    raw = ROOT / cfg["paths"]["raw"]; out = path(cfg, "processed")
    r = pd.read_csv(raw / "ratings.csv", dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
    movies = pd.read_csv(raw / "movies.csv")
    rep = {"original": {"ratings": len(r), "users": int(r.userId.nunique()), "movies": int(r.movieId.nunique())}}
    r = r.drop_duplicates(["userId", "movieId"])
    for _ in range(10):  # filtro iterativo hasta estabilizar
        uc, mc = r.userId.map(r.userId.value_counts()), r.movieId.map(r.movieId.value_counts())
        keep = (uc >= s["min_ratings_user"]) & (mc >= s["min_ratings_movie"])
        if keep.all(): break
        r = r[keep]
    rep["after_min_filters"] = {"ratings": len(r), "users": int(r.userId.nunique()), "movies": int(r.movieId.nunique())}
    rng = np.random.default_rng(cfg["seed"])
    users = r.userId.unique()
    if len(users) > s["n_users"]:
        users = rng.choice(users, s["n_users"], replace=False)
    r = r[r.userId.isin(users)]
    r = r[r.movieId.map(r.movieId.value_counts()) >= s["min_ratings_movie_after_sample"]]
    rep["subset"] = {"ratings": len(r), "users": int(r.userId.nunique()), "movies": int(r.movieId.nunique())}
    rep["criteria"] = s
    rep["sparsity"] = 1 - len(r) / (rep["subset"]["users"] * rep["subset"]["movies"])
    r.reset_index(drop=True).to_parquet(out / "ratings_subset.parquet")
    movies[movies.movieId.isin(r.movieId.unique())].reset_index(drop=True).to_parquet(out / "movies.parquet")
    (out / "subset_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
