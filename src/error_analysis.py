"""Análisis de errores del mejor modelo manual (ya elegido en validación) sobre test, por segmentos."""
import json
import joblib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from .common import load_config, load_splits, load_movies, path

def main():
    cfg = load_config(); train, _, test = load_splits(cfg); movies = load_movies(cfg).set_index("movieId")
    models, res, fig = path(cfg, "models"), path(cfg, "results"), path(cfg, "figures")
    name = json.loads((res / "best_config.json").read_text())["best_manual"]
    d = test.copy(); d["pred"] = joblib.load(models / f"{name}_best.joblib").predict(test); d["err"] = d.pred - d.rating; d["abs"] = d.err.abs(); d["sq"] = d.err ** 2
    uc, mc = train.userId.value_counts(), train.movieId.value_counts()
    d["user_activity"] = pd.qcut(d.userId.map(uc).fillna(0), 4, duplicates="drop").astype(str)
    d["movie_popularity"] = pd.qcut(d.movieId.map(mc).fillna(0), 4, duplicates="drop").astype(str)
    yr = movies.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float); d["decade"] = (d.movieId.map(yr) // 10 * 10).astype("Int64").astype(str)
    out = {}
    def seg(col, frame):
        g = frame.groupby(col).agg(n=("err", "size"), bias=("err", "mean"), mae=("abs", "mean"), mse=("sq", "mean")); g["rmse"] = np.sqrt(g.pop("mse")); return g.round(4)
    for col in ("rating", "user_activity", "movie_popularity", "decade"):
        out[col] = seg(col, d); out[col].to_csv(res / f"errors_by_{col}.csv")
    g = d.assign(genre=d.movieId.map(movies.genres).str.split("|")).explode("genre"); out["genre"] = seg("genre", g); out["genre"].to_csv(res / "errors_by_genre.csv")
    d["cold_movie"] = ~d.movieId.isin(mc.index); out["cold_movie"] = seg("cold_movie", d); out["cold_movie"].to_csv(res / "errors_cold_movie.csv")
    cluster_file = res / "cluster_assignments.csv"
    if cluster_file.exists():
        ca = pd.read_csv(cluster_file)[["movieId", "cluster"]]
        d = d.merge(ca, on="movieId", how="left")
        d["cluster"] = d["cluster"].fillna(-1).astype(int).astype(str)
        out["cluster"] = seg("cluster", d); out["cluster"].to_csv(res / "errors_by_cluster.csv")
        _, axc = plt.subplots(figsize=(8, 3.6))
        out["cluster"]["rmse"].plot.bar(ax=axc, color="#4c78a8"); axc.set_title("RMSE por cluster de películas")
        plt.tight_layout(); plt.savefig(fig / "error_by_cluster.png", dpi=130); plt.close()
    _, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    out["rating"]["bias"].plot.bar(ax=ax[0], color="#0f9f8f"); ax[0].set_title(f"Sesgo (pred - real) por rating real · {name}")
    out["movie_popularity"]["rmse"].plot.bar(ax=ax[1], color="#f5a00f"); ax[1].set_title("RMSE por cuartil de popularidad")
    plt.tight_layout(); plt.savefig(fig / "error_segments.png", dpi=130); plt.close()
    for k, v in out.items(): print(f"\n== {k} ==\n{v.to_string()}")

if __name__ == "__main__":
    main()
