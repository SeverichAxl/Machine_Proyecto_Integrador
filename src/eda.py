"""EDA sobre TRAIN únicamente (no se mira val/test para tomar decisiones). Genera figuras y eda_summary.json."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from .common import load_config, load_train_val, load_movies, path

def main():
    cfg = load_config(); train, _ = load_train_val(cfg); movies = load_movies(cfg)
    fig, res = path(cfg, "figures"), path(cfg, "results"); S = {}
    S["rating_mean"], S["rating_std"] = float(train.rating.mean()), float(train.rating.std())
    S["rating_share"] = train.rating.value_counts(normalize=True).sort_index().round(4).to_dict()
    S["pct_ratings_ge4"] = float((train.rating >= 4).mean())
    uc, mc = train.userId.value_counts(), train.movieId.value_counts()
    S["ratings_per_user"] = uc.describe().round(1).to_dict(); S["ratings_per_movie"] = mc.describe().round(1).to_dict()
    ms = np.sort(mc.to_numpy())[::-1].cumsum() / mc.sum()
    S["share_ratings_top10pct_movies"] = float(ms[int(0.1 * len(ms))]); S["share_ratings_top1pct_movies"] = float(ms[max(int(0.01 * len(ms)) - 1, 0)])
    S["sparsity_train"] = float(1 - len(train) / (len(uc) * len(mc)))
    yr = pd.to_datetime(train.timestamp, unit="s").dt.year
    S["mean_rating_by_year"] = train.groupby(yr).rating.mean().round(3).to_dict()
    pop = train.movieId.map(mc); S["corr_movie_popularity_vs_mean_rating"] = float(train.groupby("movieId").rating.mean().corr(mc.reindex(train.groupby("movieId").rating.mean().index)))
    g = movies.set_index("movieId").genres.str.split("|").explode(); S["genre_counts"] = g.value_counts().to_dict()
    S["missing_values"] = train.isna().sum().to_dict(); S["duplicates_user_movie"] = int(train.duplicated(["userId", "movieId"]).sum())
    _, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    train.rating.value_counts().sort_index().plot.bar(ax=ax[0], color="#0f9f8f"); ax[0].set_title("Distribución de ratings (train)")
    ax[1].hist(np.log10(uc), bins=40, color="#0f9f8f"); ax[1].set_title("Ratings por usuario (log10)")
    plt.tight_layout(); plt.savefig(fig / "eda_ratings_users.png", dpi=130); plt.close()
    _, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].loglog(np.arange(1, len(mc) + 1), np.sort(mc.to_numpy())[::-1], color="#0f9f8f"); ax[0].set_title("Popularidad de películas (rango vs #ratings)")
    train.groupby(yr).rating.mean().plot(ax=ax[1], marker="o", color="#f5a00f"); ax[1].set_title("Rating medio por año de calificación")
    plt.tight_layout(); plt.savefig(fig / "eda_popularity_drift.png", dpi=130); plt.close()
    (res / "eda_summary.json").write_text(json.dumps(S, indent=2, default=str), encoding="utf-8"); print(json.dumps(S, indent=2, default=str)[:1500])

if __name__ == "__main__":
    main()
