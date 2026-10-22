"""Clustering de películas: genome de tags -> PCA -> KMeans. k por silueta+codo, estabilidad por semillas, interpretación."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from .common import load_config, load_train_val, load_movies, path, ROOT, set_seed

def main():
    cfg = load_config(); set_seed(cfg["seed"]); cc = cfg["clustering"]; raw = ROOT / cfg["paths"]["raw"]
    res, fig = path(cfg, "results"), path(cfg, "figures"); train, _ = load_train_val(cfg); movies = load_movies(cfg)
    tags = pd.read_csv(raw / "genome-tags.csv"); gs = pd.read_csv(raw / "genome-scores.csv", dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"})
    gs = gs[gs.movieId.isin(movies.movieId)]
    mids = np.sort(gs.movieId.unique()); tix = pd.Index(tags.tagId.to_numpy())
    M = np.zeros((len(mids), len(tags)), np.float32); M[pd.Index(mids).get_indexer(gs.movieId), tix.get_indexer(gs.tagId)] = gs.relevance.to_numpy()
    print(f"Películas con genome en el subconjunto: {len(mids)} de {len(movies)}")
    pca = PCA(cc["pca_components"], random_state=cfg["seed"]).fit(M); Z = pca.transform(M)
    var = float(pca.explained_variance_ratio_.sum()); print("Varianza explicada PCA:", round(var, 3))
    sel = []
    for k in range(cc["k_min"], cc["k_max"] + 1):
        km = KMeans(k, n_init=10, random_state=cfg["seed"]).fit(Z)
        sel.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_, sample_size=min(cc["silhouette_sample"], len(Z)), random_state=cfg["seed"])})
    sel = pd.DataFrame(sel); sel.to_csv(res / "cluster_k_selection.csv", index=False)
    k = cc["k_final"] or int(sel.loc[sel.silhouette.idxmax(), "k"])
    labs = [KMeans(k, n_init=10, random_state=s).fit(Z).labels_ for s in range(cc["n_seeds"])]
    ari = [adjusted_rand_score(labs[0], l) for l in labs[1:]]; labels = labs[0]
    print(f"k={k}; ARI medio entre semillas = {np.mean(ari):.3f}")
    _, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(sel.k, sel.inertia, "o-"); ax[0].set_title("Codo (inercia)"); ax[1].plot(sel.k, sel.silhouette, "o-", color="#f5a00f"); ax[1].set_title("Silueta")
    ax[2].scatter(Z[:, 0], Z[:, 1], c=labels, s=4, cmap="tab10"); ax[2].set_title(f"PCA 2D coloreado por cluster (k={k})")
    plt.tight_layout(); plt.savefig(fig / "clustering_selection.png", dpi=130); plt.close()
    m = movies.set_index("movieId").loc[mids]; G = m.genres.str.get_dummies("|"); stats = train.groupby("movieId").rating.agg(["mean", "count"]).reindex(mids)
    year = m.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float); tag_names = tags.set_index("tagId").tag
    rows = []
    for c in range(k):
        idx = labels == c; lift = M[idx].mean(0) - M.mean(0); top_tags = tag_names.iloc[np.argsort(-lift)[:8]].tolist()
        gl = (G[idx].mean() / G.mean()).sort_values(ascending=False); top_g = [g for g in gl.index[:4] if G[idx][g].mean() > 0.1]
        rows.append({"cluster": c, "n_movies": int(idx.sum()), "top_tags": ", ".join(top_tags), "top_genres": ", ".join(top_g),
                     "mean_rating": float(stats["mean"][idx].mean()), "median_year": float(year[idx].median()), "median_n_ratings": float(stats["count"][idx].median())})
    pd.DataFrame(rows).to_csv(res / "cluster_profiles.csv", index=False)
    pd.DataFrame({"movieId": mids, "title": m.title.to_numpy(), "cluster": labels}).to_csv(res / "cluster_assignments.csv", index=False)
    (res / "cluster_summary.txt").write_text(f"k={k}\npca_var={var:.3f}\nari_mean={np.mean(ari):.3f}\nari_all={ari}\n")
    print(pd.DataFrame(rows).to_string())

if __name__ == "__main__":
    main()
