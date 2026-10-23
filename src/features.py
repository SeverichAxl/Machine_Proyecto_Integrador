"""Features tabulares SIN leakage. Todo lo que se aprende del target (sesgos, medias, afinidad por género)
se calcula solo con datos de ajuste: en train mediante out-of-fold; en val/test con todo train."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .common import path


def make_static(movies):
    m = movies.set_index("movieId")
    g = m.genres.str.get_dummies("|")
    g = g.drop(columns=[c for c in g.columns if c.startswith("(no genres")], errors="ignore").astype("float32")
    year = m.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    return {"G": g, "year": year.fillna(year.median())}

class FeatureBuilder:
    def __init__(self, static, damping=10, genre_k=5, count_scale=1.0):
        self.static, self.d, self.k, self.cs = static, damping, genre_k, count_scale

    def _G(self, ids):
        return self.static["G"].reindex(ids).fillna(0).to_numpy()

    def fit(self, df):
        mu = df.rating.mean(); r = df.rating - mu
        m = r.groupby(df.movieId).agg(["sum", "count"]); bi = m["sum"] / (m["count"] + self.d)
        r2 = r - df.movieId.map(bi)
        u = r2.groupby(df.userId).agg(["sum", "count"]); bu = u["sum"] / (u["count"] + self.d)
        self.mu, self.bi, self.bu, self.n_m, self.n_u = mu, bi, bu, m["count"], u["count"]
        self.u_mean = df.rating.groupby(df.userId).mean()
        G, uid = self._G(df.movieId), df.userId.to_numpy()
        self.usum = pd.DataFrame(G * r.to_numpy()[:, None], index=uid).groupby(level=0).sum()
        self.ucnt = pd.DataFrame(G, index=uid).groupby(level=0).sum()
        return self

    def transform(self, df):
        G, uid, mid = self._G(df.movieId), df.userId, df.movieId
        o = pd.DataFrame(index=df.index)
        o["bu"] = uid.map(self.bu).fillna(0).to_numpy()
        o["bi"] = mid.map(self.bi).fillna(0).to_numpy()
        o["user_mean"] = uid.map(self.u_mean).fillna(self.mu).to_numpy()
        o["log_n_user"] = np.log1p(uid.map(self.n_u).fillna(0).to_numpy() * self.cs)
        o["log_n_movie"] = np.log1p(mid.map(self.n_m).fillna(0).to_numpy() * self.cs)
        year = mid.map(self.static["year"]).fillna(self.static["year"].median()).to_numpy()
        o["year"] = year
        o["age_at_rating"] = pd.to_datetime(df.timestamp, unit="s").dt.year.to_numpy() - year
        us = self.usum.reindex(uid).fillna(0).to_numpy(); uc = self.ucnt.reindex(uid).fillna(0).to_numpy()
        o["genre_affinity"] = (us * G).sum(1) / ((uc * G).sum(1) + self.k)
        o["n_genres"] = G.sum(1)
        for j, c in enumerate(self.static["G"].columns):
            o["g_" + c] = G[:, j]
        return o.astype("float32")

def oof_features(df, static, damping, genre_k, n_folds, seed):
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    cs = n_folds / (n_folds - 1)
    parts = [FeatureBuilder(static, damping, genre_k, cs).fit(df.iloc[a]).transform(df.iloc[b]) for a, b in kf.split(df)]
    return pd.concat(parts).loc[df.index]

def build_feature_sets(cfg, train, val, movies, cache=True):
    """Construye features de train/val. No toca test: final_eval transforma test con el
    FeatureBuilder ajustado en train, sin necesidad de materializar feat_test."""
    out, models = path(cfg, "processed"), path(cfg, "models")
    names = ["feat_train", "feat_val"]
    if cache and all((out / f"{n}.parquet").exists() for n in names) and (models / "feature_builder.joblib").exists():
        sets = [pd.read_parquet(out / f"{n}.parquet") for n in names]
        if [len(s) for s in sets] == [len(train), len(val), len(test)]:
            return (joblib.load(models / "feature_builder.joblib"), *sets)
        print("Caché de features desactualizado; se recalcula.")
    f = cfg["features"]; static = make_static(movies)
    Xtr = oof_features(train, static, f["damping"], f["genre_k"], cfg["oof_folds"], cfg["seed"])
    fb = FeatureBuilder(static, f["damping"], f["genre_k"], 1.0).fit(train)
    Xva = fb.transform(val)
    for n, X in zip(names, (Xtr, Xva)):
        X.to_parquet(out / f"{n}.parquet")
    joblib.dump(fb, models / "feature_builder.joblib")
    return fb, Xtr, Xva
