import numpy as np, pandas as pd
from numba import njit
from scipy import sparse

def _clip(x): return np.clip(x, 0.5, 5.0)

class GlobalMean:
    def fit(self, tr): self.mu = float(tr.rating.mean()); return self
    def predict(self, df): return np.full(len(df), self.mu)

class MovieMean:
    def __init__(self, damping=10): self.d = damping
    def fit(self, tr):
        self.mu = float(tr.rating.mean()); g = tr.groupby("movieId").rating.agg(["sum", "count"])
        self.m = (g["sum"] + self.d * self.mu) / (g["count"] + self.d); return self
    def predict(self, df): return df.movieId.map(self.m).fillna(self.mu).to_numpy()

class BiasBaseline:
    """mu + b_u + b_i con amortiguación (damping), ajustado alternando."""
    def __init__(self, damping=10, n_iter=10): self.d, self.n_iter = damping, n_iter
    def fit(self, tr):
        mu = float(tr.rating.mean()); bu = pd.Series(0.0, index=tr.userId.unique()); bi = pd.Series(0.0, index=tr.movieId.unique())
        for _ in range(self.n_iter):
            g = (tr.rating - mu - tr.userId.map(bu)).groupby(tr.movieId); bi = g.sum() / (g.count() + self.d)
            g = (tr.rating - mu - tr.movieId.map(bi)).groupby(tr.userId); bu = g.sum() / (g.count() + self.d)
        self.mu, self.bu, self.bi = mu, bu, bi; return self
    def predict(self, df):
        return _clip(self.mu + df.userId.map(self.bu).fillna(0).to_numpy() + df.movieId.map(self.bi).fillna(0).to_numpy())

class Popularity:
    """Solo ranking: puntaje = log(1 + #ratings en train)."""
    def fit(self, tr): self.c = tr.movieId.value_counts(); return self
    def predict(self, df): return np.log1p(df.movieId.map(self.c).fillna(0).to_numpy())

class ItemKNN:
    """Item-KNN sobre residuos del baseline de sesgos; similitud coseno con contracción por co-ocurrencias."""
    def __init__(self, k=50, shrink=20, damping=10): self.k, self.shrink, self.damping = k, shrink, damping
    def fit(self, tr):
        self.base = BiasBaseline(self.damping).fit(tr)
        res = (tr.rating.to_numpy() - self.base.predict(tr)).astype(np.float32)
        self.users = pd.Index(np.sort(tr.userId.unique())); self.items = pd.Index(np.sort(tr.movieId.unique()))
        u, i = self.users.get_indexer(tr.userId), self.items.get_indexer(tr.movieId)
        nu, ni = len(self.users), len(self.items); self.ni = ni
        R = sparse.csr_matrix((res, (u, i)), shape=(nu, ni)); B = sparse.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(nu, ni))
        norms = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-9
        Rc, Bc = R.tocsc(), B.tocsc(); k = min(self.k, ni - 1)
        self.nbr_i = np.zeros((ni, k), np.int32); self.nbr_s = np.zeros((ni, k), np.float32)
        for s in range(0, ni, 500):
            e = min(s + 500, ni)
            dot = (Rc[:, s:e].T @ R).toarray(); co = (Bc[:, s:e].T @ B).toarray()
            sim = dot / (norms[s:e, None] * norms[None, :]) * (co / (co + self.shrink))
            sim[np.arange(e - s), np.arange(s, e)] = -1
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            self.nbr_i[s:e] = idx; self.nbr_s[s:e] = np.take_along_axis(sim, idx, 1)
        self.nbr_s = np.maximum(self.nbr_s, 0)
        key = u.astype(np.int64) * ni + i; o = np.argsort(key); self.keys, self.vals = key[o], res[o]
        return self
    def predict(self, df):
        out = self.base.predict(df).copy()
        u, i = self.users.get_indexer(df.userId), self.items.get_indexer(df.movieId)
        idx = np.where((u >= 0) & (i >= 0))[0]
        for s in range(0, len(idx), 100_000):
            ch = idx[s:s + 100_000]
            nb, sm = self.nbr_i[i[ch]], self.nbr_s[i[ch]]
            q = u[ch][:, None].astype(np.int64) * self.ni + nb
            pos = np.clip(np.searchsorted(self.keys, q), 0, len(self.keys) - 1)
            hit = self.keys[pos] == q
            w = sm * hit
            out[ch] += (w * np.where(hit, self.vals[pos], 0.0)).sum(1) / (w.sum(1) + 0.1)
        return _clip(out)

@njit(cache=True)
def _epoch(u, i, r, perm, P, Q, bu, bi, mu, lr, reg):
    for t in perm:
        a, b = u[t], i[t]
        dot = 0.0
        for f in range(P.shape[1]):
            dot += P[a, f] * Q[b, f]
        e = r[t] - (mu + bu[a] + bi[b] + dot)
        bu[a] += lr * (e - reg * bu[a]); bi[b] += lr * (e - reg * bi[b])
        for f in range(P.shape[1]):
            pf, qf = P[a, f], Q[b, f]
            P[a, f] += lr * (e * qf - reg * pf); Q[b, f] += lr * (e * pf - reg * qf)

class FunkSVD:
    """Factorización matricial con sesgos entrenada por SGD (Funk SVD)."""
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, epochs=20, seed=42):
        self.nf, self.lr, self.reg, self.epochs, self.seed = n_factors, lr, reg, epochs, seed
    def fit(self, tr, val=None):
        rng = np.random.default_rng(self.seed)
        self.users = pd.Index(np.sort(tr.userId.unique())); self.items = pd.Index(np.sort(tr.movieId.unique()))
        u = self.users.get_indexer(tr.userId).astype(np.int64); i = self.items.get_indexer(tr.movieId).astype(np.int64)
        r = tr.rating.to_numpy().astype(np.float64); self.mu = float(r.mean())
        self.P = rng.normal(0, 0.1, (len(self.users), self.nf)); self.Q = rng.normal(0, 0.1, (len(self.items), self.nf))
        self.bu = np.zeros(len(self.users)); self.bi = np.zeros(len(self.items)); self.history = []
        for ep in range(self.epochs):
            _epoch(u, i, r, rng.permutation(len(r)), self.P, self.Q, self.bu, self.bi, self.mu, self.lr, self.reg)
            if val is not None:
                self.history.append(float(np.sqrt(np.mean((val.rating.to_numpy() - self.predict(val)) ** 2))))
        return self
    def predict(self, df):
        u, i = self.users.get_indexer(df.userId), self.items.get_indexer(df.movieId)
        p = np.full(len(df), self.mu)
        ku, ki = u >= 0, i >= 0
        p[ku] += self.bu[u[ku]]; p[ki] += self.bi[i[ki]]
        both = ku & ki
        p[both] += (self.P[u[both]] * self.Q[i[both]]).sum(1)
        return _clip(p)

class TabularModel:
    """Featurizador (ajustado solo con train) + estimador sklearn/FLAML."""
    def __init__(self, featurizer, estimator, cols): self.fe, self.est, self.cols = featurizer, estimator, cols
    def predict(self, df): return _clip(np.asarray(self.est.predict(self.fe.transform(df)[self.cols])))
