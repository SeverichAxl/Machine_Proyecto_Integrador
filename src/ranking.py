"""Evaluación top-K con candidatos muestreados: relevantes del usuario (rating >= umbral en el split evaluado)
+ n_neg películas que el usuario nunca calificó. Recall@K = hits / min(#relevantes, K); NDCG@K binario."""
import numpy as np, pandas as pd

def ranking_eval(model, train, eval_df, seen_df, cfg, seed=42):
    rc = cfg["ranking"]; K = rc["k"]; rng = np.random.default_rng(seed)
    rel = eval_df[eval_df.rating >= rc["relevant_threshold"]]
    items = train.movieId.unique(); item_set = set(items.tolist())
    relg = rel.groupby("userId").movieId.apply(list)
    users = relg.index.to_numpy()
    if len(users) > rc["max_users"]:
        users = rng.choice(users, rc["max_users"], replace=False)
    seen = seen_df.groupby("userId").movieId.apply(set); allev = eval_df.groupby("userId").movieId.apply(set)
    asof = eval_df.groupby("userId").timestamp.min()
    rows = []
    for u in users:
        pos = [m for m in relg[u] if m in item_set]
        if not pos: continue
        excl = seen.get(u, set()) | allev.get(u, set()); negs = set()
        while len(negs) < rc["n_neg"]:
            negs |= {m for m in rng.choice(items, rc["n_neg"] * 2).tolist() if m not in excl}
        negs = list(negs)[:rc["n_neg"]]
        rows += [(u, m, 1, asof[u]) for m in pos] + [(u, m, 0, asof[u]) for m in negs]
    c = pd.DataFrame(rows, columns=["userId", "movieId", "label", "timestamp"])
    c["score"] = np.asarray(model.predict(c)) + rng.random(len(c)) * 1e-9
    c = c.sort_values(["userId", "score"], ascending=[True, False])
    c["rank"] = c.groupby("userId").cumcount()
    top = c[c["rank"] < K]; npos = c.groupby("userId").label.sum()
    recall = (top.groupby("userId").label.sum().reindex(npos.index).fillna(0) / np.minimum(npos, K)).mean()
    dcg = (top.label / np.log2(top["rank"] + 2)).groupby(top.userId).sum().reindex(npos.index).fillna(0)
    ideal = npos.clip(upper=K).map(lambda n: (1 / np.log2(np.arange(2, n + 2))).sum())
    return {f"ndcg@{K}": float((dcg / ideal).mean()), f"recall@{K}": float(recall), "n_users_rank": int(len(npos))}
