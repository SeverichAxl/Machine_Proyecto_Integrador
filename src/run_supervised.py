"""Baselines + 3 familias clásicas (Item-KNN, Funk SVD, Ridge/HGB sobre features) con tuning SOLO en validación.
Guarda el mejor modelo de cada familia en models/ y la tabla de validación. El test NO se usa aquí."""
import itertools, json, time
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import load_config, load_train_val, load_movies, path, set_seed, rmse, mae
from .features import build_feature_sets
from .models import GlobalMean, MovieMean, BiasBaseline, Popularity, ItemKNN, FunkSVD, TabularModel
from .ranking import ranking_eval

def main():
    cfg = load_config(); set_seed(cfg["seed"]); mc = cfg["models"]
    train, val = load_train_val(cfg); movies = load_movies(cfg)
    models_dir, res_dir = path(cfg, "models"), path(cfg, "results")
    rows, best = [], {}

    def evaluate(family, params, model, fit_s):
        pr = model.predict(val) if family != "popularity" else None
        row = {"family": family, "params": json.dumps(params), "fit_seconds": round(fit_s, 1),
               "val_rmse": rmse(val.rating, pr) if pr is not None else np.nan,
               "val_mae": mae(val.rating, pr) if pr is not None else np.nan}
        row.update(ranking_eval(model, train, val, train, cfg, cfg["seed"]))
        rows.append(row); print(row)
        key = row["val_rmse"] if pr is not None else -row[f"ndcg@{cfg['ranking']['k']}"]
        if family not in best or key < best[family]["key"]:
            best[family] = {"key": key, "model": model, "params": params, "val_rmse": row["val_rmse"]}

    def run(family, params, make, fit_args=()):
        t = time.time(); m = make().fit(*fit_args); evaluate(family, params, m, time.time() - t)

    run("global_mean", {}, GlobalMean, (train,))
    run("popularity", {}, Popularity, (train,))
    for d in mc["damping"]:
        run("movie_mean", {"damping": d}, lambda: MovieMean(d), (train,))
        run("bias_baseline", {"damping": d}, lambda: BiasBaseline(d), (train,))
    for k, s in itertools.product(mc["knn"]["k"], mc["knn"]["shrink"]):
        run("item_knn", {"k": k, "shrink": s}, lambda: ItemKNN(k, s), (train,))
    sv = mc["svd"]
    for nf, lr, reg in itertools.product(sv["n_factors"], sv["lr"], sv["reg"]):
        run("funk_svd", {"n_factors": nf, "lr": lr, "reg": reg, "epochs": sv["epochs"]},
            lambda: FunkSVD(nf, lr, reg, sv["epochs"], cfg["seed"]), (train, val))

    fb, Xtr, Xva = build_feature_sets(cfg, train, val, movies)
    cols = list(Xtr.columns); y = train.rating.to_numpy()
    for a in mc["ridge"]["alpha"]:
        t = time.time(); est = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, y)
        evaluate("ridge_features", {"alpha": a}, TabularModel(fb, est, cols), time.time() - t)
    h = mc["hgb"]
    for md, lr in itertools.product(h["max_depth"], h["learning_rate"]):
        t = time.time(); est = HistGradientBoostingRegressor(max_depth=md, learning_rate=lr, max_iter=h["max_iter"], random_state=cfg["seed"]).fit(Xtr, y)
        evaluate("hgb_features", {"max_depth": md, "learning_rate": lr, "max_iter": h["max_iter"]}, TabularModel(fb, est, cols), time.time() - t)

    summary = {}
    for fam, b in best.items():
        joblib.dump(b["model"], models_dir / f"{fam}_best.joblib")
        summary[fam] = {"params": b["params"], "val_rmse": b["val_rmse"]}
    classic = {f: v for f, v in summary.items() if f in ("item_knn", "funk_svd", "ridge_features", "hgb_features")}
    summary["best_manual"] = min(classic, key=lambda f: classic[f]["val_rmse"])
    (res_dir / "best_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(res_dir / "val_results.csv", index=False)
    print("Mejor modelo manual (por RMSE en validación):", summary["best_manual"])

if __name__ == "__main__":
    main()

