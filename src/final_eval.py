"""ÚNICO punto donde se toca el test: evalúa los modelos ya elegidos en validación (sin reajustar nada)."""
import json
import joblib, numpy as np, pandas as pd
from .common import load_config, load_splits, path, rmse, mae
from .ranking import ranking_eval

LABELS = {"bias_baseline": "Baseline (sesgo usuario+película)", "popularity": "Baseline ranking (popularidad)",
          "item_knn": "Item-KNN", "funk_svd": "Funk SVD", "ridge_features": "Ridge + features",
          "hgb_features": "HistGB + features", "automl_best": "Mejor de AutoML (FLAML)"}

def main():
    cfg = load_config(); train, val, test = load_splits(cfg); models, res = path(cfg, "models"), path(cfg, "results")
    best = json.loads((res / "best_config.json").read_text()); seen = pd.concat([train, val]); K = cfg["ranking"]["k"]
    rows = []
    for key, label in LABELS.items():
        f = models / (key if key == "automl_best" else key + "_best") ; f = f.with_suffix(".joblib")
        if not f.exists(): continue
        m = joblib.load(f); row = {"modelo": label, "params": json.dumps(best.get(key, {}).get("params", {}))}
        if key != "popularity":
            p = m.predict(test); row.update(test_rmse=rmse(test.rating, p), test_mae=mae(test.rating, p))
        row.update(ranking_eval(m, train, test, seen, cfg, cfg["seed"]))
        if key in best and best[key].get("val_rmse") == best[key].get("val_rmse"): row["val_rmse"] = best[key]["val_rmse"]
        rows.append(row); print(row)
    df = pd.DataFrame(rows); df.to_csv(res / "test_results.csv", index=False)
    (res / "test_results.md").write_text(df.drop(columns=["params"]).round(4).to_markdown(index=False), encoding="utf-8")
    print(df.round(4).to_string())

if __name__ == "__main__":
    main()
