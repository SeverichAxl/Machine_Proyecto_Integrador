"""AutoML (FLAML) sobre regresión de rating con las MISMAS features y la MISMA partición temporal."""
import json, time
import joblib, pandas as pd
from flaml import AutoML
from .common import load_config, load_train_val, path, set_seed
from .models import TabularModel

def main():
    cfg = load_config(); set_seed(cfg["seed"]); a = cfg["automl"]
    train, val = load_train_val(cfg); proc, models, res = path(cfg, "processed"), path(cfg, "models"), path(cfg, "results")
    Xtr, Xva = pd.read_parquet(proc / "feat_train.parquet"), pd.read_parquet(proc / "feat_val.parquet")
    fb = joblib.load(models / "feature_builder.joblib")
    automl = AutoML(); t = time.time()
    automl.fit(X_train=Xtr, y_train=train.rating, X_val=Xva, y_val=val.rating, task="regression", metric="rmse",
               time_budget=a["time_budget"], estimator_list=a["estimator_list"], seed=cfg["seed"],
               eval_method="holdout", log_file_name=str(res / "automl.log"), verbose=1)
    info = {"tool": "FLAML", "time_budget_s": a["time_budget"], "elapsed_s": round(time.time() - t, 1), "seed": cfg["seed"],
            "metric": "rmse", "estimators": a["estimator_list"], "best_estimator": automl.best_estimator,
            "best_config": automl.best_config, "val_rmse": float(automl.best_loss),
            "n_features": Xtr.shape[1], "protocol": "misma partición temporal; validación = split val; test no usado"}
    (res / "automl_summary.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    joblib.dump(TabularModel(fb, automl, list(Xtr.columns)), models / "automl_best.joblib")
    print(json.dumps(info, indent=2, default=str))

if __name__ == "__main__":
    main()
