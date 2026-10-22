"""Auditoría anti-leakage sobre un dataset sintético pequeño.

Cada prueba ataca una fuga distinta. Si una falla, el mensaje indica qué invariante
se rompió; no se corrige nada de forma silenciosa.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common import ROOT
from src.features import FeatureBuilder, make_static, oof_features
from src.split import main as split_main


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "seed": 42,
        "paths": {
            "raw": str(tmp_path / "raw"),
            "processed": str(tmp_path / "processed"),
            "models": str(tmp_path / "models"),
            "results": str(tmp_path / "results"),
            "figures": str(tmp_path / "figures"),
        },
        "split": {"train": 0.70, "val": 0.15},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for key in cfg["paths"].values():
        Path(key).mkdir(parents=True, exist_ok=True)
    return path


def _synthetic_ratings(n_users: int = 6, n_each: int = 10) -> pd.DataFrame:
    rows = []
    for u in range(1, n_users + 1):
        for j in range(n_each):
            rows.append(
                {
                    "userId": u,
                    "movieId": (j % 5) + 1,
                    "rating": float(0.5 + (j % 10) * 0.5),
                    "timestamp": 1_600_000_000 + u * 10_000 + j * 10,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def split_data(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setenv("ML_CONFIG", str(cfg_path))
    processed = tmp_path / "processed"
    _synthetic_ratings().to_parquet(processed / "ratings_subset.parquet", index=False)
    split_main()
    out = {name: pd.read_parquet(processed / f"{name}.parquet") for name in ("train", "val", "test")}
    report = json.loads((processed / "split_report.json").read_text(encoding="utf-8"))
    return out, report


def test_split_temporal_por_usuario(split_data):
    """(1) Para cada usuario: max(train) <= min(val) <= max(val) <= min(test)."""
    splits, _ = split_data
    tr, va, te = splits["train"], splits["val"], splits["test"]
    max_tr = tr.groupby("userId").timestamp.max()
    min_va = va.groupby("userId").timestamp.min()
    max_va = va.groupby("userId").timestamp.max()
    min_te = te.groupby("userId").timestamp.min()
    for u in tr.userId.unique():
        assert u in min_va.index, f"usuario {u} sin filas en val"
        assert u in min_te.index, f"usuario {u} sin filas en test"
        assert max_tr[u] <= min_va[u], f"fuga temporal train->val en usuario {u}"
        assert min_va[u] <= max_va[u], f"val vacío/inconsistente en usuario {u}"
        assert max_va[u] <= min_te[u], f"fuga temporal val->test en usuario {u}"


def test_val_test_usuarios_contenidos_en_train(split_data):
    """(2) Todos los usuarios de val/test aparecen en train (no hay usuarios fríos)."""
    splits, report = split_data
    train_users = set(splits["train"].userId.unique())
    for name in ("val", "test"):
        users = set(splits[name].userId.unique())
        assert users <= train_users, f"{name} contiene usuarios fuera de train: {sorted(users - train_users)}"
        assert report[name]["pct_ratings_cold_user"] == 0.0


def _features_fixture():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(120):
        rows.append(
            {
                "userId": int(rng.integers(1, 9)),
                "movieId": int(rng.integers(1, 6)),
                "rating": float(rng.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
                "timestamp": 1_600_000_000 + i * 100,
            }
        )
    df = pd.DataFrame(rows)
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["A (1990)", "B (1995)", "C (2000)", "D (2005)", "E (2010)"],
            "genres": ["Drama", "Comedy|Drama", "Action", "Horror|Thriller", "Comedy"],
        }
    )
    return df, movies


def test_train_features_son_out_of_fold():
    """(3) La fila i no aporta su propio rating a sus medias/sesos en train."""
    df, movies = _features_fixture()
    static = make_static(movies)
    damping, genre_k, n_folds, seed = 10, 5, 5, 42
    X = oof_features(df, static, damping, genre_k, n_folds, seed)

    from sklearn.model_selection import KFold

    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(df))
    cs = n_folds / (n_folds - 1)
    for i in [0, 7, 23, 51, 99, 119]:
        fit_idx, val_idx = next((a, b) for a, b in folds if i in b)
        expected = FeatureBuilder(static, damping, genre_k, cs).fit(df.iloc[fit_idx]).transform(df.iloc[[i]])
        for col in ("bi", "bu", "user_mean", "genre_affinity"):
            assert np.isclose(X.iloc[i][col], expected.iloc[0][col], atol=1e-5), (
                f"fila {i}: {col} no es out-of-fold ({X.iloc[i][col]} vs {expected.iloc[0][col]})"
            )

        changed = df.copy()
        changed.loc[i, "rating"] = 5.0 if changed.loc[i, "rating"] != 5.0 else 1.0
        X_changed = oof_features(changed, static, damping, genre_k, n_folds, seed)
        for col in ("bi", "bu", "user_mean", "genre_affinity"):
            assert np.isclose(X.iloc[i][col], X_changed.iloc[i][col], atol=1e-6), (
                f"fila {i}: {col} cambió al modificar su propio rating (hay fuga)"
            )


def test_val_test_solo_estadisticos_de_train():
    """(4) Cambiar ratings de val/test no altera sus features: solo se usan estadísticos de train."""
    df, movies = _features_fixture()
    static = make_static(movies)
    train, val, test = df.iloc[:80].copy(), df.iloc[80:100].copy(), df.iloc[100:].copy()
    fb = FeatureBuilder(static, 10, 5, 1.0).fit(train)

    val_orig, test_orig = fb.transform(val), fb.transform(test)
    val_changed = val.copy()
    val_changed["rating"] = 1.0
    test_changed = test.copy()
    test_changed["rating"] = 5.0
    val_new, test_new = fb.transform(val_changed), fb.transform(test_changed)

    pd.testing.assert_frame_equal(val_orig, val_new)
    pd.testing.assert_frame_equal(test_orig, test_new)

    known_user = int(train.userId.iloc[0])
    assert np.isclose(val_orig.loc[val.userId == known_user, "user_mean"].iloc[0], train[train.userId == known_user].rating.mean())

    cold = val.copy()
    cold["userId"] = 999_999
    cold_feats = fb.transform(cold)
    assert np.allclose(cold_feats["user_mean"], fb.mu), "usuario frío no cae al mu de train"
    assert np.allclose(cold_feats["bu"], 0.0), "usuario frío no cae a sesgo 0"


def test_solo_final_eval_y_error_analysis_usan_test():
    """(5) Ningún módulo de src/ lee test.parquet salvo final_eval.py y error_analysis.py.

    `common.py` contiene el helper `load_splits`; se permite solo como definición.
    Se inspecciona el AST: llamadas a load_splits y literales 'test.parquet'.
    """
    allowed = {"final_eval.py", "error_analysis.py", "common.py"}
    violations = []
    for path in sorted((ROOT / "src").glob("*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_splits":
                violations.append(f"{path.name}:{node.lineno} llama a load_splits (lee test.parquet)")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "test.parquet" in node.value:
                violations.append(f"{path.name}:{node.lineno} menciona test.parquet")
    assert not violations, "Lecturas de test fuera de los módulos permitidos:\n" + "\n".join(violations)
