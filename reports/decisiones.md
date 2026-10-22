# Decisiones de diseño y auditoría

## 2026-10-22 · Endurecimiento anti-leakage (P2)

**Hallazgo.** La auditoría de `src/` encontró que `eda.py`, `run_supervised.py`, `run_automl.py` y `run_clustering.py` llamaban a `load_splits(cfg)`, que lee `test.parquet`, aunque no usaran test para elegir modelos, hiperparámetros ni k. La decisión metodológica era correcta, pero la lectura técnica del archivo no era comprobable y violaba el criterio “solo `final_eval.py` y `error_analysis.py` leen test”.

**Cambio.**
- Se añadió `load_train_val(cfg)` en `src/common.py`.
- `eda.py`, `run_supervised.py`, `run_automl.py` y `run_clustering.py` ahora leen solo train/val.
- `build_feature_sets` dejó de materializar `feat_test`; `final_eval` transforma test con el `FeatureBuilder` ajustado en train.
- `final_eval.py` y `error_analysis.py` conservan `load_splits` porque son los únicos módulos autorizados a leer test.

**Verificación.** `tests/test_leakage.py` (5 pruebas) pasa: orden temporal por usuario, usuarios de val/test contenidos en train, features OOF, val/test solo con estadísticos de train y lectura de test restringida por AST.

**Efecto en resultados.** Ninguno: no se reajustó ningún modelo ni se cambió ninguna métrica; solo se eliminó la lectura innecesaria de test en módulos que no debían hacerlo.

## 2026-10-22 · Cierre y reproducibilidad (P8)

**Cambio.** `requirements.txt` pasó de rangos (`>=`) a versiones exactas (`==`) para fijar el entorno auditado. Se agregó `reports/results/reproducibility.md` con la corrida limpia.

**Verificación.** En una copia limpia con venv nuevo y `data/processed`, `models`, `reports/results` y `reports/figures` vacíos, el pipeline completo reprodujo de forma exacta los datos, split, EDA, modelos manuales, clustering final (`k=7`) y errores por segmento. Única desviación: AutoML/FLAML varió `2.17e-04` en RMSE test porque LightGBM no es determinista con múltiples hilos; la configuración elegida fue la misma y no cambió ninguna conclusión. La fila `k=6` de `cluster_k_selection.csv` también varió en la silueta (`3.45e-05`), sin afectar el `k` final.
