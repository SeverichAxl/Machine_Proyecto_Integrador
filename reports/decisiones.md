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

## 2026-10-23 · Ajuste de configuración, corrección en features y re-auditoría (P3–P8)

**Cambios al proyecto (correcciones del usuario).**
- `src/features.py`: se corrigió un `NameError` introducido en `build_feature_sets` (la lista de tamaños usaba una variable inexistente `test`; ahora usa `[len(train), len(val)]`).
- `config/config.yaml`: `subset.n_users` se probó temporalmente en **2 000** para una verificación rápida y luego se **restauró a 15 000**, que es la configuración final del entregable.

**Configuración final y efecto.** Se re-ejecutó el pipeline completo con `n_users=15000` y se regeneraron todos los artefactos, figuras, notebooks, `model_selection.md`, `automl_report.md`, `informe.docx` y este cierre. Cifras destacadas: subconjunto de 15 000 usuarios / 2 291 670 ratings (sparsity 0.98496); mejor modelo manual Item-KNN (k=50, shrink=50) con RMSE test 0.8450; clustering en **k=7** (silueta 0.107, PCA 0.662, ARI 0.838); AutoML/FLAML eligió `lgbm` (val RMSE 0.8477).

**Nota sobre SVD.** La verificación de épocas con la mejor regularización (`reg=0.02`) muestra que en este subconjunto el mejor punto está en la época 30 con 20 factores (0.8241) y en la época 24 con 50 factores (0.8275); a 24 épocas con 20 factores el grid obtiene 0.8261. El modelo reportado mantiene las 24 épocas fijadas en `config/config.yaml` por consistencia con el grid auditado; el detalle queda en `reports/results/model_selection.md` e informe §4. No cambia el modelo elegido.

**Verificación P8 (copia limpia, `n_users=15000`).** Los baselines, modelos manuales, clustering final (`k=7`) y errores por segmento son reproducibles de forma exacta. Desviaciones: (i) AutoML/FLAML eligió el mismo estimador y configuración (`lgbm`) pero la métrica varió por hilos (Δ RMSE test `9.15e-04`), sin alterar la elección ni las conclusiones; (ii) la inercia de `k=12` varió `0.0078`, con `k=7`, perfiles y asignaciones idénticos. Detalle en `reports/results/reproducibility.md`.
