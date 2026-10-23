# Checklist de cierre (P8)

| Requisito de la guía | Estado | Evidencia | Nota |
|---|---|---|---|
| Fuente registrada | Cumplido | `data/raw/SOURCE.md` | MovieLens 25M, URL, fecha 2026-10-22 y SHA256 del zip. |
| Subconjunto justificado | Cumplido | `data/processed/subset_report.json`, informe §2.1 | 15 000 usuarios, 2 291 670 ratings, sparsity 0.98496; `subset.n_users` configurable; filtros documentados. |
| Split correcto | Cumplido | `data/processed/split_report.json`, `tests/test_leakage.py` | Temporal por usuario 70/15/15; val/test sin usuarios fríos. |
| Sin leakage | Cumplido | `tests/test_leakage.py` (5/5), `reports/decisiones.md` | Test solo en `final_eval.py` y `error_analysis.py`; features OOF. |
| Baseline | Cumplido | `reports/results/val_results.csv`, `test_results.csv` | Media global, popularidad, media por película y baseline de sesgos. |
| 3 modelos supervisados | Cumplido | `reports/results/val_results.csv`, informe §4 | Item-KNN, Funk SVD, Ridge + features y HistGB + features. |
| Métricas justificadas | Cumplido | `reports/results/model_selection.md`, informe §4 | RMSE principal; MAE, NDCG@10 y Recall@10 secundarias; misma partición. |
| Errores analizados | Cumplido | `reports/results/errors_by_*.csv`, `error_segments.png`, `error_by_cluster.png` | Rating, actividad, popularidad, género, década, cold movie y cluster. |
| Clustering interpretado | Cumplido | `reports/results/cluster_profiles.csv`, `notebooks/03_clustering.ipynb`, informe §5 | PCA + KMeans k=7, perfiles, estabilidad y cruce con error. |
| AutoML comparado | Cumplido | `reports/results/automl_report.md`, `automl_summary.json`, informe §6 | FLAML 2.7.0, 600 s, mismo protocolo; comparación con mejor manual. |
| README con orden de ejecución | Cumplido | `README.md` | Pipeline de 9 pasos, pruebas y entregables. |
| `requirements.txt` | Cumplido | `requirements.txt` | Versiones fijadas con `==` para la corrida auditada. |
| Reproducción desde cero (misma semilla) | Cumplido con salvedad | `reports/results/reproducibility.md` | Pipeline analítico exacto; AutoML/FLAML (mismo `lgbm` y configuración) varió Δ RMSE test `9.15e-04` por hilos de LightGBM, e inercia de `k=12` con Δ 0.0078 (k=7, perfiles y asignaciones idénticos). |

## Pendientes / opcionales
- **Segunda semilla en AutoML (P6, opcional).** No se ejecutó; el informe declara que no se midió la variación por semilla. Hacerlo costaría ~10 min extra y no es requisito obligatorio.
- **Cold start real.** No hay usuarios fríos y solo 0.0368% de ratings de test son de películas frías; no se puede validar robustez real con este split.
- **Cobertura del genoma.** El clustering usa 10 158 películas del subconjunto (100% de este), pero solo el 22.1% del dataset completo; los perfiles no cubren la cola larga sin genome.
