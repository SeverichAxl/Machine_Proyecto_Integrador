# Auditoría de reproducibilidad (P8)

**Fecha:** 2026-10-22  
**Objetivo:** correr el pipeline completo desde cero en un entorno limpio y comprobar que reproduce las cifras con la misma semilla (`seed=42`).

## Entorno de la corrida limpia
- Copia del proyecto en `D:\opencode-temp\ml-movielens-clean` (fuera del workspace, para no tocar los artefactos originales).
- `data/processed`, `models`, `reports/results` y `reports/figures` eliminados antes de empezar.
- `data/raw/ml-25m` copiado desde el proyecto original; `src.download` confirmó que ya estaba descargado (no se repitió la descarga; el hash del zip está en `data/raw/SOURCE.md`).
- `venv` nuevo con Python 3.14.0 e instalación desde `requirements.txt` con versiones fijadas (`==`).
- Mismos `config/config.yaml`, mismas rutas lógicas, misma partición temporal y misma semilla.

## Comandos ejecutados (en orden)
```text
python -m pytest tests/test_leakage.py
python -m src.download
python -m src.make_subset
python -m src.split
python -m src.eda
python -m src.run_supervised
python -m src.run_clustering
python -m src.run_automl
python -m src.final_eval
python -m src.error_analysis
```

## Resultado: reproducibilidad exacta
Comparación byte a byte / numérica contra los archivos del proyecto original:

| Artefacto | Resultado |
|---|---|
| `subset_report.json` | idéntico |
| `split_report.json` | idéntico |
| `eda_summary.json` | idéntico |
| `best_config.json` | idéntico |
| `val_results.csv` (métricas, sin `fit_seconds`) | diferencia máxima `0.0` |
| `test_results.csv` (baseline, Item-KNN, Funk SVD, Ridge, HistGB) | diferencia máxima `0.0` |
| `errors_by_rating.csv` | diferencia máxima `0.0` |
| `errors_by_user_activity.csv` | diferencia máxima `0.0` |
| `errors_by_movie_popularity.csv` | diferencia máxima `0.0` |
| `errors_by_decade.csv` | diferencia máxima `0.0` |
| `errors_by_genre.csv` | diferencia máxima `0.0` |
| `errors_cold_movie.csv` | diferencia máxima `0.0` |
| `errors_by_cluster.csv` | diferencia máxima `0.0` |
| `cluster_profiles.csv` | diferencia máxima `0.0` |
| `cluster_assignments.csv` | diferencia máxima `0.0` |
| `cluster_summary.txt` | idéntico (`k=7`, `pca_var=0.662`, `ari_mean=0.838`) |
| `tests/test_leakage.py` | 5 pruebas pasan |

## Desviaciones observadas (no afectan conclusiones)
1. **AutoML/FLAML.** El mejor estimador y la mejor configuración son idénticos, pero la métrica varía porque LightGBM no es determinista con múltiples hilos:
   - `val_rmse`: `0.8478464556` → `0.8480474479` (Δ `2.01e-04`).
   - `test_rmse`: `0.8643315623` → `0.8645480234` (Δ `2.17e-04`).
   - `test_mae`: `0.6557587364` → `0.6559345070` (Δ `1.76e-04`).
   - `ndcg@10`: `0.2840637580` → `0.2857238180` (Δ `1.66e-03`).
   - `recall@10`: `0.3170789683` → `0.3209351190` (Δ `3.86e-03`).
2. **Selección de k (fila `k=6`).** `cluster_k_selection.csv` coincide en k=3..5 y k=7..12; en k=6 la inercia difiere en `0.1094` y la silueta en `3.45e-05`. El k elegido (`k=7`) y todos los artefactos derivados son idénticos.

**Interpretación.** Todo el pipeline analítico (datos, split, EDA, baselines, modelos manuales, clustering final y análisis de errores) es determinista y reproducible con la semilla fija. La única variación está en el entrenamiento de LightGBM dentro de FLAML, donde el resultado difiere en el cuarto decimal de RMSE; no cambia el orden de los modelos, la selección de `k=7` ni ninguna conclusión del informe.

**Nota.** El proyecto original conserva los artefactos de la corrida principal; esta auditoría vive en `reports/results/reproducibility.md`.
