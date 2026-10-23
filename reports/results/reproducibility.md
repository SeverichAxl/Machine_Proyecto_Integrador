# Auditoría de reproducibilidad (P8)

**Fecha:** 2026-10-23  
**Objetivo:** correr el pipeline completo desde cero en un entorno limpio y comprobar que reproduce las cifras con la misma semilla (`seed=42`) sobre la configuración vigente (`subset.n_users = 15000`).

## Entorno de la corrida limpia
- Copia del proyecto en `D:\opencode-temp\ml-movielens-clean` (fuera del workspace, para no tocar los artefactos originales).
- `data/processed`, `models`, `reports/results` y `reports/figures` eliminados antes de empezar.
- `data/raw/ml-25m` copiado desde el proyecto original; `src.download` confirmó que ya estaba descargado (no se repitió la descarga; el hash del zip está en `data/raw/SOURCE.md`).
- `venv` con Python 3.14.0 e instalación desde `requirements.txt` con versiones fijadas (`==`): pandas 3.1.0, scikit-learn 1.9.1, numba 0.68.0, FLAML 2.7.0, LightGBM 4.7.0, numpy 2.5.4.
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

## Resultado: reproducibilidad exacta en el pipeline analítico
Comparación numérica contra los archivos del proyecto original:

| Artefacto | Resultado |
|---|---|
| `subset_report.json` | idéntico (15 000 usuarios, 2 291 670 ratings, sparsity 0.9849598) |
| `split_report.json` | idéntico |
| `eda_summary.json` | idéntico |
| `best_config.json` | idéntico (`best_manual = item_knn`) |
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
| `tests/test_leakage.py` | 5 pruebas pasan (en workspace y en la copia limpia) |

Los baselines, Item-KNN, Funk SVD, Ridge, HistGB, el clustering final y todos los análisis de error son **bit a bit reproducibles** con la semilla fija.

## Desviaciones observadas (no afectan la elección del modelo)
1. **AutoML/FLAML.** El mejor estimador (`lgbm`) y la mejor configuración son idénticos en ambas corridas, pero la métrica varía porque LightGBM no es determinista con múltiples hilos:
   - `val_rmse`: `0.8477254382` → `0.8485859545` (Δ `8.61e-04`).
   - `test_rmse`: `0.8642194703` → `0.8651343433` (Δ `9.15e-04`).
   - `test_mae`: `0.6556544014` → `0.6565329182` (Δ `8.79e-04`).
   - `ndcg@10`: `0.2833654361` → `0.2902006290` (Δ `6.84e-03`).
   - `recall@10`: `0.3163196429` → `0.3256404762` (Δ `9.32e-03`).
   - En ambas corridas AutoML sigue **por detrás de Item-KNN** y en el rango de HistGB manual, así que no cambia la elección ni las conclusiones.
2. **Inercia de KMeans.** `cluster_k_selection.csv` coincide en silueta en todo el rango y en inercia salvo la fila `k=12`, donde la inercia difiere en `0.007812` por redondeo de punto flotante. El `k` elegido (`k=7`), los perfiles y las asignaciones son idénticos.

## Interpretación
El pipeline analítico (datos, split, EDA, baselines, modelos manuales, clustering final y análisis de errores) es determinista y reproducible con la semilla fija. La única variación está en el entrenamiento de LightGBM dentro de FLAML, donde el RMSE difiere en el cuarto decimal y puede cambiar el ranking fino del mejor trial, sin alterar el orden de los modelos ni ninguna conclusión del informe. La selección del mejor modelo (Item-KNN) y la elección de `k=7` quedan reproducidas.

**Nota.** El proyecto original conserva los artefactos de la corrida principal; esta auditoría vive en `reports/results/reproducibility.md`.
