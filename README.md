<<<<<<< HEAD
# Recomendación de películas · MovieLens 25M (SIS-351, EC2 Parte I)

Pipeline reproducible CRISP-DM: formulación, EDA, baselines, 3+ modelos clásicos, clustering de películas y AutoML.

**Formulación:** predecir el rating (regresión, RMSE/MAE) + evaluación complementaria top-K (NDCG@10, Recall@10) sobre la misma partición temporal por usuario.

## Entorno
```bash
python -m venv .venv && .venv\Scripts\activate    # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
Colab: abrir `notebooks/00_colab_runner.ipynb` (clona el repo, instala y ejecuta el pipeline).

## Orden de ejecución (desde la raíz del repo)
| # | Comando | Qué hace | Salida |
|---|---|---|---|
| 1 | `python -m src.download` | Descarga MovieLens 25M y registra fuente/fecha/hash | `data/raw/`, `SOURCE.md` |
| 2 | `python -m src.make_subset` | Filtro de actividad mínima + muestreo de usuarios completos | `data/processed/subset_report.json` |
| 3 | `python -m src.split` | Partición temporal por usuario 70/15/15 | `train/val/test.parquet`, `split_report.json` |
| 4 | `python -m src.eda` | EDA sobre train | `reports/figures/`, `reports/results/eda_summary.json` |
| 5 | `python -m src.run_supervised` | Baselines + Item-KNN + Funk SVD + Ridge/HistGB, tuning en validación | `reports/results/val_results.csv`, `models/` |
| 6 | `python -m src.run_clustering` | PCA del genoma de tags + KMeans + interpretación | `reports/results/cluster_*.csv` |
| 7 | `python -m src.run_automl` | FLAML con las mismas features y partición | `reports/results/automl_summary.json` |
| 8 | `python -m src.final_eval` | Evaluación en test (única vez) | `reports/results/test_results.md` |
| 9 | `python -m src.error_analysis` | Errores por rating, actividad, popularidad, género, año | `reports/results/errors_by_*.csv` |

Todo se controla desde `config/config.yaml` (semilla, filtros, grids, presupuesto AutoML).

## Pruebas y entregables
```bash
python -m pytest tests/test_leakage.py      # 5 pruebas anti-leakage
```
- `notebooks/01_eda.ipynb`: EDA interpretado (solo train).
- `notebooks/02_errores.ipynb`: evaluación final y errores por segmento.
- `notebooks/03_clustering.ipynb`: PCA + KMeans, perfiles y cruce con error.
- `reports/results/model_selection.md`: tuning y justificación del modelo.
- `reports/results/automl_report.md`: benchmark FLAML.
- `reports/results/reproducibility.md`: corrida limpia y desviaciones observadas.
- `reports/checklist.md`: checklist de cierre P8 y pendientes.
- `reports/informe.docx`: informe de 8–12 páginas.

## Control de leakage
Ver `AGENTS.md` y `src/features.py`: los sesgos/medias se calculan out-of-fold en train y solo con train en val/test; el split es temporal; el test se lee solo en `final_eval.py` y `error_analysis.py`.

## Fuente de datos
GroupLens, MovieLens 25M. Verificar cifras y versión en el README oficial del dataset; la fecha y hash de descarga quedan en `data/raw/SOURCE.md`.

## Apoyos declarados
Código generado con asistencia de IA (Claude) y revisado por el equipo. Librerías: pandas, scikit-learn, numba, FLAML, matplotlib.
=======
# Machine_Proyecto_Integrador
>>>>>>> 163ec17f3218528d6bd1a38fdb94708d9115cdda
