# Prompts para el agente (VS Code). Pégalos en orden. Cada uno asume que el agente leyó `AGENTS.md`.

## P0 · Puesta en marcha (local, 16 GB)
Lee AGENTS.md y README.md. Crea el entorno virtual, instala requirements.txt y ejecuta `python -m src.download`, luego make_subset, split, eda y run_supervised. Si hay problemas de memoria, baja `subset.n_users` en config/config.yaml (prueba 5000) y repórtame cuánto cambió. Al terminar muéstrame subset_report.json, split_report.json y val_results.csv. No modifiques la lógica anti-leakage.

## P1 · EDA interpretado (notebooks/01_eda.ipynb)
Con `reports/results/eda_summary.json` y las figuras de `reports/figures/`, crea notebooks/01_eda.ipynb que cargue SOLO train. Elige 3 a 5 hallazgos que cambien decisiones. Candidatos: rating sesgado a valores altos, cola larga de popularidad, actividad desigual por usuario, sparsity, deriva temporal. Cada hallazgo debe tener: gráfico, número concreto y una línea final "Por eso decidimos X" (filtro, métrica, baseline, split, feature). Incluye diagnóstico de calidad (faltantes, duplicados, outliers de actividad) y la tabla de leakage (qué podría filtrarse y cómo lo evitamos). No inventes cifras: léelas de los archivos.

## P2 · Auditoría anti-leakage + pruebas
Revisa src/ y crea tests/test_leakage.py con pytest sobre un dataset sintético pequeño que verifique: (1) para cada usuario, max(timestamp en train) <= min(timestamp en val) <= max(val) <= min(test); (2) todos los usuarios de val/test están en train; (3) las features de train son out-of-fold (la media de película de una fila no incluye su propio rating); (4) val/test solo usan estadísticos de train; (5) ningún módulo salvo final_eval.py y error_analysis.py lee test.parquet. Reporta cualquier violación sin corregirla en silencio.

## P3 · Tuning y justificación del modelo
Ejecuta run_supervised con los grids de config/config.yaml (amplía un poco si el tiempo lo permite, siempre desde config y evaluando solo en validación). Genera reports/results/model_selection.md con una tabla: modelo, RMSE val, MAE val, NDCG@10, Recall@10, tiempo de ajuste, interpretabilidad, manejo de cold start. Redacta una justificación de elección basada en error, costo, interpretabilidad y cold start (no solo la mejor métrica). Si Funk SVD queda mal, revisa lr/reg/epochs y muestra la curva `history` de RMSE por época.

## P4 · Evaluación final y errores
Ejecuta final_eval y error_analysis (una sola vez cada uno). Crea notebooks/02_errores.ipynb que lea reports/results/errors_by_*.csv y las figuras, e interprete: regresión hacia la media por rating real, error según actividad del usuario y popularidad de la película, géneros y décadas difíciles, y el segmento cold start. Cierra con limitaciones y una conclusión respaldada solo por números de los CSV.

## P5 · No supervisado
Ejecuta `python -m src.run_clustering` y crea notebooks/03_clustering.ipynb. Justifica la representación (genoma de tags -> PCA, varianza explicada) y la elección de k (codo + silueta + estabilidad ARI entre semillas). Interpreta cada cluster con cluster_profiles.csv (tags dominantes, géneros, rating medio, año, popularidad) y pon nombres tentativos marcados como hipótesis, no como categorías reales. Explica limitaciones (el genoma cubre solo parte de las películas, clusters poco separados si la silueta es baja). Bonus: cruza los clusters con el error del mejor modelo (RMSE por cluster) y comenta.

## P6 · AutoML como benchmark
Ejecuta `python -m src.run_automl` con time_budget de 600 s (config). Documenta en reports/results/automl_report.md: herramienta y versión, presupuesto, semilla, métrica, estimadores explorados, mejor configuración, tiempo real, y la tabla mejor manual vs AutoML bajo el mismo protocolo (val y test). Discute trade-offs (rendimiento, tiempo, interpretabilidad, control, reproducibilidad) y aclara con honestidad que AutoML tabular no reemplaza a la factorización matricial colaborativa porque no recibe la estructura usuario-película salvo por los features. Si hay tiempo, repite con otra semilla y reporta la variación.

## P7 · Informe (docx de 8 a 12 páginas)
Con los resultados de reports/results/*, la sección reports/01_problema.md y los notebooks, genera reports/informe.docx con: 1 Problema, 2 Datos (fuente, versión, fecha, subconjunto, tabla de variables, EDA con 3-5 hallazgos, calidad, leakage), 3 Preparación y pipeline, 4 Modelado supervisado, 5 No supervisado, 6 AutoML, 7 Evaluación y errores (tabla baseline/A/B/C/AutoML), 8 Conclusiones y continuidad a la Parte II (el genoma y los títulos/tags servirán para embeddings y recuperación semántica; el pipeline será base del MLOps). 8-12 páginas sin portada, referencias ni anexos; tablas largas al anexo. Todas las cifras deben venir de los archivos de resultados; si falta una, déjala marcada [PENDIENTE] en vez de inventarla.

## P8 · Cierre del repositorio
Verifica la lista de la guía: fuente registrada, subconjunto justificado, split correcto, sin leakage, baseline, 3 modelos, métricas justificadas, errores, clustering interpretado, AutoML comparado, README con orden de ejecución, requirements.txt. Corre todo el pipeline desde cero en un entorno limpio y confirma que reproduce las cifras (misma semilla). Reporta lo que falte como lista, sin maquillarlo.
