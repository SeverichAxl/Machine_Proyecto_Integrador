# Reglas del proyecto (SIS-351 · Proyecto Integrador Parte I · Grupo 4 · MovieLens 25M)

Estas reglas son obligatorias: la rúbrica penaliza su incumplimiento. Léelas antes de tocar código.

## Metodología (no negociable)
1. NUNCA usar `test` para elegir modelos, hiperparámetros, features, umbrales ni k de clustering. Solo `final_eval.py` y `error_analysis.py` leen el test, y solo con modelos ya elegidos en validación.
2. Todo lo que se aprende de los datos (medias, sesgos, escaladores, PCA, selección de variables) se ajusta (`fit`) SOLO con train y en val/test solo se aplica (`transform`).
3. Medias de usuario/película y cualquier feature derivada del rating: out-of-fold dentro de train (ver `src/features.py`); en val/test se calculan con todo train. Nunca con el dataset completo.
4. Partición temporal por usuario (70/15/15). Prohibido el split aleatorio.
5. Semilla fija (`seed` en `config/config.yaml`) en todo proceso aleatorio. Parámetros solo en `config/`, nunca hardcodeados.
6. Comparación justa: mismas filas, misma partición, mismas métricas para baselines, modelos y AutoML.
7. Registrar todo filtrado, muestreo o descarte (ver `data/processed/subset_report.json` y `split_report.json`).
8. Clustering/PCA siempre con interpretación (tags, géneros, rating medio, año, popularidad). No tratar clusters como categorías reales sin evidencia.
9. Una mejora de métrica no basta: reportar costo (tiempo), errores por segmento y limitaciones.
10. Declarar en el informe todo código de ejemplo, librería, tutorial o IA usada como apoyo.

## Convenciones
- Código en `src/` (ejecutar desde la raíz con `python -m src.<modulo>`), notebooks solo para narrativa/figuras.
- Métrica principal: RMSE (rating). Secundarias: MAE, NDCG@10 y Recall@10 (ranking con candidatos muestreados).
- No subir `data/raw`, `data/processed` ni `models/*.joblib` al repo.
- Si cambias una decisión de diseño, documéntala en `reports/decisiones.md` con la razón.
