# 3.1 Comprensión del problema (borrador listo para el informe)

**Contexto y necesidad.** Las plataformas de cine y streaming ofrecen decenas de miles de títulos; el usuario no puede explorarlos y necesita ayuda para descubrir películas que disfrutará. Un sistema de recomendación estima qué tan bien le caerá una película que aún no ha visto.

**Problema formulado (en términos de usuario).** Dado un usuario con un historial de calificaciones y una película que aún no ha calificado, ¿qué calificación (0.5 a 5) le daría? Con esa estimación se ordenan las películas candidatas y se le muestran las mejores.

**Objetivo.** Construir y comparar, con un protocolo reproducible y sin leakage, un baseline y al menos tres enfoques clásicos de recomendación sobre MovieLens 25M, complementados con clustering de películas e AutoML como benchmark.

**Variable objetivo.** `rating` (regresión). Evaluación complementaria de ranking: dado el conjunto de películas bien valoradas (rating ≥ 4) por el usuario en el futuro, ¿aparecen entre sus 10 primeras recomendaciones? (NDCG@10, Recall@10).

**Criterios de éxito.**
- *Métricas:* RMSE principal, MAE secundaria; NDCG@10 y Recall@10 como evaluación complementaria. El modelo elegido debe superar al baseline de sesgos usuario+película de forma consistente en validación y test.
- *Interpretabilidad:* poder explicar por qué una película se recomienda (sesgos, géneros, vecinos) y describir los clusters con tags y géneros.
- *Factibilidad:* entrenar y evaluar en una máquina de 16 GB de RAM / Colab en minutos, con subconjunto documentado.

**Justificación de RMSE.** Penaliza más los errores grandes (predecir 5 cuando es 1), que son los más dañinos para la confianza del usuario; MAE se reporta por ser más fácil de interpretar. Al ser el rating discreto y acotado, ambas métricas son comparables entre modelos. El ranking complementa porque un buen RMSE no garantiza un buen orden de recomendaciones.

**Restricciones y riesgos.**
| Riesgo | Efecto | Mitigación |
|---|---|---|
| Sparsity de la matriz usuario-película | Pocas co-calificaciones para vecinos/factores | Filtros de actividad mínima, contracción (shrinkage) |
| Cold start (usuarios/películas nuevos) | Sin historial en train | Fallback a sesgos; reportar % de ratings en frío y error en ese segmento |
| Sesgo de popularidad | El modelo favorece lo popular | Baseline de popularidad, análisis de error por popularidad |
| Data leakage (medias con todo el dataset, split aleatorio) | Métricas infladas | Split temporal, medias out-of-fold y solo con train |
| Subconjunto vs 25M completos | Resultados no extrapolables | Documentar criterio de muestreo y datos descartados |
| Tiempo y recursos | Imposible iterar con 25M | Muestreo de usuarios completos, grids pequeños |

## Tabla de variables (verificar cifras en el README oficial del dataset)
| Archivo | Variables | Significado |
|---|---|---|
| ratings.csv | userId, movieId, rating, timestamp | Calificación 0.5–5 (pasos de 0.5) de un usuario a una película; timestamp en segundos UTC |
| movies.csv | movieId, title, genres | Título con año entre paréntesis; géneros separados por `\|` |
| tags.csv | userId, movieId, tag, timestamp | Etiquetas libres de usuarios |
| genome-scores.csv | movieId, tagId, relevance | Relevancia (0–1) de cada tag del genoma para una película (cubre solo una parte de las películas) |
| genome-tags.csv | tagId, tag | Nombre de cada tag del genoma |
| links.csv | movieId, imdbId, tmdbId | Enlaces externos |
