# AutoML como benchmark (FLAML)

## Configuración del experimento

| Elemento | Valor |
|---|---|
| Herramienta | FLAML (`flaml.AutoML`) |
| Versión | 2.7.0 |
| Presupuesto | 600 s (`config/config.yaml: automl.time_budget`) |
| Tiempo real medido | **616.1 s** (2.7% sobre el presupuesto) |
| Semilla | 42 (`config/config.yaml: seed`) |
| Métrica | RMSE |
| Validación | split temporal `val` (misma partición y mismas filas que el resto de modelos) |
| Test | no usado durante la búsqueda; `final_eval.py` lo evalúa después con el modelo congelado |
| Features | 28 columnas de `feat_train.parquet` / `feat_val.parquet` (las mismas que Ridge/HistGB) |
| Estimadores solicitados | `lgbm`, `xgboost`, `rf`, `extra_tree` |
| Estimadores realmente evaluados | **solo `lgbm`** (11 trials; el presupuesto se agotó antes de explorar los otros tres) |

**Honestidad sobre la búsqueda:** FLAML recibió la lista de cuatro estimadores, pero su log (`reports/results/automl.log`) muestra 11 evaluaciones consecutivas de `lgbm` y ninguna de `xgboost`, `rf` o `extra_tree`. Por tanto, este benchmark mide “FLAML con presupuesto de 600 s”, no una comparación completa entre familias.

## Mejor configuración encontrada

```json
{
  "learner": "lgbm",
  "n_estimators": 6101,
  "num_leaves": 4,
  "min_child_samples": 49,
  "learning_rate": 0.08925235460224837,
  "log_max_bin": 9,
  "colsample_bytree": 0.7588420562199519,
  "reg_alpha": 42.76465181451931,
  "reg_lambda": 0.031765883524111874
}
```

RMSE de validación del mejor trial: **0.847846**.

## Mejor manual vs AutoML bajo el mismo protocolo

| Modelo | RMSE val | RMSE test | MAE test | NDCG@10 test | Recall@10 test | Tiempo de ajuste |
|---|---:|---:|---:|---:|---:|---:|
| Item-KNN (mejor manual por RMSE) | **0.819816** | **0.845029** | **0.634926** | **0.402945** | **0.433034** | 11.2 s |
| Funk SVD (mejor manual de factorización) | 0.826117 | 0.845296 | 0.640046 | 0.295132 | 0.331952 | 19.0 s |
| HistGB + features (mejor tabular manual) | 0.847197 | 0.863531 | 0.655080 | 0.285727 | 0.322883 | 58.0 s |
| AutoML FLAML (mejor trial) | 0.847846 | 0.864332 | 0.655759 | 0.284064 | 0.317079 | 616.1 s |

Lectura:

- AutoML queda **0.000649** por encima de HistGB manual en validación y **0.000801** por encima en test. La diferencia es pequeña, pero no hay mejora: el ajuste manual con el mismo feature set es igual o mejor y tarda ~10.6 veces menos.
- Frente a Item-KNN, AutoML pierde **0.028030** de RMSE en validación (3.4%) y **0.019303** en test (2.3%). El modelo colaborativo aprovecha la estructura usuario-película que el tabular no recibe directamente.
- En ranking, AutoML también queda por debajo de Item-KNN (NDCG@10 0.2841 vs 0.4029; Recall@10 0.3171 vs 0.4330).

## Trade-offs

| Dimensión | AutoML | Modelo manual |
|---|---|---|
| Rendimiento | Peor que Item-KNN y que HistGB manual en este protocolo | Item-KNN gana en RMSE y ranking; HistGB manual empata en RMSE tabular |
| Tiempo | 616.1 s para 11 trials | Item-KNN 11.2 s; HistGB 58.0 s; Funk SVD 19.0 s |
| Interpretabilidad | Baja: configuración ganadora y modelo de caja negra | Item-KNN explica por vecinos; HistGB da importancias; el baseline de sesgos es totalmente explicable |
| Control | Bajo: la búsqueda decide qué probar y cuándo parar | Alto: grids explícitos en `config/config.yaml` |
| Reproducibilidad | Semilla fija, pero la búsqueda depende del tiempo de pared y de la máquina | Más determinista con la misma semilla y los mismos grids |
| Cold start | Features con fillna(0)/media; no aporta señal nueva | El baseline de sesgos es un fallback explícito; Item-KNN cae a ese baseline |

## Advertencia metodológica

AutoML tabular **no reemplaza** a la factorización matricial colaborativa. FLAML recibe 28 features derivadas (sesgos OOF, medias, géneros, año, popularidad, afinidad), pero no la matriz dispersa usuario-película ni las co-calificaciones. Item-KNN y Funk SVD explotan directamente esa estructura; AutoML solo la ve a través de resúmenes. Por eso no es sorprendente que pierda frente a Item-KNN, y por eso su papel aquí es el de **benchmark de esfuerzo/rendimiento**, no el de candidato principal.

## Repetición con otra semilla

No se ejecutó una segunda semilla dentro de este presupuesto de tiempo. El resultado debe leerse como una sola realización (semilla 42, 616.1 s); no se reporta variación entre semillas porque no se midió.
