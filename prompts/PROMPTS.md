# Ejecución y pruebas

Guía breve para ejecutar el pipeline MovieLens 25M de principio a fin. Ejecuta los comandos desde la raíz del repositorio (`ml-movielens`), en una terminal PowerShell, y espera a que termine cada etapa antes de iniciar la siguiente.

## Preparar el entorno

La primera vez, crea y activa el entorno virtual e instala las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En sesiones posteriores basta con activar `.venv`. Si PowerShell bloquea la activación, usa una terminal compatible o ajusta la política de ejecución de tu usuario según las reglas de tu equipo.

## Ejecutar el proyecto

El pipeline transforma los datos paso a paso. Ejecuta estos módulos en orden:

```powershell
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

El flujo descarga y documenta los datos, prepara un subconjunto y crea una partición temporal por usuario. Después explora únicamente train, compara modelos supervisados con validación, analiza clusters y ejecuta AutoML. Al final evalúa en test y desglosa los errores por segmentos. La configuración central está en `config/config.yaml`; los resultados se guardan principalmente en `data/`, `models/` y `reports/`.

No omitas ni reordenes etapas: cada comando usa artefactos generados por los anteriores. El conjunto test se reserva para la evaluación final; no debe usarse para elegir modelos, parámetros ni features.

## Ejecutar las pruebas

Desde la raíz y con el entorno activo:

```powershell
python -m pytest tests/test_leakage.py
```

Estas pruebas comprueban las protecciones contra leakage. Ejecútalas para validar los cambios de código; no sustituyen la ejecución del pipeline ni la evaluación final.
