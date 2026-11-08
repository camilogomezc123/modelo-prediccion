# Modelo de predicción de egresos de UCI

Modelos para estimar, desde el primer día de UCI, qué pacientes tienen posibilidad de salida temprana,
y para proyectar la ocupación de camas de UCI y piso.

> Los datos de pacientes **no** están en este repositorio. Para correr los scripts, coloque
> `egresos uci 2024-2026.xlsx` (hoja "Consolidado por ingreso") en esta carpeta.

## Uso diario: una sola página con todos los modelos

Con XAMPP encendido: **http://localhost/Modelo%20de%20prediccion/** (solo abre desde este equipo, ver `.htaccess`).
Se llenan los datos del paciente al ingreso a UCI y la página muestra, en una sola consulta:

- Salida temprana desde el día 1 (`09_modelos_dia1.py`): paso a piso en 3 días con posibilidad ALTA/MEDIA/BAJA y salida de UCI en 2 y 3 días (solo planeación).
- UCI más de 7 días con dos estimaciones (ensamble del proyecto y UCI-IA), días esperados y probabilidad por rango.
- Camino de salida (UCI-IA y proyecto), salida de UCI día a día los próximos 14 días y ocupación semi-Markov a 1, 3 y 7 días.

`index.php` llama a `prediccion_unificada.py`, que deja un proceso local (127.0.0.1:8765) con los modelos en memoria:
la primera consulta tarda unos 12 s y las siguientes unos 2 s. `index.php#demo` carga un paciente ficticio de ejemplo.
Sin edad, SOFA o Barthel no se da ninguna predicción.

## Flujo

| Script | Qué hace |
|---|---|
| `preparacion.py` | Carga, limpieza y construcción de variables compartida por todos los scripts |
| `01`–`03` | Análisis, selección y confrontación de variables |
| `04_modelos.py` | Modelos de estancia en UCI > 7 días y salida directa desde UCI |
| `05_prediccion_cruzada.py` | Predicción cruzada entre variables (A+B→C y C→A/B) |
| `06_modelos_multiestado_ml.py` | Markov multiestado, riesgos competitivos, Cox y machine learning |
| `07_integracion.py` | Ensamble final calibrado y modelo de ocupación (semi-Markov, `ocupacion.py`) |
| `08_busqueda_automatica.py` | Búsqueda automática (Optuna, 8 familias de algoritmos) para salida temprana de UCI |
| `09_modelos_dia1.py` | Modelos de salida temprana de UCI con solo variables base, listos para `predecir.py` |
| `predecir.py` | Predicción para pacientes nuevos: `python predecir.py plantilla_pacientes.xlsx` |

Reglas de seguridad: los modelos se entrenan solo con datos completos y **no dan predicción individual
si falta SOFA, Barthel o edad**; no se asume ningún valor por el paciente.

Evidencia científica y variables recomendadas para capturar el día 1: `evidencia_variables_primer_dia.md`
y `ficha_captura_dia1_uci.xlsx`.

Entorno: Python 3.12; ejecutar con `python -X utf8`.

## Módulo `uci_control` (integrado desde UCI-IA, rama `claude/uci-control-ai`)

| Pieza | Archivo | Documentación |
|---|---|---|
| Destino y estancia al ingreso | `uci_control/prediccion/trayectoria.py` | `docs/MODELO_TRAYECTORIA.md` |
| Egreso diario y ocupación a 7 días | `uci_control/prediccion/ocupacion.py` | `docs/EGRESO_DIARIO.md` |
| Criterios de alta, barreras, plan de camas y aprendizaje del estado diario | `uci_control/camas/` | `docs/GESTION_CAMAS.md` |

Se trajo el paquete completo porque estos módulos dependen de `conectores`, `config`, `modelos`,
`diario` y `prediccion.riesgo`. `datos_prueba/` genera datos **sintéticos** para las pruebas.

```bash
pip install pandas openpyxl scikit-learn pytest
python -m pytest tests/test_trayectoria.py tests/test_ocupacion.py tests/test_aprendizaje.py
python -m uci_control modelo-trayectoria --egresos "egresos uci 2024-2026.xlsx" --salida modelos/
python -m uci_control modelo-ocupacion --egresos "egresos uci 2024-2026.xlsx" --camas-totales 66 --salida modelos/
```

No se versionan modelos (`*.joblib`), bases del historial (`*.sqlite`) ni el Excel de egresos.
