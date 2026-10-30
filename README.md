# Modelo de predicción de egresos de UCI

Modelos para estimar, desde el primer día de UCI, qué pacientes tienen posibilidad de salida temprana,
y para proyectar la ocupación de camas de UCI y piso.

> Los datos de pacientes **no** están en este repositorio. Para correr los scripts, coloque
> `egresos uci 2024-2026.xlsx` (hoja "Consolidado por ingreso") en esta carpeta.

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
| `predecir.py` | Predicción para pacientes nuevos: `python predecir.py plantilla_pacientes.xlsx` |

Reglas de seguridad: los modelos se entrenan solo con datos completos y **no dan predicción individual
si falta SOFA, Barthel o edad**; no se asume ningún valor por el paciente.

Evidencia científica y variables recomendadas para capturar el día 1: `evidencia_variables_primer_dia.md`
y `ficha_captura_dia1_uci.xlsx`.

Entorno: Python 3.12; ejecutar con `python -X utf8`.
