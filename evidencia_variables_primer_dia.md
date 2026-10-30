# Variables del primer día de UCI para predecir la salida temprana: evidencia y recomendaciones

Fuente de la literatura: PubMed (búsqueda del 2026-10-30).

## Qué tan bien se puede predecir la estancia en UCI con datos del día 1

- **APACHE IV** (Zimmerman et al., *Crit Care Med* 2006; 116.209 pacientes de 104 UCI): con datos del día 1, el modelo explicó solo el **21,5 % de la variación individual** de la estancia en UCI y el 62 % de la variación entre UCI. Los autores concluyen que es útil para grupos de pacientes pero **"limitado para pacientes individuales"**, sobre todo en estancias muy cortas (≤1,7 días) o muy largas. [DOI](https://doi.org/10.1097/01.CCM.0000240233.01711.D9)
- **Machine learning con eICU y MIMIC-III** (*Diagnostics* 2021): para estancia prolongada en UCI, el mejor modelo (gradient boosting) alcanzó un **AUROC de 0,742** con decenas de variables fisiológicas y de laboratorio. [DOI](https://doi.org/10.3390/diagnostics11122242)

**Conclusión para este proyecto:** un AUC cercano a 0,70 con 6 variables administrativas y de escala está en el rango esperable. Un modelo "perfecto" no existe ni en las bases internacionales. Lo que sí sube el rendimiento es **agregar variables clínicas del día 1**, como las de la tabla siguiente.

## Variables fáciles de tomar el día 1 que recomienda la evidencia

| Variable | Cómo se toma | Evidencia | ¿Está hoy en la base? |
|---|---|---|---|
| **Ventilación mecánica invasiva el día 1** (sí/no) | Registro de enfermería | Es variable del modelo de estancia APACHE IV [DOI](https://doi.org/10.1097/01.CCM.0000240233.01711.D9) | No |
| **Días de hospitalización antes de la UCI** | Fecha de ingreso hospitalario y fecha de ingreso a UCI | Es variable del modelo de estancia APACHE IV (estancia previa a la UCI) | Parcial: solo "procedencia piso" |
| **Tipo de ingreso** (posquirúrgico programado, posquirúrgico urgente, médico) | Una pregunta al ingreso | La fuente de admisión y la cirugía de urgencia forman parte de APACHE IV | No |
| **Escala de Fragilidad Clínica (CFS 1–9)** | 1 minuto, preguntando por el estado 2 semanas antes del ingreso | La fragilidad se asocia a peores desenlaces en UCI (metaanálisis, *Intensive Care Med* 2017 [DOI](https://doi.org/10.1007/s00134-017-4867-0)); la CFS se asocia a mayor estancia y peor destino al egreso (revisión sistemática [DOI](https://doi.org/10.2147/CIA.S388160)); la fragilidad, la funcionalidad y la cognición predicen desenlaces en mayores de 80 años en UCI (estudio VIP2 [DOI](https://doi.org/10.1007/s00134-019-05853-1)) | No (el Barthel es parecido) |
| **Estado nutricional al ingreso** | Diagnóstico o tamizaje nutricional | La desnutrición al ingreso predice mayor estancia en UCI pediátrica [DOI](https://doi.org/10.1016/j.jand.2019.06.250); en esta base es el diagnóstico que más alarga la estancia | Sí (diagnóstico E40–E46) |
| **SOFA y NEWS al ingreso** | Escalas habituales | Ya los usa el modelo; el SOFA es la variable más importante | Sí (NEWS solo desde mediados de 2025) |
| **Signos vitales al ingreso con hora** | Primera toma en UCI | Los de la base actual parecen ser del final de la estancia (fuga de información) | Sí, pero con la hora equivocada |
| **Vasopresores el día 1** (sí/no) | Registro de enfermería | Forma parte del componente cardiovascular del SOFA; registrarlo por separado permite usarlo aunque falte el SOFA completo | No |

## Recomendación práctica

1. **Agregar a la ficha de ingreso a UCI**, con la plantilla [ficha_captura_dia1_uci.xlsx](ficha_captura_dia1_uci.xlsx): ventilación mecánica, vasopresores, tipo de ingreso, fecha de ingreso a UCI y CFS. Toman menos de 2 minutos por paciente.
2. **SOFA, Barthel y edad deben registrarse siempre**: sin ellos el modelo no da predicción.
3. Con unos **6 meses de datos** de las variables nuevas, se reentrena con `08_busqueda_automatica.py`, que ya está preparado para incorporarlas.
