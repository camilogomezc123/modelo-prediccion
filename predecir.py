"""Predicción para pacientes nuevos o actualmente hospitalizados.

Uso:
    python predecir.py pacientes.xlsx [salida.xlsx]

Columnas del Excel de entrada (ver plantilla_pacientes.xlsx):
  obligatorias: ingreso, tercero, diagnostico, edad, servicio_inicial,
                puntuacion_sofa_inicial, barthel_inicial
  opcionales para proyectar ocupación: servicio_actual (UCI / HOSPITALIZACION), dias_en_servicio
  `diagnostico` usa el formato de la base: "I219 - DESCRIPCION | E440 - ..." (el primero es el principal).

Si existen modelos/dia1_*.joblib (09_modelos_dia1.py) suma, desde el primer día de UCI, la probabilidad
de salir de UCI en 2 y 3 días (solo planeación) y la de pasar a piso en 3 días con posibilidad ALTA/MEDIA/BAJA.

Salida (Excel): hoja Predicciones (probabilidades y nivel de riesgo por paciente), hoja Ocupacion
(camas esperadas en 1, 3 y 7 días si se dieron servicio_actual y dias_en_servicio) y hoja Notas.
"""
import sys
from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from preparacion import construir_variables, X_modelo, OBJETIVOS, REQUERIDAS
import ocupacion

DIR_MODELOS = Path(__file__).parent / "modelos"
OBLIGATORIAS = ["ingreso", "tercero", "diagnostico", "edad", "servicio_inicial",
                "puntuacion_sofa_inicial", "barthel_inicial"]
ETIQUETA = {"uci_prolongada": "UCI_mas_7_dias", "egreso_directo_uci": "sale_de_clinica_desde_UCI"}
# Salida temprana de UCI desde el primer día (09_modelos_dia1.py, solo variables base)
ETIQUETA_DIA1 = {"salida_uci_2d": "sale_de_UCI_en_2_dias", "salida_uci_3d": "sale_de_UCI_en_3_dias",
                 "piso_3d": "pasa_a_piso_en_3_dias"}
# Sin condición de egreso, "sale desde UCI" mezcla altas a casa con posibles fallecidos o remisiones:
# solo sirve para planear camas de piso, no como pronóstico. No se le asigna nivel de riesgo.
# La salida de UCI en 2 o 3 días incluye esas salidas directas; solo el paso a piso es con certeza vivo.
SOLO_PLANEACION = {"egreso_directo_uci", "salida_uci_2d", "salida_uci_3d"}


def nivel(p, u):
    return np.where(p >= u["youden"], "ALTO", np.where(p < u["sens90"], "BAJO", "MEDIO"))


def predecir(entrada):
    faltan = [c for c in OBLIGATORIAS if c not in entrada]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(faltan)}")
    avisos = []
    modelos = {obj: joblib.load(DIR_MODELOS / f"modelo_{obj}.joblib") for obj in OBJETIVOS}
    primero = next(iter(modelos.values()))
    df = construir_variables(entrada, aseguradoras=primero["aseguradoras"])

    # Sin SOFA, Barthel o edad no se da riesgo individual: no se asume ningún valor por el paciente
    faltantes = df[REQUERIDAS].isna()
    completo = ~faltantes.any(axis=1).values
    salida = entrada[["ingreso"]].copy()
    salida["datos"] = np.where(completo, "completos", "INCOMPLETOS: falta " + faltantes.apply(
        lambda f: ", ".join(c for c in REQUERIDAS if f[c]), axis=1))
    if (~completo).any():
        avisos.append(f"{(~completo).sum()} paciente(s) sin SOFA, Barthel o edad: no reciben predicción de riesgo. "
                      "Registre el dato y vuelva a predecir.")
    X = X_modelo(df[completo], categorias=primero["categorias"])
    for obj, m in modelos.items():
        prob = pd.Series(np.nan, index=salida.index)
        riesgo = pd.Series("SIN PREDICCIÓN", index=salida.index)
        if completo.any():
            p = m["modelo"].predict_proba(X)[:, 1]
            prob[completo], riesgo[completo] = p.round(3), nivel(p, m["umbrales"])
        salida[f"prob_{ETIQUETA[obj]}"] = prob
        if obj not in SOLO_PLANEACION:
            salida[f"riesgo_{ETIQUETA[obj]}"] = riesgo

    dia1 = {obj: joblib.load(DIR_MODELOS / f"dia1_{obj}.joblib")
            for obj in ETIQUETA_DIA1 if (DIR_MODELOS / f"dia1_{obj}.joblib").exists()}
    for obj, m in dia1.items():
        prob = pd.Series(np.nan, index=salida.index)
        posibilidad = pd.Series("SIN PREDICCIÓN", index=salida.index)
        if completo.any():
            p = m["modelo"].predict_proba(df.loc[completo, m["columnas"]])[:, 1]
            u = m["umbrales"]
            prob[completo] = p.round(3)
            posibilidad[completo] = np.where(p >= u["youden"], "ALTA", np.where(p < u["sens90"], "BAJA", "MEDIA"))
        salida[f"prob_{ETIQUETA_DIA1[obj]}"] = prob
        if obj not in SOLO_PLANEACION:
            salida[f"posibilidad_{ETIQUETA_DIA1[obj]}"] = posibilidad
    modelos.update(dia1)

    ocup = None
    if {"servicio_actual", "dias_en_servicio"} <= set(entrada):
        servicio = entrada["servicio_actual"].astype(str).str.strip().str.upper()
        validos = servicio.isin(["UCI", "HOSPITALIZACION"]) & pd.to_numeric(entrada["dias_en_servicio"], errors="coerce").notna()
        if (~validos).any():
            avisos.append(f"{(~validos).sum()} paciente(s) sin servicio_actual/dias_en_servicio válidos: no entran en la ocupación.")
        if validos.any():
            mod_oc = joblib.load(DIR_MODELOS / "modelo_ocupacion.joblib")
            proy = ocupacion.proyectar(mod_oc, df[validos.values], servicio[validos].values,
                                       pd.to_numeric(entrada.loc[validos, "dias_en_servicio"]).values)
            salida = salida.join(proy.round(3))
            ocup = pd.DataFrame({f"en {h} día(s)": {
                "camas UCI esperadas": proy[f"P_UCI_{h}d"].sum(),
                "camas de piso esperadas": proy[f"P_piso_{h}d"].sum(),
                "egresos esperados (acumulado)": proy[f"P_egreso_{h}d"].sum()} for h in (1, 3, 7)}).round(1)
            ocup.insert(0, "hoy", [(servicio[validos] == "UCI").sum(), (servicio[validos] == "HOSPITALIZACION").sum(), 0])
            promedio = validos.values & ~completo
            salida["proyeccion_ocupacion"] = np.where(~validos.values, "sin proyección",
                                                      np.where(promedio, "con perfil promedio (faltan datos)", "individual"))
            if promedio.any():
                avisos.append(f"{promedio.sum()} paciente(s) con datos incompletos se proyectan en la ocupación con el "
                              "perfil promedio de la cohorte (ocupan cama aunque no tengan predicción individual).")
    return salida, ocup, modelos, avisos


def notas(modelos, avisos):
    filas = [("Fecha de predicción", date.today().isoformat())]
    for obj, m in modelos.items():
        filas += [(f"{m['objetivo']}: modelo", m.get("nombre_modelo", m.get("familia"))),
                  (f"{m['objetivo']}: AUC en prueba (feb-ago 2026)", f"{m['auc_prueba']:.3f} (IC95 {m['ic95_prueba']})"),
                  (f"{m['objetivo']}: tasa histórica", f"{m['tasa_base']:.0%}")]
        if obj in SOLO_PLANEACION:
            filas += [(f"{m['objetivo']}: uso", "SOLO planeación de camas de piso. No es pronóstico: sin condición de "
                                                "egreso, la salida desde UCI puede ser alta a casa o fallecimiento/remisión. "
                                                "Probabilidad alta NO significa buena evolución.")]
        elif obj in ETIQUETA_DIA1:
            filas += [(f"{m['objetivo']}: posibilidad ALTA", f"probabilidad >= {m['umbrales']['youden']:.2f}"),
                      (f"{m['objetivo']}: posibilidad BAJA", f"probabilidad < {m['umbrales']['sens90']:.2f} (por debajo "
                                                            "queda solo ~10 % de los que pasan a piso en 3 días)")]
        else:
            filas += [(f"{m['objetivo']}: riesgo ALTO", f"probabilidad >= {m['umbrales']['youden']:.2f}"),
                      (f"{m['objetivo']}: riesgo BAJO", f"probabilidad < {m['umbrales']['sens90']:.2f} (por debajo queda solo ~10 % de los casos)")]
    filas += [("Ocupación", "Modelo semi-Markov. Cubre solo a los pacientes listados: no incluye ingresos nuevos "
                            "ni reingresos de piso a UCI."),
              ("Uso", "Herramienta de apoyo a la gestión de camas; no reemplaza el criterio clínico.")]
    filas += [("Aviso", a) for a in avisos]
    return pd.DataFrame(filas, columns=["concepto", "valor"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ruta = Path(sys.argv[1])
    destino = Path(sys.argv[2]) if len(sys.argv) > 2 else ruta.with_name(f"predicciones_{date.today():%Y%m%d}.xlsx")
    salida, ocup, modelos, avisos = predecir(pd.read_excel(ruta))
    with pd.ExcelWriter(destino) as w:
        salida.to_excel(w, sheet_name="Predicciones", index=False)
        if ocup is not None:
            ocup.to_excel(w, sheet_name="Ocupacion")
        notas(modelos, avisos).to_excel(w, sheet_name="Notas", index=False)
    print(f"{len(salida)} paciente(s) procesados -> {destino}")
    for a in avisos:
        print("Aviso:", a)
    if ocup is not None:
        print("\nOcupación esperada:\n" + ocup.to_string())


if __name__ == "__main__":
    main()
