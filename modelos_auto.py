"""Variables adicionales, objetivos de "salida temprana de UCI" y ensambles para la búsqueda automática.

Todas las variables se conocen el primer día de UCI:
  - historia previa del paciente (ingresos anteriores en la base, días de UCI previos, reingreso a 30 días)
  - código CIE-10 de 3 caracteres del diagnóstico principal (codificado por target encoding)
  - carga de la UCI el día del ingreso (pacientes que ya estaban en UCI ese día)
"""
import numpy as np
import pandas as pd

from preparacion import ARCHIVO, NUMERICAS_MODELO, CATEGORICAS_MODELO

OBJETIVOS_AUTO = {
    "salida_uci_2d": "Sale de UCI en 2 días o menos",
    "salida_uci_3d": "Sale de UCI en 3 días o menos",
    "piso_3d": "Pasa de UCI a piso en 3 días o menos (salida temprana con certeza viva)",
}
EXTRAS_HISTORIA = ["hist_ingresos_previos", "hist_dias_uci_previos", "hist_reingreso_30d"]
EXTRAS_CARGA = ["carga_uci_dia_ingreso"]
CONJUNTOS = {
    "base": {"num": NUMERICAS_MODELO, "cat": CATEGORICAS_MODELO, "te": []},
    "base+historia": {"num": NUMERICAS_MODELO + EXTRAS_HISTORIA, "cat": CATEGORICAS_MODELO, "te": []},
    "base+historia+dx3": {"num": NUMERICAS_MODELO + EXTRAS_HISTORIA, "cat": CATEGORICAS_MODELO, "te": ["dx3"]},
    "completo": {"num": NUMERICAS_MODELO + EXTRAS_HISTORIA + EXTRAS_CARGA, "cat": CATEGORICAS_MODELO, "te": ["dx3"]},
}


def historia_cruda():
    """Todos los ingresos del archivo (sin filtro de fechas) para calcular historia y carga."""
    h = pd.read_excel(ARCHIVO, sheet_name=0, usecols=["ingreso", "paciente_id", "fecha_ingreso_min",
                                                       "fecha_salida_max", "dias_uci", "secuencia_servicios"])
    return h


def agregar_extras(df, historia):
    """Agrega variables de historia, dx3 y carga de UCI. `historia` = historia_cruda()."""
    df = df.copy()
    df["dx3"] = df["diagnostico"].fillna("").astype(str).str.strip().str[:3].str.upper().replace("", "NA")

    h = historia.sort_values("fecha_ingreso_min")
    previos, dias_prev, reing = [], [], []
    por_paciente = {k: g for k, g in h.groupby("paciente_id")}
    for _, r in df.iterrows():
        g = por_paciente.get(r.get("paciente_id"))
        if g is None or pd.isna(r.get("fecha_ingreso_min")):
            previos.append(0); dias_prev.append(0); reing.append(0)
            continue
        ant = g[(g["fecha_salida_max"] < r["fecha_ingreso_min"]) & (g["ingreso"] != r.get("ingreso"))]
        previos.append(len(ant))
        dias_prev.append(int(ant["dias_uci"].sum()))
        reing.append(int(len(ant) > 0 and (r["fecha_ingreso_min"] - ant["fecha_salida_max"].max()).days <= 30))
    df["hist_ingresos_previos"], df["hist_dias_uci_previos"], df["hist_reingreso_30d"] = previos, dias_prev, reing

    # Carga de UCI: ingresos que empezaron en UCI y seguían en UCI el día del ingreso del paciente
    u = h[h["secuencia_servicios"].str.startswith("UCI")]
    ini = u["fecha_ingreso_min"].values
    fin = (u["fecha_ingreso_min"] + pd.to_timedelta(u["dias_uci"], unit="D")).values
    if "fecha_ingreso_min" in df and df["fecha_ingreso_min"].notna().any():
        t = df["fecha_ingreso_min"].values.reshape(-1, 1)
        df["carga_uci_dia_ingreso"] = ((ini < t) & (t < fin)).sum(axis=1)
    else:
        df["carga_uci_dia_ingreso"] = np.nan
    return df


def agregar_objetivos(df):
    df = df.copy()
    df["salida_uci_2d"] = (df["dias_uci"] <= 2).astype(int)
    df["salida_uci_3d"] = (df["dias_uci"] <= 3).astype(int)
    df["piso_3d"] = (df["secuencia_servicios"].str.contains("UCI → HOSPITALIZACION") & (df["dias_uci"] <= 3)).astype(int)
    return df


def columnas(conjunto):
    c = CONJUNTOS[conjunto]
    return c["num"] + c["cat"] + c["te"]


class Ensamble:
    """Promedio simple o apilado (regresión logística sobre las probabilidades) de modelos ya ajustados."""

    def __init__(self, modelos, apilador=None):
        self.modelos = modelos
        self.apilador = apilador
        self.classes_ = np.array([0, 1])

    def _base(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.modelos])

    def predict_proba(self, X):
        P = self._base(X)
        p = self.apilador.predict_proba(P)[:, 1] if self.apilador is not None else P.mean(axis=1)
        return np.column_stack([1 - p, p])
