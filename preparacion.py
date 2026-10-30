"""Carga, limpieza y construcción de variables comunes a los scripts de análisis y modelos."""
import numpy as np
import pandas as pd

ARCHIVO = "egresos uci 2024-2026.xlsx"
INICIO, FIN, CORTE_TEMPORAL = "2024-10-01", "2026-08-31", "2026-02-01"

BANDERAS_DX = {
    "desnutricion": ("E40", "E41", "E42", "E43", "E44", "E46"),
    "sepsis": ("A40", "A41", "R65"),
    "neumonia": ("J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69"),
    "falla_respiratoria": ("J96", "J80"),
    "iam": ("I21", "I22"),
    "cardiopatia_isquemica_cronica": ("I25",),
    "falla_cardiaca": ("I50",),
    "acv": ("I60", "I61", "I62", "I63", "I64", "I67"),
    "arritmia": ("I47", "I48", "I49"),
    "valvulopatia": ("I34", "I35", "I36"),
    "cancer": tuple(f"C{i:02d}" for i in range(100)),
    "erc": ("N17", "N18"),
    "diabetes": ("E10", "E11", "E14"),
    "hta": ("I10",),
    "trauma": ("S", "T0", "T1"),
    "hemorragia_digestiva": ("K92",),
    "obstetrico": ("O",),
}


def cargar():
    df = pd.read_excel(ARCHIVO, sheet_name=0)
    df = df[df["fecha_ingreso_min"].between(INICIO, FIN)]
    return construir_variables(df).reset_index(drop=True)


def construir_variables(df, aseguradoras=None):
    """Limpieza y variables derivadas. Sirve para la base histórica y para pacientes nuevos
    (en ese caso las columnas de desenlace y de signos vitales pueden no existir).
    `aseguradoras`: niveles de entrenamiento; las demás se agrupan en "OTRA"."""
    df = df.rename(columns={"score_inicial": "news_inicial", "score_final": "news_final",
                            "ta_alta": "ta_sistolica", "ta_baja": "ta_diastolica"}).copy()

    # Valores fisiológicamente imposibles -> faltante
    for c, lo, hi in [("edad", 15, 105), ("peso", 25, 250), ("fc", 20, 250),
                      ("ta_sistolica", 50, 260), ("ta_diastolica", 20, 160), ("temp_piel", 30, 43)]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda s: s.between(lo, hi))
    for c in ["puntuacion_sofa_inicial", "barthel_inicial", "news_inicial"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"ta_sistolica", "ta_diastolica"} <= set(df):
        invertida = df["ta_diastolica"] >= df["ta_sistolica"]
        df.loc[invertida, ["ta_sistolica", "ta_diastolica"]] = np.nan
        df["tam"] = (df["ta_sistolica"] + 2 * df["ta_diastolica"]) / 3

    if "sexo_id" in df:
        df["sexo_m"] = (df["sexo_id"] == "M").astype(float).where(df["sexo_id"].notna())
    df["procedencia_piso"] = (df["servicio_inicial"].str.strip().str.upper() == "HOSPITALIZACION").astype(int)
    if "fecha_ingreso_min" in df:
        df["mes_ingreso"] = df["fecha_ingreso_min"].dt.month
        df["dia_semana_ingreso"] = df["fecha_ingreso_min"].dt.dayofweek
        df["fin_de_semana"] = (df["dia_semana_ingreso"] >= 5).astype(int)

    codigos = df["diagnostico"].fillna("").astype(str).str.split(r" \| ").apply(
        lambda l: [c.strip()[:4].upper() for c in l if c.strip()])
    df["n_diagnosticos"] = codigos.str.len()
    principal = codigos.str[0].fillna("")
    df["capitulo_dx"] = principal.str[0].replace("", "NA").fillna("NA")
    for nombre, pref in BANDERAS_DX.items():
        df[f"dxp_{nombre}"] = principal.str.startswith(pref).astype(int)            # diagnóstico principal
        df[f"dxa_{nombre}"] = codigos.apply(lambda l: int(any(c.startswith(pref) for c in l)))  # cualquiera

    t = df["tercero"].astype(str).str.strip()
    if aseguradoras is None:
        vc = t.value_counts()
        aseguradoras = list(vc[vc >= 30].index) + ["OTRA"]
    df["aseguradora"] = t.where(t.isin(aseguradoras), "OTRA")

    if "dias_uci" in df:
        df["uci_prolongada"] = (df["dias_uci"] > 7).astype(int)
    if "servicio_final" in df:
        df["egreso_directo_uci"] = (df["servicio_final"] == "UCI").astype(int)
    return df


DXP = [f"dxp_{k}" for k in BANDERAS_DX]
# Variables finales: conocidas al ingreso a UCI y sin fuga de información (ver 02_seleccion_variables.py)
NUMERICAS_MODELO = ["edad", "procedencia_piso", "puntuacion_sofa_inicial", "barthel_inicial"] + DXP
CATEGORICAS_MODELO = ["aseguradora", "capitulo_dx"]
OBJETIVOS = {"uci_prolongada": "Estancia en UCI > 7 días",
             "egreso_directo_uci": "Salida de la clínica directamente desde UCI (condición de egreso desconocida)"}


# SOFA, Barthel y edad faltan mucho más en estancias cortas (se registran durante la estancia):
# su ausencia es información del desenlace. Los modelos se entrenan solo con datos completos y no
# se da predicción individual cuando falta alguna (ver predecir.py).
REQUERIDAS = ["edad", "puntuacion_sofa_inicial", "barthel_inicial"]


def solo_completos(df):
    return df[df[REQUERIDAS].notna().all(axis=1)].reset_index(drop=True)


def X_modelo(df, categorias=None):
    """Matriz de predictores. `categorias` fija los niveles de entrenamiento al predecir datos nuevos."""
    X = df[NUMERICAS_MODELO + CATEGORICAS_MODELO].copy()
    for c in CATEGORICAS_MODELO:
        X[c] = pd.Categorical(X[c], categories=categorias[c]) if categorias else X[c].astype("category")
    return X


CONTINUAS = ["edad", "puntuacion_sofa_inicial", "barthel_inicial"]
# dxp_cancer, dxp_obstetrico y dxp_trauma coinciden con capítulos CIE-10 completos (C, O, S/T):
# en modelos estadísticos se omiten para evitar colinealidad perfecta.
DXP_ESTADISTICO = [c for c in DXP if c not in ("dxp_cancer", "dxp_obstetrico", "dxp_trauma")]


def diseno(df, ref=None):
    """Variables estandarizadas + dummies. `ref` reutiliza medias/desvíos/niveles del entrenamiento.
    Sin indicadores de faltante: un valor faltante queda en 0 = promedio de la cohorte (solo ocurre
    al proyectar ocupación de pacientes con datos incompletos; el entrenamiento usa datos completos)."""
    if ref is None:
        ref = {"media": df[CONTINUAS].mean(), "sd": df[CONTINUAS].std(),
               "aseg": [a for a in df["aseguradora"].value_counts().index[1:]],
               "cap": [c for c in df["capitulo_dx"].value_counts().index[1:8]]}
    X = pd.DataFrame(index=df.index)
    for c in CONTINUAS:
        X[c] = ((df[c] - ref["media"][c]) / ref["sd"][c]).fillna(0.0)
    X["procedencia_piso"] = df["procedencia_piso"].astype(float)
    for c in DXP_ESTADISTICO:
        X[c] = df[c].astype(float)
    for a in ref["aseg"]:
        X[f"aseg_{a[:25]}"] = (df["aseguradora"] == a).astype(float)
    for c in ref["cap"]:
        X[f"cap_{c}"] = (df["capitulo_dx"] == c).astype(float)
    X = X.loc[:, X.sum() > 0] if "cols" not in ref else X.reindex(columns=ref["cols"], fill_value=0.0)
    ref["cols"] = list(X.columns)
    return X, ref
