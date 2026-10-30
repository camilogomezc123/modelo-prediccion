"""Predicción cruzada: cada variable como objetivo y como predictor de las demás.

Escala común de "poder predictivo" (0 = nada, 1 = perfecto):
  - objetivo binario o categórico: (AUC - 0,5) / 0,5   (AUC multiclase = promedio uno-contra-resto)
  - objetivo numérico: R² (negativos -> 0); los días se transforman con log(1 + x)
Validación: 5 particiones agrupadas por paciente (StratifiedGroupKFold / GroupKFold).

Se incluyen también variables excluidas del modelo de producción (finales, signos vitales,
días) porque aquí el fin es entender cómo se relacionan, no predecir al ingreso.
"""
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from preparacion import cargar

SEMILLA = 42
# nombre corto -> (columna, tipo)   tipo: "bin", "cat", "num"
VARIABLES = {
    "edad": ("edad", "num"),
    "sexo_m": ("sexo_m", "bin"),
    "proced_piso": ("procedencia_piso", "bin"),
    "sofa_ini": ("puntuacion_sofa_inicial", "num"),
    "barthel_ini": ("barthel_inicial", "num"),
    "news_ini": ("news_inicial", "num"),
    "aseguradora": ("aseguradora", "cat"),
    "cap_dx": ("capitulo_dx_top", "cat"),
    "n_dx": ("n_diagnosticos", "num"),
    "fc": ("fc", "num"),
    "ta_sist": ("ta_sistolica", "num"),
    "ta_diast": ("ta_diastolica", "num"),
    "temp": ("temp_piel", "num"),
    "peso": ("peso", "num"),
    "sofa_fin": ("puntuacion_sofa_final", "num"),
    "barthel_fin": ("barthel_final", "num"),
    "news_fin": ("news_final", "num"),
    "dias_uci": ("log_dias_uci", "num"),
    "dias_hosp": ("log_dias_hosp", "num"),
    "traslados": ("numero_traslados", "num"),
    "uci_>7d": ("uci_prolongada", "bin"),
    "egreso_uci": ("egreso_directo_uci", "bin"),
}
TRIADAS = [
    ("sofa_ini", "barthel_ini", "uci_>7d"),
    ("sofa_ini", "cap_dx", "egreso_uci"),
    ("edad", "barthel_ini", "sofa_ini"),
    ("aseguradora", "cap_dx", "egreso_uci"),
    ("sofa_ini", "news_ini", "uci_>7d"),
    ("dias_uci", "proced_piso", "egreso_uci"),
    ("sofa_ini", "sofa_fin", "egreso_uci"),
    ("barthel_ini", "barthel_fin", "dias_hosp"),
    ("ta_diast", "sofa_fin", "egreso_uci"),
    ("edad", "cap_dx", "aseguradora"),
]


def preparar():
    df = cargar()
    df["log_dias_uci"] = np.log1p(df["dias_uci"])
    df["log_dias_hosp"] = np.log1p(df["dias_hospitalizacion"])
    top = df["capitulo_dx"].value_counts().index[:8]
    df["capitulo_dx_top"] = df["capitulo_dx"].where(df["capitulo_dx"].isin(top), "otros")
    return df


def modelos(tipo):
    if tipo == "num":
        return {"Lineal (Ridge)": Ridge(1.0),
                "Árbol": DecisionTreeRegressor(max_depth=5, min_samples_leaf=30, random_state=SEMILLA),
                "kNN": KNeighborsRegressor(25),
                "RandomForest": RandomForestRegressor(300, min_samples_leaf=10, n_jobs=-1, random_state=SEMILLA),
                "GradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                  min_samples_leaf=40, random_state=SEMILLA),
                "Red neuronal": MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=600, early_stopping=True,
                                             random_state=SEMILLA)}
    return {"Logística": LogisticRegression(C=1.0, max_iter=5000),
            "Árbol": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=SEMILLA),
            "kNN": KNeighborsClassifier(25),
            "RandomForest": RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=SEMILLA),
            "GradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                               min_samples_leaf=40, random_state=SEMILLA),
            "Red neuronal": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=600, early_stopping=True,
                                          random_state=SEMILLA)}


def pipeline(estimador, predictores):
    num = [VARIABLES[p][0] for p in predictores if VARIABLES[p][1] != "cat"]
    cat = [VARIABLES[p][0] for p in predictores if VARIABLES[p][1] == "cat"]
    pasos = []
    if num:
        pasos.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        pasos.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return make_pipeline(ColumnTransformer(pasos), estimador)


def poder(df, predictores, objetivo, estimador):
    """Poder predictivo (0-1) de `predictores` sobre `objetivo` con CV agrupada por paciente."""
    col, tipo = VARIABLES[objetivo]
    d = df[df[col].notna()].reset_index(drop=True)
    X = d[[VARIABLES[p][0] for p in predictores]]
    y = d[col].values
    grupos = d["paciente_id"].values
    if tipo == "num":
        folds = GroupKFold(5).split(X, y, grupos)
        pred = np.zeros(len(y))
    else:
        y = pd.factorize(y)[0]
        folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(X, y, grupos)
        pred = np.zeros((len(y), y.max() + 1))
    modelo = pipeline(estimador, predictores)
    for tr, te in folds:
        m = clone(modelo).fit(X.iloc[tr], y[tr])
        if tipo == "num":
            pred[te] = m.predict(X.iloc[te])
        else:
            pred[np.ix_(te, m.classes_)] = m.predict_proba(X.iloc[te])
    if tipo == "num":
        return max(r2_score(y, pred), 0.0), "R²"
    auc = roc_auc_score(y, pred[:, 1]) if tipo == "bin" else roc_auc_score(y, pred, multi_class="ovr")
    return max((auc - 0.5) / 0.5, 0.0), f"AUC={auc:.3f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df = preparar()
    nombres = list(VARIABLES)

    # 1. Matriz por pares A -> B (árbol de decisión con una sola variable)
    print("=" * 100 + "\n1. MATRIZ A -> B: poder predictivo de la variable de la FILA sobre la de la COLUMNA (0-100)\n" + "=" * 100)
    mat = pd.DataFrame(index=nombres, columns=nombres, dtype=float)
    for a, b in itertools.permutations(nombres, 2):
        est = modelos(VARIABLES[b][1])["Árbol"]
        mat.loc[a, b] = poder(df, [a], b, est)[0]
    mat.to_csv("prediccion_cruzada_pares.csv", sep=";", decimal=",", encoding="utf-8-sig")
    print((mat * 100).round(0).astype("Int64").to_string())

    print("\n-- Pares más asimétricos (A predice B mucho mejor que B predice A) --")
    print("(solo pares del mismo tipo: numérica-numérica o categórica-categórica, para comparar en la misma escala)")
    asim = []
    es_num = lambda v: VARIABLES[v][1] == "num"
    for a, b in itertools.combinations(nombres, 2):
        if es_num(a) != es_num(b):
            continue
        asim.append({"A": a, "B": b, "A->B": mat.loc[a, b], "B->A": mat.loc[b, a],
                     "diferencia": abs(mat.loc[a, b] - mat.loc[b, a])})
    asim = pd.DataFrame(asim).sort_values("diferencia", ascending=False)
    print((asim.head(15).set_index(["A", "B"]) * 100).round(1).to_string())

    # 2. Todas contra una, con 6 algoritmos
    print("\n" + "=" * 100 + "\n2. CADA VARIABLE PREDICHA POR TODAS LAS DEMÁS (0-100), 6 ALGORITMOS\n" + "=" * 100)
    filas = []
    for obj in nombres:
        resto = [v for v in nombres if v != obj]
        for nombre_m, est in modelos(VARIABLES[obj][1]).items():
            filas.append({"objetivo": obj, "modelo": nombre_m, "poder": poder(df, resto, obj, est)[0]})
        print(f"  listo: {obj}", flush=True)
    todas = pd.DataFrame(filas).pivot(index="objetivo", columns="modelo", values="poder").loc[nombres]
    todas["mejor"] = todas.max(axis=1)
    todas["mejor_modelo"] = todas.drop(columns="mejor").idxmax(axis=1)
    todas.to_csv("prediccion_cruzada_todas.csv", sep=";", decimal=",", encoding="utf-8-sig")
    t = todas.copy()
    num_cols = [c for c in t if c != "mejor_modelo"]
    t[num_cols] = (t[num_cols] * 100).round(1)
    print(t.sort_values("mejor", ascending=False).to_string())

    # 3. Triadas: A+B -> C, A+C -> B, B+C -> A
    print("\n" + "=" * 100 + "\n3. TRIADAS: las tres direcciones (0-100; mejor de GradientBoosting y Logística/Lineal)\n" + "=" * 100)
    filas = []
    for tri in TRIADAS:
        for obj in tri:
            pred = [v for v in tri if v != obj]
            ms = modelos(VARIABLES[obj][1])
            res = {n: poder(df, pred, obj, ms[n]) for n in ["GradientBoosting", list(ms)[0]]}
            n_mejor = max(res, key=lambda n: res[n][0])
            filas.append({"triada": " / ".join(tri), "dirección": f"{' + '.join(pred)} -> {obj}",
                          "poder": round(res[n_mejor][0] * 100, 1), "detalle": res[n_mejor][1], "modelo": n_mejor})
    tri = pd.DataFrame(filas)
    tri.to_csv("prediccion_cruzada_triadas.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    for nombre_t, g in tri.groupby("triada", sort=False):
        print(f"\n[{nombre_t}]")
        print(g.drop(columns="triada").to_string(index=False))


if __name__ == "__main__":
    main()
