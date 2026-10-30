"""Integración final: modelos de producción, reglas interpretables y proyección de ocupación.

1. Ensamble (promedio de probabilidades de logística + splines, random forest y gradient boosting,
   cada uno calibrado) frente a cada modelo por separado. Se elige por AUC en CV agrupada por
   paciente sobre el entrenamiento (no por la prueba) y se informa la prueba temporal.
2. El elegido se reentrena con toda la cohorte y se guarda en modelos/ con sus puntos de corte.
3. Reglas de un árbol de decisión de profundidad 3 en unidades reales (apoyo clínico).
4. Proyección de ocupación semi-Markov (ocupacion.py): ajuste, validación retrospectiva
   semanal en el periodo de prueba frente al Markov exponencial y a "nadie se mueve", y guardado.
Ejecutar después de 04_modelos.py (sobrescribe modelos/modelo_*.joblib con la versión final).
"""
import sys
import json
import importlib.util
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from preparacion import (cargar, solo_completos, X_modelo, OBJETIVOS, CORTE_TEMPORAL, CATEGORICAS_MODELO, CONTINUAS,
                         NUMERICAS_MODELO)
import ocupacion


def _modulo(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, archivo)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m04 = _modulo("m04", "04_modelos.py")
m06 = _modulo("m06", "06_modelos_multiestado_ml.py")
SEMILLA = 42
# Mejores configuraciones de cada familia según 04_modelos.py
CONFIG = {
    "uci_prolongada": {"C": 0.05, "hgb": dict(learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=80)},
    "egreso_directo_uci": {"C": 1.0, "hgb": dict(learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=40)},
}


def miembros(obj):
    c = CONFIG[obj]
    return {
        "Logística + splines": CalibratedClassifierCV(make_pipeline(
            m04.preprocesador(True), LogisticRegression(C=c["C"], max_iter=5000)), method="sigmoid", cv=5),
        "Random forest": CalibratedClassifierCV(make_pipeline(
            m04.preprocesador(False), RandomForestClassifier(500, min_samples_leaf=10, max_features="sqrt",
                                                             n_jobs=-1, random_state=SEMILLA)),
            method="isotonic", cv=5),
        "Gradient boosting": CalibratedClassifierCV(HistGradientBoostingClassifier(
            max_iter=int(12 / c["hgb"]["learning_rate"]), l2_regularization=1.0, categorical_features="from_dtype",
            random_state=SEMILLA, **c["hgb"]), method="isotonic", cv=5),
    }


def ensamble(obj):
    return VotingClassifier(list(miembros(obj).items()), voting="soft")


def umbrales(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return {"youden": float(thr[np.argmax(tpr - fpr)]), "sens90": float(thr[np.argmax(tpr >= 0.90)])}


# ----------------------------------------------------------------------------------------------
# Reglas del árbol en unidades reales
# ----------------------------------------------------------------------------------------------
def reglas_arbol(df, y, nombre):
    pre = ColumnTransformer([
        ("cont", SimpleImputer(strategy="median"), CONTINUAS),
        ("bin", "passthrough", [c for c in NUMERICAS_MODELO if c not in CONTINUAS]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAS_MODELO)])
    arbol = make_pipeline(pre, DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=SEMILLA))
    arbol.fit(X_modelo(df), y)
    t = arbol[-1].tree_
    nombres = [n.split("__")[-1].replace("missingindicator_", "falta_") for n in arbol[0].get_feature_names_out()]
    lineas = [f"Reglas para: {nombre} (tasa global {y.mean():.0%}). Árbol de profundidad 3 entrenado con toda la cohorte.", ""]

    def recorrer(nodo, condiciones):
        if t.children_left[nodo] == -1:
            n = int(t.n_node_samples[nodo])
            tasa = t.value[nodo][0][1] / t.value[nodo][0].sum()
            nivel = "ALTO" if tasa >= 1.5 * y.mean() else "BAJO" if tasa <= 0.67 * y.mean() else "medio"
            lineas.append(f"SI {' Y '.join(condiciones)}\n   -> {tasa:.0%} con el desenlace (n={n}) [{nivel}]")
            return
        f, u = nombres[t.feature[nodo]], t.threshold[nodo]
        binaria = f.startswith(("dxp_", "procedencia", "falta_", "aseguradora_", "capitulo_dx_"))
        izq = f"NO {f}" if binaria else f"{f} <= {u:.1f}"
        der = f"{f}" if binaria else f"{f} > {u:.1f}"
        recorrer(t.children_left[nodo], condiciones + [izq])
        recorrer(t.children_right[nodo], condiciones + [der])

    recorrer(0, [])
    return "\n".join(lineas)


# ----------------------------------------------------------------------------------------------
# Validación retrospectiva de la ocupación
# ----------------------------------------------------------------------------------------------
def censo(df, fecha):
    """Pacientes en UCI o en piso tras UCI en `fecha` (solo trayectorias con duración exacta)."""
    d = df[df["secuencia_servicios"].isin(["UCI", "UCI → HOSPITALIZACION"])].copy()
    d["fin_uci"] = d["fecha_ingreso_min"] + pd.to_timedelta(d["dias_uci"], unit="D")
    d["fin_piso"] = d["fin_uci"] + pd.to_timedelta(d["dias_hospitalizacion"], unit="D")
    en_uci = (d["fecha_ingreso_min"] <= fecha) & (fecha < d["fin_uci"])
    en_piso = (d["fin_uci"] <= fecha) & (fecha < d["fin_piso"])
    d = d[en_uci | en_piso].copy()
    d["servicio_actual"] = np.where(en_uci[en_uci | en_piso], "UCI", "HOSPITALIZACION")
    d["dias_en_servicio"] = np.where(d["servicio_actual"] == "UCI", (fecha - d["fecha_ingreso_min"]).dt.days,
                                     (fecha - d["fin_uci"]).dt.days)
    return d


def validar_ocupacion(df, modelo, Q):
    fechas = pd.date_range(pd.Timestamp(CORTE_TEMPORAL) + pd.Timedelta(days=14), "2026-08-24", freq="7D")
    filas = []
    for fecha in fechas:
        d = censo(df, fecha)
        if d.empty:
            continue
        p = ocupacion.proyectar(modelo, d, d["servicio_actual"].values, d["dias_en_servicio"].values)
        for h in (1, 3, 7):
            futuro = fecha + pd.Timedelta(days=h)
            obs_uci = ((d["fecha_ingreso_min"] <= futuro) & (futuro < d["fin_uci"])).sum()
            obs_piso = ((d["fin_uci"] <= futuro) & (futuro < d["fin_piso"])).sum()
            P = expm(Q * h)
            ini = np.where(d["servicio_actual"] == "UCI", 0, 1)
            filas.append({"fecha": fecha, "h": h, "obs_uci": obs_uci, "obs_piso": obs_piso,
                          "semi_uci": p[f"P_UCI_{h}d"].sum(), "semi_piso": p[f"P_piso_{h}d"].sum(),
                          "markov_uci": P[ini, 0].sum(), "markov_piso": P[ini, 1].sum(),
                          "nadie_uci": (d["servicio_actual"] == "UCI").sum(),
                          "nadie_piso": (d["servicio_actual"] == "HOSPITALIZACION").sum()})
    r = pd.DataFrame(filas)
    res = []
    for h, g in r.groupby("h"):
        fila = {"horizonte (días)": h, "semanas": len(g), "censo UCI observado (media)": g.obs_uci.mean(),
                "censo piso observado (media)": g.obs_piso.mean()}
        for m, etiqueta in [("semi", "semi-Markov"), ("markov", "Markov exponencial"), ("nadie", "nadie se mueve")]:
            fila[f"error UCI {etiqueta}"] = (g[f"{m}_uci"] - g.obs_uci).abs().mean()
            fila[f"error piso {etiqueta}"] = (g[f"{m}_piso"] - g.obs_piso).abs().mean()
        res.append(fila)
    return pd.DataFrame(res).set_index("horizonte (días)").T


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    Path("modelos").mkdir(exist_ok=True)
    df = solo_completos(cargar())
    X = X_modelo(df)
    categorias = {c: list(X[c].cat.categories) for c in CATEGORICAS_MODELO}
    aseguradoras = sorted(df["aseguradora"].unique())
    tr = (df["fecha_ingreso_min"] < CORTE_TEMPORAL).values
    resumen = {}

    for obj, nombre in OBJETIVOS.items():
        y = df[obj].values
        print(f"\n{'=' * 90}\n{nombre.upper()}\n{'=' * 90}")
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA)
                     .split(X[tr], y[tr], df.loc[tr, "paciente_id"]))
        oof = {n: m04.oof(m, X[tr], y[tr], folds) for n, m in miembros(obj).items()}
        oof["Ensamble (promedio de los 3)"] = np.mean(list(oof.values()), axis=0)
        candidatos = dict(miembros(obj), **{"Ensamble (promedio de los 3)": ensamble(obj)})
        filas = []
        for n, m in candidatos.items():
            p = clone(m).fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
            auc, lo, hi = m04.auc_ic(y[~tr], p)
            filas.append({"modelo": n, "AUC_cv": roc_auc_score(y[tr], oof[n]), "Brier_cv": brier_score_loss(y[tr], oof[n]),
                          "AUC_prueba": auc, "IC95_prueba": f"{lo:.3f}-{hi:.3f}", "Brier_prueba": brier_score_loss(y[~tr], p)})
        tabla = pd.DataFrame(filas).sort_values("AUC_cv", ascending=False)
        print(tabla.round(4).to_string(index=False))
        elegido = tabla.iloc[0]["modelo"]
        print(f"\nElegido por AUC en CV: {elegido}")

        # Modelo final con toda la cohorte; puntos de corte desde predicciones fuera de muestra
        folds_todo = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(X, y, df["paciente_id"]))
        p_oof = m04.oof(candidatos[elegido], X, y, folds_todo)
        cortes = umbrales(y, p_oof)
        final = clone(candidatos[elegido]).fit(X, y)
        fila = tabla.set_index("modelo").loc[elegido]
        joblib.dump({"modelo": final, "nombre_modelo": elegido, "objetivo": nombre, "categorias": categorias,
                     "aseguradoras": aseguradoras, "umbrales": cortes, "tasa_base": float(y.mean()),
                     "auc_prueba": float(fila["AUC_prueba"]), "ic95_prueba": fila["IC95_prueba"]},
                    f"modelos/modelo_{obj}.joblib")
        print(f"Puntos de corte: riesgo ALTO si p >= {cortes['youden']:.3f} (Youden); "
              f"BAJO si p < {cortes['sens90']:.3f} (captura el 90 % de los casos)")
        resumen[obj] = {"objetivo": nombre, "modelo": elegido, "auc_prueba": round(float(fila["AUC_prueba"]), 3),
                        "ic95_prueba": fila["IC95_prueba"], "brier_prueba": round(float(fila["Brier_prueba"]), 3),
                        "umbral_alto": round(cortes["youden"], 3), "umbral_bajo": round(cortes["sens90"], 3)}

        texto = reglas_arbol(df, y, nombre)
        Path(f"modelos/reglas_{obj}.txt").write_text(texto, encoding="utf-8")
        print("\n" + texto)

    print(f"\n{'=' * 90}\nPROYECCIÓN DE OCUPACIÓN (semi-Markov)\n{'=' * 90}")
    modelo_tr = ocupacion.ajustar(df[tr].reset_index(drop=True))
    Q = m06.matriz_Q(m06.conteos_markov(df[tr]))
    # El censo incluye a todos los pacientes (también con datos incompletos): todos ocupan cama
    val = validar_ocupacion(cargar(), modelo_tr, Q)
    print("Validación retrospectiva semanal (feb-ago 2026): error absoluto medio en número de camas,")
    print("para los pacientes presentes ese día (no incluye ingresos nuevos):")
    print(val.round(2).to_string())
    val.to_csv("modelos/validacion_ocupacion.csv", sep=";", decimal=",", encoding="utf-8-sig")
    joblib.dump(ocupacion.ajustar(df), "modelos/modelo_ocupacion.joblib")
    resumen["ocupacion"] = {"modelo": "semi-Markov (Cox por causa + Cox de piso)",
                            "error_medio_camas_uci_7d": round(float(val.loc["error UCI semi-Markov", 7]), 2),
                            "error_medio_camas_piso_7d": round(float(val.loc["error piso semi-Markov", 7]), 2)}

    Path("modelos/resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nModelos finales guardados en modelos/")


if __name__ == "__main__":
    main()
