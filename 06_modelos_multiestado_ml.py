"""Comparación de familias de modelos: clásicos, multiestado (Markov), supervivencia y machine learning.

Mismas variables (conocidas al ingreso a UCI) y misma validación temporal para todos:
entrenamiento con ingresos < 2026-02-01, prueba con ingresos posteriores.

A. Modelo multiestado de Markov en tiempo continuo (estados UCI, HOSPITALIZACIÓN, EGRESO).
   La verosimilitud de un Markov homogéneo solo necesita, por paciente, el número de transiciones
   de cada tipo y el tiempo total en cada estado; ambos están en la base (secuencia_servicios,
   dias_uci, dias_hospitalizacion). Con covariables: regresión de Poisson por transición con
   offset log(tiempo en el estado) = intensidades proporcionales (equivalente al paquete msm de R).
B. Supervivencia: incidencia acumulada con riesgos competitivos (Aalen-Johansen) para el primer
   episodio de UCI (traslado a piso vs. egreso directo) y modelo de Cox para los días en UCI.
C. Clasificadores: logística clásica, logística con splines, árbol de decisión, random forest,
   gradient boosting, XGBoost, LightGBM y red neuronal; explicación SHAP con LightGBM.
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
import xgboost as xgb
from scipy.linalg import expm
from lifelines import CoxPHFitter, AalenJohansenFitter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from preparacion import (cargar, solo_completos, X_modelo, diseno, OBJETIVOS, CORTE_TEMPORAL, NUMERICAS_MODELO,
                         CATEGORICAS_MODELO, CONTINUAS)

SEMILLA = 42
BINARIAS = [c for c in NUMERICAS_MODELO if c not in CONTINUAS]
ESTADOS = ["UCI", "HOSPITALIZACION", "EGRESO"]
TRANSICIONES = [("UCI", "HOSPITALIZACION"), ("UCI", "EGRESO"), ("HOSPITALIZACION", "UCI"), ("HOSPITALIZACION", "EGRESO")]


# ----------------------------------------------------------------------------------------------
# A. Modelo multiestado de Markov
# ----------------------------------------------------------------------------------------------
def conteos_markov(df):
    """Transiciones por paciente y tiempo en cada estado."""
    filas = []
    for _, r in df.iterrows():
        estados = [s.strip() for s in r["secuencia_servicios"].split("→")] + ["EGRESO"]
        n = {f"{a}->{b}": 0 for a, b in TRANSICIONES}
        for a, b in zip(estados[:-1], estados[1:]):
            n[f"{a}->{b}"] += 1
        en_hosp = "HOSPITALIZACION" in estados
        filas.append({**n, "T_UCI": max(r["dias_uci"], 0.5),
                      "T_HOSPITALIZACION": max(r["dias_hospitalizacion"], 0.5) if en_hosp else 0.0})
    return pd.DataFrame(filas, index=df.index)


def matriz_Q(c):
    q = {f"{a}->{b}": c[f"{a}->{b}"].sum() / c[f"T_{a}"].sum() for a, b in TRANSICIONES}
    Q = np.zeros((3, 3))
    for (a, b) in TRANSICIONES:
        Q[ESTADOS.index(a), ESTADOS.index(b)] = q[f"{a}->{b}"]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def describir_markov(df, c):
    print("\n-- A1. Transiciones observadas (conteo) --")
    print(pd.Series({f"{a} -> {b}": c[f"{a}->{b}"].sum() for a, b in TRANSICIONES}).to_string())
    Q = matriz_Q(c)
    print("\n-- A2. Matriz de intensidades Q (transiciones por paciente-día) --")
    print(pd.DataFrame(Q, index=ESTADOS, columns=ESTADOS).round(4).to_string())
    print("\nEstancia media por visita: UCI = {:.1f} días | hospitalización = {:.1f} días".format(-1 / Q[0, 0], -1 / Q[1, 1]))
    P = -Q[:2, :] / Q[[0, 1], [0, 1]][:, None]
    np.fill_diagonal(P[:, :2], 0)
    print("Probabilidad del siguiente estado al salir de UCI: piso {:.2f}, egreso {:.2f} | al salir de piso: UCI {:.2f}, egreso {:.2f}"
          .format(P[0, 1], P[0, 2], P[1, 0], P[1, 2]))

    print("\n-- A3. Probabilidad de estar en cada estado t días después de entrar a UCI (Markov) vs. observado --")
    solo_un_episodio = df["secuencia_servicios"].isin(["UCI", "UCI → HOSPITALIZACION"])
    filas = []
    for t in [1, 3, 7, 14, 30]:
        p = expm(Q * t)[0]
        filas.append({"día": t, "P(UCI)": p[0], "P(piso)": p[1], "P(egreso)": p[2],
                      "observado: sigue en UCI": (df.loc[solo_un_episodio, "dias_uci"] > t).mean()})
    print(pd.DataFrame(filas).round(3).to_string(index=False))

    print("\n-- A4. Intensidades por grupo de SOFA inicial --")
    g = pd.cut(df["puntuacion_sofa_inicial"], [-1, 1, 4, 8, 30], labels=["0-1", "2-4", "5-8", ">8"])
    filas = []
    for nivel in ["0-1", "2-4", "5-8", ">8"]:
        Qg = matriz_Q(c[g == nivel])
        filas.append({"SOFA": nivel, "n": int((g == nivel).sum()), "UCI->piso": Qg[0, 1], "UCI->egreso": Qg[0, 2],
                      "piso->UCI": Qg[1, 0], "piso->egreso": Qg[1, 2], "días UCI/visita": -1 / Qg[0, 0],
                      "P(egreso directo | sale de UCI)": Qg[0, 2] / -Qg[0, 0]})
    print(pd.DataFrame(filas).round(3).to_string(index=False))


def ajustar_markov(X, c):
    """Una regresión de Poisson por transición con offset log(tiempo en riesgo)."""
    modelos, tablas = {}, {}
    for a, b in TRANSICIONES:
        en_riesgo = c[f"T_{a}"] > 0
        y = c.loc[en_riesgo, f"{a}->{b}"]
        Xa = sm.add_constant(X[en_riesgo], has_constant="add")
        m = sm.GLM(y, Xa, family=sm.families.Poisson(), offset=np.log(c.loc[en_riesgo, f"T_{a}"])).fit()
        modelos[(a, b)] = m
        ci = np.exp(m.conf_int())
        tablas[f"{a[:4]}->{b[:4]}"] = pd.DataFrame({"HR": np.exp(m.params), "inf": ci[0], "sup": ci[1], "p": m.pvalues})
    return modelos, tablas


def predecir_markov(modelos, X, procedencia_piso):
    Xc = sm.add_constant(X, has_constant="add")
    q = {k: m.predict(Xc, offset=np.zeros(len(X))) for k, m in modelos.items()}
    sal_uci = q[("UCI", "HOSPITALIZACION")] + q[("UCI", "EGRESO")]
    p_uci_egreso = q[("UCI", "EGRESO")] / sal_uci
    p_piso_uci = q[("HOSPITALIZACION", "UCI")] / (q[("HOSPITALIZACION", "UCI")] + q[("HOSPITALIZACION", "EGRESO")])
    # Cadena absorbente: probabilidad de que la última salida sea desde UCI
    p_final_uci = p_uci_egreso / (1 - (1 - p_uci_egreso) * p_piso_uci)
    p_final_uci = np.where(procedencia_piso == 1, p_piso_uci * p_final_uci, p_final_uci)
    return {"uci_prolongada": np.exp(-sal_uci * 7), "egreso_directo_uci": p_final_uci}


# ----------------------------------------------------------------------------------------------
# B. Supervivencia
# ----------------------------------------------------------------------------------------------
def riesgos_competitivos(df):
    un_episodio = df["secuencia_servicios"].isin(["UCI", "UCI → HOSPITALIZACION", "HOSPITALIZACION → UCI",
                                                  "HOSPITALIZACION → UCI → HOSPITALIZACION"])
    d = df[un_episodio]
    t = d["dias_uci"].clip(lower=0.5).values
    evento = np.where(d["servicio_final"] == "UCI", 2, 1)  # 1 = traslado a piso, 2 = egreso directo
    grupos = pd.cut(d["puntuacion_sofa_inicial"], [-1, 1, 4, 8, 30], labels=["0-1", "2-4", "5-8", ">8"])
    print(f"\n-- B1. Incidencia acumulada (Aalen-Johansen), un solo episodio de UCI (n={len(d)}) --")
    filas = []
    for nombre, m in [("todos", np.ones(len(d), bool))] + [(f"SOFA {g}", (grupos == g).values) for g in ["0-1", "2-4", "5-8", ">8"]]:
        for causa, etiqueta in [(1, "traslado a piso"), (2, "egreso directo")]:
            aj = AalenJohansenFitter(calculate_variance=False, seed=SEMILLA).fit(t[m], evento[m], event_of_interest=causa)
            ci = aj.cumulative_density_
            filas.append({"grupo": nombre, "destino": etiqueta,
                          **{f"día {k}": float(ci[ci.index <= k].iloc[-1, 0]) for k in [1, 3, 7, 14, 30]}})
    print(pd.DataFrame(filas).round(3).to_string(index=False))


def ajustar_cox(X, dias):
    d = X.copy()
    d["T"] = dias.clip(lower=0.5).values
    d["E"] = 1  # todos salen de UCI (no hay censura en una base de egresos)
    cph = CoxPHFitter(penalizer=0.01).fit(d, "T", "E")
    return cph


# ----------------------------------------------------------------------------------------------
# C. Clasificadores
# ----------------------------------------------------------------------------------------------
def preprocesador(splines=False, escalar=True):
    cont = [SimpleImputer(strategy="median")]
    cont += [SplineTransformer(n_knots=4, degree=3)] if splines else []
    cont += [StandardScaler()] if escalar else []
    return ColumnTransformer([
        ("cont", make_pipeline(*cont), CONTINUAS),
        ("bin", "passthrough", BINARIAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAS_MODELO)])


def clasificadores(spw):
    return {
        "Logística clásica": make_pipeline(preprocesador(), LogisticRegression(C=1.0, max_iter=5000)),
        "Logística + splines": make_pipeline(preprocesador(True), LogisticRegression(C=0.3, max_iter=5000)),
        "Árbol de decisión": make_pipeline(preprocesador(), DecisionTreeClassifier(random_state=SEMILLA)),
        "Random forest": make_pipeline(preprocesador(), RandomForestClassifier(
            500, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=SEMILLA)),
        "Gradient boosting (sklearn)": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0,
            categorical_features="from_dtype", random_state=SEMILLA),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                                     enable_categorical=True, tree_method="hist", random_state=SEMILLA),
        "LightGBM": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                       random_state=SEMILLA, verbose=-1),
        "Red neuronal (MLP)": make_pipeline(preprocesador(), MLPClassifier(
            (32, 16), alpha=1e-2, max_iter=800, early_stopping=True, random_state=SEMILLA)),
    }


def profundidad_arbol(Xtr, ytr, folds):
    mejor = (0, None)
    for prof in [2, 3, 4, 5, 6, 8]:
        for hoja in [20, 50, 100]:
            m = make_pipeline(preprocesador(), DecisionTreeClassifier(max_depth=prof, min_samples_leaf=hoja,
                                                                       random_state=SEMILLA))
            p = np.zeros(len(ytr))
            for tr, te in folds:
                p[te] = clone(m).fit(Xtr.iloc[tr], ytr[tr]).predict_proba(Xtr.iloc[te])[:, 1]
            auc = roc_auc_score(ytr, p)
            if auc > mejor[0]:
                mejor = (auc, (prof, hoja))
    return mejor[1]


def auc_ic(y, p, n=1000):
    rng = np.random.default_rng(SEMILLA)
    v = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return f"{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    df = solo_completos(cargar())
    tr = (df["fecha_ingreso_min"] < CORTE_TEMPORAL).values
    print(f"Entrenamiento {tr.sum()} | prueba {(~tr).sum()} ingresos")
    resultados = {obj: [] for obj in OBJETIVOS}

    def registrar(obj, familia, modelo, p):
        y = df.loc[~tr, obj].values
        p = np.asarray(p, dtype=float)
        resultados[obj].append({"familia": familia, "modelo": modelo, "AUC": roc_auc_score(y, p),
                                "IC95": auc_ic(y, p), "Brier": brier_score_loss(y, np.clip(p, 0, 1))})

    # ---------------- A. Markov multiestado ----------------
    print("\n" + "=" * 100 + "\nA. MODELO MULTIESTADO DE MARKOV (UCI <-> HOSPITALIZACIÓN -> EGRESO)\n" + "=" * 100)
    c = conteos_markov(df)
    describir_markov(df, c)
    Xtr_d, ref = diseno(df[tr])
    Xte_d, _ = diseno(df[~tr], ref)
    modelos_mk, tablas = ajustar_markov(Xtr_d, c[tr])
    print("\n-- A5. Efecto de cada variable sobre cada transición (HR = razón de intensidades; * p<0,05) --")
    hr = pd.DataFrame({k: t["HR"].round(2).astype(str) + np.where(t["p"] < 0.05, "*", "") for k, t in tablas.items()})
    print(hr.drop(index="const").to_string())
    pd.concat(tablas, axis=1).to_csv("markov_hazard_ratios.csv", sep=";", decimal=",", encoding="utf-8-sig")
    pred_mk = predecir_markov(modelos_mk, Xte_d, df.loc[~tr, "procedencia_piso"].values)
    for obj in OBJETIVOS:
        registrar(obj, "Multiestado", "Markov tiempo continuo + covariables", pred_mk[obj])
    # Markov sin covariables: solo distingue el estado de inicio (procedencia)
    Q = matriz_Q(c[tr])
    p_ue, p_hu = Q[0, 2] / -Q[0, 0], Q[1, 0] / -Q[1, 1]
    p_final = p_ue / (1 - (1 - p_ue) * p_hu)
    registrar("egreso_directo_uci", "Multiestado", "Markov sin covariables (solo procedencia)",
              np.where(df.loc[~tr, "procedencia_piso"].values == 1, p_hu * p_final, p_final))

    # ---------------- B. Supervivencia ----------------
    print("\n" + "=" * 100 + "\nB. SUPERVIVENCIA: RIESGOS COMPETITIVOS Y COX\n" + "=" * 100)
    riesgos_competitivos(df)
    cph = ajustar_cox(Xtr_d, df.loc[tr, "dias_uci"])
    print("\n-- B2. Cox para días en UCI (HR > 1 = sale antes de UCI; * p<0,05) --")
    s = cph.summary
    print((s["exp(coef)"].round(2).astype(str) + np.where(s["p"] < 0.05, "*", "")).to_string())
    print(f"Índice C en prueba: {cph.score(Xte_d.assign(T=df.loc[~tr, 'dias_uci'].clip(lower=0.5).values, E=1), scoring_method='concordance_index'):.3f}")
    p_cox = cph.predict_survival_function(Xte_d, times=[7]).iloc[0].values
    registrar("uci_prolongada", "Supervivencia", "Cox (P de seguir en UCI al día 7)", p_cox)

    # ---------------- C. Clasificadores ----------------
    print("\n" + "=" * 100 + "\nC. CLASIFICADORES: CLÁSICOS Y MACHINE LEARNING\n" + "=" * 100)
    X = X_modelo(df)
    Xtr, Xte = X[tr], X[~tr]
    for obj, nombre in OBJETIVOS.items():
        y = df[obj].values
        ytr = y[tr]
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA)
                     .split(Xtr, ytr, df.loc[tr, "paciente_id"]))
        prof, hoja = profundidad_arbol(Xtr, ytr, folds)
        print(f"\n### {nombre}: árbol de decisión elegido por CV: profundidad {prof}, mínimo {hoja} por hoja")
        for etiqueta, modelo in clasificadores(1).items():
            if etiqueta == "Árbol de decisión":
                modelo.set_params(decisiontreeclassifier__max_depth=prof, decisiontreeclassifier__min_samples_leaf=hoja)
            modelo.fit(Xtr, ytr)
            familia = "Clásico" if etiqueta.startswith(("Logística", "Árbol")) else "Machine learning"
            registrar(obj, familia, etiqueta, modelo.predict_proba(Xte)[:, 1])
            if etiqueta == "LightGBM":
                contrib = modelo.predict(Xte, pred_contrib=True)[:, :-1]
                shap_imp = pd.Series(np.abs(contrib).mean(axis=0), index=Xte.columns).sort_values(ascending=False)
                print("\n-- SHAP (LightGBM): impacto medio |SHAP| por variable, datos de prueba --")
                print(shap_imp[shap_imp > 0].round(4).to_string())

        # Árbol interpretable de profundidad 3 (reglas para uso clínico)
        arbol = make_pipeline(preprocesador(escalar=False), DecisionTreeClassifier(  # reglas en unidades realesmax_depth=3, min_samples_leaf=50,
                                                                      random_state=SEMILLA)).fit(Xtr, ytr)
        registrar(obj, "Clásico", "Árbol interpretable (profundidad 3)", arbol.predict_proba(Xte)[:, 1])
        nombres = [n.split("__")[-1] for n in arbol[0].get_feature_names_out()]
        print("\n-- Reglas del árbol interpretable (profundidad 3); 'class' = desenlace más probable --")
        print(export_text(arbol[-1], feature_names=nombres, show_weights=True, decimals=1))

    # ---------------- Comparación final ----------------
    print("\n" + "=" * 100 + "\nCOMPARACIÓN FINAL EN PRUEBA (ingresos feb-ago 2026)\n" + "=" * 100)
    tablas = []
    for obj, nombre in OBJETIVOS.items():
        t = pd.DataFrame(resultados[obj]).sort_values("AUC", ascending=False)
        print(f"\n### {nombre}")
        print(t.round(3).to_string(index=False))
        tablas.append(t.assign(objetivo=obj))
    pd.concat(tablas).to_csv("comparacion_familias_modelos.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
