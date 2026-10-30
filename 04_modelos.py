"""Construcción, comparación y validación de los modelos finales.

Para cada objetivo (uci_prolongada, egreso_directo_uci):
  1. Entrenamiento con ingresos < 2026-02-01; prueba con ingresos posteriores (validación temporal).
  2. Selección de algoritmo e hiperparámetros con CV de 5 particiones agrupada por paciente.
  3. Calibración de probabilidades (isotónica vs. sigmoide, elegida por Brier en CV).
  4. Métricas en prueba: AUC con IC 95 % bootstrap, AUC-PR, Brier, calibración, punto de corte.
  5. Modelo final reentrenado con toda la cohorte y guardado en modelos/.
"""
import sys
import json
import itertools
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from preparacion import (cargar, solo_completos, X_modelo, OBJETIVOS, CORTE_TEMPORAL, NUMERICAS_MODELO,
                         CATEGORICAS_MODELO)

CONTINUAS = ["edad", "puntuacion_sofa_inicial", "barthel_inicial"]
BINARIAS = [c for c in NUMERICAS_MODELO if c not in CONTINUAS]
SEMILLA = 42


def preprocesador(splines):
    continuas = [SimpleImputer(strategy="median")]
    continuas += [SplineTransformer(n_knots=4, degree=3)] if splines else []
    continuas += [StandardScaler()]
    return ColumnTransformer([
        ("cont", make_pipeline(*continuas), CONTINUAS),
        ("bin", "passthrough", BINARIAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAS_MODELO),
    ])


def candidatos():
    for C in [0.05, 0.3, 1.0]:
        yield f"Logística+splines C={C}", make_pipeline(
            preprocesador(True), LogisticRegression(C=C, max_iter=5000))
    for hoja, feats in itertools.product([10, 30], ["sqrt", 0.5]):
        yield f"RandomForest hoja={hoja} feats={feats}", make_pipeline(
            preprocesador(False), RandomForestClassifier(n_estimators=500, min_samples_leaf=hoja, max_features=feats,
                                                         n_jobs=-1, random_state=SEMILLA))
    for lr, hojas, minimo in itertools.product([0.03, 0.06], [8, 16], [40, 80]):
        yield f"GradientBoosting lr={lr} hojas={hojas} min={minimo}", HistGradientBoostingClassifier(
            max_iter=int(12 / lr), learning_rate=lr, max_leaf_nodes=hojas, min_samples_leaf=minimo,
            l2_regularization=1.0, categorical_features="from_dtype", random_state=SEMILLA)


def oof(modelo, X, y, folds):
    pred = np.zeros(len(y))
    for tr, te in folds:
        pred[te] = clone(modelo).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return pred


def auc_ic(y, p, n=1000):
    rng = np.random.default_rng(SEMILLA)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def pendiente_calibracion(y, p):
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(logit, y)
    return lr.coef_[0][0], lr.intercept_[0]


def umbral_youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]


def metricas_umbral(y, p, u):
    pred = p >= u
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, tn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    return {"sensibilidad": tp / (tp + fn), "especificidad": tn / (tn + fp),
            "VPP": tp / max(tp + fp, 1), "VPN": tn / max(tn + fn, 1), "% marcados": pred.mean()}


def odds_ratios(df, y):
    """OR interpretables de una logística simple (sin splines), IC 95 % por bootstrap."""
    escalas = {"edad": 10, "puntuacion_sofa_inicial": 1, "barthel_inicial": 10}
    X = df[NUMERICAS_MODELO].copy()
    X = X.fillna(X.median())
    for c in CATEGORICAS_MODELO:
        ref = df[c].mode()[0]
        X = pd.concat([X, pd.get_dummies(df[c], prefix=c, dtype=float).drop(columns=f"{c}_{ref}")], axis=1)
    X = X.loc[:, X.sum() >= 20]
    for c, k in escalas.items():
        X[c] = X[c] / k
    ajustar = lambda Xb, yb: LogisticRegression(C=1e4, max_iter=10000).fit(Xb, yb).coef_[0]
    coef = ajustar(X, y)
    rng = np.random.default_rng(SEMILLA)
    boots = np.array([ajustar(X.iloc[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))])
    t = pd.DataFrame({"OR": np.exp(coef), "IC95_inf": np.exp(np.percentile(boots, 2.5, axis=0)),
                      "IC95_sup": np.exp(np.percentile(boots, 97.5, axis=0))}, index=X.columns)
    t.index = [f"{c} (por {escalas[c]})" if c in escalas else c for c in t.index]
    t["significativo"] = np.where((t.IC95_inf > 1) | (t.IC95_sup < 1), "sí", "")
    return t.sort_values("OR", ascending=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    df = solo_completos(cargar())
    X = X_modelo(df)
    categorias = {c: list(X[c].cat.categories) for c in CATEGORICAS_MODELO}
    entrena = (df["fecha_ingreso_min"] < CORTE_TEMPORAL).values
    Path("modelos").mkdir(exist_ok=True)
    print(f"Entrenamiento: {entrena.sum()} ingresos (< {CORTE_TEMPORAL}) | Prueba: {(~entrena).sum()} ingresos")
    resumen = {}

    for obj, nombre in OBJETIVOS.items():
        y = df[obj].values
        Xtr, ytr, Xte, yte = X[entrena], y[entrena], X[~entrena], y[~entrena]
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA)
                     .split(Xtr, ytr, df.loc[entrena, "paciente_id"]))
        print(f"\n{'=' * 90}\n{nombre.upper()}  [{obj}]  prevalencia entrenamiento {ytr.mean():.1%}, "
              f"prueba {yte.mean():.1%}\n{'=' * 90}")

        # 1. Comparación de algoritmos
        filas = []
        for etiqueta, modelo in candidatos():
            p = oof(modelo, Xtr, ytr, folds)
            filas.append({"modelo": etiqueta, "AUC_cv": roc_auc_score(ytr, p),
                          "Brier_cv": brier_score_loss(ytr, p), "_obj": modelo})
        comp = pd.DataFrame(filas).sort_values("AUC_cv", ascending=False)
        print("\n-- Comparación de algoritmos (CV agrupada por paciente, datos de entrenamiento) --")
        print(comp.drop(columns="_obj").round(4).to_string(index=False))
        mejor_por_familia = comp.groupby(comp.modelo.str.split().str[0]).head(1)

        # 2. Calibración del mejor de cada familia y evaluación en prueba
        print("\n-- Mejor de cada familia, calibrado, evaluado en PRUEBA (ingresos feb-ago 2026) --")
        evals = []
        for _, fila in mejor_por_familia.iterrows():
            mejor_cal = None
            for metodo in ["sigmoid", "isotonic"]:
                cal = CalibratedClassifierCV(clone(fila["_obj"]), method=metodo, cv=folds)
                p_cv = oof(CalibratedClassifierCV(clone(fila["_obj"]), method=metodo, cv=3), Xtr, ytr, folds)
                b = brier_score_loss(ytr, p_cv)
                if mejor_cal is None or b < mejor_cal[1]:
                    mejor_cal = (metodo, b, cal, p_cv)
            metodo, _, cal, p_cv = mejor_cal
            cal.fit(Xtr, ytr)
            p = cal.predict_proba(Xte)[:, 1]
            auc, lo, hi = auc_ic(yte, p)
            pend, interc = pendiente_calibracion(yte, p)
            evals.append({"modelo": fila["modelo"], "calibración": metodo, "AUC": auc, "IC95": f"{lo:.3f}-{hi:.3f}",
                          "AUC_PR": average_precision_score(yte, p), "Brier": brier_score_loss(yte, p),
                          "Brier_ref": brier_score_loss(yte, np.full(len(yte), ytr.mean())),
                          "pred_media": p.mean(), "observado": yte.mean(), "pendiente_cal": pend,
                          "_cal": cal, "_p": p, "_p_cv": p_cv, "_base": fila["_obj"]})
        ev = pd.DataFrame(evals).sort_values("AUC", ascending=False)
        print(ev.drop(columns=[c for c in ev if c.startswith("_")]).round(3).to_string(index=False))

        # El elegido: mayor AUC en CV (no en prueba, para no sobreajustar a la prueba)
        elegido = ev.set_index("modelo").loc[mejor_por_familia.iloc[0]["modelo"]]
        p = elegido["_p"]
        print(f"\nModelo elegido (mejor AUC en CV): {elegido.name} con calibración {elegido['calibración']}")

        print("\n-- Calibración por deciles de riesgo (prueba) --")
        dec = pd.DataFrame({"p": p, "y": yte})
        dec["decil"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=range(1, 11))
        print(dec.groupby("decil", observed=True).agg(n=("y", "size"), riesgo_predicho=("p", "mean"),
                                                      tasa_observada=("y", "mean")).round(3).T.to_string())

        u = umbral_youden(ytr, elegido["_p_cv"])  # punto de corte elegido en entrenamiento
        print(f"\n-- Puntos de corte (elegidos en entrenamiento, medidos en prueba) --")
        cortes = {"Youden": u}
        for sens in [0.80, 0.90]:
            fpr, tpr, thr = roc_curve(ytr, elegido["_p_cv"])
            cortes[f"sensibilidad {sens:.0%}"] = thr[np.argmax(tpr >= sens)]
        print(pd.DataFrame({k: {"umbral": v, **metricas_umbral(yte, p, v)} for k, v in cortes.items()}).T
              .round(3).to_string())

        print("\n-- Odds ratios (logística simple, toda la cohorte; IC 95 % bootstrap) --")
        ors = odds_ratios(df, y)
        print(ors.round(2).to_string())
        ors.to_csv(f"modelos/odds_ratios_{obj}.csv", sep=";", decimal=",", encoding="utf-8-sig")

        # Modelo final: mismo algoritmo y calibración, reentrenado con toda la cohorte
        folds_todo = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA)
                          .split(X, y, df["paciente_id"]))
        final = CalibratedClassifierCV(clone(elegido["_base"]), method=elegido["calibración"], cv=folds_todo).fit(X, y)
        joblib.dump({"modelo": final, "categorias": categorias, "umbral_youden": float(u),
                     "objetivo": nombre, "variables": list(X.columns)}, f"modelos/modelo_{obj}.joblib")
        resumen[obj] = {"objetivo": nombre, "algoritmo": elegido.name, "calibracion": elegido["calibración"],
                        "auc_prueba": round(float(elegido["AUC"]), 3), "ic95": elegido["IC95"],
                        "brier_prueba": round(float(elegido["Brier"]), 3), "umbral_youden": round(float(u), 3),
                        "n_entrenamiento": int(entrena.sum()), "n_prueba": int((~entrena).sum())}

    Path("modelos/resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nModelos guardados en modelos/")


if __name__ == "__main__":
    main()
