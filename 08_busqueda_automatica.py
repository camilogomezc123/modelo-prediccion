"""Búsqueda automática de modelos para "salida temprana de UCI" desde el primer día.

Uso:  python 08_busqueda_automatica.py [horas]      (por defecto 6 horas)

Recorre objetivos x conjuntos de variables x familias de algoritmos. En cada combinación, Optuna
busca hiperparámetros maximizando el AUC de CV de 5 particiones agrupada por paciente SOLO en el
entrenamiento (ingresos < 2026-02-01). La prueba temporal (feb-ago 2026) se informa pero nunca se
usa para elegir. Al final: ensambles (promedio y apilado) con lo mejor de cada familia, elección
del mejor por CV, calibración, reentrenamiento con toda la cohorte y guardado en modelos/auto_*.

Salidas: modelos/auto_ranking.csv (se actualiza tras cada estudio), modelos/auto_log.txt,
modelos/auto_<objetivo>.joblib, modelos/auto_reporte.md.
"""
import sys
import json
import time
import traceback
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler, TargetEncoder

from preparacion import cargar, solo_completos, CORTE_TEMPORAL, CONTINUAS
from modelos_auto import (OBJETIVOS_AUTO, CONJUNTOS, columnas, historia_cruda, agregar_extras,
                          agregar_objetivos, Ensamble)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEMILLA = 42
DIR = Path("modelos")
LOG = DIR / "auto_log.txt"
RANKING = DIR / "auto_ranking.csv"
FAMILIAS = ["logistica", "random_forest", "extra_trees", "hist_gb", "xgboost", "lightgbm", "catboost", "red_neuronal"]


def log(msg):
    linea = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def preprocesador(conjunto, splines=False, escalar=True):
    c = CONJUNTOS[conjunto]
    cont = [x for x in c["num"] if x in CONTINUAS]
    resto = [x for x in c["num"] if x not in CONTINUAS]
    pasos_cont = [SimpleImputer(strategy="median")]
    pasos_cont += [SplineTransformer(n_knots=4, degree=3)] if splines else []
    pasos_cont += [StandardScaler()] if escalar else []
    partes = [("cont", make_pipeline(*pasos_cont), cont),
              ("resto", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) if escalar
               else SimpleImputer(strategy="median"), resto),
              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), c["cat"])]
    if c["te"]:
        partes.append(("te", TargetEncoder(target_type="binary", random_state=SEMILLA), c["te"]))
    return ColumnTransformer(partes)


def construir(familia, p, conjunto):
    """Estimador (pipeline) a partir de los hiperparámetros p."""
    if familia == "logistica":
        return make_pipeline(preprocesador(conjunto, splines=p["splines"]),
                             LogisticRegression(C=p["C"], penalty=p["penalty"], solver="liblinear", max_iter=5000))
    if familia in ("random_forest", "extra_trees"):
        cls = RandomForestClassifier if familia == "random_forest" else ExtraTreesClassifier
        return make_pipeline(preprocesador(conjunto, escalar=False), cls(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"], class_weight=p["class_weight"], n_jobs=-1, random_state=SEMILLA))
    if familia == "hist_gb":
        return make_pipeline(preprocesador(conjunto, escalar=False), HistGradientBoostingClassifier(
            learning_rate=p["lr"], max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2"], random_state=SEMILLA))
    if familia == "xgboost":
        return make_pipeline(preprocesador(conjunto, escalar=False), xgb.XGBClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["lr"], max_depth=p["max_depth"], subsample=p["subsample"],
            colsample_bytree=p["colsample"], min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
            tree_method="hist", n_jobs=-1, random_state=SEMILLA, verbosity=0))
    if familia == "lightgbm":
        return make_pipeline(preprocesador(conjunto, escalar=False), lgb.LGBMClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["lr"], num_leaves=p["num_leaves"],
            min_child_samples=p["min_child_samples"], subsample=p["subsample"], subsample_freq=1,
            colsample_bytree=p["colsample"], reg_lambda=p["reg_lambda"], random_state=SEMILLA, verbose=-1, n_jobs=-1))
    if familia == "catboost":
        return make_pipeline(preprocesador(conjunto, escalar=False), CatBoostClassifier(
            iterations=p["iterations"], depth=p["depth"], learning_rate=p["lr"], l2_leaf_reg=p["l2"],
            random_seed=SEMILLA, verbose=0, thread_count=-1, allow_writing_files=False))
    if familia == "red_neuronal":
        capas = (p["n1"],) if p["capas"] == 1 else (p["n1"], p["n2"])
        return make_pipeline(preprocesador(conjunto), MLPClassifier(
            hidden_layer_sizes=capas, alpha=p["alpha"], learning_rate_init=p["lr"], max_iter=1000,
            early_stopping=True, random_state=SEMILLA))
    raise ValueError(familia)


def sugerir(trial, familia):
    s = trial.suggest_float
    i = trial.suggest_int
    c = trial.suggest_categorical
    if familia == "logistica":
        return {"C": s("C", 1e-3, 10, log=True), "penalty": c("penalty", ["l1", "l2"]), "splines": c("splines", [False, True])}
    if familia in ("random_forest", "extra_trees"):
        return {"n_estimators": i("n_estimators", 200, 800, step=100), "max_depth": c("max_depth", [None, 6, 10, 16]),
                "min_samples_leaf": i("min_samples_leaf", 1, 60, log=True),
                "max_features": c("max_features", ["sqrt", 0.3, 0.5, 0.8]),
                "class_weight": c("class_weight", [None, "balanced"])}
    if familia == "hist_gb":
        return {"lr": s("lr", 0.01, 0.2, log=True), "max_iter": i("max_iter", 100, 800, step=50),
                "max_leaf_nodes": i("max_leaf_nodes", 4, 64, log=True), "min_samples_leaf": i("min_samples_leaf", 10, 200, log=True),
                "l2": s("l2", 1e-3, 10, log=True)}
    if familia == "xgboost":
        return {"n_estimators": i("n_estimators", 100, 800, step=50), "lr": s("lr", 0.01, 0.2, log=True),
                "max_depth": i("max_depth", 2, 8), "subsample": s("subsample", 0.5, 1.0),
                "colsample": s("colsample", 0.4, 1.0), "min_child_weight": s("min_child_weight", 1, 30, log=True),
                "reg_lambda": s("reg_lambda", 1e-3, 30, log=True)}
    if familia == "lightgbm":
        return {"n_estimators": i("n_estimators", 100, 800, step=50), "lr": s("lr", 0.01, 0.2, log=True),
                "num_leaves": i("num_leaves", 4, 64, log=True), "min_child_samples": i("min_child_samples", 10, 200, log=True),
                "subsample": s("subsample", 0.5, 1.0), "colsample": s("colsample", 0.4, 1.0),
                "reg_lambda": s("reg_lambda", 1e-3, 30, log=True)}
    if familia == "catboost":
        return {"iterations": i("iterations", 200, 1000, step=100), "depth": i("depth", 3, 8),
                "lr": s("lr", 0.01, 0.2, log=True), "l2": s("l2", 1, 30, log=True)}
    if familia == "red_neuronal":
        return {"capas": c("capas", [1, 2]), "n1": i("n1", 8, 128, log=True), "n2": i("n2", 4, 64, log=True),
                "alpha": s("alpha", 1e-5, 1e-1, log=True), "lr": s("lr", 1e-4, 1e-2, log=True)}
    raise ValueError(familia)


def oof(modelo, X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = clone(modelo).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


def auc_ic(y, p, n=500):
    rng = np.random.default_rng(SEMILLA)
    v = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return f"{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}"


def umbrales(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return {"youden": float(thr[np.argmax(tpr - fpr)]), "sens90": float(thr[np.argmax(tpr >= 0.90)])}


def guardar_ranking(filas):
    pd.DataFrame(filas).drop(columns=["_oof", "_test"], errors="ignore") \
        .sort_values(["objetivo", "auc_cv"], ascending=[True, False]).to_csv(RANKING, sep=";", decimal=",", index=False, encoding="utf-8-sig")


def main():
    horas = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    fin = time.time() + horas * 3600
    DIR.mkdir(exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    log(f"=== Inicio de la búsqueda automática ({horas} h) ===")

    df = agregar_objetivos(agregar_extras(solo_completos(cargar()), historia_cruda()))
    tr = (df["fecha_ingreso_min"] < CORTE_TEMPORAL).values
    grupos = df.loc[tr, "paciente_id"].values
    log(f"Cohorte con datos completos: {len(df)} (entrenamiento {tr.sum()}, prueba {(~tr).sum()})")

    combinaciones = [(o, c, f) for o in OBJETIVOS_AUTO for c in CONJUNTOS for f in FAMILIAS]
    reserva_final = 0.12 * horas * 3600  # tiempo reservado para ensambles y modelos finales
    filas = []

    for k, (obj, conj, fam) in enumerate(combinaciones):
        restante = fin - reserva_final - time.time()
        if restante <= 60:
            log("Se agotó el tiempo de búsqueda; paso a ensambles y modelos finales.")
            break
        presupuesto = max(60, restante / (len(combinaciones) - k))
        y = df[obj].values
        X = df[columnas(conj)]
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(X[tr], y[tr], grupos))
        try:
            def objetivo_optuna(trial):
                p = sugerir(trial, fam)
                return roc_auc_score(y[tr], oof(construir(fam, p, conj), X[tr], y[tr], folds))

            estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMILLA))
            estudio.optimize(objetivo_optuna, timeout=presupuesto, n_trials=200, catch=(Exception,))
            if not estudio.best_trials:
                raise RuntimeError("ningún intento terminó")
            p = estudio.best_params
            modelo = construir(fam, p, conj)
            p_oof = oof(modelo, X[tr], y[tr], folds)
            p_test = clone(modelo).fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
            fila = {"objetivo": obj, "conjunto": conj, "familia": fam, "auc_cv": roc_auc_score(y[tr], p_oof),
                    "auc_prueba": roc_auc_score(y[~tr], p_test), "ic95_prueba": auc_ic(y[~tr], p_test),
                    "ap_prueba": average_precision_score(y[~tr], p_test), "brier_prueba": brier_score_loss(y[~tr], p_test),
                    "intentos": len(estudio.trials), "parametros": json.dumps(p), "_oof": p_oof, "_test": p_test}
            filas.append(fila)
            log(f"[{k + 1}/{len(combinaciones)}] {obj} | {conj} | {fam}: AUC CV {fila['auc_cv']:.4f} | "
                f"prueba {fila['auc_prueba']:.4f} ({fila['ic95_prueba']}) | {fila['intentos']} intentos")
            guardar_ranking(filas)
        except Exception:
            log(f"[{k + 1}/{len(combinaciones)}] ERROR en {obj} | {conj} | {fam}:\n{traceback.format_exc()}")

    # ---------------- Ensambles y modelo final por objetivo ----------------
    reporte = ["# Búsqueda automática: salida temprana de UCI desde el primer día", "",
               f"Generado {time.strftime('%Y-%m-%d %H:%M')}. Elección por AUC en CV (entrenamiento < {CORTE_TEMPORAL}); "
               "la prueba temporal (feb-ago 2026) solo se informa.", ""]
    resumen = {}
    for obj, nombre in OBJETIVOS_AUTO.items():
        try:
            f_obj = [f for f in filas if f["objetivo"] == obj]
            if not f_obj:
                continue
            y = df[obj].values
            # Mejor conjunto de variables por CV y, dentro de él, lo mejor de cada familia
            mejor_conj = max(f_obj, key=lambda f: f["auc_cv"])["conjunto"]
            top = sorted([f for f in f_obj if f["conjunto"] == mejor_conj], key=lambda f: -f["auc_cv"])[:5]
            P_oof = np.column_stack([f["_oof"] for f in top])
            P_test = np.column_stack([f["_test"] for f in top])
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA)
                         .split(P_oof, y[tr], df.loc[tr, "paciente_id"].values))
            ap_oof = oof(LogisticRegression(C=1.0), pd.DataFrame(P_oof), y[tr], folds)
            apilador = LogisticRegression(C=1.0).fit(P_oof, y[tr])
            for etiqueta, pcv, pte in [("ensamble_promedio", P_oof.mean(1), P_test.mean(1)),
                                       ("ensamble_apilado", ap_oof, apilador.predict_proba(P_test)[:, 1])]:
                filas.append({"objetivo": obj, "conjunto": mejor_conj, "familia": etiqueta,
                              "auc_cv": roc_auc_score(y[tr], pcv), "auc_prueba": roc_auc_score(y[~tr], pte),
                              "ic95_prueba": auc_ic(y[~tr], pte), "ap_prueba": average_precision_score(y[~tr], pte),
                              "brier_prueba": brier_score_loss(y[~tr], pte), "intentos": 0,
                              "parametros": json.dumps([f["familia"] for f in top]), "_oof": pcv, "_test": pte})
            guardar_ranking(filas)

            elegido = max([f for f in filas if f["objetivo"] == obj], key=lambda f: f["auc_cv"])
            log(f"{obj}: elegido {elegido['familia']} ({elegido['conjunto']}) AUC CV {elegido['auc_cv']:.4f}, "
                f"prueba {elegido['auc_prueba']:.4f}")

            # Reentrenar con toda la cohorte, calibrado (isotónica, CV interna)
            X_todo = df[columnas(elegido["conjunto"])]
            def calibrado(f):
                return CalibratedClassifierCV(construir(f["familia"], json.loads(f["parametros"]), f["conjunto"]),
                                              method="isotonic", cv=5)
            if elegido["familia"].startswith("ensamble"):
                bases = [calibrado(f).fit(X_todo, y) for f in top]
                final = Ensamble(bases, apilador if elegido["familia"] == "ensamble_apilado" else None)
            else:
                final = calibrado(elegido).fit(X_todo, y)

            # Puntos de corte desde las predicciones fuera de muestra del entrenamiento
            cortes = umbrales(y[tr], elegido["_oof"])
            # Importancia por permutación (modelo individual de mayor AUC CV ajustado en entrenamiento)
            ref = top[0]
            m_ref = construir(ref["familia"], json.loads(ref["parametros"]), ref["conjunto"]).fit(X_todo[tr], y[tr])
            imp = permutation_importance(m_ref, X_todo[~tr], y[~tr], scoring="roc_auc", n_repeats=10, random_state=SEMILLA)
            importancia = pd.Series(imp.importances_mean, index=X_todo.columns).sort_values(ascending=False)

            joblib.dump({"modelo": final, "columnas": columnas(elegido["conjunto"]), "conjunto": elegido["conjunto"],
                         "familia": elegido["familia"], "objetivo": nombre, "umbrales": cortes,
                         "tasa_base": float(y.mean()), "auc_cv": float(elegido["auc_cv"]),
                         "auc_prueba": float(elegido["auc_prueba"]), "ic95_prueba": elegido["ic95_prueba"],
                         "aseguradoras": sorted(df["aseguradora"].unique())}, DIR / f"auto_{obj}.joblib")
            resumen[obj] = {k: elegido[k] for k in ["familia", "conjunto", "auc_cv", "auc_prueba", "ic95_prueba", "brier_prueba"]}

            rk = pd.DataFrame([f for f in filas if f["objetivo"] == obj]).drop(columns=["_oof", "_test", "parametros"])
            reporte += [f"## {nombre}", "", f"Tasa: {y.mean():.0%}. Elegido: **{elegido['familia']}** con variables "
                        f"**{elegido['conjunto']}** — AUC CV {elegido['auc_cv']:.3f}, AUC prueba {elegido['auc_prueba']:.3f} "
                        f"(IC95 {elegido['ic95_prueba']}). Corte ALTA probabilidad de salida temprana: p ≥ {cortes['youden']:.2f}.", "",
                        "### Las 10 mejores combinaciones (por AUC CV)", "",
                        rk.sort_values("auc_cv", ascending=False).head(10).round(4).to_markdown(index=False), "",
                        "### Mejor AUC CV por conjunto de variables", "",
                        rk.groupby("conjunto")[["auc_cv", "auc_prueba"]].max().round(4).to_markdown(), "",
                        "### Importancia de variables (permutación, prueba)", "",
                        importancia.round(4).to_frame("pérdida de AUC").to_markdown(), ""]
        except Exception:
            log(f"ERROR en el modelo final de {obj}:\n{traceback.format_exc()}")

    Path(DIR / "auto_reporte.md").write_text("\n".join(reporte), encoding="utf-8")
    Path(DIR / "auto_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("=== Fin de la búsqueda automática ===")


if __name__ == "__main__":
    main()
