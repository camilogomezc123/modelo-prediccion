"""Revisión detallada y selección de variables para los modelos de UCI.

Objetivos (definidos con datos conocidos al ingreso a UCI):
  - uci_prolongada: dias_uci > 7
  - egreso_directo_uci: servicio_final == "UCI" (sale de la clínica desde UCI, no pasa a piso)

Cohorte: ingresos entre 2024-10-01 y 2026-08-31. Fuera de esa ventana la base solo contiene
estancias largas (ingresos antiguos aún hospitalizados en oct-2024) o cortas (ingresos de
sep-2026 que ya egresaron), lo que sesgaría el objetivo.

Validación: StratifiedGroupKFold agrupando por paciente (455 pacientes tienen >1 ingreso)
y validación temporal (entrena antes de 2026-02-01, prueba después).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from preparacion import ARCHIVO, INICIO, FIN, CORTE_TEMPORAL, BANDERAS_DX, cargar


DXP = [f"dxp_{k}" for k in BANDERAS_DX]
DXA = [f"dxa_{k}" for k in BANDERAS_DX]
GRUPOS = {
    "edad": ["edad"],
    "sexo": ["sexo_m"],
    "procedencia_piso": ["procedencia_piso"],
    "sofa_inicial": ["puntuacion_sofa_inicial"],
    "barthel_inicial": ["barthel_inicial"],
    "news_inicial": ["news_inicial"],
    "signos_vitales": ["fc", "ta_sistolica", "ta_diastolica", "tam", "temp_piel"],
    "peso": ["peso"],
    "aseguradora": ["aseguradora"],
    "dx_principal": ["capitulo_dx"] + DXP,
    "dx_secundarios": ["n_diagnosticos"] + DXA,
    "calendario": ["mes_ingreso", "dia_semana_ingreso", "fin_de_semana"],
}
GRUPOS["dia_semana"] = ["dia_semana_ingreso", "fin_de_semana"]
BASE = ["edad", "procedencia_piso", "sofa_inicial", "barthel_inicial", "aseguradora", "dx_principal", "dia_semana"]
CONJUNTOS = {
    "todas (incluye variables con fuga)": [g for g in GRUPOS if g != "dia_semana"],
    "BASE recomendada": BASE,
    "BASE + news_inicial": BASE + ["news_inicial"],
    "BASE + dx_secundarios (fuga parcial)": BASE + ["dx_secundarios"],
    "BASE + signos_vitales (fuga)": BASE + ["signos_vitales"],
    "BASE sin aseguradora": [g for g in BASE if g != "aseguradora"],
    "BASE sin sofa": [g for g in BASE if g != "sofa_inicial"],
}
CATEG = ["aseguradora", "capitulo_dx"]
OBJETIVOS = ["uci_prolongada", "egreso_directo_uci"]


def X_de(df, grupos):
    cols = [c for g in grupos for c in GRUPOS[g]]
    X = df[cols].copy()
    for c in CATEG:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=16,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          categorical_features="from_dtype", random_state=42)


def auc_cv(df, grupos, y, folds):
    X = X_de(df, grupos)
    pred = np.zeros(len(df))
    for tr, te in folds:
        pred[te] = gbm().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y, pred), [roc_auc_score(y[te], pred[te]) for _, te in folds]


def auc_temporal(df, grupos, y):
    X = X_de(df, grupos)
    tr = (df["fecha_ingreso_min"] < CORTE_TEMPORAL).values
    m = gbm().fit(X[tr], y[tr])
    return roc_auc_score(y[~tr], m.predict_proba(X[~tr])[:, 1])


def univariado(df, y):
    filas = []
    for g, cols in GRUPOS.items():
        if g == "dia_semana":
            continue
        for c in cols:
            v = df[c]
            if c in CATEG:
                tabla = pd.crosstab(v, y)
                chi2, p, _, _ = stats.chi2_contingency(tabla)
                efecto = np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))
                filas.append((g, c, "V de Cramér", efecto, p, v.notna().mean()))
            else:
                m = v.notna()
                if v[m].nunique() < 2:
                    continue
                auc = roc_auc_score(y[m], v[m])
                p = stats.mannwhitneyu(v[m][y[m] == 1], v[m][y[m] == 0]).pvalue
                filas.append((g, c, "AUC univariado", auc, p, m.mean()))
    return pd.DataFrame(filas, columns=["grupo", "variable", "metrica", "efecto", "p_valor", "completitud"])


def logistica(df, y):
    """Regresión logística L1 (casos con SOFA y Barthel) como contraste interpretable."""
    m = df["puntuacion_sofa_inicial"].notna() & df["barthel_inicial"].notna()
    cols = [c for g in GRUPOS for c in GRUPOS[g] if c not in CATEG and g not in ("news_inicial", "dia_semana")]
    X = pd.get_dummies(df.loc[m, cols + CATEG], columns=CATEG, drop_first=True, dtype=float)
    modelo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(penalty="l1", solver="liblinear", C=0.05, max_iter=2000))
    modelo.fit(X, y[m.values])
    coef = pd.Series(modelo[-1].coef_[0], index=X.columns)
    return coef[coef != 0].sort_values(key=abs, ascending=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    df = cargar()
    grupos_pac = df["paciente_id"].values
    print(f"Cohorte {INICIO} a {FIN}: {len(df)} ingresos, {df.paciente_id.nunique()} pacientes")
    todos = [g for g in GRUPOS if g != "dia_semana"]  # dia_semana es un subconjunto de calendario
    resumen = []

    for obj in OBJETIVOS:
        y = df[obj].values
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, y, grupos_pac))
        print(f"\n{'=' * 90}\nOBJETIVO: {obj}  (prevalencia {y.mean():.1%})\n{'=' * 90}")

        uni = univariado(df, y)
        uni.to_csv(f"univariado_{obj}.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
        u = uni.assign(fuerza=np.where(uni.metrica == "AUC univariado", (uni.efecto - 0.5).abs(), uni.efecto))
        print("\n-- Univariado (más fuertes; AUC < 0.5 = protector) --")
        print(u.sort_values("fuerza", ascending=False).head(22).round(4).to_string(index=False))

        base, por_fold = auc_cv(df, todos, y, folds)
        base_t = auc_temporal(df, todos, y)
        print(f"\nModelo completo: AUC CV por paciente = {base:.3f} (folds {min(por_fold):.3f}-{max(por_fold):.3f})"
              f" | AUC temporal = {base_t:.3f}")

        # Importancia por eliminación de grupos (más fiable que permutación con variables correlacionadas)
        filas = []
        for g in todos:
            resto = [x for x in todos if x != g]
            a, f = auc_cv(df, resto, y, folds)
            solo, _ = auc_cv(df, [g], y, folds)
            filas.append({"grupo": g, "perdida_auc_al_quitar": base - a,
                          "folds_que_empeoran": sum(b > c for b, c in zip(por_fold, f)),
                          "auc_solo_grupo": solo,
                          "perdida_auc_temporal": base_t - auc_temporal(df, resto, y)})
        abl = pd.DataFrame(filas).sort_values("perdida_auc_al_quitar", ascending=False)
        print("\n-- Aporte de cada grupo (pérdida de AUC al quitarlo) --")
        print(abl.round(4).to_string(index=False))

        # Permutación por variable dentro de cada fold
        X = X_de(df, todos)
        imps = []
        for tr, te in folds:
            mdl = gbm().fit(X.iloc[tr], y[tr])
            imps.append(permutation_importance(mdl, X.iloc[te], y[te], scoring="roc_auc",
                                               n_repeats=5, random_state=0).importances_mean)
        imps = pd.DataFrame(imps, columns=X.columns)
        perm = pd.DataFrame({"media": imps.mean(), "folds_positivos": (imps > 0.001).sum()})
        print("\n-- Permutación por variable (top 20) --")
        print(perm.sort_values("media", ascending=False).head(20).round(4).to_string())

        print("\n-- Logística L1 (casos completos SOFA+Barthel, coeficientes estandarizados) --")
        print(logistica(df, y).head(25).round(3).to_string())

        # Sensibilidad 1: ¿el modelo aprende de que el SOFA falte?
        con = df["puntuacion_sofa_inicial"].notna().values
        pred = np.zeros(len(df))
        for tr, te in folds:
            pred[te] = gbm().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        print(f"\nSensibilidad: AUC en casos CON SOFA = {roc_auc_score(y[con], pred[con]):.3f} | "
              f"SIN SOFA = {roc_auc_score(y[~con], pred[~con]):.3f} | "
              f"prevalencia con/sin SOFA = {y[con].mean():.2f}/{y[~con].mean():.2f}")
        # Sensibilidad 2: diagnósticos secundarios vs solo principal
        sin_sec, _ = auc_cv(df, [g for g in todos if g != "dx_secundarios"], y, folds)
        print(f"Sensibilidad: sin diagnósticos secundarios AUC = {sin_sec:.3f} (vs {base:.3f})")

        # Conjuntos candidatos tras excluir variables con fuga (signos vitales y diagnósticos
        # secundarios contienen información del final de la estancia)
        print("\n-- Conjuntos candidatos (AUC CV por paciente | AUC temporal) --")
        for nombre, conj in CONJUNTOS.items():
            a, f = auc_cv(df, conj, y, folds)
            print(f"  {nombre:40s} {a:.3f} (folds {min(f):.3f}-{max(f):.3f}) | {auc_temporal(df, conj, y):.3f}")

        abl["objetivo"] = obj
        resumen.append(abl)

    pd.concat(resumen).to_csv("aporte_grupos_variables.csv", sep=";", decimal=",", index=False,
                              encoding="utf-8-sig")


if __name__ == "__main__":
    main()
