"""Confrontación de las variables finales entre sí.

1. Asociación entre predictores (Spearman, V de Cramér, razón de correlación eta).
2. Colinealidad (VIF).
3. Cruces clave entre variables (perfil de cada una dentro de la otra).
4. Confusión: ¿el efecto de la aseguradora se explica por el tipo de paciente?
5. Interacciones sobre cada objetivo: tasas por pares de variables y ganancia de AUC al
   permitir que cada par interactúe en el modelo.
"""
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from preparacion import cargar, OBJETIVOS

VARS = ["edad", "procedencia_piso", "puntuacion_sofa_inicial", "barthel_inicial",
        "news_inicial", "aseguradora", "capitulo_dx"]
CAT = {"aseguradora", "capitulo_dx", "procedencia_piso"}
NOMBRE_CORTO = {"edad": "edad", "procedencia_piso": "proced_piso", "puntuacion_sofa_inicial": "sofa",
                "barthel_inicial": "barthel", "news_inicial": "news", "aseguradora": "asegur",
                "capitulo_dx": "cap_dx"}


def cramer_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(t)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))


def eta(cat, num):
    d = pd.DataFrame({"c": cat, "x": num}).dropna()
    medias = d.groupby("c")["x"].agg(["mean", "size"])
    entre = (medias["size"] * (medias["mean"] - d["x"].mean()) ** 2).sum()
    return np.sqrt(entre / ((d["x"] - d["x"].mean()) ** 2).sum())


def asociacion(df):
    m = pd.DataFrame(index=VARS, columns=VARS, dtype=float)
    for a, b in itertools.product(VARS, VARS):
        if a == b:
            m.loc[a, b] = 1.0
        elif a in CAT and b in CAT:
            m.loc[a, b] = cramer_v(df[a], df[b])
        elif a in CAT:
            m.loc[a, b] = eta(df[a], df[b])
        elif b in CAT:
            m.loc[a, b] = eta(df[b], df[a])
        else:
            m.loc[a, b] = stats.spearmanr(df[a], df[b], nan_policy="omit")[0]
    return m.rename(index=NOMBRE_CORTO, columns=NOMBRE_CORTO)


def vif(df):
    X = df[["edad", "procedencia_piso", "puntuacion_sofa_inicial", "barthel_inicial"]]
    # Referencia = categoría más frecuente; con una referencia pequeña el VIF se infla artificialmente
    dummies = [pd.get_dummies(df[c], prefix=c, dtype=float).drop(columns=f"{c}_{df[c].mode()[0]}")
               for c in ["aseguradora", "capitulo_dx"]]
    X = pd.concat([X] + dummies, axis=1)
    X = X.fillna(X.median()).astype(float)
    X = X.loc[:, X.std() > 0]
    corr_inv = np.linalg.pinv(np.corrcoef(X.values, rowvar=False))
    return pd.Series(np.diag(corr_inv), index=X.columns).sort_values(ascending=False)


def grupos(df):
    g = pd.DataFrame(index=df.index)
    g["edad"] = pd.cut(df["edad"], [0, 50, 70, 120], labels=["<50", "50-70", ">70"])
    g["sofa"] = pd.cut(df["puntuacion_sofa_inicial"], [-1, 1, 4, 8, 30], labels=["0-1", "2-4", "5-8", ">8"])
    g["barthel"] = pd.cut(df["barthel_inicial"], [-1, 40, 85, 99, 100], labels=["<=40", "45-85", "90-95", "100"])
    g["proced"] = df["procedencia_piso"].map({0: "directo UCI", 1: "desde piso"})
    top = df["capitulo_dx"].value_counts().index[:6]
    g["cap_dx"] = df["capitulo_dx"].where(df["capitulo_dx"].isin(top), "otros")
    top_a = df["aseguradora"].value_counts().index[:5]
    g["asegur"] = df["aseguradora"].where(df["aseguradora"].isin(top_a), "OTRA").str[:22]
    return g


def tasa_cruzada(g, y, a, b):
    d = pd.DataFrame({"a": g[a], "b": g[b], "y": y}).dropna()
    t = d.pivot_table(index="a", columns="b", values="y", aggfunc="mean", observed=True)
    n = d.pivot_table(index="a", columns="b", values="y", aggfunc="size", observed=True)
    return (t * 100).round(0).where(n >= 20).astype("Float64")  # celdas con < 20 casos ocultas


def ganancia_interacciones(df, y, folds):
    """AUC del modelo aditivo (sin interacciones) vs. permitir interacción de cada par."""
    feats = ["edad", "procedencia_piso", "puntuacion_sofa_inicial", "barthel_inicial", "aseguradora", "capitulo_dx"]
    X = df[feats].copy()
    for c in ["aseguradora", "capitulo_dx"]:
        X[c] = X[c].astype("category")

    def auc(cst):
        pred = np.zeros(len(X))
        for tr, te in folds:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=16,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               categorical_features="from_dtype",
                                               interaction_cst=cst, random_state=42)
            pred[te] = m.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        return roc_auc_score(y, pred)

    idx = list(range(len(feats)))
    aditivo = auc("no_interactions")
    completo = auc(None)
    filas = []
    for i, j in itertools.combinations(idx, 2):
        cst = [[i, j]] + [[k] for k in idx if k not in (i, j)]
        filas.append({"par": f"{NOMBRE_CORTO[feats[i]]} x {NOMBRE_CORTO[feats[j]]}", "ganancia_auc": auc(cst) - aditivo})
    return aditivo, completo, pd.DataFrame(filas).sort_values("ganancia_auc", ascending=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    df = cargar()
    g = grupos(df)

    print("=" * 90 + "\n1. ASOCIACIÓN ENTRE PREDICTORES\n" + "=" * 90)
    print("(num-num: Spearman; cat-cat: V de Cramér; cat-num: eta. |valor| > 0,5 = fuerte)")
    m = asociacion(df)
    print(m.round(2).to_string())
    m.to_csv("asociacion_predictores.csv", sep=";", decimal=",", encoding="utf-8-sig")

    print("\n" + "=" * 90 + "\n2. COLINEALIDAD (VIF; > 5 preocupante)\n" + "=" * 90)
    print(vif(df).head(10).round(2).to_string())

    print("\n" + "=" * 90 + "\n3. CRUCES ENTRE VARIABLES\n" + "=" * 90)
    print("\n-- SOFA inicial (mediana) y Barthel inicial (mediana) por capítulo diagnóstico --")
    print(df.groupby(g["cap_dx"]).agg(n=("ingreso", "size"), sofa=("puntuacion_sofa_inicial", "median"),
                                       barthel=("barthel_inicial", "median"), edad=("edad", "median"),
                                       pct_desde_piso=("procedencia_piso", "mean")).round(2).to_string())
    print("\n-- Mezcla diagnóstica por aseguradora (% de sus ingresos) --")
    print((pd.crosstab(g["asegur"], g["cap_dx"], normalize="index") * 100).round(0).to_string())
    print("\n-- Gravedad por aseguradora --")
    print(df.groupby(g["asegur"]).agg(n=("ingreso", "size"), sofa=("puntuacion_sofa_inicial", "mean"),
                                       barthel=("barthel_inicial", "mean"), edad=("edad", "mean"),
                                       pct_desde_piso=("procedencia_piso", "mean")).round(2).to_string())
    print("\n-- Barthel inicial por grupo de edad (% en cada categoría) --")
    print((pd.crosstab(g["edad"], g["barthel"], normalize="index") * 100).round(0).to_string())
    print("\n-- SOFA por grupo de Barthel (% en cada categoría) --")
    print((pd.crosstab(g["barthel"], g["sofa"], normalize="index") * 100).round(0).to_string())
    print("\n-- Procedencia por grupo de SOFA (% desde piso) --")
    print((df.groupby(g["sofa"], observed=True)["procedencia_piso"].mean() * 100).round(0).to_string())

    print("\n" + "=" * 90 + "\n4. CONFUSIÓN: EFECTO DE LA ASEGURADORA ANTES Y DESPUÉS DE AJUSTAR\n" + "=" * 90)
    for obj, nombre in OBJETIVOS.items():
        y = df[obj]
        base = pd.get_dummies(df["aseguradora"], prefix="", prefix_sep="", dtype=float)
        base = base.drop(columns="NUEVA EMPRESA PROMOTORA DE SALUD SA")  # referencia: la más frecuente
        ajuste = pd.concat([df[["edad", "procedencia_piso", "puntuacion_sofa_inicial", "barthel_inicial"]],
                            pd.get_dummies(df["capitulo_dx"], prefix="cap", drop_first=True, dtype=float)], axis=1)
        ajuste = ajuste.fillna(ajuste.median())
        ors = {}
        for etiqueta, X in [("OR crudo", base), ("OR ajustado", pd.concat([base, ajuste], axis=1))]:
            lr = LogisticRegression(C=1e4, max_iter=5000).fit((X - X.mean()) / X.std().replace(0, 1), y)
            ors[etiqueta] = np.exp(lr.coef_[0][:base.shape[1]] / X.std().replace(0, 1).values[:base.shape[1]])
        print(f"\n{nombre} (referencia: Nueva EPS)")
        print(pd.DataFrame(ors, index=base.columns).assign(n=df["aseguradora"].value_counts()).round(2)
              .sort_values("n", ascending=False).to_string())

    print("\n" + "=" * 90 + "\n5. INTERACCIONES SOBRE LOS OBJETIVOS\n" + "=" * 90)
    pares = [("sofa", "barthel"), ("sofa", "cap_dx"), ("edad", "barthel"), ("sofa", "proced"),
             ("cap_dx", "asegur"), ("edad", "sofa")]
    for obj, nombre in OBJETIVOS.items():
        y = df[obj].values
        print(f"\n### {nombre} (tasa global {y.mean():.0%}); celdas = % con el desenlace")
        for a, b in pares:
            print(f"\n-- {a} (filas) x {b} (columnas) --")
            print(tasa_cruzada(g, y, a, b).to_string())
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, y, df["paciente_id"]))
        aditivo, completo, inter = ganancia_interacciones(df, y, folds)
        print(f"\nModelo aditivo (sin interacciones) AUC = {aditivo:.3f} | con todas las interacciones = {completo:.3f}")
        print("Ganancia de AUC al permitir solo la interacción de cada par:")
        print(inter.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
