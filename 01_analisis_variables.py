"""Análisis de variables determinantes de la estancia (egreso) en UCI/hospitalización.

Variable objetivo: dias_total (estancia total del ingreso). Se analiza también dias_uci
y estancia prolongada (dias_total > p75).
Solo se usan como predictores variables disponibles AL INGRESO; las variables que se
conocen al egreso (SOFA/Barthel/score final, servicio_final, traslados, secuencia,
dias_uci/dias_hospitalizacion) se reportan aparte porque generan fuga de información.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error

ARCHIVO = "egresos uci 2024-2026.xlsx"
SALIDA = "variables_importantes.csv"


def cargar():
    df = pd.read_excel(ARCHIVO, sheet_name=0)
    # score_* es el NEWS (National Early Warning Score); ta_alta/ta_baja son tensión sistólica/diastólica
    df = df.rename(columns={"score_inicial": "news_inicial", "score_final": "news_final",
                            "ta_alta": "ta_sistolica", "ta_baja": "ta_diastolica"})

    # Limpieza de valores fisiológicamente imposibles
    df["peso"] = df["peso"].where(df["peso"].between(25, 250))
    df["fc"] = df["fc"].where(df["fc"].between(20, 250))
    df["ta_sistolica"] = df["ta_sistolica"].where(df["ta_sistolica"].between(40, 260))
    df["ta_diastolica"] = df["ta_diastolica"].where(df["ta_diastolica"].between(15, 160))
    df["temp_piel"] = df["temp_piel"].where(df["temp_piel"].between(30, 43))
    df["tam"] = (df["ta_sistolica"] + 2 * df["ta_diastolica"]) / 3
    df = df.drop(columns=["talla"])  # 82 % en cero

    df["sexo_m"] = (df["sexo_id"] == "M").astype(float).where(df["sexo_id"].notna())
    df["ingresa_por_uci"] = (df["servicio_inicial"] == "UCI").astype(int)
    df["mes_ingreso"] = df["fecha_ingreso_min"].dt.month
    df["dia_semana_ingreso"] = df["fecha_ingreso_min"].dt.dayofweek
    df["fin_de_semana"] = (df["dia_semana_ingreso"] >= 5).astype(int)
    df["sofa_ini_faltante"] = df["puntuacion_sofa_inicial"].isna().astype(int)
    df["barthel_ini_faltante"] = df["barthel_inicial"].isna().astype(int)

    # Diagnósticos (CIE-10)
    codigos = df["diagnostico"].fillna("").str.split(r" \| ").apply(lambda l: [c[:4] for c in l if c])
    df["n_diagnosticos"] = codigos.str.len()
    df["dx_principal"] = codigos.str[0].fillna("NA")
    df["capitulo_dx"] = df["dx_principal"].str[0]
    banderas = {
        "dx_desnutricion": ("E40", "E41", "E42", "E43", "E44", "E46"),
        "dx_sepsis": ("A40", "A41", "R65"),
        "dx_neumonia": ("J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69"),
        "dx_falla_respiratoria": ("J96", "J80"),
        "dx_iam": ("I21", "I22"),
        "dx_cardiopatia_isquemica_cronica": ("I25",),
        "dx_falla_cardiaca": ("I50",),
        "dx_acv": ("I60", "I61", "I62", "I63", "I64", "I67"),
        "dx_hta": ("I10",),
        "dx_diabetes": ("E10", "E11", "E14"),
        "dx_erc": ("N18", "N17"),
        "dx_cancer": tuple(f"C{i:02d}" for i in range(100)),
        "dx_trauma": ("S", "T0", "T1"),
        "dx_arritmia": ("I47", "I48", "I49"),
        "dx_valvulopatia": ("I34", "I35", "I36"),
        "dx_hemorragia_digestiva": ("K92",),
        "dx_obstetrico": ("O",),
    }
    for nombre, prefijos in banderas.items():
        df[nombre] = codigos.apply(lambda l: int(any(c.startswith(prefijos) for c in l)))

    # Aseguradora: categorías con < 30 casos se agrupan
    top = df["tercero"].str.strip().value_counts()
    df["aseguradora"] = df["tercero"].str.strip().where(df["tercero"].str.strip().isin(top[top >= 30].index), "OTRA")
    return df


NUMERICAS = ["edad", "sexo_m", "puntuacion_sofa_inicial", "barthel_inicial", "news_inicial",
             "peso", "fc", "ta_sistolica", "ta_diastolica", "tam", "temp_piel", "n_diagnosticos",
             "ingresa_por_uci", "mes_ingreso", "dia_semana_ingreso", "fin_de_semana",
             "sofa_ini_faltante", "barthel_ini_faltante"]
CATEGORICAS = ["aseguradora", "capitulo_dx", "dx_principal"]
# Variables conocidas solo al egreso (fuga de información si se usan para predecir)
POST_EGRESO = ["puntuacion_sofa_final", "barthel_final", "news_final", "numero_traslados",
               "servicio_final", "dias_uci", "dias_hospitalizacion"]


def eta2(y, g):
    """Proporción de varianza de y explicada por el grupo g (sobre log-estancia)."""
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    media = d["y"].mean()
    entre = d.groupby("g")["y"].agg(["mean", "size"]).pipe(lambda t: (t["size"] * (t["mean"] - media) ** 2).sum())
    return entre / ((d["y"] - media) ** 2).sum()


def univariado(df, objetivo):
    y = np.log1p(df[objetivo])
    filas = []
    banderas = [c for c in df.columns if c.startswith("dx_") and c != "dx_principal"]
    for c in NUMERICAS + banderas:
        m = df[c].notna()
        rho, p = stats.spearmanr(df.loc[m, c], y[m])
        filas.append({"variable": c, "tipo": "numérica/binaria", "efecto": rho, "metrica": "Spearman rho",
                      "p_valor": p, "n": int(m.sum())})
    for c in CATEGORICAS:
        grupos = [y[df[c] == k] for k in df[c].dropna().unique() if (df[c] == k).sum() >= 10]
        h, p = stats.kruskal(*grupos)
        filas.append({"variable": c, "tipo": "categórica", "efecto": eta2(y, df[c]), "metrica": "eta²",
                      "p_valor": p, "n": int(df[c].notna().sum())})
    return pd.DataFrame(filas)


def matriz(df):
    banderas = [c for c in df.columns if c.startswith("dx_") and c != "dx_principal"]
    X = df[NUMERICAS + banderas + ["aseguradora", "capitulo_dx"]].copy()
    for c in ["aseguradora", "capitulo_dx"]:
        X[c] = X[c].astype("category")
    return X


def importancia_multivariada(df):
    X = matriz(df)
    y = np.log1p(df["dias_total"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    modelo = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=24,
                                           min_samples_leaf=30, l2_regularization=1.0,
                                           categorical_features="from_dtype", random_state=42)
    cv = cross_val_score(modelo, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    modelo.fit(Xtr, ytr)
    pred = modelo.predict(Xte)
    print(f"\n[Regresión log(dias_total)] R² CV5 = {cv.mean():.3f} ± {cv.std():.3f} | "
          f"R² test = {r2_score(yte, pred):.3f} | MAE test (días) = "
          f"{mean_absolute_error(np.expm1(yte), np.expm1(pred)):.1f}")
    pi = permutation_importance(modelo, Xte, yte, n_repeats=15, random_state=42, scoring="r2")
    imp_reg = pd.Series(pi.importances_mean, index=X.columns, name="perm_imp_regresion")

    # Clasificación: estancia prolongada (> p75)
    p75 = df["dias_total"].quantile(0.75)
    yc = (df["dias_total"] > p75).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, yc, test_size=0.25, random_state=42, stratify=yc)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=24,
                                         min_samples_leaf=30, l2_regularization=1.0,
                                         categorical_features="from_dtype", random_state=42)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f"[Clasificación estancia > {p75:.0f} días] AUC test = {auc:.3f}")
    pi = permutation_importance(clf, Xte, yte, n_repeats=15, random_state=42, scoring="roc_auc")
    imp_clf = pd.Series(pi.importances_mean, index=X.columns, name="perm_imp_clasificacion")
    return pd.concat([imp_reg, imp_clf], axis=1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = cargar()
    print(f"Registros: {len(df)} | estancia mediana {df.dias_total.median():.0f} d, "
          f"p75 {df.dias_total.quantile(.75):.0f} d")

    uni = univariado(df, "dias_total").set_index("variable")
    uni_uci = univariado(df, "dias_uci").set_index("variable")[["efecto"]].rename(columns={"efecto": "efecto_dias_uci"})
    imp = importancia_multivariada(df)
    res = uni.join(uni_uci).join(imp, how="outer")
    res["perm_imp_regresion"] = res["perm_imp_regresion"].fillna(0)
    res = res.sort_values("perm_imp_regresion", ascending=False)
    res.to_csv(SALIDA, encoding="utf-8-sig", sep=";", decimal=",")

    pd.set_option("display.width", 200)
    print("\n=== Ranking (variables disponibles al ingreso) ===")
    print(res.round(4).to_string())

    print("\n=== Variables post-egreso (NO usar como predictores; solo referencia) ===")
    y = np.log1p(df["dias_total"])
    for c in POST_EGRESO:
        v = df[c] if pd.api.types.is_numeric_dtype(df[c]) else (df[c] == "UCI").astype(int)
        m = v.notna()
        print(f"{c:28s} Spearman rho = {stats.spearmanr(v[m], y[m])[0]:+.3f}")

    print("\n=== Estancia mediana por grupo ===")
    for c in ["ingresa_por_uci", "capitulo_dx", "aseguradora"]:
        print(df.groupby(c)["dias_total"].agg(["size", "median", "mean"]).sort_values("size", ascending=False).head(10).round(1).to_string(), "\n")
    df["edad_grupo"] = pd.cut(df["edad"], [0, 40, 60, 75, 120])
    df["sofa_grupo"] = pd.cut(df["puntuacion_sofa_inicial"], [-1, 1, 4, 8, 30])
    df["barthel_grupo"] = pd.cut(df["barthel_inicial"], [-1, 40, 85, 99, 100])
    for c in ["edad_grupo", "sofa_grupo", "barthel_grupo", "n_diagnosticos"]:
        print(df.groupby(c, observed=True)["dias_total"].agg(["size", "median", "mean"]).round(1).to_string(), "\n")
    for c in ["dx_desnutricion", "dx_sepsis", "dx_neumonia", "dx_falla_respiratoria", "dx_iam",
              "dx_cardiopatia_isquemica_cronica", "dx_acv", "dx_falla_cardiaca", "dx_erc", "dx_cancer"]:
        g = df.groupby(c)["dias_total"].median()
        print(f"{c:34s} n={df[c].sum():5d}  mediana con={g.get(1, np.nan):5.1f}  sin={g.get(0, np.nan):5.1f}")


if __name__ == "__main__":
    main()
