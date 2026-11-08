"""Modelos de salida temprana de UCI para uso diario (solo variables base del primer día).

Uso:  python 09_modelos_dia1.py      (requiere haber corrido 08_busqueda_automatica.py)

La búsqueda 08 eligió para 3 días y piso conjuntos con historia del paciente, CIE-10 de 3 caracteres
y carga de UCI. En pacientes nuevos esas variables no se pueden calcular con certeza (la historia y
la carga dependen de que la base histórica esté al día) y su ganancia fue menor que el ruido
(<= 0,005 de AUC CV). Para no asumir valores, aquí se toma la mejor combinación con variables base
de cada objetivo (hiperparámetros ya optimizados en 08), se calibra y se reentrena con toda la cohorte.

Salida: modelos/dia1_<objetivo>.joblib (los usa predecir.py) y modelos/dia1_resumen.json.
"""
import importlib
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from preparacion import cargar, solo_completos, CORTE_TEMPORAL
from modelos_auto import OBJETIVOS_AUTO, columnas, agregar_objetivos

busqueda = importlib.import_module("08_busqueda_automatica")
warnings.filterwarnings("ignore")
CONJUNTO = "base"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ranking = pd.read_csv(busqueda.RANKING, sep=";", decimal=",", encoding="utf-8-sig")
    df = agregar_objetivos(solo_completos(cargar()))
    tr = (df["fecha_ingreso_min"] < CORTE_TEMPORAL).values
    X = df[columnas(CONJUNTO)]
    resumen = {}
    for obj, nombre in OBJETIVOS_AUTO.items():
        y = df[obj].values
        filas = ranking[(ranking["objetivo"] == obj) & (ranking["conjunto"] == CONJUNTO)
                        & ranking["familia"].isin(busqueda.FAMILIAS)]
        mejor = filas.sort_values("auc_cv", ascending=False).iloc[0]
        params = json.loads(mejor["parametros"])
        modelo = CalibratedClassifierCV(busqueda.construir(mejor["familia"], params, CONJUNTO), method="isotonic", cv=5)

        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=busqueda.SEMILLA)
                     .split(X[tr], y[tr], df.loc[tr, "paciente_id"].values))
        p_oof = busqueda.oof(modelo, X[tr], y[tr], folds)
        p_test = clone(modelo).fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
        info = {"familia": mejor["familia"], "auc_cv": roc_auc_score(y[tr], p_oof),
                "auc_prueba": roc_auc_score(y[~tr], p_test), "ic95_prueba": busqueda.auc_ic(y[~tr], p_test),
                "brier_prueba": brier_score_loss(y[~tr], p_test)}
        cortes = busqueda.umbrales(y[tr], p_oof)
        final = clone(modelo).fit(X, y)
        joblib.dump({"modelo": final, "columnas": columnas(CONJUNTO), "conjunto": CONJUNTO, "objetivo": nombre,
                     "umbrales": cortes, "tasa_base": float(y.mean()), "aseguradoras": sorted(df["aseguradora"].unique()),
                     "categorias": {c: sorted(df[c].unique()) for c in ["aseguradora", "capitulo_dx"]}, **info},
                    busqueda.DIR / f"dia1_{obj}.joblib")
        resumen[obj] = {**info, "umbrales": cortes, "tasa_base": float(y.mean())}
        print(f"{obj}: {info['familia']} | AUC CV {info['auc_cv']:.3f} | prueba {info['auc_prueba']:.3f} "
              f"({info['ic95_prueba']}) | Brier {info['brier_prueba']:.3f} | cortes {cortes}")
    (busqueda.DIR / "dia1_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
