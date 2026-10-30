"""Proyección de ocupación con un modelo semi-Markov (UCI -> piso / egreso; piso -> egreso).

A diferencia del Markov de tiempo continuo (06_modelos_multiestado_ml.py), aquí la probabilidad de
salir de un estado depende de cuántos días lleva el paciente en él (riesgo base no paramétrico de
Cox), lo que corrige que la mayoría de salidas de UCI ocurren en los primeros días.

  - Salida de UCI: riesgos competitivos con dos Cox específicos por causa (1 = traslado a piso,
    2 = egreso directo). Se usan ingresos con un solo episodio de UCI, cuya duración es exacta.
  - Estancia en piso tras UCI: Cox con ingresos "UCI → HOSPITALIZACION" (duración exacta).
  - Limitación: no modela reingresos de piso a UCI (7 % de los ingresos) ni ingresos nuevos.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from preparacion import diseno

TMAX = 150  # días máximos de la grilla


def ajustar(df):
    """Ajusta los tres modelos de Cox. `df` es la salida de preparacion.cargar()."""
    X, ref = diseno(df)
    un_episodio = df["secuencia_servicios"].str.count("UCI") == 1
    t_uci = df.loc[un_episodio, "dias_uci"].clip(lower=1)
    causa = np.where(df.loc[un_episodio, "servicio_final"] == "UCI", 2, 1)
    def cox(filas, T, E):
        Xs = X[filas]
        Xs = Xs.loc[:, Xs.std() > 0]  # p. ej. procedencia_piso es constante en "UCI → HOSPITALIZACION"
        return CoxPHFitter(penalizer=0.05).fit(Xs.assign(T=np.asarray(T), E=np.asarray(E)), "T", "E")

    modelos = {nombre: cox(un_episodio, t_uci, (causa == k).astype(int))
               for k, nombre in [(1, "uci_a_piso"), (2, "uci_a_egreso")]}
    piso = df["secuencia_servicios"] == "UCI → HOSPITALIZACION"
    modelos["piso_a_egreso"] = cox(piso, df.loc[piso, "dias_hospitalizacion"].clip(lower=1), np.ones(piso.sum()))
    return {"cox": modelos, "ref": ref}


def _riesgo_discreto(cph, X):
    """Probabilidad de salir el día t (t = 1..TMAX) dado que seguía el día t-1; matriz n x (TMAX+1)."""
    H0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
    grilla = np.arange(TMAX + 1)
    H = np.array([H0[H0.index <= t].iloc[-1] if (H0.index <= t).any() else 0.0 for t in grilla])
    dH = np.diff(H, prepend=0.0)
    # Después del último tiempo observado se mantiene el riesgo medio de las 2 últimas semanas observadas
    ultimo = int(min(H0.index.max(), TMAX))
    if ultimo < TMAX:
        dH[ultimo + 1:] = dH[max(ultimo - 13, 1):ultimo + 1].mean()
    r = cph.predict_partial_hazard(X[cph.params_.index]).values.reshape(-1, 1)
    return 1 - np.exp(-dH.reshape(1, -1) * r)


def proyectar(modelo, df, servicio_actual, dias_en_servicio, horizontes=(1, 3, 7)):
    """Probabilidad de estar en UCI, piso o egresado dentro de h días para cada paciente.

    df: pacientes con variables de preparacion.construir_variables.
    servicio_actual: "UCI" o "HOSPITALIZACION" por paciente; dias_en_servicio: días cumplidos ahí.
    """
    X, _ = diseno(df, dict(modelo["ref"]))
    h1 = _riesgo_discreto(modelo["cox"]["uci_a_piso"], X)
    h2 = _riesgo_discreto(modelo["cox"]["uci_a_egreso"], X)
    hw = _riesgo_discreto(modelo["cox"]["piso_a_egreso"], X)
    total = h1 + h2
    factor = np.where(total > 1, 1 / np.maximum(total, 1e-12), 1.0)  # las dos causas no pueden sumar > 1
    h1, h2 = h1 * factor, h2 * factor
    S = np.cumprod(1 - (h1 + h2), axis=1)         # S[:, t] = P(sigue en UCI al final del día t)
    S[:, 0] = 1.0
    Sw = np.cumprod(1 - hw, axis=1)
    Sw[:, 0] = 1.0

    servicio = np.asarray(servicio_actual)
    dias = np.clip(np.asarray(dias_en_servicio, dtype=float).round().astype(int), 0, TMAX - max(horizontes) - 1)
    filas = []
    for i in range(len(X)):
        d = dias[i]
        fila = {}
        for h in horizontes:
            if servicio[i] == "UCI":
                base = max(S[i, d], 1e-12)
                p_uci = S[i, d + h] / base
                t = np.arange(d + 1, d + h + 1)
                p_piso = float(np.sum(S[i, t - 1] * h1[i, t] / base * Sw[i, d + h - t]))
            else:
                p_uci = 0.0
                p_piso = Sw[i, d + h] / max(Sw[i, d], 1e-12)
            fila[f"P_UCI_{h}d"] = p_uci
            fila[f"P_piso_{h}d"] = p_piso
            fila[f"P_egreso_{h}d"] = max(0.0, 1 - p_uci - p_piso)
        filas.append(fila)
    return pd.DataFrame(filas, index=df.index)
