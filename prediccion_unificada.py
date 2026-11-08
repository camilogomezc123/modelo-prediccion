"""Predicción de un paciente con todos los modelos: los de este proyecto y los de uci_control (UCI-IA).

Uso:
    python prediccion_unificada.py < paciente.json        (imprime el resultado en JSON)
    python prediccion_unificada.py --catalogos            (aseguradoras y diagnósticos para el formulario)

Entrada (JSON): edad, sexo (M/F), tercero (asegurador), diagnostico (CIE-10 principal),
servicio_inicial (UCI / HOSPITALIZACION), puntuacion_sofa_inicial, barthel_inicial,
news_inicial (opcional), fecha_ingreso_uci (AAAA-MM-DD), dias_en_uci (días que ya lleva).

Igual que predecir.py: sin edad, SOFA o Barthel ningún modelo da predicción (no se asume el dato).
Lo usa index.php; los modelos se cargan de modelos/ y de salida/uci_control/.
"""
import json
import sys
import warnings
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

from preparacion import REQUERIDAS  # noqa: E402
import predecir as pr  # noqa: E402

DIR_UCI_IA = RAIZ / "salida" / "uci_control"
# El consolidado deriva la mortalidad SOFA del puntaje por rangos fijos (verificado 1 a 1 en la base).
SOFA_MORTALIDAD = [(1, "0%"), (4, "6-8%"), (7, "22-25%"), (9, "32-35%"), (11, "50%"), (14, "80%"), (99, "92%")]
# AUC de UCI-IA medidos solo en pacientes con edad, SOFA y Barthel (prueba desde 2025-12-01, 1262 ingresos,
# 2026-11-07). Su informe da 0,77 / 0,75 / 0,70 porque incluye faltantes, que anuncian estancias cortas.
AUC_UCI_IA_COMPLETOS = {"uci_prolongada": 0.734, "alta_directa": 0.720, "rebote_uci": 0.617}
PUERTO = 8765
NOMBRE_CAMINO = {"sale_desde_uci": "Sale de la clínica desde UCI", "piso_y_alta": "Pasa a piso y egresa",
                 "vuelve_a_uci": "Vuelve a UCI en el mismo ingreso"}


def _numero(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def sofa_texto(puntaje):
    return next(t for tope, t in SOFA_MORTALIDAD if puntaje <= tope)


def _nuestros(p, fecha_uci, lleva):
    entrada = pd.DataFrame([{"ingreso": "PACIENTE", "tercero": p.get("tercero") or "",
                             "diagnostico": p.get("diagnostico") or "", "edad": p["edad"], "sexo_id": p.get("sexo"),
                             "servicio_inicial": p.get("servicio_inicial") or "UCI",
                             "puntuacion_sofa_inicial": p["puntuacion_sofa_inicial"],
                             "barthel_inicial": p["barthel_inicial"], "servicio_actual": "UCI",
                             "dias_en_servicio": lleva, "fecha_ingreso_min": pd.Timestamp(fecha_uci)}])
    salida, _, modelos, _ = pr.predecir(entrada)
    f = salida.iloc[0]

    def ficha(obj, columna, nivel=None):
        m = modelos[obj]
        return {"prob": float(f[columna]), "nivel": None if nivel is None else str(f[nivel]),
                "auc": round(float(m["auc_prueba"]), 3), "ic95": "IC95 " + m["ic95_prueba"],
                "tasa_base": round(float(m["tasa_base"]), 3), "planeacion": obj in pr.SOLO_PLANEACION,
                "modelo": m.get("nombre_modelo", m.get("familia"))}

    out = {"uci_prolongada": ficha("uci_prolongada", "prob_UCI_mas_7_dias", "riesgo_UCI_mas_7_dias"),
           "egreso_directo_uci": ficha("egreso_directo_uci", "prob_sale_de_clinica_desde_UCI")}
    for obj, etiqueta in pr.ETIQUETA_DIA1.items():
        if obj in modelos:
            nivel = f"posibilidad_{etiqueta}" if obj not in pr.SOLO_PLANEACION else None
            out[obj] = ficha(obj, f"prob_{etiqueta}", nivel)
    if "P_UCI_1d" in salida:
        out["ocupacion"] = {f"{h}d": {"uci": round(float(f[f"P_UCI_{h}d"]), 3), "piso": round(float(f[f"P_piso_{h}d"]), 3),
                                      "egreso": round(float(f[f"P_egreso_{h}d"]), 3)} for h in (1, 3, 7)}
    return out


_JSON_EGRESO = {}


def _json_egreso(ruta):
    from uci_control.prediccion import ocupacion as o
    clave = (ruta, ruta.stat().st_mtime)
    if clave not in _JSON_EGRESO:
        _JSON_EGRESO.clear()
        _JSON_EGRESO[clave] = o.exportar_json(o.ModeloEgresoDiario.cargar(ruta))
    return _JSON_EGRESO[clave]


def _uci_ia(p, fecha_uci, lleva, avisos):
    from uci_control.prediccion import trayectoria as t, ocupacion as o
    rutas = {"tray": DIR_UCI_IA / "modelo_trayectoria_tablero.joblib", "egreso": DIR_UCI_IA / "modelo_ocupacion.joblib"}
    if not rutas["tray"].exists():
        avisos.append("No están entrenados los modelos de UCI-IA (python -m uci_control modelo-trayectoria ...).")
        return None
    dx = t.diagnostico_principal(t.codigos_cie10(p.get("diagnostico")))
    valores = {"edad": p["edad"], "sexo": p.get("sexo") if p.get("sexo") in ("M", "F") else None,
               "asegurador": t.grupo_asegurador(p.get("tercero")), "dx3": dx[:3] if dx else None,
               "agrupador": t.agrupador_diagnostico([dx]) if dx else "OTROS",
               "sofa_mortalidad": t.sofa_a_mortalidad(sofa_texto(p["puntuacion_sofa_inicial"])),
               "barthel": p["barthel_inicial"], "news": p.get("news_inicial")}
    modelo = t.ModeloTrayectoria.cargar(rutas["tray"])
    f = modelo.predecir(valores).iloc[0]
    uci = [round(float(f[f"dias_uci_p{q}"]), 1) for q in (10, 50, 90)]
    out = {"p_uci_prolongada": round(float(f["p_uci_prolongada"]), 3),
           "p_alta_directa": round(float(f["p_alta_directa"]), 3),
           "p_rebote_uci": round(float(f["p_rebote_uci"]), 3),
           "camino": {NOMBRE_CAMINO[n]: round(float(f[f"p_{n}"]), 3) for n in t.TRAYECTORIAS if f"p_{n}" in f},
           "rangos_uci": {e: round(float(f[f"p_dias_uci_{e}"]), 3) for e in t.BANDAS["dias_uci"][1] if f"p_dias_uci_{e}" in f},
           "dias_uci": uci, "dias_total": [round(float(f[f"dias_total_p{q}"]), 1) for q in (10, 50, 90)],
           "dias_restantes": t.restante_condicional(*uci, lleva), "auc_completos": AUC_UCI_IA_COMPLETOS,
           "acierto_camino": (modelo.metricas.get("trayectoria") or {}).get("acierto_clase_mas_probable")}
    if rutas["egreso"].exists():
        js = _json_egreso(rutas["egreso"])
        curva = o.curva_json(js, {**valores, "dias_en_uci": None, "dia_semana_egreso": None}, fecha_uci, lleva, 14)
        out["curva_14_dias"] = [{"fecha": c["fecha"].isoformat(), "dia": "lmmjvsd"[c["fecha"].weekday()],
                                 "sale": round(c["sale"], 4), "sigue": round(c["sigue"], 4)} for c in curva]
    return out


def predecir_paciente(p: dict) -> dict:
    p = dict(p)
    for c in ("edad", "puntuacion_sofa_inicial", "barthel_inicial", "news_inicial", "dias_en_uci"):
        p[c] = _numero(p.get(c))
    nombres = {"edad": "edad", "puntuacion_sofa_inicial": "SOFA inicial", "barthel_inicial": "Barthel inicial"}
    faltan = [nombres[c] for c in REQUERIDAS if p.get(c) is None]
    if p.get("edad") is not None and not 15 <= p["edad"] <= 105:
        faltan.append("edad válida (15 a 105)")
    if not p.get("diagnostico"):
        faltan.append("diagnóstico CIE-10")
    if faltan:
        return {"estado": "incompleto", "faltan": faltan,
                "mensaje": "Sin estos datos no se da ninguna predicción: el modelo no asume valores por el paciente."}
    try:
        fecha_uci = date.fromisoformat(p.get("fecha_ingreso_uci") or "")
    except ValueError:
        fecha_uci = date.today()
    lleva = int(max(p.get("dias_en_uci") or 0, 0))
    avisos = []
    if p.get("news_inicial") is None:
        avisos.append("Sin NEWS inicial: los modelos de UCI-IA lo tratan como dato no registrado (así se entrenaron, "
                      "porque el NEWS existe solo desde mediados de 2025). Los modelos del proyecto no usan NEWS.")
    if lleva > 0:
        avisos.append(f"Lleva {lleva} día(s): las probabilidades al ingreso no cambian; la estancia restante y "
                      "la salida día a día sí tienen en cuenta los días que lleva.")
    return {"estado": "ok", "fecha_ingreso_uci": fecha_uci.isoformat(), "dias_en_uci": lleva,
            "proyecto": _nuestros(p, fecha_uci, lleva), "uci_ia": _uci_ia(p, fecha_uci, lleva, avisos), "avisos": avisos}


@lru_cache(maxsize=1)
def catalogos() -> dict:
    """Aseguradoras con modelo propio y diagnósticos principales frecuentes (código y descripción CIE-10)."""
    import joblib
    m = joblib.load(pr.DIR_MODELOS / "modelo_uci_prolongada.joblib")
    dx = pd.read_excel(RAIZ / "egresos uci 2024-2026.xlsx", sheet_name=0, usecols=["diagnostico"])["diagnostico"]
    principal = dx.dropna().astype(str).str.split(r" \| ").str[0].str.strip()
    vc = principal.value_counts()
    return {"aseguradoras": [a for a in m["aseguradoras"] if a != "OTRA"],
            "diagnosticos": [{"codigo": d.split(" - ")[0][:4], "texto": d} for d in vc[vc >= 3].index]}


def _cargar_con_cache():
    """En el servidor, cada .joblib se carga una vez y se recarga solo si el archivo cambia."""
    import joblib
    original, cache = joblib.load, {}

    def cargar(ruta, *a, **k):
        r = Path(ruta)
        clave = (str(r.resolve()), r.stat().st_mtime)
        if clave not in cache:
            cache[clave] = original(ruta, *a, **k)
        return cache[clave]
    joblib.load = cargar


def servidor():
    """Proceso local con los modelos en memoria (solo escucha en 127.0.0.1). Lo arranca index.php."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    _cargar_con_cache()

    class Manejador(BaseHTTPRequestHandler):
        def _responder(self, res):
            cuerpo = json.dumps(res, ensure_ascii=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            self._responder({"estado": "activo"} if self.path == "/estado" else catalogos())

        def do_POST(self):
            try:
                datos = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                self._responder(predecir_paciente(datos))
            except Exception as e:
                self._responder({"estado": "error", "mensaje": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    class Servidor(HTTPServer):
        allow_reuse_address = False  # en Windows, True deja a dos procesos escuchar el mismo puerto

    try:
        srv = Servidor(("127.0.0.1", PUERTO), Manejador)
    except OSError:
        return  # ya hay otro proceso activo
    predecir_paciente({"edad": 60, "puntuacion_sofa_inicial": 3, "barthel_inicial": 90, "diagnostico": "I214"})  # precarga
    srv.serve_forever()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--servidor" in sys.argv:
        servidor()
        sys.exit(0)
    try:
        res = catalogos() if "--catalogos" in sys.argv else predecir_paciente(json.loads(sys.stdin.buffer.read().decode("utf-8")))
    except Exception as e:  # el error vuelve a la página en vez de dejarla en blanco
        res = {"estado": "error", "mensaje": f"{type(e).__name__}: {e}"}
    print(json.dumps(res, ensure_ascii=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
