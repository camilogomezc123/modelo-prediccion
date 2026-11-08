<?php
// Predicción unificada de egresos UCI: modelos del proyecto + modelos de UCI-IA (uci_control).
// El cálculo lo hace prediccion_unificada.py. Primero se intenta el proceso local con los modelos en
// memoria (127.0.0.1:8765); si no está activo se arranca y esta consulta se resuelve por línea de comandos.
const PYTHON = 'C:\\Users\\kevin\\AppData\\Local\\Programs\\Python\\Python312\\python.exe';
const PUERTO = 8765;
$script = __DIR__ . DIRECTORY_SEPARATOR . 'prediccion_unificada.py';
$cache_catalogos = __DIR__ . '/salida/catalogos_formulario.json';

function activo() {
    $s = @fsockopen('127.0.0.1', PUERTO, $errno, $errstr, 0.3);
    if ($s) { fclose($s); return true; }
    return false;
}

function arrancar($script) {
    // Evita lanzar varios procesos mientras el primero carga los modelos (tarda unos 12 s)
    $marca = sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'prediccion_uci_arranque.txt';
    if (is_file($marca) && time() - filemtime($marca) < 120) return;
    touch($marca);
    pclose(popen('start "" /B "' . PYTHON . '" -X utf8 "' . $script . '" --servidor > NUL 2>&1', 'r'));
}

function por_servidor($metodo, $cuerpo = null) {
    $ctx = stream_context_create(['http' => ['method' => $metodo, 'timeout' => 90, 'ignore_errors' => true,
        'header' => "Content-Type: application/json\r\n", 'content' => $cuerpo ?? '']]);
    return @file_get_contents('http://127.0.0.1:' . PUERTO . '/', false, $ctx);
}

function por_consola($script, $args, $entrada = '') {
    $p = proc_open('"' . PYTHON . '" -X utf8 "' . $script . '" ' . $args, [0 => ['pipe', 'r'], 1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $tubos);
    if (!is_resource($p)) return json_encode(['estado' => 'error', 'mensaje' => 'No se pudo ejecutar Python.']);
    fwrite($tubos[0], $entrada); fclose($tubos[0]);
    $salida = stream_get_contents($tubos[1]); $error = stream_get_contents($tubos[2]);
    fclose($tubos[1]); fclose($tubos[2]); proc_close($p);
    return trim($salida) !== '' ? $salida : json_encode(['estado' => 'error', 'mensaje' => 'Python no respondió: ' . substr($error, -600)]);
}

function consultar($script, $metodo, $cuerpo = null) {
    if (activo()) {
        $r = por_servidor($metodo, $cuerpo);
        if ($r !== false && $r !== '') return $r;
    } else {
        arrancar($script);
    }
    return $metodo === 'POST' ? por_consola($script, '', $cuerpo) : por_consola($script, '--catalogos');
}

$accion = $_GET['accion'] ?? '';
if ($accion === 'predecir' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json; charset=utf-8');
    echo consultar($script, 'POST', file_get_contents('php://input'));
    exit;
}
if ($accion === 'catalogos') {
    header('Content-Type: application/json; charset=utf-8');
    if (!is_file($cache_catalogos)) {
        $r = consultar($script, 'GET');
        $js = json_decode($r, true);
        if (is_array($js) && isset($js['aseguradoras'])) {
            @mkdir(dirname($cache_catalogos), 0777, true);
            file_put_contents($cache_catalogos, $r);
        }
        echo $r;
    } else {
        echo file_get_contents($cache_catalogos);
    }
    exit;
}
if (!activo()) arrancar($script);  // precarga los modelos mientras se llena el formulario
?><!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicción de egresos UCI</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f4f2; --surface-1: #fcfcfb; --borde: #dcdbd6;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #77766f;
  --series-1: #2a78d6; --pista: #e8e7e3; --acento: #2a78d6;
  --aviso-fondo: #fff6e0; --aviso-borde: #eda100;
  --error-fondo: #fde8e8; --error-borde: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --borde: #34332f;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --series-1: #3987e5; --pista: #2a2a27; --acento: #3987e5;
    --aviso-fondo: #2e2610; --aviso-borde: #c98500;
    --error-fondo: #331a1a; --error-borde: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --borde: #34332f;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --series-1: #3987e5; --pista: #2a2a27; --acento: #3987e5;
  --aviso-fondo: #2e2610; --aviso-borde: #c98500;
  --error-fondo: #331a1a; --error-borde: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { padding: 20px 16px 8px; max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
header p { margin: 0; color: var(--text-secondary); }
main { max-width: 1180px; margin: 0 auto; padding: 8px 16px 40px; display: grid; gap: 16px;
  grid-template-columns: minmax(280px, 340px) minmax(0, 1fr); align-items: start; }
@media (max-width: 860px) { main { grid-template-columns: minmax(0, 1fr); } }
.tarjeta { background: var(--surface-1); border: 1px solid var(--borde); border-radius: 10px; padding: 16px; }
h2 { font-size: 16px; margin: 0 0 12px; }
h3 { font-size: 14px; margin: 0 0 6px; color: var(--text-secondary); font-weight: 600; }
form label { display: block; font-size: 13px; color: var(--text-secondary); margin: 10px 0 4px; }
form label .op { color: var(--text-muted); font-weight: normal; }
input, select { width: 100%; padding: 8px 10px; font: inherit; color: var(--text-primary);
  background: var(--surface-0); border: 1px solid var(--borde); border-radius: 6px; }
input:focus, select:focus { outline: 2px solid var(--acento); outline-offset: 1px; }
.fila2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
button { margin-top: 16px; width: 100%; padding: 10px; font: inherit; font-weight: 600; color: #fff;
  background: var(--acento); border: 0; border-radius: 6px; cursor: pointer; }
button:disabled { opacity: .6; cursor: wait; }
.regla { font-size: 12px; color: var(--text-muted); margin-top: 10px; }
#resultados { display: grid; gap: 16px; }
.vacio { color: var(--text-secondary); }
.rejilla { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(min(200px, 100%), 1fr)); }
.indicador { border: 1px solid var(--borde); border-radius: 8px; padding: 12px; }
.indicador .valor { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
.indicador .nombre { font-size: 13px; color: var(--text-secondary); }
.indicador .detalle { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
.etiqueta { display: inline-block; font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 99px;
  border: 1px solid var(--borde); color: var(--text-primary); margin-left: 6px; vertical-align: middle; }
.barra-h { display: grid; grid-template-columns: 150px 1fr 48px; gap: 8px; align-items: center; font-size: 13px; margin: 6px 0; }
.barra-h .pista { background: var(--pista); border-radius: 4px; height: 12px; }
.barra-h .relleno { background: var(--series-1); height: 12px; border-radius: 4px; }
.barra-h .num { text-align: right; font-variant-numeric: tabular-nums; }
.dos { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
@media (max-width: 620px) { .dos { grid-template-columns: 1fr; } .barra-h { grid-template-columns: 110px 1fr 44px; } }
.tabla-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--borde); }
th { color: var(--text-secondary); font-weight: 600; }
.grafico { position: relative; }
.grafico svg { width: 100%; height: 220px; display: block; }
.grafico .eje { fill: var(--text-muted); font-size: 11px; }
.grafico .rejilla-linea { stroke: var(--borde); stroke-width: 1; }
.grafico rect.dato { fill: var(--series-1); }
.grafico rect.blanco { fill: transparent; cursor: pointer; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--borde); border-radius: 6px; padding: 6px 8px; font-size: 12px; white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none; }
details summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; margin-top: 8px; }
.aviso { background: var(--aviso-fondo); border-left: 4px solid var(--aviso-borde); padding: 10px 12px; border-radius: 6px; font-size: 13px; }
.error { background: var(--error-fondo); border-left: 4px solid var(--error-borde); padding: 10px 12px; border-radius: 6px; }
.nota { font-size: 12px; color: var(--text-muted); margin: 8px 0 0; }
</style>
</head>
<body>
<header>
  <h1>Predicción de egresos UCI</h1>
  <p>Todos los modelos en una sola consulta: los del proyecto (salida temprana desde el día 1, estancia, ocupación) y los de UCI-IA (camino, días de estancia, salida día a día).</p>
</header>
<main>
  <section class="tarjeta" aria-labelledby="t-form">
    <h2 id="t-form">Datos del paciente al ingreso a UCI</h2>
    <form id="formulario" autocomplete="off">
      <div class="fila2">
        <div><label for="edad">Edad</label><input id="edad" name="edad" type="number" min="15" max="105" required></div>
        <div><label for="sexo">Sexo</label>
          <select id="sexo" name="sexo"><option value="M">Masculino</option><option value="F">Femenino</option></select></div>
      </div>
      <label for="diagnostico">Diagnóstico principal (CIE-10)</label>
      <input id="diagnostico" name="diagnostico" list="lista-dx" placeholder="Ej. I214 o escriba el diagnóstico" required>
      <datalist id="lista-dx"></datalist>
      <label for="tercero">Asegurador</label>
      <input id="tercero" name="tercero" list="lista-aseg" placeholder="Escriba o elija">
      <datalist id="lista-aseg"></datalist>
      <label for="servicio_inicial">Procedencia</label>
      <select id="servicio_inicial" name="servicio_inicial">
        <option value="UCI">Ingreso directo a UCI (urgencias, cirugía, remitido)</option>
        <option value="HOSPITALIZACION">Desde hospitalización (piso)</option>
      </select>
      <div class="fila2">
        <div><label for="sofa">SOFA inicial (0–24)</label><input id="sofa" name="puntuacion_sofa_inicial" type="number" min="0" max="24" required></div>
        <div><label for="barthel">Barthel inicial (0–100)</label><input id="barthel" name="barthel_inicial" type="number" min="0" max="100" step="5" required></div>
      </div>
      <label for="news">NEWS inicial <span class="op">(opcional)</span></label>
      <input id="news" name="news_inicial" type="number" min="0" max="20">
      <div class="fila2">
        <div><label for="fecha">Fecha de ingreso a UCI</label><input id="fecha" name="fecha_ingreso_uci" type="date" required></div>
        <div><label for="lleva">Días que lleva en UCI</label><input id="lleva" name="dias_en_uci" type="number" min="0" max="365" value="0"></div>
      </div>
      <button id="boton" type="submit">Predecir</button>
      <p class="regla">Sin edad, SOFA o Barthel no se da ninguna predicción: el modelo no asume datos del paciente.</p>
    </form>
  </section>
  <section id="resultados" aria-live="polite">
    <div class="tarjeta vacio">Llene los datos y presione <strong>Predecir</strong>. La primera consulta del día tarda unos segundos mientras se cargan los modelos.</div>
  </section>
</main>
<script>
const $ = (s, r = document) => r.querySelector(s);
const pct = v => v == null || isNaN(v) ? '—' : Math.round(v * 100) + ' %';
const el = (tag, attrs = {}, ...hijos) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === 'class' ? e.className = v : e.setAttribute(k, v);
  for (const h of hijos.flat()) if (h != null) e.append(h instanceof Node ? h : document.createTextNode(h));
  return e;
};
$('#fecha').value = new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);

fetch('?accion=catalogos').then(r => r.json()).then(c => {
  for (const a of c.aseguradoras || []) $('#lista-aseg').append(el('option', { value: a }));
  for (const d of c.diagnosticos || []) $('#lista-dx').append(el('option', { value: d.texto }));
}).catch(() => {});

function indicador(nombre, m, nota, tipoNivel = 'Posibilidad') {
  return el('div', { class: 'indicador' },
    el('div', { class: 'nombre' }, nombre),
    el('div', { class: 'valor' }, pct(m.prob), m.nivel ? el('span', { class: 'etiqueta' }, tipoNivel + ' ' + m.nivel) : null,
       m.planeacion ? el('span', { class: 'etiqueta' }, 'Solo planeación') : null),
    el('div', { class: 'detalle' }, (m.tasa_base != null ? `Tasa histórica ${pct(m.tasa_base)} · ` : '') + `AUC ${m.auc} (${m.ic95}) · ${m.modelo}`),
    nota ? el('div', { class: 'detalle' }, nota) : null);
}

function barras(datos) {
  return el('div', {}, ...Object.entries(datos).map(([k, v]) => el('div', { class: 'barra-h' },
    el('span', {}, k), el('div', { class: 'pista' }, el('div', { class: 'relleno', style: `width:${Math.max(v * 100, 0.5)}%` })),
    el('span', { class: 'num' }, pct(v)))));
}

function graficoCurva(curva) {
  const W = 700, H = 220, m = { l: 36, r: 8, t: 10, b: 38 };
  const maxV = Math.max(0.05, ...curva.map(c => c.sale));
  const tope = Math.ceil(maxV * 20) / 20;
  const ancho = (W - m.l - m.r) / curva.length, y = v => H - m.b - (v / tope) * (H - m.t - m.b);
  const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Probabilidad de salir de UCI cada día');
  const add = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt) e.textContent = txt; svg.append(e); return e; };
  for (let i = 0; i <= 4; i++) {
    const v = tope * i / 4;
    add('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: 'rejilla-linea' });
    add('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'eje' }, Math.round(v * 100) + '%');
  }
  const caja = el('div', { class: 'grafico' }), tip = el('div', { class: 'tooltip' });
  curva.forEach((c, i) => {
    const x = m.l + i * ancho, h = H - m.b - y(c.sale), bw = ancho - 2;
    add('path', { class: 'dato', fill: 'var(--series-1)', d: `M${x + 1},${H - m.b} v${-Math.max(h - 4, 0)} q0,-4 4,-4 h${bw - 8} q4,0 4,4 v${Math.max(h - 4, 0)} z` });
    const f = c.fecha.slice(8, 10) + '/' + c.fecha.slice(5, 7);
    add('text', { x: x + ancho / 2, y: H - m.b + 14, 'text-anchor': 'middle', class: 'eje' }, f);
    add('text', { x: x + ancho / 2, y: H - m.b + 28, 'text-anchor': 'middle', class: 'eje' }, c.dia.toUpperCase());
    const b = add('rect', { x, y: m.t, width: ancho, height: H - m.t - m.b + 30, class: 'blanco' });
    const mostrar = ev => {
      tip.style.display = 'block';
      tip.innerHTML = '';
      tip.append(el('strong', {}, c.fecha), el('br'), `Sale ese día: ${pct(c.sale)}`, el('br'), `Sigue en UCI al final: ${pct(c.sigue)}`);
      const r = caja.getBoundingClientRect();
      tip.style.left = Math.min(ev.clientX - r.left + 10, r.width - 170) + 'px'; tip.style.top = (ev.clientY - r.top - 50) + 'px';
    };
    b.addEventListener('mousemove', mostrar); b.addEventListener('mouseleave', () => tip.style.display = 'none');
  });
  caja.append(svg, tip);
  return caja;
}

function pintar(r) {
  const cont = $('#resultados'); cont.innerHTML = '';
  if (r.estado === 'incompleto') {
    cont.append(el('div', { class: 'tarjeta' }, el('div', { class: 'error' }, el('strong', {}, 'Sin predicción. '),
      'Falta: ' + r.faltan.join(', ') + '. ' + r.mensaje)));
    return;
  }
  if (r.estado !== 'ok') {
    cont.append(el('div', { class: 'tarjeta' }, el('div', { class: 'error' }, el('strong', {}, 'Error: '), r.mensaje || 'sin respuesta')));
    return;
  }
  const p = r.proyecto, u = r.uci_ia;

  cont.append(el('section', { class: 'tarjeta' }, el('h2', {}, 'Salida temprana desde el primer día de UCI'),
    el('div', { class: 'rejilla' },
      p.piso_3d ? indicador('Pasa a piso en 3 días o menos', p.piso_3d, 'Salida temprana con certeza viva: es la que lleva nivel de posibilidad.') : null,
      p.salida_uci_2d ? indicador('Sale de UCI en 2 días o menos', p.salida_uci_2d) : null,
      p.salida_uci_3d ? indicador('Sale de UCI en 3 días o menos', p.salida_uci_3d) : null),
    el('p', { class: 'nota' }, '“Sale de UCI” incluye salidas directas de la clínica, que pueden ser fallecimientos o remisiones (la base no trae la condición de egreso): solo sirve para planear camas.')));

  const est = el('section', { class: 'tarjeta' }, el('h2', {}, 'Estancia en UCI'));
  const izq = el('div', {}, el('h3', {}, 'Más de 7 días en UCI: dos estimaciones'), el('div', { class: 'rejilla' },
    indicador('Modelo del proyecto', p.uci_prolongada, null, 'Riesgo'),
    u ? indicador('Modelo UCI-IA', { prob: u.p_uci_prolongada, auc: u.auc_completos.uci_prolongada, ic95: 'con datos completos',
      tasa_base: null, modelo: 'gradient boosting' }) : null),
    el('p', { class: 'nota' }, 'Si difieren mucho, la incertidumbre es alta para este paciente.'));
  est.append(izq);
  if (u) {
    est.append(el('div', { class: 'dos', style: 'margin-top:16px' },
      el('div', {}, el('h3', {}, 'Días esperados'), el('table', {},
        el('tr', {}, el('th', {}, ''), el('th', {}, 'Mediana'), el('th', {}, 'Rango 80 %')),
        el('tr', {}, el('td', {}, 'En UCI'), el('td', {}, u.dias_uci[1]), el('td', {}, `${u.dias_uci[0]} a ${u.dias_uci[2]}`)),
        el('tr', {}, el('td', {}, 'Hospitalización total'), el('td', {}, u.dias_total[1]), el('td', {}, `${u.dias_total[0]} a ${u.dias_total[2]}`)),
        el('tr', {}, el('td', {}, 'Le faltan en UCI'), el('td', {}, u.dias_restantes), el('td', {}, `lleva ${r.dias_en_uci}`)))),
      el('div', {}, el('h3', {}, 'Probabilidad por rango de días en UCI'), barras(u.rangos_uci))));
  }
  cont.append(est);

  if (u) {
    cont.append(el('section', { class: 'tarjeta' }, el('h2', {}, 'Camino de salida'),
      el('div', { class: 'dos' },
        el('div', {}, el('h3', {}, 'UCI-IA (suma 100 %)'), barras(u.camino),
          el('p', { class: 'nota' }, `La opción más probable acierta ${pct(u.acierto_camino)} en la prueba. Reingreso a UCI: AUC ${u.auc_completos.rebote_uci} con datos completos.`)),
        el('div', {}, el('h3', {}, 'Proyecto'), el('div', { class: 'rejilla' },
          indicador('Sale de la clínica desde UCI', p.egreso_directo_uci))))));
  }

  if (u && u.curva_14_dias) {
    const tabla = el('table', {}, el('tr', {}, el('th', {}, 'Fecha'), el('th', {}, 'Sale ese día'), el('th', {}, 'Sigue en UCI')),
      ...u.curva_14_dias.map(c => el('tr', {}, el('td', {}, `${c.fecha} (${c.dia})`), el('td', {}, pct(c.sale)), el('td', {}, pct(c.sigue)))));
    cont.append(el('section', { class: 'tarjeta' }, el('h2', {}, 'Probabilidad de salir de UCI cada día (próximos 14 días)'),
      graficoCurva(u.curva_14_dias), el('details', {}, el('summary', {}, 'Ver tabla'), tabla),
      el('p', { class: 'nota' }, 'Modelo de egreso diario de UCI-IA: tiene en cuenta los días que lleva y el día de la semana (salen menos pacientes sábado y domingo).')));
  }

  if (p.ocupacion) {
    cont.append(el('section', { class: 'tarjeta' }, el('h2', {}, 'Dónde estará (modelo semi-Markov del proyecto)'),
      el('div', { class: 'tabla-scroll' }, el('table', {}, el('tr', {}, el('th', {}, ''), el('th', {}, 'En 1 día'), el('th', {}, 'En 3 días'), el('th', {}, 'En 7 días')),
        ...[['En UCI', 'uci'], ['En piso', 'piso'], ['Egresó de la clínica', 'egreso']].map(([n, k]) =>
          el('tr', {}, el('td', {}, n), ...['1d', '3d', '7d'].map(h => el('td', {}, pct(p.ocupacion[h][k])))))))));
  }

  const avisos = [...(r.avisos || []), 'Herramienta de apoyo para la gestión de camas: no reemplaza el criterio clínico. Con datos del día 1 ningún modelo predice con certeza (APACHE IV explica el 21,5 % de la variación individual).'];
  cont.append(el('section', { class: 'tarjeta' }, ...avisos.map(a => el('div', { class: 'aviso', style: 'margin-bottom:8px' }, a))));
}

$('#formulario').addEventListener('submit', async ev => {
  ev.preventDefault();
  const datos = Object.fromEntries(new FormData(ev.target));
  const boton = $('#boton'); boton.disabled = true; boton.textContent = 'Calculando…';
  try {
    const r = await fetch('?accion=predecir', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(datos) });
    pintar(await r.json());
  } catch (e) {
    pintar({ estado: 'error', mensaje: String(e) });
  } finally {
    boton.disabled = false; boton.textContent = 'Predecir';
  }
});

// index.php#demo llena un paciente de ejemplo (ficticio) y predice, para mostrar o probar la página
if (location.hash === '#demo') {
  const demo = { edad: 68, sexo: 'M', diagnostico: 'I214', tercero: 'NUEVA EMPRESA PROMOTORA DE SALUD SA',
    servicio_inicial: 'UCI', puntuacion_sofa_inicial: 3, barthel_inicial: 90, news_inicial: 4, dias_en_uci: 0 };
  for (const [k, v] of Object.entries(demo)) $(`[name="${k}"]`).value = v;
  $('#formulario').requestSubmit();
}
</script>
</body>
</html>
