# Mosca v2: emulación biofísica cerrada de *Drosophila*

Mosca v2 es una simulación de navegación olfativa en lazo cerrado inspirada en
*Drosophila melanogaster*. Conecta un entorno 2D, señales sensoriales de dos
antenas y una Spiking Neural Network (SNN) implementada con Brian2. La
conectividad se extrae de Hemibrain/NeuPrint mediante Navis y conserva los
conteos reales de sinapsis como pesos relativos.

No utiliza Deep Learning, PyTorch, TensorFlow ni Aprendizaje por Refuerzo. El
comportamiento emerge de la dinámica LIF, la conectividad biológica extraída y
el acoplamiento sensorimotor explícito.

## Arquitectura del circuito

```text
Environment2D (fuente de alimento y gradiente de olor)
       |
       +--> antena izquierda / antena derecha de FlyAgent
                    |
                    v
       corrientes proporcionales de entrada a PN_L / PN_R
                    |
                    v
 Projection Neurons (PN) -> interneuronas / Central Complex (E-PG, P-EG)
                    |
                    v
 Descending Neurons (DN) izquierda / derecha
                    |
                    v
 omega = k_motor * (spikes_DN_left - spikes_DN_right)
                    |
                    v
 FlyAgent.step(v, omega) -> nueva pose -> nueva lectura antenal
```

`extract_circuit.py` amplía el subcircuito con dos rutas reales y continuas
PN -> interneuronas -> DN de Hemibrain. Los nodos E-PG/P-EG representan el
componente de Cuerpo Central incluido en el subcircuito. Los pesos son conteos
de sinapsis reales, escalados proporcionalmente como corrientes sinápticas por
`BrainSNN`.

## Módulo de Cuerpos Pedunculados

La Fase 6A incorpora un módulo asociativo extraído de Hemibrain para preparar
la futura modulación dopaminérgica:

```text
PN -> Kenyon Cell (KC) -> Mushroom Body Output Neuron (MBON)
                                  ^
                                  |
                         DAN (PAM/PPL)
```

Para cada canal lateral, el exportador selecciona la conexión directa de mayor
peso PN -> KC, la salida KC -> MBON de mayor peso y la proyección PAM/PPL más
fuerte hacia ese MBON. En la topología actual esto añade 2 KCs, 2 MBONs, 2
DANs y 6 sinapsis reales. Las sinapsis KC -> MBON son plásticas: un pulso de
recompensa activa las DANs y abre una traza dopaminérgica de 100 ms. Cada spike
KC dentro de esa ventana de elegibilidad deprime el peso KC -> MBON un 2 %,
con límite inferior cero. El resto de la conectividad conserva los pesos
anatómicos fijos.

## Estructura del proyecto

| Archivo | Responsabilidad |
|---|---|
| `environment.py` | Define `Environment2D`, el gradiente de olor y el agente `FlyAgent` con antenas y cinemática diferencial. |
| `extract_circuit.py` | Consulta Hemibrain/NeuPrint mediante Navis, descubre rutas PN -> DN y PN -> KC -> MBON con proyecciones DAN, y exporta conectividad real más un esqueleto SWC por neurona. Usa una reserva sintética solo si faltan credenciales o datos. |
| `brain_snn.py` | Implementa `BrainSNN`: neuronas LIF de Brian2, sinapsis ponderadas, monitores de voltaje/spikes y la API `step`. |
| `main.py` | Ejecuta el lazo cerrado olor -> SNN -> DNs -> giro, y muestra trayectoria, raster y actividad motora. |
| `dashboard_3d.py` | Renderiza el entorno 2D, las morfologías 3D reales y la telemetría de la simulación en un panel Plotly. |
| `circuit_data.json` | Subcircuito exportado: IDs biológicos, tipos neuronales, lateralidad y pesos de sinapsis. |
| `tokens.json` | Token local de NeuPrint; es opcional y está ignorado por Git. |

## Instalación

Los comandos siguientes están escritos para PowerShell en Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "numpy<2" brian2 navis neuprint-python matplotlib pandas plotly streamlit
```

`numpy<2` mantiene compatibilidad con la versión actual de Brian2. Para
regenerar conectividad real de Hemibrain, cree un archivo local ignorado
`tokens.json`:

```json
{
  "neuprint_token": "TU_TOKEN_DE_NEUPRINT"
}
```

Después ejecute:

```powershell
python extract_circuit.py --skeleton-dir neuron_skeletons
```

El exportador también acepta `NEUPRINT_AUTH_TOKEN` o `HEMIBRAIN_TOKEN` como
variables de entorno. Si no se configura un token, genera un circuito de
reserva marcado explícitamente como sintético y `BrainSNN` no lo aceptará como
datos biológicos reales.

Con credenciales válidas, la extracción descarga los 18 esqueletos reales con
`navis.interfaces.neuprint.fetch_skeletons` y los guarda como SWC en
`neuron_skeletons/`, más un `manifest.json`. Las coordenadas están en
nanómetros y el dashboard las utiliza directamente.

## Ejecución

Con el entorno virtual activo y un `circuit_data.json` con
`"weights": "real_synapse_counts"`:

```powershell
python main.py
```

La simulación predeterminada ejecuta 200 pasos de 5 ms. Para modificar su
duración:

```powershell
python main.py --steps 400 --dt-ms 5
```

La ejecución imprime, para cada paso, las concentraciones de ambas antenas y
su diferencia, corrientes inyectadas a `PN_L`/`PN_R`, spikes de DNs y el valor
final de `omega`. Para suprimir esta telemetría:

```powershell
python main.py --quiet
```

La fuente de alimento es configurable en cada ejecución:

```powershell
python main.py --odor-a-pos -2 -2 --odor-b-pos 2 -2
```

### Experimento Pavloviano multi-olor

El protocolo usa dos fuentes a distancia equitativa: **Olor A (CS-)** se
proyecta principalmente a `PN_L`, mientras que **Olor B (CS+)** se proyecta
principalmente a `PN_R`. Durante entrenamiento, alcanzar Olor B a menos de
`--reward-distance` activa las DANs y guarda los pesos KC→MBON aprendidos:

```powershell
python main.py --protocol training --weights-file trained_weights.json --odor-a-pos -2 -2 --odor-b-pos 2 -2 --telemetry training.jsonl
```

La prueba restaura esos pesos, mantiene ambos olores equidistantes y no activa
dopamina. La ruta autónoma recibe el sesgo de memoria asociado al CS+:

```powershell
python main.py --protocol testing --weights-file trained_weights.json --odor-a-pos -2 -2 --odor-b-pos 2 -2 --telemetry testing.jsonl
streamlit run analyzer_dashboard.py -- --telemetry testing.jsonl
```

La pestaña **Experimento Pavloviano** muestra el Índice de Aprendizaje
`(tiempo cerca de Olor B - tiempo cerca de Olor A) / tiempo total`, la
trayectoria de preferencia y la matriz KC→MBON pre/post-entrenamiento.

## Dashboard 3D

`main.py` puede transmitir una muestra JSONL completa y *flush-safe* en cada
paso `dt`. Cada muestra contiene pose, posición de comida, corrientes PN en
pA, spikes por neurona y de DN, y velocidad angular:

```powershell
python main.py --telemetry telemetry.jsonl
python dashboard_3d.py --telemetry telemetry.jsonl --show
```

El dashboard genera `dashboard_3d.html` con tres sub-vistas: entorno y
trayectoria 2D, los 18 esqueletos 3D reales (amarillo para una neurona con
spike en la muestra más reciente) y las series de corrientes, diferencial de
spikes DN y velocidad angular. Para monitorizar una ejecución larga, inicie
`dashboard_3d.py --watch` en otra terminal y refresque el HTML en el
navegador; el archivo se reemplaza atómicamente tras cada actualización.

Al entrar en el radio de recompensa (`--reward-distance`, por defecto `0.6`),
`main.py` inyecta `--dopamine-current-pa` (por defecto `700 pA`) en las DANs
durante el paso actual. La telemetría incorpora `learning` con distancia,
refuerzo, corriente dopaminérgica, spikes DAN y peso medio KC -> MBON. El
dashboard expresa este último como porcentaje de su valor inicial y marca cada
evento de recompensa.

## Suite de Analítica e Inspección

Para auditar una ejecución terminada o un `telemetry.jsonl` que se esté
actualizando, ejecute:

```powershell
streamlit run analyzer_dashboard.py -- --telemetry telemetry.jsonl
```

La aplicación Streamlit presenta métricas ejecutivas de recompensas, cambio de
peso KC -> MBON, distancia mínima y eficiencia de trayectoria. Sus pestañas
permiten inspeccionar la plasticidad con eventos DAN, el raster por rol
biológico y corrientes/omega/spikes DN, y un playback temporal del entorno 2D
con la red 3D basada en los SWC de `neuron_skeletons/`.

El selector lateral **Modo de Mapeo Cerebral 3D** ofrece cuatro lecturas de la
misma muestra temporal: spikes binarios (amarillo), gradiente Viridis de
potencial de membrana V_m/GCaMP, luminancia y opacidad por identidad de capa
según la tasa instantánea, y mapa Plasma de plasticidad KC -> MBON. Este último
sitúa marcadores entre las morfologías KC y MBON, escalados y coloreados por el
peso sináptico como porcentaje del peso inicial.

La ventana de Matplotlib contiene:

1. El gradiente de olor, la fuente de alimento y la trayectoria completa.
2. El raster de spikes de la SNN.
3. Los spikes por paso de las DNs izquierda y derecha.

## Métricas del experimento

La conversión sensorial usa `k_sensor = 450 pA/concentración`, un sesgo de
`125 pA` y un máximo de `350 pA` por PN. Esto conserva el contraste entre
antenas sin llevar a las PNs a un régimen de disparo saturado. La separación
de cada antena al centro de la mosca es `0.5` unidades.

Con el circuito real actual y `k_motor = 0.02`, las ejecuciones de 200 pasos
produjeron:

| Posición de comida | Distancia inicial -> final | Giro acumulado | Spikes DN izquierda / derecha |
|---|---:|
| Arriba a la derecha `(2, 2)` | 2.83 -> 1.11 | +1.29 rad | 30 / 0 |
| Abajo a la izquierda `(-2, -2)` | 2.83 -> 1.54 | -2.82 rad | 22 / 0 |

La asimetría emergente de las DNs controla el giro cuando aporta un diferencial
lateral. Si ambas DNs están silenciosas en un paso, el controlador usa
temporalmente la actividad de PNs —spikes y corriente subumbral normalizada—
para no perder el contraste olfativo.

Las rutas anatómicas PN→DN izquierda y derecha tienen números de sinapsis e
intermediarios distintos. `circuit_data.json` conserva esos conteos reales y
declara una `simulation_gain` de lectura motora para la sinapsis final de DN_R;
la ganancia equilibra el disparo bilateral bajo entrada PN idéntica sin alterar
la conectividad anatómica exportada. La telemetría incluye tanto
`dn_spikes` por lateralidad como `dn_spikes_by_id` para auditar cada DN.

## Procedencia y reproducibilidad

`circuit_data.json` registra su procedencia en `metadata.source` y declara
`metadata.weights`. Antes de interpretar resultados, confirme que los valores
sean `hemibrain_neuprint_via_navis` y `real_synapse_counts`, respectivamente.
Los conteos sinápticos son conectividad anatómica; el modelo LIF usa una escala
de corriente para convertirlos en dinámica simulable, no infiere plasticidad ni
aprendizaje.
