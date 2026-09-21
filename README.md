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

## Estructura del proyecto

| Archivo | Responsabilidad |
|---|---|
| `environment.py` | Define `Environment2D`, el gradiente de olor y el agente `FlyAgent` con antenas y cinemática diferencial. |
| `extract_circuit.py` | Consulta Hemibrain/NeuPrint mediante Navis, descubre rutas PN -> DN y exporta conectividad real. Usa una reserva sintética solo si faltan credenciales o datos. |
| `brain_snn.py` | Implementa `BrainSNN`: neuronas LIF de Brian2, sinapsis ponderadas, monitores de voltaje/spikes y la API `step`. |
| `main.py` | Ejecuta el lazo cerrado olor -> SNN -> DNs -> giro, y muestra trayectoria, raster y actividad motora. |
| `circuit_data.json` | Subcircuito exportado: IDs biológicos, tipos neuronales, lateralidad y pesos de sinapsis. |
| `tokens.json` | Token local de NeuPrint; es opcional y está ignorado por Git. |

## Instalación

Los comandos siguientes están escritos para PowerShell en Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "numpy<2" brian2 navis neuprint-python matplotlib pandas
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
python extract_circuit.py
```

El exportador también acepta `NEUPRINT_AUTH_TOKEN` o `HEMIBRAIN_TOKEN` como
variables de entorno. Si no se configura un token, genera un circuito de
reserva marcado explícitamente como sintético y `BrainSNN` no lo aceptará como
datos biológicos reales.

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

La ventana de Matplotlib contiene:

1. El gradiente de olor, la fuente de alimento y la trayectoria completa.
2. El raster de spikes de la SNN.
3. Los spikes por paso de las DNs izquierda y derecha.

## Métricas del experimento

Con el circuito real actual, comida en `(4.0, 2.0)`, `k_sensor = 1200` y
`k_motor = 0.04`, la ejecución de 200 pasos produjo:

| Métrica | Resultado |
|---|---:|
| Distancia inicial a la comida | 4.47 unidades |
| Distancia final a la comida | 0.59 unidades |
| Spikes DN izquierda / derecha | 34 / 20 |
| Pasos con giro no nulo | 48 |

La asimetría emergente de actividad de las DNs produce una velocidad angular
suave y una trayectoria curvada hacia la fuente de olor.

## Procedencia y reproducibilidad

`circuit_data.json` registra su procedencia en `metadata.source` y declara
`metadata.weights`. Antes de interpretar resultados, confirme que los valores
sean `hemibrain_neuprint_via_navis` y `real_synapse_counts`, respectivamente.
Los conteos sinápticos son conectividad anatómica; el modelo LIF usa una escala
de corriente para convertirlos en dinámica simulable, no infiere plasticidad ni
aprendizaje.
