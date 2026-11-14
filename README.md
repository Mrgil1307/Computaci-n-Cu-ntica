# Operadores fermiónicos: Jordan–Wigner, simulación y validación

Proyecto reproducible de Computación Cuántica. Construido a partir del alcance del avance: validación del mapeo Jordan–Wigner para 2 y 3 modos, con PennyLane como herramienta de comparación y extensiones físicas de 4 modos. **No se reportan pruebas PennyLane como ejecutadas en el entorno de creación** porque no está instalada aquí.

## Instalación

```bash
python -m pip install -r requirements.txt
python codigo/ejecutar_todo.py
```

Para ejecutar únicamente NumPy/SciPy y las figuras:

```bash
python codigo/generar_base.py
python codigo/generar_avanzados.py
python -m pytest -q tests
```

Para ejecutar PennyLane (cuando esté instalado):

```bash
python codigo/pennylane_real.py
python codigo/pennylane_extendido.py
```

No se necesita un computador cuántico físico. Se utiliza `default.qubit`; las puertas `PauliRot` forman la aproximación digital, no un `QubitUnitary` global. La instalación necesita acceso a paquetes PyPI en un equipo con conexión.

## Organización

- `codigo/nucleo.py`: construcción Jordan–Wigner, CAR, Hamiltonianos, Trotter, descomposición de Pauli.
- `codigo/generar_base.py`: figuras 01–05 y tablas de la validación y Trotter del sistema de 3 modos.
- `codigo/avanzado.py`: sectorización, espectros, propagación exacta, correlaciones y entropía de modos.
- `codigo/generar_avanzados.py`: figuras 07–12 y tablas de estudios con 4 modos.
- `codigo/pennylane_real.py`: mapeo fermión–qubit nativo con `FermiWord`/`jordan_wigner` y QNodes que miden probabilidades, comparando con NumPy.
- `codigo/pennylane_extendido.py`: circuitos de 3 modos, ocupaciones, tiros finitos y dibujo de un circuito. Se ejecuta cuando PennyLane esté disponible.
- `codigo/ejecutar_todo.py`: genera resultados y ejecuta tests; si PennyLane no está disponible, lo indica y no la declara validada.
- `notebooks/proyecto_integrado.ipynb`: cuaderno de trabajo con explicación física y ejecución por etapas.
- `datos`: archivos CSV sin inventar resultados y archivo JSON de resumen.
- `figuras`: PNG de alta resolución y PDF vectorial para uso en LaTeX.
- `tests`: pruebas reproducibles.

## Convenciones y ecuaciones

- Base de ocupación: `|n0 n1 ... nN-1>` y qubit 0 a la izquierda del producto tensorial.
- `hbar = 1`, escala de hopping `t = 1`, cadena abierta.
- `a_j = (prod_{k<j} Z_k) (X_j + iY_j)/2`.
- `H = -t sum_{i=0}^{N-2}(a†_i a_{i+1} + a†_{i+1} a_i) + U sum_{i=0}^{N-2} n_i n_{i+1}`.
- Modelo físico principal adicional: `N=4`, dos fermiones, estado inicial `|1100>`, `U={0,1,2,4,6}`, `tau in [0,8]` con 161 muestras.
- La probabilidad analizada corresponde a `|1001>`. Se informa su **máximo en el intervalo simulado**, no el máximo absoluto a todos los tiempos.
- Correlación de una partícula: `G_ij=<a†_i a_j>`; covarianza: `<n_i n_j>-<n_i><n_j>`.
- Entropía de von Neumann de una partición de **modos** en la base de ocupación (`01|23`). Es diferente del entrelazamiento entre partículas indistinguibles.

## Resultados efectivamente verificados

Los códigos NumPy/SciPy pasaron **11 pruebas automáticas**. La mayor desviación de conservación, de hermiticidad o de conmutadores en el barrido de interacciones de 4 modos fue `4.4465e-15`. Para `U=2`, `tau=2`, ocho pasos de Trotter y estado `|1100>`:

| Método | Infidelidad 1-F | CNOT estimadas |
|---|---:|---:|
| Trotter 1er orden | 0.2823457131 | 144 |
| Trotter 2o orden | 0.0040520290 | 288 |

Los CNOT se estiman con la descomposición estándar de cada cadena de Pauli mediante escalera de CNOT, sin optimización por transpiler, sin penalización por conectividad y sin incluir ruido real. La mejora de precisión depende del Hamiltoniano, tiempo y cantidad de pasos.

### Qué falta verificar en una instalación con PennyLane

Ejecutar `codigo/pennylane_real.py` y `codigo/pennylane_extendido.py`, inspeccionar los CSV `validacion_pennylane.csv`, `comparacion_circuitos_pennylane.csv`, `07_pennylane_circuitos_y_shots.csv` y comprobar que los errores NumPy–PennyLane pasan sus tolerancias. Cualquier incompatibilidad de API deberá corregirse en la versión instalada antes de presentar esos datos como resultados científicos.

## Interpretación para el artículo

1. Las CAR prueban la corrección algebraica de Jordan–Wigner en los casos calculados.
2. El experimento sin paridad muestra que una asignación local ingenua viola signos fermiónicos.
3. El modelo físico de dos fermiones muestra la redistribución de probabilidad por interacción.
4. Los espectros por sector separan cambios de energía y número de partículas.
5. La covarianza de ocupación, la coherencia de una partícula y la entropía de modos son observables diferentes; no deben confundirse.
6. Trotter compara precisión frente a recursos de circuito; la validación de PennyLane permitirá relacionarlo con implementación en qubits.

No se han utilizado resultados de hardware cuántico real ni se ha demostrado ventaja cuántica.

## Consolidación final PennyLane (nuevo)

El script `codigo/ejecutar_todo.py` ahora, después de ejecutar las pruebas nativas, llama a `codigo/consolidar_validacion.py`. Este programa lee los CSV de PennyLane, valida sus valores y produce la tabla de verificación y dos figuras nuevas. Para evitar reportar un resultado no ejecutado, los informes base de NumPy no deben tomarse como veredicto final sobre PennyLane. Lee `VALIDACION_PENNYLANE.md` y `datos/resumen_validacion_final.json`.
