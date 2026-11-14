# Validación definitiva de Jordan–Wigner y los circuitos

## Resultado ya comprobado en el computador del estudiante

Registro de terminal comunicado en la conversación (PennyLane 0.45.1):

- 11 pruebas automáticas: superadas.
- Error máximo del mapeo Jordan–Wigner comparado con NumPy: **0.0**.
- Error máximo de probabilidad del circuito frente al producto matricial de Pauli: **2.4424906541753444e−15**.
- Módulo adicional de PennyLane: **14 filas** generadas sin fallo.

Estas cifras son del registro de ejecución aportado por el usuario, no de una ejecución nativa en el entorno de creación del paquete.

## Corrección de informes

El archivo `datos/resumen.json` es el resumen de los generadores clásicos y, por diseño heredado, contiene un campo `pennylane_executed: false` o `pennylane_status: not executed`. Eso no refleja el resultado posterior de los módulos nativos. **No debe usarse como evidencia definitiva**.

El nuevo programa `codigo/consolidar_validacion.py` verifica la existencia e integridad de tres CSV **reales**, calcula los errores máximos y genera:

- `datos/resumen_validacion_final.json` — estado unificado de PennyLane.
- `datos/tabla_validacion_final.csv` — tabla de pruebas y errores.
- `figuras/14_validacion_pennylane_real.png` — comparación exacta / circuito por orden y pasos.
- `figuras/15_variacion_shots_pennylane.png` — efecto del muestreo finito (una realización por configuración).

**Los resultados marcados como verificados proceden solo de CSV generados durante una ejecución real.**

## Ejecutar en Windows

Desde la carpeta que contiene `codigo`, `datos`, `figuras`, `tests`:

```powershell
python codigo/ejecutar_todo.py
```

Si ya tienes los tres CSV generados y solo quieres consolidar:

```powershell
python codigo/consolidar_validacion.py
```

## Qué podemos afirmar científicamente

- La comparación entre mapeos y matrices demuestra consistencia numérica en los casos evaluados; el argumento algebraico general sigue siendo necesario.
- La concordancia entre circuito y Trotter matricial es una **validación de implementación**; la diferencia entre Trotter y evolución exacta mide el error de aproximación algorítmica.
- Los circuitos son simulados por `default.qubit`, no ejecutados sobre un procesador cuántico real.
- El resultado de shots mide fluctuación estadística de mediciones ideales; **no es ruido experimental del hardware**.
