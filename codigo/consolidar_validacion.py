"""Consolida resultados de pruebas nativas de PennyLane y resultados NumPy.
No inventa ejecuciones: exige CSV efectivamente generados para declarar VALIDADO.
Genera reporte JSON, tabla CSV y una figura cuantitativa reproducible.
"""
from pathlib import Path
import csv, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'datos'
F = ROOT/'figuras'
F.mkdir(exist_ok=True)


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def numeric(rows, key):
    arr = np.array([float(row[key]) for row in rows], dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f'Valores no finitos en {key}')
    return arr


def main():
    inputs = {
        'jw': D/'validacion_pennylane.csv',
        'circuit': D/'comparacion_circuitos_pennylane.csv',
        'shots': D/'07_pennylane_circuitos_y_shots.csv',
    }
    missing = [p.name for p in inputs.values() if not p.is_file()]
    base = D/'resumen.json'
    raw = json.loads(base.read_text(encoding='utf8')) if base.is_file() else {}
    consolidated = {
        'validated': False,
        'source': 'CSV generados por scripts nativos de PennyLane, no cifras supuestas',
        'missing_files': missing,
        'base_model': raw.get('model', 'tres/cuatro modos, consultar generadores'),
    }
    if missing:
        (D/'resumen_validacion_final.json').write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding='utf8')
        print('Validación PennyLane INCOMPLETA; faltan:', ', '.join(missing))
        return 1

    jw, circ, shots = (load_csv(inputs[k]) for k in ('jw','circuit','shots'))
    if len(jw) != 18 or len(circ) != 48 or len(shots) != 14:
        raise ValueError(f'Conteos inesperados: JW={len(jw)}, circuitos={len(circ)}, shots={len(shots)}')
    e_jw = numeric(jw, 'error_Frobenius')
    e_circuit = numeric(circ,'error_PennyLane_vs_Trotter_numpy')
    tv_exact = numeric(circ,'distancia_total_vs_exacta')
    analytic = [r for r in shots if r['mode']=='analytic']
    sampled = [r for r in shots if r['mode']=='finite_shots']
    e_extra = numeric(analytic,'max_probability_error')
    e_sampled = numeric(sampled,'max_probability_error')
    if any(v > 1e-9 for v in (e_jw.max(), e_circuit.max(), e_extra.max())):
        raise AssertionError('La validación PennyLane no pasó la tolerancia numérica 1e-9')

    consolidated.update({
        'validated':True,
        'missing_files':[],
        'jw_cases':len(jw),
        'circuit_cases':len(circ),
        'shot_cases':len(sampled),
        'jw_max_error_frobenius':float(e_jw.max()),
        'circuit_max_probability_error':float(e_circuit.max()),
        'circuit_extra_max_probability_error':float(e_extra.max()),
        'max_total_variation_vs_exact_digital':float(tv_exact.max()),
        'shot_max_probability_deviation':float(e_sampled.max()),
        'note':'El muestreo con shots muestra fluctuación estadística; no representa ruido de hardware físico.',
        'note2':'La comparación con Trotter matricial verifica la implementación del circuito; la distancia frente a evolución exacta mide el error algorítmico, que no tiene por qué ser cero.'
    })
    (D/'resumen_validacion_final.json').write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding='utf8')

    with (D/'tabla_validacion_final.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.writer(f)
        wr.writerow(['prueba','casos','error_maximo','umbral','resultado'])
        wr.writerow(['Jordan-Wigner nativo',len(jw),float(e_jw.max()),1e-9,'APROBADO'])
        wr.writerow(['PennyLane vs. Trotter matricial',len(circ),float(e_circuit.max()),1e-9,'APROBADO'])
        wr.writerow(['Circuitos adicionales',len(analytic),float(e_extra.max()),1e-9,'APROBADO'])
        wr.writerow(['Muestreo shots (variabilidad, NO criterio de fallo)',len(sampled),float(e_sampled.max()),'', 'INFORMATIVO'])

    # Figura principal: error algorítmico vs número de pasos (PennyLane real)
    colors = {'first':'#3274C8','second':'#E36B55'}
    fig,ax=plt.subplots(1,2,figsize=(11.7,4.35),layout='constrained')
    for order,color,marker in ((1,colors['first'],'o'),(2,colors['second'],'s')):
        rows=[r for r in circ if int(r['orden'])==order and abs(float(r['tiempo'])-2.0)<1e-9]
        rows.sort(key=lambda r:int(r['pasos']))
        x=np.array([int(r['pasos']) for r in rows]); y=np.array([float(r['distancia_total_vs_exacta']) for r in rows]);
        ax[0].loglog(x,np.maximum(y,1e-16),marker=marker,color=color,lw=2.2,label=f'Trotter orden {order}')
        ax[1].semilogx(x,np.array([float(r['error_PennyLane_vs_Trotter_numpy']) for r in rows]),marker=marker,color=color,lw=2.2,label=f'Orden {order}')
    ax[0].set(title='(a) Error algorítmico frente a evolución exacta',xlabel='Pasos de Trotter',ylabel='Distancia de variación total')
    ax[1].set(title='(b) Reproducibilidad del circuito',xlabel='Pasos de Trotter',ylabel='Error máximo de probabilidad')
    for a in ax:
        a.grid(alpha=.15,which='both'); a.legend(frameon=False,fontsize=9)
        a.spines[['top','right']].set_visible(False)
    fig.suptitle('Validación independiente con PennyLane · N=3, U/t=2, τ=2',fontweight='bold',fontsize=13)
    fig.savefig(F/'14_validacion_pennylane_real.png',dpi=240,bbox_inches='tight');plt.close(fig)

    # Honest shot results: no automatic 1/sqrt shots claim based on only 1 sample per N.
    fig,ax=plt.subplots(figsize=(7.7,4.4),layout='constrained')
    for order,color,marker in ((1,colors['first'],'o'),(2,colors['second'],'s')):
        filtered=sorted([r for r in sampled if int(r['order'])==order],key=lambda r:int(r['shots']))
        ax.loglog([int(r['shots']) for r in filtered],[max(float(r['max_probability_error']),1e-15) for r in filtered],marker=marker,ms=8,lw=1.7,color=color,label=f'Orden {order}')
    ax.set(title='Muestreo finito de circuitos (una realización por configuración)',xlabel='Número de mediciones (shots)',ylabel='Desviación máxima de probabilidad')
    ax.grid(alpha=.15,which='both');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(F/'15_variacion_shots_pennylane.png',dpi=240,bbox_inches='tight');plt.close(fig)
    print('PennyLane VALIDADO según CSV: JW=',len(jw), 'circuitos=',len(circ),'shots=',len(sampled))
    print('Error máximo JW:',e_jw.max(),'error máximo circuitos:',e_circuit.max())
    return 0

if __name__=='__main__':
    raise SystemExit(main())
