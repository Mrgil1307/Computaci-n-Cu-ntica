"""Validación adicional de PennyLane: ocupaciones, muestreo de circuitos y recursos.
Se ejecuta sólo cuando PennyLane está instalado. Todas las salidas se etiquetan como
PennyLane únicamente si esta ejecución termina sin errores.
"""
import csv
from pathlib import Path
import numpy as np
import pennylane as qml
from nucleo import hamiltonian,basis_state,pauli_decompose,evolve_trotter,evolve_exact
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'datos';FIG=ROOT/'figuras'
def make_circuit(bits,terms,steps,order,shots=None):
    n=len(bits);dev=qml.device('default.qubit',wires=n,shots=shots)
    active=[(w,c) for w,c in terms if any(q!='I' for q in w)]
    @qml.qnode(dev)
    def circuit(t):
        qml.BasisState(np.asarray(bits,dtype=int),wires=range(n))
        dt=t/steps
        for _ in range(steps):
            if order==1:
                for w,c in active:qml.PauliRot(2*c*dt,w,wires=range(n))
            else:
                for w,c in active:qml.PauliRot(c*dt,w,wires=range(n))
                for w,c in reversed(active):qml.PauliRot(c*dt,w,wires=range(n))
        return qml.probs(wires=range(n))
    return circuit

def main():
    DATA.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    n=3;bits=[1,1,0];t=2.;u=2.;steps=8
    h=hamiltonian(n,u);terms=pauli_decompose(h);initial=basis_state(bits)
    rows=[]
    for order in (1,2):
        for s in (2,4,8,16):
            c=make_circuit(bits,terms,s,order)
            q=np.asarray(c(t),float);matrix=np.abs(evolve_trotter(terms,initial,t,s,order))**2
            exact=np.abs(evolve_exact(h,initial,t))**2
            err=float(np.max(abs(q-matrix)))
            if err>1e-9:raise AssertionError(f'PennyLane vs. producto Pauli manual: {err}')
            means=[float(sum(q[k]*int(format(k,f'0{n}b')[j]) for k in range(2**n))) for j in range(n)]
            rows.append(['analytic',order,s,0,err,float(sum(abs(q-exact))/2),*means])
    # Reproducible shot-sampling with the simulator: interpretation differs from quantum-hardware noise.
    for shots in (100,1000,10000):
        for order in (1,2):
            circuit=make_circuit(bits,terms,steps,order,shots=shots)
            result=np.asarray(circuit(t),float)
            reference=np.abs(evolve_trotter(terms,initial,t,steps,order))**2
            means=[float(sum(result[k]*int(format(k,f'0{n}b')[j]) for k in range(2**n))) for j in range(n)]
            rows.append(['finite_shots',order,steps,shots,float(max(abs(result-reference))),float(sum(abs(result-reference))/2),*means])
    with (DATA/'07_pennylane_circuitos_y_shots.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['mode','order','steps','shots','max_probability_error','total_variation','n0','n1','n2']);w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,3))
    qml.draw_mpl(make_circuit(bits,terms,1,2))(1.)
    drawn=plt.gcf();drawn.savefig(FIG/'13_circuito_pennylane.png',dpi=200,bbox_inches='tight');drawn.savefig(FIG/'13_circuito_pennylane.pdf',bbox_inches='tight');plt.close(drawn);plt.close(fig)
    print('PennyLane validado:',qml.__version__,'filas:',len(rows))
if __name__=='__main__':main()
