"""Comprobación nativa y QNode con puertas PauliRot reales en PennyLane.
Uso: python codigo/pennylane_real.py. Requiere pennylane instalado.
No se emplea QubitUnitary ni se usa el resultado de SciPy como puerta.
"""
import csv
from pathlib import Path
import numpy as np
import pennylane as qml
from nucleo import (operators,hamiltonian,pauli_decompose,basis_state,evolve_exact,
                    evolve_trotter,observables,fidelity)
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datos'

def native_jw_tests():
    rows=[]
    for n in (2,3,4):
        a,ad=operators(n)
        for mode in range(n):
            for dagger,reference in ((False,a[mode]),(True,ad[mode])):
                sign='+' if dagger else '-'
                word=qml.FermiWord({(0,mode):sign})
                mapped=qml.jordan_wigner(word)
                matrix=np.asarray(qml.matrix(mapped,wire_order=list(range(n))),complex)
                err=float(np.linalg.norm(matrix-reference))
                rows.append([n,mode,'creación' if dagger else 'aniquilación',err])
                if err>1e-10:raise AssertionError(f'Fallo JW: {(n,mode,sign,err)}')
    with (DATA/'validacion_pennylane.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.writer(f);wr.writerow(['N','modo','operador','error_Frobenius']);wr.writerows(rows)
    return max(r[-1] for r in rows)

def make_qnode(n,bits,terms,steps,order):
    dev=qml.device('default.qubit',wires=n,shots=None)
    active=[(w,c) for w,c in terms if set(w)!={'I'}]
    @qml.qnode(dev)
    def circuit(total_time):
        qml.BasisState(np.asarray(bits,dtype=int),wires=range(n))
        dt=total_time/steps
        for _ in range(steps):
            if order==1:
                for word,c in active:qml.PauliRot(2*c*dt,word,wires=range(n))
            else:
                for word,c in active:qml.PauliRot(c*dt,word,wires=range(n))
                for word,c in reversed(active):qml.PauliRot(c*dt,word,wires=range(n))
        return qml.probs(wires=range(n))
    return circuit

def main():
    DATA.mkdir(exist_ok=True)
    n=3;u=2.;bits=[1,1,0];initial=basis_state(bits)
    H=hamiltonian(n,u);terms=pauli_decompose(H)
    error_jw=native_jw_tests();rows=[]
    for order in (1,2):
        for steps in (1,2,4,8,16,32):
            circuit=make_qnode(n,bits,terms,steps,order)
            for t in (0.5,1.,2.,3.):
                probs=np.asarray(circuit(t),float)
                exact=evolve_exact(H,initial,t)
                trotter=evolve_trotter(terms,initial,t,steps,order)
                target=abs(trotter)**2
                e=float(np.max(abs(probs-target)))
                if e>1e-9:raise AssertionError(f'Diferencia circuito-PauliRot vs NumPy: {e}')
                classical=np.sum(abs(probs-abs(exact)**2))/2
                rows.append([order,steps,t,e,classical])
    with (DATA/'comparacion_circuitos_pennylane.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.writer(f);wr.writerow(['orden','pasos','tiempo','error_PennyLane_vs_Trotter_numpy','distancia_total_vs_exacta']);wr.writerows(rows)
    print('PennyLane version:',qml.__version__,'\nError máximo JW:',error_jw,'\nError circuito:',max(x[3] for x in rows))
    print('Pruebas nativas completadas. Datos guardados.')

if __name__=='__main__':main()
