import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codigo'))
import numpy as np
from nucleo import *

def test_car():
    for n in (2,3,4): assert car_error(n).max()<1e-12

def test_control_wrong(): assert car_error(3,False).max()>0.1

def test_decomposition():
    for n in (2,3):
        h=hamiltonian(n,2);terms=pauli_decompose(h)
        assert np.linalg.norm(sum((c*pauli_matrix(w) for w,c in terms),np.zeros_like(h))-h)<1e-12

def test_trotter_and_probability():
    h=hamiltonian(3,2);terms=pauli_decompose(h);psi=basis_state([1,1,0])
    for order in (1,2):
        st=evolve_trotter(terms,psi,1.4,16,order)
        assert abs(np.linalg.norm(st)-1)<1e-11
        assert abs(observables(st,3)[1].sum()-2)<1e-11

def test_convergence():
    h=hamiltonian(3,2);terms=pauli_decompose(h);psi=basis_state([1,1,0]);ex=evolve_exact(h,psi,2.)
    for order in (1,2):
        assert 1-fidelity(ex,evolve_trotter(terms,psi,2.,64,order)) < 1-fidelity(ex,evolve_trotter(terms,psi,2.,4,order))
