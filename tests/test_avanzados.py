import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codigo'))
import numpy as np
from nucleo import operators,hamiltonian,basis_state,pauli_decompose,evolve_trotter,evolve_exact,fidelity
from avanzado import sector,spectra,dynamics,onebody_and_cov,entropy,validate,resources

def test_sector_count():
    assert [len(sector(4,k)) for k in range(5)]==[1,4,6,4,1]
def test_spectra_complete():
    for u in [0,2,6]:
        h=hamiltonian(4,u)
        assert np.allclose(sorted([r[2] for r in spectra(4,u)]),np.linalg.eigvalsh(h),atol=1e-12)
def test_dynamics_vs_expm():
    h=hamiltonian(4,2);psi=basis_state([1,1,0,0]);t=np.array([0.,0.3,1.,2.])
    res=dynamics(h,psi,t)
    assert all(np.linalg.norm(st-evolve_exact(h,psi,tk))<1e-12 for st,tk in zip(res,t))
def test_conservation_and_correlations():
    h=hamiltonian(4,2);st=dynamics(h,basis_state([1,1,0,0]),np.linspace(0,5,25));checks=validate(h,st,4)
    assert max(checks.values())<1e-11
    occ,rho,cov=onebody_and_cov(st,4)
    assert np.allclose(occ.sum(axis=1),2)
    assert np.max(abs(rho-rho.transpose(0,2,1).conj()))<1e-12
    assert np.max(abs(cov-cov.transpose(0,2,1)))<1e-12
    assert np.max(abs(np.trace(rho,axis1=1,axis2=2)-2))<1e-12

def test_entropy_limits():
    n=4
    assert abs(entropy(basis_state([1,1,0,0]),n,[0,1]))<1e-12
    psi=(basis_state([1,0,1,0])+basis_state([0,1,0,1]))/np.sqrt(2)
    assert abs(entropy(psi,n,[0,1])-1)<1e-12

def test_trotter_second_order_improvement_specific_case():
    n=4;h=hamiltonian(n,2);psi=basis_state([1,1,0,0]);terms=pauli_decompose(h);ref=evolve_exact(h,psi,2)
    f1=fidelity(ref,evolve_trotter(terms,psi,2,8,1));f2=fidelity(ref,evolve_trotter(terms,psi,2,8,2))
    assert f2>f1 and (1-f2)<0.01
    a,b=resources(terms,8,1);c,d=resources(terms,8,2)
    assert c==2*a and d==2*b
