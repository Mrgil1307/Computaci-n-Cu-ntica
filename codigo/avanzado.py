"""Experimentos físicos: cadena abierta, orden |n0...nN-1|, hbar=t=1."""
import numpy as np
from nucleo import operators,hamiltonian,basis_state,pauli_decompose

def sector(n,m):
    return np.array([i for i in range(2**n) if i.bit_count()==m])

def spectra(n,u):
    h=hamiltonian(n,u)
    return [(m,k,float(e)) for m in range(n+1) for k,e in enumerate(np.linalg.eigvalsh(h[np.ix_(sector(n,m),sector(n,m))]))]

def dynamics(h,psi,times):
    e,v=np.linalg.eigh(h)
    return (np.exp(-1j*np.outer(times,e))*(v.conj().T@psi))@v.T

def onebody_and_cov(states,n):
    a,ad=operators(n)
    ns=[ad[i]@a[i] for i in range(n)]
    oc=np.empty((len(states),n)); coh=np.empty((len(states),n,n),complex);cov=np.empty((len(states),n,n))
    for k,st in enumerate(states):
        oc[k]=[np.vdot(st,o@st).real for o in ns]
        for i in range(n):
            for j in range(n):
                coh[k,i,j]=np.vdot(st,ad[i]@a[j]@st)
                cov[k,i,j]=(np.vdot(st,ns[i]@ns[j]@st).real-oc[k,i]*oc[k,j])
    return oc,coh,cov

def entropy(state,n,subset):
    subset=tuple(subset);rest=tuple(i for i in range(n) if i not in subset)
    matrix=state.reshape([2]*n).transpose(subset+rest).reshape(2**len(subset),-1)
    p=np.linalg.svd(matrix,compute_uv=False)**2;p=p[p>1e-14]
    return float(-np.sum(p*np.log2(p)))

def validate(h,states,n):
    a,ad=operators(n)
    num=sum((ad[i]@a[i] for i in range(n)),np.zeros_like(h))
    par=np.diag([(-1)**i.bit_count() for i in range(2**n)])
    energies=np.einsum('ti,ij,tj->t',states.conj(),h,states)
    return {'hermitian':float(np.linalg.norm(h-h.conj().T)), 'number_commutator':float(np.linalg.norm(h@num-num@h)), 'parity_commutator':float(np.linalg.norm(h@par-par@h)), 'norm_drift':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1))), 'energy_drift':float(np.max(abs(energies-energies[0])))}

def resources(terms,steps,order):
    words=[word for word,c in terms if any(x!='I' for x in word)]
    factor=steps*(1 if order==1 else 2)
    return len(words)*factor, sum(2*(sum(x!='I' for x in word)-1) for word in words)*factor
