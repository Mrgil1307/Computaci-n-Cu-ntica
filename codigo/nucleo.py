"""Jordan-Wigner con orden de qubits |n0 n1 ...>, hbar=t=1.
Referencias exactas de NumPy/SciPy, independientes de PennyLane.
"""
from itertools import product
from functools import reduce
import numpy as np
from scipy.linalg import expm

I=np.eye(2, dtype=complex); X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex); Z=np.diag([1,-1]).astype(complex)
PAULI={'I':I,'X':X,'Y':Y,'Z':Z}

def tensor(mats):
    return reduce(np.kron,mats)

def jw_annihilation(n,j,parity=True):
    if not 0<=j<n: raise ValueError('índice fuera de rango')
    return tensor([Z if parity and k<j else (X+1j*Y)/2 if k==j else I for k in range(n)])

def operators(n,parity=True):
    a=[jw_annihilation(n,j,parity) for j in range(n)]
    return a,[o.conj().T for o in a]

def pauli_matrix(word):
    return tensor([PAULI[x] for x in word])

def pauli_decompose(h,tol=1e-11):
    n=int(round(np.log2(h.shape[0]))); assert h.shape==(2**n,2**n)
    out=[]
    for tup in product('IXYZ',repeat=n):
        w=''.join(tup); p=pauli_matrix(w)
        coef=np.trace(p@h)/(2**n)
        if abs(coef)>tol:
            if abs(coef.imag)>tol:raise ValueError('Hamiltoniano no hermítico')
            out.append((w,float(coef.real)))
    return out

def hamiltonian(n=3,u=2.0,hopping=1.0):
    a,ad=operators(n);h=np.zeros((2**n,2**n),complex)
    for i in range(n-1):
        h-=hopping*(ad[i]@a[i+1]+ad[i+1]@a[i])
        h+=u*(ad[i]@a[i])@(ad[i+1]@a[i+1])
    return h

def basis_state(bits):
    n=len(bits); v=np.zeros(2**n,complex);v[int(''.join(map(str,bits)),2)]=1
    return v

def evolve_exact(h,psi,t):return expm(-1j*t*h)@psi

def apply_pauli_rotation(state,word,angle):
    # exp(-i angle P/2) = cos(angle/2)I-i sin(angle/2)P
    return np.cos(angle/2)*state-1j*np.sin(angle/2)*(pauli_matrix(word)@state)

def evolve_trotter(terms,psi,t,steps,order=1):
    if steps<1:raise ValueError('steps debe ser positivo')
    st=psi.astype(complex).copy();dt=t/steps
    active=[(w,c) for w,c in terms if set(w)!={'I'}]
    for _ in range(steps):
        if order==1:
            for w,c in active:st=apply_pauli_rotation(st,w,2*c*dt)
        elif order==2:
            for w,c in active:st=apply_pauli_rotation(st,w,c*dt)
            for w,c in reversed(active):st=apply_pauli_rotation(st,w,c*dt)
        else:raise ValueError('order debe ser 1 o 2')
    return st

def observables(state,n):
    probs=abs(state)**2
    occupations=np.array([sum(probs[i]*int(format(i,f'0{n}b')[j]) for i in range(2**n)) for j in range(n)])
    return probs,occupations

def fidelity(psi,phi):return float(abs(np.vdot(psi,phi))**2)

def car_error(n,parity=True):
    a,ad=operators(n,parity);ident=np.eye(2**n);zeros=np.zeros((2**n,2**n))
    errors=np.empty((3,n,n))
    for i in range(n):
        for j in range(n):
            errors[0,i,j]=np.linalg.norm(a[i]@ad[j]+ad[j]@a[i]-(ident if i==j else zeros))
            errors[1,i,j]=np.linalg.norm(a[i]@a[j]+a[j]@a[i])
            errors[2,i,j]=np.linalg.norm(ad[i]@ad[j]+ad[j]@ad[i])
    return errors
