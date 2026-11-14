from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nucleo import hamiltonian,basis_state,pauli_decompose,evolve_trotter,fidelity
from avanzado import sector,spectra,dynamics,onebody_and_cov,entropy,validate,resources
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'figuras';D=ROOT/'datos';F.mkdir(exist_ok=True);D.mkdir(exist_ok=True)
C=['#087F8C','#3478DB','#7954B6','#E98B43','#D45F63']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','pdf.fonttype':42})
def save(fig,name):
    fig.savefig(F/(name+'.png'),dpi=230,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def table(name,header,rows):
    with (D/name).open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
def run():
    n=4; psi=basis_state([1,1,0,0]);times=np.linspace(0,8,161);us=[0,1,2,4,6]
    probs=[];occs=[];checks=[];row=[]
    for u in us:
        h=hamiltonian(n,u);st=dynamics(h,psi,times);oc,_,_=onebody_and_cov(st,n)
        p=abs(st[:,int('1001',2)])**2;probs.append(p);occs.append(oc);checks.append((u,validate(h,st,n)))
        row.extend([[u,t,p[k],*oc[k]] for k,t in enumerate(times)])
    table('02_interacciones.csv',['U','tau','P1001','n0','n1','n2','n3'],row)
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    im=ax[0,0].imshow(probs,origin='lower',aspect='auto',extent=[0,8,-.5,4.5],cmap='magma');ax[0,0].set_yticks(range(5),us);ax[0,0].set_ylabel('Interacción U/t');ax[0,0].set_title('(a) Mapa de probabilidad |1001⟩');fig.colorbar(im,ax=ax[0,0],label='Probabilidad')
    for k,u in enumerate(us):ax[0,1].plot(times,probs[k],c=C[k],label=f'U={u}')
    ax[0,1].legend(ncol=2,fontsize=8);ax[0,1].set_title('(b) Transferencia según interacción')
    im=ax[1,0].imshow(occs[2].T,origin='lower',aspect='auto',extent=[0,8,-.5,3.5],cmap='viridis',vmin=0,vmax=1);fig.colorbar(im,ax=ax[1,0],label='Ocupación');ax[1,0].set_yticks(range(4));ax[1,0].set_ylabel('Modo');ax[1,0].set_title('(c) Ocupaciones: U=2')
    ax[1,1].bar(range(5),[p.max() for p in probs],color=C);ax[1,1].set_xticks(range(5),us);ax[1,1].set_title('(d) Probabilidad máxima');ax[1,1].set_xlabel('U/t')
    for a in ax.flat:a.set_xlabel('Tiempo τ' if a not in (ax[1,1],) else 'U/t')
    save(fig,'07_interacciones_4_modos')
    # Spectral study
    rows=[];grid=np.linspace(0,6,61)
    for u in grid:rows.extend([[u,*item] for item in spectra(n,u)])
    table('03_espectros.csv',['U','particulas','indice','energia'],rows)
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for m in range(n+1):
        for j in range(len(sector(n,m))):
            selected=np.array([r for r in rows if r[1]==m and r[2]==j]);ax[0].plot(selected[:,0],selected[:,3],c=C[m],alpha=.8,lw=1.3,label=f'{m} partículas' if j==0 else None)
        selected=np.array([r for r in rows if r[1]==m and r[2]==0]);ax[1].plot(selected[:,0],selected[:,3],c=C[m],lw=2.3,label=f'Número={m}')
    ax[0].set_title('(a) Niveles por sector de partículas');ax[1].set_title('(b) Energía fundamental de cada sector')
    for a in ax:a.set_xlabel('U/t');a.set_ylabel('Energía / t');a.legend(fontsize=8)
    save(fig,'08_espectros_sectores')
    # Correlations / entanglement
    h=hamiltonian(n,2);states=dynamics(h,psi,times);oc,rho,cov=onebody_and_cov(states,n)
    shots=[0,1,2,4];fig,ax=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for k,t in enumerate(shots):
        idx=np.argmin(abs(times-t));a=ax[0,k];im=a.imshow(cov[idx],vmin=-.25,vmax=.25,cmap='RdBu_r');a.set_title(f'Covarianza; τ={t:g}')
        b=ax[1,k];im2=b.imshow(abs(rho[idx]),vmin=0,vmax=1,cmap='viridis');b.set_title(f'|⟨a†ᵢ aⱼ⟩|; τ={t:g}')
        for z in (a,b):z.set_xticks(range(n));z.set_yticks(range(n));z.set_xlabel('Modo j');z.set_ylabel('Modo i')
    fig.colorbar(im,ax=list(ax[0,:]),shrink=.7,label='Covarianza');fig.colorbar(im2,ax=list(ax[1,:]),shrink=.7,label='Coherencia');save(fig,'09_correlaciones')
    ent=np.array([entropy(st,n,(0,1)) for st in states]);part=np.array([1/np.sum(abs(st)**4) for st in states]);table('04_entropia.csv',['tau','entropy_01_23','participation'],zip(times,ent,part))
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    ax[0].plot(times,ent,c=C[2],lw=2);ax[0].set_title('(a) Entropía de partición de modos');ax[0].set_ylabel('bits')
    ax[1].plot(times,part,c=C[3],lw=2);ax[1].set_title('(b) Participación en Fock');ax[1].set_ylabel('1/Σp²')
    for j in range(n):ax[2].plot(times,oc[:,j],label=f'Modo {j}',c=C[j]);ax[2].legend(fontsize=7,ncol=2);ax[2].set_title('(c) Ocupaciones conservando N=2')
    for a in ax:a.set_xlabel('Tiempo τ')
    save(fig,'10_entropia_coherencia')
    # Circuit-cost & error analysis
    terms=pauli_decompose(h);exact=dynamics(h,psi,[2.])[0];res=[]
    for order in (1,2):
        for steps in (1,2,4,8,16,32,64):
            approx=evolve_trotter(terms,psi,2.,steps,order);rot,cnot=resources(terms,steps,order)
            res.append([order,steps,rot,cnot,max(0.,1-fidelity(exact,approx)),.5*np.sum(abs(abs(exact)**2-abs(approx)**2))])
    table('06_trotter_costos.csv',['orden','pasos','PauliRot','CNOT_estimadas','infidelidad','distancia_probabilidades'],res)
    fig,ax=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for o,col in [(1,C[1]),(2,C[3])]:
        data=np.array([r for r in res if r[0]==o]);label=f'Orden {o}'
        ax[0].loglog(data[:,1],np.maximum(data[:,4],1e-16),'o-',label=label,c=col)
        ax[1].loglog(data[:,3],np.maximum(data[:,4],1e-16),'o-',label=label,c=col)
        ax[2].semilogx(data[:,1],data[:,5],'o-',label=label,c=col)
    ax[0].set_title('(a) Convergencia');ax[1].set_title('(b) Precisión vs. CNOT estimadas');ax[2].set_title('(c) Error en probabilidades')
    ax[0].set_xlabel('Pasos');ax[1].set_xlabel('CNOT estimadas');ax[2].set_xlabel('Pasos')
    ax[0].set_ylabel('1−F');ax[1].set_ylabel('1−F');ax[2].set_ylabel('Distancia de variación total')
    for a in ax:a.legend(fontsize=8);a.grid(alpha=.15)
    save(fig,'11_costo_fidelidad')
    # scaling
    scal=[]
    for m in range(2,6):
        ts=pauli_decompose(hamiltonian(m,2));weights=[sum(v!='I' for v in w) for w,c in ts if any(v!='I' for v in w)]
        scal.append([m,2**m,4**m,len(weights),max(weights)])
    table('06_escalamiento.csv',['N','dimension','elementos_matriz','cadenas_Pauli_no_identidad','peso_maximo'],scal)
    fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    ax[0].semilogy([r[0] for r in scal],[r[2] for r in scal],'o-',c=C[2],lw=2);ax[0].set_title('(a) Elementos de matriz densa, 4ᴺ')
    ax[1].plot([r[0] for r in scal],[r[3] for r in scal],'o-',c=C[0],label='Cadenas Pauli');ax[1].plot([r[0] for r in scal],[r[4] for r in scal],'s--',c=C[3],label='Peso máximo');ax[1].legend(fontsize=8);ax[1].set_title('(b) Estructura de Jordan–Wigner')
    for a in ax:a.set_xlabel('Número de modos N')
    save(fig,'12_escalamiento')
    result={'model':'Four-site open chain; two particles; initial state |1100>; hopping=1; hbar=1','interaction_maxima':[{'U':u,'maximum':float(p.max()),'time':float(times[p.argmax()])} for u,p in zip(us,probs)],'max_validation_error':max(max(v.values()) for _,v in checks),'entropy_max':float(ent.max()),'trotter_at_8':[{'order':r[0],'infidelity':r[4],'estimated_CNOT':r[3]} for r in res if r[1]==8],'pennylane_status':'not executed; requires installed PennyLane'}
    (D/'resumen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
