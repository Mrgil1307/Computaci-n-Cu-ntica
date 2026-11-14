"""Genera figuras vectoriales y raster a partir de datos reales, sin PennyLane."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from nucleo import *
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'datos';FIG=ROOT/'figuras'
DATA.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
C={'navy':'#142840','cyan':'#00A6A6','blue':'#367FE3','violet':'#8156C6','coral':'#EF6B65','gold':'#E9A23B'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':C['navy'],'text.color':C['navy'],'axes.edgecolor':'#B9C7D3','axes.grid':False,'savefig.facecolor':'white'})

def savefig(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=240,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def csv_out(name,header,rows):
    with (DATA/name).open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def run():
    h=hamiltonian(3,2);terms=pauli_decompose(h);psi=basis_state([1,1,0]);times=np.linspace(0,4,81);steps_array=[1,2,4,8,16,32,64]
    rec=[];fids={};errors={};results={};total=[]
    for order in (1,2):
        error_map=np.zeros((len(steps_array),len(times)))
        for k,steps in enumerate(steps_array):
            for z,t in enumerate(times):
                ex=evolve_exact(h,psi,float(t));ap=evolve_trotter(terms,psi,float(t),steps,order)
                fidelity_=fidelity(ex,ap);err=1-fidelity_
                error_map[k,z]=err
                if z%10==0:rec.append([order,steps,t,err,*observables(ap,3)[1]])
        errors[order]=error_map
    csv_out('01_trotter_convergencia.csv',['orden','pasos','tiempo','infidelidad','n0','n1','n2'],rec)
    fig,axs=plt.subplots(1,2,figsize=(12.6,4.35),sharey=True,constrained_layout=True)
    for ax,order in zip(axs,(1,2)):
        im=ax.imshow(np.log10(np.maximum(errors[order],1e-15)),origin='lower',aspect='auto',extent=[times[0],times[-1],-.5,len(steps_array)-.5],cmap='magma',vmin=-15,vmax=0)
        ax.set(yticks=range(len(steps_array)),yticklabels=steps_array,xlabel=r'Tiempo $\tau$',title=f'(a) Trotter 1.º orden' if order==1 else '(b) Suzuki–Trotter 2.º orden')
    axs[0].set_ylabel('Número de pasos');cb=fig.colorbar(im,ax=axs,shrink=.86,pad=.015);cb.set_label(r'$\log_{10}(1-F)$')
    fig.suptitle('Convergencia de la simulación digital · 3 modos · 2 fermiones · U/t = 2',fontsize=14,fontweight='bold')
    savefig(fig,'01_mapa_fidelidad_trotter')
    fig,ax=plt.subplots(figsize=(8.2,5.0),constrained_layout=True)
    at=np.argmin(abs(times-2))
    for order,color,mark in [(1,C['blue'],'o'),(2,C['coral'],'s')]:
        ax.loglog(steps_array,np.maximum(errors[order][:,at],1e-16),marker=mark,lw=2.5,ms=6,color=color,label=f'Orden {order}')
    ax.set(xlabel='Número de pasos de Trotter',ylabel=r'Infidelidad $1-F$',title=r'Convergencia a tiempo fijo $\tau=2$')
    ax.grid(True,which='both',alpha=.17);ax.legend();savefig(fig,'02_convergencia_ordenes')
    # Heatmaps operator CAR correct vs wrong
    good=car_error(3,True);wrong=car_error(3,False)
    fig,axs=plt.subplots(2,3,figsize=(11.4,6),constrained_layout=True)
    for row,arr in enumerate((good,wrong)):
        for j in range(3):
            ax=axs[row,j]; im=ax.imshow(arr[j],cmap='YlOrRd',vmin=0,vmax=max(1,float(wrong.max())))
            ax.set(xticks=range(3),yticks=range(3),xlabel='Índice j',ylabel='Índice i' if j==0 else '',title=['{aᵢ,aⱼ†}','{aᵢ,aⱼ}','{aᵢ†,aⱼ†}'][j])
            for i in range(3):
                for k in range(3):ax.text(k,i,f'{arr[j,i,k]:.2f}',ha='center',va='center',fontsize=9,color=C['navy'])
    axs[0,0].text(-.5,-1.25,'Con cadenas Z: relaciones CAR preservadas',fontweight='bold')
    axs[1,0].text(-.5,-1.25,'Sin cadenas Z: falla la anticonmutación',fontweight='bold')
    fig.colorbar(im,ax=axs,shrink=.75,label='Error de Frobenius')
    savefig(fig,'03_prueba_paridad_car')
    csv_out('02_car_con_vs_sin_paridad.csv',['paridad','familia','i','j','error'],[[parity,k,i,j,a[k,i,j]] for parity,a in [('con_Z',good),('sin_Z',wrong)] for k in range(3) for i in range(3) for j in range(3)])
    # Pauli terms matrix visualization
    fig,ax=plt.subplots(figsize=(9.2,4.6),constrained_layout=True)
    labels=[w for w,c in terms];values=[c for w,c in terms];bars=ax.barh(labels,values,color=[C['blue'] if c<0 else C['cyan'] for c in values],height=.68)
    ax.axvline(0,color=C['navy'],lw=1);ax.set(xlabel=r'Coeficiente $c_\alpha/t$',title=r'Descomposición exacta $H=\sum_\alpha c_\alpha P_\alpha$ · 3 modos')
    for rect,c in zip(bars,values):ax.text(c+(.035 if c>0 else -.035),rect.get_y()+rect.get_height()/2,f'{c:+.2f}',ha='left' if c>0 else 'right',va='center',fontsize=9)
    ax.set_xlim(min(values)-.27,max(values)+.27);savefig(fig,'04_hamiltoniano_pauli')
    csv_out('03_hamiltoniano_pauli.csv',['palabra','coeficiente'],terms)
    reconstructed=sum((c*pauli_matrix(w) for w,c in terms),np.zeros_like(h))
    reconerr=np.linalg.norm(h-reconstructed)
    # One- vs two-fermion sector snapshots
    fig,axs=plt.subplots(2,4,figsize=(12.5,5.4),constrained_layout=True)
    for r,(bits,u) in enumerate([([1,0,0],0.),([1,1,0],2.)]):
        H=hamiltonian(3,u);st=basis_state(bits)
        for col,t in enumerate([0.,.75,1.5,2.25]):
            p=abs(evolve_exact(H,st,t))**2
            # four-sector states with correct counts
            sector=sum(bits);labels=[format(i,'03b') for i in range(8) if format(i,'03b').count('1')==sector];p_sector=[p[int(label,2)] for label in labels]
            ax=axs[r,col];bars=ax.bar(labels,p_sector,color=[C['cyan'],C['violet'],C['coral']],alpha=.92)
            ax.set_ylim(0,1.04);ax.set_title(rf'$\tau={t:g}$');ax.tick_params(axis='x',rotation=45)
            if col==0:ax.set_ylabel(f'{sector} fermión(es)\nProbabilidad')
            for bar,val in zip(bars,p_sector):
                if val>.065:ax.text(bar.get_x()+bar.get_width()/2,val+.025,f'{val:.2f}',ha='center',fontsize=8)
    fig.suptitle('Distribuciones de Fock · propagación y exclusión de Pauli',fontweight='bold',fontsize=14)
    savefig(fig,'05_fock_instantaneas')
    summary={'n_mods':3,'interaction_u_over_t':2,'pauli_terms':len(terms),'pauli_reconstruction_error':float(reconerr),'max_car_error_with_parity':float(good.max()),'max_car_error_without_parity':float(wrong.max()),'trotter_infidelity_t2_steps_8_first':float(errors[1][3,at]),'trotter_infidelity_t2_steps_8_second':float(errors[2][3,at]),'pennylane_executed':False}
    (DATA/'resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':run()
