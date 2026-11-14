"""Ejecuta las seis fases. La fase PennyLane es opcional y nunca se declara validada si falla."""
import importlib.util,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def invoke(script):
    subprocess.run([sys.executable,str(ROOT/'codigo'/script)],check=True,cwd=ROOT)
def main():
    invoke('generar_base.py');invoke('generar_avanzados.py')
    subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],check=True,cwd=ROOT)
    if importlib.util.find_spec('pennylane') is None:
        print('PENNYLANE NO INSTALADO: se omiten pruebas nativas (sin falsificar resultados).')
    else:
        invoke('pennylane_real.py');invoke('pennylane_extendido.py')
        invoke('consolidar_validacion.py')
if __name__=='__main__':main()
