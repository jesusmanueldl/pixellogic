import argparse,csv
from pathlib import Path
import numpy as np
import e22,e23,e15
from e11_helpers import degrade

def run(repetitions=30,seed=42,out='occlusion_e23',selected=None):
 folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
 samples=[('corto',b'Hola'),('referencia',b'Hola desde Tabasco'),('largo',('PixelLogic E22: '+'prueba cientifica '*3).encode())]
 rows=[]
 for i,(name,payload) in enumerate(samples):
  if selected and name not in selected: continue
  base=e15.render(e22.build_symbol(payload),600)
  for r in range(repetitions):
   s=seed+i*100000+40000+r
   img=degrade(base,'occlusion',.16,np.random.default_rng(s))
   got,route=e23.decode_image(img,return_stage=True)
   old=route=='e22'
   rows.append(dict(message=name,seed=s,e22=int(old),e23=int(got==payload),wrong=int(got is not None and got!=payload),route=route))
  segment=rows[-repetitions:]
  print(name,'E22:',sum(r['e22'] for r in segment),'E23:',sum(r['e23'] for r in segment),'routes:',{k:sum(r['route']==k for r in segment) for k in set(r['route'] for r in segment)},flush=True)
 with (folder/'trials_occlusion_e23.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=30);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='occlusion_e23');p.add_argument('--selected',nargs='*');a=p.parse_args();run(a.repetitions,a.seed,a.out,a.selected)
