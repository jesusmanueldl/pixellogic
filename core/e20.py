"""E20 research benchmark: multi-trial GMD-style erasure ranking, exact matched E19 control.
WARNING known geometry and simulation, not an autonomous optical reader.
"""
import argparse,random,math,zlib,csv
from pathlib import Path
import numpy as np
from e18 import make_grid,render,damage_white_square,sample,suspect_cells,positions_to_bytes
from rs_erasure import decode_erasures, encode
from rs_joint import decode_joint

def valid(out):
 return out is not None and len(out)>=4 and zlib.crc32(out[:-4]).to_bytes(4,'big')==out[-4:]

def decode_gmd(got,k,bits,pos,max_errors=3,max_trials=5):
 """Multi-trial erasure prefixes ordered by visually derived reliability only.
 First checks hard errors, then detected erasures; additional candidates based on white local density.
 CRC prevents counting unchecked algebraic candidates. No true mask or reference message used.
 """
 # Fast exact-codeword path: no algebraic search if intact and CRC passes.
 if len(got)>k and encode(got[:k],len(got)-k)==got and valid(got[:k]):return got[:k],'clean_fast'
 hard=decode_joint(got,k,[],max_unknown_errors=max_errors)
 if valid(hard):return hard,'hard'
 suspected=suspect_cells(bits)
 strong=[i for i,p in enumerate(pos) if any(suspected[r,c] for r,c in p)]
 q=decode_joint(got,k,strong,max_unknown_errors=max_errors)
 if valid(q):return q,'strong'
 try:
  q=decode_erasures(got,k,strong)
  if valid(q):return q,'fallback'
 except (ValueError,IndexError,ZeroDivisionError):pass
 # Graded visual whiteness: uses only sampled bit layout; does not use original or true damaged positions
 white=(bits==0).astype(np.float32)
 import cv2
 density=cv2.boxFilter(white,-1,(5,5),normalize=True,borderType=cv2.BORDER_CONSTANT)
 ranked=sorted(range(len(pos)), key=lambda i:(-max(float(density[r,c]) for r,c in pos[i]),i))
 parity=len(got)-k
 # Standard GMD incremental unreliable-symbol prefix sizes, bounded runtime
 sizes=sorted(set(min(parity, t) for t in (2,4,6,8,10) if t<=parity))[:max_trials]
 for t in sizes:
  erasures=ranked[:t]
  if set(erasures)==set(strong):continue
  q=decode_joint(got,k,erasures,max_unknown_errors=max_errors)
  if valid(q):return q,'gmd'
 return None,'fail'

def run(repetitions=4,seed=42,max_errors=3,out='resultados_e20'):
 folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
 rows=[]
 for size in (18,70):
  for frac in (.08,.16):
   for layout in ('row','spread'):
    rng=random.Random(7000+size);msg=bytes(rng.randrange(256) for _ in range(size))
    parity=math.ceil((size+4)*.75);side=29 if size==18 else 45
    grid,pos,code,k=make_grid(msg,parity,side,layout);img=render(grid)
    for rep in range(repetitions):
     s=seed+size*100000+int(frac*100)*1000+rep
     damaged,_=damage_white_square(img,frac,s);bits,_=sample(damaged,side)
     got=positions_to_bytes(bits,pos)
     guess=suspect_cells(bits); strong=[i for i,p in enumerate(pos) if any(guess[r,c] for r,c in p)]
     prev=[]
     try:q=decode_erasures(got,k,strong);prev.append(q)
     except (ValueError,IndexError,ZeroDivisionError):pass
     prev.extend([decode_joint(got,k,[],max_unknown_errors=max_errors),decode_joint(got,k,strong,max_unknown_errors=max_errors)])
     old=any(valid(x) and x[:-4]==msg for x in prev)
     outmsg,method=decode_gmd(got,k,bits,pos,max_errors=max_errors)
     new=valid(outmsg) and outmsg[:-4]==msg
     rows.append(dict(size=size,frac=frac,layout=layout,seed=s,old=int(old),gmd=int(new),method=method,regression=int(old and not new),gain=int(new and not old)))
     print('.',end='',flush=True)
    print(f' {size}B {frac:.0%} {layout} done',flush=True)
 with (folder/'trials_e20.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
 print('TRIALS',len(rows),'E19',sum(r['old'] for r in rows),'E20',sum(r['gmd'] for r in rows),'GAINS',sum(r['gain'] for r in rows),'REGRESSIONS',sum(r['regression'] for r in rows))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--max-errors',type=int,default=3);p.add_argument('--out',default='resultados_e20');a=p.parse_args();run(a.repetitions,a.seed,a.max_errors,a.out)
