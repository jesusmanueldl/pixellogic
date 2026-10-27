"""E23: E22-compatible autonomous decoding with header template rescue.
Strict E22-first, fallback compares valid encoded headers to sampled header.
No reference payload, ground-truth geometry or occlusion mask is supplied.
"""
import csv,argparse,zlib,math
from pathlib import Path
import cv2,numpy as np
import e22,e15
from e13 import preprocess
from e11_helpers import degrade
from experiment_v03 import bits_of,hamming_encode
from rs_erasure import encode
from e20 import decode_gmd,valid

HEADER_CACHE={}
def candidates(side):
 if side not in HEADER_CACHE:
  opts=[]
  for length in range(1,181):
   n=length+4+e22.parity_for(length)
   if not e22._fits(side,n) or 140+8*n>side*side:continue
   # Only lengths whose minimum fitted side is this side
   first=next((s for s in e22.SIDES if s*s>=140+8*n and e22._fits(s,n)),None)
   if first!=side:continue
   hb,_=hamming_encode(bits_of(e22.header(side,length)))
   opts.append((length,np.asarray(hb,dtype=np.uint8)))
  HEADER_CACHE[side]=opts
 return HEADER_CACHE[side]

def decode_body_rescue(inner, maximum_header_errors=26,max_errors=2):
 side=inner.shape[0]
 observed=inner.ravel()[:140]
 ranked=sorted(((int(np.count_nonzero(observed!=hb)),length) for length,hb in candidates(side)))
 if not ranked or ranked[0][0]>maximum_header_errors:return None,None
 # Never blindly accept a guessed header: packet CRC must also validate.
 for dist,length in ranked[:3]:
  if dist>maximum_header_errors or dist>ranked[0][0]+3:break
  k=length+4;n=k+e22.parity_for(length)
  try:
   pos=e22.positions(side,n)
   received=bytes(sum(int(inner[r,c])<<(7-j) for j,(r,c) in enumerate(pp)) for pp in pos)
   packet,_=decode_gmd(received,k,inner,pos,max_errors=max_errors,max_trials=5)
   if valid(packet) and len(packet)==k:return packet[:-4],dist
  except (IndexError,ValueError,ZeroDivisionError):pass
 return None,None

def decode_image(image,methods=('clahe_pre','flatfield_pre','e12_otsu'),max_errors=2,return_stage=False):
 if image is None:return (None,'no_image') if return_stage else None
 if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
 # Strict E22 pipeline first; any successful E22 result is preserved.
 p=e22.decode_image(image,methods,max_errors)
 if p is not None:return (p,'e22') if return_stage else p
 stage='no_quad'
 for method in methods:
  norm=preprocess(image,method)
  for quad in e15.quads(norm):
   stage='quad'
   for side in e22.SIDES:
    sample=e15.sample_quad(norm,quad,side)
    if min(sample[0].mean(),sample[-1].mean(),sample[:,0].mean(),sample[:,-1].mean())<.65:continue
    stage='frame'
    for turn in range(4):
     oriented=np.rot90(sample,turn)
     if np.count_nonzero(oriented[2,4:16]!=e15.SIGNATURE)>3:continue
     stage='signature'
     inner=oriented[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
     result,dist=decode_body_rescue(inner,max_errors=max_errors)
     if result is not None:return (result,'header_rescue') if return_stage else result
 return (None,stage) if return_stage else None

def run(repetitions=30,seed=42,pixels=600,out='resultados_e23'):
 folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
 samples=[('corto',b'Hola'),('referencia',b'Hola desde Tabasco'),('largo',('PixelLogic E22: '+'prueba cientifica '*3).encode())]
 scenarios=[('clean',0),('illumination',.8),('perspective',.065),('blur',2.5),('occlusion',.16)]
 rows=[]
 for i,(label,payload) in enumerate(samples):
  original=e15.render(e22.build_symbol(payload),pixels)
  cv2.imwrite(str(folder/f'simbolo_{label}.png'),original)
  for si,(scenario,strength) in enumerate(scenarios):
   for rep in range(repetitions):
    sd=seed+i*100000+si*10000+rep
    img=original if scenario=='clean' else degrade(original,scenario,strength,np.random.default_rng(sd))
    old=e22.decode_image(img)
    new,route=decode_image(img,return_stage=True)
    rows.append(dict(label=label,scenario=scenario,seed=sd,e22=int(old==payload),e23=int(new==payload),wrong=int(new is not None and new!=payload),route=route))
   group=rows[-repetitions:]
   print(f'{label:11} {scenario:12} E22 {sum(r["e22"] for r in group):2}/{repetitions} | E23 {sum(r["e23"] for r in group):2}/{repetitions} | rescue {sum(r["route"]=="header_rescue" for r in group)}',flush=True)
 with (folder/'trials_e23.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
 print('Gains',sum(r['e23'] and not r['e22'] for r in rows),'regressions',sum(r['e22'] and not r['e23'] for r in rows),'wrong',sum(r['wrong'] for r in rows))
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=5);p.add_argument('--seed',type=int,default=42);p.add_argument('--pixels',type=int,default=600);p.add_argument('--out',default='resultados_e23');a=p.parse_args();run(a.repetitions,a.seed,a.pixels,a.out)
