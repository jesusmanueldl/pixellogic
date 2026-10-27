"""E32 bounded-cost erasure/one-error decoder experiment for E31 geometry.
Maintains E31 symbol format. No clock-based assertion; bounded algebraic searches.
"""
import argparse,csv,time,zlib
from pathlib import Path
import numpy as np
import cv2
import e31,e22,e23,e20,e15
from e11_helpers import degrade
from rs_erasure import encode,decode_erasures,mul,inv,coeff
from e18 import suspect_cells


def _valid_code(packet,got,k,erasures):
    if not e20.valid(packet) or len(packet)!=k: return False
    recon=encode(packet,len(got)-k)
    return all(recon[i]==got[i] for i in range(len(got)) if i not in erasures)

def single_unknown(got,k):
    """Exact single unknown symbol correction in systematic Cauchy code.
    Locate via first two independent parity syndromes, validate all parity and CRC.
    O(k*parity) per codeword. Does not handle erasures.
    """
    p=len(got)-k
    if p<2:return None
    baseline=encode(got[:k],p)
    if e20.valid(got[:k]) and baseline==got:return got[:k]
    d0=got[k]^baseline[k]
    d1=got[k+1]^baseline[k+1]
    if not d0 or not d1:return None
    for i in range(k):
        delta=mul(d0,inv(coeff(i,0,k)))
        if mul(coeff(i,1,k),delta)!=d1:continue
        corrected=bytearray(got[:k]);corrected[i]^=delta
        if e20.valid(corrected) and encode(corrected,p)[k:]==got[k:]:return bytes(corrected)
    return None

def fast_gmd(got,k,bits,pos,max_errors=0,max_trials=5):
    p=len(got)-k
    if len(got)>255:return None,'capacity'
    if encode(got[:k],p)==got and e20.valid(got[:k]):return got[:k],'clean'
    # One unknown error (including errors in parity with intact data).
    if e20.valid(got[:k]):return got[:k],'crc_payload'
    q=single_unknown(got,k)
    if q is not None:return q,'one_error'
    if p<2:return None,'fail'
    suspect=suspect_cells(bits)
    strong=[i for i,pp in enumerate(pos) if any(suspect[r,c] for r,c in pp)]
    white=(bits==0).astype(np.float32)
    density=cv2.boxFilter(white,-1,(5,5),normalize=True,borderType=cv2.BORDER_CONSTANT)
    ranked=sorted(range(len(pos)),key=lambda i:(-max(float(density[r,c]) for r,c in pos[i]),i))
    candidates=[strong]+[ranked[:n] for n in (2,4,6,8,10,14) if n<=p]
    checked=set()
    for erasures in candidates[:1+max_trials]:
        es=frozenset(erasures)
        if es in checked or len(es)>p:continue
        checked.add(es)
        # Matrix solving only for missing data bytes (bounded by erasure count).
        if sum(i<k for i in es)>12:continue
        try:packet=decode_erasures(got,k,es)
        except (ValueError,IndexError,ZeroDivisionError):continue
        if _valid_code(packet,got,k,es):return packet,'erasures'
    return None,'fail'

# e22 and e23 originally import a reference to decode_gmd; explicitly bind to
# E32's bounded algorithm in this copied experimental package only.
e22.decode_gmd=fast_gmd
e23.decode_gmd=fast_gmd

def decode_image(image,return_info=False):
    return e31.decode_image(image,return_info=return_info)

def run(repetitions=5,seeds=(42,142,888),out='resultados_e32'):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    messages=[('corto',b'Hola'),('referencia',b'Hola desde Tabasco'),('largo',('PixelLogic E22: '+'prueba cientifica '*3).encode())]
    conditions=[('clean',0),('illumination',.8),('perspective',.065),('blur',2.5),('occlusion',.16)]
    rows=[]
    for seed in seeds:
        for index,(label,payload) in enumerate(messages):
            original=e15.render(e31.build_symbol(payload),600)
            for ci,(condition,strength) in enumerate(conditions):
                for rep in range(repetitions):
                    sd=seed+index*100000+{'clean':0,'illumination':10000,'perspective':20000,'blur':30000,'occlusion':40000}[condition]+rep
                    im=original if condition=='clean' else degrade(original,condition,strength,np.random.default_rng(sd))
                    t=time.perf_counter();result=decode_image(im);secs=time.perf_counter()-t
                    rows.append(dict(seed=seed,label=label,condition=condition,damage_seed=sd,success=int(result==payload),wrong=int(result is not None and result!=payload),seconds=round(secs,5)))
                rr=rows[-repetitions:]
                print(seed,label,condition,f'{sum(r["success"] for r in rr)}/{repetitions}',f'max {max(r["seconds"] for r in rr):.3f}s',flush=True)
    with (folder/'trials_e32.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repetitions',type=int,default=5);parser.add_argument('--seeds',type=int,nargs='+',default=[42,142,888]);parser.add_argument('--out',default='resultados_e32');a=parser.parse_args();run(a.repetitions,a.seeds,a.out)
