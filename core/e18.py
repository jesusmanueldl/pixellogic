"""E18: optical erasure-localization DIAGNOSTIC (known grid alignment).
Uses E17 systematic Cauchy MDS erasure-only code. Unknown symbol errors cannot be corrected.
Positions of occlusion are NEVER provided to inferred-erasure decoder.
"""
import argparse,csv,math,random,zlib
from pathlib import Path
import numpy as np
import cv2
from e17 import layout_positions
from rs_erasure import encode,decode_erasures


def make_grid(message,parity,side,layout='spread'):
    packet=message+zlib.crc32(message).to_bytes(4,'big')
    if len(packet)+parity>255:raise ValueError('GF(256) length')
    code=encode(packet,parity)
    pos=layout_positions(len(code),side,layout)
    grid=np.zeros((side,side),dtype=np.uint8)
    for value,pp in zip(code,pos):
        for j,(r,c) in enumerate(pp):grid[r,c]=(value>>(7-j))&1
    return grid,pos,code,len(packet)

def render(grid,module_px=12):
    return cv2.resize((255-255*grid).astype(np.uint8),(grid.shape[0]*module_px,grid.shape[1]*module_px),interpolation=cv2.INTER_NEAREST)

def damage_white_square(img,frac,seed):
    rng=random.Random(seed); n=img.shape[0];w=max(1,round(n*frac));r=rng.randrange(n-w+1);c=rng.randrange(n-w+1)
    result=img.copy();result[r:r+w,c:c+w]=255
    return result,(r,c,w)

def sample(img,side):
    z=img.shape[0]//side
    tiles=img.reshape(side,z,side,z)[:,z//4:3*z//4,:,z//4:3*z//4]
    avg=tiles.mean(axis=(1,3))
    return (avg<128).astype(np.uint8),avg

def suspect_cells(bits,window=5):
    # White-dominant local window is a weak optical surrogate for an occluded patch;
    # it cannot uniquely distinguish missing data from genuine white modules.
    white=(bits==0).astype(np.uint8)
    frac=cv2.boxFilter(white.astype(np.float32),-1,(window,window),normalize=True,borderType=cv2.BORDER_CONSTANT)
    return frac>.93

def positions_to_bytes(bits,pos):
    return bytes(sum(int(bits[r,c])<<(7-j) for j,(r,c) in enumerate(p)) for p in pos)

def verified_decode(received,k,erasures,message):
    try:
        decoded=decode_erasures(received,k,erasures)
    except (ValueError,IndexError,ZeroDivisionError):return False
    if decoded is None or len(decoded)<4:return False
    body,crc=decoded[:-4],decoded[-4:]
    return body==message and zlib.crc32(body).to_bytes(4,'big')==crc

def trial(size=18,frac=.16,seed=42,layout='spread',window=5):
    rng=random.Random(7000+size)
    msg=bytes(rng.randrange(256) for _ in range(size))
    parity=math.ceil((size+4)*.75)
    side=29 if size==18 else 45
    grid,pos,code,k=make_grid(msg,parity,side,layout)
    img=render(grid)
    damaged,region=damage_white_square(img,frac,seed)
    sampled,_=sample(damaged,side)
    inferred=suspect_cells(sampled,window)
    original=positions_to_bytes(grid,pos)
    received=positions_to_bytes(sampled,pos)
    true_b={i for i,(a,b) in enumerate(zip(original,received)) if a!=b}
    # Oracle: byte errors identified from comparison with original. Unavailable to real reader.
    visual_b={i for i,p in enumerate(pos) if any(inferred[r,c] for r,c in p)}
    no=verified_decode(received,k,[],msg)
    guess=verified_decode(received,k,sorted(visual_b),msg)
    oracle=verified_decode(received,k,sorted(true_b),msg)
    tp=len(visual_b&true_b);fp=len(visual_b-true_b);fn=len(true_b-visual_b)
    return dict(size=size,occlusion=frac,seed=seed,layout=layout,window=window,parity=parity,actual_corrupted=len(true_b),predicted_erasures=len(visual_b),true_positive=tp,false_positive=fp,false_negative=fn,without_erasures=int(no),inferred_erasures=int(guess),oracle_erasures=int(oracle))

def run(reps=30,seed=42,out='resultados_e18'):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for size in (18,70):
        for severity in (.08,.16):
            for layout in ('row','spread'):
                for rep in range(reps):
                    s=seed+size*100000+int(severity*100)*1000+rep
                    rows.append(trial(size,severity,s,layout))
    with (folder/'trials_e18.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    summary=[]
    for size in (18,70):
        for severity in (.08,.16):
            for layout in ('row','spread'):
                group=[r for r in rows if r['size']==size and r['occlusion']==severity and r['layout']==layout]
                row={'size':size,'occlusion':severity,'layout':layout,'n':len(group)}
                for key in ('without_erasures','inferred_erasures','oracle_erasures','actual_corrupted','predicted_erasures','true_positive','false_positive','false_negative'):
                    row[key]=sum(r[key] for r in group)/len(group)
                summary.append(row)
    with (folder/'summary_e18.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,summary[0].keys());w.writeheader();w.writerows(summary)
    print(f'E18 DIAGNOSTIC: {len(rows)} simulations; FIXED, KNOWN grid geometry; erasure-only MDS')
    for q in summary:
        print(f"{q['size']:2}B {q['occlusion']:.0%} {q['layout']:6} no={q['without_erasures']:.1%} inferred={q['inferred_erasures']:.1%} oracle={q['oracle_erasures']:.1%} false_erasures={q['false_positive']:.2f}")
    print('Not an autonomous image decoder; no QR claims. Files:',folder.resolve())
    return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repetitions',type=int,default=30);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--out',default='resultados_e18');a=ap.parse_args();run(a.repetitions,a.seed,a.out)
