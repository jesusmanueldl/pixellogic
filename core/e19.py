"""E19 offline fixed-geometry diagnostic. Input messages are used only to SCORE results.
Decoder receives sampled bytes, predicted erasures, code parameters, CRC structure.
Oracle is a separate upper bound and never used by non-oracle methods.
"""
import argparse, csv, math, random, zlib
from pathlib import Path
from e18 import make_grid,render,damage_white_square,sample,suspect_cells,positions_to_bytes
from rs_joint import decode_joint
from rs_erasure import decode_erasures

def valid(out):
    return out is not None and len(out)>=4 and zlib.crc32(out[:-4]).to_bytes(4,'big')==out[-4:]

def experiment(size,frac,seed,layout,max_errors=4):
    rng=random.Random(7000+size);msg=bytes(rng.randrange(256) for _ in range(size))
    parity=math.ceil((size+4)*.75); side=29 if size==18 else 45
    grid,pos,code,k=make_grid(msg,parity,side,layout)
    img=render(grid);damaged,_=damage_white_square(img,frac,seed)
    bits,_=sample(damaged,side);got=positions_to_bytes(bits,pos)
    suspected=suspect_cells(bits)
    candidate=[i for i,p in enumerate(pos) if any(suspected[r,c] for r,c in p)]
    true_erased=[i for i,(a,b) in enumerate(zip(code,got)) if a!=b]
    results={}
    # E18 no vs inferred: preserve original baseline and identical masks.
    for name,erasures in [('e18_none',[]),('e18_inferred',candidate)]:
        try:out=decode_erasures(got,k,erasures)
        except (ValueError,ZeroDivisionError,IndexError):out=None
        results[name]=int(valid(out) and out[:-4]==msg)
    # Correct errors of unknown position; combined errors and inferred erasures.
    for name,erasures in [('joint_hard',[]),('joint_inferred',candidate)]:
        out=decode_joint(got,k,erasures,max_unknown_errors=max_errors)
        results[name]=int(valid(out) and out[:-4]==msg)
    # Only for diagnostic, not a viable reader.
    try:ideal=decode_erasures(got,k,true_erased)
    except (ValueError,ZeroDivisionError,IndexError):ideal=None
    results['hybrid']=int(any(results[x] for x in ('e18_inferred','joint_hard','joint_inferred')))
    results['oracle']=int(valid(ideal) and ideal[:-4]==msg)
    return dict(size=size,occlusion=frac,seed=seed,layout=layout,actual_corrupted=len(true_erased),predicted_erasures=len(candidate),false_positive=len(set(candidate)-set(true_erased)),false_negative=len(set(true_erased)-set(candidate)),**results)

def run(repetitions=5,seed=42,out='resultados_e19',max_errors=4):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for size in (18,70):
      for severity in (.08,.16):
       for layout in ('row','spread'):
        for rep in range(repetitions):
         s=seed+size*100000+int(severity*100)*1000+rep
         rows.append(experiment(size,severity,s,layout,max_errors))
    with (folder/'trials_e19.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
    metrics=['e18_none','e18_inferred','joint_hard','joint_inferred','hybrid','oracle']
    summary=[]
    for size in (18,70):
     for severity in (.08,.16):
      for layout in ('row','spread'):
       subset=[r for r in rows if (r['size'],r['occlusion'],r['layout'])==(size,severity,layout)]
       summary.append(dict(size=size,occlusion=severity,layout=layout,n=len(subset),**{m:sum(r[m] for r in subset)/len(subset) for m in metrics}))
    with (folder/'summary_e19.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,summary[0].keys());writer.writeheader();writer.writerows(summary)
    print(f'E19: {len(rows)} identical-mask, known-geometry optical-grid tests; no QR claims')
    for m in metrics:print('%-18s %6.2f%%'%(m,100*sum(r[m] for r in rows)/len(rows)))
    print('Output:',folder.resolve())
    return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--repetitions',type=int,default=5);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--max-errors',type=int,default=4);ap.add_argument('--out',default='resultados_e19');a=ap.parse_args();run(a.repetitions,a.seed,a.out,a.max_errors)
