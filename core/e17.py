"""E17: exact-erasure upper bound (geometry supplied by simulator).
This is NOT an optical decoder and cannot be compared end-to-end with QR.
"""
import argparse,csv,math,random,os
import numpy as np
from rs_erasure import encode,decode_erasures

def layout_positions(n,side,mode):
    # Map each 8-bit byte as a unit; each byte is in a 2x4 patch.
    if n*8>side*side:raise ValueError('insufficient grid capacity')
    slots=[(r,c) for r in range(0,side-1,2) for c in range(0,side-3,4)]
    if len(slots)<n:raise ValueError('insufficient 2x4 slots')
    if mode=='row': chosen=slots[:n]
    elif mode=='column':chosen=sorted(slots,key=lambda z:(z[1],z[0]))[:n]
    elif mode=='diagonal':
        # reorder by diagonal traversal, deterministic
        chosen=sorted(slots,key=lambda z:(z[0]//2+z[1]//4,z[0]))[:n]
    elif mode=='spread':
        # cyclic stride coprime with full set, distribute symbols across field
        length=len(slots);stride=max(1,length//2+1)
        while math.gcd(stride,length)!=1:stride+=1
        chosen=[slots[(i*stride)%length] for i in range(n)]
    else:raise ValueError('unknown layout')
    return [[(r+q//4,c+q%4) for q in range(8)] for r,c in chosen]

def mask(side,kind,severity,rng):
    out=np.zeros((side,side),dtype=bool)
    s=max(1,int(round(side*severity)))
    if kind=='square':
        r=rng.randrange(side-s+1);c=rng.randrange(side-s+1);out[r:r+s,c:c+s]=True
    elif kind=='horizontal':
        r=rng.randrange(side-s+1);out[r:r+s,:]=True
    elif kind=='vertical':
        c=rng.randrange(side-s+1);out[:,c:c+s]=True
    elif kind=='random':
        count=int(round(side*side*severity*severity))
        for x in rng.sample(range(side*side),count):out[x//side,x%side]=True
    else:raise ValueError(kind)
    return out

def run(trials=100,seed=42,out='resultados_e17',sizes=(18,70)):
    os.makedirs(out,exist_ok=True)
    rows=[]
    # RS rate approximately Hamming 4/7; parity=ceil(3*k/4).
    for k in sizes:
        parity=math.ceil(3*k/4);n=k+parity
        # keep same physical module grid for different layouts at same payload
        side=29 if k==18 else 45
        if k+parity>255:raise ValueError('too long')
        rng_payload=random.Random(1000+k)
        payload=bytes(rng_payload.randrange(256) for _ in range(k))
        code=encode(payload,parity)
        positions={mode:layout_positions(n,side,mode) for mode in ('row','column','diagonal','spread')}
        for kind in ('square','horizontal','vertical','random'):
            for severity in (0.08,0.16):
                for run_id in range(trials):
                    rng=random.Random(seed*1000000+k*10000+run_id*17+int(severity*100)+sum(map(ord,kind))*100)
                    damaged=mask(side,kind,severity,rng)
                    for mode,pos in positions.items():
                        erased=[i for i,pixels in enumerate(pos) if any(damaged[r,c] for r,c in pixels)]
                        recv=bytearray(code)
                        for i in erased:recv[i]=0
                        result=decode_erasures(recv,k,erased)
                        rows.append({'payload_bytes':k,'parity_bytes':parity,'modules_side':side,'kind':kind,'severity':severity,'trial':run_id,'layout':mode,'erased_symbols':len(erased),'capacity_ok':int(len(erased)<=parity),'payload_exact':int(result==payload),'oracle_erasures':1})
    with open(os.path.join(out,'trials_e17.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    keys={}
    for r in rows:
        key=(r['payload_bytes'],r['kind'],r['severity'],r['layout'])
        keys.setdefault(key,[]).append(r)
    summary=[]
    for key,items in sorted(keys.items()):
        summary.append({'payload_bytes':key[0],'kind':key[1],'severity':key[2],'layout':key[3],'trials':len(items),'recovery':sum(r['payload_exact'] for r in items)/len(items),'mean_erased_bytes':sum(r['erased_symbols'] for r in items)/len(items)})
    with open(os.path.join(out,'summary_e17.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    print(f'E17: {len(rows)} trials; ORACLE erasures; NOT optical decoding')
    for k in sizes:
        print(f'Payload {k}B:')
        for mode in ('row','column','diagonal','spread'):
            xx=[r for r in rows if r['payload_bytes']==k and r['layout']==mode]
            print('  %-9s %.1f%% recovered; %.2f erased symbols avg'%(mode,100*sum(r['payload_exact'] for r in xx)/len(xx),sum(r['erased_symbols'] for r in xx)/len(xx)))
    print('Saved:',os.path.abspath(out))
    return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repetitions',type=int,default=100);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--out',default='resultados_e17')
    a=ap.parse_args();run(a.repetitions,a.seed,a.out)
