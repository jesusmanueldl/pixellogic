"""E31 distributed isolated anchors, prototype format v2 payload, independent optical finder.
DO NOT interpret simulated accuracy as camera validation or QR superiority.
"""
import csv, argparse, time
from pathlib import Path
import cv2
import numpy as np
import e15,e22,e23
from e13 import preprocess
from e11_helpers import degrade

ANCHOR=3

def build_symbol(payload):
    g=e22.build_symbol(payload).copy()
    n=g.shape[0]
    g[:4,:]=0;g[-4:,:]=0;g[:,:4]=0;g[:,-4:]=0
    for y in (0,n-3):
        for x in (0,n-3):g[y:y+3,x:x+3]=1
    # asymmetry is carried by valid header: test all rotations
    return g

def _candidates(image):
    binary=cv2.threshold(cv2.GaussianBlur(image,(3,3),0),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    n,_,stats,cents=cv2.connectedComponentsWithStats(binary,8)
    H,W=image.shape
    found=[]
    for i in range(1,n):
        x,y,w,h,a=map(int,stats[i]);size=(w+h)/2
        if size<12 or size>min(H,W)*.20 or a<100:continue
        ratio=min(w,h)/max(w,h)
        if ratio<.73 or a/(w*h)<.72:continue
        found.append((float(a),np.array(cents[i],dtype=np.float32),size))
    return sorted(found,key=lambda z:z[0],reverse=True)[:18]

def _assign(points):
    """Return ordered corners TL, TR, BR, BL on observed anchor centers."""
    p=np.asarray(points,np.float32)
    s=p.sum(axis=1);d=p[:,0]-p[:,1]
    ix=[np.argmin(s),np.argmax(d),np.argmax(s),np.argmin(d)]
    if len(set(ix))!=4:return None
    return p[ix]

def _rects(image):
    cand=_candidates(image)
    if len(cand)<3:return []
    from itertools import combinations
    # Rank 4-point sets by near-square geometry and consistency of candidate sizes.
    ranked=[]
    for combo in combinations(cand[:10],4):
        sizes=np.array([v[2] for v in combo]);p=np.array([v[1] for v in combo])
        q=_assign(p)
        if q is None:continue
        edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        if np.min(edges)<60:continue
        aspect=np.max(edges)/np.min(edges)
        if aspect>1.6 or np.max(sizes)/np.min(sizes)>1.4:continue
        ranked.append((aspect+np.std(sizes)/np.mean(sizes),q,float(np.mean(sizes))))
    ranked.sort(key=lambda z:z[0])
    out=ranked[:4]
    # Three surviving anchors: right-angle triangle determines missing fourth.
    for combo in combinations(cand[:9],3):
        p=np.array([v[1] for v in combo]);sz=np.array([v[2] for v in combo])
        if np.max(sz)/np.min(sz)>1.3:continue
        for right in range(3):
            a,b=[p[k]-p[right] for k in range(3) if k!=right]
            la=np.linalg.norm(a);lb=np.linalg.norm(b)
            if min(la,lb)<60 or max(la,lb)/min(la,lb)>1.25:continue
            orth=abs(np.dot(a,b)/(la*lb))
            if orth>.14:continue
            missing=p[right]+a+b
            q=_assign(np.vstack([p,missing]))
            if q is None:continue
            out.append((2+orth,q,float(np.mean(sz))))
            break
    return out[:16]

def decode_image(image,return_info=False):
    if image is None:return (None,{'stage':'empty'}) if return_info else None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Each candidate is independent of known symbol location, size and payload.
    for method in ('e12_otsu','clahe_pre','flatfield_pre'):
        im=preprocess(image,method)
        for _,q,size in _rects(im)[:7]:
            inferred_n=3.0*float(np.median(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)))/max(size,1)+3
            for side in e22.SIDES:
                if abs((side+8)-inferred_n)>4.5:continue
                n=side+8
                # source anchor centers at modules 1.5 and n-1.5; target corners module boundaries
                # recover full grid outer corners from anchor quad using homography
                xy=np.float32([[1.5,1.5],[n-1.5,1.5],[n-1.5,n-1.5],[1.5,n-1.5]])
                H=cv2.getPerspectiveTransform(xy,q)
                bounds=cv2.perspectiveTransform(np.float32([[[0,0],[n,0],[n,n],[0,n]]]),H)[0]
                sample=e15.sample_quad(im,bounds.astype(np.float32),side)
                for turn in range(4):
                    inner=np.rot90(sample,turn)[4:-4,4:-4]
                    # Algebraic decoding only after inexpensive metadata correlation.
                    observed=inner.ravel()[:140]
                    options=e23.candidates(side)
                    if not options:continue
                    mindist=min(int(np.count_nonzero(observed!=hb)) for _,hb in options)
                    if mindist>26:continue
                    budget=0 if side>=37 else 2
                    result=e22.decode_body(inner,max_errors=budget)
                    if result is None:result,_=e23.decode_body_rescue(inner,max_errors=budget)
                    if result is not None:
                        info={'stage':'success','method':method,'side':side}
                        return (result,info) if return_info else result
    return (None,{'stage':'failed'}) if return_info else None

def benchmark(reps=10,seed=42,out='resultados_e31'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    messages=[('corto',b'Hola'),('referencia',b'Hola desde Tabasco'),('largo',('PixelLogic E22: '+'prueba cientifica '*3).encode())]
    rows=[]
    for index,(label,data) in enumerate(messages):
        original=e15.render(build_symbol(data),600)
        cv2.imwrite(str(out/f'simbolo_e31_{label}.png'),original)
        for condition,strength in [('clean',0),('occlusion',.16),('perspective',.065),('blur',2.5)]:
            for j in range(reps):
                s=seed+index*100000+ {'clean':0,'occlusion':40000,'perspective':20000,'blur':30000}[condition]+j
                im=original if condition=='clean' else degrade(original,condition,strength,np.random.default_rng(s))
                t=time.perf_counter();decoded=decode_image(im);elapsed=time.perf_counter()-t
                rows.append(dict(label=label,condition=condition,seed=s,success=int(decoded==data),wrong=int(decoded is not None and decoded!=data),seconds=round(elapsed,4)))
            subset=rows[-reps:]
            print(label,condition,sum(r['success'] for r in subset),'/',reps, 'mean_seconds',round(np.mean([r['seconds'] for r in subset]),3),flush=True)
    with (out/'trials_e31.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=5);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='resultados_e31');a=p.parse_args();benchmark(a.repetitions,a.seed,a.out)
