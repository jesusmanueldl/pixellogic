"""E13 photometric ablation, retaining E12 geometry and encoding; simulation only."""
import argparse,csv
from pathlib import Path
import cv2,numpy as np
from e12 import make_symbol,render_grid,decode_bits,find_and_decode
from e11_helpers import degrade

METHODS=('e12_otsu','clahe_pre','flatfield_pre','adaptive_pre')
def extract_samples(image,side):
    n=side+6
    blur=cv2.GaussianBlur(image,(3,3),0)
    binary=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area<image.size*.08:continue
        hull=cv2.convexHull(c)
        approx=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
        if len(approx)!=4:continue
        pts=approx.reshape(4,2).astype(np.float32)
        sums=pts.sum(axis=1);diff=pts[:,0]-pts[:,1]
        ordered=np.array([pts[np.argmin(sums)],pts[np.argmax(diff)],pts[np.argmax(sums)],pts[np.argmin(diff)]],np.float32)
        candidates.append((area,ordered))
    if not candidates:return []
    samples=[]
    for _,quad in sorted(candidates,key=lambda item:-item[0])[:4]:
        target=np.float32([[0,0],[n*16-1,0],[n*16-1,n*16-1],[0,n*16-1]])
        H=cv2.getPerspectiveTransform(quad,target)
        square=cv2.warpPerspective(image,H,(n*16,n*16),borderValue=255)
        sample=square.reshape(n,16,n,16)[:,4:12,:,4:12].mean(axis=(1,3)).astype(np.float32)
        samples.append((sample,square))
    return samples

def binarize(sample,square,method):
    n=sample.shape[0]
    if method=='e12_otsu':
        _,th=cv2.threshold(sample.astype('uint8'),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return (th<128).astype(np.uint8)
    if method=='calibrated':
        # Outer row/column is black, next ring is white. Per-column
        # interpolation accounts for horizontal illumination gradient.
        black=np.median(np.stack((sample[0,:],sample[-1,:])),axis=0)
        white=np.median(np.stack((sample[1,:],sample[-2,:])),axis=0)
        threshold=(black+white)/2
        return (sample<threshold[None,:]).astype(np.uint8)
    if method=='adaptive':
        block=max(3,(n//3)|1)
        mask=cv2.adaptiveThreshold(sample.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,2)
        return (mask<128).astype(np.uint8)
    if method=='clahe_otsu':
        gray=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4)).apply(square)
        sam=gray.reshape(n,16,n,16)[:,4:12,:,4:12].mean(axis=(1,3)).astype(np.uint8)
        return (cv2.threshold(sam,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]<128).astype(np.uint8)
    raise ValueError(method)

def preprocess(image,method):
    if method=='clahe_pre':
        return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(image)
    if method=='flatfield_pre':
        background=cv2.GaussianBlur(image,(0,0),35).astype(np.float32)
        return np.clip(image.astype(np.float32)*180/(background+1),0,255).astype(np.uint8)
    if method=='adaptive_pre':
        return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,71,8)
    if method=='e12_otsu':return image
    raise ValueError(method)

def decode_method(image,payload,info,method):
    # Preserve EXACT E12 localization and decoder; change only photometric preprocessing.
    normalized=preprocess(image,method)
    valid,exact,ber,found=find_and_decode(normalized,payload,info)
    return valid,found,ber

def cases():
    return [('clean',0),('blur',1.),('blur',2.5),('illumination',.4),('illumination',.8),('occlusion',.08),('occlusion',.16),('perspective',.025),('perspective',.065)]

def run(repetitions=30,seed=42,pixels=600,out=Path('resultados_e13'),payload=b'Hola desde Tabasco'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grid,count,words,side=make_symbol(payload)
    base=render_grid(grid,pixels); info=(count,words,side)
    results=[]
    for ix,(kind,strength) in enumerate(cases()):
        for i in range(repetitions):
            ss=seed+i+ix*10000
            img=base if kind=='clean' else degrade(base,kind,strength,np.random.default_rng(ss))
            old=find_and_decode(img,payload,info)[0]
            rec={'scenario':kind,'strength':strength,'seed':ss,'e12_otsu':int(old)}
            for meth in METHODS[1:]: rec[meth]=int(decode_method(img,payload,info,meth)[0])
            results.append(rec)
            if i==0:
                path=out/'examples';path.mkdir(exist_ok=True)
                cv2.imwrite(str(path/f'{kind}_{strength}.png'),img)
        chunk=results[-repetitions:]
        print(f'{kind:12} {strength:5.3f} | '+ ' | '.join(f'{m}={np.mean([r[m] for r in chunk]):.1%}' for m in METHODS))
    summary=[]
    for kind,s in cases():
        sub=[r for r in results if r['scenario']==kind and r['strength']==s]
        summary.append(dict(scenario=kind,strength=s,n=len(sub),**{m:float(np.mean([r[m] for r in sub])) for m in METHODS}))
    for filename,data in [('trials_e13.csv',results),('summary_e13.csv',summary)]:
        with (out/filename).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=data[0].keys());writer.writeheader();writer.writerows(data)
    print('No claims of QR superiority: PixelLogic E12 encoding, simulated scenes only.')
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=30);p.add_argument('--seed',type=int,default=42);p.add_argument('--pixels',type=int,default=600);p.add_argument('--out',type=Path,default=Path('resultados_e13'));p.add_argument('--text',default='Hola desde Tabasco')
    a=p.parse_args();payload=a.text.encode('ascii')
    if len(payload)!=18:p.error('Requires 18 ASCII bytes, matching E12')
    run(a.repetitions,a.seed,a.pixels,a.out,payload)
