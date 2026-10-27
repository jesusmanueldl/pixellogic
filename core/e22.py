"""E22 research prototype: autonomous framed visual marker + Cauchy-MDS/GMD.
No oracle geometry/payload/error positions are passed to decode_image.
Research-only; synthetic channel and no QR superiority claims.
"""
import argparse,csv,math,struct,zlib
from pathlib import Path
import cv2,numpy as np
import e15
from experiment_v03 import bits_of,bytes_of,hamming_encode,hamming_decode
from rs_erasure import encode
from e20 import decode_gmd,valid
from e13 import preprocess
from e11_helpers import degrade

SIDES=(25,29,33,37,41,45,49,53,57,61)

def parity_for(length):return math.ceil((length+4)*.75)

def positions(side,n):
    start=math.ceil(140/side)
    slots=[(r,c) for r in range(start,side-1,2) for c in range(0,side-3,4)]
    if n>len(slots):raise ValueError('symbol too small')
    stride=max(1,len(slots)//2+1)
    while math.gcd(stride,len(slots))!=1:stride+=1
    chosen=[slots[(i*stride)%len(slots)] for i in range(n)]
    return [[(r+j//4,c+j%4) for j in range(8)] for r,c in chosen]

def header(side,length):
    prefix=struct.pack('>2sBBH',b'PL',2,side,length)
    return prefix+zlib.crc32(prefix).to_bytes(4,'big')

def build_symbol(payload):
    payload=bytes(payload)
    if not 1<=len(payload)<=180:raise ValueError('supported payload length 1..180')
    packet=payload+zlib.crc32(payload).to_bytes(4,'big')
    parity=parity_for(len(payload))
    code=encode(packet,parity)
    side=next((s for s in SIDES if s*s>=140+8*len(code) and _fits(s,len(code))),None)
    if side is None:raise ValueError('unsupported capacity')
    pos=positions(side,len(code))
    flat=np.fromiter(((i*73+19)%2 for i in range(side*side)),np.uint8,count=side*side)
    hdr,_=hamming_encode(bits_of(header(side,len(payload))))
    flat[:140]=hdr
    inner=flat.reshape(side,side)
    for byte,pp in zip(code,pos):
        for j,(r,c) in enumerate(pp):inner[r,c]=(byte>>(7-j))&1
    n=side+2*e15.BORDER
    grid=np.zeros((n,n),np.uint8)
    grid[0,:]=grid[-1,:]=1;grid[:,0]=grid[:,-1]=1
    grid[2,4:16]=e15.SIGNATURE;grid[2,2]=1;grid[-3,-3]=1
    grid[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]=inner
    return grid

def _fits(side,n):
    start=math.ceil(140/side)
    return len(range(start,side-1,2))*len(range(0,side-3,4))>=n

def decode_body(inner,max_errors=2):
    side=inner.shape[0]
    try:
        hbits,_=hamming_decode(inner.ravel()[:140]);h=bytes_of(hbits)
        if len(h)!=10 or h[:2]!=b'PL' or h[2]!=2 or h[3]!=side or zlib.crc32(h[:6]).to_bytes(4,'big')!=h[6:]:return None
        length=int.from_bytes(h[4:6],'big')
        if not 1<=length<=180:return None
        k=length+4;parity=parity_for(length); n=k+parity
        pos=positions(side,n)
        received=bytes(sum(int(inner[r,c])<<(7-j) for j,(r,c) in enumerate(pp)) for pp in pos)
        packet,_=decode_gmd(received,k,inner,pos,max_errors=max_errors,max_trials=5)
        if valid(packet) and len(packet)==k:return packet[:-4]
    except (ValueError,IndexError,ZeroDivisionError,struct.error):pass
    return None

def decode_image(image,methods=('clahe_pre','flatfield_pre','e12_otsu'),max_errors=2):
    if image is None:return None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for method in methods:
        normalized=preprocess(image,method)
        for quad in e15.quads(normalized):
            for side in SIDES:
                # Check header/frame before costly algebraic decoding.
                grid=e15.sample_quad(normalized,quad,side)
                if min(grid[0].mean(),grid[-1].mean(),grid[:,0].mean(),grid[:,-1].mean())<.75:continue
                for turn in range(4):
                    rotated=np.rot90(grid,turn)
                    if np.count_nonzero(rotated[2,4:16]!=e15.SIGNATURE)>1:continue
                    decoded=decode_body(rotated[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER],max_errors)
                    if decoded is not None:return decoded
    return None

def run(repetitions=3,seed=42,pixels=600,out='resultados_e22'):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    samples=[('corto',b'Hola'),('referencia',b'Hola desde Tabasco'),('largo',('PixelLogic E22: '+'prueba cientifica '*3).encode())]
    rows=[]
    for i,(label,payload) in enumerate(samples):
        grid=build_symbol(payload);original=e15.render(grid,pixels)
        cv2.imwrite(str(folder/f'simbolo_{label}.png'),original)
        for si,(scenario,strength) in enumerate([('clean',0),('illumination',.8),('perspective',.065),('blur',2.5),('occlusion',.16)]):
            for rep in range(repetitions):
                sd=seed+i*100000+si*10000+rep
                img=original if scenario=='clean' else degrade(original,scenario,strength,np.random.default_rng(sd))
                obtained=decode_image(img)
                rows.append({'label':label,'length':len(payload),'scenario':scenario,'seed':sd,'success':int(obtained==payload),'wrong':int(obtained is not None and obtained!=payload)})
            subset=rows[-repetitions:]
            print(f'{label:11} {scenario:12} {sum(r["success"] for r in subset)}/{repetitions}',flush=True)
    with (folder/'trials_e22.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=3);p.add_argument('--seed',type=int,default=42);p.add_argument('--pixels',type=int,default=600);p.add_argument('--out',default='resultados_e22');a=p.parse_args();run(a.repetitions,a.seed,a.pixels,a.out)
