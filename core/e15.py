"""PixelLogic E15: autonomous self-describing non-QR framed symbol.
Header and payload use Hamming(7,4); header carries version, size, length, CRC32.
No ground-truth payload, side length or codeword count is supplied to decoder.
"""
import argparse, csv, zlib, struct
from pathlib import Path
import cv2
import numpy as np
from experiment_v03 import bits_of,bytes_of,hamming_encode,hamming_decode,make_packet
from e13 import preprocess,cases
from e11_helpers import degrade

BORDER=4
SIDES=(17,21,25,29,33,37,41,45,49)
SIGNATURE=np.array([1,1,0,1,0,0,1,0,1,1,0,1],dtype=np.uint8)
METHODS=('clahe_pre','flatfield_pre','e12_otsu')

def header_bytes(side,length):
    first=struct.pack('>2sBBH',b'PL',1,side,length)
    return first+zlib.crc32(first).to_bytes(4,'big')

def build_symbol(payload):
    payload=bytes(payload)
    if not payload or len(payload)>65535:raise ValueError('payload length outside 1..65535')
    data,_=hamming_encode(bits_of(make_packet(payload)))
    side=next((s for s in SIDES if 140+len(data)<=s*s),None)
    if side is None:raise ValueError('payload too large for E15 supported dimensions')
    hdr,_=hamming_encode(bits_of(header_bytes(side,len(payload))))
    body=np.zeros(side*side,dtype=np.uint8)
    body[:len(hdr)]=hdr
    body[len(hdr):len(hdr)+len(data)]=data
    # Tail is intentionally not part of the coded payload.
    body[len(hdr)+len(data):]=np.arange(len(body)-len(hdr)-len(data),dtype=np.uint8)%2
    n=side+2*BORDER
    grid=np.zeros((n,n),np.uint8)
    grid[0,:]=1;grid[-1,:]=1;grid[:,0]=1;grid[:,-1]=1
    grid[2,4:16]=SIGNATURE
    grid[2,2]=1;grid[-3,-3]=1
    grid[BORDER:-BORDER,BORDER:-BORDER]=body.reshape(side,side)
    return grid

def render(grid,pixels=600):
    if pixels<240:raise ValueError('pixels>=240')
    margin=round(pixels*.1);w=pixels-2*margin
    image=np.full((pixels,pixels),255,np.uint8)
    image[margin:margin+w,margin:margin+w]=cv2.resize((1-grid)*255,(w,w),interpolation=cv2.INTER_NEAREST)
    return image

def decode_body(inner):
    side=inner.shape[0]
    raw=inner.ravel()
    if len(raw)<140:return None
    try:
        header,_=hamming_decode(raw[:140])
        h=bytes_of(header)
        if len(h)!=10 or h[:2]!=b'PL' or h[2]!=1 or h[3]!=side:return None
        if zlib.crc32(h[:6]).to_bytes(4,'big')!=h[6:10]:return None
        length=int.from_bytes(h[4:6],'big')
        if length<1:return None
        nbits=(length+6)*14
        if 140+nbits>len(raw):return None
        restored,_=hamming_decode(raw[140:140+nbits])
        packet=bytes_of(restored)
        if len(packet)!=length+6 or packet[:2]!=length.to_bytes(2,'big'):return None
        payload=packet[2:2+length]
        if packet[-4:]!=zlib.crc32(payload).to_bytes(4,'big'):return None
        return payload
    except (ValueError,IndexError,struct.error):return None

def quads(image):
    binary=cv2.threshold(cv2.GaussianBlur(image,(3,3),0),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours,key=cv2.contourArea,reverse=True)[:8]:
        if cv2.contourArea(c)<image.size*.065:continue
        hull=cv2.convexHull(c)
        p=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
        if len(p)!=4:continue
        q=p.reshape(4,2).astype(np.float32)
        sums=q.sum(axis=1);dif=q[:,0]-q[:,1]
        yield np.float32([q[np.argmin(sums)],q[np.argmax(dif)],q[np.argmax(sums)],q[np.argmin(dif)]])

def sample_quad(image,quad,side):
    n=side+2*BORDER
    resolution=n*12
    target=np.float32([[0,0],[resolution-1,0],[resolution-1,resolution-1],[0,resolution-1]])
    H=cv2.getPerspectiveTransform(quad,target)
    warped=cv2.warpPerspective(image,H,(resolution,resolution),borderValue=255)
    samples=warped.reshape(n,12,n,12)[:,3:9,:,3:9].mean(axis=(1,3)).astype(np.uint8)
    otsu=cv2.threshold(samples,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return (otsu<128).astype(np.uint8)

def decode_image(image,methods=METHODS):
    """Return recovered payload or None; supported sizes are discovered from symbol geometry/CRC."""
    if image is None:return None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for method in methods:
        normalized=preprocess(image,method)
        for quad in quads(normalized):
            for side in SIDES:
                sample=sample_quad(normalized,quad,side)
                if min(sample[0].mean(),sample[-1].mean(),sample[:,0].mean(),sample[:,-1].mean())<.75:continue
                for rotation in range(4):
                    oriented=np.rot90(sample,rotation)
                    if np.count_nonzero(oriented[2,4:16]!=SIGNATURE)>1:continue
                    payload=decode_body(oriented[BORDER:-BORDER,BORDER:-BORDER])
                    if payload is not None:return payload
    return None

def run(repetitions=30,seed=42,pixels=600,out=Path('resultados_e15')):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    payloads=[('corto',b'Hola'),('referencia',b'Hola desde Tabasco'),('largo',('PixelLogic E15: '+('prueba cientifica '*3)).encode())]
    scenarios=[('clean',0),('illumination',.8),('perspective',.065),('blur',2.5),('occlusion',.16)]
    rows=[]
    for pi,(label,payload) in enumerate(payloads):
        grid=build_symbol(payload);base=render(grid,pixels)
        cv2.imwrite(str(out/f'simbolo_{label}.png'),base)
        for ci,(kind,strength) in enumerate(scenarios):
            for r in range(repetitions):
                s=seed+pi*100000+ci*10000+r
                img=base if kind=='clean' else degrade(base,kind,strength,np.random.default_rng(s))
                decoded=decode_image(img)
                rows.append(dict(payload=label,bytes=len(payload),side=grid.shape[0],scenario=kind,strength=strength,seed=s,valid=int(decoded==payload)))
            section=rows[-repetitions:]
            print(f'{label:11} {kind:12} {sum(x["valid"] for x in section)}/{repetitions} = {np.mean([x["valid"] for x in section]):.1%}')
    with (out/'trials_e15.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('CAUTION: synthetic-only simulations, no QR superiority inferred; symbol is a research prototype.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repetitions',type=int,default=30);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--pixels',type=int,default=600);ap.add_argument('--out',type=Path,default=Path('resultados_e15'))
    args=ap.parse_args();run(args.repetitions,args.seed,args.pixels,args.out)
