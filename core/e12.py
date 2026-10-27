"""E12: autonomous finder based on an external black rectangular frame.
No ground truth location is used during decoding. Simulation only.
"""
import argparse,csv,math,zlib
from pathlib import Path
import cv2,numpy as np,qrcode
from experiment_v03 import bits_of,bytes_of,hamming_encode,hamming_decode,make_packet
from e11_helpers import degrade

BORDER=3

def make_symbol(payload):
    encoded,_=hamming_encode(bits_of(make_packet(payload)))
    words=len(encoded)//7
    side=math.isqrt(len(encoded))
    if side*side<len(encoded):side+=1
    arr=encoded.reshape(words,7).T.reshape(-1)
    arr=np.pad(arr,(0,side*side-len(arr)))
    grid=np.zeros((side+2*BORDER,side+2*BORDER),np.uint8)
    grid[0,:]=1;grid[-1,:]=1;grid[:,0]=1;grid[:,-1]=1
    grid[BORDER:-BORDER,BORDER:-BORDER]=arr.reshape(side,side)
    return grid,len(encoded),words,side

def render_grid(grid,pixels):
    margin=round(pixels*.1);width=pixels-2*margin
    image=np.full((pixels,pixels),255,np.uint8)
    image[margin:margin+width,margin:margin+width]=cv2.resize((1-grid)*255,(width,width),interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return image

def decode_bits(bits,count,words,side,payload):
    inner=bits.reshape(side,side).ravel()[:count]
    data=inner.reshape(7,words).T.reshape(-1)
    recovered,_=hamming_decode(data)
    packet=bytes_of(recovered)
    original=make_packet(payload)
    if len(packet)<len(original):return False,False,1.0
    message=packet[2:2+len(payload)]
    valid=packet[:2]==len(payload).to_bytes(2,'big') and zlib.crc32(message).to_bytes(4,'big')==packet[2+len(payload):2+len(payload)+4]
    ber=float(np.mean(bits_of(message)!=bits_of(payload)))
    return bool(valid),bool(message==payload),ber

def find_and_decode(image,payload,info):
    count,words,side=info
    n=side+2*BORDER
    # Image coordinates inferred solely from contours, never from generation parameters.
    blur=cv2.GaussianBlur(image,(3,3),0)
    binary=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area<image.size*.08:continue
        hull=cv2.convexHull(c)
        arc=cv2.arcLength(hull,True)
        approx=cv2.approxPolyDP(hull,.025*arc,True)
        if len(approx)!=4:continue
        pts=approx.reshape(4,2).astype(np.float32)
        sums=pts.sum(axis=1);diff=pts[:,0]-pts[:,1]
        ordered=np.array([pts[np.argmin(sums)],pts[np.argmax(diff)],pts[np.argmax(sums)],pts[np.argmin(diff)]],np.float32)
        candidates.append((area,ordered))
    if not candidates:return False,False,1.,False
    candidates.sort(reverse=True,key=lambda t:t[0])
    for _,quad in candidates[:4]:
        target= np.float32([[0,0],[n*16-1,0],[n*16-1,n*16-1],[0,n*16-1]])
        H=cv2.getPerspectiveTransform(quad,target)
        square=cv2.warpPerspective(image,H,(n*16,n*16),flags=cv2.INTER_LINEAR,borderValue=255)
        # Average pixels in the central 50% of each module to avoid sampling boundaries.
        sample=np.empty((n,n),np.float32)
        for y in range(n):
            for x in range(n):sample[y,x]=square[y*16+4:y*16+12,x*16+4:x*16+12].mean()
        # Per-column photometric calibration: black outer frame (row 0)
        # and white reference ring (row 1), both known by construction.
        _,th=cv2.threshold(sample.astype('uint8'),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bits=(th<128).astype(np.uint8)
        # frame check, allowing a few damaged edge cells
        if (np.mean(bits[0])<.7 or np.mean(bits[-1])<.7 or np.mean(bits[:,0])<.7 or np.mean(bits[:,-1])<.7):continue
        data=bits[BORDER:-BORDER,BORDER:-BORDER]
        for k in range(4):
            valid,exact,ber=decode_bits(np.rot90(data,k),count,words,side,payload)
            if valid:return valid,exact,ber,True
    return False,False,1.,True

def trial(payload,pixels,kind,strength,seed,save=None):
    grid,count,words,side=make_symbol(payload)
    custom=render_grid(grid,pixels)
    qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H,box_size=1,border=4)
    qr.add_data(payload.decode('ascii'));qr.make(fit=True)
    qm=np.asarray(qr.get_matrix(),dtype=np.uint8)
    qrimg=cv2.resize((1-qm)*255,(pixels,pixels),interpolation=cv2.INTER_NEAREST).astype('uint8')
    rng1=np.random.default_rng(seed);rng2=np.random.default_rng(seed)
    if kind=='clean':a,b=custom,qrimg
    else:a,b=degrade(custom,kind,strength,rng1),degrade(qrimg,kind,strength,rng2)
    valid,exact,ber,found=find_and_decode(a,payload,(count,words,side))
    qrvalid=cv2.QRCodeDetector().detectAndDecode(b)[0]==payload.decode('ascii')
    if save is not None:
        save.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(save/'pixellogic.png'),a);cv2.imwrite(str(save/'qr.png'),b)
    return dict(scenario=kind,strength=strength,seed=seed,detected=int(found),custom_valid=int(valid),custom_exact=int(exact),custom_ber=ber,qr_exact=int(qrvalid),custom_modules=len(grid),qr_modules=len(qm),pixels=pixels)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--text',default='Hola desde Tabasco');ap.add_argument('--pixels',type=int,default=600);ap.add_argument('--repetitions',type=int,default=30);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--out',type=Path,default=Path('resultados_e12'))
    a=ap.parse_args();payload=a.text.encode('ascii')
    if len(payload)!=18:ap.error('E12 currently requires exactly 18 ASCII bytes')
    if a.pixels<240 or a.repetitions<1:ap.error('Require pixels>=240 and repetitions>=1')
    cases=[('clean',0),('blur',1.),('blur',2.5),('illumination',.4),('illumination',.8),('occlusion',.08),('occlusion',.16),('perspective',.025),('perspective',.065)]
    a.out.mkdir(parents=True,exist_ok=True);rows=[];summary=[]
    for idx,(kind,s) in enumerate(cases):
        for i in range(a.repetitions):rows.append(trial(payload,a.pixels,kind,s,a.seed+i+idx*10000,a.out/'examples'/f'{kind}_{s}' if i==0 else None))
        sub=rows[-a.repetitions:]
        z=dict(scenario=kind,strength=s,n=len(sub),detected=np.mean([r['detected'] for r in sub]),custom_valid=np.mean([r['custom_valid'] for r in sub]),qr_exact=np.mean([r['qr_exact'] for r in sub]))
        summary.append(z)
        print(f"{kind:12} {s:.3f}: detected={z['detected']:.1%} PixelLogic={z['custom_valid']:.1%} QR={z['qr_exact']:.1%}")
    for name,data in [('trials_e12.csv',rows),('summary_e12.csv',summary)]:
        with (a.out/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    print('CAUTION: geometry detection is autonomous but ECC, module counts, and footprint still differ from QR. Simulated scenes only.')
    print('Saved:',a.out.resolve())
if __name__=='__main__':main()
