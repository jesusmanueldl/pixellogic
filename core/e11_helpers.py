"""E11: controlled simulated optical channel. Custom symbol uses ORACLE localization.
No claim of competitive optical detection. QR and custom have equal pixel canvas, NOT equal ECC.
"""
import argparse,csv,zlib,math
from pathlib import Path
import cv2,numpy as np,qrcode
from experiment_v03 import bits_of,bytes_of,hamming_encode,hamming_decode,make_packet

def custom_symbol(payload):
    encoded,_=hamming_encode(bits_of(make_packet(payload)))
    # 7x48 Hamming codewords for 18 B payload -> 336 bits
    k=len(encoded)//7
    side=int(math.ceil(math.sqrt(len(encoded))))
    cw=encoded.reshape(k,7)
    flattened=cw.T.reshape(-1)  # block interleaver: positions spaced k apart
    n=side*side
    if n>len(flattened): flattened=np.pad(flattened,(0,n-len(flattened)))
    return flattened.reshape(side,side), len(encoded), k

def custom_decode(image, payload, encoded_size, word_count, border, side):
    # ORACLE: the user must know the frame bounding box exactly
    crop=image[border:image.shape[0]-border,border:image.shape[1]-border]
    if crop.size==0: return False,False,1.
    samples=cv2.resize(crop,(side,side),interpolation=cv2.INTER_AREA)
    visual=(samples<128).astype(np.uint8).reshape(-1)[:encoded_size]
    inv=visual.reshape(7,word_count).T.reshape(-1)
    decoded,_=hamming_decode(inv)
    packet=bytes_of(decoded)
    original=make_packet(payload)
    message=packet[2:2+len(payload)]
    valid=(len(packet)>=len(original) and packet[:2]==len(payload).to_bytes(2,'big') and zlib.crc32(message).to_bytes(4,'big')==packet[2+len(payload):2+len(payload)+4])
    ber=float(np.mean(bits_of(message)!=bits_of(payload))) if len(message)==len(payload) else 1.
    return bool(valid),bool(message==payload),ber

def build_images(payload, pixels):
    # QR with error correction H, let library choose version, maintain equal total side length
    qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H,box_size=1,border=4)
    qr.add_data(payload.decode('ascii'));qr.make(fit=True)
    qr_modules=np.asarray(qr.get_matrix(),dtype=np.uint8)
    code, count, wc=custom_symbol(payload)
    margin=max(4,round(pixels*.10))
    def draw(mod,quiet):
        available=pixels-2*quiet
        # nearest-neighbor with same final resolution, custom gets oracle coordinates
        inner=cv2.resize((1-mod)*255,(available,available),interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        canvas=np.full((pixels,pixels),255,dtype=np.uint8)
        canvas[quiet:quiet+available,quiet:quiet+available]=inner
        return canvas
    custom=draw(code,margin)
    # qr includes standard 4-module quiet zone, centered occupying whole canvas
    qr_img=cv2.resize((1-qr_modules)*255,(pixels,pixels),interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return custom,qr_img,(count,wc,margin,code.shape[0]),qr_modules.shape[0]

def degrade(im,kind,strength,rng):
    n=im.shape[0];out=im.copy()
    if kind=='blur':
        s=max(.01,strength);out=cv2.GaussianBlur(out,(0,0),s)
    elif kind=='illumination':
        x=np.linspace(0,1,n)[None,:];field=(1-strength*.9*x)
        out=np.clip(out.astype(float)*field+strength*24,0,255).astype(np.uint8)
    elif kind=='occlusion':
        # white square of side fraction strength, same pixel location for two codes
        z=max(1,round(n*strength));xx=int(rng.integers(0,n-z+1));yy=int(rng.integers(0,n-z+1));out[yy:yy+z,xx:xx+z]=255
    elif kind=='perspective':
        d=n*strength;offs=rng.uniform(-d,d,(4,2)).astype(np.float32)
        src=np.float32([[0,0],[n-1,0],[n-1,n-1],[0,n-1]])
        H=cv2.getPerspectiveTransform(src,src+offs)
        out=cv2.warpPerspective(out,H,(n,n),borderValue=255,flags=cv2.INTER_LINEAR)
    else: raise ValueError(kind)
    return out

def trial(payload,pixels,kind,strength,seed,save=None):
    a,b,info,qsize=build_images(payload,pixels)
    # independent RNG seeded identically => same damage geometry
    da=degrade(a,kind,strength,np.random.default_rng(seed))
    db=degrade(b,kind,strength,np.random.default_rng(seed))
    cnt,wc,margin,side=info
    # This is an oracle-decoder, and not corrected for perspective intentionally
    cv,cp,ber=custom_decode(da,payload,cnt,wc,margin,side)
    qrread=cv2.QRCodeDetector().detectAndDecode(db)[0]
    qvalid=(qrread==payload.decode('ascii'))
    if save:
        save.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(save/'custom_clean.png'),a)
        cv2.imwrite(str(save/'qr_clean.png'),b)
        cv2.imwrite(str(save/'custom_damage.png'),da)
        cv2.imwrite(str(save/'qr_damage.png'),db)
    return {'scenario':kind,'strength':strength,'seed':seed,'qr_exact':int(qvalid),'custom_oracle_packet':int(cv),'custom_oracle_payload':int(cp),'custom_oracle_ber':ber,'qr_modules_including_quiet':qsize,'custom_modules':side,'canvas_pixels':pixels}

def main():
    p=argparse.ArgumentParser();p.add_argument('--text',default='Hola desde Tabasco');p.add_argument('--pixels',type=int,default=600);p.add_argument('--repetitions',type=int,default=20);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,default=Path('resultados_e11'))
    a=p.parse_args();payload=a.text.encode('ascii')
    if len(payload)!=18: p.error('v0.11 requires exactly 18 ASCII bytes (same as original controlled experiments)')
    if a.pixels<240 or a.repetitions<1:p.error('pixels>=240 and repetitions>=1 required')
    a.out.mkdir(parents=True,exist_ok=True)
    cases=[('blur',1.),('blur',2.5),('illumination',.4),('illumination',.8),('occlusion',.08),('occlusion',.16),('perspective',.025),('perspective',.065)]
    rows=[]
    for kind,s in cases:
        for i in range(a.repetitions):
            rows.append(trial(payload,a.pixels,kind,s,a.seed+i+10000*cases.index((kind,s)),save=(a.out/'examples'/f'{kind}_{s}' if i==0 else None)))
    with (a.out/'trials_e11.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    sums=[]
    for kind,s in cases:
        part=[r for r in rows if r['scenario']==kind and r['strength']==s]
        sums.append(dict(scenario=kind,strength=s,n=len(part),qr_rate=np.mean([r['qr_exact'] for r in part]),custom_oracle_rate=np.mean([r['custom_oracle_packet'] for r in part]),custom_oracle_ber=np.mean([r['custom_oracle_ber'] for r in part])))
    with (a.out/'summary_e11.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sums[0]);w.writeheader();w.writerows(sums)
    print('E11 simulated optical channel; samples:',len(rows),'|',a.pixels,'pixel-square images')
    for s in sums:print(f"{s['scenario']:12} {s['strength']:5.3f}: QR={s['qr_rate']:.1%}; custom ORACLE={s['custom_oracle_rate']:.1%}")
    print('WARNING: NOT FAIR END-TO-END QR COMPARISON; custom has oracle localization and different ECC/visual overhead.')
    print('Saved:',a.out.resolve())
if __name__=='__main__':main()
