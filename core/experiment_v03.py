"""PixelLogic Lab v0.3: controlled Hamming(7,4) experiment, no external deps beyond NumPy.
Same encoded information, same footprint and matched damage masks for every model.
"""
import argparse, csv, math, zlib
from pathlib import Path
import numpy as np

# Systematic Hamming(7,4): positions 1,2,4 are parity; 3,5,6,7 are data.
DATA_IDX=np.array([2,4,5,6]); PAR_IDX=np.array([0,1,3])

def bits_of(data): return np.unpackbits(np.frombuffer(data,dtype=np.uint8)).astype(np.uint8)
def bytes_of(bits): return np.packbits(np.asarray(bits,dtype=np.uint8)).tobytes()

def hamming_encode(bits):
    x=np.asarray(bits,dtype=np.uint8)
    pad=(-len(x))%4
    if pad:x=np.pad(x,(0,pad))
    c=np.zeros((len(x)//4,7),dtype=np.uint8)
    c[:,DATA_IDX]=x.reshape(-1,4)
    # parity checking over one-indexed positions
    for p in [1,2,4]:
        mask=np.array([(i & p)!=0 for i in range(1,8)])
        c[:,p-1]=np.bitwise_xor.reduce(c[:,mask],axis=1)
    return c.reshape(-1),pad

def hamming_decode(bits):
    c=np.asarray(bits,dtype=np.uint8).reshape(-1,7).copy()
    syndrome=np.zeros(len(c),dtype=int)
    for p in (1,2,4):
        mask=np.array([(i & p)!=0 for i in range(1,8)])
        syndrome+=p*np.bitwise_xor.reduce(c[:,mask],axis=1)
    bad=np.where(syndrome>0)[0]
    c[bad,syndrome[bad]-1]^=1
    # Double errors are not reliably detected by plain (7,4) code.
    return c[:,DATA_IDX].reshape(-1),len(bad)

def cellular_step(a,shift):
    """Reversible local additive coupling, even/odd row partition, periodic columns."""
    b=a.copy()
    b[1::2]^=np.roll(b[0::2],shift,axis=1)
    b[0::2]^=np.roll(b[1::2],-shift,axis=1)
    return b

def cellular_inverse(a,shift):
    b=a.copy()
    b[0::2]^=np.roll(b[1::2],-shift,axis=1)
    b[1::2]^=np.roll(b[0::2],shift,axis=1)
    return b

def transform(bits,shape,method,rounds,permutation=None,inverse=False):
    a=np.asarray(bits,dtype=np.uint8).copy()
    if method=='direct':return a
    if method=='interleaver':
        return a[np.argsort(permutation)] if inverse else a[permutation]
    if method!='cellular':raise ValueError(method)
    x=a.reshape(shape)
    if inverse:
        for t in range(rounds-1,-1,-1): x=cellular_inverse(x,1+t%3)
    else:
        for t in range(rounds):x=cellular_step(x,1+t%3)
    return x.reshape(-1)

def error_mask(shape,rate,kind,rng):
    h,w=shape; count=int(round(h*w*rate)); mask=np.zeros(h*w,dtype=bool)
    if count==0:return mask
    if kind=='random':
        mask[rng.choice(h*w,size=count,replace=False)]=True
    elif kind=='burst':
        start=int(rng.integers(0,h*w))
        mask[(start+np.arange(count))%(h*w)]=True
    elif kind=='rectangle':
        # Actual 2D localized blemish; exactly count sites, connected where possible.
        height=min(h,max(1,int(round(math.sqrt(count*h/w)))))
        width=min(w,max(1,int(math.ceil(count/height))))
        r=int(rng.integers(h));c=int(rng.integers(w))
        points=[((r+rr)%h)*w+(c+cc)%w for rr in range(height) for cc in range(width)]
        mask[points[:count]]=True
    else:raise ValueError(kind)
    return mask

def make_packet(payload):return len(payload).to_bytes(2,'big')+payload+zlib.crc32(payload).to_bytes(4,'big')

def experiment(payload,rates,kinds,repetitions,rounds,seed):
    packet=make_packet(payload); raw=bits_of(packet)
    protected,pad=hamming_encode(raw)
    width=28 # multiple of 7; pad only complete Hamming codewords
    rows=math.ceil(len(protected)/width)
    if rows%2:rows+=1
    pad_cw=(rows*width-len(protected))//7
    protected=np.concatenate((protected,np.tile(hamming_encode(np.zeros(4,dtype=np.uint8))[0],pad_cw)))
    shape=(rows,width); N=len(protected)
    perm=np.random.default_rng(seed+98765).permutation(N)
    methods=['direct','interleaver','cellular']; records=[]
    encoded={m:transform(protected,shape,m,rounds,perm) for m in methods}
    for kind in kinds:
        for rate in rates:
            for rep in range(repetitions):
                rng=np.random.default_rng(seed+rep+10007*kinds.index(kind)+100003*rates.index(rate))
                mask=error_mask(shape,rate,kind,rng)
                for method in methods:
                    corrupted=encoded[method].copy();corrupted[mask]^=1
                    restored=transform(corrupted,shape,method,rounds,perm,inverse=True)
                    decoded,corrected=hamming_decode(restored)
                    output=bytes_of(decoded[:len(raw)])
                    extracted=output[2:2+len(payload)]
                    valid= (len(output)==len(packet) and output[:2]==len(payload).to_bytes(2,'big')
                            and zlib.crc32(extracted).to_bytes(4,'big')==output[-4:])
                    errors=int(np.count_nonzero(bits_of(extracted)!=bits_of(payload)))
                    records.append(dict(method=method,kind=kind,rate=rate,repeat=rep,seed=seed+rep,
                        rounds=rounds,visual_bits=N,payload_bits=len(payload)*8,
                        flipped_bits=int(mask.sum()),recovered_payload_errors=errors,
                        ber=errors/(len(payload)*8),payload_equal=(extracted==payload),
                        valid_packet=valid,corrected_codewords=corrected))
    return records,shape

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--text',default='Hola desde Tabasco')
    p.add_argument('--rates',default='0,0.01,0.02,0.05,0.10')
    p.add_argument('--kinds',default='random,burst,rectangle')
    p.add_argument('--repetitions',type=int,default=100)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path,default=Path('resultados_v03'))
    a=p.parse_args();rates=[float(t) for t in a.rates.split(',')];kinds=a.kinds.split(',')
    if not a.text or a.repetitions<1 or a.rounds<0 or a.rounds>1000 or any(r<0 or r>1 for r in rates) or any(k not in ('random','burst','rectangle') for k in kinds):p.error('Invalid parameters')
    records,shape=experiment(a.text.encode('utf-8'),rates,kinds,a.repetitions,a.rounds,a.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    group={}
    for r in records:group.setdefault((r['method'],r['kind'],r['rate']),[]).append(r)
    rows=[]
    for (method,kind,rate),rs in sorted(group.items()):
        rows.append(dict(method=method,kind=kind,rate=rate,n=len(rs),mean_ber=float(np.mean([x['ber'] for x in rs])),
            exact_payload_rate=float(np.mean([x['payload_equal'] for x in rs])),
            valid_packet_rate=float(np.mean([x['valid_packet'] for x in rs])),visual_bits=rs[0]['visual_bits']))
    with (a.out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f'PixelLogic v0.3: {len(records)} ensayos; matriz {shape[0]}x{shape[1]}; {records[0]["visual_bits"]} bits')
    print('Igual carga util, Hamming(7,4), imagen y mascaras de errores.')
    for method in ('direct','interleaver','cellular'):
        rr=[r for r in records if r['method']==method and r['rate']>0]
        print(f'{method:12} BER={np.mean([r["ber"] for r in rr]):.3%}; recuperacion exacta={np.mean([r["payload_equal"] for r in rr]):.3%}')
    print('Archivos:',a.out.resolve())
if __name__=='__main__':main()
