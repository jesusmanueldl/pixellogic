import e22,e15
import numpy as np
from e11_helpers import degrade
from e13 import preprocess

def probe(img, payload):
 stages={'quad':False,'frame':False,'signature':False,'header':False,'body':False}
 for method in ('clahe_pre','flatfield_pre','e12_otsu'):
  norm=preprocess(img,method)
  for q in e15.quads(norm):
   stages['quad']=True
   for side in e22.SIDES:
    g=e15.sample_quad(norm,q,side)
    if min(g[0].mean(),g[-1].mean(),g[:,0].mean(),g[:,-1].mean())<.75:continue
    stages['frame']=True
    for turn in range(4):
     rot=np.rot90(g,turn)
     if np.count_nonzero(rot[2,4:16]!=e15.SIGNATURE)>1:continue
     stages['signature']=True
     inner=rot[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
     from experiment_v03 import hamming_decode,bytes_of
     import zlib
     try:
      hb,_=hamming_decode(inner.ravel()[:140]);h=bytes_of(hb)
      if len(h)==10 and h[:2]==b'PL' and h[2]==2 and h[3]==side and zlib.crc32(h[:6]).to_bytes(4,'big')==h[6:]:stages['header']=True
     except Exception:pass
     if e22.decode_body(inner)==payload:stages['body']=True;return stages
 return stages

if __name__=='__main__':
 for i,(name,payload) in enumerate([('corto',b'Hola'),('referencia',b'Hola desde Tabasco'),('largo',('PixelLogic E22: '+'prueba cientifica '*3).encode())]):
  base=e15.render(e22.build_symbol(payload),600)
  stats={k:0 for k in ('quad','frame','signature','header','body')}
  for rep in range(30):
   sd=42+i*100000+4*10000+rep
   img=degrade(base,'occlusion',.16,np.random.default_rng(sd))
   r=probe(img,payload)
   for k in stats:stats[k]+=int(r[k])
  print(name,stats,flush=True)
