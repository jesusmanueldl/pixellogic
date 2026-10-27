"""E29 research: fast geometry hypothesis ordering, CRC-validated; E28 fallback."""
import heapq,time
import cv2,numpy as np
import e15,e22,e23,e25,e28
from e13 import preprocess

def decode_image(image, *, shortlist=20, fallback=True, max_candidates=120, return_info=False):
    if image is None:
        info={'route':'no_image','ranked':0,'seconds':0}
        return (None,info) if return_info else None
    t0=time.perf_counter()
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Fast path: no ranking overhead for already-readable images.
    p0,info0=e28.decode_image(image,max_candidates=12,return_info=True)
    if p0 is not None:
        info={"route":"fast_"+info0["route"],"ranked":0,"seconds":time.perf_counter()-t0}
        return (p0,info) if return_info else p0
    # Rank the remaining candidates using only observed pixels, not ground-truth geometry.
    chosen=[];counter=0
    for method in ('e12_otsu','clahe_pre'):
        norm=preprocess(image,method)
        quads=list(e15.quads(norm))[:3]+e25.bounds_quad(norm)
        for quad in quads:
            center=quad.mean(axis=0)
            for scale in (1.0,1.02,1.05,1.09):
                q=(center+(quad-center)*scale).astype(np.float32)
                for side in e22.SIDES:
                    grid=e15.sample_quad(norm,q,side)
                    # Signature / frame both cheap and discriminative (robust to partial occlusion)
                    frame=np.concatenate([grid[0],grid[-1],grid[1:-1,0],grid[1:-1,-1]])
                    frame_bad=int(np.count_nonzero(frame==0))
                    for turn in range(4):
                        oriented=np.rot90(grid,turn)
                        sig=int(np.count_nonzero(oriented[2,4:16]!=e15.SIGNATURE))
                        if sig>5:continue
                        cost=sig*4+frame_bad*0.08
                        # Keep complete evidence only for top candidates.
                        if len(chosen)<shortlist:
                            counter+=1;heapq.heappush(chosen,(-cost,counter,oriented.copy(),sig))
                        elif cost < -chosen[0][0]:
                            counter+=1;heapq.heapreplace(chosen,(-cost,counter,oriented.copy(),sig))
    ranked=sorted(chosen,key=lambda t:-t[0]);tries=0
    for _,_,grid,sig in ranked:
        tries+=1
        inner=grid[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
        if sig<=2:
            p=e22.decode_body(inner)
            if p is not None:
                info={'route':'ranked_header','ranked':tries,'seconds':time.perf_counter()-t0}
                return (p,info) if return_info else p
        p,_=e23.decode_body_rescue(inner,max_errors=2)
        if p is not None:
            info={'route':'ranked_rescue','ranked':tries,'seconds':time.perf_counter()-t0}
            return (p,info) if return_info else p
    if fallback:
        p,old=e28.decode_image(image,max_candidates=max_candidates,return_info=True)
        info={'route':'fallback_'+old['route'],'ranked':tries,'seconds':time.perf_counter()-t0}
    else:
        p=None;info={'route':'exhausted','ranked':tries,'seconds':time.perf_counter()-t0}
    return (p,info) if return_info else p
