"""E30: line-supported bounded geometric recovery experiment.
Classical Hough lines (Duda-Hart 1972); candidate CRC-protected. No image oracle.
E28 kept intact as baseline. Research prototype: synthetic validation only.
"""
import time
import cv2
import numpy as np
import e15,e22,e23,e28
from e13 import preprocess

def hough_quads(im):
    h,w=im.shape
    ed=cv2.Canny(im,45,140)
    lines=cv2.HoughLinesP(ed,1,np.pi/180,threshold=34,minLineLength=int(.19*min(h,w)),maxLineGap=int(.12*min(h,w)))
    if lines is None:return []
    lines=lines.reshape(-1,4)
    hor=[];ver=[]
    for x1,y1,x2,y2 in lines:
        dx=x2-x1;dy=y2-y1
        if abs(dx)>=2.5*abs(dy):hor.extend((y1,y2))
        if abs(dy)>=2.5*abs(dx):ver.extend((x1,x2))
    if len(hor)<4 or len(ver)<4:return []
    # Quantile-based trimming ignores isolated short spurious segments; quantiles 0/100 and 5/95.
    out=[]
    for lo,hi in [(0,100),(5,95)]:
        xl,xr=np.percentile(ver,[lo,hi]);yt,yb=np.percentile(hor,[lo,hi])
        if min(xr-xl,yb-yt)<.22*min(h,w):continue
        out.append(np.float32([[xl,yt],[xr,yt],[xr,yb],[xl,yb]]))
    return out

def decode_image(image, *, max_algebra=36, return_info=False):
    t0=time.perf_counter()
    if image is None:
        return (None,{'route':'no_image','candidates':0,'seconds':0}) if return_info else None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Legacy bounded search is first: exact backward compatibility on its successes.
    payload,prev=e28.decode_image(image,max_candidates=30,return_info=True)
    if payload is not None:
        info={'route':'e28_'+prev['route'],'candidates':prev['candidates'],'seconds':time.perf_counter()-t0}
        return (payload,info) if return_info else payload
    used=0
    for method in ('e12_otsu','clahe_pre'):
        normalized=preprocess(image,method)
        for base in hough_quads(normalized):
            ctr=base.mean(axis=0)
            for scale in (1.0,1.025,1.06):
                quad=np.float32(ctr+(base-ctr)*scale)
                for side in e22.SIDES:
                    grid=e15.sample_quad(normalized,quad,side)
                    for rotation in range(4):
                        oriented=np.rot90(grid,rotation)
                        if int(np.count_nonzero(oriented[2,4:16]!=e15.SIGNATURE))>3:continue
                        if used>=max_algebra:break
                        used+=1
                        inner=oriented[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
                        result=e22.decode_body(inner,max_errors=1)
                        if result is None:result,_=e23.decode_body_rescue(inner,max_errors=1)
                        if result is not None:
                            info={'route':'hough','candidates':used,'seconds':time.perf_counter()-t0}
                            return (result,info) if return_info else result
                    if used>=max_algebra:break
                if used>=max_algebra:break
            if used>=max_algebra:break
        if used>=max_algebra:break
    info={'route':'not_recovered','candidates':used,'seconds':time.perf_counter()-t0}
    return (None,info) if return_info else None
