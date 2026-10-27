"""E28 bounded geometric candidate decoder. Does not need source payload/location.
Hard cap on algebraic proposals; ordinary E22/E23 are retained as separate baselines.
"""
import cv2,numpy as np
import e15,e22,e23,e25
from e13 import preprocess

def decode_image(image, *, max_candidates=120, return_info=False):
    if image is None:
        return (None,{'route':'none','candidates':0}) if return_info else None
    if image.ndim==3: image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    used=0
    # Bounded candidate search, no unbounded E25 fallback.
    for method in ('e12_otsu','clahe_pre'):
        norm=preprocess(image,method)
        quads=list(e15.quads(norm))[:3]+e25.bounds_quad(norm)
        for quad in quads:
            center=quad.mean(axis=0)
            for scale in (1.0,1.02,1.05,1.09):
                q=(center+(quad-center)*scale).astype(np.float32)
                for side in e22.SIDES:
                    grid=e15.sample_quad(norm,q,side)
                    for turn in range(4):
                        oriented=np.rot90(grid,turn)
                        sig=int(np.count_nonzero(oriented[2,4:16]!=e15.SIGNATURE))
                        if sig>5: continue
                        if used>=max_candidates:
                            info={'route':'budget','candidates':used}
                            return (None,info) if return_info else None
                        used+=1
                        inner=oriented[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
                        if sig<=2:
                            p=e22.decode_body(inner)
                            if p is not None:
                                return (p,{'route':'header','candidates':used}) if return_info else p
                        p,_=e23.decode_body_rescue(inner,max_errors=2)
                        if p is not None:
                            return (p,{'route':'rescue','candidates':used}) if return_info else p
    info={'route':'exhausted','candidates':used}
    return (None,info) if return_info else None
