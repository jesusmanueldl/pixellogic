"""E26 candidate: constrained quad expansion after failed E25, with CRC verification.
No known symbol coordinates, message, or erasure oracle.
"""
import numpy as np
import cv2
import e25,e23,e15,e22
from e13 import preprocess

def expanded_quads(q):
    center=q.mean(axis=0)
    for scale in (1.02,1.05,1.09):
        yield (center+(q-center)*scale).astype(np.float32)

def decode_image(image,return_stage=False):
    result,stage=e25.decode_image(image,True)
    if result is not None:return (result,stage) if return_stage else result
    if image is None:return (None,'no_image') if return_stage else None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for method in ('e12_otsu',):
        norm=preprocess(image,method)
        for base in e25.bounds_quad(norm):
            for quad in expanded_quads(base):
                for side in e22.SIDES:
                    grid=e15.sample_quad(norm,quad,side)
                    for r in range(4):
                        oriented=np.rot90(grid,r)
                        if np.count_nonzero(oriented[2,4:16]!=e15.SIGNATURE)>6:continue
                        inner=oriented[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
                        p=e22.decode_body(inner)
                        if p is None:p,_=e23.decode_body_rescue(inner,max_errors=2)
                        if p is not None:return (p,'expanded_quad') if return_stage else p
    return (None,'failed') if return_stage else None
