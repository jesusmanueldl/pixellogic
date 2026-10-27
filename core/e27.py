"""E27 prioritized geometric CRC-verified decoder. No known location/content or occlusion oracle.
E23 primary; prioritized bbox variants before E25; E26 remains available for offline audit.
"""
import cv2,numpy as np
import e23,e25,e15,e22
from e13 import preprocess

def decode_image(image,return_stage=False):
    if image is None:return (None,'no_image') if return_stage else None
    payload,route=e23.decode_image(image,return_stage=True)
    if payload is not None:return (payload,route) if return_stage else payload
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Method choice determined only by the input pixels, not ground truth.
    norm=preprocess(image,'e12_otsu')
    for quad in e25.bounds_quad(norm):
      c=quad.mean(axis=0)
      for scale in (1.0,1.02,1.05,1.09):
       q=(c+(quad-c)*scale).astype(np.float32)
       for side in e22.SIDES:
        grid=e15.sample_quad(norm,q,side)
        for rotation in range(4):
         oriented=np.rot90(grid,rotation)
         sig=int(np.count_nonzero(oriented[2,4:16]!=e15.SIGNATURE))
         if sig>6:continue
         inner=oriented[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
         if sig<=2:
          p=e22.decode_body(inner)
          if p is not None:return (p,'fast_header') if return_stage else p
         p,_=e23.decode_body_rescue(inner,max_errors=2)
         if p is not None:return (p,'fast_rescue') if return_stage else p
    # Retain E25's wider preprocessing search as bounded logical fallback;
    # no invented geometry or knowledge of source data.
    p,route=e25.decode_image(image,True)
    return (p,route) if return_stage else p
