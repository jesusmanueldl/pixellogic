"""E25 experimental: geometry recovery from global ink extent. E23 preserved."""
import cv2, numpy as np
import e23, e15, e22
from e13 import preprocess

def bounds_quad(im):
    bw=cv2.threshold(cv2.GaussianBlur(im,(3,3),0),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    n,lab,stats,cen=cv2.connectedComponentsWithStats(bw,8)
    # Use all foreground: synthetic and controlled clean white background; reject excessive darkness.
    yy,xx=np.nonzero(bw)
    if len(xx)<im.size*.025:return []
    x0,x1=int(xx.min()),int(xx.max());y0,y1=int(yy.min()),int(yy.max())
    if min(x1-x0,y1-y0)<im.shape[0]*.25:return []
    q=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    # Alternate robust tilted rectangle based on foreground points
    rect=cv2.minAreaRect(np.column_stack([xx,yy]).astype(np.float32))
    pts=cv2.boxPoints(rect)
    sums=pts.sum(axis=1); diff=pts[:,0]-pts[:,1]
    q2=np.float32([pts[np.argmin(sums)],pts[np.argmax(diff)],pts[np.argmax(sums)],pts[np.argmin(diff)]])
    return [q,q2]

def decode_image(im,return_stage=False):
    p,route=e23.decode_image(im,return_stage=True)
    if p is not None:return (p,route) if return_stage else p
    if im.ndim==3:im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    for method in ('e12_otsu','clahe_pre','flatfield_pre'):
      norm=preprocess(im,method)
      for q in bounds_quad(norm):
        for side in e22.SIDES:
          grid=e15.sample_quad(norm,q,side)
          for r in range(4):
            oriented=np.rot90(grid,r)
            # Signature optional: CRC of header and payload is decisive.
            inner=oriented[e15.BORDER:-e15.BORDER,e15.BORDER:-e15.BORDER]
            p=e22.decode_body(inner)
            if p is not None:return (p,'bounding_header') if return_stage else p
            if np.count_nonzero(oriented[2,4:16]!=e15.SIGNATURE)<=2:
              p,_=e23.decode_body_rescue(inner,max_errors=2)
              if p is not None:return (p,'bounding_rescue') if return_stage else p
    return (None,'failed') if return_stage else None
