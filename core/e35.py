"""E35 experimental: perspective-aware recovery of E31 distributed anchors."""
from itertools import combinations
import cv2
import numpy as np
import e31,e22,e23,e15,e32
from e13 import preprocess


def anchors(image, thresholds=(65,85,105,125,145)):
    height,width=image.shape
    proposals=[]
    for threshold in thresholds:
        binary=(cv2.GaussianBlur(image,(3,3),0)<threshold).astype(np.uint8)
        n,_,stats,centers=cv2.connectedComponentsWithStats(binary,8)
        for i in range(1,n):
            x,y,w,h,a=map(int,stats[i]); ratio=min(w,h)/max(w,h)
            if a<250 or w<18 or h<18 or w>min(width,height)*.18 or h>min(width,height)*.18 or ratio<.4:continue
            fill=a/(w*h)
            if fill<.5:continue
            p=centers[i]; sz=np.sqrt(w*h)
            if any(np.linalg.norm(p-v[1])<10 for v in proposals):continue
            proposals.append((float(a),p.astype(np.float32),float(sz)))
    return sorted(proposals,key=lambda z:z[0],reverse=True)[:14]


def candidates(image):
    a=anchors(image)
    out=[]
    for combo in combinations(a[:10],4):
        p=np.array([v[1] for v in combo],np.float32)
        q=e31._assign(p)
        if q is None:continue
        edges=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
        if min(edges)<100:continue
        area=cv2.contourArea(q.reshape(-1,1,2));
        if area<.025*image.size:continue
        sizes=np.array([v[2] for v in combo]);
        if sizes.max()/sizes.min()>2.8:continue
        # quadrilateral convexity
        if not cv2.isContourConvex(q.reshape(-1,1,2)):continue
        score=-area/max(image.size,1)+.01*(sizes.max()/sizes.min())
        out.append((score,q,float(np.median(sizes))))
    out.sort(key=lambda x:x[0]);return out[:8]


def decode_image(image,return_info=False):
    prior,info=e32.decode_image(image,return_info=True)
    if prior is not None:return (prior,info) if return_info else prior
    if image is None:return (None,{'stage':'empty'}) if return_info else None
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    for label in ('e12_otsu','flatfield_pre','clahe_pre'):
        pre=preprocess(gray,label)
        # use source grey for geometric sample; threshold is applied inside sample_quad
        for _,q,size in candidates(pre):
            for side in e22.SIDES:
                n=side+8
                # geometrical ratio rough prior, perspective can cause anisotropy
                edge=np.median(np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1))
                if not .45 < 3*edge/(size*(n-3)) < 2.2:continue
                xy=np.float32([[1.5,1.5],[n-1.5,1.5],[n-1.5,n-1.5],[1.5,n-1.5]])
                H=cv2.getPerspectiveTransform(xy,q)
                bounds=cv2.perspectiveTransform(np.float32([[[0,0],[n,0],[n,n],[0,n]]]),H)[0]
                for sample_im in (pre,gray):
                    sample=e15.sample_quad(sample_im,bounds.astype(np.float32),side)
                    for turn in range(4):
                        inner=np.rot90(sample,turn)[4:-4,4:-4]
                        options=e23.candidates(side)
                        if not options:continue
                        observed=inner.ravel()[:140]
                        mindist=min(int(np.count_nonzero(observed!=hb)) for _,hb in options)
                        if mindist>22:continue
                        result=e22.decode_body(inner,max_errors=0)
                        if result is None:result,_=e23.decode_body_rescue(inner,max_errors=0)
                        if result is not None:
                            info={'stage':'success','method':'e35_'+label,'side':side}
                            return (result,info) if return_info else result
    return (None,{'stage':'failed'}) if return_info else None
