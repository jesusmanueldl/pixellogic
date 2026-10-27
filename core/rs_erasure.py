"""Systematic Cauchy Reed-Solomon-like MDS code over GF(256), erasure-only.
Any <= parity lost bytes recoverable if positions known. NOT an errors decoder.
"""
POLY=0x11D
EXP=[0]*512; LOG=[0]*256
v=1
for i in range(255):
    EXP[i]=v; LOG[v]=i
    v<<=1
    if v&256: v^=POLY
for i in range(255,512): EXP[i]=EXP[i-255]
def mul(a,b): return 0 if not a or not b else EXP[LOG[a]+LOG[b]]
def inv(a):
    if not a: raise ZeroDivisionError()
    return EXP[255-LOG[a]]
def coeff(i,j,k):
    # x_i=i, y_j=k+j distinct for k+parity<=255
    return inv(i^(k+j))
def encode(payload, parity):
    k=len(payload)
    if not 1<=k<=255 or parity<1 or k+parity>255: raise ValueError('invalid length')
    result=bytearray(payload)
    for j in range(parity):
        out=0
        for i,x in enumerate(payload): out ^= mul(x,coeff(i,j,k))
        result.append(out)
    return bytes(result)
def solve(a,b):
    n=len(b)
    a=[list(r)+[int(z)] for r,z in zip(a,b)]
    for col in range(n):
        p=next((i for i in range(col,n) if a[i][col]),None)
        if p is None: raise ValueError('singular')
        a[col],a[p]=a[p],a[col]
        fac=inv(a[col][col]); a[col]=[mul(z,fac) for z in a[col]]
        for r in range(n):
            if r==col or not a[r][col]:continue
            fac=a[r][col]
            a[r]=[u^mul(fac,w) for u,w in zip(a[r],a[col])]
    return bytes(row[-1] for row in a)
def decode_erasures(received,k,erasures):
    n=len(received); p=n-k
    bad=set(erasures)
    if not 1<=k<n<=255 or any(i<0 or i>=n for i in bad): raise ValueError('invalid positions')
    if len(bad)>p:return None
    missing=sorted(i for i in bad if i<k)
    good_parity=[j for j in range(p) if k+j not in bad]
    if len(good_parity)<len(missing):return None
    if not missing:return bytes(received[:k])
    a=[]; b=[]
    for j in good_parity[:len(missing)]:
        row=[coeff(i,j,k) for i in missing]
        rhs=received[k+j]
        for i in range(k):
            if i not in bad:rhs ^= mul(received[i],coeff(i,j,k))
        a.append(row);b.append(rhs)
    recovered=solve(a,b)
    out=bytearray(received[:k])
    for i,x in zip(missing,recovered):out[i]=x
    return bytes(out)
