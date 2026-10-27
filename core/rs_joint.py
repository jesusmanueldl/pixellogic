"""Compatible algebraic Reed–Solomon errors+erasures decoder for E17/E18 Cauchy MDS.
Berlekamp–Welch solution on equivalent generalized Reed–Solomon evaluations.
Reference implementation; not optimized for real-time or arbitrary >255 symbols.
"""
from rs_erasure import mul,inv,encode

def evalpoly(poly,x):
    v=0
    for c in reversed(poly):v=mul(v,x)^c
    return v

def divpoly(num,den):
    r=list(num);q=[0]*max(1,len(num)-len(den)+1)
    while len(r)>=len(den):
        idx=len(r)-len(den); factor=mul(r[-1],inv(den[-1]));q[idx]=factor
        for j,d in enumerate(den):r[idx+j]^=mul(factor,d)
        while r and r[-1]==0:r.pop()
    return q,r

def solve_overdetermined(matrix,rhs,unknown):
    aug=[list(row)+[v] for row,v in zip(matrix,rhs)]
    rows=len(aug); pivots=[]; r=0
    for col in range(unknown):
        pivot=next((i for i in range(r,rows) if aug[i][col]),None)
        if pivot is None:return None
        aug[r],aug[pivot]=aug[pivot],aug[r]
        scale=inv(aug[r][col]);aug[r]=[mul(x,scale) for x in aug[r]]
        for i in range(rows):
            if i!=r and aug[i][col]:
                f=aug[i][col];aug[i]=[a^mul(f,b) for a,b in zip(aug[i],aug[r])]
        pivots.append(col);r+=1
    if any(not any(row[:unknown]) and row[-1] for row in aug):return None
    answer=[0]*unknown
    for i,col in enumerate(pivots):answer[col]=aug[i][-1]
    return answer

def scales(n,k):
    # D'(i) for data positions and D(j) for parity positions; i and j represented GF(256) byte values.
    out=[]
    for a in range(n):
        v=1
        for i in range(k):
            if a!=i:v=mul(v,a^i)
        out.append(v)
    return out

def decode_joint(received,k,erasures=(),max_unknown_errors=None):
    """Return payload bytes or None. Correct v unknown byte errors plus e erasures if 2v+e<=n-k.
    No original message, true error mask or oracle inputs are used.
    """
    n=len(received); parity=n-k; bad=set(int(i) for i in erasures)
    if not 1<=k<n<=255 or len(bad)>parity or any(i<0 or i>=n for i in bad):return None
    factors=scales(n,k)
    y=[mul(received[i],factors[i]) for i in range(n)]
    indices=[i for i in range(n) if i not in bad]
    limit=(parity-len(bad))//2
    if max_unknown_errors is not None:limit=min(limit,max_unknown_errors)
    for v in range(limit+1):
        # monic error locator L degree v; Q=N L, deg(Q)<k+v
        width=k+2*v
        mat=[];rhs=[]
        for a in indices:
            powers=[1]
            for _ in range(max(k+v,v+1)):powers.append(mul(powers[-1],a))
            mat.append(powers[:k+v]+[mul(y[a],powers[t]) for t in range(v)])
            rhs.append(mul(y[a],powers[v]))
        solution=solve_overdetermined(mat,rhs,width)
        if solution is None:continue
        quotient,remainder=divpoly(solution[:k+v],solution[k+v:]+[1])
        if any(remainder):continue
        decoded=bytes(mul(evalpoly(quotient,i),inv(factors[i])) for i in range(k))
        regenerated=encode(decoded,parity)
        errors=sum(regenerated[i]!=received[i] for i in indices)
        if errors<=v and 2*errors+len(bad)<=parity:return decoded
    return None
