"""Independent (main-window) ansatz-free search for the best Gaussian state of two coupled pendulums.
Parametrisation in the STANDARD coordinates (q1,q2,p1,p2): mean q free on the torus, S = L L^T with a general lower-triangular L
(log-diagonal). No frame, no scales and no starting point taken from the formula. Gap by the exact Gaussian law (paper B, Thm 1):
   kappa - R = tr(F (kappa I - K)) / tr F,  F = (S + s^2 I)^{-1},  K = [[0, Mbar],[Mbar, 0]],  Mbar = (I - E V'')/2 ."""
import numpy as np, sys, json
from scipy.optimize import minimize
def kappa(e): return (1+e+np.sqrt(1+e*e))/2
def EVpp(m,S,e):
    c1=np.cos(m[0])*np.exp(-S[0,0]/2); c2=np.cos(m[1])*np.exp(-S[1,1]/2)
    c12=np.cos(m[0]-m[1])*np.exp(-(S[0,0]+S[1,1]-2*S[0,1])/2)
    return np.array([[c1+e*c12,-e*c12],[-e*c12,c2+e*c12]])
def unpack(x):
    m=x[:2]; L=np.zeros((4,4)); L[np.tril_indices(4)]=x[2:12]; d=np.diag_indices(4); L[d]=np.exp(x[[2,4,7,11]])
    return m, L@L.T
def gap(x,s,e):
    m,S=unpack(x); k=kappa(e)
    M=(np.eye(2)-EVpp(m,S,e))/2; K=np.block([[np.zeros((2,2)),M],[M,np.zeros((2,2))]])
    w,U=np.linalg.eigh(S+s*s*np.eye(4)); F=(U/w)@U.T
    return float(np.trace(F@(k*np.eye(4)-K))/np.trace(F))/s
def obj(x,s,e):
    # box: standard deviations along the Cholesky axes between 1e-12*sigma and e^7 ~ 1100; off-diagonals |.| <= 1e3
    if np.any(x[[2,4,7,11]]>7) or np.any(x[[2,4,7,11]]<np.log(s)-27) or np.any(np.abs(x[[3,5,6,8,9,10]])>1e3): return 1e6
    v=gap(x,s,e)
    if not np.isfinite(v): return 1e6
    # Theorem (paper B): kappa - R >= kappa - lambda_max(K) >= 0; a value below that bound is a round-off failure
    m,S=unpack(x); M=(np.eye(2)-EVpp(m,S,e))/2
    if v*s < kappa(e)-np.linalg.eigvalsh(M)[-1]-1e-9: return 1e6
    return v
def search(s,e,starts,seed):
    rng=np.random.default_rng(seed); out=[]
    for i in range(starts):
        x0=np.zeros(12); x0[:2]=rng.uniform(0,2*np.pi,2)
        x0[[2,4,7,11]]=rng.uniform(np.log(s)-2,2,4); x0[[3,5,6,8,9,10]]=rng.normal(0,0.3,6)*np.exp(rng.uniform(np.log(s),1,6))
        f=lambda x: obj(x,s,e)
        r=minimize(f,x0,method='Powell',options={'xtol':1e-10,'ftol':1e-13,'maxfev':40000})
        r=minimize(f,r.x,method='Nelder-Mead',options={'xatol':1e-10,'fatol':1e-14,'maxfev':40000,'maxiter':40000})
        r=minimize(f,r.x,method='Powell',options={'xtol':1e-12,'ftol':1e-15,'maxfev':40000})
        out.append((float(r.fun),[float(v) for v in r.x]))
    return out
if __name__=='__main__':
    e=float(sys.argv[1]); s=float(sys.argv[2]); starts=int(sys.argv[3]); seed=int(sys.argv[4])
    res=search(s,e,starts,seed); vals=sorted(v for v,_ in res)
    best=min(res,key=lambda t:t[0])
    json.dump({'eps':e,'sigma':s,'starts':starts,'seed':seed,'values':vals,'best_x':best[1]},open(f'm1_e{e}_s{s}_seed{seed}.json','w'))
    print(e,s,'best',vals[0],'median',np.median(vals),'n within 1e-3 of best',sum(v<vals[0]+1e-3 for v in vals),flush=True)
