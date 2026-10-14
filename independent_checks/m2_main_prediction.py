"""Main-window prediction of Gamma for open chains of n pendulums (independent of the lab; recorded with time and hash, not sent to the lab).
V = -sum cos q_i - eps sum cos(q_i - q_{i+1}).  kappa = max_q lambda_max((I - V'')/2), found by multistart optimisation."""
import numpy as np, json, hashlib, datetime, itertools
from scipy.optimize import minimize
from scipy.linalg import sqrtm
exec(open('check_paperC.py').read().split('def Vpp_coupled')[0].split('n_ok=n=0')[1].replace('def check','def _unused'))
def Vpp_chain(n,e):
    def f(q):
        H=np.diag(np.cos(q)).astype(float)
        for i in range(n-1):
            c=e*np.cos(q[i]-q[i+1]); H[i,i]+=c; H[i+1,i+1]+=c; H[i,i+1]-=c; H[i+1,i]-=c
        return H
    return f
def top(n,e):
    f=Vpp_chain(n,e); lm=lambda q: -np.linalg.eigvalsh((np.eye(n)-f(q))/2)[-1]
    best=None
    for corner in itertools.product([0,np.pi],repeat=n):
        r=minimize(lm,np.array(corner)+1e-3,method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-15,'maxiter':20000})
        if best is None or r.fun<best.fun-1e-12: best=r
    rng=np.random.default_rng(0)
    for _ in range(40):
        r=minimize(lm,rng.uniform(0,2*np.pi,n),method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-15,'maxiter':20000})
        if r.fun<best.fun-1e-9: best=r
    return np.mod(best.x,2*np.pi), -best.fun
out=[]
for n in [3,4]:
    for e in [0.1,0.3]:
        q0,k=top(n,e); q0=np.round(q0/np.pi)*np.pi if np.allclose(q0/np.pi,np.round(q0/np.pi),atol=1e-5) else q0
        kg,B,C,G,Bq,ep,oth=Gamma_generic(Vpp_chain(n,e),q0)
        out.append({'n':n,'eps':e,'q0_over_pi':(q0/np.pi).round(6).tolist(),'kappa':kg,'kappa_search':k,'Gamma':G,'eig_Bq':np.linalg.eigvalsh(Bq).tolist()})
        print(n,e,'q0/pi',(q0/np.pi).round(4),'kappa',round(kg,8),'Gamma',round(G,6),'eig Bq',np.round(np.linalg.eigvalsh(Bq),5))
rec={'time_utc':datetime.datetime.utcnow().isoformat(),'predictions':out}
s=json.dumps(rec,indent=1); open('m2_main_prediction.json','w').write(s); print('sha256',hashlib.sha256(s.encode()).hexdigest())
