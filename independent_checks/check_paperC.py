"""Verification of the analytical statements of paper C (main window).  Independent of the stage-L code:
B is computed from a closed form and, separately, from numerical derivatives of the potential; the family gap of Theorem 2 is
evaluated with the exact block form of F.  Exit code 0 iff all checks pass."""
import numpy as np, json, sys
from scipy.linalg import sqrtm
from scipy.optimize import minimize
import os
STUDY=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','study')
RES=os.path.join(STUDY,'results','')
n_ok=n=0
def check(name, ok, info=''):
    global n_ok, n; n+=1; n_ok+=bool(ok); print(('PASS ' if ok else 'FAIL ')+name+('   '+info if info else ''), flush=True)

# ---------------- generic machinery for H = |p|^2/2 + V(q) ----------------
def hess(fun, q, h=1e-4):
    k=len(q); H=np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            e_i=np.eye(k)[i]*h; e_j=np.eye(k)[j]*h
            H[i,j]=(fun(q+e_i+e_j)-fun(q+e_i-e_j)-fun(q-e_i+e_j)+fun(q-e_i-e_j))/(4*h*h)
    return H
def Gamma_generic(Vpp, q0):
    """kappa, B (in the basis w, e_2..e_n), C, Gamma from V'' alone (numerical Hessian of e+^T V'' e+)."""
    M=(np.eye(len(q0))-Vpp(q0))/2; lam,U=np.linalg.eigh(M); k=lam[-1]; ep=U[:,-1]; others=U[:,:-1]
    g=lambda q: ep@Vpp(q)@ep
    Bq=hess(g,q0)/4; P=np.column_stack([ep/np.sqrt(2)]+[others[:,j] for j in range(others.shape[1])])
    B=P.T@Bq@P; C=np.diag([2*k]+[k]*(len(q0)-1))
    Bs=np.real(sqrtm((B+B.T)/2+1e-300*np.eye(len(B)))); G=2*np.trace(np.real(sqrtm(Bs@C@Bs)))
    return k,B,C,G,Bq,ep,others
def Vpp_coupled(e):
    def f(q):
        c1,c2,c12=np.cos(q[0]),np.cos(q[1]),np.cos(q[0]-q[1])
        return np.array([[c1+e*c12,-e*c12],[-e*c12,c2+e*c12]])
    return f
def closed_form(e):
    """Eq. (coupledBq): kappa, e+, B_q, B, Gamma in closed form."""
    k=(1+e+np.sqrt(1+e*e))/2; v=np.array([e,1-np.sqrt(1+e*e)]); ep=v/np.linalg.norm(v); e1,e2=ep; d=(e1-e2)**2
    Bq=np.array([[e1**2+e*d,-e*d],[-e*d,-e2**2+e*d]])/4; em=np.array([-e2,e1])
    P=np.column_stack([ep/np.sqrt(2),em]); B=P.T@Bq@P; C=np.diag([2*k,k])
    Bs=np.real(sqrtm(B)); return k,ep,Bq,B,2*np.trace(np.real(sqrtm(Bs@C@Bs)))

# ---------------- 1. kappa and e+ ----------------
for e in [0.1,0.3,1.0]:
    k,ep,Bq,B,G=closed_form(e); M=(np.eye(2)-Vpp_coupled(e)(np.array([np.pi,0])))/2
    lam,U=np.linalg.eigh(M); check(f'kappa(eps={e}) closed form = top eigenvalue of M(q0)', abs(lam[-1]-k)<1e-14, f'{k:.12f}')
    check(f'e+(eps={e}) closed form is the top eigenvector', abs(abs(ep@U[:,-1])-1)<1e-13)
    qs=np.linspace(0,2*np.pi,401); mx=max(np.linalg.eigvalsh((np.eye(2)-Vpp_coupled(e)(np.array([a,b])))/2)[-1] for a in qs for b in qs)
    check(f'kappa(eps={e}) is the maximum over a 401^2 grid of the torus', mx<=k+1e-12, f'grid max {mx:.12f}')

# ---------------- 2. B: closed form vs numerical derivatives vs stage-L finite differences ----------------
sys.path.insert(0,STUDY); import l3_coupled as L3
for e in [1e-3,0.1,0.3,1.0]:
    k,ep,Bq,B,G=closed_form(e); kg,Bg,Cg,Gg,_,_,_=Gamma_generic(Vpp_coupled(e),np.array([np.pi,0.0]))
    check(f'B closed form = numerical Hessian of e+^T V\'\' e+ (eps={e})', np.abs(np.abs(B)-np.abs(Bg)).max()<1e-6, f'max diff {np.abs(np.abs(B)-np.abs(Bg)).max():.1e}')
    check(f'Gamma closed form = generic computation (eps={e})', abs(G-Gg)<1e-6, f'{G:.7f}')
    lab=L3.gap_coefficient(e)['matrix']
    check(f'Gamma = stage-L finite-difference value (eps={e})', abs(G-lab)<2e-6, f'{G:.7f} vs {lab:.7f}')
    check(f'B_q is positive semidefinite at the maximum (eps={e})', np.linalg.eigvalsh(Bq).min()>-1e-14)

# ---------------- 3. Lemma 3 (matrix identity) ----------------
rng=np.random.default_rng(0)
for t in range(5):
    A=rng.normal(size=(3,3)); B=A@A.T; Cd=np.diag(rng.uniform(0.5,2,3))
    Ch=np.sqrt(Cd); X=Ch@B@Ch; target=2*np.trace(np.real(sqrtm(X)))
    Ts=Ch@np.linalg.inv(np.real(sqrtm(X)))@Ch; val=np.trace(B@Ts)+np.trace(Cd@np.linalg.inv(Ts))
    f=lambda y: (lambda Lm: np.trace(B@Lm@Lm.T)+np.trace(Cd@np.linalg.inv(Lm@Lm.T)))(np.tril(y.reshape(3,3))+np.diag(np.exp(np.diag(y.reshape(3,3)))-np.diag(y.reshape(3,3))))
    best=min(minimize(f,rng.normal(size=9)*0.3,method='BFGS').fun for _ in range(3))
    check(f'matrix identity, random 3x3 case {t}: T* attains 2 tr sqrt(C^1/2 B C^1/2), numerical inf agrees', abs(val-target)<1e-10 and abs(best-target)<1e-6, f'{target:.8f} / {best:.8f}')

# ---------------- 4. Proposition 1 (momentum directions are free) ----------------
e=0.3; m=np.array([2.9,0.2])
def Kbar(S): c=Vpp_coupled(e); return L3.EVpp(np.concatenate([m,[0,0]]),S,e)
A=rng.normal(size=(4,4)); S1=A@A.T*0.1; S2=S1.copy(); S2[2:,2:]+=np.diag([5.,7.]); S2[0,2]+=0.3; S2[2,0]+=0.3
w=np.linalg.eigvalsh(S2); 
check('averaged strain depends on S only through S_qq (momentum block and q-p correlations changed)', np.allclose(Kbar(S1),Kbar(S2),atol=1e-15) and w.min()>0)

# ---------------- 5. Theorem 2: family gap -> sigma [tr(BT) + tr(C T^-1)] + O(sigma^2) ----------------
def family_gap(e, s, T):
    """(kappa-R)/sigma for the family (eq:family), F in exact block form; Mbar by the closed-form Gaussian averages."""
    k,ep,Bq,B,G=closed_form(e); em=np.array([-ep[1],ep[0]])
    v=np.concatenate([ep,ep])/np.sqrt(2); wv=np.concatenate([ep,-ep])/np.sqrt(2); a=np.concatenate([em,[0,0]]); b=np.concatenate([[0,0],em])
    Lm=np.column_stack([wv,a]); S=s*Lm@T@Lm.T+np.outer(b,b)
    mean=np.array([np.pi,0,0,0]); M=(np.eye(2)-L3.EVpp(mean,S,e))/2; K=np.block([[np.zeros((2,2)),M],[M,np.zeros((2,2))]])
    F=np.outer(v,v)/s**2 + Lm@np.linalg.inv(T+s*np.eye(2))@Lm.T/s + np.outer(b,b)/(1+s*s)
    return np.trace(F@(k*np.eye(4)-K))/np.trace(F)/s
for e in [0.1,0.3]:
    k,ep,Bq,B,G=closed_form(e); C=np.diag([2*k,k])
    for name,T in [('T*',np.real(np.linalg.inv(sqrtm(B))@sqrtm(sqrtm(B)@C@sqrtm(B))@np.linalg.inv(sqrtm(B)))),('random T',np.array([[3.,0.5],[0.5,9.]]))]:
        lead=np.trace(B@T)+np.trace(C@np.linalg.inv(T))
        errs=[abs(family_gap(e,s,T)-lead)/s for s in [1e-3,1e-4,1e-5]]
        check(f'Theorem 2, eps={e}, {name}: |gap/sigma - leading| / sigma stays bounded', max(errs)<50 and errs[-1]<=errs[0]*1.2+1e-6, ' '.join(f'{x:.3g}' for x in errs))
    Ts=np.real(np.linalg.inv(sqrtm(B))@sqrtm(sqrtm(B)@C@sqrtm(B))@np.linalg.inv(sqrtm(B)))
    check(f'Theorem 2 at T*: family gap -> Gamma (eps={e}, sigma=1e-6)', abs(family_gap(e,1e-6,Ts)-G)<1e-4, f'{family_gap(e,1e-6,Ts):.7f} vs {G:.7f}')

# ---------------- 6. Corollary 4 (uncoupled) and one degree of freedom ----------------
Vpp_unc=lambda q: np.diag([np.cos(q[0]), 0.5*np.cos(q[1])])     # V = -cos q1 - 0.5 cos q2 ; max strain 1 at q1 = pi
k,B,C,G,_,_,_=Gamma_generic(Vpp_unc,np.array([np.pi,0.0]))
check('uncoupled second degree of freedom costs nothing: Gamma = 1 (pendulum)', abs(G-1)<1e-6 and abs(k-1)<1e-12, f'{G:.8f}')
k,B,C,G,_,_,_=Gamma_generic(lambda q: np.array([[np.cos(q[0])]]),np.array([np.pi]))
check('one degree of freedom: Gamma = 2 sqrt(2 beta kappa) = 1 for the pendulum', abs(G-1)<1e-6)

# ---------------- 7. weak-coupling series ----------------
for e in [1e-3,1e-4,1e-5]:
    G=closed_form(e)[4]; ser=1+np.sqrt(e)+0.75*e+0.125*e**1.5
    check(f'series 1 + sqrt(e) + 3e/4 + e^1.5/8: error/e^2 < 1.1 at e={e}', abs(G-ser)/e**2<1.1, f'{(G-ser)/e**2:.4f}')

# ---------------- 8. stage-L numerical optimum vs Gamma, and the predicted shape ----------------
d=json.load(open(RES+'l3_single.json'))+json.load(open(RES+'l3_eps1.json'))
for r in d:
    if r['eps']>0 and r['sigma']<=1e-5:
        G=closed_form(r['eps'])[4]
        dev=(r['gap_over_sigma']-G)/r['sigma']      # the next order is O(sigma): slopes about -0.9, -1.7, -2.4, -4 (stage L)
        check(f"numerical optimum = Gamma + O(sigma), |dev|/sigma < 10 (eps={r['eps']}, sigma={r['sigma']:g})", abs(dev)<10, f"{r['gap_over_sigma']:.6f} vs {G:.6f}, (opt-Gamma)/sigma = {dev:.2f}")
for e,obs in [(0.1,(1.98,2.57,0.076)),(0.3,(1.93,2.10,0.170))]:
    k,ep,Bq,B,G=closed_form(e); C=np.diag([2*k,k]); Bh=np.real(sqrtm(B)); T=np.linalg.inv(Bh)@np.real(sqrtm(Bh@C@Bh))@np.linalg.inv(Bh)
    sd=np.sqrt(np.diag(T)); cr=T[0,1]/sd[0]/sd[1]
    check(f'predicted optimal shape matches stage-L optimum (eps={e})', abs(sd[0]-obs[0])<0.01 and abs(sd[1]-obs[1])<0.01 and abs(cr-obs[2])<0.002, f'{sd[0]:.3f} {sd[1]:.3f} {cr:.3f}')

# ---------------- 9. singular covariance: R is continuous as width is added (Sec. II, Gaussian states) ----------------
def R_pend(S, s2):   # pendulum, exact Gaussian average: E[cos q] = -exp(-S_qq/2) at q0 = pi; S_qq = (S_vv + 2 S_vw + S_ww)/2 in frame (v, w)
    P=np.array([1,1])/2**.5; sqq=P@S@P; Mb=(1+np.exp(-sqq/2))/2
    K=np.array([[Mb,0],[0,-Mb]]); F=np.linalg.inv(S+s2*np.eye(2)); return np.trace(F@K)/np.trace(F)
for sig in [0.1,0.01]:
    S=sig*4*np.diag([0.0,1.0])   # zero width along v, T* = 4 along w
    r0=R_pend(S,sig**2); d=[abs(R_pend(S+eta*np.eye(2),sig**2)-r0) for eta in [1e-4,1e-6,1e-8]]
    check(f'singular state is the limit of non-degenerate ones, |R(S+eta I)-R(S)| = O(eta) (sigma={sig})', d[1]<d[0]*0.02 and d[2]<d[1]*0.02 and d[2]<1e-5, ' '.join(f'{x:.1e}' for x in d))

# ---------------- 10. Eq. (kappa): the largest strain is on the positive side, sup lmax(M) >= sup(-lmin(M)) ----------------
def side_ratio(Vpp, n, npts=181, nrand=20000):
    if n==2:
        t=np.linspace(0,2*np.pi,npts); Q=[np.array([a,b]) for a in t for b in t]
    else:
        Q=list(np.random.default_rng(1).uniform(0,2*np.pi,(nrand,n)))
    hi=lo=-np.inf
    for q in Q:
        w=np.linalg.eigvalsh((np.eye(n)-Vpp(q))/2); hi=max(hi,w[-1]); lo=max(lo,-w[0])
    return hi,lo
def Vpp_chain(e,n):
    def f(q):
        H=np.diag(np.cos(q)).astype(float)
        for i in range(n-1):
            c=e*np.cos(q[i]-q[i+1]); H[i,i]+=c; H[i+1,i+1]+=c; H[i,i+1]-=c; H[i+1,i]-=c
        return H
    return f
def Vpp_prod(e): return lambda q: np.array([[np.cos(q[0])+e*np.cos(q[0])*np.cos(q[1]),-e*np.sin(q[0])*np.sin(q[1])],[-e*np.sin(q[0])*np.sin(q[1]),2*np.cos(q[1])+e*np.cos(q[0])*np.cos(q[1])]])
systems=[(f'coupled eps={e}',Vpp_coupled(e),2) for e in [-0.3,-0.1,0.1,0.3,1,2,3]]+[(f'chain n={m} eps={e}',Vpp_chain(e,m),m) for m in [3,4] for e in [0.1,0.3]]+[(f'product eps={e}',Vpp_prod(e),2) for e in [0.3,1]]
for name,f,nd in systems:
    hi,lo=side_ratio(f,nd)
    check(f'largest strain on the positive side, Eq. (kappa): {name}', lo<0.9*hi, f'sup lmax = {hi:.4f}, sup(-lmin) = {lo:.4f}')

print(f'\n{n_ok}/{n} checks passed'); sys.exit(0 if n_ok==n else 1)
