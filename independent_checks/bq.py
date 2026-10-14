# Analytic B for separable H = |p|^2/2 + V(q):
#   lambda_max of Mbar = E[M(q)], M = (I - V'')/2, q ~ N(q0, S_qq):  lambda = kappa - tr(B_q S_qq) + O(S^2),
#   B_q = -1/2 Hess_q [ e+^T M(q) e+ ]_{q0} = 1/4 Hess_q [ e+^T V''(q) e+ ]_{q0}   (e+ frozen).
# Plane (W+, q_j): W+ = (e+, -e+)/sqrt2 has q-part e+/sqrt2;  q_j = (e_j, 0).  B = P^T B_q P,  P = [e+/sqrt2, e_2..e_n].
import sys, numpy as np
import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','study'))
import l3_coupled as L
from scipy.linalg import sqrtm
def analytic(eps):
    q0=np.array([np.pi,0.0]); c=np.cos(q0[0]-q0[1])
    Vpp=np.array([[np.cos(q0[0])+eps*c,-eps*c],[-eps*c,np.cos(q0[1])+eps*c]]); M=(np.eye(2)-Vpp)/2
    ev,U=np.linalg.eigh(M); ep,em=U[:,1],U[:,0]; k=ev[1]
    e1,e2=ep; d=(e1-e2)**2
    # Hess of g(q)=e1^2 cos q1 + e2^2 cos q2 + eps (e1-e2)^2 cos(q1-q2) at q0
    H=np.array([[-e1**2*np.cos(q0[0])-eps*d*c, eps*d*c],[eps*d*c, -e2**2*np.cos(q0[1])-eps*d*c]])
    Bq=H/4; P=np.stack([ep/np.sqrt(2),em],1); B=P.T@Bq@P
    C=np.diag([2*k,k]); Bh=np.real(sqrtm(B))
    return k,B,2*np.trace(np.real(sqrtm(Bh@C@Bh))),np.linalg.eigvalsh(Bq)
for eps in [1e-4,1e-3,1e-2,0.1,0.3,1.0]:
    k,B,coef,eq=analytic(eps); g=L.gap_coefficient(eps)
    print(f"eps={eps:<7} kappa={k:.6f} coef_analytic={coef:.7f} lab_fd={g['matrix']:.7f} 1+sqrt(eps)+3eps/4={1+np.sqrt(eps)+0.75*eps:.7f}  B={np.round(B,5).tolist()} lab_B={np.round(g['B'],5).tolist()} eig(Bq)={np.round(eq,5)}")
