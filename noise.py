

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import matplotlib.pyplot as plt
from fgbuster import CMB, Dust, Synchrotron, MixingMatrix
from fgbuster.observation_helpers import standardize_instrument, get_observation
from fgbuster.cosmology import _get_Cl_cmb
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import healpy as hp
import numdifftools as nd
import math as m

instr = np.load('/Users/alicepirotta/Desktop/APC/MCMC/instrument_LB_IMOv1.npy', allow_pickle=True).item()
instr_ = {}
instr_['frequency'] = np.array([instr[f]['freq'] for f in instr.keys()])
instr_['depth_p'] = np.array([instr[f]['P_sens'] for f in instr.keys()])
instr_['fwhm'] = np.array([instr[f]['beam'] for f in instr.keys()])
instr_['depth_i'] = instr_['depth_p']/np.sqrt(2)
instrument = standardize_instrument(instr_)

nside = 64
freq_maps = get_observation(instrument, 'd0s0', noise=True, nside=nside)
print(freq_maps.shape)

components= [CMB(), Dust(50.), Synchrotron(50.)]

A = MixingMatrix(*components)
A_ev = A.evaluator(instrument.frequency)
#invN=np.linalg.inv(np.eye(len(instrument.frequency)))
invN = np.diag((hp.nside2resol(nside, arcmin=True) / instrument.depth_p)**2)

x0 = np.array([1.3,19,-2.5])
x1 = np.array([1.3,19,-2.5, 0.8,1.3])
x2 =np.array([1.54, 20, -3, 1.1, 0.4, 1.5, 0.8, 1.2, 0.6, 0.3, 1.5, 0.5, 1.5, 1.2, 1.1, 0.8, 1.4, 0.7, 1.4, 0.4, 0.6, 0.8, 1.2, 1.5])  






Cl_all_g21 = []
seeds = 10
for i in range (seeds):
    np.random.seed(i)
    #noise =np.random.random
    freq_maps = get_observation(instrument, 'd0s0', noise=True, nside=nside)[:,1:,:]
    

    def spectral_likelihood(y):
        Bd, T, Bs, a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, r, t, u, v, w, z = y
        s = 1
        G = np.diag([a,b,c,d,e,f,g,h,i,l,m,n,o,p,q,r,s,t,u,v,w,z])
        invNd = np.einsum('ij,jsp->isp', invN, freq_maps)
        A_maxL =G.dot(A_ev(np.array([Bd,T,Bs]))) 
        logL = 0
        AtNd= np.einsum('ji,jsp->isp', A_maxL, invNd)
        AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
        logL = logL + np.einsum('isp,ij,jsp->', AtNd, AtNA, AtNd)
        #print(i,logL)
        if logL != logL:
            return 0.0
        return -logL

        
    min_= minimize (spectral_likelihood,x2,method='trust-constr', tol= 1e-18)
        
    
    invNd = np.einsum('ij,jsp->isp', invN, freq_maps)
    A_maxL =A_ev(min_.x) 
    AtNd= np.einsum('ji,jsp->isp', A_maxL, invNd)
    AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
    s = np.einsum('cg,gsp->csp', AtNA,AtNd)
    print(AtNA.shape)
    print(AtNA.shape)
    s = s[0]
    print(s.shape)
    zeros=np.zeros((1, 49152))
    unione = np.vstack((zeros, s))
        
        
    Cl =hp.anafast(unione)[2,2:]
    Cl_all_g21.append(Cl)



Cl_all_g21=np.array(Cl_all_g21)
Cl_mean_g21=np.mean(Cl_all_g21, axis=0) 
Cl_std_g21 = np.std(Cl_all_g21, axis=0)



ell = np.arange(2,192)
Dl_mean_g21 = (ell*(ell+1)*Cl_mean_g21)/(2*m.pi)
Dl_std_g21 = (ell*(ell+1)*Cl_std_g21)/(2*m.pi)




Cl_BB_r1 = _get_Cl_cmb(Alens=0.0, r=1.)[2][2:192]
Cl_BB_r001 = _get_Cl_cmb(Alens=0.0, r=0.001)[2][2:192]
Dl_BB = (ell*(ell+1)*Cl_BB_r001)/(2*m.pi)

Cl_lens = _get_Cl_cmb(Alens=1.0, r=0.)[2][2:192]
Dl_lens = (ell*(ell+1)*Cl_lens)/(2*m.pi)

Data_g21 = Cl_mean_g21 + Cl_lens
fsky = 1
F_g21 = np.sum((2*ell+1) * fsky / 2*Cl_BB_r1** 2 / Data_g21 ** 2)
sigma_r_fisher_g21 = np.sqrt(1.0 / F_g21)
print(F_g21)
print(sigma_r_fisher_g21)





plt.loglog(ell, Dl_mean_g21, label = "Dl_BB_g21")
plt.loglog(ell, Dl_lens, label = "Dl_lensing_g21")
plt.loglog(ell, Dl_BB, label = "Dl_BB_r0.001_g21")
plt.fill_between(ell,Dl_mean_g21+Dl_std_g21, Dl_mean_g21-Dl_std_g21, alpha=0.2, label="Dl_BB_std_g21")



plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell$* ($\ell$+1)*C$\ell$/2$\pi$')
plt.legend()
plt.show()




