import warnings
warnings.filterwarnings("ignore")


import numpy as np
import matplotlib.pyplot as plt
from fgbuster import CMB, Dust, Synchrotron, MixingMatrix
from fgbuster.observation_helpers import standardize_instrument, get_observation
import matplotlib.pyplot as plt
import emcee
import random
from getdist import MCSamples, plots



#INSTRUMENT
instr = np.load('/Users/alicepirotta/Desktop/APC/MCMC/instrument_LB_IMOv1.npy', allow_pickle=True).item()
instr_ = {}
instr_['frequency'] = np.array([instr[f]['freq'] for f in instr.keys()])
instr_['depth_p'] = np.array([instr[f]['P_sens'] for f in instr.keys()])
instr_['fwhm'] = np.array([instr[f]['beam'] for f in instr.keys()])
instr_['depth_i'] = instr_['depth_p']/np.sqrt(2)
instrument = standardize_instrument(instr_)
print(instr_['frequency'])


#SKY MAP
nside = 4
freq_maps = get_observation(instrument, 'd0s0', noise=False, nside=nside)
components= [CMB(),Dust(50.),Synchrotron(50.)]
A = MixingMatrix(*components)
A_ev = A.evaluator(instrument.frequency)


#NOISE
invN=np.linalg.inv(np.eye(len(instrument.frequency)))
#invN = np.diag((hp.nside2resol(nside, arcmin=True) / instrument.depth_p)**2)


#INITIAL PARAMETER
x0 =np.array([1.54,20,-3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])



def lnprior(y):
    Bd, Td, Bs, a, b, c, d, e, f, g, i, l, m , n, o, p, q, r, s, t, u, v, w, z = y
    if ((Bd < 0.) or (Bd > 2.) or
        (Td < 10.) or (Td > 30.) or
        (Bs < -4.) or (Bs > -2.) or  
        (a < 0.) or
        (b < 0.) or
        (c < 0.) or
        (d < 0.) or
        (e < 0.) or
        (f < 0.) or
        (g < 0.) or
        (i < 0.) or
        (l < 0.) or
        (m < 0.) or
        (n < 0.) or
        (o < 0.) or
        (p < 0.) or
        (q < 0.) or
        (r < 0.) or
        (s < 0.) or
        (t < 0.) or
        (u < 0.) or
        (v < 0.) or
        (w < 0.) or
        (z < 0.)):
        return -np.inf
    else:
        return 0.0
    

def spec_likelihood(y):
    Bd, T, Bs, a, b, c, d, e, f, g, i, l, m, n, o, p, q, r, s, t, u, v, w, z = y
    h= 1
    G = np.diag([a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, r, s, t, u, v, w, z])
    A =G.dot(A_ev(np.array([Bd,T,Bs]))) 
    logL = 0
    AinvNd= np.einsum('ba,ac,cde->bde', A.T, invN, freq_maps)
    AtinvNA = np.linalg.inv(np.einsum('ab,bc,ce->ae',A.T,invN,A))
    logL = logL - np.einsum('abc,cd,dba->', AinvNd.T, AtinvNA, AinvNd)
    if logL != logL:
        return 0.0
    return logL

# def spec_likelihood(y):
#     Bd, T, Bs, a, b, c, d, e, f, g, i, l, m, n, o, p, q, r, s, t, u, v, w, z = y
#     h= 1
#     G = np.diag([a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, r, s, t, u, v, w, z])
#     invNd = np.einsum('ij,jsp->isp', invN, freq_maps)
#     A_maxL =G.dot(A_ev(np.array([Bd,T,Bs]))) 
#     logL = 0
#     AtNd= np.einsum('ji,jsp->isp', A_maxL, invNd)
#     AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
#     logL = logL + np.einsum('isp,ij,jsp->', AtNd, AtNA, AtNd)
#     if logL != logL:
#         return 0.0
#     return logL

def lnprob(x):
    lp = lnprior(x)
    return lp + spec_likelihood(x)


random.seed(10)
nwalkers, ndim = 50, 24
pos = np.random.uniform(low=x0 * (1 - 1 / 4), high= x0 * (1 + 1 / 4), size=(nwalkers, ndim))


#SAMPLE
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos,1000, progress=True)



#AUTOCORRELATION TIME
tau = sampler.get_autocorr_time(quiet=True)
samples = sampler.get_chain(discard=3*int(max(tau)), thin=int(max(tau)), flat=True)
np.save("1000000_thin.npy",samples)


s1 = MCSamples(samples=samples, names =["Bd", "T", "Bs", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g22" ])
g = plots.get_subplot_plotter()
g.triangle_plot([s1], filled=True, title_limit= True)
plt.show()







