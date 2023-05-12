import warnings
warnings.filterwarnings("ignore")

import pysm3
import pysm3.units as u
from astropy import units as u
import pysm3.units as u
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from fgbuster import CMB, Dust, Synchrotron, MixingMatrix
from fgbuster.observation_helpers import standardize_instrument, get_observation,  get_instrument
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import emcee
import corner
import healpy as hp


instr = np.load('/Users/alicepirotta/Desktop/APC/MCMC/instrument_LB_IMOv1.npy', allow_pickle=True).item()
instr_ = {}
instr_['frequency'] = np.array([instr[f]['freq'] for f in instr.keys()])
instr_['depth_p'] = np.array([instr[f]['P_sens'] for f in instr.keys()])
instr_['fwhm'] = np.array([instr[f]['beam'] for f in instr.keys()])
instr_['depth_i'] = instr_['depth_p']/np.sqrt(2)
instrument = standardize_instrument(instr_)
print(instr_['frequency'])



nside = 64

freq_maps = get_observation(instrument, 'd0s0', noise=False, nside=nside)



components= [CMB(),Dust(50.),Synchrotron(50.)]
A = MixingMatrix(*components)
A_ev = A.evaluator(instrument.frequency)

#invN=np.linalg.inv(np.eye(len(instrument.frequency)))
invN = np.diag((hp.nside2resol(nside, arcmin=True) / instrument.depth_p)**2)


x0 =np.array([1.54,20,-3])

def lnprior(i):
    Bd, Td, Bs = i
    if ((Bd < 0.) or (Bd > 2.) or
        (Td < 10.) or (Td > 30.) or
        (Bs < -4.) or (Bs > -2.)):
        return -np.inf
    else:
        return 0.0
    
def likelihood(i):
    Bd, T, Bs=i
    invNd = np.einsum('ij,jsp->isp', invN, freq_maps)
    A_maxL =A_ev(np.array([Bd,T,Bs]))
    logL = 0
    AtNd= np.einsum('ji,jsp->isp', A_maxL, invNd)
    AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
    logL = logL + np.einsum('isp,ij,jsp->', AtNd, AtNA, AtNd)
    if logL != logL:
        return 0.0
    return logL

def lnprob(x):
    lp = lnprior(x)
    return lp + likelihood(x)


ndim,nwalkers=3,30
pos = np.random.uniform(low=[x0[0] * (1 - 1 / 4), x0[1] * (1 - 1 / 4), x0[2] * (1 - 1 / 4)], high=[x0[0] * (1 + 1 / 4), x0[1] * (1 + 1 / 4), x0[2] * (1 + 1 / 4)], size=(nwalkers, ndim))


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos,20000, progress=True)
samples = sampler.get_chain(flat=True)
fig = corner.corner(samples, labels=["Bd", "T", "Bs"])
print(samples)
np.save("3d",samples)

plt.show()




