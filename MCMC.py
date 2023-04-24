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


instr = np.load('/Users/alicepirotta/Desktop/APC/FgBuster/instrument_LB_IMOv1.npy', allow_pickle=True).item()
instr_ = {}
instr_['frequency'] = np.array([instr[f]['freq'] for f in instr.keys()])
instr_['depth_p'] = np.array([instr[f]['P_sens'] for f in instr.keys()])
instr_['fwhm'] = np.array([instr[f]['beam'] for f in instr.keys()])
instr_['depth_i'] = instr_['depth_p']/np.sqrt(2)
instrument = standardize_instrument(instr_)
print(instr_['frequency'])



nside = 64
freq_mapsQ = get_observation(instrument, 'd0s0',nside=nside)[:,1,:]
freq_mapsU = get_observation(instrument, 'd0s0',nside=nside)[:,2,:]
freq_maps = get_observation(instrument, 'd0s0',nside=nside)
freq_mapsP = np.sqrt(np.power(freq_mapsQ,2)+np.power(freq_mapsU,2))

components= [CMB(),Dust(50.),Synchrotron(50.)]
A = MixingMatrix(*components)
A_ev = A.evaluator(instrument.frequency)

invN=np.eye(len(instrument.frequency))

x0 =np.array([1.54,20,-3,0.1,0.2,0.3])

def lnprior(i):
    Bd, Td, Bs, x, y, z = i
    if ((Bd < -3.) or (Bd > 6.0) or
        (Td < 10.) or (Td > 30.) or
        (Bs < -10.) or (Bs > 5.) or 
        (x < -5.) or (x > 5.) or 
        (y < -5.) or (y > 5.) or 
        (z < -5.) or (z > 5.)) :
        return -np.inf
    else:
        return 0.0
    
def likelihood(i):
    Bd, T, Bs, x, y, z = i
    G = np.diag([x,x,x,x,x,x,x,x,x,x,x,x,y,y,y,y,y,z,z,z,z,z])
    invNG= np.einsum('ij,js->is',invN,G)
    invNGd = np.einsum('ij,jsp->isp', invNG, freq_maps)
    A_maxL = A_ev(np.array([Bd,T,Bs])) 
    logL = 0
    AtNd= np.einsum('ji,jsp->isp', A_maxL, invNGd)
    AtNA = np.linalg.inv(A_maxL.T.dot(invNG).dot(A_maxL))
    logL = logL + np.einsum('isp,ij,jsp->', AtNd, AtNA, AtNd)
    if logL != logL:
        return 0.0
    return -logL

def lnprob(x):
    lp = lnprior(x)
    return lp + likelihood(x)


ndim,nwalkers=6,50
pos = np.random.uniform(low=[x0[0] * (1 - 1 / 6), x0[1] * (1 - 1 / 6), x0[2] * (1 - 1 / 6), x0[3] * (1 - 1 / 6), x0[4] * (1 - 1 / 6), x0[5] * (1 - 1 / 6)], high=[x0[0] * (1 + 1 / 6), x0[1] * (1 + 1 / 6), x0[2] * (1 + 1 / 6), x0[3] * (1 + 1 / 6),x0[4] * (1 + 1 / 6), x0[5] * (1 + 1 / 6)], size=(nwalkers, ndim))
#pos = x0 + 0.001 * np.random.randn(30, 6)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos, 10000, progress=True)
samples = sampler.get_chain(flat=True)
flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)
fig = corner.corner(flat_samples, labels=["Bd","T","Bs","g1","g2","g3"])
plt.show()


