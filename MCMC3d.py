
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


instr = np.load('/Users/alicepirotta/Desktop/APC/MCMC/instrument_LB_IMOv1.npy', allow_pickle=True).item()
instr_ = {}
instr_['frequency'] = np.array([instr[f]['freq'] for f in instr.keys()])
instr_['depth_p'] = np.array([instr[f]['P_sens'] for f in instr.keys()])
instr_['fwhm'] = np.array([instr[f]['beam'] for f in instr.keys()])
instr_['depth_i'] = instr_['depth_p']/np.sqrt(2)
instrument = standardize_instrument(instr_)
print(instr_['frequency'])



nside = 4
freq_mapsQ = get_observation(instrument, 'd1s1',nside=nside)[:,1,:]
freq_mapsU = get_observation(instrument, 'd1s1',nside=nside)[:,2,:]
freq_maps = get_observation(instrument, 'd1s1',nside=nside)
freq_mapsP = np.sqrt(np.power(freq_mapsQ,2)+np.power(freq_mapsU,2))

components= [CMB(),Dust(50.),Synchrotron(50.)]
A = MixingMatrix(*components)
A_ev = A.evaluator(instrument.frequency)

invN=np.eye(len(instrument.frequency))
invNd = np.einsum('ij,jsp->isp', invN, freq_maps)
invNd_P = np.einsum('ij,jp->ip', invN, freq_mapsP)


x0 = np.array([5,-10])

def lnprior(x):
    Bd, Bs = x
    if ((Bd < 0.)  or
        (Bs > 0.)):
        return -np.inf
    else:
        return 0.0

    

dati = np.einsum('ijp,klm-> ik',freq_maps,freq_maps)



def likelihood(y):
    Bd, Bs = y 
    Td=20
    A_maxL =A_ev(np.array([Bd,Td,Bs]))
    logL = 0
    AtN = A_maxL.T.dot(invN)
    NA= invN.dot(A_maxL)
    AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
    logL = logL + np.trace(AtNA.dot(AtN).dot(dati).dot(NA))
    if logL != logL:
        return 0.0
    return logL

def lnprob(x):
    lp = lnprior(x)
    return lp + likelihood(x)


ndim,nwalkers=2,30
np.random.seed(10)
pos = np.random.uniform(low=[x0[0] * (1 - 1 / 4), x0[1] * (1 - 1 / 4)], high=[x0[0] * (1 + 1 / 4), x0[1] * (1 + 1 / 4)], size=(nwalkers, ndim)) 
print(pos)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos, 10000, progress=True)
samples = sampler.get_chain(flat=True)
flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)
fig = corner.corner(flat_samples, labels=["Bd","Bs"], show_titles= True)
np.save("aver2d",samples)
plt.show()

posterior = sampler(samples, samples=1000, mu_init=-1.)
fig, ax = plt.subplots()
ax.plot(posterior)
_ = ax.set(xlabel='sample', ylabel='mu')
fig.show()

