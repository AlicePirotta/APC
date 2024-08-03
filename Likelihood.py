import warnings
warnings.filterwarnings("ignore")

import pysm3.units as u
import numpy as np
import matplotlib.pyplot as plt

from fgbuster import CMB, Dust, Synchrotron, MixingMatrix
from fgbuster.observation_helpers import standardize_instrument, get_observation

import matplotlib.pyplot as plt


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

def spectral_likelihood(i):
    Bd, T, Bs, x, y, z= i
    x, y = 1, 1
    G = np.diag([x,x,x,x,x,x,x,x,x,x,x,x,y,y,y,y,y,z,z,z,z,z])
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


# g3 = np.linspace(0.5,1.7,500)
# result= []
# for i in g3:
#     result.append(spectral_likelihood(np.array([1.54,20,-3,1,1,i])))
# print(np.min(result))


# plt.plot(g3,result, 'o')
# plt.title(r'Likelihood varying g3')
# plt.xlabel(r'slope index, g3')
# plt.ylabel('Likelihood value')
# plt.show()


Betad = np.linspace(1,2,10)
g3 = np.linspace(0.5,1.5,10)
result = np.zeros((10,10))
for i,val_i in enumerate(Betad):
    for j,val_j in enumerate(g3):
        result[i,j] = spectral_likelihood(np.array([val_i,20,-3,1,1,val_j]))      
   
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(Betad, g3, result.T,100)
ax.contour(Betad, g3, result.T,levels=[ np.max(result)-2.3])

ax.set_xlabel(r'$\beta_d$')
ax.set_ylabel(r'g3')
ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.show()


# Betad = np.linspace(0,6,100)
# Betas = np.linspace(-4,-2,100)
# result = np.zeros((100,100))
# for i,val_i in enumerate(Betad):
#     for j,val_j in enumerate(Betas):
#         result[i,j] = spectral_likelihood(np.array([val_i,20,val_j,0.1,0.2,0.3]))      
   
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.contourf(Betad, Betas, result.T,100)
# ax.contour(Betad, Betas, result.T,levels=[ np.max(result)-2.3])

# ax.set_xlabel(r'$\beta_d$')
# ax.set_ylabel(r'$\beta_s$')
# ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

# plt.show()