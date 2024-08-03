import autograd.numpy as np 
import matplotlib.pyplot as plt
from fgbuster import CMB, Dust, Synchrotron, MixingMatrix, basic_comp_sep
from fgbuster.observation_helpers import standardize_instrument, get_observation
from fgbuster.cosmology import _get_Cl_cmb, _get_Cl_noise
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import healpy as hp
import numdifftools.core as nd
import math as m
from fgbuster.algebra import  W_dBdB, W_dB, W, _mmm, _utmv, _mmv, comp_sep
import pylab as pl
import numdifftools

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from fgbuster import CMB, Dust, Synchrotron, MixingMatrix
from fgbuster.observation_helpers import standardize_instrument, get_observation
import matplotlib.pyplot as plt
import emcee
from getdist import MCSamples, plots
import os
import healpy as hp

nside = 64
model = 'd0s0'

#INSTRUMENT
base_path = os.path.realpath(os.path.dirname(__file__))
instr_file = base_path+'/instrument_LB_IMOv1.npy'
instr = np.load(instr_file, allow_pickle=True).item()
instr_ = {}
instr_['frequency'] = np.array([instr[f]['freq'] for f in instr.keys()])
instr_['depth_p'] = np.array([instr[f]['P_sens'] for f in instr.keys()])
instr_['fwhm'] = np.array([instr[f]['beam'] for f in instr.keys()])
instr_['depth_i'] = instr_['depth_p']/np.sqrt(2)
instrument = standardize_instrument(instr_)
print(instr_['frequency'])

d_fgs_N = get_observation(instrument, model, noise=True, nside=nside)
d_fgs = get_observation(instrument, model, noise=False, nside=nside)

#take only the Q and U, not I
freq_maps_N= d_fgs_N[:,1:,:]
freq_maps= d_fgs[:,1:,:]

n_freqs = freq_maps.shape[0]#22
n_stokes = freq_maps.shape[1]#2 
n_pix = freq_maps.shape[2]#49152



components= [CMB(), Dust(50.), Synchrotron(50.)]

M = MixingMatrix(*components) 
M_ev = M.evaluator(instrument.frequency)

invN = np.diag(hp.nside2resol(nside, arcmin=True) / (instrument.depth_p))**2
N = np.diag((instrument.depth_p / hp.nside2resol(nside, arcmin=True))**2)


gains_true = np.ones(21)
x_spectrals= M.defaults
x_true = np.concatenate((x_spectrals, gains_true))
x_init = np.random.uniform(low=x_true * (1 - 1 / 40), high= x_true * (1 + 1 / 40))
print(x_init)



dati = np.einsum('abc,cbx-> ax',freq_maps,freq_maps.T)


def aver_likelihood(y):
    Bd, T, Bs, a, b, c, d, e, f, g, i, l, m, n, o, p, q, r, s, t, u, v, w, z = y
    h = 1
    G = np.diag([a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, r, s, t, u, v, w, z ])
    A =G.dot(M_ev(np.array([Bd,T,Bs]))) 
    # A = np.einsum('fe, ec -> fc', G, M_ev(np.array([Bd,T,Bs])))
    logL=0
    NA= np.einsum('ab,bc->ac', invN,A)
    AtNA = np.linalg.inv(np.einsum('ab,bc,cd->ad',A.T,invN,A))
    AtN= np.einsum('ab,bc->ac', A.T, invN)
    P = np.einsum ('ab,bc,cd->ad',NA,AtNA,AtN)
    dN = dati+N
    logL = logL - np.trace(np.einsum('ab,bc->ac',P,dN))/2
    if logL != logL:
        return 0.0
    return logL

bnds = ((1.4, 1.6), (19, 21), (-3.5,-2.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5))
options={'maxfun':1000000000}
result= minimize(aver_likelihood, x_init, method='TNC', tol = 1e-18, options=options, bounds=bnds )
print(result)
print(result.x)


fisher = numdifftools.Hessian(aver_likelihood)(x_init)
sigma_params_square = np.linalg.inv(fisher)
print(sigma_params_square)

def A_ev_new(y):
    Bd, T, Bs, a, b, c, d, e, f, g, i, l, m, n, o, p, q, r, s, t, u, v, w, z = y
    h = 1
    G = np.diag([a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, r, s, t, u, v, w, z ])
    # A_maxL = np.einsum('fe, ec -> fc', G, M_ev(np.array([Bd,T,Bs])))
    A_maxL =G.dot(M_ev(np.array([Bd,T,Bs])))
    return A_maxL

#evaluate the matrix A at maximum of the average likelihood
A_maxL=A_ev_new(result.x) #(n_freq,n_component) 
print(A_maxL)


AtN= np.einsum('ab,bc->ac', A_maxL.T, invN)
AtNA = np.linalg.inv(np.einsum('ab,bc,cd->ad',A_maxL.T,invN,A_maxL))


comp_maps=np.einsum ('cg,gs,sij->cij', AtNA,AtN, freq_maps)


residual_maps_QU = comp_maps[0]
zeros=np.zeros((1, 49152))
residual_maps_IQU = np.vstack((zeros, residual_maps_QU))



#multipole range 
lmin= 2
lmax= 2*nside-1
ell = np.arange(lmin,lmax+1)


#power spectrum of the total residual 
Cl_BB_residual =hp.anafast(residual_maps_IQU)[2,lmin:lmax+1] #with [2] I select the B mode #closed to bias
Dl_BB_residual = (ell*(ell+1)*Cl_BB_residual)/(2*m.pi)


#Cl theory 
Cl_BB_r1 = _get_Cl_cmb(Alens=0.0, r=1.)[2][lmin:lmax+1]
Cl_BB_r_001 = _get_Cl_cmb(Alens=0.0, r=0.001)[2] [lmin:lmax+1]

#Dl theory
Dl_BB_r001 = (ell*(ell+1)*Cl_BB_r_001)/(2*m.pi)


#lensing
Cl_lens = _get_Cl_cmb(Alens=1.0, r=0.)[2][lmin:lmax+1]
Dl_lens = (ell*(ell+1)*Cl_lens)/(2*m.pi)

#plot power spectra
plt.loglog(ell, Dl_BB_residual, label = "Dl_BB_residual")
plt.loglog(ell, Dl_lens, label = "Dl_lensing")
plt.loglog(ell, Dl_BB_r001, label = "Dl_BB_r0.001")
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell$* ($\ell$+1)*C$\ell$/2$\pi$')
plt.legend()



Data = Cl_BB_residual + Cl_lens #it is needed by the cosmological likelihood
fsky = 1.0
F = np.sum((2*ell+1) * fsky / 2*Cl_BB_r1** 2 / Data ** 2) #it is a scalar because I am using comsological likelihood
sigma_r_fisher = np.sqrt(1.0 / F)

i_cmb = M.components.index('CMB')
W_maxL = np.einsum ('cg,gs->cs', AtNA,AtN)[i_cmb,:]
print(W_maxL)


#this function takes the gain parameters 21 from the minimization and add the one that is fixed at 1
def G_values_tot(gain_params):
    a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, s, t, u, v, w, z = gain_params
    r= 1
    G = [a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, r, s, t, u, v, w, z]
    return G

gain_params_tot= G_values_tot(result.x[3:]) 

#it gives you the derivative of the gain respect to each gains
def G_diff(gain_params_tot_):
    G_dB = []
    for i,_ in enumerate(result.x[3:]):
        zeros= np.zeros((len(gain_params_tot_),len(gain_params_tot_)))
        zeros[i] = 1
        G_dB.append(zeros)
    return np.array(G_dB)


def zeros(M_dB_):
    M_dB=[]
    zero_column=np.zeros(22)
    for i,_ in enumerate(result.x[:3]):
        if i == 2: 
            M_dB_tot=np.vstack((zero_column,zero_column,M_dB_[i,:,0], zero_column, zero_column, zero_column, 
                                zero_column, zero_column, zero_column, zero_column, zero_column, zero_column, 
                                zero_column, zero_column, zero_column, zero_column, zero_column, zero_column, 
                                zero_column, zero_column, zero_column, zero_column, zero_column, zero_column))   
        else:
            M_dB_tot=np.vstack((zero_column,M_dB_[i,:,0], zero_column, zero_column, zero_column, zero_column, 
                                zero_column, zero_column, zero_column, zero_column, zero_column, zero_column, 
                                zero_column, zero_column, zero_column, zero_column, zero_column, zero_column, 
                                zero_column, zero_column, zero_column, zero_column, zero_column, zero_column))
        M_dB.append(M_dB_tot)
    return np.array(M_dB)



def A_diff_new(params_values):
    spectral_params_values = params_values[:3]
    # gain_params_values = params_values[3:]
    

    #first term
    G_dB=G_diff(gain_params_tot)  #(gain_param,freq,freq)=(21,22,22)
    zeros_matrix=np.zeros((3,22,22))
    G_dB_tot= np.concatenate((zeros_matrix,G_dB), axis=0) #(param, freq, freq)=(24,22,22)
  
    M_ev_val = M_ev(spectral_params_values) #(freq,component)=(22,3)
    # derivatives of G wrt. spectral parameters is zero
    G_dB_M = np.einsum('gif, fc-> gci',  G_dB_tot, M_ev_val) #(param, component, freq)= (24, 3, 22)
   
    

    #second term
    M_dB_ev = M.diff_evaluator(instrument.frequency)
    M_dB_val = np.array(M_dB_ev(spectral_params_values))#(spec_param,freq,component)=(3,22,1)
    M_dB_tot=zeros(M_dB_val)#(component, param, freq)=(3,24,22)
    M_dB_tot=np.array(M_dB_tot)
   
    G = np.diag(np.array(gain_params_tot)) #(freq,freq)=(22,22)
    G_M_dB_val = np.einsum('abc, cd-> bad', M_dB_tot,G) #(param, component, freq)= (24,3,22)
  
 
    A_dB_new= G_dB_M+G_M_dB_val#(params,freq,component)=(24,22,3)
    return A_dB_new


A_dB_maxL= A_diff_new(result.x)
print(A_dB_maxL)

def A_diff_diff_new(params_values):
    spectral_params_values = params_values[:3]
    # gain_params_values = params_values[3:]
    
    #first term
    G_dB=G_diff(gain_params_tot)  #(gain_param,freq,freq)=(21,22,22)
    zeros_matrix=np.zeros((3,22,22))
    G_dB_tot= np.concatenate((zeros_matrix,G_dB), axis=0) #(param, freq, freq)=(24,22,22)
    
    M_dB_ev = M.diff_evaluator(instrument.frequency)
    M_dB_val = np.array(M_dB_ev(spectral_params_values))#(spec_param,freq,component)=(3,22,1)
    M_dB_tot=zeros(M_dB_val)#(component, param, freq)=(3,24,22)
    M_dB_tot=np.array(M_dB_tot)

    G_dB_M_dB = np.einsum('abc,dec->aedb', G_dB_tot,M_dB_tot) #(24,24,3,22)
    
    

    # #second term
    M_dB_dB_ev = M.diff_diff_evaluator(instrument.frequency)
    M_dB_dB_val= np.array(M_dB_dB_ev(spectral_params_values))#(3,3)

    M_dB_dB_all = np.zeros((24,24,3,22))
    M_dB_dB_all[0,0,0]= M_dB_dB_val[0,0][:,0]
    M_dB_dB_all[0,1,1]= M_dB_dB_val[0,1][:,0]
    M_dB_dB_all[1,1,0]= M_dB_dB_val[1,0][:,0]
    M_dB_dB_all[1,1,1]= M_dB_dB_val[1,1][:,0]
    M_dB_dB_all[2,2,2]= M_dB_dB_val[2,1][:,0]
    
    
    A_dB_dB_new=  G_dB_M_dB +  G_dB_M_dB +M_dB_dB_all
    print(A_dB_dB_new.shape)

    return A_dB_dB_new

A_dBdB_maxL= A_diff_diff_new(result.x)
print(A_dBdB_maxL)


#W_dB_maxL

a = -np.linalg.inv(np.einsum('cf, fx, xs-> cs',A_maxL.T, invN, A_maxL)) #(3,3)
b = np.einsum('fcg, fx, xs-> cgs',A_dB_maxL.T, invN, A_maxL)+ np.einsum('cf, fx, gsx-> cgs',A_maxL.T, invN, A_dB_maxL)
#(3,24,3)
c = np.linalg.inv(np.einsum('cf, fx, xs-> cs',A_maxL.T, invN, A_maxL))#(3,3)
d = np.einsum('fcg, fx-> cgx',A_dB_maxL.T, invN)#(3,24,22)
e = np.einsum( 'cf, fx-> cx', A_maxL.T, invN)#(3,22)
f = np.einsum ( 'ab, bcd-> acd', a,b)
g = np.einsum ( 'ab, bc-> ac', c,e)


W_dB_maxL = np.einsum('abc, cd-> bad', f,g) + np.einsum('cx, xgf-> gcf', c, d)  #(params_tot,component,freq)=(24,3,22)
print(W_dB_maxL)
W_dB_maxL = W_dB_maxL [:, i_cmb]



#W_dB_dB_maxL

h = np.einsum ( 'bacd,be,ef->cdaf', A_dBdB_maxL.T, invN, A_maxL )
i = np.einsum ( 'bac,bd,efd->ceaf', A_dB_maxL.T, invN, A_dB_maxL )
l = np.einsum ( 'ab,bc,defc->deaf', A_maxL.T,invN, A_dBdB_maxL)

p = h+i+i+l
o = np.einsum (  'ab,bc,cd->ad', a,c,e)


H = np.einsum('ab,bcd,de,efg,gh,hi->cfai', a,b,c,b,c,e)
B =  np.einsum ( 'ab,cdea->cdeb',o,p)
C = np.einsum('ab,bcd,de,fg,ghi,if->chae', a,b,e,a,b,c)
D = np.einsum('ab,bcd,de,efg->cfag', a,b,c,d)
E = np.einsum ('ab,cbde,cf->deaf', c,A_dBdB_maxL.T,invN)


W_dB_dB_maxL = - H + B + C + D + D + E
print(W_dB_dB_maxL)
W_dB_dB_maxL =  W_dB_dB_maxL[:, :, i_cmb]

if n_stokes == 3:  
    d_spectra = freq_maps
else:  # Only P is provided, add T for map2alm
    d_spectra = np.zeros((n_freqs, 3, freq_maps.shape[2]), dtype=freq_maps.dtype)
    d_spectra[:, 1:] = freq_maps

# Compute cross-spectra
almBs = [hp.map2alm(freq_map, lmax=lmax, iter=10)[2] for freq_map in d_spectra]
Cl_fgs = np.zeros((n_freqs, n_freqs, lmax+1), dtype=freq_maps.dtype)
for f1 in range(n_freqs):
    for f2 in range(n_freqs):
        if f1 > f2:
            Cl_fgs[f1, f2] = Cl_fgs[f2, f1]
        else:
            Cl_fgs[f1, f2] = hp.alm2cl(almBs[f1], almBs[f2], lmax=lmax)

Cl_fgs = Cl_fgs[..., lmin:] / fsky



V_maxL = np.einsum('ij,ij...->...', sigma_params_square, W_dB_dB_maxL)

# Check dimentions
assert ((n_freqs,) == W_maxL.shape == W_dB_maxL.shape[1:]== W_dB_dB_maxL.shape[2:] == V_maxL.shape)
assert (len(result.x) == W_dB_maxL.shape[0] == W_dB_dB_maxL.shape[0] == W_dB_dB_maxL.shape[1])

# elementary quantities defined in Stompor, Errard, Poletti (2016)
Cl_xF = {}
Cl_xF['yy'] = _utmv(W_maxL, Cl_fgs.T, W_maxL)  # (ell,)= (191)
Cl_xF['YY'] = _mmm(W_dB_maxL, Cl_fgs.T, W_dB_maxL.T)  # (ell, param, param)=(191,24,24)
Cl_xF['yz'] = _utmv(W_maxL, Cl_fgs.T, V_maxL )  # (ell,)
Cl_xF['Yy'] = _mmv(W_dB_maxL, Cl_fgs.T, W_maxL)  # (ell, param)
Cl_xF['Yz'] = _mmv(W_dB_maxL, Cl_fgs.T, V_maxL)  # (ell, param)

print(V_maxL)

Cl_noise = _get_Cl_noise(instrument, A_maxL, lmax)[ i_cmb,  i_cmb, lmin:]

# bias and statistical foregrounds residuals
noise = Cl_noise
bias = Cl_xF['yy'] + 2 * Cl_xF['yz'] #should be 0 
stat = np.einsum('ij, lij -> l', sigma_params_square, Cl_xF['YY'])  
var = stat**2 + 2 * np.einsum('li, ij, lj -> l', Cl_xF['Yy'], sigma_params_square, Cl_xF['Yy'])
# noise_stat= noise + stat

#control that Cl_BB_residual = noise + stat


lmin= 2
lmax= 127
ell = np.arange(lmin, lmax+1)


Cl_fid = {}
Cl_fid['BB'] = _get_Cl_cmb(Alens=0.1, r=0.001)[2][lmin:lmax+1]
Cl_fid['BuBu'] = _get_Cl_cmb(Alens=0.0, r=1.0)[2][lmin:lmax+1]
Cl_fid['BlBl'] = _get_Cl_cmb(Alens=1.0, r=0.0)[2][lmin:lmax+1]



fig = pl.figure( figsize=(14,12), facecolor='w', edgecolor='k' )
ax = pl.gca()
left, bottom, width, height = [0.2, 0.2, 0.15, 0.2]
ax0 = fig.add_axes([left, bottom, width, height])
ax0.set_title(r'$\ell_{\min}=$'+str(lmin)+\
    r'$ \rightarrow \ell_{\max}=$'+str(lmax), fontsize=16)

# ax.loglog(ell, Cl_BB_r_001, color='DarkGray', linestyle='-', label='BB tot', linewidth=2.0)
ax.loglog(ell, Cl_BB_r_001 , color='DarkGray', linestyle='--', label='primordial BB for r='+str(0.001), linewidth=2.0)
ax.loglog(ell, stat, 'DarkOrange', label='statistical residuals', linewidth=2.0)
ax.loglog(ell, bias, 'DarkOrange', linestyle='--', label='systematic residuals', linewidth=2.0)
ax.loglog(ell, noise, 'DarkBlue', linestyle='--', label='noise after component separation', linewidth=2.0)
# ax.loglog(ell, Cl_BB_residual, 'DarkGreen', linestyle='--', label='residual', linewidth=1.0)
ax.legend()
ax.set_xlabel('$\ell$', fontsize=20)
ax.set_ylabel('$C_\ell$ [$\mu$K-arcmin]', fontsize=20)
ax.set_xlim(lmin,lmax)

## 5.1. data 
Cl_obs = Cl_fid['BB'] + Cl_noise
dof = (2 * ell + 1) * fsky
YY = Cl_xF['YY']
tr_SigmaYY = np.einsum('ij, lji -> l', sigma_params_square, YY)

## 5.2. modeling
def cosmo_likelihood(r_):
    # S16, Appendix C
    Alens=1.0
    Cl_model = Cl_fid['BlBl'] * Alens + Cl_fid['BuBu'] * r_ + Cl_noise
    dof_over_Cl = dof / Cl_model
    ## Eq. C3
    U = np.linalg.inv(fisher + np.dot(YY.T, dof_over_Cl))
        
    ## Eq. C9
    first_row = np.sum(dof_over_Cl * (Cl_obs * (1 - np.einsum('ij, lji -> l', U, YY) / Cl_model) + tr_SigmaYY))
    second_row = - np.einsum('l, m, ij, mjk, kf, lfi',
        dof_over_Cl, dof_over_Cl, U, YY, sigma_params_square, YY)
    trCinvC = first_row + second_row
       
    ## Eq. C10
    first_row = np.sum(dof_over_Cl * (Cl_xF['yy'] + 2 * Cl_xF['yz']))
    ### Cyclicity + traspose of scalar + grouping terms -> trace becomes
    ### Yy_ell^T U (Yy + 2 Yz)_ell'
    trace = np.einsum('li, ij, mj -> lm', Cl_xF['Yy'], U, Cl_xF['Yy'] + 2 * Cl_xF['Yz'])
    second_row = - _utmv(dof_over_Cl, trace, dof_over_Cl)
    trECinvC = first_row + second_row

    ## Eq. C12
    logdetC = np.sum(dof * np.log(Cl_model)) - np.log(np.linalg.det(U))

    # Cl_hat = Cl_obs + tr_SigmaYY

    ## Bringing things together
    return trCinvC + trECinvC + logdetC



# Likelihood maximization
r_grid = np.logspace(-5,0,num=500)
logL = np.array([cosmo_likelihood(r_loc) for r_loc in r_grid])
ind_r_min = np.argmin(logL)
r0 = r_grid[ind_r_min]
if ind_r_min == 0:
    bound_0 = 0.0
    bound_1 = r_grid[1]
    # pl.figure()
    # pl.semilogx(r_grid, logL, 'r-')
    # pl.show()
elif ind_r_min == len(r_grid)-1:
    bound_0 = r_grid[-2]
    bound_1 = 1.0
    # pl.figure()
    # pl.semilogx(r_grid, logL, 'r-')
    # pl.show()
else:
    bound_0 = r_grid[ind_r_min-1]
    bound_1 = r_grid[ind_r_min+1]
print('bounds on r = ', bound_0, ' / ', bound_1)
print('starting point = ', r0)
res_Lr = minimize(cosmo_likelihood, [r0], bounds=[(bound_0,bound_1)])
print ('    ===>> fitted r = ', res_Lr['x'])

print ('======= ESTIMATION OF SIGMA(R) =======')
def sigma_r_computation_from_logL(r_loc):
    THRESHOLD = 1.00
    # THRESHOLD = 2.30 when two fitted parameters
    delta = np.abs( cosmo_likelihood(r_loc) - res_Lr['fun'] - THRESHOLD )
        
    # print r_loc, cosmo_likelihood(r_loc),  res_Lr['fun']
    return delta

if res_Lr['x'] != 0.0:
    sr_grid = np.logspace(np.log10(res_Lr['x']), 0, num=25)
else:
    sr_grid = np.logspace(-5,0,num=25)

slogL = np.array([sigma_r_computation_from_logL(sr_loc) for sr_loc in sr_grid ])
ind_sr_min = np.argmin(slogL)
sr0 = sr_grid[ind_sr_min]
print('ind_sr_min = ', ind_sr_min)
print('sr_grid[ind_sr_min-1] = ', sr_grid[ind_sr_min-1])
print('sr_grid[ind_sr_min+1] = ', sr_grid[ind_sr_min+1])
print('sr_grid = ', sr_grid)
if ind_sr_min == 0:
    print('case # 1')
    bound_0 = res_Lr['x']
    bound_1 = sr_grid[1]
elif ind_sr_min == len(sr_grid)-1:
    print('case # 2')
    bound_0 = sr_grid[-2]
    bound_1 = 1.0
else:
    print('case # 3')
    bound_0 = sr_grid[ind_sr_min-1]
    bound_1 = sr_grid[ind_sr_min+1]
print('bounds on sigma(r) = ', bound_0, ' / ', bound_1)
print('starting point = ', sr0)
res_sr = minimize(sigma_r_computation_from_logL, sr0,
        bounds=[(bound_0.item(),bound_1.item())],
        # item required for test to pass but reason unclear. sr_grid has
        # extra dimension? 
        )
print ('    ===>> sigma(r) = ', res_sr['x'] -  res_Lr['x'])
# res.cosmo_params = {}
# res.cosmo_params['r'] = (res_Lr['x'], res_sr['x']- res_Lr['x'])


    ###############################################################################






