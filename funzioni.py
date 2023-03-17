import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np




def get_pixel_value(frequencies,npix,Stokes,sky):
    values=[]
    for n in frequencies:
        map=sky.get_emission(n * u.GHz)[Stokes]
        pixel = map[npix].value
        values.append(pixel)
    return values

def get_pixel_value_norm(frequencies,ref_frequencies,npix,Stokes,sky):
    values=[]
    for n in frequencies:
        map=sky.get_emission(n * u.GHz)[Stokes]
        pixel=map[npix].value
        map_ref=sky.get_emission(ref_frequencies* u.GHz)[Stokes]
        val= map_ref[npix].value
        norm=pixel/val
        values.append(norm)
    return values
    

def allpixel(frequencies,Stokes,sky):
    values=[]
    err = []
    for n in frequencies:
        map=sky.get_emission(n * u.GHz)[Stokes]
        rms  = np.sqrt(np.mean(map.value**2))
        std_dev = np.std(map.value)
        values.append(rms)
        err.append(std_dev)
    return values, err


def allpixel_norm(frequencies,Stokes,sky,value):
    values=[]
    err = []
    for n in frequencies:
        map=sky.get_emission(n * u.GHz)[Stokes]
        rms  = (np.sqrt(np.mean(map.value**2)))/value
        std_dev = (np.std(map.value))/value
        values.append(rms)
        err.append(std_dev)
    return values, err


def polarizzazione(Q,U):
    pol_values=[]
    for i in np.arange(0,len(Q)):
        P=np.sqrt((Q[i])**2+(U[i])**2)
        pol_values.append(P)
    return pol_values
    


