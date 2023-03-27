import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np



def get_pixel_value_norm(frequencies,ref_frequencies,sky):
    values=[]
    for n,val_n in enumerate(frequencies):
        map=sky.get_emission(val_n*u.GHz)
        norm= np.zeros_like(map.value)
        for j,val_j in enumerate(ref_frequencies):
            map_ref=sky.get_emission(val_j*u.GHz)
            norm[j,:]=map[j,:]/map_ref[j,:] 
        values.append(norm)
    return values




def get_pixel_value(frequencies,npix,Stokes,sky):
    """ 
    take all the frequencies, the pixel, Stokes parameter and the sky you want and return a value for each frequencies. This value can be the Intensity,
    Q or U. In the plot they are the points.
    """
    values=[]
    for n in frequencies:
        map=sky.get_emission(n * u.GHz)[Stokes]
        pixel = map[npix].value
        values.append(pixel)
    return values




def allpixel(frequencies,Stokes,sky):
    """
    This function takes all the frequencies, without specify the pixel because it takes all the pixels.
    The scope is to return the root mean square of all the values (Intensity, Q or U) from the map and take the 
    root mean square and the standard deviation for all the values. It return a value for each frequency, in the 
    plot they are the points. 
    """
    values=[]
    err = []
    for n in frequencies:
        map=sky.get_emission(n * u.GHz)[Stokes]
        rms  = np.sqrt(np.mean(map.value**2))
        std_dev = np.std(map.value)
        values.append(rms)
        err.append(std_dev)
    return values, err


def allpixel_norm(frequencies,St,sky,value):
    """
    The same as before with also the normalization
    """
    values=[]
    err = []
    for n in frequencies:
        map=sky.get_emission(n * u.GHz)[St]
        rms  = (np.sqrt(np.mean(map.value**2)))/value
        std_dev = (np.std(map.value))/value
        values.append(rms)
        err.append(std_dev)
    return values, err


def polarization(Q,U):
    """
    This function wants to compute the polarization, that is the square root of the sum of Q and U, both on the second power.
    """
    pol_values=[]
    for i in np.arange(0,len(Q)):
        P=np.sqrt((Q[i])**2+(U[i])**2)
        pol_values.append(P)
    return pol_values
   




