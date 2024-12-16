import numpy as np
import scipy as sp


def Sf_renorm(Sf_list: np.ndarray, f_list: np.ndarray, 
              t, 
              spd_renorm_method="trapz", spec_res=None,
              trunc_freq=None, fk_list_input=None):
    
    """
    Coarse-grains the noise spectrum at the resolution of 2pi/tau (wp) or the user-defined resolution, 
    based on how slowly the noise spectrum varies or the resolution required.
    
    By default, we use the trapz spd_renorm_method for this renormalization, which is fast but perhaps not very accurate
    One can also choose the sinc2 spd_renorm_method as described in the paper.

    Parameters   :

    Sf_list      : np.ndarray
                   Samples of the noise spectrum to be coarse-grained

    f_list       : np.ndarray
                   Frequencies at which the noise spectrum has been sampled
    
    t            : float
                   Time array, to specify 2pi/tau

    spd_renorm_method       : str
                   spd_renorm_method to use for the renormalization. Options are 'trapz' and 'sinc2' (default is 'trapz')

    spec_res     : float
                   Minimum resolution at which to coarse-grain the noise spectrum (default is None) 

    trunc_freq   : np.ndarray
                   Frequency range or values to consider for the coarse-graining (default is None)  

    """

    if np.amin(t) == np.amax(t):
        f_fund = 1 / abs(t)
    else:
        f_fund = 1 / abs(np.amax(t) - np.amin(t))
    if fk_list_input is not None:
        fk_list = fk_list_input
    else:
        if trunc_freq is None:
            f_min = np.amin(f_list)
            f_max = np.amax(f_list)
        else:
            f_min = np.amin(trunc_freq)
            f_max = np.amax(trunc_freq)
        fk_list = np.arange(np.floor((f_min+f_fund/2)/f_fund), np.ceil((f_max-f_fund/2)/f_fund)+1) * f_fund
    
    Sf_interp = sp.interpolate.CubicSpline(f_list, Sf_list)
    Sfk_list = np.zeros(len(fk_list))

    if spd_renorm_method == 'trapz':

        def spec_func(f):
            return np.heaviside(f-np.amin(f_list), 0) * np.heaviside(np.amax(f_list)-f, 0) * Sf_interp(f)
        for f_ in range(len(fk_list)):
            Sfk_list[f_] = sp.integrate.quad(spec_func, fk_list[f_]-f_fund/2, fk_list[f_]+f_fund/2)[0] / f_fund

    if spd_renorm_method == 'sinc2':

        def spec_func(f):
            return (np.heaviside(f-np.amin(f_list), 0) * np.heaviside(np.amax(f_list)-f, 0) * Sf_interp(f) *
                        np.sinc(2*(f-center)/f_fund/2)**2)/f_fund
        for f_ in range(len(fk_list)):
            center = fk_list[f_]
            Sfk_list[f_] = sp.integrate.quad(spec_func, fk_list[f_]-5*f_fund/2, fk_list[f_]+5*f_fund/2)[0]

    return fk_list, Sfk_list
