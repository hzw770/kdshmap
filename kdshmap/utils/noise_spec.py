import numpy as np
import scipy as sp


def Sf_renorm(Sf_list, f_list, t, method="trapz", trunc_freq=(0, 0), fk_list_input=None):

    if np.amin(t) == np.amax(t):
        f_fund = 1 / abs(t)
    else:
        f_fund = 1 / abs(np.amax(t) - np.amin(t))
    if fk_list_input is not None:
        fk_list = fk_list_input
    else:
        if np.amin(trunc_freq) == np.amax(trunc_freq):
            f_min = np.amin(f_list)
            f_max = np.amax(f_list)
        else:
            f_min = np.amin(trunc_freq)
            f_max = np.amax(trunc_freq)
        fk_list = np.arange(np.floor((f_min+f_fund/2)/f_fund), np.ceil((f_max-f_fund/2)/f_fund)+1) * f_fund
    Sf_interp = sp.interpolate.CubicSpline(f_list, Sf_list)
    Sfk_list = np.zeros(len(fk_list))

    if method == 'trapz':

        def spec_func(f):
            return np.heaviside(f-np.amin(f_list), 0) * np.heaviside(np.amax(f_list)-f, 0) * Sf_interp(f)
        for f_ in range(len(fk_list)):
            Sfk_list[f_] = sp.integrate.quad(spec_func, fk_list[f_]-f_fund/2, fk_list[f_]+f_fund/2)[0] / f_fund

    if method == 'sinc2':

        def spec_func(f):
            return (np.heaviside(f-np.amin(f_list), 0) * np.heaviside(np.amax(f_list)-f, 0) * Sf_interp(f) *
                        np.sinc(2*(f-center)/f_fund/2)**2)/f_fund
        for f_ in range(len(fk_list)):
            center = fk_list[f_]
            Sfk_list[f_] = sp.integrate.quad(spec_func, fk_list[f_]-5*f_fund/2, fk_list[f_]+5*f_fund/2)[0]

    return fk_list, Sfk_list
