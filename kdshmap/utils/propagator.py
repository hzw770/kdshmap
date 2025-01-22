import qutip as q
import numpy as np
import scipy as sp


def propagator(H, t_list, options=q.Options(atol=1e-10, rtol=1e-10), solver='qutip',
               u0_list: np.ndarray = None):
    """
    Purpose: generate a list of propagators, useful for Fourier expansion
    Input: Hamiltonian (Qobj or list), t_list (numpy array)
    Return: numpy array
    """

    if solver == 'qutip':
        prop_qobj = q.propagator(H, t_list, options=options, c_ops=[])
        prop_array = np.zeros((len(t_list), prop_qobj[0].dims[0][0], prop_qobj[0].dims[0][0]), dtype=complex)
        if u0_list is None:
            for t_ in range(len(t_list)):
                prop_array[t_] = prop_qobj[t_].full()
            return prop_array
        else:
            # For construction later, used for calculation in a rotating frame
            if len(u0_list) == len(t_list):
                for t_ in range(len(t_list)):
                    prop_array[t_] = (u0_list[t_] * prop_qobj[t_] * u0_list[0].dag()).full()
            else:
                return None
            return prop_array

    if solver == 'magnus':
        # For construction later, maybe more efficient for numerics
        return None


def propagator_fft(prop_array, t_list, trunc_freq=None):

    """
    Purpose: perform FFT on the propagator x propagator^dagger
             Not the propagator itself!
             The propagator is generated above
             This is meaningful once we have more than one noise operator
    Output: the propagator x propagator^dagger in the frequency basis
    """

    fk_list = -np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])
    prop_array_dag = np.conjugate(np.swapaxes(prop_array, 1, 2))
    prop_super_array = np.einsum('ijk,ilm->ijmkl', prop_array_dag, prop_array)
    prop_array_fft = np.fft.fft(prop_super_array[0:len(t_list)-1], axis=0)/(len(t_list)-1)

    argsort = np.argsort(fk_list)
    fk_list = fk_list[argsort]
    prop_array_fft = prop_array_fft[argsort]

    if trunc_freq is not None:
        argwhere = np.argwhere(fk_list <= trunc_freq[1]).transpose()[0]
        fk_list = fk_list[argwhere]
        prop_array_fft = prop_array_fft[argwhere]
        argwhere = np.argwhere(fk_list >= trunc_freq[0]).transpose()[0]
        fk_list = fk_list[argwhere]
        prop_array_fft = prop_array_fft[argwhere]

    return fk_list, prop_array_fft


def propagator_rotate(H_rot, t_list, options=q.Options(atol=1e-10, rtol=1e-10), solver='qutip'):
    # Rotated propagator in a rotating frame, will be constructed later
    if solver == 'qutip':
        return q.propagator(H_rot, t_list, options=options, c_ops=[])
    if solver == 'magnus':
        return None

