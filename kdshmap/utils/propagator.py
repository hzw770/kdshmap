import qutip as q
import numpy as np
import scipy as sp


def propagator(H, t_list, options=dict(atol=1e-10, rtol=1e-10), solver_type='qutip',
               u0_list: np.ndarray = None):

    """
    Given a Hamiltonian and a list of time values, return the propagator at each time.
    Can specify the solver_type to be used, either 'qutip' or 'magnus', for a Magnus
    expansion type, perturbative calculation of the propagator.
    Parameters:
    H         : Qobj or list
                Hamiltonian operator or list of Hamiltonian operators

    t_list    : numpy array
                List of time values at which to calculate the propagator

    options   : dict
                Options for the solver_type, default is atol=1e-10, rtol=1e-10
    solver_type    : str (optional)
                solver_type to be used, either 'qutip' or 'magnus'

    u0_list   : numpy array (optional)
                A unitary frame transformations, if the Hamiltonian is provided in a different frame
    Returns   :
    prop_array: numpy array
                Array of propagators at each time value

    """

    if solver_type == 'qutip':
        prop_qobj = q.propagator(H, t_list, options=options, c_ops=[])
        prop_array = np.zeros((len(t_list), prop_qobj[0].shape[0], prop_qobj[0].shape[0]), dtype=complex)
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

    if solver_type == 'magnus':
        # For construction later, maybe more efficient for numerics
        return None


def propagator_superop_fft(prop_array, t_list, trunc_freq=None):


    """
    Given samples of the propagator in the time basis, compute the closed system superoperator (U^\dag \otimes U^T),
    transform it to the frequency basis through an FFT, and return it.
    Parameters            :
    prop_array            : numpy array
                            Array of propagators (numpy arrays) at each time value

    t_list                : numpy array
                            List of time values at which the propagators are calculated/sampled
    trunc_freq            : tuple (optional)
                            Tuple of two floats, the lower and upper bounds of the frequency range
                            in which the DFT is to be calculated
    Returns                 :
    fk_list                 : numpy array
                              List of frequencies at which the DFT is calculated (ordered negative to positive)

    prop_superop_array_fft  : numpy array
                              Array of 'super-propagators' in the frequency basis : Convention followed is the opposite of that in np.fft.fft
    """

    fk_list = -np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])
    prop_array_dag = np.conjugate(np.swapaxes(prop_array, 1, 2))
    prop_superop_array = np.einsum('ijk,ilm->ijmkl', prop_array_dag, prop_array)
    prop_superop_array_fft = np.fft.fft(prop_superop_array[0:len(t_list)-1], axis=0)/(len(t_list)-1)

    argsort = np.argsort(fk_list)
    fk_list = fk_list[argsort]
    prop_superop_array_fft = prop_superop_array_fft[argsort]

    if trunc_freq is not None:
        argwhere = np.argwhere(fk_list <= trunc_freq[1]).transpose()[0]
        fk_list = fk_list[argwhere]
        prop_superop_array_fft = prop_superop_array_fft[argwhere]
        argwhere = np.argwhere(fk_list >= trunc_freq[0]).transpose()[0]
        if len(argwhere) == 0:
            raise Exception('no filter_ops, change trunc_freq')
        fk_list = fk_list[argwhere]
        prop_superop_array_fft = prop_superop_array_fft[argwhere]

    return fk_list, prop_superop_array_fft


def propagator_rotate(H_rot, t_list, options=dict(atol=1e-10, rtol=1e-10), solver_type='qutip'):
    # Rotated propagator in a rotating frame, will be constructed later
    if solver_type == 'qutip':
        return q.propagator(H_rot, t_list, options=options, c_ops=[])
    if solver_type == 'magnus':
        return None

