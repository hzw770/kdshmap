from typing import Union
from typing import Tuple

from ..utils.propagator import propagator
from ..utils.propagator import propagator_superop_fft
from ..utils.map import filter_weight
import qutip as q
import numpy as np
import matplotlib.pyplot as plt




def generate_filter(H: Union[list, q.qobj.Qobj],
                    t_list: np.ndarray,
                    noise_op: q.qobj.Qobj,
                    trunc_freq: Tuple = None,
                    options = dict(atol=1e-10, rtol=1e-10),
                    solver_type: str = 'qutip',
                    u0_list: np.ndarray = None,
                    prop_array: np.ndarray = None,
                    prop_superop_array_fft: np.ndarray = None):
    """
    Generate the filter strength for a given noise operator.
    Parameters:
    H                       : Union[list, q.qobj.Qobj]
                              Hamiltonian of the system, given as a list of qutip.Qobj or a single qutip.Qobj.
    t_list                  : np.ndarray
                              To calculate the filter operators at t_list[-1], the propagators are evaluated at
                              values within [0, t_list[-1]].
    noise_op                : q.qobj.Qobj
                              Noise operator for which the filter strength is calculated.
    trunc_freq              : Tuple
                              Tuple of the lower and upper frequency bounds for the filter strength.

    options                 : dict
                              Options for the solver_type.

    solver_type             : str
                              Specifies the solver_type to be used.

    u0_list                 : np.ndarray
                              Initial state of the system.

    prop_array              : np.ndarray
                              Array of the propagator.
    prop_superop_array_fft  : np.ndarray
                              Array of the Fourier transformed propagator superoperators.

    """
    if prop_superop_array_fft is None:
        if prop_array is None:
            prop_array = propagator(H, t_list, options, solver_type=solver_type, u0_list=u0_list)
        fk_list, prop_superop_array_fft = propagator_superop_fft(prop_array, t_list, trunc_freq=None)

    fk_list, filter_strength, _ = filter_weight(prop_array, t_list, noise_op, trunc_freq, prop_superop_array_fft=prop_superop_array_fft)
    return fk_list, filter_strength


def plot_filter(H, t_list, noise_op, trunc_freq=None, options=dict(atol=1e-10, rtol=1e-10), solver_type='qutip', u0_list=None,
                filters=None, ax=None, prop_array: np.ndarray = None, prop_superop_array_fft: np.ndarray = None):


    """
    Plot the filter strength for a given noise operator.
    Parameters:
    H                       : Union[list, q.qobj.Qobj]
                              Hamiltonian of the system, given as a list of qutip.Qobj or a single qutip.Qobj.
    t_list                  : np.ndarray
                              To calculate the filter operators at t_list[-1], the propagators are evaluated at
                              values within [0, t_list[-1]].

    noise_op                : q.qobj.Qobj
                              Noise operator for which the filter strength is calculated.
    trunc_freq              : Tuple
                              Tuple of the lower and upper frequency bounds for the filter strength.
    options                 : dict
                              Options for the solver_type.
    solver_type             : str
                              Specifies the solver_type to be used.
    u0_list                 : np.ndarray
                              Initial state of the system.
    filters                 : list
                              List of filter weights and frequencies, if already calculated.
    ax                      : plt.Axes
                              Axes object for plotting.
    prop_array              : np.ndarray
                              Array of the propagator.
    prop_superop_array_fft  : np.ndarray
                              Array of the Fourier transformed propagator superoperators.

    """

    if filters is None:
        filters = [None, None]

    if any(x is None for x in filters):
        fk_list, filter_strength = generate_filter(H, t_list, noise_op, trunc_freq=trunc_freq, options=options,
                                                   solver_type='qutip', u0_list=u0_list, prop_array=prop_array,
                                                   prop_superop_array_fft=prop_superop_array_fft)
    else:
        fk_list, filter_strength = filters

    if ax is None:
        ax = plt.subplot()
    ax.step(fk_list, filter_strength, where='mid')
    ax.fill_between(fk_list, filter_strength, step="mid", alpha=0.4)
    ax.set_xlabel(r'frequency (unit of time$^{-1}$)')
    ax.set_ylabel('filter strength')
    ax.set_xlim(trunc_freq[0], trunc_freq[1])
    ax.set_ylim(0.0,)

    return ax


def plot_filter_Sf(H: Union[list, q.qobj.Qobj],
                   t_list: np.ndarray,
                   noise_op: q.qobj.Qobj,
                   f_list: np.ndarray,
                   Sf_list: np.ndarray,
                   trunc_freq: Tuple = None,
                   options=dict(atol=1e-10, rtol=1e-10),
                   solver_type: str = 'qutip',
                   u0_list: np.ndarray = None,
                   filters: list = None,
                   ax=None, prop_array: np.ndarray = None,
                   prop_superop_array_fft: np.ndarray = None):

    """
    Plot the given bath spectral density, evaluated at the filter frequencies.
    Parameters:
    H                       : Union[list, q.qobj.Qobj]
                              Hamiltonian of the system, given as a list of qutip.Qobj or a single qutip.Qobj.
    t_list                  : np.ndarray
                              To calculate the filter operators at t_list[-1], the propagators are evaluated at
                              values within [0, t_list[-1]].
    noise_op                : q.qobj.Qobj
                              Noise operator for which the filter strength is calculated.
    f_list                  : np.ndarray
                              Frequencies at which the bath spectral density is evaluated.
    Sf_list                 : np.ndarray
                              Bath spectral density evaluated at the frequencies in f_list.
    trunc_freq              : Tuple
                              Bounds for the plot
    options                 : dict
                              Options for the solver_type.
    solver_type             : str
                              Specifies the solver_type to be used.
    u0_list                 : np.ndarray
                              Initial state of the system.
    filters                 : list
                              List of filter weights and frequencies, if already calculated.
    ax                      : plt.Axes
                              Axes object for plotting.
    prop_array              : np.ndarray
                              Array of the propagator.
    prop_superop_array_fft  : np.ndarray
                              Array of the Fourier transformed propagator superoperators.

    """
    if filters is None:
        filters = [None, None]

        if prop_superop_array_fft is None:
            if prop_array is None:
                prop_array = propagator(H, t_list, options, solver_type=solver_type, u0_list=u0_list)
        fk_list, prop_superop_array_fft = propagator_superop_fft(prop_array, t_list, trunc_freq=None)

    if trunc_freq is None:
        trunc_freq = (np.amin(f_list), np.amax(f_list))

    if ax is None:
        ax = plt.subplot()

    plot_filter(H, t_list, noise_op, trunc_freq=trunc_freq, options=options, solver_type=solver_type, u0_list=u0_list,
                filters=filters, ax=ax, prop_array=prop_array, prop_superop_array_fft=prop_superop_array_fft)
    ax2 = ax.twinx()
    ax2.plot(f_list, Sf_list, lw=2, alpha=1, color='k')
    ax2.set_xlim(trunc_freq[0], trunc_freq[1])
    ax2.set_ylim(0.0,)
    ax2.set_ylabel('noise amplitude')

    return ax


def plot_filter_Sf_multiple(H: Union[list, q.qobj.Qobj],
                            t_list: np.ndarray,
                            noise_ops: list,
                            f_list_list: list,
                            Sf_list_list: list,
                            trunc_freq_list: list = None,
                            options=dict(atol=1e-10, rtol=1e-10),
                            solver_type: str = 'qutip',
                            u0_list: np.ndarray = None,
                            filters_list: list = None,
                            ax=None,
                            prop_array: np.ndarray = None,
                            prop_superop_array_fft: np.ndarray = None):
    """
    Plot the filter strength for a list of noise operators.

    Parameters:
    H                       : Union[list, q.qobj.Qobj]
                              Hamiltonian of the system, given as a list of qutip.Qobj or a single qutip.Qobj.
    t_list                  : np.ndarray
                              To calculate the filter operators at t_list[-1], the propagators are evaluated at
                              values within [0, t_list[-1]].
    noise_op                : q.qobj.Qobj
                              Noise operator for which the filter strength is calculated.
    trunc_freq              : Tuple
                              Tuple of the lower and upper frequency bounds for the filter strength.
    options                 : dict
                              Options for the solver_type.
    solver_type             : str
                              Specifies the solver_type to be used.
    u0_list                 : np.ndarray
                              Initial state of the system.
    filters                 : list
                              List of filter weights and frequencies, if already calculated.
    ax                      : plt.Axes
                              Axes object for plotting.
    prop_array              : np.ndarray
                              Array of the propagator.
    prop_superop_array_fft  : np.ndarray
                              Array of the Fourier transformed propagator superoperators.

    """
    if filters_list is None:
        filters_list = [[None] * len(noise_ops), [None] * len(noise_ops)]
        if prop_superop_array_fft is None:
            if prop_array is None:
                prop_array = propagator(H, t_list, options, solver_type=solver_type, u0_list=u0_list)
            fk_list, prop_superop_array_fft = propagator_superop_fft(prop_array, t_list, trunc_freq=None)

    if len(noise_ops) == 1:

        return plot_filter_Sf(H, t_list, noise_ops[0], f_list_list[0], Sf_list_list[0], trunc_freq=trunc_freq_list[0],
                              options=options, solver_type=solver_type, u0_list=u0_list, filters=[filters_list[0][0], filters_list[1][0]], ax=None,
                              prop_array=prop_array, prop_superop_array_fft=prop_superop_array_fft)

    if ax is None:
        fig, ax = plt.subplots(len(noise_ops), 1)

    for n_ in range(len(noise_ops)):

        if trunc_freq_list is None:
            trunc_freq = None
        else:
            trunc_freq = trunc_freq_list[n_]

        plot_filter_Sf(H, t_list, noise_ops[n_], f_list_list[n_], Sf_list_list[n_], trunc_freq=trunc_freq,
                       options=options, solver_type=solver_type, u0_list=u0_list, filters=[filters_list[0][n_], filters_list[1][n_]],
                       ax=ax[n_], prop_array=prop_array, prop_superop_array_fft=prop_superop_array_fft)
    plt.tight_layout()
    return ax
