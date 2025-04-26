from typing import Union
from typing import Tuple

from ..utils.stabilized import propagator_stabilized_superop_fft
from ..utils.stabilized import filter_weight_stabilized_for_state
import qutip as q
import numpy as np
import matplotlib.pyplot as plt


def generate_filter_stabilized_for_state(H: Union[list, q.qobj.Qobj],
                    density0: q.qobj.Qobj,
                    t_list: np.ndarray,
                    noise_op: q.qobj.Qobj,
                    c_ops: list,
                    trunc_freq: Tuple = None,
                    options = dict(atol=1e-10, rtol=1e-10),
                    solver_type: str = 'qutip',
                    u0_list: np.ndarray = None,
                    prop_fft: np.array = None,
                    prop_final = None):
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
    if any(element is None for element in [prop_fft, prop_final]):
        fk_list, prop_fft, prop_final = propagator_stabilized_superop_fft(H, c_ops, t_list, options, solver_type)

    fk_list, filter_weights_state = filter_weight_stabilized_for_state(prop_fft, prop_final, noise_op, density0, t_list, trunc_freq)
    return fk_list, filter_weights_state


def plot_filter_stabilized_for_state(H, density0, t_list, noise_op, c_ops,trunc_freq=None, options=dict(atol=1e-10, rtol=1e-10),
                                     solver_type='qutip', u0_list=None,
                                     ax=None, prop_fft = None,prop_final = None):


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


    fk_list, filter_weights_state = generate_filter_stabilized_for_state(H, density0, t_list, noise_op, c_ops, trunc_freq, options, solver_type, u0_list, prop_fft, prop_final)

    if ax is None:
        ax = plt.subplot()
    ax.step(fk_list, filter_weights_state, where='mid')
    ax.fill_between(fk_list, filter_weights_state, step="mid", alpha=0.4)
    ax.set_xlabel(r'frequency (unit of time$^{-1}$)')
    ax.set_ylabel('filter strength')
    ax.set_xlim(trunc_freq[0], trunc_freq[1])
    ax.set_ylim(0.0,)

    return ax


def plot_filter_Sf_stabilized_for_state(H: Union[list, q.qobj.Qobj],
                   density0: q.qobj.Qobj,
                   t_list: np.ndarray,
                   noise_op: q.qobj.Qobj,
                   c_ops: list,
                   f_list: np.ndarray,
                   Sf_list: np.ndarray,
                   trunc_freq: Tuple = None,
                   options=dict(atol=1e-10, rtol=1e-10),
                   solver_type: str = 'qutip',
                   u0_list: np.ndarray = None,
                   ax=None, prop_fft = None,
                   prop_final = None):

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

    if trunc_freq is None:
        trunc_freq = (np.amin(f_list), np.amax(f_list))

    if ax is None:
        ax = plt.subplot()

    plot_filter_stabilized_for_state(H, density0, t_list, noise_op, c_ops, trunc_freq, options, solver_type, u0_list,
                 ax, prop_fft, prop_final)

    ax2 = ax.twinx()
    ax2.plot(f_list, Sf_list, lw=2, alpha=1, color='k')
    ax2.set_xlim(trunc_freq[0], trunc_freq[1])
    ax2.set_ylim(0.0,)
    ax2.set_ylabel('noise amplitude')

    return ax


def plot_filter_Sf_stabilized_for_state_multiple(H: Union[list, q.qobj.Qobj],
                            density0: q.qobj.Qobj,
                            t_list: np.ndarray,
                            noise_ops: list,
                            c_ops: list,
                            f_list_list: list,
                            Sf_list_list: list,
                            trunc_freq_list: list = None,
                            options=dict(atol=1e-10, rtol=1e-10),
                            solver_type: str = 'qutip',
                            u0_list: np.ndarray = None,
                            ax=None):


    if len(noise_ops) == 1:
        return plot_filter_Sf_stabilized_for_state(H, density0, t_list, noise_ops[0], c_ops, f_list_list[0], Sf_list_list[0], trunc_freq=trunc_freq_list[0],
                              options=options, solver_type=solver_type, u0_list=u0_list, ax=None, prop_fft=None, prop_final=None)

    if ax is None:
        fig, ax = plt.subplots(len(noise_ops), 1)

    for n_ in range(len(noise_ops)):

        if trunc_freq_list is None:
            trunc_freq = None
        else:
            trunc_freq = trunc_freq_list[n_]

        plot_filter_Sf_stabilized_for_state(H, density0, t_list, noise_ops[n_], c_ops, f_list_list[n_], Sf_list_list[n_], trunc_freq=trunc_freq,
                       options=options, solver_type=solver_type, u0_list=u0_list, ax=ax[n_])
    plt.tight_layout()
    return ax