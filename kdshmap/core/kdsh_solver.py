from typing import Tuple
from typing import Union

import qutip as q
import numpy as np

from ..utils.propagator import propagator
from ..utils.propagator import propagator_superop_fft

from ..utils.map import filter_weight
from ..utils.operations import damped_density, damped_densities, expect, decoh_error
from .filter_func import plot_filter_Sf
from .filter_func import plot_filter_Sf_multiple


from .map_evol import generate_map_single, generate_maps


class KeldyshSolver:

    prop_array: np.ndarray = None
    prop_superop_array_fft: np.ndarray = None
    fk_list_full: np.ndarray = None
    kdshmap_list: np.ndarray = None
    kdshmap_final: np.ndarray = None
    Sfk_list: Union[list, np.ndarray] = None
    fk_list: Union[list, np.ndarray] = None
    filter_strength: Union[list, np.ndarray] = None
    density_list: np.ndarray = None
    density_final: q.qobj.Qobj = None
    expect: list = None
    error_list = None
    error_final: float = None
    filter_op: list = None
    e_ops: list = None

    def __init__(self,
                 H: Union[list, q.qobj.Qobj],
                 t_list_sub: np.ndarray,
                 minimal_step: float,
                 noise_ops: Union[list, q.qobj.Qobj],
                 f_list: Union[list, np.ndarray],
                 Sf_list: Union[list, np.ndarray],
                 density0: q.qobj.Qobj = None,
                 e_ops: list = None,
                 trunc_freq: Union[list, Tuple] = None,
                 options = dict(atol=1e-10, rtol=1e-10),
                 solver_type: str = 'qutip',
                 u0_list: np.ndarray = None,
                 spd_renorm_method: str = 'trapz',
                 goal: str = None):
        """
        Attributes defined:
        H              : Union[list, q.qobj.Qobj]
                         Hamiltonian of the system. If list, it should contain Hamiltonian at each time step.
        t_list_sub     : np.ndarray
                         Time steps for which the map will be calculated.
        minimal_step   : float
                         Minimal time step for which propagators are calculated. Should be small enough to describe the fastest oscillations in the quantum state.
        noise_ops      : Union[list, q.qobj.Qobj]
                         Noise operators for the system. If list, it should contain noise operators at each time step.
        f_list         : Union[list, np.ndarray]
                         List of frequencies for the noise operators.
        Sf_list        : Union[list, np.ndarray]
                         List of noise power spectral densities.
        density0       : q.qobj.Qobj
                         Initial density of the system.
        U_target       : q.qobj.Qobj
                         Target unitary operator.
        e_ops          : list
                         List of operators for which the expectation values will be calculated.
        trunc_freq     : Union[list, Tuple]
                         Frequency range for the noise operators.
        options        : dict
                         Options for the solver_type.
        solver_type    : str
                         solver_type for the propagator.
        u0_list        : np.ndarray
                         Closed system propagator
        spd_renorm_method         : str
                         method for the map calculation (trapz or sinc2).
        goal           : str
                         Goal of the calculation. If None or default, and expectation operators are provided,
                         the expectation values will be calculated.
                         If None or default and no expectation operators are provided, the final density will be calculated.
                         Else the maps are generated and returned.

        """
        if options is None:
            options = dict(atol=1e-10, rtol=1e-10)
        if e_ops is None:
            e_ops = []
        if type(e_ops) != list:
            e_ops = [e_ops]

        self.H = H
        self.density0 = density0
        self.t_list_sub = t_list_sub
        self.minimal_step = minimal_step
        self.noise_ops = noise_ops
        self.f_list = f_list
        self.Sf_list = Sf_list
        self.e_ops = e_ops
        self.trunc_freq = trunc_freq
        self.options = options
        self.solver_type = solver_type
        self.u0_list = u0_list
        self.spd_renorm_method = spd_renorm_method

        if np.amax(self.t_list_sub) - np.amin(self.t_list_sub) != 0:
            self.N_expand = int((self.t_list_sub[1] - self.t_list_sub[0])/abs(self.minimal_step))
            self.t_list_full = np.linspace(np.amin(self.t_list_sub), np.amax(self.t_list_sub),
                                           self.N_expand * (len(self.t_list_sub) - 1) + 1)
        else:
            self.N_expand = int(abs(np.amax(self.t_list_sub)) / self.minimal_step)
            self.t_list_full = np.linspace(0, abs(np.amax(self.t_list_sub)),
                                           self.N_expand + 1)

        self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver_type=self.solver_type, u0_list=u0_list)
        self.fk_list_full, self.prop_superop_array_fft = propagator_superop_fft(self.prop_array, self.t_list_full, trunc_freq=None)
        self.system_maps = None

        # If no specific goal is given, the goal will be speculated based on input
        if goal is None:
            return

        if goal == 'default':
            if len(e_ops) != 0 and density0 is not None:

                self.calc_all()

            elif len(e_ops) == 0 and density0 is not None:

                if len(t_list_sub) != 0:
                    self.generate_densities()
                else:
                    self.generate_density_final()

            elif len(e_ops) == 0 and density0 is None:

                if len(t_list_sub) != 0:
                    self.generate_maps()
                    self.generate_errors()
                else:
                    self.generate_map_final()
                    self.generate_error_final()

        # If goal is specified, only particular job will be performed
        else:

            if goal == 'map':
                self.generate_map_final()

            if goal == 'maps':
                self.generate_maps()

            if goal == 'filter':
                self.plot_filter_Sf()

            if goal == 'final_density':
                self.generate_density_final()

            if goal == 'densities':
                self.generate_densities(output_type='qutip')

            if goal == 'expect':
                self.generate_expect()

            if goal == 'error':
                self.generate_error_final()

            if goal == 'errors':
                self.generate_errors()

    def plot_filter_Sf(self):

        """
        Class spd_renorm_method that plots the filter strengths and the noise power spectral density.
        Returns:
        ax : matplotlib.pyplot.Axes
             Axes object containing the plot.
        """

        if type(self.noise_ops) == q.qobj.Qobj:
            if any(x is None for x in [self.fk_list, self.filter_strength]):
                if self.prop_superop_array_fft is None:
                    if self.prop_array is None:
                        self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver_type=self.solver_type, u0_list=self.u0_list)
                    self.fk_list_full, self.prop_superop_array_fft = propagator_superop_fft(self.prop_array, self.t_list_full, trunc_freq=None)
                self.fk_list, self.filter_strength, self.filter_op = filter_weight(self.prop_array, self.t_list_full, self.noise_ops,
                                                                             self.trunc_freq, prop_superop_array_fft=self.prop_superop_array_fft)

            ax = plot_filter_Sf(self.H, self.t_list_full, self.noise_ops, self.f_list, self.Sf_list, trunc_freq=self.trunc_freq,
                                options=self.options, solver_type=self.solver_type, u0_list=self.u0_list, filters=[self.fk_list, self.filter_strength])

        elif type(self.noise_ops) == list:

            if type(self.trunc_freq) != list:
                trunc_freq_list = [None] * len(self.noise_ops)
            else:
                trunc_freq_list = self.trunc_freq

            if any(x is None for x in [self.fk_list_full, self.prop_superop_array_fft]):
                self.fk_list_full, self.prop_superop_array_fft = propagator_superop_fft(self.prop_array, self.t_list_full, trunc_freq=None)

            if any(x is None for x in [self.fk_list, self.filter_strength]):

                self.fk_list = [None] * len(self.noise_ops)
                self.filter_strength = [None] * len(self.noise_ops)
                self.filter_op = [None] * len(self.noise_ops)

                for n_ in range(len(self.noise_ops)):

                    self.filter_op[n_] = np.einsum('ijmkl,kl->ijm', self.prop_superop_array_fft, self.noise_ops[n_].full())
                    filter_op_dag = np.conjugate(np.swapaxes(self.filter_op[n_], 1, 2))
                    filter_op_dag_filter_op = np.einsum('ijk,ikl->ijl', filter_op_dag, self.filter_op[n_])

                    self.filter_strength[n_] = (np.trace(filter_op_dag_filter_op, axis1=1, axis2=2) -
                                                abs(np.trace(self.filter_op[n_], axis1=1, axis2=2))**2/self.noise_ops[n_].shape[-1]).real

                    self.fk_list[n_] = self.fk_list_full

                    if trunc_freq_list[n_] is not None:
                        argwhere = np.argwhere(self.fk_list[n_] <= trunc_freq_list[n_][1]).transpose()[0]
                        self.fk_list[n_] = self.fk_list[n_][argwhere]
                        self.filter_op[n_] = self.filter_op[n_][argwhere]
                        self.filter_strength[n_] = self.filter_strength[n_][argwhere]
                        argwhere = np.argwhere(self.fk_list[n_] >= trunc_freq_list[n_][0]).transpose()[0]
                        self.fk_list[n_] = self.fk_list[n_][argwhere]
                        self.filter_op[n_] = self.filter_op[n_][argwhere]
                        self.filter_strength[n_] = self.filter_strength[n_][argwhere]

            ax = plot_filter_Sf_multiple(self.H, self.t_list_full, self.noise_ops, self.f_list, self.Sf_list,
                                         trunc_freq_list=trunc_freq_list, options=self.options, solver_type=self.solver_type, u0_list=self.u0_list,
                                         filters_list=[self.fk_list, self.filter_strength])

        else:
            raise Exception('Wrong noise op type')

        return ax

    def generate_map_final(self):

        """
        Class spd_renorm_method that generates the Keldysh map for the final time specified.
        It also calculates the propagator and the Fourier transform of the propagator superoperator for the given frequencies.
        Attributes added:
        fk_list_full            : np.ndarray
                                  List of frequencies for the fourier transform of the propagator superoperator.
        prop_superop_array_fft  : np.ndarray
                                  FFT samples of the propagator superoperator at the values specified by fk_list_full.
        kdshmap_final           : np.ndarray
                                  Keldysh map for the final time step.
        fk_list                 : np.ndarray
                                  List of frequencies for the noise operators.
        Sfk_list                : np.ndarray
                                  List of noise power spectral densities.
        Returns:
        kdshmap_final : np.ndarray
                        Keldysh map for the final time step.
        """

        if self.prop_array is None:
            self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver_type=self.solver_type, u0_list=self.u0_list)

        if any(x is None for x in [self.fk_list_full, self.prop_superop_array_fft]):
            self.fk_list_full, self.prop_superop_array_fft = propagator_superop_fft(self.prop_array, self.t_list_full, trunc_freq=None)

        self.kdshmap_final, self.fk_list, self.Sfk_list = generate_map_single(self.H, self.t_list_full, self.noise_ops,
                                                                              self.f_list, self.Sf_list, trunc_freq=self.trunc_freq,
                                                                              options=self.options, solver_type=self.solver_type,
                                                                              u0_list=self.u0_list, spd_renorm_method=self.spd_renorm_method,
                                                                              prop_array=self.prop_array, filter_ops=[self.fk_list, self.filter_op],
                                                                              output='all', prop_superop_array_fft=self.prop_superop_array_fft,
                                                                              fk_list=self.fk_list_full)

        return self.kdshmap_final

    def generate_maps(self):

        self.kdshmap_list = generate_maps(self.H, self.t_list_sub, self.minimal_step, self.noise_ops, self.f_list,
                                          self.Sf_list, t_list_full=self.t_list_full, trunc_freq=self.trunc_freq,
                                          options=self.options, solver_type=self.solver_type, u0_list=self.u0_list,
                                          spd_renorm_method=self.spd_renorm_method)
        if self.kdshmap_final is None:
            self.kdshmap_final = self.kdshmap_list[-1]
        return self.kdshmap_list

    def generate_error_final(self):

        """
        Class spd_renorm_method that calculates the error for the final time step specified.
        If the target unitary is specified, the error is calculated as the difference between the realized map and the target unitary.
        """

        if self.prop_array is None:
            self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver_type=self.solver_type, u0_list=self.u0_list)

        if self.kdshmap_final is None:
            self.generate_map_final()

        dimension = self.prop_array[-1].shape[-1]

        decoh_map = np.einsum('jk,lm->jmkl', np.conjugate(np.swapaxes(self.prop_array[-1], 0, 1)), self.prop_array[-1])
        decoh_map = np.matmul(decoh_map.reshape(dimension*dimension, dimension*dimension), self.kdshmap_final)
        self.error_final = decoh_error(decoh_map).real
        return self.error_final

    def generate_errors(self):

        if self.prop_array is None:
            self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver_type=self.solver_type, u0_list=self.u0_list)

        if self.kdshmap_list is None:
            self.generate_maps()


        dimension = self.prop_array[-1].shape[-1]
        self.error_list = np.zeros(len(self.kdshmap_list))
        for t_, kdshmap_t in enumerate(self.kdshmap_list):
            decoh_map = np.einsum('jk,lm->jmkl', np.conjugate(np.swapaxes(self.prop_array[int(self.N_expand * t_)], 0, 1)),
                                 self.prop_array[int(self.N_expand * t_)])
            decoh_map = np.matmul(decoh_map.reshape(dimension * dimension, dimension * dimension), kdshmap_t)
            self.error_list[t_] = decoh_error(decoh_map).real

        return

    def generate_density_final(self, density0=None):
        """
        Class spd_renorm_method that generates the final density matrix for the system.
        Attributes added:
        fk_list_full            : np.ndarray
                                  List of frequencies for the fourier transform of the propagator superoperator.
        prop_superop_array_fft  : np.ndarray
                                  FFT samples of the propagator superoperator at the values specified by fk_list_full.
        kdshmap_final           : np.ndarray
                                  Keldysh map for the final time step.
        fk_list                 : np.ndarray
                                  List of frequencies for the noise operators.
        Sfk_list                : np.ndarray
                                  List of noise power spectral densities.
        density_final           : q.qobj.Qobj
                                  Final density matrix for the system @t_list_sub[-1].
        """
        if density0 is None:
            density0 = self.density0

        if density0 is None:
            raise Exception('No initial density given')

        if self.kdshmap_final is None:
            self.generate_map_final()

        density_final = damped_density(density0, self.kdshmap_final)

        if density0 == self.density0:
            self.density_final = density_final

        return density_final

    def generate_densities(self, density0=None, output_type='numpy'):
        """
        Class spd_renorm_method that generates the density matrices for all the time steps in t_list_sub.
        Attributes added:
        density_list : np.ndarray
                       List of density matrices for all the time steps in t_list_sub (numpy array).

        Parameters:
        density0     : q.qobj.Qobj
                       Initial density matrix for the system.
        output_type  : str
                       Output type for the density matrices. Can be 'numpy' or 'qutip'.
        Returns:
        density_list : np.ndarray
                       List of density matrices for all the time steps in t_list_sub.
        """


        if density0 is None:
            density0 = self.density0

        if density0 is None:
            raise Exception('No initial density matrix provided')

        if self.kdshmap_list is None:
            self.generate_maps()

        density_list = damped_densities(density0, self.kdshmap_list, output_type='numpy')
        if density0 == self.density0:
            self.density_list = density_list

        if output_type == 'numpy':
            return density_list

        elif output_type == 'qutip':
            return damped_densities(density0, self.kdshmap_list, output_type='qutip')

        else:
            raise Exception('Output cannot be specified in that datatype')

    def generate_expect(self, e_ops=None, density0=None):

        """
        Class spd_renorm_method that calculates the expectation values for the given operators.

        Parameters:
        e_ops     : list
                    List of operators for which the expectation values will be calculated.
                    If None, the operators specified in the object will be used.

        density0  : q.qobj.Qobj
                    Initial density matrix for the system.
                    If None, the initial density matrix specified in the object will be used.

        Returns:
        expect_list : list
                      List of expectation values for the list of given operators.
        """

        if e_ops is None:
            if self.e_ops is None or self.e_ops == []:
                raise Exception('No expect given')

        elif e_ops != self.e_ops:
            if type(e_ops) == list:
                self.e_ops += e_ops
            elif type(e_ops) == q.qobj.Qobj:
                self.e_ops.append(e_ops)

        if density0 is None:
            density0 = self.density0
            self.density_list = self.generate_densities(density0=density0)
            density_list = self.density_list
        else:
            density_list = self.generate_densities(density0=density0)

        expect_list = expect(density0, self.e_ops, damped_density_list=density_list)
        if self.density0 == density0:
            self.expect = expect_list

        if self.system_maps is None:
            self.system_maps = self.kdshmap_list * 0
            dimension = self.prop_array[-1].shape[-1]
            for t_ in range(len(self.t_list_sub)):
                self.system_maps[t_] = np.einsum('jk,lm->jmkl', self.prop_array[int(self.N_expand*t_)], np.conjugate(np.swapaxes(self.prop_array[int(self.N_expand*t_)], 0, 1))).reshape(dimension * dimension, dimension * dimension)

        expect_noise_free_list = expect(density0, self.e_ops, kdshmap_list=self.system_maps)

        return expect_list, expect_noise_free_list

    def calc_all(self):

        self.generate_expect()
        return None


