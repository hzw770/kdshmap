from typing import Tuple
from typing import Union

import qutip as q
import numpy as np

from ..utils.propagator import propagator
from ..utils.propagator import propagator_fft

from ..utils.map import filter_weight
from ..utils.operations import damped_density, damped_densities, expect, decoh_error
from .filter_func import plot_filter_Sf
from .filter_func import plot_filter_Sf_multiple


from .map_evol import generate_map_single, generate_maps
from .error import generate_error_single, generate_errors


class KeldyshSolver:

    prop_array: np.ndarray = None
    prop_array_fft: np.ndarray = None
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
    fft: list = None
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
                 options=q.Options(atol=1e-10, rtol=1e-10),
                 solver: str = 'qutip',
                 u0_list: np.ndarray = None,
                 method: str = 'trapz',
                 goal: str = None):

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
        self.solver = solver
        self.u0_list = u0_list
        self.method = method

        if np.amax(self.t_list_sub) - np.amin(self.t_list_sub) != 0:
            N_expand = int((self.t_list_sub[1] - self.t_list_sub[0])/abs(self.minimal_step))
            self.t_list_full = np.linspace(np.amin(self.t_list_sub), np.amax(self.t_list_sub),
                                           N_expand * (len(self.t_list_sub) - 1) + 1)
        else:
            N_expand = int(abs(np.amax(self.t_list_sub)) / self.minimal_step)
            self.t_list_full = np.linspace(0, abs(np.amax(self.t_list_sub)),
                                           N_expand + 1)

        self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver=self.solver, u0_list=u0_list)
        self.fk_list_full, self.prop_array_fft = propagator_fft(self.prop_array, self.t_list_full, trunc_freq=None)

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

        if type(self.noise_ops) == q.qobj.Qobj:
            if any(x is None for x in [self.fk_list, self.filter_strength]):
                if self.prop_array_fft is None:
                    if self.prop_array is None:
                        self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver=self.solver, u0_list=self.u0_list)
                    self.fk_list_full, self.prop_array_fft = propagator_fft(self.prop_array, self.t_list_full, trunc_freq=None)
                self.fk_list, self.filter_strength, self.fft = filter_weight(self.prop_array, self.t_list_full, self.noise_ops,
                                                                             self.trunc_freq, prop_array_fft=self.prop_array_fft)

            ax = plot_filter_Sf(self.H, self.t_list_full, self.noise_ops, self.f_list, self.Sf_list, trunc_freq=self.trunc_freq,
                                options=self.options, solver=self.solver, u0_list=self.u0_list, filters=[self.fk_list, self.filter_strength])

        elif type(self.noise_ops) == list:

            if type(self.trunc_freq) != list:
                trunc_freq_list = [None] * len(self.noise_ops)
            else:
                trunc_freq_list = self.trunc_freq

            if any(x is None for x in [self.fk_list_full, self.prop_array_fft]):
                self.fk_list_full, self.prop_array_fft = propagator_fft(self.prop_array, self.t_list_full, trunc_freq=None)

            if any(x is None for x in [self.fk_list, self.filter_strength]):

                self.fk_list = [None] * len(self.noise_ops)
                self.filter_strength = [None] * len(self.noise_ops)
                self.fft = [None] * len(self.noise_ops)

                for n_ in range(len(self.noise_ops)):

                    self.fft[n_] = np.einsum('ijmkl,kl->ijm', self.prop_array_fft, self.noise_ops[n_].full())
                    fft_dag = np.conjugate(np.swapaxes(self.fft[n_], 1, 2))
                    fft_dag_fft = np.einsum('ijk,ikl->ijl', fft_dag, self.fft[n_])

                    self.filter_strength[n_] = (np.trace(fft_dag_fft, axis1=1, axis2=2) -
                                                abs(np.trace(self.fft[n_], axis1=1, axis2=2))**2/self.noise_ops[n_].shape[-1]).real

                    self.fk_list[n_] = self.fk_list_full

                    if trunc_freq_list[n_] is not None:
                        argwhere = np.argwhere(self.fk_list[n_] <= trunc_freq_list[n_][1]).transpose()[0]
                        self.fk_list[n_] = self.fk_list[n_][argwhere]
                        self.fft[n_] = self.fft[n_][argwhere]
                        self.filter_strength[n_] = self.filter_strength[n_][argwhere]
                        argwhere = np.argwhere(self.fk_list[n_] >= trunc_freq_list[n_][0]).transpose()[0]
                        self.fk_list[n_] = self.fk_list[n_][argwhere]
                        self.fft[n_] = self.fft[n_][argwhere]
                        self.filter_strength[n_] = self.filter_strength[n_][argwhere]

            ax = plot_filter_Sf_multiple(self.H, self.t_list_full, self.noise_ops, self.f_list, self.Sf_list,
                                         trunc_freq_list=trunc_freq_list, options=self.options, solver=self.solver, u0_list=self.u0_list,
                                         filters_list=[self.fk_list, self.filter_strength])

        else:
            raise Exception('Wrong noise op type')

        return ax

    def generate_map_final(self):

        if self.prop_array is None:
            self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver=self.solver, u0_list=self.u0_list)

        if any(x is None for x in [self.fk_list_full, self.prop_array_fft]):
            self.fk_list_full, self.prop_array_fft = propagator_fft(self.prop_array, self.t_list_full, trunc_freq=None)

        self.kdshmap_final, self.fk_list, self.Sfk_list = generate_map_single(self.H, self.t_list_full, self.noise_ops,
                                                                              self.f_list, self.Sf_list, trunc_freq=self.trunc_freq,
                                                                              options=self.options, solver=self.solver,
                                                                              u0_list=self.u0_list, method=self.method,
                                                                              prop_array=self.prop_array, ffts=[self.fk_list, self.fft],
                                                                              output='all', prop_array_fft=self.prop_array_fft,
                                                                              fk_list=self.fk_list_full)

        return self.kdshmap_final

    def generate_maps(self):

        self.kdshmap_list = generate_maps(self.H, self.t_list_sub, self.minimal_step, self.noise_ops, self.f_list,
                                          self.Sf_list, t_list_full=self.t_list_full, trunc_freq=self.trunc_freq,
                                          options=self.options, solver=self.solver, u0_list=self.u0_list,
                                          method=self.method)

        return self.kdshmap_list

    def generate_error_final(self):

        if self.prop_array is None:
            self.prop_array = propagator(self.H, self.t_list_full, options=self.options, solver=self.solver, u0_list=self.u0_list)

        if self.kdshmap_final is None:
            self.generate_map_final()

        dimension = self.prop_array[-1].shape[-1]

        decoh_map = np.einsum('jk,lm->jmkl', np.conjugate(np.swapaxes(self.prop_array[-1], 0, 1)), self.prop_array[-1])
        decoh_map = np.matmul(decoh_map.reshape(dimension*dimension, dimension*dimension), self.kdshmap_final)
        self.error_final = decoh_error(decoh_map).real
        return self.error_final

    def generate_errors(self):

        return None

    def generate_density_final(self, density0=None):

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

        if density0 is None:
            density0 = self.density0

        if density0 is None:
            raise Exception('No initial density given')

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
            raise Exception('No such data type')

    def generate_expect(self, e_ops=None, density0=None):

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

        if density0 != self.density0 or self.density_list is None:
            density_list = self.generate_densities(density0=density0)
        else:
            density_list = self.density_list

        expect_list = expect(density0, self.e_ops, damped_density_list=density_list)
        if self.density0 == density0:
            self.expect = expect_list

        return expect_list

    def calc_all(self):

        self.generate_expect()
        return None


