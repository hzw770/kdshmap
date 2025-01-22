from typing import Union
from typing import Tuple


from ..utils.propagator import propagator
from ..utils.propagator import propagator_superop_fft

from ..utils.noise_spec import Sf_renorm
from ..utils.map import filter_weight
from ..utils.map import filter_weight_progressive
from pathos.multiprocessing import ProcessingPool

from ..utils.map import kdshmap
from ..utils.map import kdshmap_multi_n_ops
import numpy as np
import qutip as q


def generate_map_single(H: Union[list, q.qobj.Qobj],
                        t_list: np.ndarray,
                        noise_op: Union[list, q.qobj.Qobj],
                        f_list: Union[list, np.ndarray],
                        Sf_list: Union[list, np.ndarray],
                        trunc_freq: Union[Tuple, list] = None,
                        options=q.Options(atol=1e-10, rtol=1e-10),
                        solver_type: str = 'qutip',
                        u0_list: np.ndarray = None,
                        spd_renorm_method: str = 'trapz',
                        prop_array: np.ndarray = None,
                        filter_ops: list = None,
                        output: str = 'map',
                        prop_superop_array_fft: np.ndarray = None,
                        fk_list: Union[np.ndarray, list] = None):
    """
    Generate a single dynamical map from 0 to tau
    In the following, the code is conditioned on whether noise_op is a single noise op or a list of ops
    """

    if prop_array is None:
        prop_array = propagator(H, t_list, options, solver_type=solver_type, u0_list=u0_list)
    if prop_superop_array_fft is None:
        _, prop_superop_array_fft = propagator_superop_fft(prop_array, t_list, trunc_freq=None)

    if type(noise_op) == q.qobj.Qobj:  # If only one noise op is given -- some users may be only interested in this

        if trunc_freq is None:
            trunc_freq = (np.amin(f_list), np.amax(f_list))
        dimension = noise_op.full().shape[-1]

        if filter_ops is None:  # for single noise op, filter_ops should be [fk_list, filter_op]
            fk_list, _, filter_op = filter_weight(prop_array, t_list, noise_op, trunc_freq=trunc_freq, prop_superop_array_fft=prop_superop_array_fft)

        elif type(filter_ops) == list:
            if any(x is None for x in filter_ops):
                fk_list, _, filter_op = filter_weight(prop_array, t_list, noise_op, trunc_freq=trunc_freq, prop_superop_array_fft=prop_superop_array_fft)

            else:
                fk_list, filter_op = filter_ops
        else:
            raise Exception('no right filter_ops')

        # step 2, renormalize
        fk_list, Sfk_list = Sf_renorm(Sf_list, f_list, t_list, spd_renorm_method=spd_renorm_method, trunc_freq=trunc_freq, fk_list_input=fk_list)

        # step 3, kdshmap
        exp_lindb = kdshmap(filter_op, Sfk_list, t_list)
        system_map = np.einsum('jk,lm->jmkl', prop_array[-1], np.conjugate(np.swapaxes(prop_array[-1], 0, 1)))
        system_map = np.matmul(system_map.reshape(dimension*dimension, dimension*dimension), exp_lindb)
        if output == 'all':
            return system_map, fk_list, Sfk_list
        return system_map

    elif type(noise_op) == list:  # Instead, a list of noise_ops

        dimension = noise_op[0].full().shape[-1]  # take the dimension of the system
        fk_list_list = [np.array([None])] * len(noise_op)
        Sfk_list_list = [np.array([None])] * len(noise_op)
        filter_op_list = [np.array([None])] * len(noise_op)

        if trunc_freq.__class__ != list:  # trunc_freq in this case should also be a list
            trunc_freq = [None] * len(noise_op)  # if not, we will create a list

        if filter_ops.__class__ == list and any(x is not None for x in filter_ops):  # filter_ops here should be lists of fk_list and filter_op, i.e., [fk_list_list, filter_op_list]

            for n_ in range(len(noise_op)):
                fk_list_list[n_] = filter_ops[0][n_]  # if given, we take the n_ th filter_ops
                filter_op_list[n_] = filter_ops[1][n_]

        else:

            if any(x is None for x in [prop_superop_array_fft, fk_list]):
                fk_list, prop_superop_array_fft = propagator_superop_fft(prop_array, t_list, trunc_freq=None)

            for n_ in range(len(noise_op)):
                fk_list_list[n_] = fk_list
                filter_op_list[n_] = np.einsum('ijmkl,kl->ijm', prop_superop_array_fft, noise_op[n_].full())

                if trunc_freq[n_] is not None:
                    argwhere = np.argwhere(fk_list_list[n_] <= trunc_freq[n_][1]).transpose()[0]
                    fk_list_list[n_] = fk_list_list[n_][argwhere]
                    filter_op_list[n_] = filter_op_list[n_][argwhere]
                    argwhere = np.argwhere(fk_list_list[n_] >= trunc_freq[n_][0]).transpose()[0]
                    fk_list_list[n_] = fk_list_list[n_][argwhere]
                    filter_op_list[n_] = filter_op_list[n_][argwhere]

        fk_list_list[n_], Sfk_list_list[n_] = Sf_renorm(Sf_list[n_], f_list[n_], t_list, spd_renorm_method=spd_renorm_method,
                                                        trunc_freq=trunc_freq[n_], fk_list_input=fk_list_list[n_])
        exp_lindb = kdshmap_multi_n_ops(filter_op_list, Sfk_list_list, t_list)

        system_map = np.einsum('jk,lm->jmkl', prop_array[-1], np.conjugate(np.swapaxes(prop_array[-1], 0, 1)))
        system_map = np.matmul(system_map.reshape(dimension*dimension, dimension*dimension), exp_lindb)
        if output == 'all':
            return system_map, fk_list_list, Sfk_list_list
        return system_map


def generate_maps(H: Union[list, q.qobj.Qobj],
                  t_list_sub: np.ndarray,
                  minimal_step: float,
                  noise_op: Union[list, q.qobj.Qobj],
                  f_list: Union[list, np.ndarray],
                  Sf_list: Union[list, np.ndarray],
                  t_list_full: np.ndarray = None,
                  trunc_freq: Union[Tuple, list] = None,
                  options=q.Options(atol=1e-10, rtol=1e-10),
                  solver_type: str = 'qutip',
                  u0_list: np.ndarray = None,
                  spd_renorm_method: str = 'trapz',
                  prop_array: np.ndarray = None,
                  multicore: str ='pathos'):

    if t_list_full is None:
        N_expand = int((t_list_sub[1]-t_list_sub[0])/minimal_step)
        t_list = np.linspace(t_list_sub[0], t_list_sub[-1], N_expand*(len(t_list_sub)-1)+1)
    else:
        t_list = t_list_full
        N_expand = int((t_list_sub[1]-t_list_sub[0])/minimal_step)

    if prop_array is None:
        prop_array = propagator(H, t_list, options, solver_type=solver_type, u0_list=u0_list)

    dimension = prop_array.shape[-1]
    system_map_list = np.zeros((len(t_list_sub), dimension*dimension, dimension*dimension), dtype=complex)
    system_map_list[0] = np.eye(dimension*dimension)
    if multicore == 'pathos':

        def run(x):
            return generate_map_single(H, t_list[0:N_expand*x+1], noise_op, f_list, Sf_list, trunc_freq=trunc_freq,
                                       options=options, solver_type=solver_type, u0_list=u0_list, spd_renorm_method=spd_renorm_method,
                                       prop_array=prop_array[0:N_expand*x+1], filter_ops=None, output='map')
        results = ProcessingPool().map(run, np.arange(1, len(t_list_sub)))
        for j in range(1, len(t_list_sub)):
            system_map_list[j] = results[j-1]

    else:
        for j in range(len(t_list_sub)):

            if t_list_sub[j] == np.amin(t_list_sub):
                system_map_list[j] = np.eye(dimension*dimension)
                continue

            system_map_list[j] = generate_map_single(H, t_list[0:N_expand*j+1], noise_op, f_list, Sf_list, trunc_freq=trunc_freq,
                                                     options=options, solver_type=solver_type, u0_list=u0_list, spd_renorm_method=spd_renorm_method,
                                                     prop_array=prop_array[0:N_expand*j+1], filter_ops=None, output='map')

    return system_map_list
