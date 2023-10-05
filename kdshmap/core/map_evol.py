from typing import Union
from typing import Tuple


from ..utils.propagator import propagator
from ..utils.propagator import propagator_fft

from ..utils.noise_spec import Sf_renorm
from ..utils.map import filter_weight
from ..utils.map import filter_weight_progressive
from pathos.multiprocessing import ProcessingPool

from ..utils.map import kdshmap
from ..utils.map import kdshmap_nops
import numpy as np
import qutip as q


def generate_map_single(H: Union[list, q.qobj.Qobj],
                        t_list: np.ndarray,
                        noise_op: Union[list, q.qobj.Qobj],
                        f_list: Union[list, np.ndarray],
                        Sf_list: Union[list, np.ndarray],
                        trunc_freq: Union[Tuple, list] = (0, 0),
                        options=q.Options(atol=1e-10, rtol=1e-10),
                        solver: str = 'qutip',
                        u0_list: np.ndarray = None,
                        method: str = 'trapz',
                        prop_array: np.ndarray = None,
                        ffts: list = None,
                        output: str = 'map',
                        prop_array_fft: np.ndarray = None,
                        fk_list: Union[np.ndarray, list] = None):
    """
    """

    if prop_array is None:
            prop_array = propagator(H, t_list, options, solver=solver, u0_list=u0_list)

    if type(noise_op) == q.qobj.Qobj:  # If only one noise op is given -- some users may be only interested in this

        if np.amax(trunc_freq) == np.amin(trunc_freq) or type(trunc_freq) != tuple:
            trunc_freq = (np.amin(f_list), np.amax(f_list))
        dimension = noise_op.full().shape[-1]

        if ffts is None:  # for single noise op, ffts should be [fk_list, fft]
            fk_list, _, fft = filter_weight(prop_array, t_list, noise_op, trunc_freq=trunc_freq)

        elif type(ffts) == list:
            if any(x is None for x in ffts):
                fk_list, _, fft = filter_weight(prop_array, t_list, noise_op, trunc_freq=trunc_freq)

            else:
                fk_list, fft = ffts
        else:
            raise Exception('no right ffts')


        # step 2, renormalize
        fk_list, Sfk_list = Sf_renorm(Sf_list, f_list, t_list, method=method, trunc_freq=trunc_freq, fk_list_input=fk_list)

        # step 3, kdshmap
        exp_lindb = kdshmap(fft, Sfk_list, t_list)
        system_map = np.einsum('jk,lm->jmkl', prop_array[-1], np.conjugate(np.swapaxes(prop_array[-1], 0, 1)))
        system_map = np.matmul(system_map.reshape(dimension*dimension, dimension*dimension), exp_lindb)
        if output == 'all':
            return system_map, fk_list, Sfk_list
        return system_map

    elif type(noise_op) == list:  # Instead, a list of noise_ops

        dimension = noise_op[0].full().shape[-1]  # take the dimension of the system
        fk_list_list = [np.array([None])] * len(noise_op)
        Sfk_list_list = [np.array([None])] * len(noise_op)
        fft_list = [np.array([None])] * len(noise_op)

        if trunc_freq.__class__ != list:  # trunc_freq in this case should also be a list
            trunc_freq = [(0, 0)] * len(noise_op)  # if not, we will create a list

        if ffts.__class__ == list and any(x is not None for x in ffts):  # ffts here should be lists of fk_list and fft, i.e., [fk_list_list, fft_list]

            for n_ in range(len(noise_op)):
                fk_list_list[n_] = ffts[0][n_]  # if given, we take the n_ th ffts
                fft_list[n_] = ffts[1][n_]

        else:
            if any(x is None for x in [prop_array_fft, fk_list]):
                fk_list, prop_array_fft = propagator_fft(prop_array, t_list, trunc_freq=(0, 0))

            for n_ in range(len(noise_op)):
                fk_list_list[n_] = fk_list
                fft_list[n_] = np.einsum('ijmkl,kl->ijm', prop_array_fft, noise_op[n_].full())

                if trunc_freq[n_] != (0, 0):
                    argwhere = np.argwhere(fk_list_list[n_] <= trunc_freq[n_][1]).transpose()[0]
                    fk_list_list[n_] = fk_list_list[n_][argwhere]
                    fft_list[n_] = fft_list[n_][argwhere]
                    argwhere = np.argwhere(fk_list_list[n_] >= trunc_freq[n_][0]).transpose()[0]
                    fk_list_list[n_]= fk_list_list[n_][argwhere]
                    fft_list[n_] = fft_list[n_][argwhere]

                fk_list_list[n_], Sfk_list_list[n_] = Sf_renorm(Sf_list[n_], f_list[n_], t_list, method=method,
                                                                trunc_freq=trunc_freq[n_], fk_list_input=fk_list_list[n_])

        exp_lindb = kdshmap_nops(fft_list, Sfk_list_list, t_list)

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
                  trunc_freq: Union[Tuple, list] = (0, 0),
                  options=q.Options(atol=1e-10, rtol=1e-10),
                  solver: str = 'qutip',
                  u0_list: np.ndarray = None,
                  method: str = 'trapz',
                  prop_array: np.ndarray = None,
                  output: str = None,
                  multicore: str ='pathos'):

    if t_list_full is None:
        N_expand = int((t_list_sub[1]-t_list_sub[0])/minimal_step)
        t_list = np.linspace(t_list_sub[0], t_list_sub[-1], N_expand*(len(t_list_sub)-1)+1)
    else:
        t_list = t_list_full
        N_expand = int((t_list_sub[1]-t_list_sub[0])/minimal_step)

    if prop_array is None:
        prop_array = propagator(H, t_list, options, solver=solver, u0_list=u0_list)

    dimension = prop_array.shape[-1]
    system_map_list = np.zeros((len(t_list_sub), dimension*dimension, dimension*dimension), dtype=complex)
    system_map_list[0] = np.eye(dimension*dimension)
    if multicore == 'pathos':

        def run(x):
            return generate_map_single(H, t_list[0:N_expand*x+1], noise_op, f_list, Sf_list, trunc_freq=trunc_freq,
                                       options=options, solver=solver, u0_list=u0_list, method=method,
                                       prop_array=prop_array[0:N_expand*x+1], ffts=None, output='map')
        results = ProcessingPool().map(run, np.arange(1, len(t_list_sub)))
        for j in range(1, len(t_list_sub)):
            system_map_list[j] = results[j-1]

    else:
        fk_list = None
        Sfk_list = None
        for j in range(len(t_list_sub)):

            if t_list_sub[j] == np.amin(t_list_sub):
                system_map_list[j] = np.eye(dimension*dimension)
                continue

            if j != len(t_list_sub)-1:
                system_map_list[j] = generate_map_single(H, t_list[0:N_expand*j+1], noise_op, f_list, Sf_list, trunc_freq=trunc_freq,
                                                         options=options, solver=solver, u0_list=u0_list, method=method,
                                                         prop_array=prop_array[0:N_expand*j+1], ffts=None, output='map')

            else:
                system_map_list[j], fk_list, Sfk_list = generate_map_single(H, t_list[0:N_expand*j+1], noise_op, f_list, Sf_list, trunc_freq=trunc_freq,
                                                                            options=options, solver=solver, u0_list=u0_list, method=method,
                                                                            prop_array=prop_array[0:N_expand*j+1], ffts=None, output='all')
        if output == 'all':
            return system_map_list, fk_list, Sfk_list

    return system_map_list
