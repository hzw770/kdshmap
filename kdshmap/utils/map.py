import numpy as np
import scipy as sp


def filter_weight(prop_array, t_list, noise_op, trunc_freq=(0, 0)):
    """
    Some descriptions
    """

    dimension = prop_array[0].shape[-1]
    fk_list = -np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])
    prop_array_dag = np.conjugate(np.swapaxes(prop_array, 1, 2))
    noise_op_array = noise_op.full()
    noise_op_int = np.dot(prop_array_dag, noise_op_array)
    noise_op_int = np.einsum('ijk,ikl->ijl', noise_op_int, prop_array)

    fft = np.fft.fft(noise_op_int[0:len(t_list)-1], axis=0)/(len(t_list)-1)
    fft_dag = np.conjugate(np.swapaxes(fft, 1, 2))
    fft_dag_fft = np.einsum('ijk,ikl->ijl', fft_dag, fft)
    filter_strength = np.trace(fft_dag_fft, axis1=1, axis2=2) - abs(np.trace(fft, axis1=1, axis2=2))**2/dimension

    argsort = np.argsort(fk_list)
    fk_list = fk_list[argsort]
    filter_strength = filter_strength[argsort]
    fft = fft[argsort]

    if trunc_freq != (0, 0):
        argwhere = np.argwhere(fk_list <= trunc_freq[1]).transpose()[0]
        fk_list = fk_list[argwhere]
        filter_strength = filter_strength[argwhere]
        fft = fft[argwhere]
        argwhere = np.argwhere(fk_list >= trunc_freq[0]).transpose()[0]
        fk_list = fk_list[argwhere]
        filter_strength = filter_strength[argwhere]
        fft = fft[argwhere]

    return fk_list, filter_strength.real, fft


def filter_weight_progressive(prop_array, t_list_sub, N_expand, noise_op, trunc_freq=(0, 0)):
    """
    Want to avoid doing redundant FFT
    Better use previous fft to construct new fft
    :return:
    """
    t_list = np.linspace(t_list_sub[0], t_list_sub[-1], N_expand*(len(t_list_sub)-1)+1)

    fk_list_progressive = np.array([None] * len(t_list_sub))
    filter_strength_progressive = np.array([None] * len(t_list_sub))
    fft_progressive = np.array([None] * len(t_list_sub))
    for j in range(len(t_list_sub)):
        if t_list_sub[j] == np.amin(t_list_sub):
            continue
        fk_list, filter_strength, fft = filter_weight(prop_array[0:j*N_expand+1], t_list[0:j*N_expand+1], noise_op, trunc_freq=trunc_freq)
        fk_list_progressive[j] = fk_list
        filter_strength_progressive[j] = filter_strength
        fft_progressive[j] = fft

    return fk_list_progressive, filter_strength_progressive, fft_progressive


def kdshmap(fft, Sfk_list, t, exp=True):

    if np.amax(t) != np.amin(t):
        t = np.amax(t) - np.amin(t)
    else:
        t = abs(np.amax(t))

    dimension = fft[0].shape[-1]
    identity = np.eye(dimension, dtype=complex)
    fft_dag = np.conjugate(np.swapaxes(fft, 1, 2))
    lindb = np.einsum('ijk,ilm->ijklm', fft, fft_dag)
    fft_dag_fft = np.einsum('ijk,ikl->ijl', fft_dag, fft)
    lindb += -0.5 * np.einsum('ijk,lm->ijklm', fft_dag_fft, identity)
    lindb += -0.5 * np.einsum('jk,ilm->ijklm', identity, fft_dag_fft)
    lindb = np.einsum('ijklm,i->jklm', lindb, Sfk_list) * t
    lindb = np.einsum('jklm->jmkl', lindb)
    lindb = lindb.reshape(dimension*dimension, dimension*dimension)
    if exp == True:
        return sp.linalg.expm(lindb)
    else:
        return lindb


def kdshmap_nops(fft_list: list, Sfk_list_list: list, t):

    dimension = fft_list[0][0].shape[-1]
    lindb = np.zeros((dimension*dimension, dimension*dimension), dtype=complex)
    for n_ in range(len(fft_list)):
        lindb += kdshmap(fft_list[n_], Sfk_list_list[n_], t, exp=False)
    return sp.linalg.expm(lindb)
