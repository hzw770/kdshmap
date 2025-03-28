import qutip as q
import numpy as np
import scipy as sp
import copy

def propagator_stabilized_superop_fft(H, noise_op, c_ops, t_list, options=dict(atol=1e-10, rtol=1e-10), solver_type='qutip'):

    noise_op_array = noise_op.full()
    dimension = noise_op_array.shape[-1]

    noise_op_super_left = q.sprepost(noise_op, q.qeye(dimension)).full()
    noise_op_super_right = q.sprepost(q.qeye(dimension), noise_op).full()

    noise_op_t_left = np.zeros((len(t_list), dimension * dimension, dimension * dimension), dtype=complex)
    noise_op_t_right = np.zeros((len(t_list), dimension * dimension, dimension * dimension), dtype=complex)
    noise_op_t_left[0] = noise_op_super_left
    noise_op_t_right[0] = noise_op_super_right

    if solver_type == 'qutip':
        prop_list = np.zeros((len(t_list)-1, dimension*dimension, dimension*dimension), dtype=complex)
        prop_inv_list = np.zeros((len(t_list)-1, dimension*dimension, dimension*dimension), dtype=complex)
        prop_final = np.eye(dimension*dimension, dimension*dimension)

        for t_ in range(len(t_list)-1):
            time_slice = np.linspace(t_list[t_], t_list[t_+1], 2)
            prop_list[t_] = (q.propagator(H, time_slice, options=options, c_ops=c_ops)[-1]).full()
            prop_inv_list[t_] = np.linalg.inv(prop_list[t_])
            prop_final = np.einsum('ij,jk->ik', prop_list[t_], prop_final)

        for t_ in range(len(t_list)-1):
            noise_op_t_left[t_+1] = copy.deepcopy(noise_op_t_left[0])
            noise_op_t_right[t_+1] = copy.deepcopy(noise_op_t_right[0])
            for t__ in range(t_+1):
                noise_op_t_left[t_+1] = np.einsum('ij,jk->ik',
                                                np.einsum('ij,jk->ik', prop_inv_list[t_-t__], noise_op_t_left[t_+1]),
                                                prop_list[t_-t__])
                noise_op_t_right[t_+1] = np.einsum('ij,jk->ik',
                                                np.einsum('ij,jk->ik', prop_inv_list[t_-t__], noise_op_t_right[t_+1]),
                                                prop_list[t_-t__])
    if solver_type == 'magnus':
        return None

    fk_list = -np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])
    noise_op_t_left_fft = np.fft.fft(noise_op_t_left[0:len(t_list)-1], axis=0)/(len(t_list)-1)
    noise_op_t_right_fft = np.fft.fft(noise_op_t_right[0:len(t_list)-1], axis=0)/(len(t_list)-1)

    # argsort = np.argsort(fk_list)
    # fk_list = fk_list[argsort]
    # noise_op_t_left_fft = noise_op_t_left_fft[argsort]
    # noise_op_t_right_fft = noise_op_t_right_fft[argsort]

    # if trunc_freq is not None:
    #     argwhere = np.argwhere(fk_list <= trunc_freq[1]).transpose()[0]
    #     fk_list = fk_list[argwhere]
    #     noise_op_t_left_fft = noise_op_t_left_fft[argwhere]
    #     noise_op_t_right_fft = noise_op_t_right_fft[argwhere]
    #     argwhere = np.argwhere(fk_list >= trunc_freq[0]).transpose()[0]
    #     if len(argwhere) == 0:
    #         raise Exception('no filter_ops, change trunc_freq')
    #     fk_list = fk_list[argwhere]
    #     noise_op_t_left_fft = noise_op_t_left_fft[argwhere]
    #     noise_op_t_right_fft = noise_op_t_right_fft[argwhere]

    return fk_list, noise_op_t_left_fft, noise_op_t_right_fft, prop_final


def filter_weight_stabilized_for_state(noise_op_t_left_fft, noise_op_t_right_fft, prop_final, density0, t_list, trunc_freq=None):

    fk_list = -np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])
    argsort = np.argsort(fk_list)
    fk_list = fk_list[argsort]
    noise_op_t_left_fft = noise_op_t_left_fft[argsort]
    noise_op_t_right_fft = noise_op_t_right_fft[argsort]

    if len(fk_list) % 2 == 0:
        fk_list = fk_list[:-1]
        noise_op_t_left_fft = noise_op_t_left_fft[:-1]
        noise_op_t_right_fft = noise_op_t_right_fft[:-1]

    noise_op_t_left_dag_fft = noise_op_t_left_fft[::-1]
    noise_op_t_right_dag_fft = noise_op_t_right_fft[::-1]

    lindb =  0.5 * np.einsum('ijk,ikl->ijl', noise_op_t_left_fft, noise_op_t_right_dag_fft)
    lindb += 0.5 * np.einsum('ijk,ikl->ijl', noise_op_t_right_dag_fft, noise_op_t_left_fft)
    lindb += -0.5 * np.einsum('ijk,ikl->ijl', noise_op_t_left_dag_fft, noise_op_t_left_fft)
    lindb += -0.5 * np.einsum('ijk,ikl->ijl', noise_op_t_right_fft, noise_op_t_right_dag_fft)
    lindb = np.einsum('jk,ikl->ijl', prop_final, lindb)

    density = density0.full()
    dimension = density.shape[-1]
    density = density.reshape(dimension*dimension)

    density = q.operator_to_vector(density0).full().reshape(dimension*dimension)
    

    density_decay = np.einsum('ijk,k->ij', lindb, density)
    density_decay = density_decay.reshape(len(fk_list), dimension, dimension)
    filter_weights_state = np.einsum('ijk,ijk->i', density_decay, np.conjugate(density_decay))

    argsort = np.argsort(fk_list)
    fk_list = fk_list[argsort]
    filter_weights_state = filter_weights_state[argsort]

    if trunc_freq is not None:
        argwhere = np.argwhere(fk_list <= trunc_freq[1]).transpose()[0]
        fk_list = fk_list[argwhere]
        filter_weights_state = filter_weights_state[argwhere]
        argwhere = np.argwhere(fk_list >= trunc_freq[0]).transpose()[0]
        if len(argwhere) == 0:
            raise Exception('no filter_ops, change trunc_freq')
        fk_list = fk_list[argwhere]
        filter_weights_state = filter_weights_state[argwhere]

    return fk_list, np.sqrt(filter_weights_state.real)