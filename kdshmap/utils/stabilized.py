import qutip as q
import numpy as np
import scipy as sp
import copy
from pathos.multiprocessing import ProcessingPool

def propagator_stabilized_superop_fft(H, c_ops, t_list, options=dict(atol=1e-10, rtol=1e-10), solver_type='qutip'):

    dimension = c_ops[0].shape[-1]
    if solver_type == 'qutip':
        prop_list = np.zeros((len(t_list)-1, dimension*dimension, dimension*dimension), dtype=np.complex128)
        # prop_inv_list = np.zeros((len(t_list)-1, dimension*dimension, dimension*dimension), dtype=np.complex128)
        prop_final = np.eye(dimension*dimension, dimension*dimension)


        prop_t_list = (q.propagator(H, t_list, options=options, c_ops=c_ops))
        # prop_inv_t_list = (q.propagator(H, t_list[::-1], options=options, c_ops=c_ops))
        prop_final = np.einsum('ij,jk->ik', prop_t_list[-1].full(), prop_final)

    if solver_type == 'magnus':
        return None

    fk_list = -np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])
    prop_fft = np.array([prop_t_list[j].full() for j in range(len(t_list))], dtype = complex)
    prop_fft = np.fft.fft(prop_fft[0:len(t_list)-1], axis=0)/(len(t_list)-1)

    return fk_list, prop_fft, prop_final


def filter_weight_stabilized_for_state(prop_fft, prop_final, noise_op, density0, t_list, trunc_freq=None, steady_state=True):

    noise_op_array = noise_op.full()
    noise_op_dag_array = noise_op.dag().full()

    dimension = noise_op_array.shape[-1]

    noise_op_super_left = q.sprepost(noise_op, q.qeye(dimension)).full()
    noise_op_super_right = q.sprepost(q.qeye(dimension), noise_op).full()
    noise_op_dag_super_left = q.sprepost(noise_op.dag(), q.qeye(dimension)).full()
    noise_op_dag_super_right = q.sprepost(q.qeye(dimension), noise_op.dag()).full()

    fk_list = -np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])
    argsort = np.argsort(fk_list)
    fk_list = fk_list[argsort]
    prop_fft = prop_fft[argsort]


    # if trunc_freq is not None:
    #     argwhere = np.argwhere(fk_list <= trunc_freq[1]).transpose()[0]
    #     fk_list_focus = fk_list[argwhere]
    #     argwhere = np.argwhere(fk_list_focus >= trunc_freq[0]).transpose()[0]
    #     if len(argwhere) == 0:
    #         raise Exception('no filter_ops, change trunc_freq')
    #     fk_list_focus = fk_list_focus[argwhere]
    if trunc_freq is not None:
        argwhere = np.where((fk_list<=trunc_freq[1]) & (fk_list>=trunc_freq[0]))
        fk_list_focus = fk_list[argwhere]


    lindb = np.zeros((len(fk_list_focus),  prop_fft[0].shape[-1], prop_fft[0].shape[-1]), dtype = complex)

    
    def run(fk):
        translation = int(fk*t_list[-1])
        temp_Xk = np.roll(prop_fft, translation, axis = 0)
        temp_Xk2 = np.roll(prop_fft, -translation, axis = 0)
        if steady_state is True:
            argwhere_int = np.where(fk_list == 0)[0]
        else:
            argwhere_int = np.arange(abs(translation), len(fk_list)-abs(translation))


        lindb_local = 0.5 * np.sum(np.einsum('ijk,ikl->ijl',
                                      np.einsum('ijk,ikl->ijl', prop_fft[argwhere_int]@noise_op_super_left, temp_Xk2[argwhere_int]@noise_op_dag_super_right),
                                      prop_fft[argwhere_int]), axis = 0)
        lindb_local += 0.5 * np.sum(np.einsum('ijk,ikl->ijl',
                                      np.einsum('ijk,ikl->ijl', prop_fft[argwhere_int]@noise_op_dag_super_right, temp_Xk[argwhere_int]@noise_op_super_left),
                                      prop_fft[argwhere_int]), axis = 0)
        lindb_local -= 0.5 * np.sum(np.einsum('ijk,ikl->ijl',
                                      np.einsum('ijk,ikl->ijl', prop_fft[argwhere_int]@noise_op_dag_super_left, temp_Xk[argwhere_int]@noise_op_super_left),
                                      prop_fft[argwhere_int]), axis = 0)
        lindb_local -= 0.5 * np.sum(np.einsum('ijk,ikl->ijl',
                                      np.einsum('ijk,ikl->ijl', prop_fft[argwhere_int]@noise_op_super_right, temp_Xk2[argwhere_int]@noise_op_dag_super_right),
                                      prop_fft[argwhere_int]), axis = 0)
        return lindb_local

    results = ProcessingPool().map(run, fk_list_focus)
    for fk_ in range(len(fk_list_focus)):
        lindb[fk_] = results[fk_]

    density = density0.full()
    dimension = density.shape[-1]
    density = density.reshape(dimension*dimension)

    density = q.operator_to_vector(density0).full().reshape(dimension*dimension)


    density_decay = np.einsum('ijk,k->ij', lindb, density)
    # density_decay = density_decay.reshape(len(fk_list), dimension, dimension)
    # filter_weights_state = np.einsum('ijk,ijk->i', density_decay, np.conjugate(density_decay))
    filter_weights_state = np.einsum('ij,ij->i', density_decay, np.conjugate(density_decay))
    
    return fk_list_focus, np.sqrt(filter_weights_state.real)
    