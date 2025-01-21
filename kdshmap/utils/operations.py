import numpy as np
import qutip as q


def damped_density(density, exp_lindb):

    density = density.full()
    dimension = density.shape[-1]
    density = density.reshape(dimension*dimension)
    density = np.matmul(exp_lindb, density)
    density = density.reshape(dimension, dimension)

    return q.Qobj(density)


def damped_densities(density, kdshmap_list, output_type='qutip'):

    density = density.full()
    dimension = density.shape[-1]
    density = density.reshape(dimension*dimension)
    damped_density_list = np.array([None]*len(kdshmap_list))

    if output_type == 'qutip':
        for j in range(len(kdshmap_list)):
            new_density = np.matmul(kdshmap_list[j], density)
            new_density = new_density.reshape(dimension, dimension)
            damped_density_list[j] = q.Qobj(new_density)

    if output_type == 'numpy':
        damped_density_list = np.einsum('ijk,k->ij', kdshmap_list, density)

    return damped_density_list


def expect(density,  e_op_list, damped_density_list=None, kdshmap_list=None, store_states=False):

    if damped_density_list is None:
        if kdshmap_list is None:
            raise Exception('no sufficient input')
        else:
            damped_density_list = damped_densities(density, kdshmap_list, output_type='numpy')
    dimension2 = damped_density_list.shape[-1]
    e_map = np.zeros((len(e_op_list), damped_density_list.shape[0]), dtype=complex)
    for op_ in range(len(e_op_list)):
        e_op = e_op_list[op_].full()
        e_op = (e_op.transpose()).reshape(dimension2)
        e_map[op_] = np.einsum('ij,j->i', damped_density_list, e_op)
    if store_states is False:
        return e_map
    else:
        return e_map, damped_density_list


def decoh_error(single_map):

    dimension_sq = single_map.shape[-1]
    return 1 - np.trace(single_map)/dimension_sq

