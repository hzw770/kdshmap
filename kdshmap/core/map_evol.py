# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
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


def generate_map_single(H                      : Union[list, q.qobj.Qobj],
                        t_list                 : np.ndarray,
                        noise_op               : Union[list, q.qobj.Qobj],
                        f_list                 : Union[list, np.ndarray],
                        Sf_list                : Union[list, np.ndarray],
                        trunc_freq             : Union[Tuple, list] = None,
                        options                = q.Options(atol=1e-10, rtol=1e-10),
                        solver_type            : str = 'qutip',
                        u0_list                : np.ndarray = None,
                        spd_renorm_method      : str = 'trapz',
                        prop_array             : np.ndarray = None,
                        filter_op_props        : list = None,
                        output_type            : str = 'map',
                        prop_array_superop_fft : np.ndarray = None,
                        fk_list                : Union[np.ndarray, list] = None):
    
    """
    Generate a single dynamical map to evolve the system.
    
    Parameters:
    
    H                           : Union[list, q.qobj.Qobj]
                                  The Hamiltonian of the system. If a list is given, the Hamiltonian is time-dependent.   
        
    t_list                      : np.ndarray
                                  The time list of the system. This is the time list that the system is evolved over. 
            
    noise_op                    : Union[list, q.qobj.Qobj]
                                  The noise operator(s). If a list is given, 
                                  the system couples to the bath through multiple noise operators. 
            
    f_list                      : Union[list, np.ndarray]
                                  The frequencies at which the noise spectrak density has been sampled. 
                                  Between these values Sf_renorm will interpolate the spectral density.
                                  If the system couples to multiple noise operators, a list is given.
            
    Sf_list                     : Union[list, np.ndarray]
                                  The spectral density of the noise operator(s).
             
    trunc_freq                  : Union[Tuple, list]
                                  The frequency range of the noise operator(s) that will be considered in the calculation.
                  
    options                     : q.Options
                                  Options for the solver.
                                  Default options are q.Options(atol=1e-10, rtol=1e-10)
                  
    solver_type                 : str
                                  The solver to be used for the propagator. 
                                  Default is 'qutip'. Other option is 'Magnus' (currently in development)
                  
    u0_list                     : np.ndarray
                                  List of unitary transformations specifying frame transformations since it may be easier to compute
                                  the propagator in a rotating frame. By default, the propagator is computed in the lab frame.
                  
    spd_renorm_method           : str
                                  The spd_renorm_method used for the integration of the spectral density. 
                                  Default is 'trapz'. Other option is 'sinc'. 
                          
    prop_array                  : np.ndarray
                                  The propagator. If not given, it will be calculated.  
                          
    filter_op_props             : list
                                  A list of tuples of the filter frequencies and corresponding filter superoperators.
                          
    output_type                 : str
                                  The output_type of the function. 
                                  Default is 'map'. Other option is 'all'. 
                          
    prop_array_superop_fft      : np.ndarray
                                  The filter_op_props of the propagator. If not provided, it will be calculated.
                          
    fk_list                      : Union[np.ndarray, list]
                                   The frequency list of the noise operator(s). If not provided, it will be calculated.   
                                   
    Returns:
    system_map                   : np.ndarray
                                   The Keldysh map for the given system, propagator, 
                                   noise operator(s), and spectral density.               

    """
    
    #If the propagator is not given, calculate it
    if prop_array is None:
        prop_array = propagator(H, t_list, options, solver_type=solver_type, u0_list=u0_list)
        
        
    #Calc the fft of the propagator superoperator    
    if prop_array_superop_fft is None:
        _, prop_array_superop_fft = propagator_superop_fft(prop_array, t_list, trunc_freq=None)

    # If only one noise op is given, calculate the map for that noise op
    if type(noise_op) == q.qobj.Qobj: 

        if trunc_freq is None:
            trunc_freq = (np.amin(f_list), np.amax(f_list))
        dimension = noise_op.full().shape[-1]
        
        #If the filter freqs & superoperators are not given, calculate them
        if filter_op_props is None:  
            fk_list, _, filter_op      = filter_weight(prop_array, 
                                                       t_list, 
                                                       noise_op, 
                                                       trunc_freq=trunc_freq, 
                                                       prop_array_superop_fft=prop_array_superop_fft)
        elif type(filter_op_props) == list:
            if any(x is None for x in filter_op_props):
                fk_list, _, filter_op     = filter_weight(prop_array, 
                                                          t_list, 
                                                          noise_op, 
                                                          trunc_freq=trunc_freq, 
                                                          prop_array_superop_fft=prop_array_superop_fft)

            else:
                fk_list, filter_op = filter_op_props
        else:
            raise Exception('Incorrect format for filter_op_props. It is a list of tuples of the filter frequencies (floats) and corresponding filter operators (ndarrays).')

        # Renormalize the spectral density by 'smoothening' it out and sample it at the filter frequencies
        fk_list, Sfk_list = Sf_renorm(Sf_list, 
                                      f_list, 
                                      t_list, 
                                      spd_renorm_method=spd_renorm_method, 
                                      trunc_freq=trunc_freq, 
                                      fk_list_input=fk_list)

        #Calculate the Keldysh map using the filter operators and the renormalized spectral densities at the filter frequencies
        kdsh_map = kdshmap(filter_op, Sfk_list, t_list)
        
        #Calculate the transformation matrix that took the system to the interaction picture (this transformation must be undone here)
        system_map = np.einsum('jk,lm->jmkl', prop_array[-1], np.conjugate(np.swapaxes(prop_array[-1], 0, 1)))
        
        #The entire system map is the Keldysh map, transformed to the Schrodinger picture
        system_map = np.matmul(system_map.reshape(dimension*dimension, dimension*dimension), kdsh_map)
        
        if output_type == 'all':
            
            return system_map, fk_list, Sfk_list
        
        return system_map
    
    #If multiple noise operators are given, calculate the map for each noise operator and sum them
    elif type(noise_op) == list:
        
        dimension = noise_op[0].full().shape[-1] 
        
        #Each element of these lists are a list of the frequencies sampled in the noise spectral density & the corresponding spectral densities resp.
        fk_list_list        = [np.array([None])] * len(noise_op)
        Sfk_list_list       = [np.array([None])] * len(noise_op)
        
        #Each element of these lists are lists of filter superoperators for each noise operator
        filter_op_list      = [np.array([None])] * len(noise_op)

        if trunc_freq.__class__ != list:  # trunc_freq in this case should also be a list
            trunc_freq = [None] * len(noise_op)  # if not, create the required list
            
        # filter_op_props here should be a 2D list of fk_list and fft, i.e., [fk_list_list, filter_op_list]. If any filter_op_props is None, it will be calculated.
        
        if filter_op_props.__class__ == list and any(x is not None for x in filter_op_props): 
            
            for n_ in range(len(noise_op)):
                
                fk_list_list[n_]       = filter_op_props[0][n_]  # if given, we take the n_ th filter_op_props
                filter_op_list[n_]     = filter_op_props[1][n_]

        else:

            if any(x is None for x in [prop_array_superop_fft, fk_list]):
                fk_list, prop_array_superop_fft = propagator_superop_fft(prop_array, t_list, trunc_freq=None)

            for n_ in range(len(noise_op)):
                fk_list_list[n_]   = fk_list
                filter_op_list[n_] = np.einsum('ijmkl,kl->ijm', prop_array_superop_fft, noise_op[n_].full())

                if trunc_freq[n_] is not None:
                    argwhere = np.argwhere(fk_list_list[n_] <= trunc_freq[n_][1]).transpose()[0]
                    
                    fk_list_list[n_]   = fk_list_list[n_][argwhere]
                    filter_op_list[n_] = filter_op_list[n_][argwhere]
                    
                    argwhere = np.argwhere(fk_list_list[n_] >= trunc_freq[n_][0]).transpose()[0]
                    fk_list_list[n_]   = fk_list_list[n_][argwhere]
                    filter_op_list[n_] = filter_op_list[n_][argwhere]

        fk_list_list[n_], Sfk_list_list[n_] = Sf_renorm(Sf_list[n_], 
                                                        f_list[n_], 
                                                        t_list, 
                                                        spd_renorm_method=spd_renorm_method,
                                                        trunc_freq=trunc_freq[n_], 
                                                        fk_list_input=fk_list_list[n_])
        
        exp_lindb = kdshmap_multi_n_ops(filter_op_list, Sfk_list_list, t_list)

        system_map = np.einsum('jk,lm->jmkl', prop_array[-1], np.conjugate(np.swapaxes(prop_array[-1], 0, 1)))
        system_map = np.matmul(system_map.reshape(dimension*dimension, dimension*dimension), exp_lindb)
        
        if output_type == 'all':
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
    
    """
    If the map/state is to be computed for multiple times, this function is used. It calculates the map/state for each time in t_list_sub.
    It calls the generate_map_single function for each time in t_list_sub.
    
    Parameters:
        
    H                           : Union[list, q.qobj.Qobj]
                                  The Hamiltonian of the system. If a list is given, the Hamiltonian is time-dependent.   
        
    t_list_sub                  : np.ndarray
                                  The time list of the system. This is the time list that the system is evolved over. 
                                  
    minimal_step                : float
                                  The minimal step size of the time list.
            
    noise_op                    : Union[list, q.qobj.Qobj]
                                  The noise operator(s). If a list is given, 
                                  the system couples to the bath through multiple noise operators. 
            
    f_list                      : Union[list, np.ndarray]
                                  The frequencies at which the noise spectrak density has been sampled. 
                                  Between these values Sf_renorm will interpolate the spectral density.
                                  If the system couples to multiple noise operators, a list is given.
            
    Sf_list                     : Union[list, np.ndarray]
                                  The spectral density of the noise operator(s).
                                  
    t_list_full                 : np.ndarray
                                  The full time list of the system. If not provided, it will be calculated.               
             
    trunc_freq                  : Union[Tuple, list]
                                  The frequency range of the noise operator(s) that will be considered in the calculation.
                  
    options                     : q.Options
                                  Options for the solver.
                                  Default options are q.Options(atol=1e-10, rtol=1e-10)
                  
    solver_type                 : str
                                  The solver to be used for the propagator. 
                                  Default is 'qutip'. Other option is 'Magnus' (currently in development)
                  
    u0_list                     : np.ndarray
                                  List of unitary transformations specifying frame transformations since it may be easier to compute
                                  the propagator in a rotating frame. By default, the propagator is computed in the lab frame.
                  
    spd_renorm_method           : str
                                  The spd_renorm_method used for the integration of the spectral density. 
                                  Default is 'trapz'. Other option is 'sinc'. 
                          
    prop_array                  : np.ndarray
                                  The propagator. If not given, it will be calculated.  
                          
    multicore                   : str
                                  The multicore option for multiprocessing. Default is 'pathos'. Other option is 'None'.   
                                  
    Returns:
    system_map_list             : np.ndarray
                                  The list of Keldysh maps for the given system, propagator, noise operator(s), and spectral density.       
    """
    
    
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
            return generate_map_single(H, 
                                       t_list[0:N_expand*x+1], 
                                       noise_op, 
                                       f_list, 
                                       Sf_list, 
                                       trunc_freq=trunc_freq,
                                       options=options, 
                                       solver_type=solver_type, 
                                       u0_list=u0_list, 
                                       spd_renorm_method=spd_renorm_method,
                                       prop_array=prop_array[0:N_expand*x+1], 
                                       filter_op_props=None, 
                                       output_type='map')
            
        results = ProcessingPool().map(run, np.arange(1, len(t_list_sub)))
        for j in range(1, len(t_list_sub)):
            system_map_list[j] = results[j-1]

    else:
        for j in range(len(t_list_sub)):

            if t_list_sub[j] == np.amin(t_list_sub):
                system_map_list[j] = np.eye(dimension*dimension)
                continue

            system_map_list[j] = generate_map_single(H, 
                                                     t_list[0:N_expand*j+1], 
                                                     noise_op, 
                                                     f_list, 
                                                     Sf_list, 
                                                     trunc_freq=trunc_freq,
                                                     options=options, 
                                                     solver_type=solver_type, 
                                                     u0_list=u0_list, 
                                                     spd_renorm_method=spd_renorm_method,
                                                     prop_array=prop_array[0:N_expand*j+1], 
                                                     filter_op_props=None, 
                                                     output_type='map')
    return system_map_list

