import qutip as q
import numpy as np
import scipy as sp


def propagator(H, t_list, 
               options=q.Options(atol=1e-10, rtol=1e-10), 
               solver_type='qutip',
               u0_list: np.ndarray = None):
    
    """
    Given a Hamiltonian and a list of time values, return the propagator at each time.
    Can specify the solver_type to be used, either 'qutip' or 'magnus', for a Magnus 
    expansion type, perturbative calculation of the propagator.

    Parameters:

    H         : Qobj or list
                Hamiltonian operator or list of Hamiltonian operators
    
    t_list    : numpy array
                List of time values at which to calculate the propagator
    
    options   : qutip.Options (optional)
                Options for the solver_type, default is atol=1e-10, rtol=1e-10

    solver_type    : str (optional)
                solver_type to be used, either 'qutip' or 'magnus'
    
    u0_list   : numpy array (optional)
                A unitary frame transformations, if the Hamiltonian is provided in a different frame

    Returns   :

    prop_array: numpy array
                Array of propagators at each time value
                
    """

    if solver_type == 'qutip':

        prop_qobj  = q.propagator(H, t_list, options=options, c_op_list=[])
        prop_array = np.zeros((len(t_list), prop_qobj[0].dims[0][0], prop_qobj[0].dims[0][0]), dtype=complex)

        #Checks if any frame transformation has been applied
        if u0_list is None:
            for t_ in range(len(t_list)):
                #Converts the propagator to a numpy array
                prop_array[t_] = prop_qobj[t_].full()
            return prop_array
        
        else:
            # If a frame transformation has been applied, transform the propagator back to the original frame
            if len(u0_list) == len(t_list):

                for t_ in range(len(t_list)):
                    #At each time value, the propagator is transformed back to the original frame
                    prop_array[t_] = (u0_list[t_] * prop_qobj[t_] * u0_list[0].dag()).full()
            else:
                return None
            return prop_array

    if solver_type == 'magnus':
        # For construction later, maybe more efficient for numerics (Unclear if there is an advantage in optimizing this step)
        return None


def propagator_superop_fft(prop_array, t_list, trunc_freq=None):

    """
    Given samples of the propagator in the time basis, compute the closed system superoperator (U^\dag \otimes U^T),
    transform it to the frequency basis through an FFT, and return it.

    Parameters            :

    prop_array            : numpy array
                            Array of propagators (numpy arrays) at each time value
    
    t_list                : numpy array    
                            List of time values at which the propagators are calculated/sampled

    trunc_freq            : tuple (optional)
                            Tuple of two floats, the lower and upper bounds of the frequency range 
                            in which the DFT is to be calculated

    Returns                 :
    fk_list                 : numpy array
                              List of frequencies at which the DFT is calculated (ordered negative to positive)
    
    prop_superop_array_fft  : numpy array
                              Array of 'super-propagators' in the frequency basis : Convention followed is the opposite of that in np.fft.fft 

    """

    #The inverse of the propagator at each time step (U^\dag)	
    prop_array_dag = np.conjugate(np.swapaxes(prop_array, 1, 2)) #0 axis is the time axis

    #The superoperator (U^\dag \otimes U^T) in the time basis and its fft 
    prop_superop_array     = np.einsum('ijk,ilm->ijmkl', prop_array_dag, prop_array)

    prop_superop_array_fft = np.fft.ifft(prop_superop_array[0:len(t_list)-1], axis=0)
    #Contribution from the last time step is ignored to avoid aliasing issues & ifft is called due to convention

    #List of frequencies at which the FFT is calculated
    fft_fk_list = np.fft.fftfreq(len(t_list)-1, t_list[1]-t_list[0])

    #Order the FFT frequencies from negative to positive
    argsort = np.argsort(fft_fk_list)
    fft_fk_list = fft_fk_list[argsort]
    
    #Sort the FFT of the superoperator accordingly
    prop_superop_array_fft = prop_superop_array_fft[argsort]

    if trunc_freq is not None:
        
        #Truncate the frequency range & the fft of the superoperator accordingly
        argwhere = np.argwhere(fft_fk_list <= trunc_freq[1]).transpose()[0]
        fft_fk_list = fft_fk_list[argwhere]
        prop_superop_array_fft = prop_superop_array_fft[argwhere]
        argwhere = np.argwhere(fft_fk_list >= trunc_freq[0]).transpose()[0]
        fft_fk_list = fft_fk_list[argwhere]
        prop_superop_array_fft = prop_superop_array_fft[argwhere]


    return fft_fk_list, prop_superop_array_fft
