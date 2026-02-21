#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to compute marginal and two-point distributions

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

import numpy as np
import pandas as pd
import ray
from itertools import product


def marginal_distribution(Psi, t, transform_dict, cutoff=50, resolution=100, range_out=None,
                          parallel=False,num_cpus=8):
    """marginal_distribution.

    Example of transform function (to measure cumulative detections at sentinels):

    def transform(state_vars,c_):
        state_vars['cumulative detection'] *= c_

    Parameters
    ----------
    Psi : PGF object
        Main generating function.
    t : array_like
        Time at which to evaluate the pgf
    transform_dict : dict
        Dictionary of list of transform functions. The key is the time at which the transform is applied. One
        of the transform function should have c_ as argument to transform the variables
    cutoff: int
        Characteristic scale of the exponential filter (for anti-aliasing). The filter takes the value e^(-1)
        at the cut-off.
    resolution: int
        Even integer for the number of points to use for the fft.
    range_out: tuple
        Tuple of integer for the range to output (excluding the upper bound).
    """
    radius = np.exp(-1 / cutoff) #anti-aliasing

    if (resolution % 2) > 0:
        raise ValueError("The resolution must be even")

    if range_out is None:
        range_out = (0,resolution) #not inclusive of right bound
    elif range_out[1] > resolution:
        raise ValueError("Upper bound of range larger than resolution")

    nb_eval = resolution // 2 + 1
    nodes = np.arange(resolution) / (resolution)
    resolution_out = range_out[1]-range_out[0]

    c = np.exp(2 * np.pi * complex(0, 1) * nodes[:nb_eval]) * radius

    #state variables
    state_vars = Psi.get_initial_state_vars(1.)

    Psi_c = {label:np.zeros((len(t),c.shape[0]),dtype=complex) for label in Psi.Psi0}
    char_func = {label:np.zeros((len(t),resolution_out),dtype=complex) for label in Psi.Psi0}
    distribution = {label:np.zeros((len(t),resolution_out)) for label in Psi.Psi0}

    if parallel:
        #start ray instance
        ray.shutdown()
        ray.init(num_cpus=min([num_cpus,len(c)]))

        #define function to parallelize
        Psi_id = ray.put(Psi)
        state_vars_id = ray.put(state_vars)

        @ray.remote
        def target(c_):
            Psi_ = ray.get(Psi_id)
            state_vars_ = ray.get(state_vars_id)
            out = Psi_(state_vars_,t, transform_dict=transform_dict, c_=c_)
            return out

        #compute
        result_ids = []
        for j in range(c.shape[0]):
            result_ids.append(target.remote(c[j]))
        results = ray.get(result_ids)
        #format
        for j in range(c.shape[0]):
            out = results[j]
            for label in out:
                Psi_c[label][:,j] = out[label]

    else:
        #evaluate characteristic function at nodes
        for j in range(c.shape[0]):
            out = Psi(state_vars,t, transform_dict=transform_dict, c_=c[j]) #apply transformation at beginning
            for label in out:
                Psi_c[label][:,j] = out[label]

    #obtain pdf from characteristic function
    for label in distribution:
        for ind in range(len(t)):
            spectrum = np.concatenate((Psi_c[label][ind],np.conjugate(Psi_c[label][ind][-2:0:-1])))
            distribution[label][ind,:] = np.absolute(np.fft.fft(spectrum))[range_out[0]:range_out[1]]
            distribution[label][ind,:] /= resolution
            distribution[label][ind,:] /= radius**np.arange(range_out[0],range_out[1])
            char_func[label][ind] = spectrum[range_out[0]:range_out[1]] #not exactly because of anti-aliasing

    #transform output in dataframe format
    results = pd.DataFrame(columns=['label','time','target','probability'])
    results['probability'] = np.concatenate([np.ravel(distribution[label]) for label in distribution])
    results['target'] = np.tile(np.arange(range_out[0],range_out[1]), len(t)*len(distribution))
    results['time'] = np.tile(np.repeat(t,resolution_out), len(distribution))
    results['label'] = list(np.repeat(list(distribution.keys()), resolution_out*len(t), axis=0))
    #works for labels that are tuple/list

    return results


def marginal_p0(Psi, t, transform_dict):
    """marginal_p0. Get the first term of the PGF expansion (probability to have 0)

    See marginal_distribution for an example of transform.

    Parameters
    ----------
    Psi : PGF object
        Main generating function.
    t : array_like
        Time at which to evaluate the pgf
    transform_dict : dict
        Dictionary of list of transform functions. The key is the time at which the transform is applied. One
        of the transform function should have c_ as argument to transform the variables
    """

    #state variables
    state_vars = Psi.get_initial_state_vars(1.)

    distribution = Psi(state_vars,t, transform_dict=transform_dict, c_=0)

    #transform output in dataframe format
    results = pd.DataFrame(columns=['label','time','target','probability'])
    results['probability'] = np.concatenate([distribution[label] for label in distribution])
    results['target'] = np.tile([0], len(t)*len(distribution))
    results['time'] = np.tile(t, len(distribution))
    results['label'] = list(np.repeat(list(distribution.keys()), len(t), axis=0))
    #works for labels that are tuple/list

    return results


def two_point_distribution(Psi, t, transform_dict, cutoff1=50, cutoff2=50, resolution1=100, resolution2=100,
                          range_out1=None, range_out2=None):
    """two_point_distribution.

    See marginal_distribution for an example of transform.

    Parameters
    ----------
    Psi : PGF object
        Main generating function.
    t : array_like
        Time at which to evaluate the pgf
    transform_dict : dict
        Dictionary of list of transforms. The key is the time at which the transform is applied. Two of them should have c1_ or c2_ as argument to transform the variables
    cutoff1 and cutoff2: int
        Characteristic scale of the exponential filter (for anti-aliasing). The filter takes the value e^(-1)
        at the cut-off.
    resolution1 and resolution2 : int
        Even integer for the number of points to use for the fft for each dimension.
    range_out1 and range_out2: tuple
        Tuple of integer for the range to output (excluding the upper bound).
    """
    radius1 = np.exp(-1 / cutoff1) #anti-aliasing
    radius2 = np.exp(-1 / cutoff2) #anti-aliasing

    if (resolution1 % 2) > 0 or (resolution2 % 2) > 0:
        raise ValueError("The resolutions must be even")

    if range_out1 is None:
        range_out1 = (0,resolution1) #not inclusive of right bound
    elif range_out1[1] > resolution1:
        raise ValueError("Upper bound of range larger than resolution")

    if range_out2 is None:
        range_out2 = (0,resolution2) #not inclusive of right bound
    elif range_out2[1] > resolution2:
        raise ValueError("Upper bound of range larger than resolution")

    nodes1 = np.arange(resolution1) / (resolution1)
    nodes2 = np.arange(resolution2) / (resolution2)

    resolution_out1 = range_out1[1]-range_out1[0]
    resolution_out2 = range_out2[1]-range_out2[0]

    nb_eval1 = resolution1 // 2 + 1 #to minimize computation
    nb_eval2 = resolution2 // 2 + 1


    c1 = np.exp(2 * np.pi * complex(0, 1) * nodes1[:nb_eval1]) * radius1
    c2 = np.exp(2 * np.pi * complex(0, 1) * nodes2) * radius2

    #state variables
    state_vars = Psi.get_initial_state_vars(1.)

    Psi_c = {label:np.zeros((len(t),nb_eval1,c2.shape[0]),dtype=complex) for label in Psi.Psi0}
    char_func = {label:np.zeros((len(t),resolution_out1,resolution_out2),dtype=complex) for label in Psi.Psi0}
    distribution = {label:np.zeros((len(t),resolution_out1,resolution_out2)) for label in Psi.Psi0}

    #evaluate characteristic function at nodes
    for j,k in product(range(c1.shape[0]),range(c2.shape[0])):
        out = Psi(state_vars,t, transform_dict=transform_dict, c1_=c1[j], c2_=c2[k]) #apply transformations
        for label in out:
            Psi_c[label][:,j,k] = out[label]

    #obtain pdf from characteristic function
    for label in distribution:
        for ind in range(len(t)):
            spectrum = np.zeros((resolution1,resolution2),dtype=complex)
            #recycle calculated elements
            spectrum[0:nb_eval1,:] = Psi_c[label][ind][0:nb_eval1,:]
            spectrum[nb_eval1:,nb_eval2:] = Psi_c[label][ind][nb_eval1-2:0:-1,nb_eval2-2:0:-1].conj()
            spectrum[nb_eval1:,nb_eval2-1] = Psi_c[label][ind][nb_eval1-2:0:-1,nb_eval2-1].conj()
            spectrum[nb_eval1:,0] = Psi_c[label][ind][nb_eval1-2:0:-1,0].conj()
            spectrum[nb_eval1:,1:nb_eval2-1] = Psi_c[label][ind][nb_eval1-2:0:-1,-1:nb_eval2-1:-1].conj()

            temp = np.absolute(np.fft.fft2(spectrum)) / (resolution1*resolution2)
            distribution[label][ind,:,:] = temp[range_out1[0]:range_out1[1],range_out2[0]:range_out2[1]]
            distribution[label][ind,:,:] /= np.outer(radius1**np.arange(range_out1[0],range_out1[1]),
                                                     radius2**np.arange(range_out2[0],range_out2[1]))
            char_func[label][ind] = spectrum[range_out1[0]:range_out1[1],range_out2[0]:range_out2[1]]

    #transform output in dataframe format
    results = pd.DataFrame(columns=['label','time','target1','target2','probability'])
    results['probability'] = np.concatenate([np.ravel(distribution[label]) for label in distribution])
    results['target2'] = np.tile(np.arange(range_out2[0],range_out2[1]), resolution_out1*len(t)*len(distribution))
    results['target1'] = np.tile(np.repeat(np.arange(range_out1[0],range_out1[1]),resolution_out2), len(t)*len(distribution))
    results['time'] = np.tile(np.repeat(t,resolution_out1*resolution_out2), len(distribution))
    results['label'] = np.repeat(list(distribution.keys()), resolution_out1*resolution_out2*len(t))

    return results

