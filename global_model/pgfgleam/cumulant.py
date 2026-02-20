
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to compute cumulants.

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

import numpy as np
import pandas as pd
from scipy.special import binom, gamma, loggamma, polygamma
from numpy.linalg import inv,det,eigh
from itertools import product

def marginal_cumulants(Psi, t, transform_dict, radius=10**(-4), resolution=4, range_out=None):
    """marginal_cumulants.

    Parameters
    ----------
    Psi : PGF object
        Main generating function.
    t : array_like
        Time at which to evaluate the pgf
    transform_dict : dict
        Dictionary of list of transform functions. The key is the time at which the transform is applied. One
        of the transform function should have c_ as argument to transform the variables
    radius: float
        Radius of the integral contour in the complex plane (for anti-aliasing). Be careful of numerical
        errors if it is too small.
    resolution: int
        Even integer for the number of points to use for the fft.
    range_out: tuple
        Tuple of integer for the range to output (excluding the upper bound).
    """

    if (resolution % 2) > 0:
        raise ValueError("The resolution must be even")

    if range_out is None:
        range_out = (0,resolution) #not inclusive of right bound
    elif range_out[1] > resolution:
        raise ValueError("Upper bound of range larger than resolution")

    nb_eval = resolution // 2 + 1
    nodes = np.arange(resolution) / (resolution)
    resolution_out = range_out[1]-range_out[0]

    c = np.exp(np.exp(2 * np.pi * complex(0, 1) * nodes[:nb_eval]) * radius)

    #state variables
    state_vars = Psi.get_initial_state_vars(0.)

    Psi_c = {label:np.zeros((len(t),c.shape[0]),dtype=complex) for label in Psi.Psi0}
    cumulants = {label:np.zeros((len(t),resolution_out)) for label in Psi.Psi0}

    #evaluate characteristic function at nodes
    for j in range(c.shape[0]):
        out = Psi(state_vars,t, transform_dict=transform_dict, c_=c[j]) #apply transformation at beginning
        for label in out:
            Psi_c[label][:,j] = out[label]

    #obtain pdf from characteristic function
    for label in cumulants:
        for ind in range(len(t)):
            spectrum = np.concatenate((Psi_c[label][ind],np.conjugate(Psi_c[label][ind][-2:0:-1])))
            cumulants[label][ind,:] = np.absolute(np.fft.fft(spectrum))[range_out[0]:range_out[1]]
            cumulants[label][ind,:] /= resolution*(radius)**np.arange(range_out[0],range_out[1])
            cumulants[label][ind,:] *= gamma(np.arange(range_out[0],range_out[1])+1)

    #transform output in dataframe format
    results = pd.DataFrame(columns=['label','time','target','cumulant'])
    results['cumulant'] = np.concatenate([np.ravel(cumulants[label]) for label in cumulants])
    results['target'] = np.tile(np.arange(range_out[0],range_out[1]), len(t)*len(cumulants))
    results['time'] = np.tile(np.repeat(t,resolution_out), len(cumulants))
    results['label'] = list(np.repeat(list(cumulants.keys()), resolution_out*len(t), axis=0))
    #works for labels that are tuple/list

    return results

def marginal_cumulants_origin_specific(Psi, t, transform_dict, radius=10**(-4), resolution=4, range_out=None, origin_idx=None):
    """marginal_cumulants.
    Parameters
    ----------
    Psi : PGF object
        Main generating function.
    t : array_like
        Time at which to evaluate the pgf
    transform_dict : dict
        Dictionary of list of transform functions. The key is the time at which the transform is applied. One
        of the transform function should have c_ as argument to transform the variables
    radius: float
        Radius of the integral contour in the complex plane (for anti-aliasing). Be careful of numerical
        errors if it is too small.
    resolution: int
        Even integer for the number of points to use for the fft.
    range_out: tuple
        Tuple of integer for the range to output (excluding the upper bound).
    origin_idx: int or None
        If specified, track only infections originating from this location index. 
        Uses np.ones multiplier with c_ at origin_idx position to allow normal spread 
        while tracking origin-specific cases.
    """
    if (resolution % 2) > 0:
        raise ValueError("The resolution must be even")
    if range_out is None:
        range_out = (0, resolution)  # not inclusive of right bound
    elif range_out[1] > resolution:
        raise ValueError("Upper bound of range larger than resolution")
    
    nb_eval = resolution // 2 + 1
    nodes = np.arange(resolution) / (resolution)
    resolution_out = range_out[1] - range_out[0]
    c = np.exp(np.exp(2 * np.pi * complex(0, 1) * nodes[:nb_eval]) * radius)
    
    # state variables
    state_vars = Psi.get_initial_state_vars(0.)
    Psi_c = {label: np.zeros((len(t), c.shape[0]), dtype=complex) for label in Psi.Psi0}
    cumulants = {label: np.zeros((len(t), resolution_out)) for label in Psi.Psi0}
    
    # evaluate characteristic function at nodes
    for j in range(c.shape[0]):
        if origin_idx is not None:
            # Create origin-specific tracking
            out = Psi(state_vars, t, transform_dict=transform_dict, c_=c[j], origin_idx=origin_idx)
        else:
            # Standard evaluation
            out = Psi(state_vars, t, transform_dict=transform_dict, c_=c[j])
        
        for label in out:
            Psi_c[label][:, j] = out[label]
    
    # obtain pdf from characteristic function
    for label in cumulants:
        for ind in range(len(t)):
            spectrum = np.concatenate((Psi_c[label][ind], np.conjugate(Psi_c[label][ind][-2:0:-1])))
            cumulants[label][ind, :] = np.absolute(np.fft.fft(spectrum))[range_out[0]:range_out[1]]
            cumulants[label][ind, :] /= resolution * (radius) ** np.arange(range_out[0], range_out[1])
            cumulants[label][ind, :] *= gamma(np.arange(range_out[0], range_out[1]) + 1)
    
    # transform output in dataframe format
    results = pd.DataFrame(columns=['label', 'time', 'target', 'cumulant'])
    results['cumulant'] = np.concatenate([np.ravel(cumulants[label]) for label in cumulants])
    results['target'] = np.tile(np.arange(range_out[0], range_out[1]), len(t) * len(cumulants))
    results['time'] = np.tile(np.repeat(t, resolution_out), len(cumulants))
    results['label'] = list(np.repeat(list(cumulants.keys()), resolution_out * len(t), axis=0))
    
    return results


def two_point_cumulants(Psi, t, transform_dict, radius1=10**(-4), radius2=10**(-4),
                        resolution1=4,resolution2=4,range_out1=None, range_out2=None):
    """two_point_cumulants.

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
    radius1 and radius2: float
        Radius of the integral contour in the complex plane (for anti-aliasing). Be careful of numerical
        errors if it is too small.
    resolution1 and resolution2 : int
        Even integer for the number of points to use for the fft for each dimension.
    range_out1 and range_out2: tuple
        Tuple of integer for the range to output (excluding the upper bound).
    """

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


    c1 = np.exp(np.exp(2 * np.pi * complex(0, 1) * nodes1[:nb_eval1]) * radius1)
    c2 = np.exp(np.exp(2 * np.pi * complex(0, 1) * nodes2) * radius2)

    #state variables
    state_vars = Psi.get_initial_state_vars(0.)

    Psi_c = {label:np.zeros((len(t),nb_eval1,c2.shape[0]),dtype=complex) for label in Psi.Psi0}
    cumulants = {label:np.zeros((len(t),resolution_out1,resolution_out2)) for label in Psi.Psi0}

    #evaluate characteristic function at nodes
    for j,k in product(range(c1.shape[0]),range(c2.shape[0])):
        out = Psi(state_vars,t, transform_dict=transform_dict, c1_=c1[j], c2_=c2[k]) #apply transformations
        for label in out:
            Psi_c[label][:,j,k] = out[label]

    #obtain pdf from characteristic function
    for label in cumulants:
        for ind in range(len(t)):
            spectrum = np.zeros((resolution1,resolution2),dtype=complex)
            #recycle calculated elements
            spectrum[0:nb_eval1,:] = Psi_c[label][ind][0:nb_eval1,:]
            spectrum[nb_eval1:,nb_eval2:] = Psi_c[label][ind][nb_eval1-2:0:-1,nb_eval2-2:0:-1].conj()
            spectrum[nb_eval1:,nb_eval2-1] = Psi_c[label][ind][nb_eval1-2:0:-1,nb_eval2-1].conj()
            spectrum[nb_eval1:,0] = Psi_c[label][ind][nb_eval1-2:0:-1,0].conj()
            spectrum[nb_eval1:,1:nb_eval2-1] = Psi_c[label][ind][nb_eval1-2:0:-1,-1:nb_eval2-1:-1].conj()

            temp = np.absolute(np.fft.fft2(spectrum)) / (resolution1*resolution2)
            cumulants[label][ind,:,:] = temp[range_out1[0]:range_out1[1],range_out2[0]:range_out2[1]]
            cumulants[label][ind,:,:] /= np.outer(radius1**np.arange(range_out1[0],range_out1[1]),
                                                  radius2**np.arange(range_out2[0],range_out2[1]))
            cumulants[label][ind,:,:] *= np.outer(gamma(np.arange(range_out1[0],range_out1[1])+1),
                                                  gamma(np.arange(range_out2[0],range_out2[1])+1))

    #transform output in dataframe format
    results = pd.DataFrame(columns=['label','time','target1','target2','cumulant'])
    results['cumulant'] = np.concatenate([np.ravel(cumulants[label]) for label in cumulants])
    results['target2'] = np.tile(np.arange(range_out2[0],range_out2[1]), resolution_out1*len(t)*len(cumulants))
    results['target1'] = np.tile(np.repeat(np.arange(range_out1[0],range_out1[1]),resolution_out2), len(t)*len(cumulants))
    results['time'] = np.tile(np.repeat(t,resolution_out1*resolution_out2), len(cumulants))
    results['label'] = np.repeat(list(cumulants.keys()), resolution_out1*resolution_out2*len(t))

    return results

#==========================
#Marginal distributions
#==========================

def pdf_from_marginal_cumulants_NB(dataframe, time=None):
    """pdf_from_marginal_cumulants_NB.

    We use a negative binomial to do moment matching.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Output from the two_point_cumulants function
    time : int
        time for the distribution.
    """

    if time is None:
        df = dataframe[dataframe['time'] == dataframe['time'].max()].copy() #filter to get max time
    else:
        df = dataframe[dataframe['time'] == time].copy()

    #order 1 and 2 cumulants
    mean = df[df['target'] == 1]['cumulant'].values[0]
    var = df[df['target'] == 2]['cumulant'].values[0]

    #get the parameters
    x0 = mean * mean / (var - mean)
    p = (var - mean) / (var)
    p0 = 1 - p

    def pdf(x):
        logy = loggamma(x0+x) + x0*np.log(p0) - loggamma(x0) + x*np.log(p) - loggamma(x+1)
        return np.exp(logy)

    return pdf



#==========================
#Two-point distributions
#==========================

def pdf_from_two_point_cumulants_NM(dataframe, time=None):
    """pdf_from_two_point_cumulants.

    We use a negative multinomial to do moment matching.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Output from the two_point_cumulants function
    time : int
        time for the distribution.
    """

    if time is None:
        df = dataframe[dataframe['time'] == dataframe['time'].max()].copy() #filter to get max time
    else:
        df = dataframe[dataframe['time'] == time].copy()

    #order 1 and 2 cumulants
    cov = np.zeros((2,2))
    cov[0,0] = df[(df['target1'] == 2) & (df['target2'] == 0)]['cumulant']
    cov[1,1] = df[(df['target1'] == 0) & (df['target2'] == 2)]['cumulant']
    cov[0,1] = df[(df['target1'] == 1) & (df['target2'] == 1)]['cumulant']
    cov[1,0] = cov[0,1]
    mean = np.zeros((2,))
    mean[0] = df[(df['target1'] == 1) & (df['target2'] == 0)]['cumulant']
    mean[1] = df[(df['target1'] == 0) & (df['target2'] == 1)]['cumulant']

    #get the parameters
    cov_det = det(cov)
    mean_sum = np.sum(mean)
    mean_prod = np.prod(mean)
    x0 = mean_sum * mean_prod / (cov_det-mean_prod)
    p = (cov_det-mean_prod) / (cov_det*mean_sum) * mean
    p0 = 1 - np.sum(p)


    def pdf(x1,x2):
        logy = (loggamma(x0+x1+x2) + x0*np.log(p0) - loggamma(x0) + x1*np.log(p[0]) + x2*np.log(p[1])
                - loggamma(x1+1) - loggamma(x2+1))
        return np.exp(logy)

    return pdf


def conditional_cumulants_NM(dataframe, target1_value, time=None):
    """conditional_cumulants_NM.

    We use a negative multinomial to do moment matching.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Output from the two_point_cumulants function
    time : int
        time for the distribution.
    """

    if time is None:
        df = dataframe[dataframe['time'] == dataframe['time'].max()].copy() #filter to get max time
    else:
        df = dataframe[dataframe['time'] == time].copy()

    #order 1 and 2 cumulants
    cov = np.zeros((2,2))
    cov[0,0] = df[(df['target1'] == 2) & (df['target2'] == 0)]['cumulant']
    cov[1,1] = df[(df['target1'] == 0) & (df['target2'] == 2)]['cumulant']
    cov[0,1] = df[(df['target1'] == 1) & (df['target2'] == 1)]['cumulant']
    cov[1,0] = cov[0,1]
    mean = np.zeros((2,))
    mean[0] = df[(df['target1'] == 1) & (df['target2'] == 0)]['cumulant']
    mean[1] = df[(df['target1'] == 0) & (df['target2'] == 1)]['cumulant']

    #get the parameters for the joint
    cov_det = det(cov)
    mean_sum = np.sum(mean)
    mean_prod = np.prod(mean)
    x0 = mean_sum * mean_prod / (cov_det-mean_prod)
    p = (cov_det-mean_prod) / (cov_det*mean_sum) * mean
    p0 = 1 - np.sum(p)

    #get the conditional mean and variance
    x0 = x0 + target1_value
    p0 = p0 + p[0]
    cond_mean = x0*p[1]/p0
    cond_var = x0*p[1]**2/(p0**2) + x0*p[1]/p0

    return cond_mean, cond_var

