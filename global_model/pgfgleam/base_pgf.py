#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probability generating function of a branching process on metapopulation network for early epidemic forecast.
This module contains the base class to abstract base class used to implement more complex epidemic models.

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.sparse import load_npz, csr_array, csc_array, spmatrix #base class for sparse matrices
from .utils import get_index

def load_sparse(file):
    """By default, load as sparse matrix instead of sparse array"""
    return csr_array(load_npz(file))

#==============================
#Base class for epidemic models
#==============================
class BasePGF(ABC):
    """BasePGF. Abstract base class to define PGFs encoding the state of an agent-based system.
    It uses a generalized formulation which allows more flexibility for the development
    of more complex epidemic models.
    """

    def __init__(self, umat, mmat, infection='poisson', nb_microsteps=1, cumulant=False,
                 umap=None, mmap=None, cmat1=None, cmat2=None, **kwargs):
        """__init__.

        If umat (mmat) is a list of ndarrays or matrices and umap (mmap) is not None, then
        it is assumed that the contact matrix (mobility matrix) is time-varying.

        Parameters
        ----------
        umat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Contact matrix.
        mmat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Mobility matrix.
        infection : str
            Type of offspring distribution for infectious agents.
        nb_microsteps : int
            Number of microsteps for the contagion part.
        cumulant : bool
            If true, turn the generating function into the cumulant generating function.
        umap : function
            Function mapping time to index for umat
        mmap : function
            Function mapping time to index for mmat
        cmat1: ndarray or sparse matrix
            Commuting matrix part 1. Associated to susceptible commuting and getting infected there.
        cmat2: ndarray or sparse matrix
            Commuting matrix part 2. Associated to infectious commuting and infecting individuals there.
        """
        super().__init__()
        self.umap = umap
        self.mmap = mmap
        self.Psi0 = dict() #dictionary of initial PGF functions
        self.cumulant = cumulant
        self.nb_microsteps = nb_microsteps
        self.cmat1 = cmat1
        self.cmat2 = cmat2
        if (cmat1 is None) and (cmat2 is not None) or (cmat2 is None) and (cmat1 is not None):
            raise ValueError("cmat1 & cmat2 must be defined or none of them")

        #check if time-varying contact matrix
        if isinstance(umat, list) and umap is not None:
            self.umat_list = umat
            self.umean_list = [u @ np.ones(u.shape[0]) for u in self.umat_list]
            try:
                #default/dummy umat values
                self.umat = self.umat_list[0]
            except:
                raise ValueError("Empty umat list")
        elif isinstance(umat, (np.ndarray,spmatrix,csr_array,csc_array)) and umap is None:
            self.umat = umat
            self.umat_list = None
        else:
            raise ValueError("Inconsistent umat input")
        #otherwise
        self.umean = self.umat @ np.ones(self.umat.shape[0]) #mean nb of contact per time step
        self.nb_types = self.umat.shape[0] #number of agent types

        #check if time-varying mobility matrix
        if isinstance(mmat, list) and mmap is not None:
            self.mmat_list = mmat
            self.mvec_list = [1 - m @ np.ones(m.shape[0]) for m in self.mmat_list]
            try:
                #default/dummy mmat values
                self.mmat = self.mmat_list[0]
            except:
                raise ValueError("Empty mmat list")
        elif isinstance(mmat, (np.ndarray,spmatrix,csr_array,csc_array)) and mmap is None:
            self.mmat = mmat
            self.mmat_list = None
        else:
            raise ValueError("Inconsistent mmat input")
        self.mvec = 1 - self.mmat @ np.ones(self.mmat.shape[0]) #probability to not move

        #verify consistency
        if self.mmat.shape[0] != self.nb_types:
            raise ValueError("Inconsistent shape for umat and mmat")

        #fix the infection pgf
        if infection == 'poisson':
            self.G = self.G_poisson
        elif infection == 'nb':
            self.G = self.G_nb
            if 'k' in kwargs:
                self.k = kwargs['k']
            else:
                self.k = 1 #default value for nb
        else:
            raise ValueError("Infection mispecified")

    @classmethod
    def init_from_config(cls,config,**kwargs):
        #essential parameters to fix R0
        lev = config['parameters']['leading_eigenvalue_source']
        if 'R0' in kwargs:
            R0 = kwargs['R0']
        else:
            R0 = config['parameters']['R0']
        infectious_period = config['parameters']['infectious_period']

        #get the contact matrices
        umat_loader = config['structure']['umat']
        if  isinstance(umat_loader, list):
            umat_ = [load_sparse(mat['file']) for mat in umat_loader]
            umat = [u*R0/(infectious_period*lev) for u in umat_] #rescale contact matrix to get specific R0
            def umap(t):
                val = 0
                for mat in umat_loader:
                    if mat['end_time'] and t > mat['end_time']:
                        val += 1
                return val
        elif isinstance(umat_loader, str):
            umat_ = load_sparse(umat_loader)
            umat = umat_*R0/(infectious_period*lev) #rescale contact matrix to get specific R0
            umap = None
        else:
            raise RuntimeError("Bad PGF configuration file: umat")

        #get the mobility matrices
        mmat_loader = config['structure']['mmat']
        if  isinstance(mmat_loader, list):
            mmat = [load_sparse(mat['file']) for mat in mmat_loader]
            def mmap(t):
                val = 0
                for mat in mmat_loader:
                    if mat['end_time'] and t > mat['end_time']:
                        val += 1
                return val
        elif isinstance(mmat_loader, str):
            mmat = load_sparse(mmat_loader)
            mmap = None
        else:
            raise RuntimeError("Bad PGF configuration file: mmat")

        #get the commuting matrix
        if 'cmat1' in config['structure'] and config['structure']['cmat1'] is not None:
            cmat_loader = config['structure']['cmat1']
            cmat1 = load_sparse(cmat_loader)
        else:
            cmat1 = None

        if 'cmat2' in config['structure'] and config['structure']['cmat2'] is not None:
            cmat_loader = config['structure']['cmat2']
            cmat2 = load_sparse(cmat_loader)
        else:
            cmat2 = None

        if ('cumulant' not in config['parameters']) and ('cumulant' not in kwargs):
            warnings.warn("By default, this defines a PGF not a CGF")

        #other extractions for the derived classes
        other = cls.special_init(config)

        #instantiate PGF
        obj = cls(umat, mmat, cmat1=cmat1, cmat2=cmat2, umap=umap, mmap=mmap, **other, **config['parameters'], **kwargs)

        #get the age structure and format the dataframe
        age_struc = pd.read_csv(config['structure']['age'], sep=' ', header=None)
        age_struc = age_struc.rename(columns={0:'location_id'})
        nb_age_cat = config['structure']['nb_age_cat']
        age_struc['population'] = age_struc[np.arange(1,nb_age_cat+1)].sum(axis=1)
        for col in range(1,nb_age_cat+1):
            age_struc[col] /= age_struc['population']

        #define initial conditions if any
        #we assume that age category is chosen at random (proportional to fraction) unless specified
        if 'seeds' in config and isinstance(config['seeds'],list):
            for seed in config['seeds']:
                if 'location_id' in seed and seed['location_id']:
                    if 'age_id' in seed and seed['age_id'] and 'weight' in seed and seed['weight']:
                        idx = get_index(seed['location_id'],seed['age_id'],nb_age_cat)
                        weight = seed['weight']
                    else:
                        idx = get_index(seed['location_id'],np.arange(nb_age_cat),nb_age_cat)
                        weight = age_struc.loc[age_struc['location_id'] == seed['location_id'],np.arange(1,nb_age_cat+1)].values[0]
                    obj.add_initial_conditions(idx,weight,label=seed['label'],**seed['state'])
        return obj

    def add_custom_initial_conditions(self, Psi0, label=None):
        """add_custome_initial_condition.

        This method allows adding custom initial conditions with an optional label. If no label is provided,
        it assigns a default label based on the current length of self.Psi0 dictionary.

        Parameters
        ----------
        Psi0 : Initial stage PGF
            Initial state probability generating function
        """
        if label is None:
            label = len(self.Psi0)
        self.Psi0[label] = Psi0

    def B(self, current_var, next_var, probability):
        """Generic PGF for a bernoulli random variable, to flip from one state to another"""
        return probability * next_var + (1 - probability) * current_var

    def G_poisson(self, x): #poisson infection each day (or subpart of the day)
        if self.cmat1 is None:
            out = np.exp(self.umat @ (x-1) / self.nb_microsteps)
        else:
            out = np.exp(self.cmat2 @ (self.umat @ (self.cmat1 @ (x-1))) / self.nb_microsteps)
        return out

    def G_nb(self, x): #negative binomial infection each day (or subpart of the day)
        if self.cmat1 is None:
            out = (1 + (self.umean - self.umat @ x) / (self.k*self.nb_microsteps))**(-self.k)
        else:
            out = (1 + (self.cmat2 @ self.umean - self.cmat2 @ (self.umat @ (self.cmat1 @ x))) / (self.k*self.nb_microsteps))**(-self.k)
        return out

    #----------------------------------------
    #The following methods need to be defined
    #----------------------------------------
    @abstractmethod
    def get_initial_state_vars(self, value):
        pass

    @abstractmethod
    def add_initial_conditions(self, idx, weight, label=None, **state):
        pass

    @abstractmethod
    def reaction_phase(self, state_vars):
        pass

    @abstractmethod
    def mobility_phase(self, state_vars):
        pass

    #----------------------------------
    #The following methods are optional
    #----------------------------------
    @classmethod
    def special_init(cls, config):
        """Perform other extractions for the derived classes"""
        return dict()
    #----------------------------------

    def __call__(self, state_vars, t, transform_dict=dict(), **transform_kwargs):
        """__call__ evaluates the PGF or CGF.

        Transforms are used to modify the PGF/CGF at arbitrary times.

        NOTE: if either the contact or the mobility matrix is time-varying, only the evaluation for the largest
        time max(t) is correct. We therefore recommend to use a singleton as input to avoid potential errors.

        Parameters
        ----------
        state_vars : dictionary
            Variables of the probability generating function with the name of what they track as key (e.g.,
            latent, infectious)
        t : set or array_like
            Times at which to evaluate the pgf
        """
        if not self.Psi0:
            raise RuntimeError("Initial conditions must be specified")

        if (self.umap is not None or self.mmap is not None) and len(t) > 1:
            #we use relative time to evaluate intermediate time points, therefore:
            warnings.warn("Time variation detected: Only the last time point should be trusted")

        #if we want to output the result of the cumulant generating function (CGF)
        if self.cumulant:
            state_vars_ = dict()
            for key,var in state_vars.items():
                if isinstance(var, (spmatrix,csr_array,csc_array)):
                    temp = var.copy()
                    np.exp(temp.data, out=temp.data)
                    state_vars_[key] = temp
                else:
                    state_vars_[key] = np.exp(var)
        else:
            #still copy the input state variables
            state_vars_ = {key:var.copy() for key,var in state_vars.items()}


        #prepare output
        out = {label:[] for label in self.Psi0}

        tmax = max(t)
        tset = set(t)
        for t_ in range(tmax,0,-1):
            #apply transformation to variables
            if t_ in transform_dict:
                for transform in transform_dict[t_]:
                    transform(state_vars_,**transform_kwargs)
            #evaluate at initial conditions
            if (tmax-t_) in tset:
                for label,Psi0 in self.Psi0.items():
                    val = Psi0(state_vars_)
                    if self.cumulant:
                        val = np.log(val)
                    out[label].append(val)

            #time-varying contact matrix
            if self.umap is not None:
                ind = self.umap(t_)
                self.umat = self.umat_list[ind]
                self.umean = self.umean_list[ind]

            #time-varying mobility matrix
            if self.mmap is not None:
                ind = self.mmap(t_)
                self.mmat = self.mmat_list[ind]
                self.mvec = self.mvec_list[ind]


            #===============================================================
            #Actions of the agent during a day
            #note: what happens after in this loop happens before in reality
            #===============================================================

            #Reaction phase
            #--------------
            #A number of microsteps larger than 1 can be used to try to mimic "continuous-time" dynamics
            for _ in range(self.nb_microsteps):
                self.reaction_phase(state_vars_)

            #Mobility phase
            #--------------
            self.mobility_phase(state_vars_)

        #apply transformation to variables at t = 0 (if any)
        if 0 in transform_dict:
            for transform in transform_dict[0]:
                transform(state_vars_,**transform_kwargs)

        for label,Psi0 in self.Psi0.items():
            val = Psi0(state_vars_)
            if self.cumulant:
                val = np.log(val)
            out[label].append(val)

        return out
