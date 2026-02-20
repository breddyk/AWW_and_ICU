#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probability generating function of a branching process on metapopulation network for early epidemic forecast.
This module contains a class derived from BasePGF specific for the design of a sentinel surveillance system.

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

from .base_pgf import *
from scipy.sparse import csr_array

class SentinelPGF(BasePGF):
    """SentinelPGF. PGF encoding the state of an agent-based system designed to track individuals who might
    might be detected by a sentinel surveillance system at airports."""

    def __init__(self, umat, mmat, smat, latency_period, infectious_period, post_infectious_period,
                 nb_infectious_states=2, nb_post_infectious_states=2,  infection='poisson',
                 nb_microsteps=1, detection_probability=None, cumulant=False, umap=None, mmap=None, **kwargs):
        """__init__.

        If umat (mmat) is a list of ndarrays or matrices and umap (mmap) is not None, then
        it is assumed that the contact matrix (mobility matrix) is time-varying.

        Parameters
        ----------
        umat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Contact matrix.
        mmat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Mobility matrix.
        smat : ndarray or sparse matrix
            Sentinel matrix specifying the detection on each origin-destination.
        latency_period : float
            Mean time spent in the latent state [days].
        infectious_period : float
            Mean time spent in the infectious state(s) [days].
        post_infectious_period : float
            Mean time spent in the post-infectious post_infectious state(s) [days].
        nb_infectious_states: int
            Number of infectious states
        nb_post_infectious_states: int
            Number of post_infectious states
        infection : str
            Type of offspring distribution for infectious agents.
        detection_probability: float or ndarray of float
            If not None, multiply smat by this factor
        cumulant : bool
            If true, turn the generating function into the cumulant generating function.
        umap : function
            Function mapping time to index for umat
        mmap : function
            Function mapping time to index for mmat
        """
        super().__init__(umat, mmat, infection, nb_microsteps, cumulant, umap, mmap,**kwargs)
        self.nb_infectious_states = nb_infectious_states
        self.nb_post_infectious_states = nb_post_infectious_states

        #get a sparse matrix that is the union of all possible mmat (all possible origin-destination)
        if self.mmat_list:
            union = sum(self.mmat_list)
        else:
            union = self.mmat.copy()
        nz_idx = union.nonzero()
        union_one = csr_array((np.ones(len(nz_idx[0])),nz_idx),shape=union.shape)

        #sentinel network format; only keep nonzero element of mmat if sparser
        self.smat = smat.copy() * union_one
        if detection_probability:
            self.smat[self.smat.nonzero()] *= detection_probability

        #get utility matrix for sentinel network
        nz_idx = self.smat.nonzero()
        self.smat_one = csr_array((np.ones(len(nz_idx[0])),nz_idx),shape=self.smat.shape)

        #rescale the periods on the scale of microsteps (and for each infectious/post_infectious state)
        self.latency_period = latency_period*nb_microsteps
        self.infectious_period = infectious_period*nb_microsteps/nb_infectious_states
        self.post_infectious_period = post_infectious_period*nb_microsteps/nb_post_infectious_states

    @classmethod
    def special_init(cls, config):
        """Perform other extractions for the SentinelPGF class"""
        other = dict()
        if 'smat' in config['structure']:
            other['smat'] = load_sparse(config['structure']['smat'])
        return other

    def get_initial_state_vars(self, value):
        vec = np.zeros(self.nb_types, dtype=complex)+value
        state_vars = {'latent': vec.copy(),
                      'cumulative latent': vec.copy(),
                      'cumulative importation': vec.copy()}
        for j in range(self.nb_infectious_states):
            state_vars[f'infectious {j+1}'] = vec.copy()
        for j in range(self.nb_post_infectious_states):
            state_vars[f'post_infectious {j+1}'] = vec.copy()

        nz_idx = self.smat.nonzero() #index of nonzero elements for detection
        nb_nz = len(nz_idx[0])
        state_vars['cumulative detection'] = csr_array((np.zeros(nb_nz)+value, (nz_idx[0],nz_idx[1])),
                                                        shape=self.smat.shape, dtype=complex)
        return state_vars

    def reaction_phase(self, state_vars):
        latent = state_vars['latent']
        cumulative_latent = state_vars['cumulative latent']
        infectious_list = [state_vars[f'infectious {j+1}'] for j in range(self.nb_infectious_states)]

        #new infections from infectious ; infectious advance state
        infectious_list_temp = []
        for j in range(self.nb_infectious_states):
            if j == (self.nb_infectious_states - 1):
                next_var = state_vars['post_infectious 1']
            else:
                next_var = infectious_list[j+1]
            infectious_list_temp.append(
                self.G(latent*cumulative_latent)*self.B(infectious_list[j],next_var,1./self.infectious_period))

        #latents become infectious
        state_vars['latent'] = self.B(latent, infectious_list[0], 1./self.latency_period)

        #update infectious variables
        for j in range(self.nb_infectious_states):
            state_vars[f'infectious {j+1}'] = infectious_list_temp[j]

        #post_infectious advance state
        for j in range(self.nb_post_infectious_states):
            if j == (self.nb_post_infectious_states - 1):
                next_var = 1
            else:
                next_var = state_vars[f'post_infectious {j+2}']
            state_vars[f'post_infectious {j+1}'] = self.B(state_vars[f'post_infectious {j+1}'], next_var,
                                                     1./self.post_infectious_period)

    def mobility_phase(self, state_vars):
        latent = state_vars['latent']
        cumulative_importation = state_vars['cumulative importation']

        #latents move
        state_vars['latent'] = self.mmat @ (cumulative_importation*latent) + self.mvec*latent

        #infectious move
        mmat_ = self.mmat - self.smat_one * self.mmat + self.mmat * \
                (self.smat_one - self.smat + self.smat * state_vars['cumulative detection'])
        # keep up to date with mmat a version of self.mmat - self.smat_one * self.mmat
        # and the last self.mmat could be replaced by a version that intersects with smat

        for j in range(self.nb_infectious_states):
            infectious = state_vars[f'infectious {j+1}']
            state_vars[f'infectious {j+1}'] = mmat_ @ (cumulative_importation*infectious) + self.mvec*infectious

        #post_infectious move
        for j in range(self.nb_post_infectious_states):
            post_infectious = state_vars[f'post_infectious {j+1}']
            state_vars[f'post_infectious {j+1}'] = mmat_ @ post_infectious + self.mvec*post_infectious


    def add_initial_conditions(self, idx, weight=None, nb_infectious=0, nb_latent=0, label=None, **kwargs):
        """add_initial_condition.

        Parameters
        ----------
        idx : int or array_like
            indices for the categories, e.g., if we want to distribute the probability across age and
            locations.
        weight : array_like
            probability associated to each category if idx is array_like. Uniform by default. Does nothing if
            idx is an integer.
        nb_infectious : int or array_like
            nb_infectious for each category
        nb_latent : int or array_like
            nb_latent for each category
        """
        if label is None:
            label = len(self.Psi0)
        if isinstance(idx, (np.ndarray, list)):
            if weight is None or len(weight) != len(idx) or not np.isclose(1.,sum(weight)):
                raise ValueError("weight ill-defined")
            else:
                self.Psi0[label] = lambda state_vars: \
                        np.sum(weight*state_vars["latent"][idx]*state_vars["cumulative latent"][idx])**nb_latent*\
                        np.sum(weight*state_vars["infectious 1"][idx]*state_vars["cumulative latent"][idx])**nb_infectious
        else:
            self.Psi0[label] = lambda state_vars: \
                    (state_vars["latent"][idx]*state_vars["cumulative latent"][idx])**nb_latent*\
                    (state_vars["infectious 1"][idx]*state_vars["cumulative latent"][idx])**nb_infectious
        #NOTE: we assume initial infectious were "latent" at some point in the past

