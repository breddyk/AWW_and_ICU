#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probability generating function of a branching process on metapopulation network for early epidemic forecast.
This module contains a class derived from BasePGF.

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

from .base_pgf import *

class StdPGF(BasePGF):
    """StdPGF. Basic PGF encoding the state of an agent-based system."""

    def __init__(self, umat, mmat, latency_period, infectious_period, infection='poisson',
                 nb_microsteps=1, cumulant=False, umap=None, mmap=None, infectious_travel=False,
                 **kwargs):
        """__init__.

        If umat (mmat) is a list of ndarrays or matrices and umap (mmap) is not None, then
        it is assumed that the contact matrix (mobility matrix) is time-varying.

        Parameters
        ----------
        umat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Contact matrix.
        mmat : ndarray or sparse matrix or list of ndarray or sparse matrix
            Mobility matrix.
        latency_period : float
            Mean time spent in the latent state [days].
        infectious_period : float
            Mean time spent in the infectious state [days].
        infection :
            Type of offspring distribution for infectious agents.
        cumulant : bool
            If true, turn the generating function into the cumulant generating function.
        umap : function
            Function mapping time to index for umat
        mmap : function
            Function mapping time to index for mmat
        """
        super().__init__(umat, mmat, infection, nb_microsteps, cumulant, umap, mmap,**kwargs)
        #rescale the periods on the scale of microsteps
        self.latency_period = latency_period*nb_microsteps
        self.infectious_period = infectious_period*nb_microsteps
        self.infectious_travel = infectious_travel

    def get_initial_state_vars(self, value):
        vec = np.zeros(self.nb_types, dtype=complex)+value
        state_vars = {'latent': vec.copy(),
                      'infectious': vec.copy(),
                      'cumulative latent': vec.copy(),
                      'cumulative infectious': vec.copy(),
                      'cumulative importation': vec.copy()}

        return state_vars

    def reaction_phase(self, state_vars):
        latent = state_vars['latent']
        cumulative_infectious = state_vars['cumulative infectious']
        cumulative_latent = state_vars['cumulative latent']
        infectious = state_vars['infectious']

        #new infections from infectious ; infectious advance state
        infectious_temp = self.G(latent*cumulative_latent)*self.B(infectious,1,1./self.infectious_period)

        #latents become infectious
        state_vars['latent'] = self.B(latent, infectious*cumulative_infectious, 1./self.latency_period)

        #update infectious variables
        state_vars['infectious'] = infectious_temp


    def mobility_phase(self, state_vars):
        latent = state_vars['latent']
        cumulative_importation = state_vars['cumulative importation']

        #latents move
        state_vars['latent'] = self.mmat @ (cumulative_importation*latent) + self.mvec*latent

        #infectious move
        if self.infectious_travel:
                infectious = state_vars['infectious']
                state_vars['infectious'] = self.mmat @ (cumulative_importation*infectious) + self.mvec*infectious


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
            if len(weight) != len(idx) or not np.isclose(1.,sum(weight)):
                raise ValueError("weight ill-defined")
            else:
                self.Psi0[label] = lambda state_vars:\
                        np.sum(weight*state_vars["latent"][idx]*state_vars["cumulative latent"][idx])**nb_latent*\
                        np.sum(weight*state_vars["infectious"][idx]*state_vars["cumulative infectious"][idx]*state_vars["cumulative latent"][idx])**nb_infectious
        else:
            self.Psi0[label] = lambda state_vars: \
                    (state_vars["latent"][idx]*state_vars["cumulative latent"][idx])**nb_latent*\
                    (state_vars["infectious"][idx]*state_vars["cumulative infectious"][idx]*state_vars["cumulative latent"][idx])**nb_infectious
        #NOTE: we assume initial infectious were "latent" at some point in the past
