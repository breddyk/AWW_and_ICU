#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions - slightly adapted from original pgfgleam package to incorporate turnaround time for testing

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

import numpy as np
import pandas as pd
from collections.abc import Iterable
from itertools import product
from datetime import date
from scipy.optimize import fsolve

def calculate_growth_rate(R0, infectious_period, latency_period, nb_latent_states=1, nb_infectious_states=1):
    """See the following paper:
        https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020174
    """

    func = lambda r: R0 - r*(1 + r*latency_period/nb_latent_states)**nb_latent_states/\
            (infectious_period**(-1)*(1 - (1 + r*infectious_period/nb_infectious_states)**(-nb_infectious_states)))

    return fsolve(func, 0.1)[0]

def calculate_doubling_time(*args,**kwargs):
    return np.log(2)/calculate_growth_rate(*args,**kwargs)

def get_index_default(location_id, age_id, nb_location, nb_age_cat):
    """get_index_default provide a category index for metapopulation model with age structure.

    If location_id or age_id is None, all ids are used by default.

    Parameters
    ----------
    location_id : int or iterable
        Integer label(s) for a location.
    age_id : int or iterable
        Integer label(s) for age category.
    nb_age_cat" int
        Number of age categories
    """
    if age_id and location_id:
        idx = get_index(location_id, age_id, nb_age_cat)
    elif age_id and location_id is None:
        idx = get_index(np.arange(nb_location), age_id, nb_age_cat)
    elif location_id and age_id is None:
        idx = get_index(location_id, np.arange(nb_age_cat), nb_age_cat)
    else:
        idx = get_index(np.arange(nb_location), np.arange(nb_age_cat), nb_age_cat)
    return idx

def get_index(location_id, age_id, nb_age_cat):
    """get_index provide a category index for metapopulation model with age structure.

    Parameters
    ----------
    location_id : int or iterable
        Integer label(s) for a location.
    age_id : int or iterable
        Integer label(s) for age category.
    nb_age_cat" int
        Number of age categories
    """
    if isinstance(location_id,Iterable) and isinstance(age_id,Iterable):
        idx = [nb_age_cat*i + j for i,j in product(location_id,age_id)]
    elif isinstance(location_id,Iterable) and not isinstance(age_id,Iterable):
        idx = [nb_age_cat*i + age_id for i in location_id]
    elif not isinstance(location_id,Iterable) and isinstance(age_id,Iterable):
        idx = [nb_age_cat*location_id + j for j in age_id]
    else:
        idx = nb_age_cat*location_id+age_id
    return idx


def append_stats(distribution, quantiles, groupbycols, xcol='time'):
    #calculate cumulative
    dist = distribution.copy()
    dist['cum'] = dist.groupby(groupbycols)['probability'].cumsum()

    #get mean
    mean_df = dist.copy().drop(columns=['cum'])
    mean_df['mean'] = dist[xcol]*dist['probability']
    mean_df = mean_df.groupby(by=groupbycols).sum().reset_index().drop(columns=[xcol,'probability'])

    #get quantiles
    df_list = []
    for q in quantiles:
        df = dist[dist['cum'] >= q].copy()
        idx = df.groupby(by=groupbycols)['cum'].idxmin()
        df_ = dist.loc[idx].drop(columns=['probability','cum']).sort_values(
            by=groupbycols+[xcol]).copy()
        df_['quantile'] = q
        df_list.append(df_)

    if len(df_list) > 0:
        quantile_df = pd.concat(df_list)
        quantile_df.rename(columns={xcol:'value'},inplace=True)
        quantile_df = quantile_df.pivot_table(values=['value'],index=groupbycols,
                                      columns=['quantile'])['value'].reset_index()
        quantile_df.columns = quantile_df.columns.astype(str)
        stats_df = pd.merge(mean_df,quantile_df, on=groupbycols)
    else:
        stats_df = mean_df

    #merge stats
    dist = dist.merge(stats_df, on=groupbycols)

    return dist


#alias for backward compatibility
def get_time_to_detection(*args,**kwargs):
    return get_detection_time(*args,**kwargs)

def get_detection_time(distribution, D, quantiles=[0.5], groupbycols=['label'], turnaround_time=0):
    """get_time_to_detection calculates the posterior distribution on the time to get D detections.
    Uses the distribution of # of detected cases
    Also calculates the mean detection time and desired ~quantiles, the first time such that the
    cumulative probability is bigger than a quantile q value, e.g., q = 0.5 for the median.

    Parameters
    ----------
    distribution : Pandas DataFrame
        Contains the result of a marginal distribution for a given quantity (detected)
    D : integer
        Fixed value for the number of detections
    quantiles: list of float
        q values for the desired quantiles to compute
    groupbycols: list of str
        Columns to groupby for the calculations.
    turnaround_time: integer or float
        Time delay between taking a test and receiving the result
    """
    # Make a copy of the distribution and shift the time by the turnaround_time
    shifted_distribution = distribution.copy()
    shifted_distribution['time'] = shifted_distribution['time'] + turnaround_time
    
    cum = shifted_distribution[shifted_distribution['target'] < D].groupby(by=groupbycols+['time']).sum().reset_index().drop(columns=['target'])
    detection_time = cum  # probability that target < D
    detection_time['probability_'] = cum.groupby(by=groupbycols)['probability'].shift()
    detection_time['probability'] = (detection_time['probability_'] - detection_time['probability']).abs()
    detection_time = detection_time.drop(columns={'probability_'}).dropna()
    
    # normalize
    norm = detection_time.groupby(by=groupbycols).sum().reset_index().drop(columns=['time']).rename(columns={'probability':'norm'})
    detection_time = detection_time.merge(norm, on=groupbycols)
    detection_time['probability'] /= detection_time['norm']
    detection_time = detection_time.drop(columns=['norm'])
    
    # append stats
    detection_time = append_stats(detection_time, quantiles, groupbycols)
    
    # Add rows with zero probability for times from 0 to turnaround_time - 1 AFTER all calculations
    if turnaround_time > 0:
        import pandas as pd
        
        # Get unique groups from the result
        unique_groups = detection_time[groupbycols].drop_duplicates()
        
        # Create time points from 0 to turnaround_time (not including turnaround_time)
        time_points = []
        if isinstance(turnaround_time, int):
            time_points = list(range(turnaround_time + 1))
        else:
            # For float turnaround_time, create integer values up to but not including turnaround_time
            time_points = list(range(int(turnaround_time + 1)))
        
        # Create zeros rows
        zero_rows = []
        for _, group in unique_groups.iterrows():
            group_dict = group.to_dict()
            
            # Get stats for this group to append to each new row
            stats_row = detection_time[
                detection_time[groupbycols].isin(group_dict.values()).all(axis=1)
            ].iloc[0].to_dict()
            
            # Keep only stats columns and remove 'time' and 'probability'
            stats_dict = {k: v for k, v in stats_row.items() 
                         if k not in ['time', 'probability', 'cum'] + groupbycols}
            
            for t in time_points:
                new_row = {**group_dict, **stats_dict, 'time': t, 'probability': 0.0, 'cum': 0.0}
                zero_rows.append(new_row)
        
        # Create DataFrame with zero rows and concatenate
        if zero_rows:
            zero_df = pd.DataFrame(zero_rows)
            detection_time = pd.concat([detection_time, zero_df], ignore_index=True)
            detection_time = detection_time.sort_values(by=groupbycols + ['time']).reset_index(drop=True)
    
    return detection_time
