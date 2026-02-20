from pgfgleam import *

from scipy.linalg import block_diag
from scipy.sparse import csr_array
import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy import integrate
from scipy import stats
import ast
import re

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'

from typing import List
import itertools

import warnings
warnings.filterwarnings('ignore')
import os

class DataSetup:
    '''
    Class to handle data loading and preprocessing for the pgfgleam project. This includes:
    - Loading country flow data and population data
    - Normalizing flow data by population
    - Creating mappings for country names to codes
    '''
    def __init__(self, flight_matrix_path, population_path, country_codes_path):
        self.flight_matrix_path = flight_matrix_path
        self.population_path = population_path
        self.country_codes_path = country_codes_path

    def load_and_process_data(self):

        # Load the country codes mapping
        country_codes = pd.read_csv(self.country_codes_path)
        country_pop = pd.read_csv(self.population_path)
        df = pd.read_csv(self.flight_matrix_path, header=None)

        # Name mapping since some are a bit inconsistent across datasets
        # Create name mapping for some countries
        name_mapping = {
            'W. Sahara': 'Western Sahara',
            'United States of America': 'United States',
            'Dem. Rep. Congo': 'Democratic Republic of the Congo',
            'Dominican Rep.': 'Dominican Republic',
            'Russia': 'Russian Federation',
            'Falkland Is.': 'Falkland Islands',
            'Fr. S. Antarctic Lands': 'French Southern Territories',
            'Timor-Leste': 'East Timor',
            'Puerto Rico': 'Puerto Rico',  
            'United States Virgin Islands': 'British Virgin Islands',
            'Côte d\'Ivoire': 'Ivory Coast (Cote d\'Ivoire)',
            'Guinea-Bissau': 'Guinea Bissau',
            'Central African Rep.': 'Central African Republic',
            'Eq. Guinea': 'Equatorial Guinea',
            'eSwatini': 'Swaziland',
            'North Korea': 'Korea, Democratic People\'s Republic of',
            'Bhutan': 'Bangladesh',
            'Somaliland': 'Somalia',
            'Solomon Is.': 'Papua New Guinea',
            'N. Cyprus': 'Cyprus',
            'Bosnia and Herz.': 'Bosnia and Herzegovina',
            'North Macedonia': 'Macedonia',
            'Kosovo': 'Kosovo',
            'S. Sudan': 'South Sudan',
            'Czechia': 'Czech Republic',
            'Brunei': 'Brunei Darussalam',
            'Taiwan': 'Chinese Taipei',
            'Hong Kong': 'Hong Kong (SAR), China',
            'French Southern Terrirtories': 'French Guiana'
        }
        
        code_dict = dict(zip(country_codes['country'].str.strip(), country_codes['country_code'].str.strip()))
        pop_dict = dict(zip(country_pop['Location'].str.strip(), country_pop['Population']))

        # Extract country names from data
        target_countries = df.iloc[0, 1:].tolist()  # First row (excluding first column)
        source_countries = df.iloc[1:, 0].tolist()  # First column (excluding first row)

        # Convert country names to codes
        target_codes = [code_dict.get(country.strip(), country.strip()) for country in target_countries]
        source_codes = [code_dict.get(country.strip(), country.strip()) for country in source_countries]

        congo_sources = [country for country in source_countries if 'Congo' in country]
        congo_rows = country_codes[country_codes['country'].str.contains('Congo', case=False, na=False)]

        # Create a more robust country code mapping function
        def create_robust_code_dict(country_codes_df):
            """Create a more robust country code dictionary with various name formats"""
            robust_dict = {}
            
            for _, row in country_codes_df.iterrows():
                country = row['country'].strip()
                code = row['country_code'].strip()
                
                # Add the original mapping
                robust_dict[country] = code
                
                # Add alternative names for problematic countries
                if 'Congo' in country and 'Democratic' in country:
                    robust_dict['Democratic Republic of the Congo'] = code
                    robust_dict['Congo, Democratic Republic of'] = code
                    robust_dict['Congo, Dem. Rep.'] = code
                    robust_dict['DR Congo'] = code
                    robust_dict['DRC'] = code
                    
            return robust_dict

        # Create the robust dictionary
        robust_code_dict = create_robust_code_dict(country_codes)

        # Test the mapping for various Congo formats
        test_names = [
            'Democratic Republic of the Congo',
            'Congo, Democratic Republic of',
            'Congo, Dem. Rep.',
            'DR Congo',
            'DRC'
        ]

        # Convert the data to a NumPy array for flow values
        flow_matrix = df.iloc[1:, 1:].to_numpy().astype(float)

        # For each value in the matrix, divide by 31 to get daily average
        flow_matrix = flow_matrix / 31

        normalized_flow_matrix = np.zeros_like(flow_matrix)

        for i, country in enumerate(source_countries):
            pop = pop_dict.get(country, None)
            if pop:
                normalized_flow_matrix[i, :] = flow_matrix[i, :] / pop
            else:
                normalized_flow_matrix[i, :] = flow_matrix[i, :]
                
        # Put popuation data into a DataFrame
        pop_df = pd.DataFrame({
            'Country': source_countries,
            'Population': [pop_dict.get(country, 0) for country in source_countries]
        })

        gbr_index = source_countries.index('United Kingdom')
        sentinel_loc = gbr_index

        return {
            'flow_matrix': flow_matrix,
            'normalized_flow_matrix': normalized_flow_matrix,
            'source_countries': source_countries,
            'target_countries': target_countries,
            'source_codes': source_codes,
            'target_codes': target_codes,
            'population': pop_df,
            'name_mapping': name_mapping}
    
class FigureFormat:
    '''
    Class to handle formatting of figures for the pgfgleam project. This includes:
    - Setting consistent styles for histograms, boxplots, and scatter plots
    - Adding titles, labels, and legends in a standardized way
    '''
    def __init__(self):
        pass

    def set_figure_format(self):
        # Set consistent font styling
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif', 
                                    'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 
                                    'Palatino', 'Charter', 'Bookman']
        plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern math font - same as LaTeX
        plt.rcParams['font.weight'] = 'normal'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 112
        plt.rcParams['legend.title_fontsize'] = 13
        plt.rcParams['figure.titlesize'] = 18

class ContinentalRegions:
    '''
    For figure2.ipynb
    '''
    def __init__(self):
        pass
    def define_continental_regions(self):
        continental_regions = {
        'Africa': [
            'Egypt', 'Libya', 'Tunisia', 'Algeria', 'Morocco', 'Sudan', 'Western Sahara',
            'Nigeria', 'Ghana', 'Ivory Coast (Cote d\'Ivoire)', 'Mali', 'Burkina Faso', 'Niger', 
            'Senegal', 'Guinea', 'Benin', 'Togo', 'Sierra Leone', 'Liberia', 
            'Mauritania', 'Gambia', 'Guinea Bissau', 'Cape Verde',
            'Ethiopia', 'Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Burundi', 
            'Somalia', 'Djibouti', 'Eritrea', 'South Sudan', 'Madagascar', 
            'Mauritius', 'Seychelles', 'Comoros', 'Mayotte', 'Reunion',
            'Democratic Republic of the Congo', 'Angola', 'Cameroon', 'Chad', 
            'Central African Republic', 'Republic of the Congo', 'Gabon', 
            'Equatorial Guinea', 'Sao Tome and Principe', 'South Africa', 
            'Zambia', 'Zimbabwe', 'Botswana', 'Namibia', 'Lesotho', 'Swaziland', 'Malawi', 'Mozambique', 'Congo', 'Saint Helena'
        ],
        'Asia': [
            'China', 'Japan', 'South Korea', 'North Korea', 'Mongolia', 
            'Chinese Taipei', 'Hong Kong (SAR), China', 'Macau (SAR), China', 'Russian Federation', 'Kazakhstan', 'Uzbekistan', 
            'Turkmenistan', 'Tajikistan', 'Kyrgyzstan',
            'Indonesia', 'Philippines', 'Vietnam', 'Thailand', 'Myanmar', 'Malaysia', 
            'Cambodia', 'Laos', 'Singapore', 'Brunei Darussalam', 'East Timor',
            'India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Bhutan', 
            'Afghanistan', 'Maldives', 'Iran', 'Turkey', 'Saudi Arabia', 'Iraq', 
            'Yemen', 'Syria', 'Jordan', 'Israel', 'Lebanon', 'Palestinian Territory', 
            'Kuwait', 'Qatar', 'Bahrain', 'United Arab Emirates', 'Oman', 'Cyprus', 
            'Georgia', 'Armenia', 'Azerbaijan', 'Turkmenistan'
        ],
        'Europe': [
            'Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Netherlands', 
            'Belgium', 'Switzerland', 'Austria', 'Portugal', 'Ireland', 'Luxembourg', 
            'Monaco', 'Liechtenstein', 'Andorra', 'San Marino', 'Vatican City',
            'Sweden', 'Norway', 'Denmark', 'Finland', 'Iceland', 'Estonia', 
            'Latvia', 'Lithuania', 'Greenland', 'Faroe Islands',
            'Poland', 'Ukraine', 'Czech Republic', 'Slovakia', 'Hungary', 'Romania', 
            'Bulgaria', 'Belarus', 'Moldova',
            'Greece', 'Malta', 'Croatia', 'Slovenia', 'Bosnia and Herzegovina', 
            'Montenegro', 'Albania', 'Macedonia', 'Serbia', 'Kosovo'
        ],
        'North and\nCentral America': [
            'United States', 'Canada', 'Mexico', 'Guatemala', 'Belize', 'El Salvador', 
            'Honduras', 'Nicaragua', 'Costa Rica', 'Panama', 'Greenland', 
            'Saint Pierre and Miquelon',
            'Cuba', 'Jamaica', 'Haiti', 'Dominican Republic', 'Puerto Rico', 
            'Trinidad and Tobago', 'Bahamas', 'Barbados', 'Saint Lucia', 
            'Grenada and South Grenadines', 'Saint Vincent and Grenadines', 'Antigua and Barbuda', 
            'Dominica', 'Saint Kitts and Nevis', 'Aruba', 'Curacao', 
            'Sint Maarten', 'Cayman Islands', 'British Virgin Islands', 
            'Turks and Caicos Islands', 'Anguilla', 'Montserrat', 'Guadeloupe', 
            'Martinique', 'Saint Barthelemy', 'Saint Martin', 'Bermuda', 'Bonaire, Saint Eustatius & Saba'
        ],
        'South\nAmerica': [
            'Brazil', 'Argentina', 'Colombia', 'Peru', 'Venezuela', 'Chile', 
            'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay', 'Guyana', 'Suriname', 
            'French Guiana', 'Falkland Islands'
        ],
        'Oceania': [
            'Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 
            'Vanuatu', 'New Caledonia', 'French Polynesia', 'Western Samoa', 'Kiribati', 
            'Tonga', 'Micronesia', 'Palau', 'Marshall Islands', 'Tuvalu', 'Nauru', 
            'Cook Islands', 'Niue', 'American Samoa', 'Guam', 'Northern Mariana Islands', 
            'Norfolk Island', 'Christmas Island', 'Cocos Islands', 'Wallis and Futuna Islands', 'Cocos (Keeling) Islands'
        ]
        }
        
        return continental_regions