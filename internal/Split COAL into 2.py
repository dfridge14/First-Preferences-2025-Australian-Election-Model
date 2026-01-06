import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gammaln  # Correct import for the gamma function
from scipy.stats import multivariate_t, multivariate_normal, dirichlet



# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler

base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)





name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}






COAL_ratio_df_list = []

for data_year in ['2004','2007','2010','2013','2016','2019','2022']:

    Incumbent_col_name = 'SittingMemberFl' if data_year == '2004' else 'HistoricElected'
    is_incumbent_symbol = '#' if data_year == '2004' else 'Y'

    # 1. Collect only divs which have both LP and NP in 


    # get state-to-div dict, adjusting for name changes
    div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
    div_to_state_dict = {name_changes_year_dict[data_year].get(div, div): div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}

    # create wide format eliminaation_order_dict
    DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)
    DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)

    df_simplified = DOP_By_Division.loc[(DOP_By_Division['CountNumber']==0) & (DOP_By_Division['CalculationType'] == 'Preference Percent'),]
    

    COAL_double_parties = ['NP', 'LP']
    df_filtered = df_simplified[df_simplified['PartyAb'].isin(COAL_double_parties)]

    # Then, keep only the div_nm groups that contain *both* NP and LP
    divs_with_both = (
        df_filtered.groupby('div_nm')['PartyAb']
        .nunique()
        .reset_index()
        .query('PartyAb == 2')['div_nm']
    )

    # ordered subsection of df where there are both NP and LP!
    COAL_double_divs_df = df_simplified[df_simplified['div_nm'].isin(divs_with_both) & df_simplified['PartyAb'].isin(COAL_double_parties)].sort_values(by=['div_nm','PartyAb'])
    COAL_double_divs_df = COAL_double_divs_df[['div_nm','StateAb',Incumbent_col_name,'PartyAb','CalculationValue']]
    #pivoted = COAL_double_divs_df.pivot(index=['div_nm'], columns='SittingMemberFl', values='PartyAb')
    #pivoted['NP_ratio'] = pivoted['NP'] / (pivoted['NP'] + pivoted['LP'])

    totals = COAL_double_divs_df.groupby('div_nm')['CalculationValue'].transform('sum')

    # Compute NP ratio only for NP rows (optional, others will be NaN)
    COAL_double_divs_df['NP_ratio'] = COAL_double_divs_df.apply(
        lambda row: row['CalculationValue'] / totals[row.name]
        if row['PartyAb'] == 'NP' else None,
        axis=1
    )

    # get the incumbent party
    incumbent_party = (
    COAL_double_divs_df[COAL_double_divs_df[Incumbent_col_name] == is_incumbent_symbol].groupby('div_nm')['PartyAb'].first().reindex(COAL_double_divs_df['div_nm'].unique(), fill_value='').reset_index(name='Incumbent')
)
    # merge with incumbents and clean up df
    COAL_double_divs_df = COAL_double_divs_df.merge(incumbent_party, on='div_nm', how='left')
    COAL_double_divs_df = COAL_double_divs_df.loc[~(COAL_double_divs_df['NP_ratio'].isna()),].rename(columns = {Incumbent_col_name:'election_year','StateAb':'State'})
    COAL_double_divs_df.loc[:,'election_year'] = data_year
    COAL_double_divs_df.drop(['PartyAb','CalculationValue'], axis=1, inplace = True)

    COAL_ratio_df_list.append(COAL_double_divs_df[['div_nm','State','election_year','Incumbent','NP_ratio']])


COAL_NP_ratio_df = pd.concat(COAL_ratio_df_list, ignore_index=False)

COAL_NP_ratio_df_2025 = pd.read_csv("COAL_NP_ratio_df_2025.csv", index_col = None).fillna('')

ONLY_2025 = 1

if ONLY_2025:
    COAL_NP_ratio_df = pd.concat([COAL_NP_ratio_df,COAL_NP_ratio_df_2025], ignore_index=True)

import pdb;pdb.set_trace()


#COAL_NP_ratio_df.to_csv('COAL_NP_ratio_df.csv', index = False)
    
#import pdb;pdb.set_trace()


def adjust_for_incumbent(row, compare_row):
    INCUMBENT_NP_RATIO = 0.64
    if row['Incumbent'] != compare_row['Incumbent']:
        if compare_row['Incumbent'] == 'LP':
            return compare_row['NP_ratio'] * INCUMBENT_NP_RATIO # increase NP_ratio - row doesn't have Inc
        elif not compare_row['Incumbent']:
            return compare_row['NP_ratio'] * INCUMBENT_NP_RATIO # reduce NP ratio - row has Inc
    return compare_row['NP_ratio']


from tqdm import tqdm  # For progress bar if desired
tqdm.pandas()  # optional

df = COAL_NP_ratio_df # to avoid progress_apply() difficulties - remove later!

def compute_ratios(row):
    this_year = row['election_year']
    this_div = row['div_nm']
    this_state = row['State']

    # 1. Same div, last election
    prev_year = str(int(this_year) - 3) 
    mask1 = (df['div_nm'] == this_div) & (df['election_year'] == prev_year)
    comp1 = df[mask1]
    ratio1 = comp1.apply(lambda r: adjust_for_incumbent(row, r), axis=1).mean()

    # 2. Same state, last election
    mask2 = (df['State'] == this_state) & (df['election_year'] == prev_year)
    comp2 = df[mask2]
    ratio2 = comp2.apply(lambda r: adjust_for_incumbent(row, r), axis=1).mean()

    # 3. Same div, all years except this
    mask3 = (df['div_nm'] == this_div) & (df['election_year'] != this_year)
    comp3 = df[mask3]
    ratio3 = comp3.apply(lambda r: adjust_for_incumbent(row, r), axis=1).mean()

    # 4. All divisions, last election
    mask4 = (df['election_year'] == prev_year)
    comp4 = df[mask4]
    ratio4 = comp4.apply(lambda r: adjust_for_incumbent(row, r), axis=1).mean()

    comp5 = df[df['election_year'] != this_year]
    ratio5 = comp5.apply(lambda r: adjust_for_incumbent(row, r), axis=1).mean()

    return pd.Series([ratio1, ratio2, ratio3, ratio4, ratio5], index=[
        'ratio_same_div_last_year',
        'ratio_state_last_year',
        'ratio_same_div_other_years',
        'ratio_national_last_year',
        'ratio_national_all_other_years'
    ])

# Apply the function
COAL_NP_ratio_df[['ratio_same_div_last_year',
    'ratio_state_last_year',
    'ratio_same_div_other_years',
    'ratio_national_last_year',
    'ratio_national_all_other_years']] = COAL_NP_ratio_df.progress_apply(compute_ratios, axis=1)

COAL_NP_ratio_df_prediction = COAL_NP_ratio_df.loc[COAL_NP_ratio_df['election_year']!= '2004',]

weights = {
    'ratio_same_div_last_year': 1.0,
    'ratio_state_last_year': 0.5,
    'ratio_same_div_other_years': 0.5,
    'ratio_national_last_year': 0.25,
    'ratio_national_all_other_years': 0.1
}

ratio_cols = list(weights.keys())

def weighted_estimate(row):
    # Get values and corresponding weights (excluding NaNs)
    values = []
    wts = []
    for col in ratio_cols:
        val = row[col]
        if pd.notna(val):
            values.append(val)
            wts.append(weights[col])

    if not values:
        return np.nan  # All values were NaN

    # Normalize weights
    norm_wts = np.array(wts) / np.sum(wts)
    return np.dot(values, norm_wts)

# Apply to DataFrame
COAL_NP_ratio_df_prediction.loc[:,'final_estimate'] = COAL_NP_ratio_df_prediction.apply(weighted_estimate, axis=1)



COAL_NP_ratio_df_prediction = COAL_NP_ratio_df_prediction[['div_nm','State','election_year','final_estimate']]

import pdb;pdb.set_trace()

if not ONLY_2025:
    COAL_NP_ratio_df_prediction.to_csv(f'NP_ratio_estimated_df.csv', index = False)

else:
    COAL_NP_ratio_df_prediction = COAL_NP_ratio_df_prediction.loc[COAL_NP_ratio_df_prediction['election_year'] == 2025,]
    COAL_NP_ratio_df_prediction.to_csv(f'NP_ratio_estimated_df_2025.csv', index = False)



import pdb;pdb.set_trace()
