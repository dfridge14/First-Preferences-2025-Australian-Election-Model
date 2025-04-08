import pandas as pd
import numpy as np
from itertools import product
import os,time
from datetime import datetime
from pathlib import Path

from scipy.special import logit


# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):

    print("\a")  # Rings the system bell
    os.system('echo -e "\\a"')  # Extra bell command for reliability
    os.system('tput bel')  # This forces the terminal to beep

    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler











base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

party_map = {
    'ALP': 'ALP',
    'CLR': 'ALP',
    'LP': 'COAL',
    'NP': 'COAL',
    'LNP': 'COAL',
    'LNQ': 'COAL',
    'CLP': 'COAL',
    'GRN': 'GRN',
    'GVIC': 'GRN'
}


# first, get the state results for each election


election_state_result_list = []
for election_year in [2004,2007,2010,2013,2016,2019,2022]:
    #for state in ['NSW','VIC','QLD','WA','SA','TAS','NT','ACT']:
    StateFirstPrefs = pd.read_csv(f"{election_year}HouseFirstPrefsByStateByParty.csv", skiprows=1, index_col=None).rename(columns={'StateAb':'State'})
    StateFirstPrefs = StateFirstPrefs.loc[StateFirstPrefs['PartyAb'].isin(['ALP','CLR','LP','NP','CLP','LNP','LNQ','GRN','GVIC']),['State','PartyAb','TotalPercentage']]
    StateFirstPrefs['PartyAb'] = StateFirstPrefs['PartyAb'].replace(party_map)

    grouped = StateFirstPrefs.groupby(['State', 'PartyAb'], as_index=False)['TotalPercentage'].sum()
    pivoted = grouped.pivot(index='State', columns='PartyAb', values='TotalPercentage').fillna(0).reset_index()
    pivoted.loc[:,'Election_year'] = election_year
    election_state_result_list.append(pivoted)

Federal_State_1996_01 = pd.read_csv("OldFederalStateResults.csv", index_col=None)
Nat_polls_2007_22 = pd.read_csv("NationalElectionResults.csv", index_col=None).drop('OTH', axis=1).rename(columns={'Election':'Election_year'})
Nat_polls_2007_22.loc[:,'State'] = 'NAT'
Nat_polls_2007_22.iloc[:,:3] *= 100 # same format as percentage

National_State_df = pd.concat(election_state_result_list + [Federal_State_1996_01,Nat_polls_2007_22], ignore_index=True)
National_State_df.loc[:,['COAL','ALP','GRN']] /= 100 # convert to proportions
National_State_df.loc[:,'OTH'] = 1-National_State_df.loc[:,['COAL','ALP','GRN']].sum(axis=1)
National_State_df = National_State_df[['Election_year','State','COAL','ALP','GRN','OTH']].sort_values(by=['Election_year','State'])

#import pdb;pdb.set_trace()

# convert to ALR
ref_col = 'COAL'
National_State_ALR_df = National_State_df.copy().drop(ref_col, axis=1)
National_State_ALR_df.iloc[:,2:] = np.log(National_State_df.iloc[:,2:].drop(columns=[ref_col]).div(National_State_df.iloc[:,2:][ref_col], axis=0))

National_ALR_df = National_State_ALR_df.loc[National_State_ALR_df['State']=='NAT',] 
National_ALR_df.loc[:,'Election_year'] = National_ALR_df['Election_year']
State_ALR_df =  National_State_ALR_df.loc[National_State_ALR_df['State']!='NAT',] 
State_NAT_merged = pd.merge(State_ALR_df, National_ALR_df, on='Election_year', how='left', suffixes=('','_Nat'))

# Take 1 difference between elections - equivalent to calculating change in swing

alr_cols = [col for col in ['COAL','ALP','GRN','OTH'] if col != ref_col]
# get Nat - State differences
for col in alr_cols:
    State_NAT_merged[col + '_rel_to_Nat'] = State_NAT_merged[col] - State_NAT_merged[col + '_Nat']

ALR_deviation_cols = [col + '_rel_to_Nat' for col in alr_cols]
State_NAT_merged = State_NAT_merged.sort_values(by=['State', 'Election_year'])

differenced_df = State_NAT_merged.groupby('State')[['Election_year'] + ALR_deviation_cols].apply(lambda group: group.set_index('Election_year').diff().dropna()).reset_index()

differenced_df_cleaned = differenced_df.loc[~((np.abs(differenced_df['GRN_rel_to_Nat'])>0.5) & (differenced_df['Election_year']<2004)),]
differenced_df_cleaned = differenced_df_cleaned.loc[~((np.abs(differenced_df_cleaned['OTH_rel_to_Nat'])>0.9) ),] # & (differenced_df_cleaned['State']!= 'SA') - remove due to no longer such a state effect in play!



from pingouin import multivariate_normality

result = multivariate_normality(differenced_df_cleaned.iloc[:,2:], alpha=0.05)
print(result)

import pdb;pdb.set_trace()

# I get this correlation structure! Small correlations!
#                ALP_rel_to_Nat  GRN_rel_to_Nat  OTH_rel_to_Nat
#ALP_rel_to_Nat        1.000000        0.080057        0.119370
#GRN_rel_to_Nat        0.080057        1.000000        0.052068
#OTH_rel_to_Nat        0.119370        0.052068        1.000000



# Example data loading
election_year = '2019'

df = pd.read_csv(f"StatePolls{election_year}.csv", parse_dates=['Date'])



# Filter relevant columns
parties = ['COAL', 'ALP', 'GRN', 'ON', 'OTH']

# Convert vote shares to ALR space using OTH as the reference
def to_alr(row):
    # Numerator parties are all except the reference (OTH)
    alr_values = np.log(row[parties[:-1]] / row['OTH'])
    return pd.Series(alr_values, index=[f'ALR_{p}' for p in parties[:-1]])

df_alr = df[parties].apply(to_alr, axis=1)
df = pd.concat([df, df_alr], axis=1)

# Split national and state polls
nat_df = df[df['Scope'] == 'NAT']
state_df = df[df['Scope'] != 'NAT']

# Ensure date is in datetime format
nat_df['Date'] = pd.to_datetime(nat_df['Date'])
state_df['Date'] = pd.to_datetime(state_df['Date'])

# Merge each state poll with the nearest national poll by date
def match_nearest_nat(poll):
    date_diff = abs(nat_df['Date'] - poll['Date'])
    nearest_nat = nat_df.loc[date_diff.idxmin()]
    # Difference in ALR space
    alr_diff = poll[[f'ALR_{p}' for p in parties[:-1]]] - nearest_nat[[f'ALR_{p}' for p in parties[:-1]]]
    return alr_diff

# Apply to all state polls
alr_diffs = state_df.apply(match_nearest_nat, axis=1)

# Merge results back for convenience
state_df_with_diffs = pd.concat([state_df.reset_index(drop=True), alr_diffs.rename(columns=lambda x: x + '_rel_to_nat')], axis=1)

# Now you have the ALR swing of each state poll relative to national in ALR_*_rel_to_nat columns
print(state_df_with_diffs.head())
