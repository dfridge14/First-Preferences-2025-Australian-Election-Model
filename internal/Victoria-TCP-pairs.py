import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import re
import time
import copy
import json

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
from scipy.optimize import minimize
from collections import defaultdict

import geopandas as gpd


import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler


os.chdir(Path.home() / 'Australian Election')

prev_federals = {2010:2010,2014:2013,2018:2016,2022:2022,2026:2025}


election_year = 2026


party_df = pd.read_csv(f'Victorian Election/VIC-{election_year}/VIC-{election_year}-Parties.csv')
Ideology_map = party_df.set_index('PartyAb')['Ideo_category'].to_dict()
Grand_party_df = pd.read_csv('Grand_Party_Category_df_2004_2022.csv')
Ideology_map_federal = Grand_party_df.set_index('PartyAb')['Ideo_Category'].to_dict()

if election_year == 2026:
    next_contesting_parties = pd.read_csv('Victorian Election/VIC-2026/VIC-2026-Candidates.csv').groupby('div_nm')['PartyAb'].apply(lambda x: list(dict.fromkeys(x))).to_dict()
else:
    def format_specific_party(row):
        """
        Helper function to convert 'IND' into 'IND_First_Last'.
        Leaves standard parties (e.g., 'ALP', 'ON') untouched.
        """
        ab = str(row['PartyAb']).strip()
        if ab.startswith('IND'):
            # Title case and replace spaces with underscores (e.g. "Ali Cupper" -> "Ali_Cupper")
            first = str(row['first_name']).strip().title().replace(' ', '_')
            last = str(row['surname']).strip().title().replace(' ', '_')
            return f"IND_{first}_{last}"
        return ab

    LA_Candidates_next = pd.read_csv(f'Victorian Election/VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
    LA_Candidates_next['PartyAb_Specific'] = LA_Candidates_next.apply(format_specific_party, axis=1)
    next_contesting_parties = LA_Candidates_next.groupby('div_nm')['PartyAb_Specific'].agg(list).to_dict()

GEODATA_FOLDER = Path.home() / 'Australian Election/Victorian Election/Geodata'

if election_year == 2026:
    federal_file = GEODATA_FOLDER / 'Federal-VIC-2024-shapefiles/E_VIC24_region.shp'
    state_file = GEODATA_FOLDER / 'State-VIC-2021-shapefiles/VIC-2021-New districts and regions.shp'

# Load and reproject to VicGrid for accurate square-meter calculations
fed_gdf = gpd.read_file(federal_file).to_crs(epsg=3111)
state_gdf = gpd.read_file(state_file).to_crs(epsg=3111)

# Store the total area of each original STATE electorate (instead of federal)
state_gdf['state_area'] = state_gdf.geometry.area

# Intersect the two maps 
overlap = gpd.overlay(state_gdf, fed_gdf, how='intersection')

# Calculate the area of each new overlapping piece and correspondign weights (proportion of the STATE electorate  made up by this federal piece)
overlap['overlap_area'] = overlap.geometry.area
overlap['weight'] = overlap['overlap_area'] / overlap['state_area']

# Clean up to a simple mapping dictionary/dataframe: 
# Fed_Name, State_Name, Weight
overlap = overlap.rename(columns={'DISTRICT':'State_Dist', 'Elect_div': 'Federal_Div'})
overlap['State_Dist'] = overlap['State_Dist'].str.title()
overlap_weights = overlap[['Federal_Div', 'State_Dist', 'weight']]
overlap_weights['weight'] = overlap_weights['weight'] / overlap_weights.groupby('State_Dist')['weight'].transform('sum') # normalise so state districts are perfectly split


# Obtain 2CP estimates
with open(Path.home() / "Australian Election/TCP_pair_category_dict_for_2028.pkl", "rb") as f:
    TCP_transferred_to_first = pickle.load(f)
TCP_COMBINATION_INDEX = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6,  \
                                ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14}

# 1. Filter, clean, and flatten the federal dictionary concisely
vic_fed_divs = overlap_weights['Federal_Div'].unique()

df_list = [
    df.loc[:, ~df.columns.duplicated()]
      .assign(Federal_Div=fed)
      .rename_axis('TCP_Regime')
      .reset_index()
    for fed, df in TCP_transferred_to_first.items() if fed in vic_fed_divs
]
fed_all = pd.concat(df_list, ignore_index=True)
party_cols = fed_all.columns.difference(['TCP_Regime', 'Federal_Div']).tolist()

# 2. Merge geographical weights
merged = fed_all.merge(overlap_weights, on='Federal_Div', how='inner')

# 3. Calculate Normalized Weighted Averages
vals_weighted = merged[party_cols].multiply(merged['weight'], axis=0)
weights_masked = merged[party_cols].notna().multiply(merged['weight'], axis=0)

group_keys = [merged['State_Dist'], merged['TCP_Regime']]
num = vals_weighted.groupby(group_keys).sum(min_count=1)
den = weights_masked.groupby(group_keys).sum(min_count=1)

state_aggregated_raw = num / den

# 4. Rebuild into the final state dictionary safely
state_base_dict = {}
for state, group in state_aggregated_raw.groupby(level='State_Dist'):
    # Extract the state's dataframe, locally drop 100% NaN parties, and fill remaining gaps
    df_state = group.droplevel('State_Dist').dropna(axis=1, how='all').fillna(0)
    state_base_dict[state] = df_state

# Verify
print(state_base_dict['Albert Park'].head())


### Impute ideological average and apply to next_contesting_parties set for each state electorate

TCP_state_dict = {}

for state_dist, df in state_base_dict.items():
    df_imputed = df.copy()
    current_cols = df_imputed.columns.tolist()
    
    # ---------------------------------------------------------
    # STAGE 3: Ideological Averages
    # ---------------------------------------------------------
    # Group existing columns by their federal ideology
    left_cols = [col for col in current_cols if Ideology_map_federal.get(col, 'Centre') == 'Left']
    right_cols = [col for col in current_cols if Ideology_map_federal.get(col, 'Centre') == 'Right']
    centre_cols = [col for col in current_cols if Ideology_map_federal.get(col, 'Centre') == 'Centre']
    
    # Helper to calculate row mean while ignoring 0.0 (so missing parties don't drag down the average)
    def get_ideo_mean(cols):
        if not cols:
            return pd.Series(np.nan, index=df_imputed.index)
        return df_imputed[cols].replace(0, np.nan).mean(axis=1)

    df_imputed['Left_Avg'] = get_ideo_mean(left_cols)
    df_imputed['Right_Avg'] = get_ideo_mean(right_cols)
    df_imputed['Centre_Avg'] = get_ideo_mean(centre_cols)
        
    # If Centre_Avg is NaN (no Centre parties exist), use the average of all nonzero parties in that row
    all_nonzero_mean = df_imputed[current_cols].replace(0, np.nan).mean(axis=1)
    df_imputed['Centre_Avg'] = df_imputed['Centre_Avg'].fillna(all_nonzero_mean)
    
    # ---------------------------------------------------------
    # STAGE 4: Impute Missing State Parties and Filter
    # ---------------------------------------------------------
    # Get the target parties for this specific state electorate
    target_parties = next_contesting_parties.get(state_dist, current_cols)
    
    for party in target_parties:
        if party not in df_imputed.columns:
            # Look up ideology using the state ideology map, defaulting to 'Centre'
            party_ideo = Ideology_map.get(party, 'Centre')
            
            # Borrow the corresponding ideological average
            df_imputed[party] = df_imputed[f'{party_ideo}_Avg']
            
    # Fallback: if an imputed column is NaN (e.g., no Left federal parties ran at all), 
    # fill with the overall row average, then 0 as a last resort.
    for party in target_parties:
        if df_imputed[party].isna().any():
            df_imputed[party] = df_imputed[party].fillna(all_nonzero_mean).fillna(0)

    # Keep ONLY the next contesting parties (automatically drops the _Avg columns and unused federal parties)
    TCP_state_dict[state_dist] = df_imputed[target_parties]

# Verify the final output
print(TCP_state_dict['Albert Park'].head())

if not os.path.exists(f"Victorian Election/VIC-{election_year}/TCP_state_dict_for_{election_year}.pkl"):
    with open(f"Victorian Election/VIC-{election_year}/TCP_state_dict_for_{election_year}.pkl", "wb") as f:
        pickle.dump(TCP_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

import pdb; pdb.set_trace()


