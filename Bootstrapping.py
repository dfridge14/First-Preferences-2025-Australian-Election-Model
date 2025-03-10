import pandas as pd
import numpy as np
import os, time
from pathlib import Path
import re
from sklearn.neighbors import BallTree

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

data_year = "2022"
previous_data_year = str(int(data_year)-3)




Omnipresent_parties_df_curr = pd.read_csv(f'{data_year}OmnipresentPartiesByPP.csv', index_col=None)
Omnipresent_parties_df_last = pd.read_csv(f'{previous_data_year}OmnipresentPartiesByPP.csv', index_col=None).drop('pp_id', axis=1)

# Fix up missing Lat/Longs if necessary
div_names_curr = Omnipresent_parties_df_curr['div_nm'].unique().tolist()
div_names_last = Omnipresent_parties_df_last['div_nm'].unique().tolist()
div_names = [d.upper() for d in (set(div_names_curr) | set(div_names_last))]


def remove_division_name(pp_nm, div_list_upper): # written by chatgpt
    pattern = r"\b" + r"\b|\b".join(map(re.escape, div_list_upper)) + r"\b"  # Match full word(s)
    
    # Remove matched division name, replacing it with a single space
    cleaned_pp_nm = re.sub(pattern, " ", pp_nm)
    
    # Ensure single spacing and strip extra spaces
    return " ".join(cleaned_pp_nm.split())


def remove_pp_nm_div_identifier(df, div_names):
    # remove (div_nm)
    df.loc[:,'pp_nm'] = df.loc[:,'pp_nm'].str.replace(r'\[.*?\]|\(.*?\)','',regex=True).str.strip()
    # remove 'DIVNM' PPVC
    df.loc[df['pp_nm'].str.endswith('PPVC'),'pp_nm'] = df.loc[df['pp_nm'].str.endswith('PPVC'),'pp_nm'].apply(lambda x: remove_division_name(x, div_names))

    return df





# make them comparable
Omnipresent_parties_df_curr_adj = remove_pp_nm_div_identifier(Omnipresent_parties_df_curr, div_names)
Omnipresent_parties_df_last_adj = remove_pp_nm_div_identifier(Omnipresent_parties_df_last, div_names)

Redistribution_pairs = pd.read_csv(f"RedistributionPairs{str(int(previous_data_year)+2)}.csv", index_col = None).sort_values(by='new_div')
Redistribution_pairs_new_div_dict = {new_div: group for new_div, group in Redistribution_pairs.groupby('new_div')['old_div'].apply(list).items()}
import pdb;pdb.set_trace()


# 1. Try merge on pp_nm and same div_nm
# 2. For each reciever_div, try add on pp_nm and any of reciever's divisions

Omnipresent_parties_merged = Omnipresent_parties_df_curr_adj.merge(Omnipresent_parties_df_last_adj, on=['pp_nm','div_nm'], how='left', suffixes=(f'_{data_year}', f'_{previous_data_year}'))

# unmatched:
unmatched_pps_full = Omnipresent_parties_merged.loc[(Omnipresent_parties_merged['ALP_2019'].isna()) & (Omnipresent_parties_merged['pp_nm']!='Other'),]
unmatched_pps = unmatched_pps_full[['div_nm','pp_nm']]


df2_redistributed = pd.merge(Omnipresent_parties_df_last_adj, Redistribution_pairs, left_on='div_nm', right_on='old_div')

matches = pd.merge(Omnipresent_parties_df_curr_adj, df2_redistributed.drop('div_nm', axis=1), left_on=['div_nm', 'pp_nm'], right_on=['new_div', 'pp_nm'])


matches = matches.loc[matches['pp_nm']!='Other',]

match_counts = matches.groupby(['new_div', 'pp_nm']).size().reset_index(name='count')


# match new pp_nms via coordinates!!!

PP_coords_curr = pd.read_csv(f'{data_year}_PP_data.csv', index_col=None)[['div_nm','pp_nm','Booth_type','Lat','Long']]
PP_coords_last = pd.read_csv(f'{previous_data_year}_PP_data.csv', index_col=None)[['div_nm','pp_nm','Booth_type','Lat','Long']]

PP_coords_curr = remove_pp_nm_div_identifier(PP_coords_curr, div_names)
PP_coords_last = remove_pp_nm_div_identifier(PP_coords_last, div_names)

unmatched_pps = unmatched_pps.merge(PP_coords_curr[['div_nm','pp_nm','Booth_type']], on = ['div_nm','pp_nm'],how='left') # add Booth_type columns to unmatched_pps

unmatched_pps['to_match'] = 0

import pdb;pdb.set_trace()

# perform matching procedure separately for PBs, PPVCs
merged_df_list = []
for Booth_type in ['PB','PPVC']:
    df1 = unmatched_pps.loc[unmatched_pps['Booth_type'] == Booth_type,].merge(PP_coords_curr.drop('Booth_type', axis=1), on=['div_nm','pp_nm'],how='left')
    df2 = PP_coords_last.loc[~(PP_coords_last['Lat'].isna()) & (PP_coords_last['Booth_type']==Booth_type),] # can ignore all those without coordinates!!! Zero votes, or Other

    df1[['Lat_rad', 'Long_rad']] = np.radians(df1[['Lat', 'Long']])
    df2.loc[:,['Lat_rad', 'Long_rad']] = np.radians(df2[['Lat', 'Long']]).values

    earth_radius = 6371000  

    # Build BallTree on df2 using Haversine metric
    tree = BallTree(df2[['Lat_rad', 'Long_rad']], metric='haversine')

    # Query df1 against df2 (k=1 means find the closest neighbor)
    distances, closest_indices = tree.query(df1[['Lat_rad', 'Long_rad']], k=1)


    # Convert distance from radians to meters
    distances_meters = distances[:, 0] * earth_radius

    df2_closest = df2.iloc[closest_indices.flatten()].copy().reset_index(drop=True).rename(columns={'pp_nm':'pp_nm_old','div_nm':'div_nm_old'})
    df2_closest.loc[:,'distance'] = distances_meters

    import pdb;pdb.set_trace()


    dfs_merged = pd.concat([df1,df2_closest[['div_nm_old','pp_nm_old','distance']]], axis=1)

    dfs_merged['to_match'] = 0

    # find 1. location matches within same div; 2. location matches from redistributed divs

    curr_df_div_pp_nms = PP_coords_curr[['div_nm','pp_nm','Booth_type']]
    curr_df_div_pp_nms_dict = {Booth_type: {div:pp_nms for div, pp_nms in curr_df_div_pp_nms.loc[curr_df_div_pp_nms['Booth_type']==Booth_type,].groupby('div_nm')['pp_nm'].apply(list).items()}  \
                                                                                                                for Booth_type in ['PB','PPVC']}
    
    # closest is already in new_div's divs - consider FAILED - ignore + Only if within 1500m!
    matched_same_dfs = dfs_merged.loc[(dfs_merged['div_nm']==dfs_merged['div_nm_old']) & (dfs_merged['distance']<1500) & ~(dfs_merged.apply(lambda row: row['pp_nm_old'] in curr_df_div_pp_nms_dict[Booth_type].get(row['div_nm'], []), axis=1)),:]

    # For redistirbuted areas, either same names and <1500m distance, or different names and <500 distance! (already in has too small a constituency to be cared about)
    matched_redistributed_dfs = dfs_merged.loc[(((dfs_merged['distance']<1500) & (dfs_merged['pp_nm']==dfs_merged['pp_nm_old'])) | \
                                                    ((dfs_merged['distance']<500) & (dfs_merged['pp_nm']!=dfs_merged['pp_nm_old']))) & \
                                                        (dfs_merged.apply(lambda row: row['div_nm_old'] in Redistribution_pairs_new_div_dict.get(row['div_nm'], []), axis=1)),]


    matched_same_dfs.loc[:,'to_match'] = 1
    matched_redistributed_dfs.loc[:,'to_match'] = 1

    dfs_merged.loc[matched_same_dfs.index] = matched_same_dfs
    dfs_merged.loc[matched_redistributed_dfs.index] = matched_redistributed_dfs


    import pdb;pdb.set_trace()


    merged_df_list.append(dfs_merged[['div_nm','pp_nm','div_nm_old','pp_nm_old','to_match']].copy())

mostly_matched_pps = pd.concat(merged_df_list, ignore_index=True) # make sure order is maintained! Or that index values are enough!


import pdb;pdb.set_trace()

# merge past results onto matched_pps
matched_pps = mostly_matched_pps.loc[mostly_matched_pps['to_match']==1,].merge(Omnipresent_parties_df_last_adj.rename(columns={'div_nm':'div_nm_old','pp_nm':'pp_nm_old'}), on = ['div_nm_old','pp_nm_old'], how='left')
matched_pps.columns = [col if i < len(matched_pps.columns) - 3 else f"{col}_{previous_data_year}" for i, col in enumerate(matched_pps.columns)]
matched_pps.drop(['div_nm_old','pp_nm_old','to_match'], axis=1, inplace=True)


unmatched_pps_mostly_filled = unmatched_pps_full.reset_index().merge(matched_pps, on = ['div_nm','pp_nm'], how='left',suffixes=('_nan','')).iloc[:, list(range(7)) + list(range(10, 13))]
unmatched_pps_mostly_filled = unmatched_pps_mostly_filled.loc[unmatched_pps_mostly_filled.iloc[:,-1].notna(),]






match_mask = (distances_meters <= 100) #& (df1['ID1'] != df2.iloc[indices[:, 0]]['ID1'].values)
matched_df1 = df1[match_mask].reset_index(drop=True)
matched_df2 = df2.iloc[closest_indices[match_mask, 0]].reset_index(drop=True)
result = pd.concat([matched_df1, matched_df2], axis=1)


def haversine(PP_Lat, PP_Long, SA1_Lat, SA1_Long): # written by chatgpt
    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    PP_Lat, PP_Long, SA1_Lat, SA1_Long = map(np.radians, [PP_Lat, PP_Long, SA1_Lat, SA1_Long])

    # Calculate differences in latitude and longitude
    dlat = PP_Lat - SA1_Lat.T  # PP_Lat is column vector, SA1_Lat is row vector -> transpose SA1_Lat
    dlon = PP_Long - SA1_Long.T  # Same for longitude

    # Apply Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(PP_Lat) * np.cos(SA1_Lat.T) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Compute the distances
    distance = R * c  # Distance in kilometers
    return distance



# mostly 1s, occasional 2. Only ~400 disscrepancies, goes a good way
matches['pp_nm'].unique()
Omnipresent_parties_merged.loc[(Omnipresent_parties_merged['ALP_2019'].isna()) & (Omnipresent_parties_merged['pp_nm']!='Other'),'pp_nm'].unique()
Omnipresent_parties_merged.loc[Omnipresent_parties_merged['pp_nm'].isin(set(asd) & set(qwe)),]

import pdb;pdb.set_trace()


# see if last election the pp_nm of old_divs are now the names of pps in new_div
for receiver_div in Redistribution_pairs['new_div'].unique():

    curr_pp_nms = Omnipresent_parties_df_curr.loc[(Omnipresent_parties_df_curr['div_nm']==receiver_div) & (Omnipresent_parties_df_curr['pp_nm']!='Other'),'pp_nm'].unique()

    for giver_div in Redistribution_pairs.loc[Redistribution_pairs['new_div']==receiver_div,'old_div'].unique():

        givers_last_pp_nms = Omnipresent_parties_df_last.loc[(Omnipresent_parties_df_last['div_nm']==giver_div) & (Omnipresent_parties_df_last['pp_nm']!='Other'),'pp_nm'].unique()
        common_pp_nms = set(givers_last_pp_nms) & set(curr_pp_nms)

        import pdb;pdb.set_trace()


# create swings and standardise them
pp_swings = []


divs = Omnipresent_parties_df_curr['div_nm'].unique()
Bootstrapping_dict = {new_div: set(pp_swings['standardised_swing']) for new_div in divs}