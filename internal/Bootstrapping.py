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

print(sys.executable)




base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

data_year = "2022"
previous_data_year = str(int(data_year)-3)






Omnipresent_parties_df_curr = pd.read_csv(f'{data_year}OmnipresentPartiesByPP.csv', index_col=None).drop('INFORMAL', axis = 1)
Omnipresent_parties_df_last = pd.read_csv(f'{previous_data_year}OmnipresentPartiesByPP.csv', index_col=None).drop(['pp_id','INFORMAL'], axis=1)

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
unmatched_pps_full = Omnipresent_parties_merged.loc[(Omnipresent_parties_merged.iloc[:,-1].isna()) & (Omnipresent_parties_merged['pp_nm']!='Other'),]
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

    #import pdb;pdb.set_trace()


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


    #import pdb;pdb.set_trace()


    merged_df_list.append(dfs_merged[['div_nm','pp_nm','div_nm_old','pp_nm_old','to_match']].copy())

mostly_matched_pps = pd.concat(merged_df_list, ignore_index=True) # make sure order is maintained! Or that index values are enough!


#import pdb;pdb.set_trace()

# merge past results onto matched_pps
matched_pps = mostly_matched_pps.loc[mostly_matched_pps['to_match']==1,].merge(Omnipresent_parties_df_last_adj.rename(columns={'div_nm':'div_nm_old','pp_nm':'pp_nm_old'}), on = ['div_nm_old','pp_nm_old'], how='left')
matched_pps.columns = [col if i < len(matched_pps.columns) - 3 else f"{col}_{previous_data_year}" for i, col in enumerate(matched_pps.columns)]
matched_pps.drop(['div_nm_old','pp_nm_old','to_match'], axis=1, inplace=True)


unmatched_pps_mostly_filled = unmatched_pps_full.reset_index().merge(matched_pps, on = ['div_nm','pp_nm'], how='left',suffixes=('_nan','')).iloc[:, list(range(7)) + list(range(10, 13))]
unmatched_pps_mostly_filled = unmatched_pps_mostly_filled.loc[unmatched_pps_mostly_filled.iloc[:,-1].notna(),].set_index('index')

Omnipresent_parties_merged.loc[unmatched_pps_mostly_filled.index] = unmatched_pps_mostly_filled


Omnipresent_parties_culled = Omnipresent_parties_merged[Omnipresent_parties_merged.iloc[:,-1].notna()]

Omnipresent_parties_culled = Omnipresent_parties_culled.loc[Omnipresent_parties_culled['pp_nm']!='Other',]

# Others - in redistributed states, take some from each other's by proportions!
Others_old = Omnipresent_parties_df_last.loc[Omnipresent_parties_df_last['pp_nm']=='Other']
Others_new = Omnipresent_parties_df_curr.loc[Omnipresent_parties_df_curr['pp_nm']=='Other']

Redistribution_proportions = pd.read_csv(f"Correspondence_CED_{str(int(data_year)-1)}_{str(int(data_year)-4)}_Reversed.csv", index_col=None)

Others_old_merged = Others_old.merge(Redistribution_proportions, left_on='div_nm', right_on='old_div').sort_values(by='new_div')
Others_old_merged.iloc[:,2:5] = Others_old_merged.iloc[:,2:5].multiply(Others_old_merged['proportion'], axis=0).round(0).astype(int)
Post_redistribution_Others_old = Others_old_merged.groupby('new_div', as_index=False)[Others_old_merged.columns[2:5]].sum().rename(columns={'new_div':'div_nm'})
Post_redistribution_Others_old.loc[:,'pp_nm'] = 'Other'

# get full form of Others new and old
Omnipresent_parties_Others = Others_new.merge(Post_redistribution_Others_old, on=['pp_nm','div_nm'], how='left', suffixes=(f'_{data_year}', f'_{previous_data_year}'))

Omnipresent_parties_full = pd.concat([Omnipresent_parties_culled,Omnipresent_parties_Others], ignore_index=True)

Omnipresent_parties_full = Omnipresent_parties_full.loc[~(Omnipresent_parties_full.iloc[:,6:9].sum(axis=1)<100),]
Omnipresent_parties_full = Omnipresent_parties_full.loc[~(Omnipresent_parties_full.iloc[:,3:6].sum(axis=1)<100),]

# remove those that have vote difference of over 100


def get_swings_from_merged_df(merged_df):
    data_year_cols = merged_df.columns[-6:-3]
    previous_year_cols = merged_df.columns[-3:]
    merged_df[previous_year_cols] = merged_df[previous_year_cols].div(merged_df[previous_year_cols].sum(axis=1), axis=0)
    merged_df[data_year_cols] = merged_df[data_year_cols].div(merged_df[data_year_cols].sum(axis=1), axis=0)

    swing_df = merged_df.copy()
    swing_df[data_year_cols] = (swing_df[data_year_cols].values - swing_df[previous_year_cols].values)*100

    swing_df.rename(columns = {f'COAL_{data_year}':'COAL_swing', f'ALP_{data_year}':'ALP_swing'}, inplace=True)
    swing_df = swing_df[['div_nm','COAL_swing','ALP_swing']].sort_values(by='div_nm')

    return swing_df


Omnipresent_parties_swings = get_swings_from_merged_df(Omnipresent_parties_full)
# rewrite both data sets as percentages - MAKE SURE FUNCTION DOES EXACTLY THIS:
data_year_cols = Omnipresent_parties_full.columns[-6:-3]
previous_year_cols = Omnipresent_parties_full.columns[-3:]
Omnipresent_parties_full[previous_year_cols] = Omnipresent_parties_full[previous_year_cols].div(Omnipresent_parties_full[previous_year_cols].sum(axis=1), axis=0)
Omnipresent_parties_full[data_year_cols] = Omnipresent_parties_full[data_year_cols].div(Omnipresent_parties_full[data_year_cols].sum(axis=1), axis=0)

Omnipresent_parties_swings = Omnipresent_parties_full.copy()
Omnipresent_parties_swings[data_year_cols] = (Omnipresent_parties_full[data_year_cols].values - Omnipresent_parties_full[previous_year_cols].values)*100

Omnipresent_parties_swings.rename(columns = {f'COAL_{data_year}':'COAL_swing', f'ALP_{data_year}':'ALP_swing'}, inplace=True)
Omnipresent_parties_swings = Omnipresent_parties_swings[['div_nm','COAL_swing','ALP_swing']].sort_values(by='div_nm')

import pdb;pdb.set_trace()


###
#Omnipresent_parties_swings_melted = Omnipresent_parties_swings.melt(id_vars=['div_nm'], value_vars=['COAL_swing', 'ALP_swing'], var_name='party_swing', value_name='Swing')
#PP_swings = Omnipresent_parties_swings_melted.groupby(['div_nm', 'party_swing'])['Swing'].apply(list).reset_index() 

# keep COAL and ALP cols separate - don't melt!
PP_swings = Omnipresent_parties_swings.groupby('div_nm').agg({'COAL_swing': lambda x: list(x),'ALP_swing': lambda x: list(x)}).reset_index()


# MAKE SURE THAT BY USING SETS NO INFO IS DESTROYED - UNLIKELY!




Electorate_3PPs_redistributed = pd.read_csv(f'{previous_data_year}Electorate_3PPs_redistributed.csv', index_col=None)

Electorate_3PPs_curr = Omnipresent_parties_df_curr.groupby('div_nm', as_index=False).sum(numeric_only=True)
Electorate_3PPs_curr = Electorate_3PPs_curr[['div_nm'] + Electorate_3PPs_curr.columns[-3:].tolist()]

import pdb;pdb.set_trace()


Electorate_3PPs_swings = Electorate_3PPs_curr.merge(Electorate_3PPs_redistributed, on = 'div_nm', suffixes=(f'_{data_year}',f'_{previous_data_year}'))

Electorate_3PPs_swings = get_swings_from_merged_df(Electorate_3PPs_swings)


import pdb;pdb.set_trace()


# standardise PP swings!


# get location shift and apply
PP_sample_means = PP_swings.iloc[:, 1:].apply(lambda x: x.apply(np.mean) if x.apply(isinstance, args=(list,)).all() else x)

location_shift = Electorate_3PPs_swings.iloc[:,1:] - PP_sample_means
location_shift['div_nm'] = Electorate_3PPs_swings['div_nm'].values

PP_swings_merged = Omnipresent_parties_swings.merge(location_shift, on='div_nm',how='left', suffixes = ('','_shift'))
PP_swings_merged.iloc[:,1:3] = PP_swings_merged.iloc[:,1:3].values + PP_swings_merged.iloc[:,3:5].values
PP_swings_centered = PP_swings_merged.iloc[:,:3]

#PP_swings.iloc[:,1:] = PP_swings.iloc[:, 1:].apply(lambda col: col.apply(lambda x: (np.array(x) + location_shift.values).tolist() if isinstance(x, list) else x))

#PP_swings.iloc[:, 1:].apply(lambda col: col.apply(lambda x, idx: (np.array(x) + location_shift.iloc[idx, :]).tolist() if isinstance(x, list) else x, idx=col.index))
import pdb;pdb.set_trace()


sigma2_electorates = Electorate_3PPs_swings[['COAL_swing', 'ALP_swing']].var()
sigma2_PPs = PP_swings_centered.iloc[:,1:].var()

scaling_factors = sigma2_electorates / sigma2_PPs

PP_swings_standardised = PP_swings_centered
PP_swings_standardised.iloc[:,1:] = PP_swings_standardised.iloc[:,1:] * np.sqrt(scaling_factors)
PP_swings_standardised_COAL = PP_swings_standardised.groupby('div_nm')['COAL_swing'].apply(list)
PP_swings_standardised_ALP = PP_swings_standardised.groupby('div_nm')['ALP_swing'].apply(list)

grouped_array = np.array(PP_swings_standardised_COAL.tolist(), dtype=object)
n_groups = grouped_array.shape[0]


import pdb;pdb.set_trace()

# Actual Bootstrapping

n_iterations = 500000
bootstrap_samples = np.empty((n_iterations, n_groups))


for i in range(n_groups):
    # Sample one observation from each group
    #sampled_df = PP_swings_standardised.iloc[:,:2].groupby('div_nm', group_keys=False).apply(lambda x: x.sample(n=1, replace=True), include_groups=False)

    #bootstrap_samples[i] = sampled_df['COAL_swing'].values

    bootstrap_samples[:, i] = np.random.choice(grouped_array[i], size=n_iterations, replace=True) # one div at a time!


#df = pd.DataFrame(bootstrap_samples, columns = PP_swings_standardised['div_nm'].unique())
#correlation_matrix = df.corr()

import pdb;pdb.set_trace()

from pynetcor.cor import corrcoef

# using 8 threads
# Pearson correlations between `arr1` and itself
cor_result = corrcoef(bootstrap_samples.T, threads=8)
# Compute the correlation matrix

correlation_df = pd.DataFrame(correlation_matrix, columns=PP_swings_standardised['div_nm'].unique(), index=PP_swings_standardised['div_nm'].unique())

#correlation_matrix = np.corrcoef(bootstrap_samples, rowvar=False)

import pdb;pdb.set_trace()
























# get counts - Omnipresent_parties_swings.groupby('div_nm').size().reset_index(name='count').sort_values(by='count')

import pdb;pdb.set_trace()




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
#Omnipresent_parties_merged.loc[Omnipresent_parties_merged['pp_nm'].isin(set(asd) & set(qwe)),]

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