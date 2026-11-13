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


os.chdir(Path.home() / 'Australian Election' / 'Victorian Election')

data_year = 2018
election_year = data_year+4

PollingPlaces = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-Pollingplaces-Formatted.csv')

existing_divs = PollingPlaces['div_nm'].unique().tolist()
new_divs = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-Pollingplaces-Formatted.csv')['div_nm'].unique().tolist()


GEODATA_FOLDER = Path.home() / 'Australian Election/Victorian Election/Geodata'

if election_year == 2022:
    new_divs_shp = GEODATA_FOLDER / 'State-VIC-2021-shapefiles/VIC-2021-New districts and regions.shp'
    existing_divs_shp = GEODATA_FOLDER / 'State-VIC-2013-shapefiles/STATE_ASSEMBLY_2013.shp'

elif election_year == 2014:
    new_divs_shp = GEODATA_FOLDER / 'State-VIC-2013-shapefiles/STATE_ASSEMBLY_2013.shp'
    existing_divs_shp  = GEODATA_FOLDER / 'State-VIC-2001-shapefiles/STATE_ASSEMBLY_2001.shp'

new_state_gdf = gpd.read_file(new_divs_shp).to_crs(epsg=3111)
existing_state_gdf = gpd.read_file(existing_divs_shp).to_crs(epsg=3111)

def parse_district_transfer_stats(existing_divs, new_divs):
    valid_divs = existing_divs + new_divs
    valid_divs.sort(key=len, reverse=True)

    with open(f'{election_year}-redistribution-statistics.txt', 'r') as f:
        lines = f.readlines()

    parsed_data = []

    for line in lines:
        line = line.strip()
        
        # Skip empty lines or the header line
        if not line or line.startswith("New district"):
            continue

        # Clean multiple spaces down to a single space
        line = re.sub(r'\s+', ' ', line)

        # The number of electors is always the very last 'word'
        parts = line.rsplit(' ', 1)
        if len(parts) != 2: 
            continue
            
        text_part = parts[0].strip()
        electors_str = parts[1].replace(',', '')
        
        try:
            electors = int(electors_str)
        except ValueError:
            continue # Failsafe skip if the last item wasn't actually a number

        # 2. Extract New Div and Existing Div
        new_div = None
        existing_div = None
        
        for div in valid_divs:
            if text_part.startswith(div):
                new_div = div
                # The remaining text is the existing_div
                existing_div = text_part[len(div):].strip()
                break
                
        if new_div and existing_div:
            parsed_data.append((new_div, existing_div, electors))
        else:
            print(f"Warning: Could not parse candidates from line: '{line}'")

    # 3. Export to clean CSV
    df = pd.DataFrame(parsed_data, columns=['new_div', 'existing_div', 'electors'])

    df['donation_prop'] = df['electors'] / df.groupby('existing_div')['electors'].transform('sum')
    #df.to_csv('ebc_transfers_clean.csv', index=False)

    print(f"Successfully processed {len(df)} segments.")
    #import pdb; pdb.set_trace()
    return df

def generate_pb_redistribution_mapping(
    PB_df: str, 
    old_shapefile_path: str, 
    new_shapefile_path: str,
    old_shp_name_col: str = 'DISTRICT', # Replace with the actual column name in your 2013 shapefile
    new_shp_name_col: str = 'DISTRICT',  # Replace with the actual column name in your 2021 shapefile
    data_year: int = 0
) -> dict:
    """
    Performs a double spatial join to mathematically guarantee the 
    (existing_div, new_div) assignment for every physical polling booth.
    """
    
    # 2. Filter for physical booths with valid coordinates
    # Your snippet shows pp_type as numeric (1), but checking Lat/Long is the safest filter
    df_valid = PB_df.dropna(subset=['Lat', 'Long']).copy()
    
    # 3. Convert to a GeoDataFrame using standard WGS84 GPS coordinates (EPSG:4326)
    booths_gdf = gpd.GeoDataFrame(
        df_valid, 
        geometry=gpd.points_from_xy(df_valid['Long'], df_valid['Lat']),
        crs="EPSG:4326"
    )
    
    # 4. Load the Old Shapefile (e.g., 2013 boundaries for the 2018 election)
    old_shp = gpd.read_file(old_shapefile_path)
    if old_shp.crs != "EPSG:4326":
        old_shp = old_shp.to_crs("EPSG:4326")
        
    # 5. Load the New Shapefile (e.g., 2021 boundaries for the 2022 election)
    new_shp = gpd.read_file(new_shapefile_path)
    if new_shp.crs != "EPSG:4326":
        new_shp = new_shp.to_crs("EPSG:4326")

    old_shp['DISTRICT'] = old_shp['DISTRICT'].str.title()
    new_shp['DISTRICT'] = new_shp['DISTRICT'].str.title()



    # 6. Perform the First Spatial Join (Old Boundaries)
    # Using a left join ensures booths that sit exactly on a coastline aren't dropped
    joined_old = gpd.sjoin(booths_gdf, old_shp, how="left", predicate="intersects")
    joined_old = joined_old.rename(columns={old_shp_name_col: 'existing_div_geo'})
    
    # Drop the 'index_right' column added by sjoin before doing the second join
    if 'index_right' in joined_old.columns:
        joined_old = joined_old.drop(columns=['index_right'])

    # 7. Perform the Second Spatial Join (New Boundaries)
    final_joined = gpd.sjoin(joined_old, new_shp, how="left", predicate="intersects")
    final_joined = final_joined.rename(columns={new_shp_name_col: 'new_div_geo'})
    
    # 8. Clean up formatting (Title case to match your standard names)
    final_joined['existing_div_geo'] = final_joined['existing_div_geo'].str.title()
    final_joined['new_div_geo'] = final_joined['new_div_geo'].str.title()
    
    # set existing_div_geo to div_nm to ensure origin district consisent with redistribution
    final_joined['existing_div_geo'] = final_joined['div_nm']
    
    # 9. Output to CSV for inspection
    output_df = final_joined[['pp_id', 'div_nm', 'existing_div_geo', 'new_div_geo', 'premises', 'Lat', 'Long']]
    
    #output_df.to_csv('pb_spatial_mapping_results.csv', index=False)
    #import pdb; pdb.set_trace()
    # 10. Generate the master routing dictionary for Project 3
    # Format: {pp_id: ('Old_Div', 'New_Div')}
    
    all_pairs = list(final_joined[['existing_div_geo', 'new_div_geo']].dropna().drop_duplicates().itertuples(index=False, name=None))
    pb_routing_dict = {pair: [] for pair in all_pairs}

    for _, row in final_joined.iterrows():
        # Only route if it has a valid geometric new destination
        if pd.notna(row['new_div_geo']):
            #pb_routing_dict[row['pp_id']] = (row['existing_div_geo'], row['new_div_geo'])
            pb_routing_dict[(row['existing_div_geo'], row['new_div_geo'])] += [row['pp_id']]
        else:
            if row['pp_id'] == 160 and data_year == 2010:
                pb_routing_dict[(row['existing_div_geo'], row['existing_div_geo'])] += [row['pp_id']]
            else:
                import pdb; pdb.set_trace()


    print(f"Successfully spatially mapped {len(pb_routing_dict)} physical booths.")
    #import pdb; pdb.set_trace()
    return pb_routing_dict

#import pdb; pdb.set_trace()
rediv_transfer_props_df = parse_district_transfer_stats(existing_divs, new_divs)

rediv_pb_routing = generate_pb_redistribution_mapping(
     PB_df=PollingPlaces,
     old_shapefile_path=existing_divs_shp,
     new_shapefile_path=new_divs_shp,
     old_shp_name_col='DISTRICT',
     new_shp_name_col='DISTRICT',
     data_year=data_year  
)

transfer_pairs = list(rediv_transfer_props_df[['existing_div', 'new_div']].dropna().drop_duplicates().itertuples(index=False, name=None))

accidental_transfers = [key for key in rediv_pb_routing.keys() if key not in transfer_pairs]

for pair in accidental_transfers:
    pbs = rediv_pb_routing[pair]

    if (pair[0], pair[0]) not in rediv_pb_routing.keys():

        if data_year == 2018 and pair == ('Gembrook', 'Narre Warren North'):
            rediv_pb_routing[('Gembrook', 'Berwick')] += pbs
            del rediv_pb_routing[pair]
        else:
            import pdb; pdb.set_trace()
        continue

    rediv_pb_routing[(pair[0], pair[0])] += pbs
    del rediv_pb_routing[pair]

print([key for key in rediv_pb_routing.keys() if key not in transfer_pairs])
assert not [key for key in rediv_pb_routing.keys() if key not in transfer_pairs]
#import pdb; pdb.set_trace()
print([key for key in transfer_pairs if key not in rediv_pb_routing.keys()])
extra_keys = [key for key in transfer_pairs if key not in rediv_pb_routing.keys()]
# mask = rediv_transfer_props_df[['existing_div', 'new_div']].apply(tuple, axis=1).isin(extra_keys)
# rediv_transfer_props_df[mask]

PP_votes_dict = {}
PP_votes = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Primary-Pollingplace-Formatted.csv')
PP_votes.loc[(PP_votes['pp_name'].isin(['Absent Votes','Early Votes','Provisional Votes', 'Postal Votes','Marked As Voted Votes'])),'pp_id'] = 0
for div in PP_votes['div_nm'].unique():
    PP_votes_curr = PP_votes.loc[PP_votes['div_nm']==div,]
    PP_votes_curr = PP_votes_curr.groupby(['pp_id', 'PartyAb'], as_index=False)['votes'].sum()

    PP_votes_dict[div] = PP_votes_curr.pivot_table(index='pp_id', columns='PartyAb', values='votes', aggfunc='sum').drop(columns = ['INFORMAL'])

rediv_votes_by_pp_id = {}
for pair, pp_ids in rediv_pb_routing.items():

    target_indices = [0] + pp_ids

    pb_votes_df = PP_votes_dict[pair[0]] # search for votes form existing_div
    valid_indices = pb_votes_df.index.intersection(target_indices) 
    rediv_votes_by_pp_id[pair] = pb_votes_df.loc[valid_indices].copy()

for pair in extra_keys: # no booths transferred
    pb_votes_df = PP_votes_dict[pair[0]]
    rediv_votes_by_pp_id[pair] = pb_votes_df.loc[[0]].copy()

rediv_votes_by_pp_id
#rediv_transfer_props_df.to_csv(f'rediv_transfer_props_df_for_{election_year}.csv', index = False)
rediv_transfer_props_dict = rediv_transfer_props_df.set_index(['existing_div', 'new_div'])['donation_prop'].to_dict()
with open(f'rediv_votes_by_pp_id_for_{election_year}.pkl', "wb") as f:
        pickle.dump(rediv_votes_by_pp_id, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'rediv_transfer_props_dict_for_{election_year}.pkl', "wb") as f:
        pickle.dump(rediv_transfer_props_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
import pdb; pdb.set_trace()