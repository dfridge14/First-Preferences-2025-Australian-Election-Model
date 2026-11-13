import pandas as pd
import numpy as np
import os
from pathlib import Path


import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler



data_year = 2006
state = 'VIC'

tally_room_dir = Path.home() / f"Australian Election/Victorian Election/Tally Room/{state}-{data_year}"
main_dir =  Path.home() / f"Australian Election/Victorian Election/{state}-{data_year}"


def allocate_regional_votes_to_electorate(df):
    # 1. Define the metadata columns that describe the candidate/electorate
    # Grouping by these carries them straight through to the final output
    group_cols = [
        'district_name', 'district_id', 'region_name', 'region_id', 
        'candidate_id', 'candidate_name', 'surname', 'first_name', 
        'party_code', 'group_code', 'ballot_order'
    ]
    
    # 2. Separate local booth votes from unallocated regional votes
    local_df = df[df['district_name'].notna() & (df['district_name'] != 'NA')]
    regional_df = df[df['district_name'].isna() | (df['district_name'] == 'NA')]

    # 3. Sum local votes, keeping all metadata
    local_totals = local_df.groupby(group_cols, dropna=False)['votes'].sum().reset_index()

    # 4. Calculate each electorate's share of a candidate's local vote within their region
    candidate_region_totals = local_totals.groupby(['region_name', 'candidate_id'])['votes'].transform('sum')
    local_totals['weight'] = np.where(candidate_region_totals > 0, local_totals['votes'] / candidate_region_totals, 0)

    # 5. Sum the unallocated votes pool by Region and Candidate
    unallocated_pool = regional_df.groupby(['region_name', 'candidate_id'])['votes'].sum().reset_index()
    unallocated_pool = unallocated_pool.rename(columns={'votes': 'unallocated_votes'})

    # 6. Merge the unallocated pool onto the local proportions
    merged = pd.merge(local_totals, unallocated_pool, on=['region_name', 'candidate_id'], how='left')
    merged['unallocated_votes'] = merged['unallocated_votes'].fillna(0)

    # 7. Distribute, add, and round to integer
    merged['distributed_votes'] = merged['weight'] * merged['unallocated_votes']
    merged['votes'] = (merged['votes'] + merged['distributed_votes']).round(0).astype(int)

    df = merged[group_cols + ['votes']]

    def standardise_2006_atl_rows(df):
        """
        Creates structural 'Above-the-line' rows for grouped candidates in 2006 
        with 0 votes to prevent double-counting, as ATL votes are already rolled 
        into the lead candidates.
        """
        # 1. Identify valid groups (ignore 'UG' ungrouped and 'INF'/'ZZ' informal)
        grouped_df = df[~df['group_code'].isin(['UG', 'INF', 'ZZ'])]
        
        # 2. Extract one representative row per group per electorate to act as the template
        atl_rows = grouped_df.drop_duplicates(subset=['district_name', 'group_code'], keep='first').copy()
        
        # 3. Set raw votes to 0 so we don't double-count the group total!
        atl_rows['votes'] = 0
        
        # 4. Overwrite the candidate-specific fields to match the standard ATL format
        atl_rows['candidate_name'] = 'Above-the-line'
        atl_rows['surname'] = 'Above-the-line'
        atl_rows['first_name'] = 'Above-the-line'
        atl_rows['candidate_id'] = 0
        atl_rows['ballot_order'] = 0
        
        # 5. Append the synthetic ATL rows to the original dataframe
        standardised_df = pd.concat([atl_rows, df], ignore_index=True)
        
        return standardised_df
    
    return standardise_2006_atl_rows(df)


# For 2006, combine polling place votes into Electorates-Votes...? 

if data_year == 2006:
    df = pd.read_excel(tally_room_dir / f'VIC-2006-LC-Votes-Pollingplace.xlsx')
    df = allocate_regional_votes_to_electorate(df).sort_values(by = ['district_id', 'group_code','ballot_order'])
    if not (main_dir / 'VIC-2006-LC-Votes-Electorate.xlsx').exists():
        df.to_excel(main_dir / 'VIC-2006-LC-Votes-Electorate.xlsx', index = False)
    #import pdb; pdb.set_trace()
    

if state == 'VIC':




    # rename district_name -> div_nm, party_code -> PartyAb
    party_df = pd.read_csv(main_dir / f'VIC-{data_year}-Parties.csv')
    party_mapping = party_df.set_index('party_code')['PartyAb']

    file_map = {        
        'LA_Candidates': tally_room_dir / f'VIC-{data_year}-LA-Candidates.xlsx',
        'LC_Candidates': tally_room_dir / f'VIC-{data_year}-LC-Candidates.xlsx',
        'LA_results': tally_room_dir / f'VIC-{data_year}-LA-Primary-Electorate.xlsx',
        'LC_results': tally_room_dir / f'VIC-{data_year}-LC-Votes-Electorate.xlsx' if data_year != 2006 else main_dir / 'VIC-2006-LC-Votes-Electorate.xlsx',
        'LA_PP_results': tally_room_dir / f'VIC-{data_year}-LA-Primary-Pollingplace.xlsx',
        'Polling_Places': tally_room_dir / f'VIC-{data_year}-Pollingplaces.xlsx'
    }

    for name, file_path in file_map.items():
        new_file_path =  main_dir / f"{file_path.stem}-Formatted.csv"

        if not new_file_path.exists():

          

            df = pd.read_excel(file_path)

            if name == 'Polling_Places':
                df.rename(columns={'district_name': 'div_nm', 'latitude': 'Lat','longitude':'Long'}, inplace=True)
            else:
            
                # Modify in place and map the standardized values
                df.rename(columns={'district_name': 'div_nm', 'party_code': 'PartyAb'}, inplace=True)
                df['PartyAb'] = df['PartyAb'].map(party_mapping)

                # adjust PartyAb of IND to INDX based on ballot order
                if 'LA' in name:
                    is_ind = df['PartyAb'] == 'IND'
                    df.loc[is_ind, 'PartyAb'] = ('IND' + (df[is_ind].groupby('div_nm').cumcount() + 1).astype(str)) if name != 'LA_PP_results' else ('IND' + (df[is_ind].groupby(['div_nm','pp_id']).cumcount() + 1).astype(str))
            #import pdb; pdb.set_trace()
            df.to_csv(new_file_path, index=False)
            print(f"Created: {new_file_path.name}")

        else:
            print(f"Already exists, skipping: {new_file_path.name}")
        
        #import pdb; pdb.set_trace()
        

    

            

# for pre-2022 add 2CP results

for data_year in [2006,2010,2014,2018]:
    filepath = Path.home() / f"Australian Election/Victorian Election/Tally Room/{state}-{data_year}/{state}-{data_year}-LA-2CP-Electorate.xlsx"
    main_dir =  Path.home() / f"Australian Election/Victorian Election/{state}-{data_year}"
    new_file_path =  main_dir / f"{filepath.stem}-Formatted.csv"
    df = pd.read_excel(filepath)

    df.rename(columns={'district_name': 'div_nm', 'party_code': 'PartyAb'}, inplace=True)
    df['PartyAb'] = df['PartyAb'].map(party_mapping)

    df.to_csv(new_file_path, index=False)
    print(f"Created: {new_file_path.name}")
        
    import pdb; pdb.set_trace()