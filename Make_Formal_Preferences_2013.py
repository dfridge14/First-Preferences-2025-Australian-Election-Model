import pandas as pd
import numpy as np
import os,time
from collections import Counter
import io
import os
import glob
from pathlib import Path

from multiprocessing import Pool, cpu_count, Manager



import gc


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




start = time.time()

HIGH_INT = 500
data_year = '2013'
states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']

#states = ['NSW','NT','QLD','SA','TAS','VIC','WA']


def fill_missing_number(row, num_preferences):
    zero_indices = (row == 0).to_numpy().nonzero()[0]  # Find a list of where 0 appears
    #import pdb;pdb.set_trace()
    if len(zero_indices) == 1:  # Only proceed if exactly one 0 is present
        present_numbers = set(row) - {0}  # Exclude 0 to get existing numbers
        missing_number = (set(range(1, 1+num_preferences)) - present_numbers).pop()  # Find missing number
        row.iloc[zero_indices[0]] = missing_number  # Replace 0 with missing number

    elif len(zero_indices) == 2:
        present_numbers = set(row) - {0}
        missing_numbers = sorted(list(set(range(1, 1+num_preferences)) - present_numbers))

        # allocate 2 missing numbers in order of appearance of cand_id
        try:
            row.iloc[zero_indices] = missing_numbers
        except:
            row.iloc[zero_indices] = [HIGH_INT for _ in range(len(zero_indices))] # ensure int is very high (500), so will be avoided when minimising
            #print("Duplicate row; make missing values into 500")

    else:
        row.iloc[zero_indices] = [HIGH_INT for _ in range(len(zero_indices))] # many unfilled values

    return row


def find_earliest_preference_id(preferences):
    # get indices where there may be duplication, store in multiple_min_mask
    # input: df with unique alphabetical column names and integer or nan values

    votes = preferences.idxmin(axis=1)

    min_values = preferences.min(axis=1)
    mask = preferences.eq(min_values, axis=0)
    multiple_min_mask = mask.sum(axis=1) > 1 # series with True when there are multiple minimum preferences - set as nan and deal with later

    if not votes[multiple_min_mask].empty:
        import pdb;pdb.set_trace()

    return votes, multiple_min_mask   # Returns NaN in votes if no preference for the candidate set, series if row min is not unique

def allocate_votes(df, allocation_set): 
    ### allocates vote following algorithm: First, ATL decides. If ATL not decisive, record any duplicates, try BTL. 

    allocated_votes = pd.DataFrame(index=df.index, columns=['Vote'])

    # If no vote is allocated using ATL ('Vote' is nan), use BTL
    BTL_allocations, BTL_non_unique_min = find_earliest_preference_id(df[allocation_set]) # selects lowest preference of combined BTL groups
    allocated_votes.loc[:,'Vote'] = BTL_allocations

    return allocated_votes # duplicate_indices are index object!

def allocate_Formal_preferences_to_First_Preferences(BTL):

    # produce df or dictionary of dfs with concatenated ATL&uniqueBTL and the First Preference vote in the last column

    first_prefs_set = BTL.columns.unique().tolist() # all cols including 'UG'
    
    # allocate first preferences using formal_prefs
    first_pref_allocated_votes = allocate_votes(BTL, first_prefs_set)
    formal_prefs_first_prefs = pd.concat([BTL, first_pref_allocated_votes], axis=1) # add allocated first_pref vote to formal_prefs_by_group df

    return formal_prefs_first_prefs


def get_BTL_Prefs_First_Prefs(data_year):

    for state in states:
        BTL_prefs_long = pd.read_csv(f'{data_year}SenateStateBTLPreferences{state}.csv', skiprows = 1, index_col = None, dtype={'Preference': str}, low_memory=False).iloc[:,:4].rename(columns={'CandidateId':'cand_id'})
        #BTL_prefs_long.loc[:,'Preference'] = BTL_prefs_long.loc[:,'Preference'].astype(float)
        First_prefs_senate = pd.read_csv(f'{data_year}SenateStateFirstPrefsByPollingPlace{state}.csv', skiprows = 1, skipfooter=1, index_col = None, engine = 'python').rename(columns={'CandidateID':'cand_id'})
        First_prefs_senate[['Ticket','cand_id','PartyNm']].drop_duplicates()

        # set nan values to 0 - later fill in!
        BTL_prefs_long.loc[:,'Preference'] = pd.to_numeric(BTL_prefs_long['Preference'], errors='coerce').fillna(0).astype(int) # there are some '??' values when single pref is missing

        # 1. Pivot BTL to wide by cand_id
        BTL_prefs_wide = BTL_prefs_long.pivot_table(index=['Batch','Paper'], 
                                                    columns=['cand_id'], 
                                                    values='Preference', 
                                                    aggfunc='first',
                                                    sort = False).reset_index(drop=True) 

        # these sometimes contain duplicate values, but are evidently not early enough to invalidate the vote! Proceed as with Formal Preferences
        
        # 2. Fill in single missing values
        num_preferences = len(BTL_prefs_wide.columns)
        # if one 0 missing, replace with missing; if two 0s missing, replace with both missing, in order of cand_id - 1st in ticket gets first pref! Ignore cases with duplicate indices
        BTL_prefs_wide_zeros = BTL_prefs_wide[BTL_prefs_wide.eq(0).any(axis=1)]
        BTL_prefs_wide_zeros = BTL_prefs_wide_zeros.apply(lambda row: fill_missing_number(row,num_preferences), axis=1)

        BTL_prefs_wide.loc[BTL_prefs_wide.index.isin(BTL_prefs_wide_zeros.index),] = BTL_prefs_wide_zeros.values

        # 3. Convert cand_id 
        columns_renaming_dict = First_prefs_senate.set_index('cand_id').apply(lambda row: row['Ticket'].strip(), axis=1).to_dict()
        BTL_prefs_wide.rename(columns = columns_renaming_dict, inplace=True)

        # Get minimum of each group
        BTL = BTL_prefs_wide.T.groupby(BTL_prefs_wide.columns).min().T

        BTL_first_prefs = allocate_Formal_preferences_to_First_Preferences(BTL)

        # make csv files
        BTL_first_prefs.to_csv(f"{data_year}BTLFirstPrefs{state}.csv", index = False)

    return 1

#get_BTL_Prefs_First_Prefs(data_year)

def sample_rows(Ticket, Votes, div_nm, pp_id,pp_nm, shared_dict):
    return shared_dict[Ticket].sample(n=Votes, replace=True).assign(Vote=Ticket, div_nm = div_nm,pp_id=pp_id,pp_nm=pp_nm)

def custom_sort(col):
    return (len(col), col)  # Sort first by length, then alphabetically


def sample_Formal_prefs(data_year,state):
    BTL_first_prefs = pd.read_csv(f"{data_year}BTLFirstPrefs{state}.csv", index_col = None)
    First_prefs_senate = pd.read_csv(f'{data_year}SenateStateFirstPrefsByPollingPlace{state}.csv', skiprows = 1, skipfooter=1, index_col = None, engine = 'python').rename(columns={'DivisionNm':'div_nm','PollingPlaceID':'pp_id', 'PollingPlaceNm':'pp_nm','CandidateID':'cand_id','OrdinaryVotes':'Votes'})
    First_prefs_senate = First_prefs_senate.groupby(['div_nm','pp_id','pp_nm','Ticket'], as_index=False)['Votes'].agg('sum')
    First_prefs_senate['Ticket'] = First_prefs_senate['Ticket'].str.strip()
    # Eliminate informal votes
    First_prefs_senate = First_prefs_senate.loc[First_prefs_senate['Ticket'] != 'ZZ']

    # add 'Other' votes - APPP:
    First_prefs_senate_APPP = pd.read_csv(f'{data_year}SenateFirstPrefsByDivisionByVoteType.csv', skiprows = 1,index_col = None).rename(columns={'DivisionNm':'div_nm'})
    First_prefs_senate_APPP = First_prefs_senate_APPP.loc[First_prefs_senate_APPP['StateAb'] == state,][['div_nm','Ticket','AbsentVotes','ProvisionalVotes','PrePollVotes','PostalVotes']]
    First_prefs_senate_APPP = First_prefs_senate_APPP.groupby(['div_nm','Ticket'], as_index=False)[['AbsentVotes','ProvisionalVotes','PrePollVotes','PostalVotes']].agg('sum')
    First_prefs_senate_APPP.loc[:,'Votes'] = First_prefs_senate_APPP.iloc[:,2:].sum(axis=1)
    First_prefs_senate_APPP.loc[:,'pp_nm'] = 'Other'
    First_prefs_senate_APPP.loc[:,'pp_id'] = 0
    First_prefs_senate_APPP = First_prefs_senate_APPP[['div_nm','pp_id','pp_nm','Ticket','Votes']]
    
    First_prefs_senate = pd.concat([First_prefs_senate,First_prefs_senate_APPP], ignore_index=True)
    import pdb;pdb.set_trace()


    BTL_dict = {p:group.set_index('Vote') for p, group in BTL_first_prefs.groupby('Vote')}

    #sampled_dfs = [BTL_dict[row.Ticket].sample(n=row.Votes, replace=True).assign(First_Pref=row.Ticket, div_nm = row.div_nm,pp_id=row.pp_id,pp_nm=row.pp_nm) for _, row in First_prefs_senate.iterrows()]
    #final_sampled_df = pd.concat(sampled_dfs, ignore_index=True)


    # use multiprocessing, although efficiency is limited due to shared dictionary. Still about 2-3 times faster
    if __name__ == "__main__":
        num_workers = min(cpu_count(), 12)  # Use up to 12 cores

        with Manager() as manager:  # Create a multiprocessing-safe dictionary
            shared_dict = manager.dict(BTL_dict)  

            with Pool(num_workers) as pool:
                sampled_dfs = pool.starmap(sample_rows, [(row.Ticket, row.Votes, row.div_nm, row.pp_id, row.pp_nm, BTL_dict)  for _, row in First_prefs_senate.iterrows()])

            # Combine results
            final_sampled_df = pd.concat(sampled_dfs, ignore_index=True)

            desired_order = ['div_nm', 'pp_nm'] + sorted([col for col in final_sampled_df.columns if col not in ['div_nm', 'pp_id', 'pp_nm']], key = custom_sort)
            final_sampled_df = final_sampled_df[desired_order]

            final_sampled_df.to_csv(f"{data_year}FormalPrefsSampledReduced{state}.csv", index=False)
            print("Done", state)

    return final_sampled_df


Formal_prefs = {}
Party_names = pd.DataFrame(columns = ['State','div_nm','Ticket'])


def get_2007_2013_Senate_party_names(state):
    First_prefs_senate = pd.read_csv(f'{data_year}SenateStateFirstPrefsByPollingPlace{state}.csv', skiprows = 1, skipfooter=1, index_col = None, engine = 'python').rename(columns={'DivisionNm':'div_nm','PollingPlaceID':'pp_id', 'PollingPlaceNm':'pp_nm','CandidateID':'cand_id','OrdinaryVotes':'Votes'})
    SenateCandidates = First_prefs_senate[['Ticket','PartyNm']].drop_duplicates()

    party_names_list = SenateCandidates.loc[~SenateCandidates['Ticket'].isin(['UG','ZZ']),'PartyNm'].drop_duplicates(ignore_index=True).tolist()

for state in states:
    Formal_prefs[state] = sample_Formal_prefs(data_year,state)

    First_prefs_senate = pd.read_csv(f'{data_year}SenateStateFirstPrefsByPollingPlace{state}.csv', skiprows = 1, skipfooter=1, index_col = None, engine = 'python').rename(columns={'DivisionNm':'div_nm','PollingPlaceID':'pp_id', 'PollingPlaceNm':'pp_nm','CandidateID':'cand_id','OrdinaryVotes':'Votes'})
    First_prefs_senate = First_prefs_senate.groupby(['div_nm','Ticket'], as_index=False)['Votes'].agg('sum')
    First_prefs_senate['Ticket'] = First_prefs_senate['Ticket'].str.strip()



import pdb;pdb.set_trace()



