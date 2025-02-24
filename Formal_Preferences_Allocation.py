import pandas as pd
import numpy as np
import os,time
import ast

import gc

os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()


data_year = '2022'



# Game plan:
# Four different places where FormalPrefs will be necessary:
# 1/2. Redistribution changes & candidate changes - (1. Simultaneously: ~140+110 or 2. nowcast ~110 for 2025) (IF nowcast model)
# 3. Incumbent advantage - (150) (Possibly simultaneously across all elections)
# 4. Polling_Top_x (150)
# - First Preferences required for each
# - 1/2(possibly 3) need polling-booth specifics
# Should be done simultaenously, saved to candidate-wide dfs of length 150 or 150*50 (without or with polling booth specifics)
# Write 4 csv files: 
# 1/2. Candidate changes | div_nm, pp_id vs parties     (len(parties) might differ)
# 3. Incumbent advantage | div_nm (, pp_id) vs parties  (len(parties) the same!)
# 4. Polling_Top_x | div_nm vs parties (len(parties) the same!)

# First, read and generate dict of FormalPrefs by div, from the state csv's
# Produce FPrefs once - modify existing dict
# For each of 1,2,3,4:
# If div is in 1list,2list,3list & all for 4: allocate, aggregate and add to corresponding df to eventually write to csv.







def abbreviate_party_names(party_names_list, general_party_df):
    # handle exceptions to party names
    party_abvs_list = []

    for party in party_names_list:
        #print(party)
        if party:
            if party.lower() == "Liberal/The Nationals".lower() or party.lower() == "Liberal & Nationals".lower(): # handle LIB/NAT Exception - I think best to treat them as one party in the Senate as they always contest together, and then reverse engineer House split if needed
                party_abvs_list.append('COAL')
            elif party == " Science, Pirate, Secular, Climate Emergency": # SOPA exception
                party_abvs_list.append('SOPA')
            elif party == "Labor/Country Labor":
                party_abvs_list.append('ALP')
            else:
                party_abvs_list.append(general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].iloc[0])
        else:
            party_abvs_list.append('')


        #import pdb;pdb.set_trace()

    return party_abvs_list


general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',"PartyAb"] = 'GRN' # handle exceptions, but think GVIC is the only one


def get_Senate_party_abvs_dict():
    # quickly extracts abvs from the senate without needing to read all of Formal Prefs


    Formal_prefs_dict = {}
    states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']

    #### basic version to get the party names lists for cheap - read only 2 rows each!
    for state in states: # currently only 2016 onwards
        filename = f"{data_year}FormalPrefs{state}.csv"

        state_Formal_prefs = pd.read_csv(filename, nrows=2)

        state_Formal_prefs_dict = {state: group.reset_index(drop=True).apply(
            lambda col: pd.to_numeric(col, downcast='float') if pd.api.types.is_numeric_dtype(col) else col
        ) for state, group in state_Formal_prefs.groupby("State")} 
        for key, group in state_Formal_prefs_dict.items():
            Formal_prefs_dict[key] = group # assumes no keys (divs) overlap for different states :)

    # get state-to-div dict
    div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
    div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}



    Senate_party_abvs_dict = {}
    for div in div_to_state_dict.keys():
        #import pdb;pdb.set_trace()
        state = div_to_state_dict[div] # gets StateAb
        formal_prefs_full = Formal_prefs_dict[state]
        formal_prefs = formal_prefs_full.iloc[:, 6:]
        formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
        start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated

        # store group party names (from ATL) in Senate_party_names_dict
        group_party_names = formal_prefs_full.iloc[:, 6:].columns[:start_of_BTL_index] # includes both group and party names
        party_names_list = group_party_names.str.split(':').str[-1].tolist() # records only party names
        print(div)
        party_abvs_list = abbreviate_party_names(party_names_list, general_party_df)
        Senate_party_abvs_dict[div] = party_abvs_list
    
    # write to csv
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    #Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) 

    return 1


#get_Senate_party_abvs_dict()

def optimize_dataframe(df):
    # Convert all remaining numeric columns to smallest possible int type
    df.iloc[:, 3:] = df.iloc[:, 3:].apply(pd.to_numeric, downcast='integer')

    return df


Formal_prefs_dict = {}
#states = ['NSW']
states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
for state in states: # currently only 2016 onwards

    gc.collect()


    print(state)
    filename = f"{data_year}FormalPrefs{state}.csv"

    if (state == 'NSW') & (data_year == '2019'):

        df_columns = pd.read_csv(filename, index_col=None, nrows=1).columns
        # fix malformed file using readlines:
        expected_columns = len(df_columns)  # You can adjust this number to match your actual expected columns
        batch_size = 1_000_000  # Process 1 million rows at a time
        dataframes = []  # Store DataFrames in a lis

        # Read the CSV file line by line
        with open(filename, 'r') as file:
            
            batch = []
            row_count = 0  # Keep track of rows read

            for i, line in enumerate(file, start = -1):
                if i == -1:
                    continue  # Skip the first row

                row = line.strip().split(',') # Split the line by commas

                # Fix malformed rows by padding or truncating
                if len(row) < expected_columns:
                    row.extend([np.nan] * (expected_columns - len(row)))
                elif len(row) > expected_columns:
                    import pdb;pdb.set_trace()

                    print("AHHHHHHHHHHHH")
                    row = row[:expected_columns]
                
                batch.append(row)
                row_count += 1

                # Process and store each batch
                if row_count % batch_size == 0:
                    print(f"Processing batch {len(dataframes) + 1}, rows read: {row_count}")  # Debugging
                    df_batch = pd.DataFrame(batch, columns=df_columns)
                    dataframes.append(df_batch)
                    batch = []  # Reset batch, clears memory
                    print("done", time.time() - start)

            # Process any remaining rows
            if batch:
                print(f"Processing final batch, rows read: {row_count}")  # Debugging
                df_batch = pd.DataFrame(batch, columns=df_columns)
                dataframes.append(df_batch)
                print("done", time.time() - start)

        # Combine all batches into a final DataFrame
        if dataframes:
            curr_Formal_prefs = pd.concat(dataframes, ignore_index=True)
            print("Final DataFrame shape:", curr_Formal_prefs.shape)
        else:
            print("No data was read from the file.")
        #curr_Formal_prefs = pd.concat(dataframes, ignore_index=True)

    else:
        # low memory try
        columns = pd.read_csv(filename, nrows=1).columns
        dtype_dict = {col: 'str' for col in columns[:3]}
        for col in columns[3:]:
            dtype_dict[col] = 'float32'

        curr_Formal_prefs = pd.read_csv(filename, index_col=None, na_values=["NaN", "nan"], dtype=dtype_dict)

    
    curr_Formal_prefs.rename(columns={"Division": "div_nm", "Vote Collection Point Name": "pp_nm"}, inplace=True)

    # Not enough memory --> downcast floats to lower order for numeric columns
    state_div_Formal_prefs_dict = {
        div: group.reset_index(drop=True)  # optimize_dataframe(group.reset_index(drop=True)
        for div, group in curr_Formal_prefs.groupby("div_nm")
    } 
    for key, group in state_div_Formal_prefs_dict.items():
        Formal_prefs_dict[key] = group # assumes no keys (divs) overlap for different states :)

    curr_Formal_prefs = {}

    print("done", time.time() - start)

#import pdb;pdb.set_trace()

# TO DO:
# 1. Fix up indexing so each element of dict starts with 0                          DONE
# 2. First preferences allocate - useful for all!!!                                 DONE
# 3. get ordered list of 5 parties from Incumbent Advantage csv                     DONE
# 4. For each div allocate to specific one                                          DONE
# 5. aggregate into whole or by pp_id                                               DONE
# 6. Use PollingPlacesRepository for correspondence between names and pp_id
# 7. write to csv of senate prefs                                                   DONE
# 8. For 3. Make list of all top 5 that match senate: Senate_Party_Abs per division DONE
        





general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',"PartyAb"] = 'GRN' # handle exceptions, but think GVIC is the only one


def find_earliest_preference_id(preferences):
    # get indices where there may be duplication, store in multiple_min_mask
    # input: df with unique alphabetical column names and integer or nan values

    #votes = preferences.idxmin(axis=1, skipna=True)
    #votes[preferences.isna().all(axis=1)] = np.nan 
    votes = preferences.fillna(float("inf")).idxmin(axis=1) # min in row, avoids warning
    votes = votes.where(preferences.notna().any(axis=1), other=pd.NA) # no prefs in row

    min_values = preferences.min(axis=1)
    mask = preferences.eq(min_values, axis=0)
    multiple_min_mask = mask.sum(axis=1) > 1 # series with True when there are multiple minimum preferences - set as nan and deal with later

    votes[multiple_min_mask] = np.nan

    return votes, multiple_min_mask   # Returns NaN in votes if no preference for the candidate set, series if row min is not unique



def allocate_votes(df, allocation_set): 
    ### allocates vote following algorithm: First, ATL decides. If ATL not decisive, record any duplicates, try BTL. 

    start_of_BTL_index = next(i for i, col in enumerate(df.columns) if df.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated
    ATL = df.iloc[:,:start_of_BTL_index]
    BTL = df.iloc[:,start_of_BTL_index:]
    allocated_votes = pd.DataFrame(index=df.index, columns=['Vote'])

    # ATL preferences
    allocation_set_UG = [col for col in allocation_set if col != 'UG'] # remove UG from ATL preferences if exists - only useful for First_Preferences
    ATL_preferences = ATL[allocation_set_UG]     # Filter the row to only include the candidates of interest
    allocated_votes.loc[:,'Vote'], ATL_non_unique_min = find_earliest_preference_id(ATL_preferences)

    # If no vote is allocated using ATL ('Vote' is nan), use BTL
    BTL_allocations, BTL_non_unique_min = find_earliest_preference_id(BTL[allocation_set]) # selects lowest preference of combined BTL groups
    allocated_votes.loc[:,'Vote'] = allocated_votes.loc[:,'Vote'].fillna(BTL_allocations) 

    ## BTL_non_unique_min should be unnecessary as vote MUST be Formal
    
    if allocation_set != df.columns.unique().tolist(): # if allocation_set not just first prefs, handle duplicates by adding index to list
        ATL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & ATL_non_unique_min
        BTL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & ~ATL_duplicates_series & BTL_non_unique_min # duplicate but not in ATL

        # collect indices 
        duplicates = ATL_duplicates_series+BTL_duplicates_series
        duplicate_indices = duplicates.loc[duplicates].index # return indices where duplicates == True
    else:
        duplicate_indices = [] # no duplicated 1st prefs

    return allocated_votes, duplicate_indices # duplicate_indices are index object!

def allocate_votes_duplicates(df, allocation_set):
        
        start_of_BTL_index = next(i for i, col in enumerate(df.columns) if df.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated
        ATL_dup = df.iloc[:,:start_of_BTL_index]
        BTL_dup = df.iloc[:,start_of_BTL_index:]

        # ATL preferences
        ATL_dup_preferences = ATL_dup[allocation_set]     # Filter the row to only include the candidates of interest
        duplicate_votes_series = ATL_dup_preferences.apply(lambda row: list(row[row == row.min()].index), axis = 1)

        # If no vote is allocated using ATL ('Vote' is nan), use BTL
        BTL_dup_preferences = BTL_dup[allocation_set].apply(lambda row: list(row[row == row.min()].index), axis = 1) # selects lowest preferences of combined BTL groups
        
        #empty_mask = duplicate_votes_series.apply(lambda x: len(x) == 0)  #Find empty lists, replace them with BTL_dup_preferences (effectively using a mask)
        #duplicate_votes_series.loc[empty_mask] = BTL_dup_preferences[empty_mask].values
        duplicate_votes_series.loc[duplicate_votes_series.apply(lambda x: len(x) == 0)] = BTL_dup_preferences

        return duplicate_votes_series


def abbreviate_party_names(party_names_list, general_party_df):
    # handle exceptions to party names
    party_abvs_list = []

    for party in party_names_list:
        #print(party)
        if party:
            if party == "Liberal/The Nationals" or party == "Liberal & Nationals": # handle LIB/NAT Exception - I think best to treat them as one party in the Senate as they always contest together, and then reverse engineer House split if needed
                party_abvs_list.append('COAL')
            elif party == " Science, Pirate, Secular, Climate Emergency": # SOPA exception
                party_abvs_list.append('SOPA')
            else:
                party_abvs_list.append(general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].iloc[0])
        else:
            party_abvs_list.append('')


        #import pdb;pdb.set_trace()

    return party_abvs_list


def allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict, general_party_df):

    # produce df or dictionary of dfs with concatenated ATL&uniqueBTL and the First Preference vote in the last column

    Senate_party_abvs_dict = {}

    for div in Formal_prefs_dict.keys():
        #import pdb;pdb.set_trace()
        formal_prefs_full = Formal_prefs_dict[div]
        formal_prefs = formal_prefs_full.iloc[:, 6:]
        formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
        start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated

        # store group party names (from ATL) in Senate_party_names_dict
        group_party_names = formal_prefs_full.iloc[:, 6:].columns[:start_of_BTL_index] # includes both group and party names
        party_names_list = group_party_names.str.split(':').str[-1].tolist() # records only party names
        party_abvs_list = abbreviate_party_names(party_names_list, general_party_df)
        Senate_party_abvs_dict[div] = party_abvs_list


        first_prefs_set = formal_prefs.columns.unique().tolist() # all cols including 'UG'
        # fix BTL into single group and concatenate
        ATL = formal_prefs.iloc[:,:start_of_BTL_index]
        BTL = formal_prefs.iloc[:,start_of_BTL_index:]
        BTL = BTL.apply(pd.to_numeric)
        BTL = BTL.T.groupby(BTL.columns).min().T
        formal_prefs_by_group = pd.concat([ATL, BTL], axis=1)

        # allocate first preferences using formal_prefs
        first_pref_allocated_votes = allocate_votes(formal_prefs_by_group, first_prefs_set)[0]
        formal_prefs_first_prefs = pd.concat([formal_prefs_by_group, first_pref_allocated_votes], axis=1) # add allocated first_pref vote to formal_prefs_by_group df
        Formal_prefs_dict[div] = pd.concat([Formal_prefs_dict[div].iloc[:,1:3], formal_prefs_first_prefs], axis=1)
        #print(Formal_prefs_dict[div])
        #import pdb;pdb.set_trace()

    return Formal_prefs_dict, Senate_party_abvs_dict





list1 = []
list2 = []
incumbency_advantage_dict3 = [] # div: [list of PartyAbs]
list4 = []




First_Prefs_by_PP_Complete = pd.read_csv("2022FirstPrefsByPPComplete.csv", index_col = None)
Booth_name_pp_id = First_Prefs_by_PP_Complete.iloc[:,:3].drop_duplicates()



def allocate_formal_preferences_to_allocation_set(formal_prefs_div, allocation_set):

    Final_allocated_votes = pd.DataFrame(index=formal_prefs_div.index, columns=allocation_set, data = 0) # df of allocated votes for each candidate, should preserve order of df
    Final_allocated_votes["First_Preferences"] = formal_prefs_div["Vote"]

    # building groups based on first preference vote, either allocate vote directly to allocation_set if one of them, else allocate by group among later preferences

    for party, formal_subsection in formal_prefs_div.iloc[:,2:].groupby("Vote"): # ignore first 2 rows in calculation (div/pp)
        
        if party in allocation_set:
            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party, party] = 1.0 # put in a 1 into the party column while preserving index

        else:
            allocated_votes_subsection, duplicate_indices_subsection = allocate_votes(formal_subsection, allocation_set)
            Subsection_final_votes = Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party] # just working with this subsection

            # Add allocation preferences where clear
            mask = pd.get_dummies(allocated_votes_subsection.loc[allocated_votes_subsection["Vote"].notna(), "Vote"])
            mask = mask.reindex(Subsection_final_votes.index, fill_value=0)     # Align the mask with the indices of Subsection_final_votes
            Subsection_final_votes.loc[:, mask.columns] = mask.astype(float)         # Update Subsection_final_votes with the mask

            # Add duplicate preferences 
            duplicate_for_party_df = formal_subsection[formal_subsection.index.isin(duplicate_indices_subsection)].copy()

            # iteratively add duplicate votes proportionate to # of cnadidates duplicated
            duplicate_for_party_df["Vote"] = allocate_votes_duplicates(duplicate_for_party_df, allocation_set) # get series of candidates for each duplicate votes
            
            Subsection_final_votes.iloc[:, :-1] = Subsection_final_votes.iloc[:, :-1].apply(pd.to_numeric)
            Subsection_final_votes.iloc[:,:-1] = Subsection_final_votes.iloc[:,:-1].astype(float)

            for row in duplicate_for_party_df.index:
                duplicate_vote_list = duplicate_for_party_df.loc[duplicate_for_party_df.index == row,"Vote"].iloc[0] # iloc makes it a list
                for vote in duplicate_vote_list:
                    Subsection_final_votes.loc[Subsection_final_votes.index==row, vote] = 1/len(duplicate_vote_list)
            

            # handle remainingg nan values - assign votes proportional to how rest of their subsection voted
            Subsection_final_votes = Subsection_final_votes.drop(columns=['First_Preferences'])
            Party_preferences_proportions = Subsection_final_votes.sum() / np.sum(Subsection_final_votes.sum()) # row of proportions
            mask = allocated_votes_subsection["Vote"].isna() & ~allocated_votes_subsection.index.isin(duplicate_indices_subsection)
            Subsection_final_votes.loc[mask] = pd.DataFrame([Party_preferences_proportions.values] * sum(mask), index=Subsection_final_votes.index[mask], columns=Subsection_final_votes.columns) # changed from mask.sum()

            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party,Final_allocated_votes.columns[:-1]] = Subsection_final_votes # fill out full table



    Final_allocated_votes_df = pd.concat([formal_prefs_div.iloc[:,:2], Final_allocated_votes], axis=1).drop(columns = "First_Preferences") # return 1st 3 cols & remove last


    # This is where to return in the pp_id column 
    Final_allocated_votes_aggregated_df = Final_allocated_votes_df.groupby(["div_nm", "pp_nm"], as_index=False).sum()
    #Final_allocated_votes_aggregated_df = Final_allocated_votes_df.drop(columns = ["pp_nm"]).groupby(["div_nm"], as_index=False).sum()

    # GROUP THE STARTSWITH ABSENT,PREPOLL,POSTAL,PROVISIONAL,EAV,REMOTEMT,SPECIALMT,OTHERMT TOGETHER WITH PP_ID 0, THE REST MERGE WITH PP_IDS
    Other_booth_type_prefixes = ['Remote Mobile', 'Other Mobile','Special Hospital','EAV','ABSENT','PROVISIONAL','PRE_POLL','POSTAL']
    Final_allocated_votes_aggregated_df.loc[:,"pp_nm"] = Final_allocated_votes_aggregated_df.loc[:,"pp_nm"].apply(lambda x: 'Other' if any(x.startswith(prefix) for prefix in Other_booth_type_prefixes) else x)
    Final_allocated_votes_aggregated_df = Final_allocated_votes_aggregated_df.groupby(["div_nm", "pp_nm"], as_index=False).sum() # group again

    #Booth_name_pp_id_div = Booth_name_pp_id.loc[Booth_name_pp_id["div_nm"] == div, ]

    # switch pp_nm to pp_id
    Final_allocated_votes_aggregated_df = pd.merge(Final_allocated_votes_aggregated_df, Booth_name_pp_id, on = ['div_nm','pp_nm'], how='left')
    Final_allocated_votes_aggregated_df.loc[:,'pp_nm'] = Final_allocated_votes_aggregated_df.loc[:,'pp_id']
    import pdb;pdb.set_trace()

    Final_allocated_votes_aggregated_df.drop(columns=['pp_id'], inplace=True)
    Final_allocated_votes_aggregated_df.rename(columns={"pp_nm":"pp_id"}, inplace=True)


    return Final_allocated_votes_aggregated_df



def allocate_Formal_prefs_by_1234(Formal_prefs_dict, Senate_party_abvs_dict, application_dict):
    #### application_dict is one of 1,2,3 (incumbency_advantage),4

    Final_allocated_pcts_aggregated_dict = {}

    for div in application_dict.keys(): # only apply to relevant divisions

        # convert allocation_abvs into Senate Group letters
        allocation_abvs_list = application_dict[div] # list of PartyAb to allocate to

        allocation_set = []
        # Goal is to preserve order of allocation_abvs_list in allocation_set
        for party in allocation_abvs_list:  # Iterate through allocation_abvs_list directly
            if party in Senate_party_abvs_dict[div]: 
            
                i = Senate_party_abvs_dict[div].index(party) # Find the index of the party in this div and use it to get the corresponding Senate Group name
                allocation_set.append(Formal_prefs_dict[div].columns[2:len(Senate_party_abvs_dict[div])+2][i])  # Append the corresponding group 'letter'
        #for i, party in enumerate(Senate_party_abvs_dict[div]):
        #    if party in allocation_abvs_list:
        #        allocation_set.append(Formal_prefs_dict[div].columns[2:len(Senate_party_abvs_dict[div])+2][i]) # add corresponding group 'letter' to allocation_set | len is equivalent to number of ATL parties + 3 starting columns

        # allocate to allocation_set and convert to percentages
        Final_allocated_pcts_aggregated_dict[div] = allocate_formal_preferences_to_allocation_set(Formal_prefs_dict[div], allocation_set)
        Final_allocated_pcts_aggregated_dict[div].iloc[:, 2:] = Final_allocated_pcts_aggregated_dict[div].iloc[:, 2:].div(Final_allocated_pcts_aggregated_dict[div].drop(columns=['div_nm','pp_nm']).sum(axis=1), axis=0)

    return Final_allocated_pcts_aggregated_dict


















DOP_By_PP_2022 = pd.read_csv("2022DOP_By_PP_full.csv", index_col=None)

######## TO DO: Bring in Candidate_Pairs into here, adjusted for DOP_By_PP
### Change order within DOP_By_PP to match rather arbitrary m-c1-c2 ordering (consistency)
### Apply the function to existing counts, returning final counts!



def convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div):
    ### convert allocation_abvs into Senate Group letters
    allocation_set = []
    # Goal is to preserve order of allocation_abvs_list in allocation_set
    for party in allocation_abvs_list:  # Iterate through allocation_abvs_list directly
        if party in Senate_party_abvs_dict[div]: 
            i = Senate_party_abvs_dict[div].index(party) # Find the index of the party in this div and use it to get the corresponding Senate Group name
            allocation_set.append(Formal_prefs_dict[div].columns[2:len(Senate_party_abvs_dict[div])+2][i])  # Append the corresponding group 'letter'
    return allocation_set
    

def allocate_Formal_prefs_Redistribution_change(Formal_prefs_dict, Senate_party_abvs_dict, Redistribution_pair_c1_c2_lists):
    #### want to produce a df that 

    Final_allocated_pcts_aggregated_dict = {}

    div_pair_keys = list(Redistribution_pair_c1_c2_lists.iloc[:, :2].itertuples(index=False, name=None))
    import pdb;pdb.set_trace()


    for index, row in Redistribution_pair_c1_c2_lists.iterrows(): # only apply to relevant division pairs

        c1_m_c2_dict = {}
        import pdb;pdb.set_trace()

        giver_div = row[0]

        for idx, (col_name, value) in enumerate(row.iloc[2:].items()): # iterate over the 3 lists of PartyAb
            allocation_abvs_list = ast.literal_eval(value) #(= row[col_name]) list of PartyAb to allocate to
            
            import pdb;pdb.set_trace()
            if idx>=1 and value == row.iloc[2:].iloc[idx-1]: # c1 or c2 is same as m --> don't need to repeat
                c1_m_c2_dict[col_name] = c1_m_c2_dict[row.columns[2+idx-1]] # copies previous column
            else:
                allocation_set = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, giver_div)

                # allocate to allocation_set and convert to percentages - BE CAREFUL TO DO IT PER ROW AND NOT TOTALLY
                Final_allocated_pcts_aggregated_dict[giver_div] = allocate_formal_preferences_to_allocation_set(Formal_prefs_dict[giver_div], allocation_set)
                df = Final_allocated_pcts_aggregated_dict[giver_div]
                import pdb;pdb.set_trace()
                df.iloc[:, 2:] = df.iloc[:, 2:].div(df.drop(columns=['div_nm','pp_id']).sum(axis=1), axis=0) # percentages
                c1_m_c2_dict[col_name] = df

        # do all the fancy calculations now
        import pdb;pdb.set_trace()


        # 1-> 2. Percentage transfer

        # first m candidates the same, remaining c1 - 
        m = len(ast.literal_eval(row['m_list']))
        c1 = len(ast.literal_eval(row['c1_list']))
        c2 = len(ast.literal_eval(row['c2_list']))

        if c1 > m:
            sum_c1_extras = c1_m_c2_dict['c1_list'].iloc[:,2+m:].sum(axis=1) # sum values in row for extra c1 candidates (first 2 rows are info)
            c1_m_c2_dict['transfer_percent'] = (c1_m_c2_dict['m_list'].iloc[:,2:] - c1_m_c2_dict['c1_list'].iloc[:,2:2+m])/sum_c1_extras # must be positive
        else:
            #zero_df = c1_m_c2_dict['m_list'].iloc[:,2:].loc[:, c1_m_c2_dict['m_list'].iloc[:,2:].columns] = 0
            #c1_m_c2_dict['transfer_percent'] =zero_df
            c1_m_c2_dict['transfer_percent'] = np.nan # 
        import pdb;pdb.set_trace()

        if c2 > m:
            # separately, save proportion donated by m parties, and proportions of donation total recieved by extra c2 parties
            c1_m_c2_dict['donation_proportion'] = 1 - (c1_m_c2_dict['c2_list'].iloc[:,2:2+m] / c1_m_c2_dict['m_list'].iloc[:,2:].replace(0, float('nan'))).fillna(0) # avoid division by 0
            sum_c2_etras = c1_m_c2_dict['c2_list'].iloc[:,2+m:].sum(axis=1)
            c1_m_c2_dict['receiving_proportion'] = c1_m_c2_dict['c2_list'].iloc[:,2+6:].div(sum_c2_etras, axis=0) # of c2 extra candidates, get proportion donated to each
            
            
        else: 
            c1_m_c2_dict['donation_proportion'] = 0
            c1_m_c2_dict['receiving_proportion'] = 0
            

        # 1. Combine all state division DOPByPP together for each redistribution state
        # 2. Select wide format percentages corresponding to c1 candidates
        # 3. c1-> m transition

    return df





def whole_procedure(Formal_prefs_dict,general_party_df):
    Formal_prefs_dict, Senate_party_abvs_dict = allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict, general_party_df)

    # make list of senate parties for check if they match house ones
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) # currently off
    #import pdb;pdb.set_trace()


    #application_dict = {}
    #application_dict['Bass'] = ['JLN','GRN','LP','ON','ALP']
    #application_dict['Franklin'] = ['TLOC','ALP','JLN','LP','GRN']

    Incumbent_advantage = 0
    candidate_change_redistribution = 1

    if Incumbent_advantage:
        Final_x_House_df = pd.read_csv(f"{data_year}Final_x_for_Incumbency.csv")
        #Final_x_House_df = Final_x_House_df.loc[Final_x_House_df['div_nm'].isin(list(Formal_prefs_dict.keys())),] # extra for testing!
        Final_x_House_dict = {name: list(Final_x_House_df.loc[Final_x_House_df['div_nm'] == name, 'PartyAb']) for name in Final_x_House_df['div_nm'].unique()}


        application_dict = Final_x_House_dict

        Final_allocated_pcts_aggregated_dict = allocate_Formal_prefs_by_1234(Formal_prefs_dict, Senate_party_abvs_dict, application_dict)

        import pdb;pdb.set_trace()

        Senate_votes = pd.concat([df.melt(id_vars=["div_nm"], value_vars=df.columns[1:], var_name="PartyAb", value_name="Senate_Pct") for df in Final_allocated_pcts_aggregated_dict.values()], ignore_index=True).reset_index(drop=True)
        print(Senate_votes)
        import pdb;pdb.set_trace()
        Final_x_HS_df = pd.concat([Final_x_House_df, Senate_votes.drop(columns=['div_nm', 'PartyAb'])], axis=1)[["div_nm","PartyAb","is_incumbent","is_historic_incumbent","House_Pct","Senate_Pct"]]
        Final_x_HS_df.loc[:,"Senate_Pct"] = (pd.to_numeric(Final_x_HS_df.loc[:, "Senate_Pct"].astype(str), errors="coerce") * 100).round(2) # fixes issues with string values somehow????


        #Final_x_HS_df.to_csv(f"{data_year}Final_x_HS_df.csv", index=False)
    if candidate_change_redistribution:
        Redistribution_pair_c1_c2_lists = pd.read_csv("Redistribution_pair_c1_c2_lists2024.csv", index_col = None)
        transformed_votes = allocate_Formal_prefs_Redistribution_change(Formal_prefs_dict, Senate_party_abvs_dict, Redistribution_pair_c1_c2_lists)

    return Final_allocated_pcts_aggregated_dict, Final_x_HS_df


Final_allocated_pcts_aggregated_dict, Final_x_HS_df = whole_procedure(Formal_prefs_dict,general_party_df)



import pdb;pdb.set_trace()
