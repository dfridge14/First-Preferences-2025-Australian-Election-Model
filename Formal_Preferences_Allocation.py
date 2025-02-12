import pandas as pd
import numpy as np
import os,time

os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()


data_year = '2019'



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







# old version
#formal_prefs_full = pd.read_csv("FormalPrefs_Deakin.csv")
def allocate_Formal_preferences_to_selected_party_list(div, allocation_abvs_list):

    formal_prefs_full = pd.read_csv("FormalPrefs_Deakin.csv")
    formal_prefs = formal_prefs_full.iloc[:, 6:]



    formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
    #formal_prefs = formal_prefs.where(formal_prefs.notna(), pd.NA).astype('Int64') # convert type from floats to ints
    start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated


    group_party_names = formal_prefs_full.iloc[:, 6:].columns[:start_of_BTL_index] # include both group and party names
    party_names_list = group_party_names.str.split(':').str[-1].tolist()

    general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
    general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',] = 'GRN' # handle exceptions, but think GVIC is the only one


    party_abvs_list = []
    for party in party_names_list:
        #print(party)
        if party:
            if party == "Liberal/The Nationals": # handle LIB/NAT Exception - I think best to treat them as one party in the Senate as they always contest together, and then reverse engineer House split if needed
                party_abvs_list.append('COAL')
            elif party == " Science, Pirate, Secular, Climate Emergency": # SOPA exception
                party_abvs_list.append('SOPA')
            else:
                party_abvs_list.append(general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].iloc[0])
        else:
            party_abvs_list.append('')

    allocation_abvs_list = ['COAL','ALP','GRN']
    allocation_set = []

    for i, party in enumerate(party_abvs_list):
        if party in allocation_abvs_list:
            allocation_set.append(formal_prefs.columns[:start_of_BTL_index][i]) # add corresponding group 'letter' to allocation_set


    print(party_abvs_list)
    print(party_names_list)
    print(allocation_set)

    import pdb;pdb.set_trace()
            


    #print(start_of_BTL_index)
    #print(formal_prefs.iloc[:, start_of_BTL_index])


    first_prefs_set = formal_prefs.columns.unique().tolist()

    allocation_set = ["D","H","I","K","L","N","P","U","W"]





    def find_earliest_preference_id(preferences):
        
        # Former method - too long calculating duplicates!
        # Find the candidate with the lowest preference number in each row of df preferences 
        #votes = preferences.idxmin(axis=1, skipna=True)
        #votes[preferences.isna().all(axis=1)] = np.nan
        # is_earliest_non_unique = preferences.apply(lambda row: row[row == row.min()].count() > 1, axis=1)
        # series of either None if earliest preferences is unique, or list of non-unique candidates
        #non_unique_min_cands = preferences.apply(lambda row: list(row[row == row.min()].index) if row[row == row.min()].count() > 1 else None, axis=1)

        #votes[non_unique_min_cands.notnull()] = np.nan # set non-unique cands to nan for now
        #print(votes[non_unique_min_cands.notnull()])


        if 1 == "allocation":
            # alternative that counts all cols if duplicated, returns vote candidate(s) as list
            min_values = preferences.min(axis=1)
            mask = preferences.eq(min_values, axis=0)
            earliest_cands = mask.apply(lambda row: row.index[row].tolist(), axis=1) # creates list of 1 or more candidates

            non_unique_min_cands = earliest_cands.apply(lambda x: x if len(x)>=2 else np.nan)

            # set non-decided values to nan
            earliest_cands[earliest_cands.apply(len)==0] = np.nan
            earliest_cands[non_unique_min_cands.notnull()] = np.nan

        # alternative to alternative: get indices where there may be duplication

        votes = preferences.idxmin(axis=1, skipna=True) # min in row
        votes[preferences.isna().all(axis=1)] = np.nan # no prefs in row

        min_values = preferences.min(axis=1)
        mask = preferences.eq(min_values, axis=0)
        multiple_min_mask = mask.sum(axis=1) > 1 # series with True when there are multiple minimum preferences - set as nan and deal with later

        votes[multiple_min_mask] = np.nan

        return votes, multiple_min_mask   # Return NaN in votes if no preference for the candidate set


    #formal_prefs_combined = pd.concat([formal_prefs.iloc[:,:start_of_BTL_index], combine_BTL_preferences(formal_prefs, start_of_BTL_index)], axis=1)
    #print(formal_prefs_combined)



    allocation_set = ["D","H","I","K","L","N","P","U","W"]

    def allocate_votes(df, allocation_set, allocation_type): 
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
        
        if allocation_type == "allocation": # if actually performing allocation, handle duplicates by adding index to list
            ATL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & ATL_non_unique_min
            BTL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & ~ATL_duplicates_series & BTL_non_unique_min # duplicate but not in ATL

            # collect indices 
            duplicates = ATL_duplicates_series+BTL_duplicates_series
            duplicate_indices = duplicates.loc[duplicates].index # return indices where duplicates == True

            #import pdb;pdb.set_trace()
            
            # ATL is duplicate, but BTL not decisive - is this even formal (maybe, if duplicate is 13th on BTL). Set vote to ATL_non_unique_min
            #ATL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & ATL_non_unique_min.notnull() # not yet allocated, but duplicate ATL
            #allocated_votes.loc[:,'Vote'] = allocated_votes.loc[:,'Vote'].fillna(ATL_non_unique_min[ATL_duplicates_series])

            # Only BTL duplicates, set vote to BTL_non_unique_min_cands
            #BTL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & BTL_non_unique_min.notnull() # not yet allocated, but duplicate BTL
            #allocated_votes.loc[:,'Vote'] = allocated_votes.loc[:,'Vote'].fillna(BTL_non_unique_min[BTL_duplicates_series])
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


    def allocate_formal_preferences_to_allocation_set(formal_prefs, allocation_set):

        # still yet to figure out how to go between candidates and groups in allocation set!!!!!!!!!!!!!!!!!!!!!!!!!


        Final_allocated_votes = pd.DataFrame(index=formal_prefs.index, columns=allocation_set, data = 0) # df of allocated votes for each candidate

        # fix BTL into single group
        start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated
        ATL = formal_prefs.iloc[:,:start_of_BTL_index]
        BTL = formal_prefs.iloc[:,start_of_BTL_index:]
        #BTL_voted = BTL.loc[BTL.sum(axis=1) > 0].groupby(BTL.columns, axis=1).min() # try only for potential BTL votes, but seems quick enough as is
        BTL = BTL.groupby(BTL.columns, axis=1).min()

        formal_prefs_by_group = pd.concat([ATL, BTL], axis=1)


        # allocate first preferences using formal_prefs
        first_pref_allocated_votes, redundant = allocate_votes(formal_prefs_by_group, first_prefs_set, "First_Preferences")
        formal_prefs_allocated_first_prefs = pd.concat([formal_prefs_by_group, first_pref_allocated_votes], axis=1) # add allocated first_pref vote to formal_prefs_by_group df
        Final_allocated_votes["First_Preferences"] = first_pref_allocated_votes


        # building groups based on first preference vote, either allocate vote directly to allocation_set if one of them, else allocate by group among later preferences

        for party, formal_subsection in formal_prefs_allocated_first_prefs.groupby("Vote"):
            
            if party in allocation_set:
                Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party, party] = 1 # put in a 1 into the party column while preserving index

            else:
                allocated_votes_subsection, duplicate_indices_subsection = allocate_votes(formal_subsection, allocation_set, "allocation")
                Subsection_final_votes = Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party] # just working with this subsection

                # Add allocation preferences where clear
                mask = pd.get_dummies(allocated_votes_subsection.loc[allocated_votes_subsection["Vote"].notna(), "Vote"])
                mask = mask.reindex(Subsection_final_votes.index, fill_value=0)     # Align the mask with the indices of Subsection_final_votes
                Subsection_final_votes.loc[:, mask.columns] = mask         # Update Subsection_final_votes with the mask

                # Add clear allocation preferences
                #Final_allocated_votes.loc[(Final_allocated_votes["First_Preferences"] == party) & (allocated_votes_subsection["Vote"].notna()), allocated_votes_subsection["Vote"]] = 1 # add decided votes - ------- make sure only if not na

                # Add duplicate preferences 
                duplicate_for_party_df = formal_subsection[formal_subsection.index.isin(duplicate_indices_subsection)].copy()

                # iteratively add duplicate votes proportionate to # of cnadidates duplicated
                duplicate_for_party_df["Vote"] = allocate_votes_duplicates(duplicate_for_party_df, allocation_set) # get series of candidates for each duplicate votes
                for row in duplicate_for_party_df.index:
                    duplicate_vote_list = duplicate_for_party_df.loc[duplicate_for_party_df.index == row,"Vote"].iloc[0] # iloc makes it a list
                    for vote in duplicate_vote_list:
                        Subsection_final_votes.loc[Subsection_final_votes.index==row, vote] = 1/len(duplicate_vote_list)
                


                Subsection_final_votes = Subsection_final_votes.drop(columns=['First_Preferences'])
                Party_preferences_proportions = Subsection_final_votes.sum() / np.sum(Subsection_final_votes.sum()) # row of proportions
                mask = allocated_votes_subsection["Vote"].isna() & ~allocated_votes_subsection.index.isin(duplicate_indices_subsection)
                Subsection_final_votes.loc[mask] = [Party_preferences_proportions.values] * mask.sum()
                #Subsection_final_votes.loc[allocated_votes_subsection["Vote"].isna() & ~allocated_votes_subsection.index.isin(duplicate_indices_subsection), ] = Party_preferences_proportions # if nan but not duplicate



                Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party,Final_allocated_votes.columns[:-1]] = Subsection_final_votes # fill out full table


        return Final_allocated_votes.drop(columns = "First_Preferences")



    Deakin_senate_df = allocate_formal_preferences_to_allocation_set(formal_prefs, allocation_set) # done!

    print(time.time() - start, "seconds")
    import pdb;pdb.set_trace()

    return Deakin_senate_df





























Formal_prefs_dict = {}
states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
for state in states: # currently only 2016 onwards
    filename = f"{data_year}FormalPrefs{state}.csv"

    curr_Formal_prefs = pd.read_csv(filename).rename(columns = {"Division": "div_nm"})

    # Not enough memory --> downcast floats to lower order for numeric columns
    state_div_Formal_prefs_dict = {div: group.reset_index(drop=True).apply(
        lambda col: pd.to_numeric(col, downcast='float', errors='ignore') if pd.api.types.is_numeric_dtype(col) else col
    ) for div, group in curr_Formal_prefs.groupby("div_nm")} 
    for key, group in state_div_Formal_prefs_dict.items():
        Formal_prefs_dict[key] = group # assumes no keys (divs) overlap for different states :)

    print("done", time.time() - start)


# TO DO:
# 1. Fix up indexing so each element of dict starts with 0                          DONE
# 2. First preferences allocate - useful for all!!!                                 DONE
# 3. get ordered list of 5 parties from Incumbent Advantage csv                     DONE
# 4. For each div allocate to specific one                                          DONE
# 5. aggregate into whole or by pp_id                                               DONE
# 6. Use PollingPlacesRepository for correspondence between names and pp_id
# 7. write to csv of senate prefs
# 8. For 3. Make list of all top 5 that match senate: Senate_Party_Abs per division DONE
        

Final_x_House_df = pd.read_csv(f"{data_year}Final_x_for_Incumbency.csv")
#Final_x_House_df = Final_x_House_df.loc[Final_x_House_df['div_nm'].isin(list(Formal_prefs_dict.keys())),] # extra for testing!
Final_x_House_dict = {name: list(Final_x_House_df.loc[Final_x_House_df['div_nm'] == name, 'PartyAb']) for name in Final_x_House_df['div_nm'].unique()}




general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',] = 'GRN' # handle exceptions, but think GVIC is the only one


def find_earliest_preference_id(preferences):
    # get indices where there may be duplication, store in multiple_min_mask
    # input: df with unique alphabetical column names and integer or nan values

    votes = preferences.idxmin(axis=1, skipna=True) # min in row
    votes[preferences.isna().all(axis=1)] = np.nan # no prefs in row

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
        BTL = BTL.groupby(BTL.columns, axis=1).min()
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







def allocate_formal_preferences_to_allocation_set(formal_prefs_div, allocation_set):

    Final_allocated_votes = pd.DataFrame(index=formal_prefs_div.index, columns=allocation_set, data = 0) # df of allocated votes for each candidate, should preserve order of df
    Final_allocated_votes["First_Preferences"] = formal_prefs_div["Vote"]

    # building groups based on first preference vote, either allocate vote directly to allocation_set if one of them, else allocate by group among later preferences

    for party, formal_subsection in formal_prefs_div.iloc[:,2:].groupby("Vote"): # ignore first 2 rows in calculation (div/pp)
        
        if party in allocation_set:
            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party, party] = 1 # put in a 1 into the party column while preserving index

        else:
            allocated_votes_subsection, duplicate_indices_subsection = allocate_votes(formal_subsection, allocation_set)
            Subsection_final_votes = Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party] # just working with this subsection

            # Add allocation preferences where clear
            mask = pd.get_dummies(allocated_votes_subsection.loc[allocated_votes_subsection["Vote"].notna(), "Vote"])
            mask = mask.reindex(Subsection_final_votes.index, fill_value=0)     # Align the mask with the indices of Subsection_final_votes
            Subsection_final_votes.loc[:, mask.columns] = mask         # Update Subsection_final_votes with the mask

            # Add duplicate preferences 
            duplicate_for_party_df = formal_subsection[formal_subsection.index.isin(duplicate_indices_subsection)].copy()

            # iteratively add duplicate votes proportionate to # of cnadidates duplicated
            duplicate_for_party_df["Vote"] = allocate_votes_duplicates(duplicate_for_party_df, allocation_set) # get series of candidates for each duplicate votes
            
            for row in duplicate_for_party_df.index:
                duplicate_vote_list = duplicate_for_party_df.loc[duplicate_for_party_df.index == row,"Vote"].iloc[0] # iloc makes it a list
                for vote in duplicate_vote_list:
                    Subsection_final_votes.loc[Subsection_final_votes.index==row, vote] = 1/len(duplicate_vote_list)
            

            # handle remainingg nan values - assign votes proportional to how rest of their subsection voted
            Subsection_final_votes = Subsection_final_votes.drop(columns=['First_Preferences'])
            Party_preferences_proportions = Subsection_final_votes.sum() / np.sum(Subsection_final_votes.sum()) # row of proportions
            mask = allocated_votes_subsection["Vote"].isna() & ~allocated_votes_subsection.index.isin(duplicate_indices_subsection)
            #import pdb;pdb.set_trace()
            Subsection_final_votes.loc[mask] = pd.DataFrame([Party_preferences_proportions.values] * sum(mask)) # changed from mask.sum()

            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party,Final_allocated_votes.columns[:-1]] = Subsection_final_votes # fill out full table



    Final_allocated_votes_df = pd.concat([formal_prefs_div.iloc[:,:2], Final_allocated_votes], axis=1).drop(columns = "First_Preferences") # return 1st 3 cols & remove last


    # This is where to return in the pp_id column 
    #Final_allocated_votes_aggregated_df = Final_allocated_votes_df.groupby(["div_nm", "Vote Collection Point Name"], as_index=False).sum()
    Final_allocated_votes_aggregated_df = Final_allocated_votes_df.drop(columns = ["Vote Collection Point Name"]).groupby(["div_nm"], as_index=False).sum()

    #import pdb;pdb.set_trace()

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
        Final_allocated_pcts_aggregated_dict[div].iloc[:,1:] = Final_allocated_pcts_aggregated_dict[div].iloc[:,1:]/Final_allocated_pcts_aggregated_dict[div].drop(columns=['div_nm']).sum(axis=1).iloc[0]  

    return Final_allocated_pcts_aggregated_dict





def whole_procedure(Formal_prefs_dict,general_party_df):
    Formal_prefs_dict, Senate_party_abvs_dict = allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict, general_party_df)

    # make list of senate parties for check if they match house ones
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    #Senate_parties_by_div.to_csv("Senate_parties_by_div.csv", index=False) # currently off
    #import pdb;pdb.set_trace()


    #application_dict = {}
    #application_dict['Bass'] = ['JLN','GRN','LP','ON','ALP']
    #application_dict['Franklin'] = ['TLOC','ALP','JLN','LP','GRN']

    application_dict = Final_x_House_dict

    Final_allocated_pcts_aggregated_dict = allocate_Formal_prefs_by_1234(Formal_prefs_dict, Senate_party_abvs_dict, application_dict)

    import pdb;pdb.set_trace()

    return Final_allocated_pcts_aggregated_dict



Final_allocated_pcts_aggregated_dict = whole_procedure(Formal_prefs_dict,general_party_df)

Senate_votes = pd.concat([df.melt(id_vars=["div_nm"], value_vars=df.columns[1:], var_name="PartyAb", value_name="Senate_Pct") for df in Final_allocated_pcts_aggregated_dict.values()], ignore_index=True).reset_index(drop=True)
print(Senate_votes)
import pdb;pdb.set_trace()
Final_x_HS_df = pd.concat([Final_x_House_df, Senate_votes.drop(columns=['div_nm', 'PartyAb'])], axis=1)[["div_nm","PartyAb","is_incumbent","is_historic_incumbent","House_Pct","Senate_Pct"]]
Final_x_HS_df.loc[:,"Senate_Pct"] = (pd.to_numeric(Final_x_HS_df.loc[:, "Senate_Pct"].astype(str), errors="coerce") * 100).round(2) # fixes issues with string values somehow????
#Final_x_House_df.merge(Senate_votes, on = ['div_nm','PartyAb'], how = 'left')


Final_x_HS_df.to_csv(f"{data_year}Final_x_HS_df.csv", index=False)




# find who didnt preference any i.e. NA_set
#NA_set = formal_prefs_allocated.loc[formal_prefs_allocated["Vote"].isna(),:]
#first_votes = pd.DataFrame(index=NA_set.index, columns=['1st'])
#first_votes.loc[:,'1st'] = find_earliest_preference_id(NA_set.iloc[:,:start_of_BTL_index])
#NA_set_BTL = NA_set.iloc[:,start_of_BTL_index:]
#first_votes.loc[:,'1st'] = first_votes.loc[:,'1st'].fillna(find_earliest_preference_id(NA_set_BTL.groupby(NA_set_BTL.columns, axis=1).min()))
##cs = first_votes.value_counts(dropna=False)
# check if 2nd level NaNs occur - they should not!
#NA_set_1st_prefs = pd.concat([NA_set, first_votes], axis=1)
