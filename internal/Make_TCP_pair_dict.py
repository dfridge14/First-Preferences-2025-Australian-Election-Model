import pandas as pd
import numpy as np
import pickle 
import os
from pathlib import Path
from collections import defaultdict

# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


base_dir = Path.home() / "Australian Election"
os.chdir(base_dir)


data_year = '2022'
START_OF_PREFS = 2 # Prefs begin on the 3th column (after div_nm,pp_nm) - deleted stateab to accomodate 2016 file

NAME_CHANGES_YEAR_DICT = {'2025':{},'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
NEW_SEATS_YEAR_DICT = {'2025':[],'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner','Gorton']}
BTL_ONLY_ELECTIONS = ['2007','2010','2013']


TCP_COMBINATION_INDEX = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6,  \
                                ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14}






def make_party_category_dict():

        all_parties = pd.read_csv('Grand_Party_Category_df_2004_2022.csv', index_col=None)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['CLR'],'Ideo_Category':['ALP'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['NGS'],'Ideo_Category':['Right'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['ARTS'],'Ideo_Category':['Left'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties_house = all_parties.loc[all_parties['Ideo_Category'].notna(),].iloc[:,:2].set_index('PartyAb') # excludes only senates, who don't yet have Ideology written
        party_category_dict = all_parties_house.to_dict()['Ideo_Category']
        party_category_dict['IND'] = 'Centre'
        party_category_dict['COALLP'] = 'COAL'
        party_category_dict['COALNP'] = 'COAL'

        return party_category_dict



def find_earliest_preference_id(preferences):
    # get indices where there may be duplication, store in multiple_min_mask
    # input: df with unique alphabetical column names and integer or nan values

    #votes = preferences.idxmin(axis=1, skipna=True)
    #votes[preferences.isna().all(axis=1)] = np.nan 

    #votes = preferences.astype("float32") # return to float32!
    # votes = votes.fillna(float("inf")).idxmin(axis=1) # min in row, avoids warning
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

    #print("done", time.time() - start)

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


def allocate_votes_2007_2013(df, allocation_set):
    ### amended version of allocate_votes that accounts for the pre-formatted structure of sampled BTL Formal Preferences 
    allocated_votes = pd.DataFrame(index=df.index, columns=['Vote'])

    BTL_preferences = df[allocation_set]     # Filter the row to only include the candidates of interest
    allocated_votes.loc[:,'Vote'], BTL_non_unique_min = find_earliest_preference_id(BTL_preferences)
    duplicates = allocated_votes.loc[:,'Vote'].isna() & BTL_non_unique_min
    duplicate_indices = duplicates.loc[duplicates].index # return indices where duplicates == True

    return allocated_votes, duplicate_indices

def allocate_votes_duplicates_2007_2013(df, allocation_set):

    # these will often be identical duplicates due to sampling with replacement
    
    BTL_dup_preferences = df[allocation_set]     # Filter the row to only include the candidates of interest
    duplicate_votes_series = BTL_dup_preferences.apply(lambda row: list(row[row == row.min()].index), axis = 1)

    #if not duplicate_votes_series.empty:
        #import pdb;pdb.set_trace()

    return duplicate_votes_series

# get correpondence between Booth name and pp_id for given year
Polling_Places_df = pd.read_csv(f"{data_year}GeneralPollingPlaces.csv", index_col = None, skiprows = 1)
Polling_Places_df = Polling_Places_df.iloc[:,2:6].rename(columns={'DivisionNm': 'div_nm','PollingPlaceID': 'pp_id','PollingPlaceNm':'pp_nm'})
Polling_Places_df = Polling_Places_df.loc[Polling_Places_df['PollingPlaceTypeID'].isin([1,5]),].drop('PollingPlaceTypeID', axis=1)

for div in Polling_Places_df['div_nm'].unique().tolist():
    Other_row = pd.DataFrame({"div_nm": [div],"pp_id":[0],"pp_nm":["Other"]})
    Polling_Places_df = pd.concat([Polling_Places_df,Other_row], ignore_index=True)

Booth_name_pp_id = Polling_Places_df

Booth_name_pp_id['div_nm'] = Booth_name_pp_id['div_nm'].replace(NAME_CHANGES_YEAR_DICT[data_year])



def allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_div, allocation_set, by_pp_id = False, as_percent = True):


    Final_allocated_votes = pd.DataFrame(index=Formal_prefs_div.index, columns=allocation_set, data = 0.0) # df of allocated votes for each candidate, should preserve order of df
    Final_allocated_votes["First_Preferences"] = Formal_prefs_div["Vote"]

    # building groups based on first preference vote, either allocate vote directly to allocation_set if one of them, else allocate by group among later preferences

    for party, formal_subsection in Formal_prefs_div.iloc[:,START_OF_PREFS:].groupby("Vote"): # ignore first 2 rows in calculation (div/pp)
        
        if party in allocation_set:
            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party, party] = 1.0 # put in a 1 into the party column while preserving index

        else:
            if data_year in BTL_ONLY_ELECTIONS:
                allocated_votes_subsection, duplicate_indices_subsection = allocate_votes_2007_2013(formal_subsection, allocation_set)
            else:
                allocated_votes_subsection, duplicate_indices_subsection = allocate_votes(formal_subsection, allocation_set)

            Subsection_final_votes = Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party] # just working with this subsection

            #import pdb;pdb.set_trace()

            # Add allocation preferences where clear
            mask = pd.get_dummies(allocated_votes_subsection.loc[allocated_votes_subsection["Vote"].notna(), "Vote"])
            mask = mask.reindex(Subsection_final_votes.index, fill_value=0)     # Align the mask with the indices of Subsection_final_votes
            Subsection_final_votes.loc[:, mask.columns] = mask.astype(float)         # Update Subsection_final_votes with the mask

            # Add duplicate preferences 
            duplicate_for_party_df = formal_subsection[formal_subsection.index.isin(duplicate_indices_subsection)].copy()

            # iteratively add duplicate votes proportionate to # of candidates duplicated - if there are any duplicates!
            if not duplicate_for_party_df.empty:
                if data_year in BTL_ONLY_ELECTIONS:
                    duplicate_for_party_df["Vote"] = allocate_votes_duplicates_2007_2013(duplicate_for_party_df, allocation_set)
                else:
                    duplicate_for_party_df["Vote"] = allocate_votes_duplicates(duplicate_for_party_df, allocation_set) # get series of candidates for each duplicate votes
                
                Subsection_final_votes = Subsection_final_votes.astype({col: "float64" for col in Subsection_final_votes.columns[:-1]})

                for row in duplicate_for_party_df.index:
                    duplicate_vote_list = duplicate_for_party_df.loc[duplicate_for_party_df.index == row,"Vote"].iloc[0] # iloc makes it a list
                    for vote in duplicate_vote_list:
                        Subsection_final_votes.loc[Subsection_final_votes.index==row, vote] = 1/len(duplicate_vote_list)

            #import pdb;pdb.set_trace()
            

            # handle remaining nan values - assign votes proportional to how rest of their subsection voted
            Subsection_final_votes = Subsection_final_votes.drop(columns=['First_Preferences'])
            Party_preferences_proportions = Subsection_final_votes.sum() / np.sum(Subsection_final_votes.sum()) # row of proportions
            mask = allocated_votes_subsection["Vote"].isna() & ~allocated_votes_subsection.index.isin(duplicate_indices_subsection)
            Subsection_final_votes.loc[mask] = pd.DataFrame([Party_preferences_proportions.values] * sum(mask), index=Subsection_final_votes.index[mask], columns=Subsection_final_votes.columns) # changed from mask.sum()

            #Final_allocated_votes.iloc[:, :-1] = Final_allocated_votes.iloc[:, :-1].astype(float)
            if Final_allocated_votes.columns.duplicated().any():
                import pdb;pdb.set_trace()
                print("Warning: Duplicate column names found!")
                Final_allocated_votes = Final_allocated_votes.loc[:, ~Final_allocated_votes.columns.duplicated()]  # Drop duplicates

            # Select numeric columns correctly
            numeric_columns = Final_allocated_votes.select_dtypes(include=['number']).columns  

            # Convert the selected numeric columns to float (ensuring proper data types) - CLEAN UP - THIS SHOULDN'T HAVE TO BE DONE EVERY TIME!!!
            Final_allocated_votes[numeric_columns] = Final_allocated_votes[numeric_columns].astype(float)

            #import pdb;pdb.set_trace()

            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party,Final_allocated_votes.columns[:-1]] = Subsection_final_votes.values # fill out full table


    Final_allocated_votes_df = pd.concat([Formal_prefs_div.iloc[:,:START_OF_PREFS], Final_allocated_votes], axis=1).drop(columns = "First_Preferences") # return 1st 3 cols & remove last


    # This is where to return in the pp_id column 
    if not by_pp_id:
        Final_allocated_votes_aggregated_df = Final_allocated_votes_df.drop(columns = ["pp_nm"]).groupby(["div_nm"], as_index=False).sum() # group all together
        #import pdb;pdb.set_trace()
    
    else:
        Final_allocated_votes_aggregated_df = Final_allocated_votes_df.groupby(["div_nm", "pp_nm"], as_index=False).sum()

        # GROUP THE STARTSWITH ABSENT,PREPOLL,POSTAL,PROVISIONAL,EAV,REMOTEMT,SPECIALMT,OTHERMT TOGETHER WITH PP_ID 0, THE REST MERGE WITH PP_IDS
        Other_booth_type_prefixes = ['Remote Mobile', 'Other Mobile','Special Hospital','EAV','ABSENT','PROVISIONAL','PRE_POLL','POSTAL']

        Final_allocated_votes_aggregated_df.loc[:,"pp_nm"] = Final_allocated_votes_aggregated_df.loc[:,"pp_nm"].apply(lambda x: 'Other' if any(x.startswith(prefix) for prefix in Other_booth_type_prefixes) else x)
        Final_allocated_votes_aggregated_df = Final_allocated_votes_aggregated_df.groupby(["div_nm", "pp_nm"], as_index=False).sum() # group again

        # switch pp_nm to pp_id
        Final_allocated_votes_aggregated_df = pd.merge(Final_allocated_votes_aggregated_df, Booth_name_pp_id, on = ['div_nm','pp_nm'], how='left')
        Final_allocated_votes_aggregated_df.loc[:,'pp_nm'] = Final_allocated_votes_aggregated_df.loc[:,'pp_id']

        Final_allocated_votes_aggregated_df.drop(columns=['pp_id'], inplace=True)
        Final_allocated_votes_aggregated_df.rename(columns={"pp_nm":"pp_id"}, inplace=True)

    if as_percent:
        Final_allocated_votes_aggregated_df.iloc[:, 1+by_pp_id:] = Final_allocated_votes_aggregated_df.iloc[:, 1+by_pp_id:].div(Final_allocated_votes_aggregated_df.drop(columns=['div_nm','pp_id'], errors='ignore').sum(axis=1), axis=0)


    return Final_allocated_votes_aggregated_df


def convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div):
    ### convert allocation_abvs into Senate Group letters
    #import pdb;pdb.set_trace()

    allocation_set = []
    # Goal is to preserve order of allocation_abvs_list in allocation_set
    for party in allocation_abvs_list:  # Iterate through allocation_abvs_list directly
        if party in Senate_party_abvs_dict[div]: 
            i = Senate_party_abvs_dict[div].index(party) # Find the index of the party in this div and use it to get the corresponding Senate Group name
            allocation_set.append(Formal_prefs_dict[div].columns[START_OF_PREFS:START_OF_PREFS+len(Senate_party_abvs_dict[div])][i])  # Append the corresponding group 'letter'
    return allocation_set


def get_senate_TCP(Formal_prefs_dict, Senate_party_abvs_dict, div, tcp_pair_parties, party):

    allocation_abvs_list = [p for p in tcp_pair_parties]

    allocation_set = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div)
    party_to_allocate = convert_partyab_to_senate_group_names([party], Formal_prefs_dict, Senate_party_abvs_dict, div)

    TCP_senate = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[div].loc[Formal_prefs_dict[div]['Vote'] == party_to_allocate[0],], allocation_set, by_pp_id = False, as_percent = True).iloc[0,1]

    return TCP_senate
    

def add_category_pct(row, party_category_dict):
    # Convert row to dict
    party_dict =  {k: (next(iter(v.values())) if isinstance(v, dict) else v) for k, v in row.to_dict().items()} # convert df row to a dict
    
    # Compute category averages
    category_values = defaultdict(list)
    for party, pct in party_dict.items():
        category = party_category_dict.get(party)
        if category in ['Left', 'Right', 'Centre']:
            if pd.notna(pct):
                category_values[category].append(pct)
    
    # Add category averages to row
    for category, values in category_values.items():
        row[category] = sum(values) / len(values)

    return row


def make_TCP_pair_category_dict(election_year, party_category_dict={}):

    from collections import defaultdict


    data_year = str(int(election_year) - 3)
    

    name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}

    replacement_seats_year_dict = {'2022': {'Hasluck':'Bullwinkel'}, '2019':{'Gorton':'Hawke'}, '2016':{'Canberra':'Bean', 'Maribyrnong':'Fraser'}, '2013':{'Hasluck':'Burt'}}
    abolished_divs_dict = {'2022':set(['Higgins','North Sydney']), '2016': set(['Port Adelaide']),'2019':set(['Stirling']),'2013':set(['Charlton'])}


    next_year = election_year
    # 1. Get names of next election's parties in each div for comparison to senate
    if next_year != '2025':

        DOP_By_Division_next = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'})[["div_nm","PartyAb"]].drop_duplicates()

    else:
        DOP_By_Division_next = pd.read_csv("2025Candidates_By_Division.csv", index_col = None)
    
    DOP_By_Division_next.loc[:,'PartyAb'] = DOP_By_Division_next.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
    Div_parties_next_dict = {div: group['PartyAb'].tolist() for div, group in DOP_By_Division_next.groupby("div_nm")}

    Div_parties_next_dict_COAL = {div: ['COAL' if p in ['LP', 'NP','CLP','LNP'] else p for p in Div_parties_next_dict[div]] for div in Div_parties_next_dict.keys()}


    TCP_Preference_Flows = pd.read_csv(f"{data_year}HouseTCPFlowByDivision.csv", skiprows = 1, index_col = None).rename(columns = {'DivisionNm':'div_nm','FromCandidatePartyAb':'PartyAb', \
                                                'FromCandidateBallotPosition':'Ballot_Position','ToCandidatePartyAb':'TCP_Ab','ToCandidateBallotPosition':'TCP_Ballot_Position'})
    TCP_Preference_Flows = TCP_Preference_Flows[['div_nm','PartyAb','Ballot_Position','TCP_Ab','TCP_Ballot_Position','TransferPercentage']]
    TCP_Preference_Flows = TCP_Preference_Flows.loc[TCP_Preference_Flows['Ballot_Position']>0,]
    TCP_Preference_Flows = TCP_Preference_Flows[['div_nm','PartyAb','TCP_Ab','TransferPercentage']]
    TCP_Preference_Flows['div_nm'] = TCP_Preference_Flows['div_nm'].replace(name_changes_year_dict[data_year])

    TPP_by_State = pd.read_csv(f"{data_year}HouseTPPFlowByStateByParty.csv", skiprows = 1, index_col = None)
    TPP_by_State = TPP_by_State[['StateAb', 'PartyAb', 'Australian Labor Party Transfer Percentage']]
    TPP_by_State = TPP_by_State.loc[(TPP_by_State['PartyAb'].notna()) & (TPP_by_State['PartyAb'] != 'NAFD'), ].rename(columns={'Australian Labor Party Transfer Percentage':'ALP%'})

    TPP_nationally = pd.read_csv(f"{data_year}HouseTPPFlowByParty.csv", skiprows = 1, index_col = None)[['PartyAb', 'Australian Labor Party Transfer Percentage']].rename(columns={'Australian Labor Party Transfer Percentage':'ALP%'})
    
    if next_year != '2025':
        div_to_state = pd.read_csv(f"{next_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})

    else:
        div_to_state = pd.read_csv(f"2022HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        div_to_state.loc[div_to_state['div_nm'] == 'North Sydney',] = 'Bullwinkel', 'WA'
        div_to_state = div_to_state.loc[~(div_to_state['div_nm'] == 'Higgins'),]

    div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}




    # 1. Get PartyAb: 1st alphabetically for each pair:

    # add TCP to dict!

    Preference_flows_dict = {}
    Non_classic_divs = defaultdict(list)

    def normalize_party(p):
        return 'COAL' if p in ['LP', 'NP'] else p

    def sorted_tcp_pair(tcp1, tcp2):
        return tuple(sorted([normalize_party(tcp1), normalize_party(tcp2)]))


    # Step 1: Precompute the sorted and COALified 2CP pair per division
    tcp_pairs_by_div = (
        TCP_Preference_Flows.groupby("div_nm")["TCP_Ab"]
        .unique()
        .apply(lambda x: sorted_tcp_pair(*x))
    )

    # remember actual party for senate comparison - the non-majot party in non-classics, and the COAL party in each div
    major = {'ALP', 'COAL', 'LNP', 'CLP'}
    COAL_parties_list = ['COAL','LNP','CLP','LP','NP'] # for inter-state matching
    non_ind_tcp_pairs_by_div = tcp_pairs_by_div[(~tcp_pairs_by_div.isin([('ALP','COAL'),('ALP','LNP'),('ALP','CLP')]) &  ~tcp_pairs_by_div.apply(lambda x: 'IND' in x))]
    Non_classic_div_TCP_parties = {div: next(p for p in pair if p not in major) for div, pair in non_ind_tcp_pairs_by_div.items() if any(p not in major for p in pair)}

    COAL_parties_div = {div: next(p for p in pair if p in COAL_parties_list) for div, pair in tcp_pairs_by_div.items() if any(p in COAL_parties_list for p in pair)}
    # SA, WA, TAS,ACT - make COAL into LP (only issue is 2013 WA LP-NP contests, but they will not be used for Senate matching!)

    # converting this all to just a senate party name per state
    for div, state in div_to_state_dict.items():
        if state in {'SA','WA','TAS','ACT'} : # and div in COAL_parties_div
            COAL_parties_div[div] = 'LP'
        elif state in {'VIC','NSW'}:
            COAL_parties_div[div] = 'COAL'
        elif state == 'QLD':
            COAL_parties_div[div] = 'LNP'
        elif state == 'NT':
            COAL_parties_div[div] = 'CLP'
    print(non_ind_tcp_pairs_by_div)

    # Step 2: Build the desired structure
    for _, row in TCP_Preference_Flows.iterrows():
        div = row['div_nm']
        party = row['PartyAb']
        tcp = normalize_party(row['TCP_Ab'])
        pct = row['TransferPercentage']

        if div in abolished_divs_dict[data_year]:
            continue
        
        tcp_pair = tuple(sorted([p if p not in ['LP','NP','LNP','CLP'] else 'COAL' for p in tcp_pairs_by_div[div]]))

        if tcp_pair != ('ALP', 'COAL'):
            # we have a Non-classic contest!

            tcp_pair = tuple(sorted([party_category_dict[p] if party_category_dict[p] != 'Centre' else 'IND' for p in tcp_pair]))

            if tcp_pair == ('ALP', 'COAL'):
                import pdb;pdb.set_trace() # should not happen

            if div not in Non_classic_divs[tcp_pair]:
                Non_classic_divs[tcp_pair].append(div)

        first, second = tcp_pair  # alphabetical order
        # We want % transferred to the *first* in alphabetical order
        if party_category_dict[tcp] == first:
            transfer_pct = pct
        else:
            transfer_pct = 100 - pct

        Preference_flows_dict.setdefault(div, {})

        party = party if party not in ['LP','NP','LNP','CLP'] else 'COAL'

        if party in Div_parties_next_dict_COAL[div]:
            Preference_flows_dict[div][party] = {tcp_pair: transfer_pct}

        if div in replacement_seats_year_dict[data_year].keys():
            new_div = replacement_seats_year_dict[data_year][div]
            Preference_flows_dict.setdefault(new_div, {})
            if party in Div_parties_next_dict_COAL[new_div]:
                Preference_flows_dict[new_div][party] = {tcp_pair: transfer_pct}



    #import pdb;pdb.set_trace()

    from collections import defaultdict

    # Assume `result` and `party_categories` are already defined

    # Make a new version of result with category rollups
    Preference_flows_dict_with_categories = {}

    for div, party_dict in Preference_flows_dict.items():
        category_values = defaultdict(list)
        div_result = dict(party_dict)  # copy original party entries

        # Get the TCP pair from any of the entries (they all have the same)
        tcp_pair = next(iter(next(iter(party_dict.values())).keys()))

        for party, tcp_data in party_dict.items():
            category = party_category_dict.get(party)
            if category in ['Left', 'Right', 'Centre']:
                pct = tcp_data[tcp_pair]
                category_values[category].append(pct)

        

        # Compute category averages and add them to the division result
        for category, values in category_values.items():
            avg_pct = sum(values) / len(values)
            div_result[category] = {tcp_pair: avg_pct}

        Preference_flows_dict_with_categories[div] = div_result


    ######### Now 2PP values if missing - extend to all categories!

    # STEP 1 — Fill in missing PARTY entries by div
    for div, state in div_to_state_dict.items():
        if div not in Preference_flows_dict_with_categories: # should be redundant!
            Preference_flows_dict_with_categories[div] = {}
        
        for _, row in TPP_by_State[TPP_by_State['StateAb'] == state].iterrows():
            party = row['PartyAb']
            percent_to_alp = row['ALP%']

            if party in ['LP','NP']:
                continue
            
            # Add the party only if not already present
            if party in Div_parties_next_dict_COAL[div]:
                if party not in Preference_flows_dict_with_categories[div]:
                    Preference_flows_dict_with_categories[div][party] = {}
                #import pdb;pdb.set_trace()

                if ('ALP', 'COAL') not in Preference_flows_dict_with_categories[div][party]:
                    Preference_flows_dict_with_categories[div][party][('ALP', 'COAL')] = percent_to_alp


    state_category_values = defaultdict(lambda: defaultdict(list))

    # STEP 2 — Fill in missing PARTY entries by div
    for _, row in TPP_by_State.iterrows():
        state = row['StateAb']
        party = row['PartyAb']
        percent_to_alp = row['ALP%']
        
        category = party_category_dict.get(party)
        if category and category not in ['ALP', 'COAL']:  # Skip major parties
            state_category_values[state][category].append(percent_to_alp)

    # STEP 3 — Fill in missing category averages per div
    for div, state in div_to_state_dict.items():
        #if div not in Preference_flows_dict_with_categories:
        #    continue

        subdict = Preference_flows_dict_with_categories[div]

        for category in ['Left', 'Right', 'Centre']:
            if category not in subdict:
                values = state_category_values[state].get(category)
                if values:
                    median_value = np.median(values)
                    subdict.setdefault(category, {})[('ALP', 'COAL')] = median_value
            else:
                if ('ALP','COAL') not in subdict[category]:
                    values = state_category_values[state].get(category)
                    if values:
                        median_value = np.median(values)

                        subdict[category][('ALP','COAL')] = median_value

    for div in Preference_flows_dict_with_categories.keys():
        for _, row in TPP_nationally.iterrows():

            party = row['PartyAb']
            percent_to_alp = row['ALP%']

            if party in ['LP','NP']:
                    continue
                
            # Add the party only if not already present
            if party in Div_parties_next_dict_COAL[div]:
                if party not in Preference_flows_dict_with_categories[div]:
                    Preference_flows_dict_with_categories[div][party] = {}
                #import pdb;pdb.set_trace()

                if ('ALP', 'COAL') not in Preference_flows_dict_with_categories[div][party]:
                    Preference_flows_dict_with_categories[div][party][('ALP', 'COAL')] = percent_to_alp

                


    #import pdb;pdb.set_trace()
    # reformat into dict of dicts of dfs:


    rows = []

    for division, party_dict in Preference_flows_dict_with_categories.items():
        for party, tcp_dict in party_dict.items():
            for tcp_pair, percent in tcp_dict.items():
                rows.append({
                    'division': division,
                    'tcp_pair': tcp_pair,
                    'party': party,
                    'percent': percent
                })



    # Step 2: Convert to DataFrame and pivot
    long_df = pd.DataFrame(rows)

    # Step 3: Pivot to wide format: one row per division, one column per party
    # Step 2: Create the dictionary of wide DataFrames
    result_dict = {}

    # Iterate over the unique divisions
    for div, group in long_df.groupby('division'):
        div_result = pd.DataFrame(index=range(len(TCP_COMBINATION_INDEX)), columns=Preference_flows_dict_with_categories[div].keys())
        
        # Fill the DataFrame with NaN (or 0 if preferred) for all TCP pairs initially
        div_result[:] = None  # Or use np.nan if you prefer NaN

        #if division == 'Melbourne':
        # import pdb;pdb.set_trace()


        # Iterate over each unique tcp_pair for this division
        for tcp_pair, tcp_group in group.groupby('tcp_pair'):

            if tcp_pair not in TCP_COMBINATION_INDEX.keys():
                # convert to party category:
                
                tcp_pair = tuple(sorted([party_category_dict[p] if p != 'Centre' else 'IND' for p in tcp_pair]))



            # Map tcp_pair to the corresponding row index
            row_index = TCP_COMBINATION_INDEX.get(tcp_pair, None)
            
            # If the index is not found in mapping, skip (shouldn't happen if mapping is correct)
            if row_index is None:
                #import pdb;pdb.set_trace()
                continue
            
            # Pivot the tcp_group into a wide format (party as columns, percent as values)
            for _, row in tcp_group.iterrows():
                div_result.at[row_index, row['party']] = row['percent']
        
        # Store the result for this division in the final dictionary
        result_dict[div] = div_result
        if 'ALP' not in result_dict[div]:
            result_dict[div].loc[:,'ALP'] = None
        if 'COAL' not in result_dict[div]:
            result_dict[div].loc[:,'COAL'] = None

        #if div == 'Macnamara':
        #    import pdb;pdb.set_trace()



    # Outlier - Canberra 2013 House had no Right parties!
    if data_year == '2016':
        for div in ['Bean','Canberra','Fenner']:
            result_dict[div].loc[:,'Right'] = None
            result_dict[div].loc[0,'Right'] = result_dict['Bean'].loc[0,'LDP']

        #import pdb;pdb.set_trace()



    # import Formal_prefs_dict for senate calibration
    with open(f"Formal_prefs_dict_{data_year}.pkl", "rb") as f:
        Formal_prefs_dict = pickle.load(f)

    with open(f"Senate_party_abvs_dict_{data_year}.pkl", "rb") as f:
        Senate_party_abvs_dict = pickle.load(f)


    
    LNP_ON_SEN_ADJUSTMENT_FOR_2025 = pd.DataFrame([{'ALP': 11.004743,'UAPP': 5.010368,'GRN': -6.365148,'Right': 5.010368,'Left': -6.365148}]) # Maranoa 2019


    # now, find non-classic divisions and extrapolate

    for tcp_pair in Non_classic_divs.keys():
        #import pdb; pdb.set_trace()

        if tcp_pair == ('COAL','COAL'): 
            continue

        # first get average of all results with said tcp:
        i = TCP_COMBINATION_INDEX[tcp_pair]

        series_list = []

        for div in Non_classic_divs[tcp_pair]:
            
            series_list.append(result_dict[div].iloc[i]) # curr reusults
        
        tcp_overall = pd.concat(series_list, axis=1)
        tcp_average = tcp_overall.mean(axis=1).dropna()

        #import pdb;pdb.set_trace()

        #if tcp_pair != ('COAL', 'Right'):
        #    continue


        

        # always do naive update first, then try senate comparisons

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                filtered_update = tcp_average[result_dict[div].columns.intersection(tcp_average.index)]
                result_dict[div].loc[i, filtered_update.index] = filtered_update


        if 'IND' not in tcp_pair:
            # Use senate calibration  - CURRENTLY ASSUMES TCP PARTY IN LEFT or RIGHT is ALWAYS THE SAME

            non_classic_sen_rows = []

            for div in Non_classic_divs[tcp_pair]:


                non_major_party = Non_classic_div_TCP_parties[div]

                if non_major_party in ['KAP','XEN']:
                    continue # Katter should not be used for preferences to Right!

                replace_p = {party_category_dict[non_major_party]:non_major_party}
                tcp_pair_parties = [replace_p.get(p,p) for p in tcp_pair]

                # replace COAL with LNP/LP/CLP/NP if COAL in 2CP
                if any(p in COAL_parties_list for p in tcp_pair_parties):
                    COAL_party = COAL_parties_div[div]
                    tcp_pair_parties = [COAL_party if p in COAL_parties_list else p for p in tcp_pair_parties]


                # df with cols div_nm, tcp_pair_parties, TCP, senate_TCP


                non_classic_sen_TCP_df = pd.DataFrame([{'tcp_pair_parties': tcp_pair_parties}])

                for party in result_dict[div]:
                    if result_dict[div].loc[i,party] and (party not in ['Left','Centre','Right','IND']) and party in Senate_party_abvs_dict[div]: # party must also have contested the senate
                        non_classic_senate_TCP = get_senate_TCP(Formal_prefs_dict, Senate_party_abvs_dict, div, tcp_pair_parties, party)
                        non_classic_sen_TCP_df[party] = non_classic_senate_TCP
                    else:
                        # no party info from house, or non-sen/category party
                        non_classic_sen_TCP_df[party] = np.nan # add nan to preserve party order



                # re-add categories for ['Left','Centre','Right','IND'] (IND if Centre exists) - same cols as result_dict[div] - use existing code somewhere

                non_classic_sen_TCP_df.iloc[:,1:] = add_category_pct(non_classic_sen_TCP_df.iloc[:,1:], party_category_dict) # CHECK: no errors with nan values?

                # If some unavailable e.g., IND, or non-senate, then just use an np.nan (equivalent of never embarking on this venture a la 2025 model)

                #import pdb; pdb.set_trace()

                non_classic_sen_TCP_df.iloc[:,1:] = pd.to_numeric(result_dict[div].iloc[i], errors='coerce').to_numpy() - non_classic_sen_TCP_df.iloc[:,1:].values * 100 # House minus Senate

                non_classic_sen_rows.append(non_classic_sen_TCP_df.iloc[:,1:])

            
            if (election_year == '2025') and (tcp_pair == ('COAL','Right')):
                tcp_pair_parties = ['LNP','ON']
                non_classic_sen_adjustment = LNP_ON_SEN_ADJUSTMENT_FOR_2025 # taken from Maranoa in 2019
            else:
                non_classic_sen_adjustment = pd.concat(non_classic_sen_rows).mean().to_frame().T # Multiple rows --> average over them! (Keeping any additional data)

            #import pdb; pdb.set_trace()
            
            # overwrite existing naive data where possible
            for div in result_dict.keys():
                #import pdb;pdb.set_trace()
                if div not in Non_classic_divs[tcp_pair] and (div not in NEW_SEATS_YEAR_DICT[data_year]):

                    # get correct COAL party for current div (if COAL in 2CP)
                    if any(p in COAL_parties_list for p in tcp_pair_parties):
                        COAL_party = COAL_parties_div[div]
                        tcp_pair_parties = [COAL_party if p in COAL_parties_list else p for p in tcp_pair_parties]

                    # if tcp party not in senate, skip this div - use naive
                    if any(party not in Senate_party_abvs_dict[div] for party in tcp_pair_parties): 
                        continue
                    

                    non_classic_sen_TCP_curr = pd.DataFrame()

                    for party in non_classic_sen_adjustment:
                        if pd.notna(non_classic_sen_adjustment.loc[0,party]) and (party not in ['Left','Centre','Right','IND']): # cases where senate comparison available

                            # If COAL, ensure party is matched to the correct senate version
                            if party in COAL_parties_list:
                                matched_party = next((p for p in Senate_party_abvs_dict[div] if p in COAL_parties_list), None)
                            else:
                                matched_party = party

                            if matched_party in Senate_party_abvs_dict[div]: # WHAT IF NOT??? - Then, use the original value! - MAKE SURE this case ends up being covered
                                non_classic_senate_TCP = get_senate_TCP(Formal_prefs_dict, Senate_party_abvs_dict, div, tcp_pair_parties, matched_party)
                                non_classic_sen_TCP_curr.loc[0,party] = non_classic_senate_TCP
                                #import pdb; pdb.set_trace()
                            else:
                                non_classic_sen_TCP_curr[party] = np.nan
                        else:
                            non_classic_sen_TCP_curr[party] = np.nan

                        #import pdb; pdb.set_trace()

                    # re-add categories for ['Left','Centre','Right','IND'] (IND if Centre exists) - same cols as result_dict[div] - use existing code somewhere

                    non_classic_sen_TCP_curr = add_category_pct(non_classic_sen_TCP_curr, party_category_dict)
                    #import pdb; pdb.set_trace()

                    curr_house_TCP = (non_classic_sen_TCP_curr*100 + non_classic_sen_adjustment).clip(lower=5, upper=95) # Senate + House - Senate (correct final amounts)

                    # add only results for required columns from only the new available data to results_dict for div
                    common_cols = result_dict[div].columns.intersection(curr_house_TCP.columns[curr_house_TCP.iloc[0].notna()])
                    result_dict[div].loc[i, common_cols] = curr_house_TCP.loc[0, common_cols].where(
                        pd.notna(curr_house_TCP.loc[0, common_cols]), result_dict[div].loc[i, common_cols]
                    )
                    
                    #print(div)
                    #print(result_dict[div].loc[i, ].to_frame().T)
                    


                    # This should ensure results are like previously, only overwritten where new data via senate comparisons is available

            for div in NEW_SEATS_YEAR_DICT[data_year]:
                # use supplier div to add any relevant new data to the div's result_dict
                supplier_dict = {v: k for k, v in replacement_seats_year_dict[data_year].items()}
                common_cols_with_supplier = result_dict[supplier_dict[div]].columns
                result_dict[div].loc[i,common_cols_with_supplier] = result_dict[supplier_dict[div]].loc[i,] # common columns with its supplier

        import pdb; pdb.set_trace()

        #if tcp_pair == ('COAL','Left'):
        #    import pdb;pdb.set_trace()

    for tcp_pair in [('IND','IND'),('Left','Left'),('LP','NP'),('Right','Right'),('COAL','COAL')]:

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                result_dict[div].loc[i, :] = 50.0

    for tcp_pair in [('IND','Right'),('IND','Left')]:

        # use IND-ALP/COAL i.e. 2 or 1

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                # symmetrically apply!

                source_row = result_dict[div].loc[i - 8, :].copy()

                # Invert each value ONLY if it's not None
                new_row = source_row.apply(lambda x: 100 - x if x is not None else None).astype('object')
                new_row = new_row.where(new_row.notna(), None)
                #new_row = 100 - result_dict[div].loc[i-8, :].copy() # correct indexing!
                if tcp_pair == ('IND', 'Right'):

                    if 'Right' not in new_row.index:
                        #print(div)
                        import pdb;pdb.set_trace()

                    new_row['COAL'] = new_row['Right'] 
                    
                elif tcp_pair == ('IND', 'Left'):
                    new_row['ALP'] = new_row['Left'] 

                # new_row.index.map(lambda x: 'ALP' if x == 'Left' else x)

                #result_dict[div].iloc[i,:] = new_row

                for col in result_dict[div].columns:
                    if new_row[col] is not None:  # Only overwrite if the new value isn't None
                        result_dict[div].at[i, col] = new_row[col]

            #import pdb;pdb.set_trace()
            #4


    for tcp_pair in [('Left','Right')]:
        #import pdb;pdb.set_trace()

        # use ALP/COAL

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                new_row = result_dict[div].loc[0, :].copy()

                # ALP gets Left's 
                # COAL gets Right's 
                new_row.loc['ALP'] = new_row['Left']
                new_row.loc['COAL'] = new_row['Right']

                result_dict[div].loc[i, :] = new_row

    import pdb;pdb.set_trace()


    for tcp_pair in [('ALP','Right'),('COAL','Left')]:
        #import pdb;pdb.set_trace()

        # use ALP/COAL

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():

                # ALP gets Left's 
                # COAL gets Right's 

                if tcp_pair == ('ALP','Right'):

                    new_row = result_dict[div].loc[0, :].copy()

                    new_row.loc['COAL'] = new_row['Right']

                    result_dict[div].loc[i, :] = new_row


                elif tcp_pair == ('COAL','Left'):

                    source_row = result_dict[div].loc[0, :].copy()
                    # switch order
                    new_row = source_row.apply(lambda x: 100 - x if x is not None else None).astype('object')
                    new_row = new_row.where(new_row.notna(), None)

                    new_row.loc['ALP'] = new_row['Left']              

                    for col in result_dict[div].columns:
                        if new_row[col] is not None:  # Only overwrite if the new value isn't None
                            result_dict[div].at[i, col] = new_row[col]      

                    #import pdb;pdb.set_trace()          


    #import pdb;pdb.set_trace()

    # If RIght is None, get corresponding COAL, smae for left and ALP
    Right_COAL_Right_Preferences = 100 - 58.39 # ON in 2016/9 Maranoa
    Left_COAL_Left_Preferences = 30.5 # 2016/19 Kooyong/Higgins/Melbourne
    Left_ALP_Left_Preferences = 34.0 # 2016/19 ALP/GRN Seats
    Right_COAL_IND_Preferences = 39.8 # for XEN: 2013 Indi/NE preferences
    Right_COAL_Left_Preferences = 40.0 # for 2016, from 2010 Grayndler/Batman
    Right_COAL_COAL_Preferences = 50.0


    for div in result_dict.keys():
        #print(div)
        div = div
        for i, row in result_dict[div].iterrows():
            if pd.isna(row['Right']):
                #import pdb;pdb.set_trace()
                result_dict[div].at[i, 'Right'] = row['COAL']
            if pd.isna(row['Left']):
                #import pdb;pdb.set_trace()
                result_dict[div].at[i, 'Left'] = row['ALP']




            # Fix no Right for RIght-COAL contests:
            if (data_year == '2022') and (i == 6) and (pd.isna(result_dict[div].loc[6,'Right'])):
                result_dict[div] = result_dict[div].copy()
                result_dict[div].at[6,'Right'] = Right_COAL_Right_Preferences


            
            if (data_year == '2022') and (i == 5) and (pd.isna(result_dict[div].loc[5,'Left'])):
                result_dict[div].at[5, 'Left'] = Left_COAL_Left_Preferences

            if (data_year == '2022') and (i == 3) and (pd.isna(result_dict[div].loc[3,'Left'])):
                result_dict[div].at[3, 'Left'] = Left_ALP_Left_Preferences

            if (data_year == '2019') and (i == 3) and (pd.isna(result_dict[div].loc[3,'Left'])):
                result_dict[div].at[3, 'Left'] = Left_ALP_Left_Preferences

            if (data_year == '2016') and (i == 1) and (pd.isna(result_dict[div].loc[1,'Right'])):
                result_dict[div].at[1, 'Right'] = Right_COAL_IND_Preferences

            if (data_year == '2016') and (i == 9) and (pd.isna(result_dict[div].loc[9,'Right'])):
                result_dict[div].at[9, 'Right'] = 1 - Right_COAL_IND_Preferences

            if (data_year == '2016') and (i == 5) and (pd.isna(result_dict[div].loc[5,'Right'])):
                result_dict[div].at[5, 'Right'] = Right_COAL_Left_Preferences

            if (data_year == '2016') and (i == 6) and (pd.isna(result_dict[div].loc[6,'Right'])):
                result_dict[div].at[6, 'Right'] = Right_COAL_Right_Preferences # cheating from future - but quick fix!

            if (data_year == '2016') and (i == 14) and (pd.isna(result_dict[div].loc[14,'Right'])):
                result_dict[div].at[14, 'Right'] = Right_COAL_COAL_Preferences


            # fetch all changes so far!
            row = result_dict[div].loc[i]



            if pd.isna(row['Centre']):
                #import pdb;pdb.set_trace()
                #import pdb;pdb.set_trace()
                if (row['Right']) and (row['Left']):
                    

                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['Left'])/2

                    if pd.isna(result_dict[div].at[i, 'Centre']):
                        import pdb;pdb.set_trace()
                    
                elif (row['Right']) and (row['ALP']):
                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['ALP'])/2

                elif (row['Left']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['Left'])/2

                elif (row['ALP']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['ALP'])/2

                if pd.isna(result_dict[div].at[i, 'Centre']):
                    print(div, i, row)
                    import pdb;pdb.set_trace()

                    






            

    for division in result_dict.keys():
        #print(division)
        if result_dict[division]['Centre'].isna().sum():
            import pdb;pdb.set_trace()
            1
        if result_dict[division]['Right'].isna().sum():
            import pdb;pdb.set_trace()
            6
        if result_dict[division]['Left'].isna().sum():
            import pdb;pdb.set_trace()
            5


            

    #import pdb;pdb.set_trace()


    # find COALITION_double_divs_last_year
    DOP_By_Division_curr = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'})[["div_nm","PartyAb"]].drop_duplicates()    
    DOP_By_Division_curr.loc[:,'PartyAb'] = DOP_By_Division_curr.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
    DOP_By_Division_curr = {div: group['PartyAb'].tolist() for div, group in DOP_By_Division_curr.groupby("div_nm")}

    COAL_double_divs_curr = []
    for div in DOP_By_Division_curr.keys():
        if ('LP' in DOP_By_Division_curr[div]) and ('NP' in DOP_By_Division_curr[div]):
            COAL_double_divs_curr.append(div)


    COAL_double_div_transfers = TCP_Preference_Flows.loc[((TCP_Preference_Flows['PartyAb']=='NP') & (TCP_Preference_Flows['TCP_Ab']=='LP')) | ((TCP_Preference_Flows['PartyAb']=='LP') & (TCP_Preference_Flows['TCP_Ab']=='NP')),].rename(columns = {'TransferPercentage':'COAL%'})[['div_nm','COAL%']]
    COAL_double_div_transfers = COAL_double_div_transfers.set_index('div_nm')
    COAL_double_div_transfers.loc['Average'] = COAL_double_div_transfers['COAL%'].mean()
    #import pdb;pdb.set_trace()

    COAL_double_divs_next = []
    for div in Div_parties_next_dict.keys():
        if ('LP' in Div_parties_next_dict[div]) and ('NP' in Div_parties_next_dict[div]):
            COAL_double_divs_next.append(div)



    COAL_list = ['LP','NP','CLP','LNP']


    for div in result_dict.keys():

        # convert all None to nan
        result_dict[div] = result_dict[div].where(pd.notna(result_dict[div] ), None)  # no-op if already NaN
        result_dict[div]  = result_dict[div] .astype(float)


        for i, party in enumerate(Div_parties_next_dict_COAL[div]):

            # new parties
            if party not in result_dict[div].columns:
                result_dict[div].loc[:,party] =  result_dict[div][party_category_dict[party]] # replace with the category!

            else: # fill in missing vals for old parties
                result_dict[div][party] = result_dict[div][party].combine_first(result_dict[div][party_category_dict[party]])
                #import pdb;pdb.set_trace()

            if party == 'COAL':
                PartyAb = Div_parties_next_dict[div][i]
                result_dict[div].loc[:,PartyAb] = result_dict[div]['COAL']


        # Finally, add COAL -> COAL support if COAL_double_div

        if div in COAL_double_divs_next:

            if div in COAL_double_div_transfers.index:
                COAL_Pct = COAL_double_div_transfers.loc[div].iloc[0]
            else:
                COAL_Pct = COAL_double_div_transfers.loc['Average'].iloc[0]

            for col in ['LP','NP']:
                row_indexer = [0,1,5,6]
                result_dict[div].loc[row_indexer,col] = [100 - COAL_Pct, COAL_Pct, COAL_Pct, COAL_Pct]

        #import pdb;pdb.set_trace()

        # fill in ALP/COAL rows with 0


            

            

        # Remove Centre, LeftLRight, "COAL"
        result_dict[div] = result_dict[div].drop(['Left','Centre','Right','COAL'], axis = 1)


    #import pdb;pdb.set_trace()
    cols_to_drop = ['ALP', 'LNP', 'LP', 'NP', 'CLP']

    for div in result_dict.keys():

        df_dropped = result_dict[div].drop(columns=[col for col in cols_to_drop if col in result_dict[div].columns])

        # Now check for NaN values in the remaining columns
        if df_dropped.isna().any().any():
            import pdb;pdb.set_trace()

                

    def expand_and_reorder_duplicate(df, new_names):
        return pd.concat([df[col] for col in new_names], axis=1, keys=new_names)


    

    # get them in Ballot order: 

    # Div_parties_next_dict groups all the INDs together - must use Div_Ballot_Order_next_dict instead to get all INDs in Ballot order
    if next_year != '2025':
        DOP_by_div_full = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'}).rename(columns = {'DivisionNm':'div_nm'}) 
        DOP_by_div_full = DOP_by_div_full.loc[(DOP_by_div_full['CountNumber']==0) & (DOP_by_div_full['CalculationType'] == 'Preference Count'),['div_nm', 'PartyAb']]
        DOP_by_div_full.loc[:,'PartyAb'] = DOP_by_div_full.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
        Div_Ballot_Order_next_dict = DOP_by_div_full.groupby('div_nm')['PartyAb'].apply(list).to_dict()
    else:
        Div_Ballot_Order_next_dict = Div_parties_next_dict

    for div in result_dict.keys():

        result_dict[div] = expand_and_reorder_duplicate(result_dict[div], Div_Ballot_Order_next_dict[div]) 


    #import pdb;pdb.set_trace()

    # adjust preferences of Defecting Independents in 2022 to be more favourable to their original parties (~ like ON)

    if data_year == '2022':
        Defected_INDs_Ballot_order = {'Moore':1, 'Monash':3,'Calare':7}
        for div in ['Monash','Moore','Calare']:
            idx = Defected_INDs_Ballot_order[div] - 1
            result_dict[div].iloc[:,idx] = result_dict[div]['ON']

        # Macnamara - Josh Burns Open Preference in 2025
        result_dict['Macnamara'].loc[[5],'ALP'] += 6.6

        #import pdb;pdb.set_trace()


    TCP_pair_category_dict = result_dict



    return TCP_pair_category_dict


election_year = '2025'

party_category_dict = make_party_category_dict()
TCP_pair_category_dict = make_TCP_pair_category_dict(election_year = election_year, party_category_dict=party_category_dict)


if not os.path.exists(f"TCP_pair_category_dict_for_{election_year}.pkl"):
    with open(f"TCP_pair_category_dict_for_{election_year}.pkl", "wb") as f:
        pickle.dump(TCP_pair_category_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

import pdb; pdb.set_trace()