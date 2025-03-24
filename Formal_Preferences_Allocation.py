import pandas as pd
import numpy as np
import os,time
from collections import Counter
import io
import os
import glob
from pathlib import Path
from itertools import groupby



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


FP_ID_COLUMNS = [3,4,5] # remove id columns
START_OF_PREFS = 2 # Prefs begin on the 3th column (after div_nm,pp_nm) - deleted stateab to accomodate 2016 file
BTL_ONLY_ELECTIONS = ['2007','2010','2013']

NUM_OF_INDX_LETTERS = 4
SMALL_CONST_FOR_LATENT_IND = 1e-10




incumbent_advantage_dict = {5:4.68,4:4.5,3:6}
final_cand_no_dict = {"2022":5, "2019": 4, "2016": 4,"2013": 5, "2010": 3, "2007": 4, "2004": 4,"2001":4}



is_redistribution = 0
data_year = '2013'
FINAL_CANDIDATE_NO = final_cand_no_dict[data_year]
INCUMBENT_ADVANTAGE = incumbent_advantage_dict[FINAL_CANDIDATE_NO]

NONINCUMBENT_DISADVANTAGE =  INCUMBENT_ADVANTAGE/(FINAL_CANDIDATE_NO-1)


new_seats_year_dict = {'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner','Gorton']}
name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
states_to_redistribute_dict = {'2022': ['NSW','VIC','WA','NT'],'2019': ['VIC','WA'],'2016':['ACT','NT','QLD','SA','TAS','VIC'],'2013':['ACT','NSW','WA'],'2010':['SA','VIC'],'2007':['NSW','NT','QLD','TAS','WA'],'2004':['ACT','NSW','QLD'],'2001':['QLD','SA','VIC']}



# check_house_senate_discrepancies(data_year, name_changes_year_dict) # early check without needing to load Formal Preferences - must add [data_year] to follow name_changes_year_dict if called here


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



# get state-to-div dict, adjusting for name changes
div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
div_to_state_dict = {name_changes_year_dict[data_year].get(div, div): div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}









def abbreviate_party_names(party_names_list, general_party_df):
    # handle exceptions to party names
    party_abvs_list = []

    for party in party_names_list:
        #print(party)
        if party:
            if party.lower() == "Liberal/The Nationals".lower() or party.lower() == "Liberal & Nationals".lower() or party.lower() == "Liberal/National".lower(): # handle LIB/NAT Exception - I think best to treat them as one party in the Senate as they always contest together, and then reverse engineer House split if needed
                party_abvs_list.append('COAL')
            elif party == " Science, Pirate, Secular, Climate Emergency": # SOPA exception
                party_abvs_list.append('SOPA')
            elif party == "Labor/Country Labor":
                party_abvs_list.append('ALP')
            elif party == "Science Party/Australian Cyclists Party":
                party_abvs_list.append('FTCY')
            elif party == 'Australian Sex Party/Marijuana (HEMP) Party':
                party_abvs_list.append('SXHM')
            elif party == 'A.F.N.P.P.':
                party_abvs_list.append('FNPP')
            else:
                if general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].empty:
                    import pdb;pdb.set_trace()
                    continue
                party_abvs_list.append(general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].iloc[0])
        else:
            party_abvs_list.append('')


        #import pdb;pdb.set_trace()

    return party_abvs_list


general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',"PartyAb"] = 'GRN' # handle exceptions, but think GVIC is the only one



def get_2016_Senate_party_names(state, return_PartyAbs = False):
    ### reads in the unusual 2016 Formal Prefs csv file, filling in the header column with senate groupings

    #1. LOAD DATA & CLEAN
    SenateCandidates_2016 = pd.read_csv("2016SenateCandidates.csv", index_col = None)
    SenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016["nom_ty"] == 'S',["state_ab","ticket","party_ballot_nm"]]
    SenateCandidates_2016.rename(columns={"state_ab": "StateAb", "party_ballot_nm": "party_nm"}, inplace=True)

    StateSenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016['StateAb'] == state,:]
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].fillna('') # for ungrouped

    # 2. COALITION(S) inspect if there are coalitions and give them PartyAb
    nonUG = StateSenateCandidates_2016.loc[~(StateSenateCandidates_2016['ticket']=='UG'),]
    coalition_df = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1].iloc[:,1:].drop_duplicates(ignore_index=True).groupby("ticket", as_index=False)['party_nm'].agg("/".join)


    if not coalition_df.empty:
        coalition_group_dict = coalition_df.set_index("ticket")["party_nm"].to_dict() #coalition_party_names = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1]['party_nm'].unique()
        # map dictionary
        StateSenateCandidates_2016.loc[:,"party_nm"] = StateSenateCandidates_2016["ticket"].map(coalition_group_dict).where(StateSenateCandidates_2016["ticket"].isin(coalition_group_dict), StateSenateCandidates_2016["party_nm"])

    party_names_list = StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket']!='UG','party_nm'].drop_duplicates(ignore_index=True).tolist()

    # 3. convert to PartyAb and format for Formal Preferences
    party_abvs = abbreviate_party_names(party_names_list, general_party_df)

    if return_PartyAbs:
        return party_abvs

    party_names_abvs_dict = dict(zip(party_names_list,party_abvs))
    StateSenateCandidates_2016 = StateSenateCandidates_2016.copy() # avoid warning ..?
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].replace(party_names_abvs_dict)
    StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket'] == 'UG','party_nm'] = ''

    # format string column names
    group_party_names = StateSenateCandidates_2016['ticket'].astype(str) + ':' + StateSenateCandidates_2016['party_nm'].astype(str)
    unique_groups = group_party_names[~group_party_names.str.startswith('UG')].drop_duplicates(ignore_index=True)
    group_party_names = unique_groups.tolist() + group_party_names.tolist()

    return group_party_names

def get_2007_2013_Senate_party_names(state, data_year):
    # Use PartyAbs provided in FP by Div by Vote Type - replace LPNP/LP/NP/LNP with COAL as is customary in teh correct years

    First_prefs_senate = pd.read_csv(f'{data_year}SenateFirstPrefsByDivisionByVoteType.csv', skiprows = 1,index_col = None)
    SenateCandidates = First_prefs_senate.loc[First_prefs_senate['StateAb']==state,]

    if (state in ['VIC','NSW']) or ((data_year == '2007') & (state == 'QLD')):
        SenateCandidates.loc[SenateCandidates['PartyAb'].isin(['LPNP','LNP','LP','NP']),'PartyAb'] = 'COAL'

    SenateCandidates = SenateCandidates[['Ticket','PartyAb']].drop_duplicates().fillna('')

    party_abvs = SenateCandidates.loc[SenateCandidates['Ticket']!='UG','PartyAb'].tolist()

    return party_abvs


def get_Senate_party_abvs_dict(data_year, div_to_state_dict, to_csv = False):
    # quickly extracts abvs from the senate without needing to read all of Formal Prefs


    Formal_prefs_dict = {}
    states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']

    if data_year in BTL_ONLY_ELECTIONS:
        state_party_abvs_list = {}
        for state in states:
            state_party_abvs_list[state] = get_2007_2013_Senate_party_names(state, data_year)
        
        Senate_party_abvs_dict = {}
        for div in div_to_state_dict.keys():
            state = div_to_state_dict[div]
            Senate_party_abvs_dict[div] = state_party_abvs_list[state]

    #### basic version to get the party names lists for cheap - read only 2 rows each!
    elif data_year == '2016':
        state_party_abvs_list = {}
        for state in states:
            state_party_abvs_list[state] = get_2016_Senate_party_names(state, return_PartyAbs = True)
        
        Senate_party_abvs_dict = {}
        for div in div_to_state_dict.keys():
            state = div_to_state_dict[div]
            Senate_party_abvs_dict[div] = state_party_abvs_list[state]

    else:
        for state in states: # currently only 2016 onwards
            filename = f"{data_year}FormalPrefs{state}.csv"

            state_Formal_prefs = pd.read_csv(filename, nrows=1)
            state_Formal_prefs.drop(columns=state_Formal_prefs.columns[FP_ID_COLUMNS], inplace=True)
            

            state_Formal_prefs_dict = {state: group.reset_index(drop=True).apply(
                lambda col: pd.to_numeric(col, downcast='float') if pd.api.types.is_numeric_dtype(col) else col
            ) for state, group in state_Formal_prefs.groupby("State")} 

            for key, group in state_Formal_prefs_dict.items():
                group.pop('State') # remove State for concordance with later dfs
                Formal_prefs_dict[key] = group # assumes no keys (divs) overlap for different states :)

        Senate_party_abvs_dict = {}
        for div in div_to_state_dict.keys():
            #import pdb;pdb.set_trace()
            state = div_to_state_dict[div] # gets StateAb
            formal_prefs_full = Formal_prefs_dict[state]
            formal_prefs = formal_prefs_full.iloc[:, START_OF_PREFS:]
            formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
            start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated

            # store group party names (from ATL) in Senate_party_names_dict
            group_party_names = formal_prefs_full.iloc[:, START_OF_PREFS:].columns[:start_of_BTL_index] # includes both group and party names
            party_names_list = group_party_names.str.split(':').str[-1].tolist() # records only party names
            party_abvs_list = abbreviate_party_names(party_names_list, general_party_df)
            Senate_party_abvs_dict[div] = party_abvs_list
    
    # write to csv
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) 

    return Senate_party_abvs_dict


Senate_party_abvs_dict = get_Senate_party_abvs_dict(data_year, div_to_state_dict, to_csv = False)




######### Candidate Pairs stuff


DOP_By_PP_Expand = pd.read_csv(f"{data_year}DOP_By_PP_Expand.csv", index_col=None)
DOP_By_PP_Pref_Percent = pd.read_csv(f"{data_year}DOP_By_PP_Pref_Percent.csv", index_col=None)
DOP_By_PP_Reduce = pd.read_csv(f"{data_year}DOP_By_PP_Reduce.csv", index_col=None)


def convert_to_wide_format(df, df_type):
    # converts to wide format indexed by pp_id for either First Preferences or SA1 dfs
    if df_type == "First Preferences":
        pivot_df = df.pivot_table(index=['pp_id'], 
                                columns=['PartyAb'], 
                                values='votes', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()
    if df_type == "SA1s":
        pivot_df = df.pivot(index='pp_id', columns='SA1_CODE16', values='votes')
        pivot_df = pivot_df.fillna(0)
        pivot_df = pivot_df.astype(int)
        pivot_df = pivot_df.reset_index()
    
    if df_type == "DOP":
        pivot_df = df.pivot_table(index=['CountNumber'], 
                                columns=['PartyAb'], 
                                values='CalculationValue', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()

    if df_type == "DOP_By_PP":

        pivot_df = df.pivot_table(index=['pp_id','CountNumber'], # double index for info across pp_ids
                                columns=['PartyAb'], 
                                values='CalculationValue', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()

    return pivot_df

def compute_ratio_efficient(df):
    # more efficient version using vectorised operations
    
    # Pivot table to get values in a single row
    pivot = df.pivot(index=df.columns[:-2].tolist(), columns="CalculationType", values="CalculationValue") # preserve order of parties as BallotPosition is before cand_id

    #import pdb;pdb.set_trace()

    # Compute the ratio
    ratio = pivot.get("Transfer Count", 0) / pivot.get("Preference Count", 1).replace(-np.inf, -1).fillna(0)
    ratio = ratio.replace(-np.inf, -1).fillna(0)
    pivot["CalculationValue"] = ratio

    pivot = pivot.iloc[:,-1]
    #import pdb;pdb.set_trace()

    # Reset index and transform back to the required format
    result = pivot.reset_index()

    return result

def rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, Senate_party_abvs_dict, by_pp_id = False):
    ### apprends div_nm onto any INDXs and changes COAL member parties to COAL or COALNP/COALLP for doubles
    if (DOP_table_wide.columns.isin(COAL_set).sum() == 2) & (div_to_state_dict[div] in ['VIC','NSW']): # Both members of Coalition in div!
        #import pdb;pdb.set_trace()
        for party in DOP_table_wide.columns[1+by_pp_id:]:

            # convert LP and NP to COALLP/COALNP
            if (party=='NP') | (party =='LP'):
                DOP_table_wide.rename(columns = {party: 'COAL' + party}, inplace = True) # rename to COALLP
            elif party == 'CLR':
                DOP_table_wide.rename(columns = {party: 'ALP'}, inplace = True) # rename to ALP

            elif party.startswith('IND'):
                DOP_table_wide.rename(columns = {party: party + div}, inplace = True) # e.g. IND1Goldstein
            elif party not in Senate_party_abvs_dict[div]:
                DOP_table_wide.rename(columns = {party: party + div}, inplace = True) # e.g. CECHunter

            
    
    else:
        for party in DOP_table_wide.columns[1+by_pp_id:]:

            # convert LP and NP in VIC/NSW to COAL
            if (div_to_state_dict[div] in ['VIC','NSW']) and (party in ['LP','NP']):
                DOP_table_wide.rename(columns = {party: 'COAL'}, inplace = True)
            elif party == 'CLR':
                DOP_table_wide.rename(columns = {party: 'ALP'}, inplace = True) # rename to ALP

            elif party.startswith('IND'):
                DOP_table_wide.rename(columns = {party: party + div}, inplace = True) # e.g. IND1Goldstein
            elif party not in Senate_party_abvs_dict[div] and party:
                DOP_table_wide.rename(columns = {party: party + div}, inplace = True) # e.g. CECHunter

            
    return DOP_table_wide


def create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type):
    ### Processes the data in 4 ways: 1. Fills Blanks (NAFD) with IND 2. Given IND label i.e. IND1,IND2 3. GVIC --> GRN 4. Processes INDs/COAL names correctly

    ### In future simplify the function by reducing duplication
    
    DOP_table_wide_dict = {}

    if DOP_type == "EliminationOrder":
        # get state-to-div dict

        for div in Div_DOP_dict.keys():
            #print(div)
            FP_pcts = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] == 0) & (Div_DOP_dict[div]["CalculationType"] == "Preference Percent"),]
            Transfer_pcts = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] > 0) & (Div_DOP_dict[div]["CalculationType"] == "Transfer Percent"),]
            DOP_table_long = pd.concat([FP_pcts, Transfer_pcts], ignore_index=True)

            # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
            DOP_table_long.loc[:,'PartyAb'] = DOP_table_long['PartyAb'].fillna('IND') 

            # relabel independents in order of ballot appearance if there are multiple
            target = 'IND'
            DOP_table_long['Count'] = DOP_table_long.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
            # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
            adjusted_party_names = DOP_table_long.loc[DOP_table_long["CountNumber"] == 0,].apply(
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            )
            num_pref_counts = (DOP_table_long.iloc[-1,0] + 1) # num of final count + original FP count

            DOP_table_long.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * num_pref_counts, ignore_index=True)
            DOP_table_long.loc[DOP_table_long["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!


            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
            
            # record elimination order
            Elim_order_list_part = DOP_table_wide.iloc[1:,].apply(lambda row: row[row == -100.00].index[0], axis=1).tolist()# Apply the function row-wise to get the column names
            Final_2_Parties = DOP_table_wide.iloc[-1,1:][DOP_table_wide.iloc[-1,] > 0].index.tolist()
            Elim_order_list = Elim_order_list_part + Final_2_Parties

            # give INDs distinct names based on division and convert LP and NP into COAL in Victoria & account for divs with both Coalition parties!
            COAL_set = {'NP','LP'}

            if (len(set(Elim_order_list) & COAL_set) == 2) & (div_to_state_dict[div] in ['VIC','NSW']): # Both members of Coalition in div!
                #import pdb;pdb.set_trace()
                for i, party in enumerate(Elim_order_list):

                    # convert LP and NP to COALLP/COALNP
                    if (party=='NP') | (party =='LP'):
                        Elim_order_list[i] = 'COAL' + party # rename to COALLP
                    elif party == 'CLR':
                        Elim_order_list[i] = 'ALP' # rename to ALP

                    elif party.startswith('IND'):
                        Elim_order_list[i] = party + div # e.g. IND1Goldstein
                    elif party not in Senate_party_abvs_dict[div]:
                        print(party + div)
                        Elim_order_list[i] = party + div # e.g. CECHunter

                    
            
            else:
                for i, party in enumerate(Elim_order_list):

                    # convert LP and NP in VIC/NSW to COAL
                    if (div_to_state_dict[div] in ['VIC','NSW']) and (party in ['LP','NP']):
                        Elim_order_list[i] = 'COAL'
                    elif party == 'CLR':
                        Elim_order_list[i] = 'ALP' # rename to ALP
                    elif party.startswith('IND'):
                        Elim_order_list[i] = party + div # e.g. IND1Goldstein
                    elif (party not in Senate_party_abvs_dict[div]) and (party not in ['LP','NP']):
                        Elim_order_list[i] = party + div # e.g. CECHunter
                        print(party + div)

                    

            DOP_table_wide_dict[div] = Elim_order_list[::-1] # need to still reverse




    
    if DOP_type == 'Expand':
        for div in Div_DOP_dict.keys():

            # get ratio of Transfer Count / Preference Count
            progressed_counts = Div_DOP_dict[div].loc[Div_DOP_dict[div]["CountNumber"]>0,]

            DOP_table_long = compute_ratio_efficient(progressed_counts).drop('BallotPosition', axis=1) # BallotPosition only useful in preserving order of candidates

            #import pdb;pdb.set_trace()



            # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
            DOP_table_long['PartyAb'] = DOP_table_long['PartyAb'].fillna('IND') 

            # relabel independents in order of ballot appearance if there are multiple
            target = 'IND'
            DOP_table_long['Count'] = DOP_table_long.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
            # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ... (CountNumber starts from 1)
            adjusted_party_names = DOP_table_long.loc[DOP_table_long["CountNumber"] == 1,].apply(
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            ).reset_index(drop=True)
            num_pref_counts = (DOP_table_long.iloc[-1,0] + 1) # num of final count + original FP count

            DOP_table_long.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * (num_pref_counts), ignore_index=True) # project IND# across df ; (-1 because df excludes FP count)


            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
            #import pdb;pdb.set_trace()

            DOP_table_wide = DOP_table_wide.rename(columns = {"GVIC": "GRN"}) # GVIC issue resolve!
            
            # give INDs distinct names based on division and convert LP and NP into COAL in Victoria/NSW
            COAL_set = {'NP','LP'}
            DOP_table_wide = rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, Senate_party_abvs_dict, by_pp_id = False)
            

            DOP_table_wide_dict[div] = DOP_table_wide


    if DOP_type == 'PrefPercent':
        for div in Div_DOP_dict.keys():

            #Div_DOP_dict[div] = Div_DOP_dict[div].loc[Div_DOP_dict[div]["CountNumber"]>0,]
            DOP_table_long = Div_DOP_dict[div].loc[Div_DOP_dict[div]["CalculationType"] == "Preference Percent",].reset_index(drop=True)
            DOP_table_long = DOP_table_long.copy()
            DOP_table_long = DOP_table_long.reset_index(drop=True)


            #import pdb;pdb.set_trace()

            # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
            DOP_table_long['PartyAb'] = DOP_table_long['PartyAb'].fillna('IND') 

            # relabel independents in order of ballot appearance if there are multiple
            target = 'IND'
            DOP_table_long['Count'] = DOP_table_long.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
            # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ... (CountNumber starts from 1)
            adjusted_party_names = DOP_table_long.loc[DOP_table_long["CountNumber"] == 0,].apply( # CountNumber === 0
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            ).reset_index(drop=True)
            num_pref_counts = (DOP_table_long.iloc[-1,0] + 1) # num of final count + original FP count

            DOP_table_long.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * (num_pref_counts), ignore_index=True) # project IND# across df ; (-1 because df excludes FP count)


            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
            #import pdb;pdb.set_trace()

            DOP_table_wide = DOP_table_wide.rename(columns = {"GVIC": "GRN"}) # GVIC issue resolve!
            # give INDs distinct names based on division and convert LP and NP into COAL in Victoria/NSW
            
            
            COAL_set = {'NP','LP'}
            DOP_table_wide = rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, Senate_party_abvs_dict, by_pp_id = False)

            DOP_table_wide_dict[div] = DOP_table_wide
            #import pdb;pdb.set_trace()

    if DOP_type == 'Reduce':
        for div in Div_DOP_dict.keys():
            DOP_table_long = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] > 0) & (Div_DOP_dict[div]["CalculationType"] == "Transfer Percent"),].reset_index(drop=True)
            DOP_table_long = DOP_table_long.copy()
            DOP_table_long = DOP_table_long.reset_index(drop=True)
            DOP_table_long.loc[:,'CalculationValue'] /= 100 # ensure they are in proportion terms

            # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
            DOP_table_long.loc[:,'PartyAb'] = DOP_table_long['PartyAb'].fillna('IND') 


            # relabel independents in order of ballot appearance if there are multiple
            target = 'IND'
            DOP_table_long.loc[:,'Count'] = DOP_table_long.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
            # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...
            adjusted_party_names = DOP_table_long.loc[DOP_table_long["CountNumber"] == 1,].apply(
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            ).reset_index(drop=True)
            num_pref_counts = (DOP_table_long.iloc[-1,0] + 1) # num of final count + original FP count

            DOP_table_long.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * (num_pref_counts-1), ignore_index=True) # project IND# across df ; (-1 because df excludes FP count)


            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")

            DOP_table_wide = DOP_table_wide.rename(columns = {"GVIC": "GRN"}) # GVIC issue resolve!
            #DOP_table_wide_dict[div] = DOP_table_wide.astype(int)

            COAL_set = {'NP','LP'}
            DOP_table_wide = rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, Senate_party_abvs_dict, by_pp_id = False)

            DOP_table_wide_dict[div] = DOP_table_wide

    return DOP_table_wide_dict

def convert_long_to_wide_format(DOP_table_long, div_to_state_dict, Senate_party_abvs_dict):
    ### creates dict for div_nm as key and wide DOP table for each pp_id as value

    DOP_By_PP_dict = {div: group for div, group in DOP_table_long.groupby('div_nm')}

    for div, group in DOP_By_PP_dict.items():

        target = 'IND' # relabel independents in order of ballot appearance if there are multiple

        group_sample_zero = group.loc[group['pp_id']==0,] # always will have one
        group_sample_zero = group_sample_zero.copy()
        group_sample_zero.loc[:,'Count'] = (group_sample_zero.groupby('PartyAb').cumcount() + 1)     # Count instances of the target string

        adjusted_party_names = group_sample_zero.loc[group_sample_zero["CountNumber"] == 0,].apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1)
        
        num_pref_counts = (group_sample_zero.iloc[-1,3] + 1) # num of final count + original FP count

        num_rows = len(group['pp_id'].unique()) * num_pref_counts

        group.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * (num_rows), ignore_index=True).values
        group.loc[group["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!

        DOP_table_wide = convert_to_wide_format(group, "DOP_By_PP")

        # give INDs distinct names based on division and convert LP and NP into COAL in Victoria/NSW
        COAL_set = {'NP','LP'}
        DOP_table_wide = rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, Senate_party_abvs_dict, by_pp_id = True)

        DOP_By_PP_dict[div] = DOP_table_wide

    return DOP_By_PP_dict


# create wide format eliminaation_order_dict
DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)
DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
Div_DOP_dict = {div: group.drop(columns=['div_nm']) for div, group in DOP_By_Division[["div_nm","CountNumber","BallotPosition","cand_id", "PartyAb","CalculationType", "CalculationValue"]].groupby("div_nm")}

Div_DOP_dict = {name_changes_year_dict[data_year].get(key, key): val for key, val in Div_DOP_dict.items()} # adjust for name changes

Elimination_order_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "EliminationOrder")
DOP_div_expand_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "Expand")
DOP_div_pref_percent_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "PrefPercent")
DOP_div_reduce_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "Reduce")



DOP_By_PP_Pref_Percent_wide_dict = convert_long_to_wide_format(DOP_By_PP_Pref_Percent, div_to_state_dict, Senate_party_abvs_dict)
DOP_By_PP_Expand_wide_dict = convert_long_to_wide_format(DOP_By_PP_Expand, div_to_state_dict, Senate_party_abvs_dict)
DOP_By_PP_Reduce_wide_dict = convert_long_to_wide_format(DOP_By_PP_Reduce, div_to_state_dict, Senate_party_abvs_dict)























def make_unique(names):
    """ Append suffixes to duplicate column names to make them unique. """
    counts = Counter()
    unique_names = []
    
    for name in names:
        if counts[name] > 0:
            new_name = f"{name}_dup{counts[name]}"  # Add suffix
        else:
            new_name = name
        counts[name] += 1
        unique_names.append(new_name)

    return unique_names




def split_and_convert_2016(series, cols, chunk_size=500000, output_file="processed_data.parquet", return_df=True):
    """Processes large CSV-like columns in chunks, writes to disk, and optionally returns a final DataFrame."""
    
    max_columns = len(cols)

    def parse_row(s):
        """Splits a row and converts elements to integers, handling empty values."""
        items = s.rstrip(',').split(',')
        return [int(i) if i.isdigit() else np.nan for i in items[:max_columns]] + [np.nan] * (max_columns - len(items))

    first_chunk = True
    for chunk_start in range(0, len(series), chunk_size):
        chunk = series.iloc[chunk_start : chunk_start + chunk_size]  # Select batch

        chunk_parsed = np.array([parse_row(row) for row in chunk], dtype=np.float32)  # Process batch
        
        # ✅ Convert to Pandas DataFrame (but do not store it in memory)
        chunk_df = pd.DataFrame(chunk_parsed)

        # ✅ Convert to Pandas nullable Int16 (fixes NaN issues)
        chunk_df = chunk_df.convert_dtypes().astype("Int16")

        chunk_df.columns = make_unique(cols) 

        # ✅ Write chunk directly to file (No concatenation)
        chunk_df.to_parquet(output_file, engine="fastparquet", index=False, compression="snappy", append=not first_chunk)
        first_chunk = False  # Append mode for next chunks

        # ✅ Explicitly delete unused objects to free memory
        del chunk, chunk_parsed, chunk_df
        gc.collect()  # Force garbage collection

        print(f"Processed and saved chunk {chunk_start // chunk_size + 1}")

    print("Processing complete. Data saved to", output_file)

    # ✅ Optionally return DataFrame (but loads everything into memory)
    if return_df:
        print("Loading final DataFrame into memory...")
        return pd.read_parquet(output_file, engine="fastparquet")
    
def get_2016_Senate_party_names(state):
    ### reads in the unusual 2016 Formal Prefs csv file, filling in the header column with senate groupings

    #1. LOAD DATA & CLEAN
    SenateCandidates_2016 = pd.read_csv("2016SenateCandidates.csv", index_col = None)
    SenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016["nom_ty"] == 'S',["state_ab","ticket","party_ballot_nm"]]
    SenateCandidates_2016.rename(columns={"state_ab": "StateAb", "party_ballot_nm": "party_nm"}, inplace=True)

    StateSenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016['StateAb'] == state,:]
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].fillna('') # for ungrouped

    # 2. COALITION(S) inspect if there are coalitions and give them PartyAb
    nonUG = StateSenateCandidates_2016.loc[~(StateSenateCandidates_2016['ticket']=='UG'),]
    coalition_df = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1].iloc[:,1:].drop_duplicates(ignore_index=True).groupby("ticket", as_index=False)['party_nm'].agg("/".join)


    if not coalition_df.empty:
        coalition_group_dict = coalition_df.set_index("ticket")["party_nm"].to_dict() #coalition_party_names = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1]['party_nm'].unique()
        # map dictionary
        StateSenateCandidates_2016.loc[:,"party_nm"] = StateSenateCandidates_2016["ticket"].map(coalition_group_dict).where(StateSenateCandidates_2016["ticket"].isin(coalition_group_dict), StateSenateCandidates_2016["party_nm"])

    party_names_list = StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket']!='UG','party_nm'].drop_duplicates(ignore_index=True).tolist()

    # 3. convert to PartyAb and format for Formal Preferences
    party_abvs = abbreviate_party_names(party_names_list, general_party_df)

    party_names_abvs_dict = dict(zip(party_names_list,party_abvs))
    StateSenateCandidates_2016 = StateSenateCandidates_2016.copy() # avoid warning ..?
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].replace(party_names_abvs_dict)
    StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket'] == 'UG','party_nm'] = ''

    # format string column names
    group_party_names = StateSenateCandidates_2016['ticket'].astype(str) + ':' + StateSenateCandidates_2016['party_nm'].astype(str)
    unique_groups = group_party_names[~group_party_names.str.startswith('UG')].drop_duplicates(ignore_index=True)
    group_party_names = unique_groups.tolist() + group_party_names.tolist()

    return group_party_names

def get_2016_Formal_Prefs(state):
    ### reads in the unusual 2016 Formal Prefs csv file, filling in the header column with senate groupings, and expanding the csv string into a df

    # 1. get group party names
    group_party_names = get_2016_Senate_party_names(state)


    # 2. PROCESS Formal_Prefs_file and add columns
    curr_formal_prefs_2016 = pd.read_csv(f"2016FormalPrefs{state}.csv", index_col = None, usecols=['ElectorateNm', 'VoteCollectionPointNm','Preferences'], dtype={-1: str}).rename(columns = {'ElectorateNm': 'div_nm', 'VoteCollectionPointNm': 'pp_nm'})
    curr_formal_prefs_2016 = curr_formal_prefs_2016.iloc[1:,].reset_index(drop=True) # eliminate ---- row

    last_col = curr_formal_prefs_2016.columns[-1]
    # replace any non-int or empty values with 1 (* or /)
    curr_formal_prefs_2016.iloc[:,-1] = curr_formal_prefs_2016.iloc[:,-1].apply(lambda x: ','.join(['1' if not val.isdigit() and val != '' else val for val in x.split(',')]))

    max_col_no = len(group_party_names) # number of candidate/party boxes - group_party_names is, as above, the intended header for the df
    expanded_cols = split_and_convert_2016(curr_formal_prefs_2016.iloc[:,-1],cols = group_party_names)     #expanded_cols = expanded_cols.astype('float32')

    new_column_names = group_party_names

    if len(new_column_names) == expanded_cols.shape[1]:
        expanded_cols.columns = new_column_names

    curr_formal_prefs_2016 = pd.concat([curr_formal_prefs_2016.drop(columns=[last_col]), expanded_cols], axis=1)

    return curr_formal_prefs_2016

def get_2019_NSW_Formal_Prefs(filename, state):
    df_columns = pd.read_csv(filename, index_col=None, nrows=1).columns
    # fix malformed file using readlines:
    expected_columns = len(df_columns)  # You can adjust this number to match your actual expected columns
    batch_size = 1_000_000  # Process 1 million rows at a time
    dataframes = []  # Store DataFrames in a lis

    # Read the CSV file line by line
    with open(filename, 'r') as file:
        
        buffer = io.StringIO()  # Create an in-memory file buffer

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
            
            buffer.write(",".join(map(str, row)) + "\n")
            row_count += 1

            # Process and store each batch
            if row_count % batch_size == 0:
                print(f"Processing batch {len(dataframes) + 1}, rows read: {row_count}")  # Debugging
                buffer.seek(0)  # Move to start of buffer
                df_batch = pd.read_csv(buffer, names=df_columns)
                df_batch.drop(columns=df_batch.columns[FP_ID_COLUMNS], inplace=True)
                dataframes.append(df_batch)

                buffer = io.StringIO()  # Reset buffer
                del df_batch
                gc.collect() # free memory
                batch = []  # free memory
                print("done", time.time() - start)

        # Process any remaining rows
        if buffer.tell() > 0:
            buffer.seek(0)
            df_batch = pd.read_csv(buffer, names=df_columns)
            print(f"Processing final batch, rows read: {row_count}")  # Debugging


            df_batch.drop(columns=df_batch.columns[FP_ID_COLUMNS], inplace=True) # remove id columns
            #df_batch.iloc[:, START_OF_PREFS:] = df_batch.iloc[:, START_OF_PREFS:]#.apply(pd.to_numeric, errors='coerce') #.astype('Int16')
            dataframes.append(df_batch)

            del df_batch
            gc.collect() # free memory # free memory
            print("done", time.time() - start)

    # Combine all batches into a final DataFrame
    if dataframes:
        curr_Formal_prefs = pd.concat(dataframes, ignore_index=True)
        for col in curr_Formal_prefs.columns[3:]:
            curr_Formal_prefs[col] = curr_Formal_prefs[col].astype(np.float32) # convert to float32 when finished
            #print(col)
        dataframes = [] # free memory
        print("Final DataFrame shape:", curr_Formal_prefs.shape)
    else:
        print("No data was read from the file.")

    return curr_Formal_prefs


def optimize_dataframe(df):
    # Convert all remaining numeric columns to smallest possible int type
    df.iloc[:, 3:] = df.iloc[:, 3:].apply(pd.to_numeric, downcast='integer')

    return df


Formal_prefs_dict = {}

if is_redistribution:
    states = states_to_redistribute_dict[data_year]
else:
    states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']

#states = ['NSW','VIC','WA']
#states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
for state in states: # currently only 2016 onwards

    gc.collect()
    print(state)
    filename = f"{data_year}FormalPrefs{state}.csv"

    if data_year in BTL_ONLY_ELECTIONS:
        curr_Formal_prefs = pd.read_csv(f'{data_year}FormalPrefsSampledReduced{state}.csv', index_col = None)


    elif data_year == '2016':
        curr_Formal_prefs = get_2016_Formal_Prefs(state)

        # convert to float 32
        for col in curr_Formal_prefs.columns[2:]:
            curr_Formal_prefs[col] = curr_Formal_prefs[col].fillna(np.nan).astype('float32', copy=False)
        #import pdb;pdb.set_trace()


    elif (state == 'NSW') & (data_year == '2019'):
        curr_Formal_prefs = get_2019_NSW_Formal_Prefs(filename, state) # deal with malformed csv file
        curr_Formal_prefs.drop(columns=["State"], inplace=True)


    else:
        # low memory try - convert to float32 later when using!
        columns = pd.read_csv(filename, nrows=1).columns
        dtype_dict = {col: 'str' for col in columns[:3]}
        for col in columns[3:]:
            dtype_dict[col] = 'float32'

        curr_Formal_prefs = pd.read_csv(filename, index_col=None, na_values=["NaN", "nan"], dtype=dtype_dict)
        curr_Formal_prefs.drop(columns=curr_Formal_prefs.columns[FP_ID_COLUMNS].tolist() + ['State'], inplace=True) # remove id  and State columns

    
    curr_Formal_prefs.rename(columns={"Division": "div_nm", "Vote Collection Point Name": "pp_nm"}, inplace=True)
    curr_Formal_prefs['div_nm'] = curr_Formal_prefs['div_nm'].replace(name_changes_year_dict[data_year]) # adjust for name changes

    for div, group in curr_Formal_prefs.groupby("div_nm"):
        Formal_prefs_dict[div] = group.reset_index(drop=True)  # optimize_dataframe(group.reset_index(drop=True)

    del curr_Formal_prefs

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

def allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict, general_party_df, Senate_party_abvs_dict):

    # produce df or dictionary of dfs with concatenated ATL&uniqueBTL and the First Preference vote in the last column

    Senate_party_abvs_dict = {}

    for div in Formal_prefs_dict.keys():
        #import pdb;pdb.set_trace()
        formal_prefs_full = Formal_prefs_dict[div]
        formal_prefs = formal_prefs_full.iloc[:, START_OF_PREFS:]
        formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
        start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated



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
        Formal_prefs_dict[div] = pd.concat([Formal_prefs_dict[div].iloc[:,:START_OF_PREFS], formal_prefs_first_prefs], axis=1)
        #print(Formal_prefs_dict[div])

    return Formal_prefs_dict

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

list1 = []
list2 = []
incumbency_advantage_dict3 = [] # div: [list of PartyAbs]
list4 = []



# get correpondence between Booth name and pp_id for given year
Polling_Places_df = pd.read_csv(f"{data_year}GeneralPollingPlaces.csv", index_col = None, skiprows = 1)
Polling_Places_df = Polling_Places_df.iloc[:,2:6].rename(columns={'DivisionNm': 'div_nm','PollingPlaceID': 'pp_id','PollingPlaceNm':'pp_nm'})
Polling_Places_df = Polling_Places_df.loc[Polling_Places_df['PollingPlaceTypeID'].isin([1,5]),].drop('PollingPlaceTypeID', axis=1)

for div in Polling_Places_df['div_nm'].unique().tolist():
    Other_row = pd.DataFrame({"div_nm": [div],"pp_id":[0],"pp_nm":["Other"]})
    Polling_Places_df = pd.concat([Polling_Places_df,Other_row], ignore_index=True)

Booth_name_pp_id = Polling_Places_df

Booth_name_pp_id['div_nm'] = Booth_name_pp_id['div_nm'].replace(name_changes_year_dict[data_year])

# First_Prefs_by_PP_Complete = pd.read_csv(f"{data_year}FirstPrefsByPPComplete.csv", index_col = None)
# Booth_name_pp_id = First_Prefs_by_PP_Complete.iloc[:,:3].drop_duplicates()


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


def allocate_Formal_prefs_by_1234(Formal_prefs_dict, Senate_party_abvs_dict, application_dict, by_pp_id = False, as_percent = True):
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
                allocation_set.append(Formal_prefs_dict[div].columns[START_OF_PREFS:START_OF_PREFS+len(Senate_party_abvs_dict[div])][i])  # Append the corresponding group 'letter'
       
        # allocate to allocation_set (already converted to percentages!)
        Final_allocated_pcts_aggregated_dict[div] = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[div], allocation_set, by_pp_id = False, as_percent = True)

        #Final_allocated_pcts_aggregated_dict[div].iloc[:, 1+by_pp_id:] = Final_allocated_pcts_aggregated_dict[div].iloc[:, 1+by_pp_id:].div(Final_allocated_pcts_aggregated_dict[div].drop(columns=['div_nm','pp_id'], errors='ignore').sum(axis=1), axis=0)

    return Final_allocated_pcts_aggregated_dict








######## TO DO: Bring in Candidate_Pairs into here, adjusted for DOP_By_PP
### Change order within DOP_By_PP to match rather arbitrary m-c1-c2 ordering (consistency)
### Apply the function to existing counts, returning final counts!



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

def allocate_Formal_prefs_complex(Formal_prefs_dict, Senate_party_abvs_dict, reduced_votes_by_PP, complex_pair_row, by_pp_id = True):
    # Use FP to get senate versions of A) reduced_votes = c1 B) common votes = m C) expanded votes = c2
    # calculate the % shifts in both A>B,B>C
    # apply to reduced_votes_by_PP --> c2

    c1_m_c2_dict = {}

    #import pdb;pdb.set_trace()


    row = pd.DataFrame([complex_pair_row])

    mlist = row['m_list'][0]
    c1list = row['c1_list'][0]
    c2list = row['c2_list'][0]
    m,c1,c2 = len(mlist),len(c1list),len(c2list)
    giver_div = row['old_div'][0]

    redistribution_votes = reduced_votes_by_PP     # full c1 set to start with - redistribution_votes
    #import pdb;pdb.set_trace()

    # iterate over the 3 lists of PartyAb, apply Formal Preferences, store results in c1_m_c2_dict
    for idx, (col_name, value) in enumerate(row.iloc[:,2:].items()): 
        allocation_abvs_list = value[0] #(= row[col_name]) list of PartyAb to allocate to
        
        #import pdb;pdb.set_trace()
        if idx>=1 and value[0] == row.iloc[:,2:].iloc[0,idx-1]: # c1 or c2 is same as m --> don't need to repeat
            c1_m_c2_dict[col_name] = c1_m_c2_dict[row.columns[2+idx-1]] # copies previous column
        else:
            allocation_set = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, giver_div)

            # allocate to allocation_set and convert to percentages - BE CAREFUL TO DO IT PER ROW AND NOT TOTALLY
            Final_allocated_pcts_aggregated = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[giver_div], allocation_set, by_pp_id, as_percent = True)

            c1_m_c2_dict[col_name] = Final_allocated_pcts_aggregated.iloc[:,1:]
            if by_pp_id:
                c1_m_c2_dict[col_name] = c1_m_c2_dict[col_name].set_index("pp_id").sort_index()

    #import pdb;pdb.set_trace()
    # do all the fancy calculations now
    # 1-> 2. Percentage transfer     # first m candidates the same, remaining c1 - 
    #import pdb;pdb.set_trace()

    # align together redistribution_votes, c1_m_c2_dict['transfer_percent'], c1_m_c2_dict['donation_proportion'],  c1_m_c2_dict['receiving_proportion']

    if c1 > m:
        sum_c1_extras = c1_m_c2_dict['c1_list'].iloc[:,m:].sum(axis=1) # sum values in row for extra c1 candidates (first 2 rows are info) - PPID USED AS INDEX NOW SO ALL NUMERIC
        c1_m_c2_dict['transfer_percent'] = (c1_m_c2_dict['m_list'] - c1_m_c2_dict['c1_list'].iloc[:,:m]).div(sum_c1_extras, axis=0).replace([np.inf, -np.inf, np.nan], 0) # must be positive (unless none are transferred --> no info!)
        c1_m_c2_dict['transfer_percent'].columns = mlist

        extra_c1_parties = list(set(c1list) - set(mlist))
        total_percentages_to_transfer = redistribution_votes[extra_c1_parties].sum(axis=1)

        if c1_m_c2_dict['transfer_percent'].shape[0] == 1: # (by_pp_id = False) - expand to #pp_id rows, adjusting for index
            c1_m_c2_dict['transfer_percent'] = pd.concat([c1_m_c2_dict['transfer_percent'].iloc[0].to_frame().T] * len(total_percentages_to_transfer))
            c1_m_c2_dict['transfer_percent'].index = total_percentages_to_transfer.index

        transfers_by_PP = c1_m_c2_dict['transfer_percent'].mul(total_percentages_to_transfer, axis=0) 
        
        redistribution_votes = redistribution_votes[mlist] + transfers_by_PP # add new transferred values to original redistribution votes
 
    #import pdb;pdb.set_trace()
    if c2 > m:
        # separately, save proportion donated by m parties, and proportions of donation total recieved by extra c2 parties
        c1_m_c2_dict['donation_proportion'] = 1 - (c1_m_c2_dict['c2_list'].iloc[:,:m] / c1_m_c2_dict['m_list'].replace(0, float('nan'))).fillna(0) # avoid division by 0
        sum_c2_etras = c1_m_c2_dict['c2_list'].iloc[:,m:].sum(axis=1)
        c1_m_c2_dict['receiving_proportion'] = c1_m_c2_dict['c2_list'].iloc[:,m:].div(sum_c2_etras, axis=0).replace([np.inf, -np.inf, np.nan], 0) # of c2 extra candidates, get proportion donated to each
        c1_m_c2_dict['donation_proportion'].columns = mlist
        c1_m_c2_dict['receiving_proportion'].columns = c2list[m:]

        if c1_m_c2_dict['receiving_proportion'].shape[0] == 1: # (by_pp_id = False) - expand to #pp_id rows, adjusting for index
            c1_m_c2_dict['receiving_proportion'] = pd.concat([c1_m_c2_dict['receiving_proportion'].iloc[0].to_frame().T] * len(redistribution_votes))
            c1_m_c2_dict['donation_proportion'] = pd.concat([c1_m_c2_dict['donation_proportion'].iloc[0].to_frame().T] * len(redistribution_votes))
            c1_m_c2_dict['receiving_proportion'].index = redistribution_votes.index
            c1_m_c2_dict['donation_proportion'].index = redistribution_votes.index

        total_donation_percentages = redistribution_votes.multiply(c1_m_c2_dict['donation_proportion']).sum(axis=1) # WILL COLUMNS PARTYABS ALIGN???


        receiving_percentages = c1_m_c2_dict['receiving_proportion'].mul(total_donation_percentages, axis=0)
        redistribution_votes = redistribution_votes.multiply(1 - c1_m_c2_dict['donation_proportion']) # proportions remaining for the m parties
        redistribution_votes = pd.concat([redistribution_votes, receiving_percentages], axis = 1)
    
    #import pdb;pdb.set_trace()

    return redistribution_votes


def combine_coalition(all_votes_df):
    # returns df with combined COAL columns, however with order the new COAL column is moved to last place.
    #import pdb;pdb.set_trace()

    coalition_columns = all_votes_df.loc[:,all_votes_df.columns.str.startswith('COAL')] #[all_votes_df.columns.duplicated()]
    coalition_proportions = coalition_columns.div(coalition_columns.sum(axis=1), axis=0).replace([np.inf, -np.inf, np.nan], 0) # very unlikely to be 0

    # convert coalition columns to 'COAL'
    all_votes_df["COAL"] = all_votes_df["COALLP"] + all_votes_df["COALNP"]  # Sum the two columns
    all_votes_df = all_votes_df.drop(columns=["COALLP", "COALNP"])

    return all_votes_df, coalition_proportions 

def separate_coalition(combined_votes_df, coalition_proportions, by_pp_id = True):
    ### returns df where COAL col is separated into COALLP and COALNP according to original proportions
    #import pdb;pdb.set_trace()

    if by_pp_id:
        coalition_columns = combined_votes_df["COAL"].values[:, None] * coalition_proportions
    else:
        coalition_columns = pd.DataFrame(combined_votes_df["COAL"].values[:, None] * coalition_proportions.values, index=combined_votes_df.index, columns = coalition_proportions.columns)

    #import pdb;pdb.set_trace()

    combined_votes_removed_df = combined_votes_df.drop(columns=['COAL'])
    separated_votes_df = pd.concat([combined_votes_removed_df, coalition_columns], axis=1)

    return separated_votes_df

def reduce_candidates_to_set_size(div, Reduce_dict_PP, reduced_c_size, by_pp_id = True, Coalition_double_divs = [], combine_double_divs = True, votes_to_reduce = pd.DataFrame()):
    wide_df1 = Reduce_dict_PP[div]

    if votes_to_reduce.empty:
        Final_Count_Number = wide_df1.iloc[-1,0 + by_pp_id] # last index of CountNumber (2nd column)
        if by_pp_id:
            reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == (Final_Count_Number+2)-reduced_c_size,].set_index('pp_id').iloc[:,1:]  # the correct count number!
        else:
            reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == (Final_Count_Number+2)-reduced_c_size,].iloc[:,1:].reset_index(drop=True)
        #import pdb;pdb.set_trace()

    else:
        reduced_votes_by_PP = votes_to_reduce.copy() # maybe remove INFORMAL column

        initial_candidate_no = len(votes_to_reduce.columns) # pp_id is as index!

        Final_Count_Number = wide_df1.iloc[-1,0 + by_pp_id] # CountNumber is column 0

        start_range = 1 + (Final_Count_Number + 2) - initial_candidate_no # 1 + total num of candidates - c1 # WILL THIS ALWAYS START WITH 0????? CHECK!!!
        end_range = 1 + (Final_Count_Number+2) - reduced_c_size # 1 more than desired for the range indexing

        #import pdb;pdb.set_trace()

        for i in range(start_range, end_range): 
            reduce_df = wide_df1.loc[wide_df1['CountNumber'] == i,].drop('CountNumber', axis = 1) # entire df with pp_ids as rows

            if by_pp_id:
                reduce_df.set_index('pp_id', inplace=True)


            to_reduce_party = reduce_df.columns[reduce_df.iloc[0] == -1].tolist() # party to reduce_div to will have -1 as value
            
            transferred_votes = reduced_votes_by_PP[to_reduce_party] # make into series for multiplication

            if reduce_df.shape[0] == 1: # (by_pp_id = False) - expand to #pp_id rows, adjusting for index
                reduce_df = pd.concat([reduce_df.iloc[0].to_frame().T] * len(transferred_votes))
                reduce_df.index = transferred_votes.index
            #import pdb;pdb.set_trace()
            gained_votes = reduce_df.mul(transferred_votes.values, axis = 0) # ensures only expanded_votes columns are used

            reduced_votes_by_PP = reduced_votes_by_PP.add(gained_votes) # ensures that to_reduce_party is forced to 0!

    zero_columns = reduced_votes_by_PP.columns[reduced_votes_by_PP.eq(0).all()].tolist()

    reduced_votes_by_PP.loc[:,zero_columns] = reduced_votes_by_PP[zero_columns].astype(float) # avoids type warning for 2022 McEwen Casey/Nicholls
    reduced_votes_by_PP.loc[:,zero_columns] = np.nan # convert other cols to nan
    reduced_votes_by_PP = reduced_votes_by_PP.loc[:, ~reduced_votes_by_PP.iloc[0].isna()] # remove nan columns - don't need to store extra c1 candidates!

    if (div in Coalition_double_divs) and combine_double_divs: # HOPEFULLY THIS WILL WORK EVEN IF ONLY 1 IN C1
        #import pdb;pdb.set_trace()
        reduced_votes_by_PP = combine_coalition(reduced_votes_by_PP)[0] # finally, combine 'COALLP' and 'COALNP' into 'COAL'
    elif div in Coalition_double_divs:
        reduced_votes_by_PP.rename(columns={'COALNP':'COAL', 'COALLP':'COAL'}, inplace=True)

    #import pdb;pdb.set_trace()

    return reduced_votes_by_PP

def expand_candidates_to_set_size(div, reduced_votes_by_PP, DOP_div_expand_dict, DOP_div_pref_percent_dict, c_size, expanded_c_size, Coalition_double_divs = [], combine_double_divs = True):
    ### expands candidate set to expanded_c_size, unfortunately using the DOP of the whole div as opposed to by pp_id
    ### if Coalition_double_divs, expects combined data if combine_double_divs = True


    wide_df_expand = DOP_div_expand_dict[div]

    expanded_votes = reduced_votes_by_PP

    if (div in Coalition_double_divs) and combine_double_divs: #   MIGHT FAIL IF INDEPENDENT EXPAND ALREADY DID IT, BUT TRY!
        #import pdb;pdb.set_trace()
        coalition_proportions = combine_coalition(reduce_candidates_to_set_size(div, DOP_div_pref_percent_dict, c_size, by_pp_id=False))[1] # get coalition proportions of div2; DON'T PASS Coalition_double_divs SO IT RETURNS SEPARATED!!!
        expanded_votes = separate_coalition(expanded_votes, coalition_proportions, by_pp_id=False) # split coalition into 2 again
        #c_size += 1
        # I THINK THAT NEEDED TO ADD 1 TO C_SIZE, ASSUMING THAT C2 DOES NOT INCLUDE DUPLICATED COAL.
    elif div in Coalition_double_divs:
        # just rename the one that exists!
        import pdb;pdb.set_trace()
        expanded_votes.rename(columns={'COAL':'COALLP','COAL':'COALNP'}, inplace=True) # IS THIS A MISTAKE?????????
        

    Final_Count_Number = wide_df_expand.iloc[-1,0] # CountNumber is column 0
    if expanded_c_size == 'full': # specify the full size if previously unknown what the full size is
        expanded_c_size = Final_Count_Number + 2

    start_range = 1 + (Final_Count_Number + 2) - expanded_c_size # 1 + total num of candidates - c1
    end_range = 1 + (Final_Count_Number+2) - c_size # 1 more than desired for the range indexing

    #import pdb;pdb.set_trace()

    for i in reversed(range(start_range, end_range)): #(i.e. from count 4 to count 1, where the difference 4-1=c1-m)
        expand_div = wide_df_expand.loc[wide_df_expand['CountNumber'] == i,].drop('CountNumber', axis = 1) # single row of df 

        #import pdb;pdb.set_trace()

        to_expand_party = expand_div.columns[expand_div.iloc[0] == -1].tolist() # party to expand to will have -1 as value
        
        expand_div = expand_div.iloc[0,:] # make into series for multiplication
        lost_votes = expanded_votes.mul(expand_div.reindex(expanded_votes.columns).values) # ensures only expanded_votes columns are used

        expanded_votes = expanded_votes.subtract(lost_votes)
        expanded_votes.loc[:,to_expand_party] = lost_votes.sum(axis=1)
    #import pdb;pdb.set_trace()

    return expanded_votes

def simple_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict,DOP_div_expand_dict, m,c1,c2, Coalition_double_divs = [], combine_double_divs = True, votes_to_reduce=pd.DataFrame(), by_pp_id = True):

    #import pdb;pdb.set_trace()
    if (div1 in Coalition_double_divs) and combine_double_divs:
        reduced_votes_by_PP = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, m+1, by_pp_id, Coalition_double_divs, votes_to_reduce=votes_to_reduce) # for simple redistribution, weakest COAL member must be in top m, hence use m+1
        # reduced to m+1, then combined
        #reduced_votes_by_PP = combine_coalition(reduced_votes_by_PP)[0] # don't need to store combined proportions
        #elif div1 in Coalition_double_divs: 
        # reduce only to m!
        #reduced_votes_by_PP = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, m)
    else:
        reduced_votes_by_PP = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, m, by_pp_id, Coalition_double_divs, combine_double_divs, votes_to_reduce=votes_to_reduce)

    #import pdb;pdb.set_trace()
    #if (div2 in Coalition_double_divs) and combine_double_divs:
    #    coalition_proportions = combine_coalition(reduce_candidates_to_set_size(div2, DOP_By_PP_Pref_Percent_wide_dict, m+1))[1] # get coalition proportions of div2
    #    reduced_votes_by_PP = separate_coalition(reduced_votes_by_PP, coalition_proportions) # split coalition into 2 again
    #    m = m + 1 # start from m+1 candidates
    #elif div2 in Coalition_double_divs:
    #    # Change COAL into COALLP or COALNP - only one is in the m counts
    #    m_parties_count = DOP_div_pref_percent_dict[div2].loc[DOP_div_pref_percent_dict[div2]['CountNumber'] == c2-m,].drop(columns=['CountNumber'])
    #    m_parties = m_parties_count.loc[m_parties_count.index[0]].gt(0).index[m_parties_count.loc[m_parties_count.index[0]] > 0].tolist()
    #    if 'COALLP' in m_parties:
    #        reduced_votes_by_PP.rename(columns={'COAL':'COALLP'})
    #    else:
    #        reduced_votes_by_PP.rename(columns={'COAL':'COALNP'})

    if m < c2:
        expanded_votes_by_PP = expand_candidates_to_set_size(div2, reduced_votes_by_PP, DOP_div_expand_dict, DOP_div_pref_percent_dict, m, c2, Coalition_double_divs, combine_double_divs)
    else:
        expanded_votes_by_PP = reduced_votes_by_PP
    
    return expanded_votes_by_PP

def complex_redistribution(div1,div2, DOP_By_PP_Pref_Percent_wide_dict, complex_pair_row, c1_votes = None, by_pp_id = True, Coalition_double_divs = [], combine_double_divs = True, votes_to_reduce = pd.DataFrame()):
    ### returns votes reallocated to c2, using FPA
    #import pdb;pdb.set_trace()

    if isinstance(c1_votes, pd.DataFrame): # already have reduced votes due to independent

        reduced_votes_by_PP = c1_votes 

        #if div1 in Coalition_double_divs: # might have already been done! In reduce!
        #    # combine the 2 cols into 1
        #    reduced_votes_by_PP = combine_coalition(reduced_votes_by_PP)[0]


    else: # need to construct reduced votes to c1 directly (no independent)
        m = len(complex_pair_row['m_list'])
        c1 = len(complex_pair_row['c1_list'])
        c2 = len(complex_pair_row['c2_list'])

        if (len(Coalition_double_divs) == 1) and (div1 in Coalition_double_divs):
            combine_double_divs = False if sum(p in complex_pair_row['c1_list'] for p in ['COALNP','COALNP']) == 1 else True # ONLY RELEVANT FOR LEN(combine_double_divs) == 1

        reduced_votes_by_PP = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1, by_pp_id, Coalition_double_divs, combine_double_divs, votes_to_reduce=votes_to_reduce) # want to reduce div1 to c1 candidates

    #import pdb;pdb.set_trace()

    # if complex_pair_row has any COALLP/COALNPs, now is the time to convert them to COALS for comparison to senate
    if Coalition_double_divs:
        cand_lists = ['m_list','c1_list','c2_list']
        for lst in cand_lists:
            complex_pair_row[lst] = list(dict.fromkeys(p if p not in ['COALLP', 'COALNP'] else 'COAL' for p in complex_pair_row[lst]))

    #import pdb;pdb.set_trace()

    # apply First Preferences allocation
    allocated_votes_to_c2 = allocate_Formal_prefs_complex(Formal_prefs_dict, Senate_party_abvs_dict, reduced_votes_by_PP, complex_pair_row, by_pp_id = by_pp_id)
    

    #import pdb;pdb.set_trace()

    #I THINK THIS SECTION CAN JUST BE ELIMINATED - WANT OUTPUT OF COMPLEX TO BE COMBINED, WHICH IT ALREADY IS!!!
    #if (div2 in Coalition_double_divs) and combine_double_divs:
    #    import pdb;pdb.set_trace()
    #    # separate the COAL column into 2
    #    coalition_proportions = combine_coalition(reduce_candidates_to_set_size(div2, DOP_By_PP_Pref_Percent_wide_dict, c2+1, Coalition_double_divs = [], combine_double_divs = False))[1] # get coalition proportions of div2
    #    allocated_votes_to_c2 = separate_coalition(reduced_votes_by_PP, coalition_proportions) # split coalition into 2 again 
    #    #   SHOULD IT BE C2 OR C2+1? C2+1!
    #elif div2 in Coalition_double_divs:
    #    allocated_votes_to_c2.rename(columns={'COALLP':'COAL','COALNP':'COAL'}, inplace=True)

    return allocated_votes_to_c2


def get_incumbency_advantage(div, top_party_list, party_category_dict, data_year, incumbent_party, incumbent_years):
    # estimates INCUMBENT_ADVANTAGE and NONINCUMBENT_DISADVANTAGE for each of the top x parties in div's top x. Mean estimates obtained from regression in R, using code from Incumbency Advantage Analyse
    # incumbent_party names are from {data_year}Incumbents, so they are in PartyAb form - convenient for our purposes! XEN/KAP will be treated as 'Other'

    Demographic_Classification_State_df = pd.read_csv(f'{data_year}DemographicClassification.csv', index_col=None)
    Demographic = Demographic_Classification_State_df.loc[Demographic_Classification_State_df['div_nm']==div,'Demographic'].iloc[0]

    if FINAL_CANDIDATE_NO == 5:

        # from the linear model PartyCat + elections_won*Demographic
        intercept = 4.7503
        ALP = -1.2831
        LNP = -1.9874
        LP = -3.2023
        elections_won = 0.5673
        Rural = 1.0630
        OuterMetropolitan = 2.2023
        Provincial = 1.5914
        elections_won_Rural = -0.2173
        elections_won_OuterMetropolitan = -0.7526
        elections_won_Provincial = -0.4806

        IND_estimate = 2.6227 # this is an average of all starting House-Senate differences (Inner Metropolitan and elections_won == 0) - calculated in Incumbency_Advantage_Analyse

        # calculate incremental election_won boost
        if Demographic=='Inner Metropolitan':
            elections_won_boost = 0.5673  
        elif Demographic=='Outer Metropolitan':
            elections_won_boost = 0.5673 - 0.7526
        elif Demographic == 'Provincial':
            elections_won_boost = 0.5673 - 0.4806
        else:
            elections_won_boost = 0.5673 - 0.2173


        # First, calculate incumbent_advantage
        if incumbent_party.startswith('IND'):
            INCUMBENT_ADVANTAGE = IND_estimate
        else:
            INCUMBENT_ADVANTAGE = intercept + ALP*(incumbent_party == 'ALP') + LNP*(incumbent_party in ['LNP','CLP','LNQ']) + LP*(incumbent_party == 'LP')

        INCUMBENT_ADVANTAGE += Rural*(Demographic == 'Rural') + OuterMetropolitan*(Demographic == 'Outer Metropolitan') + Provincial*(Demographic == 'Provincial')

        INCUMBENT_ADVANTAGE += elections_won_boost * incumbent_years



        # Non-incumbent disadvantage average estimates - from linear model lm(Diff_Pct ~ incumbent_party * Ideology, data = df2), where ALP, combined with 'Other' is set as reference
        intercept = -3.6114
        LNP = 4.0662    
        LP = 3.7129  
        Centre = 2.3173 
        COAL = 1.7458  
        Left = 1.5484 
        Right = 3.2802
        LNP_Centre = np.nan  
        LP_Centre = -3.2613 
        LP_COAL = -1.2323   
        LNP_Left = -4.2932 
        LP_Left = -3.3766
        LNP_Right = -5.0229 
        LP_Right = -4.1333 

        LNP_Centre = LP_Centre # best guess - should be similar

        # now, estimates for IND incumbent: averages obtained from Incumbency_Advantage_Analyse - combine additive and interaction terms for IND incumbents!
        # IND_Left, IND_ALP, IND_Centre, IND_COAL, IND_Right = -1.9984, -0.2225, -1.0298, -1.7475, -0.6298
        IND_Ideology_disadvantage = {'Left': -1.9984, 'ALP': -0.2225, 'Centre': -1.0298, 'COAL': -1.7475, 'Right': -0.6298}


        non_incumbents = [p if not p.endswith(div) else p.removesuffix(div)for p in top_party_list if p != incumbent_party] # original PartyAbs
        corresponding_Ideo_Categories = [party_category_dict[p] for p in non_incumbents]

        average_disadvantage_list = []

        for p_cat in corresponding_Ideo_Categories:
            if not incumbent_party.startswith('IND'):
                average_disadvantage = intercept + LNP*(incumbent_party in ['LNP','CLP','LNQ']) + LP*(incumbent_party == 'LP')
                average_disadvantage += (Centre*(p_cat == 'Centre') + COAL*(p_cat == 'COAL') + Left*(p_cat == 'Left') + Right * (p_cat == 'Right'))
                average_disadvantage += LP_Centre*((p_cat == 'Centre') & (incumbent_party in ['LP', 'LNP','CLP','LNQ'])) 
                average_disadvantage += LP_COAL*((p_cat == 'COAL')  & (incumbent_party == 'LP'))
                average_disadvantage += (LNP_Left*((p_cat == 'Left') & (incumbent_party in ['LNP','CLP','LNQ'])) + LP_Left*((p_cat == 'Left') & (incumbent_party == 'LP')))
                average_disadvantage += (LNP_Right*((p_cat == 'Right') & (incumbent_party in ['LNP','CLP','LNQ'])) + LP_Right*((p_cat == 'Right') & (incumbent_party == 'LP')))
            else:
                average_disadvantage = IND_Ideology_disadvantage[p_cat]

            average_disadvantage_list.append(average_disadvantage)

        # ensures that the sum of the swings negate the incumbent advantage!
        curr_sum = INCUMBENT_ADVANTAGE - sum(average_disadvantage_list)
        average_disadvantage_list_normalised = np.array(average_disadvantage_list) + curr_sum/(FINAL_CANDIDATE_NO - 1)
        #average_disadvantage_list_normalised = np.array(average_disadvantage_list) / sum(average_disadvantage_list) * INCUMBENT_ADVANTAGE ############### Fails because some can be positive and negative! Unconcstrained!

        NONINCUMBENT_DISADVANTAGE_dict = {}
        i = 0
        for p in top_party_list:
            if p != incumbent_party:
                NONINCUMBENT_DISADVANTAGE_dict[p] = average_disadvantage_list_normalised[i]
                i += 1

    return INCUMBENT_ADVANTAGE, NONINCUMBENT_DISADVANTAGE_dict

def adjust_c1_c2_for_incumbency_adv(div, expand_wide_dict, pref_percent_wide_dict, c, Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, by_pp_id = True, is_inc = True):
    ### adjusts votes of c1 candidate for incumbency advantage. If by_pp_id = False, then single row is used for entire calculation (0+by_pp_id commonly used to take into account
    ### extra column for pp_id if by_pp_id == True)
    # 1. get top 5 cands, 2. reverse incumb.advantage, 3. expand to full vote

    wide_df1 = pref_percent_wide_dict[div]
    wide_df_expand = expand_wide_dict[div] # same div here!

    # Reduce to Final 5
    Final_Count_Number = wide_df1.iloc[-1, 0 + by_pp_id] # last index of CountNumber (2nd column); 0 if calculating for entire division
    reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == Final_Count_Number - (FINAL_CANDIDATE_NO-2),] # when 5 candidates remaining
    reduced_votes_by_PP = reduced_votes_by_PP.drop(columns=[reduced_votes_by_PP.columns[0 + by_pp_id]]) # remove CountNumber col!

    # make zero columns into nan
    zero_columns = reduced_votes_by_PP.iloc[0, 0 + by_pp_id:] == 0
    zero_columns = zero_columns[zero_columns].index.tolist()

    reduced_votes_by_PP.loc[:,zero_columns] = np.nan
    #reduced_votes_by_PP.iloc[:, 1:] = reduced_votes_by_PP.iloc[:, 1:].where(reduced_votes_by_PP.iloc[:, 1:] > 0)
    reduced_votes_by_PP = reduced_votes_by_PP.sort_index()

    remaining_columns = reduced_votes_by_PP.columns[~reduced_votes_by_PP.iloc[0].isna()].tolist()[0 + by_pp_id:]
    #import pdb;pdb.set_trace()

    if not is_inc: # for expand: only need to return up to this point! i.e. don't perform incumbency adjustment
        #import pdb;pdb.set_trace()
        return reduced_votes_by_PP[remaining_columns]
    
    top_5_columns = remaining_columns

    # Reverse incumbency advantage: e.g. -4 to inc, +1 to each non-inc 
    Incumbent_in_div = Incumbency_by_div.loc[Incumbency_by_div['div_nm']==div,'PartyAb'].tolist()

    for party in Incumbent_in_div:

       

        top_party_list = top_5_columns
        incumbent_years = Incumbent_in_div.loc[Incumbent_in_div['PartyAb']==party,'elections_won'].iloc[0]
        INCUMBENT_ADVANTAGE, NONINCUMBENT_DISADVANTAGE_dict = get_incumbency_advantage(div, top_party_list, party_category_dict, data_year, incumbent_party = party, incumbent_years = incumbent_years)

         # fix up any issues with LP/NP and INDs
        if (party in ['LP','NP']) & ((div_to_state_dict[div] in ['VIC','NSW']) | ((data_year == '2007') & (div_to_state_dict[div] == 'QLD'))):
            party = 'COAL'

        if party.startswith('IND'):
            party = party + div # should be correct IND number (i.e. IND1) as Incumbents_by_div created using 2022 data

        reduced_votes_by_PP[party] -= INCUMBENT_ADVANTAGE
        reduced_votes_by_PP.update(reduced_votes_by_PP[list(NONINCUMBENT_DISADVANTAGE_dict)].add(NONINCUMBENT_DISADVANTAGE_dict)) # adds correct disadvantage

    #import pdb;pdb.set_trace()
    # expand to c1
    expanded_votes = reduced_votes_by_PP

    start_range = 1 + (Final_Count_Number + 2) - c # Final_Count_Number + 2 =  total num of candidates; ensures if c1 = FINAL_CANDIDATE_NO, nothing is done
    end_range = 1+ (Final_Count_Number+2) - FINAL_CANDIDATE_NO # 1 more than desired for the range indexing

    for i in reversed(range(start_range, end_range)): #(i.e. from count 4 to count 1, where the difference 4-1=c1-m)
        expand_div = wide_df_expand.loc[wide_df_expand['CountNumber'] == i,].drop('CountNumber', axis = 1)

        to_expand_party = expand_div.columns[(expand_div.iloc[0] == -1)].tolist()[0] # party to expand to will have -1 as value

        if by_pp_id:
            lost_votes = expanded_votes.set_index("pp_id").multiply(expand_div.set_index("pp_id")).reset_index()
            expanded_votes = expanded_votes.set_index("pp_id").subtract(lost_votes.set_index("pp_id")).reset_index()
        else: # no need to set pp_id index!
            lost_votes = expanded_votes.reset_index(drop=True)*expand_div.reset_index(drop=True)
            expanded_votes = expanded_votes.reset_index(drop=True).subtract(lost_votes)
        
        expanded_votes[to_expand_party] = lost_votes.iloc[:,0 + by_pp_id:].sum(axis=1).values
        
    #import pdb;pdb.set_trace()

    return expanded_votes

def independent_redistribution_reduce(div, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, Reduce_dict_PP, DOP_By_PP_Pref_Percent_wide_dict, c,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP, votes_to_expand = None, Coalition_double_divs = [], combine_double_divs = True, votes_to_reduce=pd.DataFrame(), IND_VOTES_ONLY = False, by_pp_id = True):
    ### adjusts initial house votes (for c candidates) for incumbency advantage via adjust_c1_c2_for_incumbency_adv, compares to the senate (for c- candidates)
    ### c should include any independent candidates
    ### distinction between Reduce_dict_PP and DOP_By_PP_Pref_Percent_wide_dict is that Reduce_dict_PP varies depending on whether need to use Reduce or Pref_Percent.
    ### for incumbency advantage to estimate non-senate proportions, must use DOP_By_PP_Pref_Percent_wide_dict; but at the end of function, when applying proportions
    ### to votes_to_reduce, then use Reduce_dict_PP, which will be DOP_By_PP_Reduce_wide_dict if votes_to_reduce exists.
    
    ### Needs to output combined COAL parties if Coalition_double_divs
    #import pdb;pdb.set_trace()

    # adjust house votes for incumbency advantage (unless no incumbent or non-senate is incumbent!) - First convert LP,NP in VIC/NSW to COAL
    if not Incumbency_by_div.loc[Incumbency_by_div["div_nm"] == div,'div_nm'].empty:

        incumbents = Incumbency_by_div.loc[Incumbency_by_div["div_nm"] == div,'PartyAb']
        if div_to_state_dict[div] in ['VIC','NSW']:
            incumbents.replace('NP','COAL', inplace=True)
            incumbents.replace('LP','COAL', inplace=True)

        # if all incumbents are non-senate candidates, no need for incumbency advantage
        if all(inc not in list_div1_FP for inc in incumbents):
            house_votes = reduce_candidates_to_set_size(div, DOP_By_PP_Pref_Percent_wide_dict, c, by_pp_id=by_pp_id)

        else: # need incumbent advantage adjustment
            house_votes = adjust_c1_c2_for_incumbency_adv(div, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, by_pp_id = by_pp_id)
            if by_pp_id:
                house_votes = house_votes.set_index('pp_id')

    else: # empty incumbent - no need for incumbency advantage
        house_votes = reduce_candidates_to_set_size(div, DOP_By_PP_Pref_Percent_wide_dict, c, by_pp_id=by_pp_id)

    # make sure totals add to 100%, not any off! And replace ones with no house votes with 0 to avoid warning
    house_votes = house_votes.mul(100/house_votes.sum(axis=1), axis=0).replace([np.nan, np.inf, -np.inf], 0) 



    # if 2 COAL parties i.e. duplicated columns (currently only NP/LP, but could be other coalitions!!!), combine their house votes
    if (div in Coalition_double_divs) and combine_double_divs:
        #house_votes, coalition_proportions = combine_coalition(house_votes)
        house_votes = combine_coalition(house_votes)[0]
    # if not combine_coalition, then only 1 COAL member in c1 - hence this just needs to be renamed, and stored to be put back later! This ignores dormant NaN columns.
    elif div in Coalition_double_divs:
        first_COAL = next((col for col in ['COALLP', 'COALNP'] if col in house_votes.columns and not pd.isna(house_votes.at[house_votes.index[0], col])), None)
        if first_COAL:
            house_votes.rename(columns={first_COAL:'COAL'}, inplace=True) 



    if div in Coalition_double_divs:
        #list_div1_FP_combined = list(dict.fromkeys(p if p not in ['COALLP', 'COALNP'] else 'COAL' for p in list_div1_FP))
        # convert to COAL and remove last instance of COAL
        list_div1_FP_combined = ['COAL' if p in ['COALLP', 'COALNP'] else p for p in list_div1_FP]
        list_div1_FP_combined.reverse()
        list_div1_FP_combined.remove('COAL')
        list_div1_FP_combined.reverse()

        senate_allocation_list = list_div1_FP_combined
        list_div1_FP = list_div1_FP_combined
    else:
        senate_allocation_list = list_div1_FP # list of parties excluding non-senate parties
    allocation_set = convert_partyab_to_senate_group_names(senate_allocation_list, Formal_prefs_dict, Senate_party_abvs_dict, div)

    #import pdb;pdb.set_trace()
    senate_votes = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[div], allocation_set, by_pp_id = by_pp_id, as_percent = True).iloc[:,1:] # only data columns
    if by_pp_id:
        senate_votes = senate_votes.set_index('pp_id').sort_index()
    senate_votes = senate_votes.div(senate_votes.sum(axis=1), axis=0)*100
    
    senate_votes.columns = senate_allocation_list

    # investigate where house vote > senate vote
    Senate_minus_IND_house = senate_votes - house_votes.loc[:,list_div1_FP]
    negative_sum = (Senate_minus_IND_house < 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
    positive_sum = (Senate_minus_IND_house > 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
    positive_sum = positive_sum.replace(0, np.nan) # redundant

    #import pdb;pdb.set_trace()

    proportions = Senate_minus_IND_house.div(positive_sum, axis=0) # negative vals will be damaged, but they will soon be ignored
    Proportion_df = Senate_minus_IND_house + proportions.mul(negative_sum, axis=0)
    Proportion_df[Proportion_df<0] = 0 # set negatives to 0
    #Proportion_df = Senate_minus_IND_house.apply(lambda row: row.where(row < 0, row + (row / row[row > 0].sum(axis=1)[0])*(negative_difference_totals.loc[row.name])), axis=1) # temporarily houses proportions of positive % differences, multiplies by negative difference totals

    if IND_VOTES_ONLY and div == 'Corangamite':
        # For IND_VOTES_ONLY: in case there are non-senates alongside INDs, rescale the allocation to only the IND's portion!
        import pdb;pdb.set_trace()
        IND_columns = [col for col in house_votes.columns if col.startswith('IND')]
        IND_proportion_of_non_senate =  house_votes.loc[:,IND_columns].sum(axis=1)/ Proportion_df.sum(axis=1)
        Proportion_df = Proportion_df.multiply(IND_proportion_of_non_senate, axis=0)

        import pdb;pdb.set_trace()


    sum_c1_extras = 100 - house_votes.loc[:,list_div1_FP].sum(axis=1) # sum values in row for extra c1 candidates
    Transfer_percent = Proportion_df.div(sum_c1_extras, axis = 0).replace([np.inf, -np.inf, np.nan], 0) # if 0 votes in house for INDs, replace everything with 0

    # get actual c1 votes (for len(list_div1_FP) parties) and allocate transfers
    c1_votes = reduce_candidates_to_set_size(div, Reduce_dict_PP, c, by_pp_id, Coalition_double_divs,combine_double_divs, votes_to_reduce=votes_to_reduce)
    non_senate_cands = [cand for cand in (set(c1_votes.columns) - set(list_div1_FP))]
    sum_non_senates = c1_votes.loc[:,c1_votes.columns.isin(non_senate_cands)].sum(axis=1)

    if Transfer_percent.shape[0] == 1: # (by_pp_id = False) - expand to #pp_id rows, adjusting for index
        Transfer_percent = pd.concat([Transfer_percent.iloc[0].to_frame().T] * len(sum_non_senates))
        Transfer_percent.index = sum_non_senates.index

    transferred_votes = Transfer_percent.mul(sum_non_senates, axis=0)
    #import pdb;pdb.set_trace()

    redistribution_votes = c1_votes.loc[:,~c1_votes.columns.isin(non_senate_cands)] + transferred_votes

    # I THINK REMOVING THIS IS FOR THE BEST!
    #if (div in Coalition_double_divs) and combine_coalition:
    #    redistribution_votes = separate_coalition(redistribution_votes, coalition_proportions)
    #elif div in Coalition_double_divs:
    #    redistribution_votes.rename(columns={'COAL':first_COAL}, inplace=True) # only 1 COAL in c2

    return redistribution_votes if not IND_VOTES_ONLY else transferred_votes


def independent_redistribution_expand(div, Formal_prefs_dict, DOP_div_expand_dict, DOP_div_pref_percent_dict, c, Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = None, Coalition_double_divs = [], combine_double_divs = True, cands_to_expand = []):
    ### use whole-div expand and pref percent dicts! CHECK IF VOTES_TO_EXPAND SHOULD BE INITIALISED TO NONE, OR IF AN OLD RELIC?
    #import pdb;pdb.set_trace()

    # c is int corresponding to the final number of candidates!

    # if only partial expansion, then list_div2_FP should be truncated!
    list_div2_FP = list_div2_FP # REDUNDANT if not cands_to_expand else [p for p in list_div2_FP if p in votes_to_expand.columns]

    # adjust house votes for incumbency advantage if is_inc is True (unless no incumbent or non-senate is incumbent!) - First convert LP,NP in VIC/NSW to COAL
    if not Incumbency_by_div.loc[Incumbency_by_div["div_nm"] == div,'div_nm'].empty:

        incumbents = Incumbency_by_div.loc[Incumbency_by_div["div_nm"] == div,'PartyAb']
        if div_to_state_dict[div] in ['VIC','NSW']:
            incumbents.replace('NP','COAL', inplace=True)
            incumbents.replace('LP','COAL', inplace=True)

        # if all incumbents are non-senate candidates, no need for incumbency advantage
        if all(inc not in list_div2_FP for inc in incumbents):
            is_inc = False

        else: # need incumbent advantage adjustment
            is_inc = True 

    else: # empty incumbent - no need for incumbency advantage
        is_inc = False

    FINAL_CANDIDATE_NO = FINAL_CANDIDATE_NO if is_inc else c

    #import pdb;pdb.set_trace()

    house_votes = adjust_c1_c2_for_incumbency_adv(div, DOP_div_expand_dict, DOP_div_pref_percent_dict, c,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, by_pp_id = False, is_inc = is_inc) # do analysis as whole division

    house_votes = house_votes.mul(100/house_votes.sum(axis=1), axis=0).reset_index(drop=True) # make sure totals add to 100%, not any off! Ensure index is 0 as for whole div only


    # if 2 COAL parties i.e. duplicated columns (currently only NP/LP, but could be other coalitions!!!), combine their house votes, or just rename to COAL if latter is irrelevant
    if (div in Coalition_double_divs) and combine_double_divs:
        #house_votes, coalition_proportions = combine_coalition(house_votes)
        house_votes = combine_coalition(house_votes)[0]
    elif div in Coalition_double_divs:
        first_COAL = next((col for col in ['COALLP', 'COALNP'] if col in house_votes.columns and not pd.isna(house_votes.at[house_votes.index[0], col])), None)
        if first_COAL:
            house_votes.rename(columns={first_COAL:'COAL'}, inplace=True) 


    if div in Coalition_double_divs:
        list_div2_FP_combined = list(dict.fromkeys(p if p not in ['COALLP', 'COALNP'] else 'COAL' for p in list_div2_FP))
        senate_allocation_list = list_div2_FP_combined
        list_div2_FP = list_div2_FP_combined
    else:
        senate_allocation_list = list_div2_FP # list of parties excluding non-senate parties
    allocation_set = convert_partyab_to_senate_group_names(senate_allocation_list, Formal_prefs_dict, Senate_party_abvs_dict, div)

    #import pdb;pdb.set_trace()
    senate_votes = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[div], allocation_set, by_pp_id = False, as_percent = True).iloc[:,1:] # only data columns
    senate_votes = senate_votes.div(senate_votes.sum(axis=1), axis=0)*100
    
    senate_votes.columns = senate_allocation_list

    # investigate where house vote > senate vote
    Senate_minus_IND_house = senate_votes - house_votes.loc[:,list_div2_FP]
    negative_sum = (Senate_minus_IND_house < 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
    positive_sum = (Senate_minus_IND_house > 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
    positive_sum = positive_sum.replace(0, np.nan) # redundant


    #import pdb;pdb.set_trace()

    # direction == 'Expand'!
    # This should ensure that house>senate cases are nullfied
    mask = Senate_minus_IND_house > 0
    add_to_house_rebalancing = Senate_minus_IND_house.mask(mask,Senate_minus_IND_house.div(positive_sum, axis=0).mul(negative_sum, axis=0)*-1) # mask to keep negative values untouched!
    house_votes.loc[:,list_div2_FP] += add_to_house_rebalancing # now fully rebalanced

    #import pdb;pdb.set_trace()

    # perform classic expand trick
    donation_proportion = 1 - np.where(senate_votes != 0, house_votes[list_div2_FP] / senate_votes, 0)
    donation_proportion = pd.DataFrame(donation_proportion, columns=house_votes[list_div2_FP].columns) # convert array to df with columns
    #donation_proportion = 1 - house_votes[list_div1_FP] / senate_votes
    #c1_m_c2_dict['donation_proportion'] = 1 - (c1_m_c2_dict['c2_list'].iloc[:,:m] / c1_m_c2_dict['m_list'].replace(0, float('nan'))).fillna(0) # avoid division by 0

    non_senate_cands = [cand for cand in (set(house_votes.columns) - set(list_div2_FP))] # REDUNDANT if not cands_to_expand else cands_to_expand  # MAJOR CHANGE!!!
    sum_non_senates = house_votes.loc[:,house_votes.columns.isin(non_senate_cands)].sum(axis=1)
    

    receiving_proportion = house_votes.loc[:,house_votes.columns.isin(non_senate_cands)].div(sum_non_senates, axis=0) # if multiple non-senates
    total_donation_percentages = (votes_to_expand * donation_proportion.iloc[0]).sum(axis=1) # ensures PartyAb columns align; NEED votes_to_expand TO BE COMBINED IF Coalition_double_divs
    receiving_percentages = total_donation_percentages.to_frame() @ receiving_proportion
    redistribution_votes = votes_to_expand*(1 - donation_proportion.iloc[0]) # proportions remaining for the m parties
    redistribution_votes = pd.concat([redistribution_votes, receiving_percentages], axis = 1)
    #import pdb;pdb.set_trace()

    # IDEAL TO KEEP COMBINED SO THAT EXPAND_TO_SET_SIZE PROCESSES EXPANSION - WILL SEE!

    #if (div in Coalition_double_divs) and combine_coalition:
    #    redistribution_votes = separate_coalition(redistribution_votes, coalition_proportions)
    #elif div in Coalition_double_divs:
    #    redistribution_votes.rename(columns={'COAL':first_COAL}, inplace=True) # only 1 COAL in c2


    return redistribution_votes



def find_c1_c2(elimination_list, common_set):
    ### finds length of subsection of list upon which all elements of common_set are seen
    
    seen = set()
    for c in range(len(elimination_list)):  # Iterate from the start
        if elimination_list[c] in common_set:
            seen.add(elimination_list[c])  # Track seen elements from the set
        if seen == common_set:  # Stop once all elements have appeared
            return c+1 # inedx+1  
        
    return 0/0 # should never happen!

def remove_non_senate_cands(list_div1_c1,list_div2_c2,Senate_parties):
    non_senate_parties_div1 = []

    for party in list_div1_c1:
        if party not in Senate_parties:
            non_senate_parties_div1.append(party)
    list_div1_FP = [x for x in list_div1_c1 if x not in non_senate_parties_div1]

    non_senate_parties_div2 = []
    for party in list_div2_c2:
        if party not in Senate_parties:
            non_senate_parties_div2.append(party)
    list_div2_FP = [x for x in list_div2_c2 if x not in non_senate_parties_div2]

    return list_div1_FP,list_div2_FP, non_senate_parties_div1, non_senate_parties_div2 # returns parties in elimination lists excluding those that are not in the senate, and then the remaining ones


def transform_to_raw_votes(redistributed_votes, giver_div, name_changes_year_dict, data_year, IS_FINAL_TRANSFORMATION = False):

    First_Prefs_By_PP_Complete = pd.read_csv(f"{data_year}FirstPrefsByPPComplete.csv", index_col = None)[['pp_id','div_nm','PartyAb','votes']]
    First_Prefs_By_PP_Complete['div_nm'] = First_Prefs_By_PP_Complete['div_nm'].replace(name_changes_year_dict)
    First_Prefs_By_PP_div = First_Prefs_By_PP_Complete.loc[First_Prefs_By_PP_Complete['div_nm'] == giver_div,].drop(columns = 'div_nm', axis = 1)

    INFORMAL_df = First_Prefs_By_PP_div[First_Prefs_By_PP_div['PartyAb'] == 'INFORMAL'].rename(columns = {'votes':'INFORMAL'}).drop(columns = ['PartyAb'], axis=1).set_index('pp_id').sort_index()
    FORMAL_df = First_Prefs_By_PP_div[First_Prefs_By_PP_div['PartyAb'] != 'INFORMAL']

    grouped_FORMAL = FORMAL_df.groupby('pp_id', as_index=False).agg({'votes': 'sum'}).set_index('pp_id').sort_index()

    # correct for any issues with 0-0 house-senate votes! PERHAPS BETTER TO REMOVE ALTOGETHER???
    grouped_FORMAL = grouped_FORMAL.reindex(redistributed_votes.index, fill_value=0) # ensure no issue when there are 0 votes in House & Senate (excluded from Formal Prefs & First_Prefs_By_PP_div)
    INFORMAL_df = INFORMAL_df.reindex(redistributed_votes.index, fill_value=0)

    mask = redistributed_votes.sum(axis=1) == 0
    redistributed_votes.loc[mask] = redistributed_votes.loc[mask].fillna(0)

    # scale redistributed_votes by grouped_FORMAL
    if not IS_FINAL_TRANSFORMATION:
        redistributed_votes_raw = (redistributed_votes / 100).mul(grouped_FORMAL['votes'], axis=0)
    else:
        if (redistributed_votes['INFORMAL'] != INFORMAL_df['INFORMAL']).sum():
            import pdb;pdb.set_trace()

        redistributed_votes_raw = redistributed_votes.drop('INFORMAL', axis=1)

    if redistributed_votes_raw.isna().any().any():
        import pdb;pdb.set_trace()


    # adjust to get integer values for votes
    redistributed_votes_rounded = redistributed_votes_raw.round().astype(int)
    redistributed_votes_sum = redistributed_votes_rounded.sum(axis=1)
    adjustment = grouped_FORMAL['votes'] - redistributed_votes_sum

    # row by row, adjust rounded votes based on total vote, rounding biggest abusers first
    for i in range(len(redistributed_votes_raw)):
        if adjustment.iloc[i] != 0:
            fractional_part = redistributed_votes_rounded.iloc[i] - redistributed_votes_raw.iloc[i] #- np.floor(redistributed_votes_raw.iloc[i]) # MAY BE ISSUE WITH INCORRECT CHOICE OF OFFENDERS TO ROUND
            order = np.argsort(adjustment.iloc[i] * fractional_part).tolist()  # Sort descending or ascending, based on the sign of adjustment.iloc[i]: + --> negatives first to add some, - --> positives first to remove some
            for idx in order[:abs(adjustment.iloc[i])]:  # Distribute adjustments for first adjustment.iloc[i] in the list
                redistributed_votes_rounded.iloc[i, idx] += np.sign(adjustment.iloc[i])
            # Ensure sum is correct after adjustment
            assert redistributed_votes_rounded.iloc[i].sum() == grouped_FORMAL['votes'].iloc[i]

    # sort columns in alphabetical order, adding on INFORMAL at the end
    redistributed_votes_rounded = redistributed_votes_rounded.sort_index(axis=1) 
    redistributed_votes_raw_plus_informal = pd.concat([redistributed_votes_rounded, INFORMAL_df], axis=1)

    return redistributed_votes_raw_plus_informal


def replace_COAL_with_COALNPLP(party_list):
    # order of COALLP and COALNP shouldnt matter!
    count = 0  # Counter for occurrences
    for i, item in enumerate(party_list):
        if item == 'COAL':
            count += 1
            if count == 1:
                party_list[i] = "COALLP"  # Replace first occurrence
            elif count == 2:
                party_list[i] = "COALNP"  # Replace second occurrence
                break  # Stop after replacing both
    return party_list

def check_Coalition_double_divs_for_simple_case(div1,div2,list_div1, list_div2, m, Coalition_double_divs, c2_dict):
    ### Performs 4 processes/checks: 
    # 1. Check if 2nd COAL happens in the excluded portion?
    # 2. Remove worst performing COAL party and check if simple now
    # 3. For 2 double divs, see if retaining both is simple
    # 4. Else, merge both and check if simple


    first_COAL_simple, combined_COAL_simple, both_separate_COAL_simple, both_combined_COAL_simple = False, False, False, False
    m_dd, list_div1_COAL_combined, list_div2_COAL_combined, list_div1_first_COAL, list_div2_first_COAL = m,[],[],[],[]


    if len(Coalition_double_divs) == 1:

        # check if one of COALNP/LP is an extra, which would make it simple!
        div = Coalition_double_divs[0]

        elimination_order_dd = list_div1.copy() if div == div1 else list_div2.copy()

        # replace first one of COALNP/LP with COAL, check if last one is irrelevant (simple)
        for j in range(len(elimination_order_dd)):
            if elimination_order_dd[j].startswith('COAL'):
                elimination_order_dd[j] = 'COAL'
                break
        # use new versions of list_divis for simplicity check
        list_div1_first_COAL = elimination_order_dd if div == div1 else list_div1 
        list_div2_first_COAL = elimination_order_dd if div == div2 else list_div2

        common_dd = set(list_div1_first_COAL) & set(list_div2_first_COAL) # now one of sets includes one 'COAL' and one 'COALLP'/'COALNP'
        m_dd = len(common_dd) if not c2_dict else len(list_div2)

        if set(list_div1_first_COAL[:m_dd]) == set(list_div2_first_COAL[:m_dd]):
            first_COAL_simple = True

        else:
            # now try combining COALs into the first one (remove the worst performing coalition partner):)
            # NOW, CHECK IF COMBINING COALS WILL HELP - iterating backwards, remove first COAL encountered, rename 2nd.
            elimination_order_dd_combined = list_div1.copy() if div == div1 else list_div2.copy()

            removed = 0
            for j in reversed(range(len(elimination_order_dd_combined))):
                if (not removed) and elimination_order_dd_combined[j].startswith('COAL'):
                    del elimination_order_dd_combined[j]
                    removed = 1
                if removed and elimination_order_dd_combined[j].startswith('COAL'):
                    elimination_order_dd_combined[j] = 'COAL'
                    break
                    
            list_div1_COAL_combined = elimination_order_dd_combined if div == div1 else list_div1 
            list_div2_COAL_combined = elimination_order_dd_combined if div == div2 else list_div2

            common_dd = set(list_div1_COAL_combined) & set(list_div2_COAL_combined) # both sets have only one COAL
            m_dd = len(common_dd) if not c2_dict else len(list_div2)

            if set(list_div1_COAL_combined[:m_dd]) == set(list_div2_COAL_combined[:m_dd]):
                combined_COAL_simple = True

    elif len(Coalition_double_divs) == 2:
        # 1. try simple check if both treated as separate parties
        if set(list_div1[:m]) == set(list_div2[:m]):
            both_separate_COAL_simple = True
            m_dd = m
        else:
            #2. try combine both and check if simple
            if not c2_dict:

                for div in Coalition_double_divs:
                    elimination_order_dd_combined = Elimination_order_dict[div].copy()
                    removed = 0
                    for j in reversed(range(len(elimination_order_dd_combined))):
                        if (not removed) and elimination_order_dd_combined[j].startswith('COAL'):
                            del elimination_order_dd_combined[j]
                            removed = 1
                        if removed and elimination_order_dd_combined[j].startswith('COAL'):
                            elimination_order_dd_combined[j] = 'COAL'
                            break
                    if div == div1:
                        list_div1_COAL_combined = elimination_order_dd_combined # CHECK TO MAKE SURE THAT PUTTING THIS BACK ONE INDENT IS CORRECT EFFECT!!!
                    else:
                        list_div2_COAL_combined = elimination_order_dd_combined

                    common_dd = set(list_div1_COAL_combined) & set(list_div2_COAL_combined) # both sets have only one COAL
                    m_dd = len(common_dd) if not c2_dict else len(list_div2)

                    if set(list_div1_COAL_combined[:m_dd]) == set(list_div2_COAL_combined[:m_dd]):
                        both_combined_COAL_simple = True

            else:
                # main difference - DO NOT use Elimination_order_dict because not accessible for div2. Apply to div1, and then copy the party choice in div2
                elimination_order_dd_combined = list_div1.copy()
                removed = 0
                for j in reversed(range(len(elimination_order_dd_combined))):
                    if (not removed) and elimination_order_dd_combined[j].startswith('COAL'):
                        removed_COAL_party = elimination_order_dd_combined[j]
                        del elimination_order_dd_combined[j]
                        removed = 1
                    if removed and elimination_order_dd_combined[j].startswith('COAL'):
                        elimination_order_dd_combined[j] = 'COAL'
                        break
                list_div1_COAL_combined = elimination_order_dd_combined
                list_div2_COAL_combined = ['COAL' if p.startswith('COAL') else p for p in list_div2 if p != removed_COAL_party] # match original removal

                common_dd = set(list_div1_COAL_combined) & set(list_div2_COAL_combined) # both sets have only one COAL
                m_dd = len(common_dd) if not c2_dict else len(list_div2_COAL_combined)

                if set(list_div1_COAL_combined[:m_dd]) == set(list_div2_COAL_combined[:m_dd]):
                    both_combined_COAL_simple = True
    
    if first_COAL_simple + combined_COAL_simple + both_separate_COAL_simple + both_combined_COAL_simple:
        import pdb;pdb.set_trace()

    return first_COAL_simple, combined_COAL_simple, both_separate_COAL_simple, both_combined_COAL_simple, m_dd, list_div1_COAL_combined, list_div2_COAL_combined, list_div1_first_COAL, list_div2_first_COAL


def create_new_seat_party_dicts(data_year):

    # 2007 Flynn is tricky, as both LP and NP contested!!!
    new_seat_party_dicts = {'2022':{'Bullwinkel':['LP','ALP','GRN','UAPP','ON']},'2019':{'Hawke':['ALP','COAL','GRN','UAPP']},'2016':{'Bean':['ALP','LP','GRN'],'Fraser':['ALP','COAL','GRN']},'2013':{'Burt':['ALP','LP','GRN']},'2010':{},'2007':{'Wright':['ALP','LNP','GRN']},'2004':{'Flynn':['ALP','COALLLLLL','GRN']},'2001':{'Bonner':['ALP','LP','GRN'],'Gorton':['ALP','LP','GRN']}}

    return new_seat_party_dicts[data_year]

def split_by_special(tail_list, non_senate_parties_div2):
    #I THINK REDUNDANT NOW
    sections = []
    for is_non_senate, group in groupby(tail_list, key=lambda x: x in non_senate_parties_div2):
        sections.append((is_non_senate, list(group)))  # Store whether it's special & the group itself
    return sections

def check_staircase_expansion(non_senate_parties_div2, reduced_votes, list_div2_FP, list_div2):
    #I THINK REDUNDANT NOW
    
    # checks if conditions are met to require a staircase expansion of independent_expand, then normal expand, then independent expand etc.

    staircase_steps = []
    staircase_required = 0
    pre_staircase_parties = []
    # if more than one Non-Senate, and there will be a further expansion, and there is a non-senate party after the first to-expand party

    if (len(non_senate_parties_div2) > 1) and (len(reduced_votes.columns) != len(list_div2_FP)):
        index_to_expand = list_div2.index(list_div2_FP[len(reduced_votes.columns)]) # gets elim_order index of party in list_div2FP that is first to expand
        if any(non_sen_p in list_div2[index_to_expand + 1:] for non_sen_p in non_senate_parties_div2):
            staircase_required = 1
            staircase_steps = split_by_special(list_div2[index_to_expand:], non_senate_parties_div2)

            if staircase_steps[-1][0]: # if last step is True, then can group it with second last step (guaranteed to exist) as ordinary expansion enough
                staircase_steps[-2][1].extend(staircase_steps[-1][1])
                staircase_steps.pop(-1)

            # non-senate parties occurring strictly before index_to_expand in list_div2
            pre_staircase_parties = [nsp for nsp in set(list_div2[:index_to_expand]) & set(non_senate_parties_div2)]

    return staircase_steps, staircase_required, pre_staircase_parties


def get_IND_latent_votes(list_div1, div1, Formal_prefs_dict, Expand_dict_PP, Reduce_dict_PP, DOP_By_PP_Pref_Percent_wide_dict,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year , votes_to_reduce, by_pp_id, Coalition_double_divs):

    ### allocate independent votes among all other parties, to track IND votes via votes_to_reduce

    c = len(list_div1)
    list_div1_FP = [p for p in list_div1 if not p.endswith(div1)] # Only INDs removed
    list_div1_INDs = [p for p in list_div1 if p.startswith('IND')]

    # MAKE A SEPARATE CLAUSE FOR CASES WHERE ALL INDS ARE REMOVED IMMEDIATELY - CAN JUST USE REDUCE_DICT_PP


    # apply Coalition_double_divs: Only 2 cases - either both parties remain (simple), or converted to COAL. Assuming conversion to COAL. If both end up remaining, then produce a separate version
    votes_to_reduce = independent_redistribution_reduce(div1, Formal_prefs_dict, Expand_dict_PP, Reduce_dict_PP, DOP_By_PP_Pref_Percent_wide_dict, c,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP, Coalition_double_divs=Coalition_double_divs, votes_to_reduce=votes_to_reduce, IND_VOTES_ONLY = True, by_pp_id=by_pp_id)
    #votes_to_reduce[list_div1_INDs] = 0.0
    # add a col for non-IND non-senates as 0
    non_IND_non_senates = [p for p in list_div1 if (p.endswith(div1) and (not p.startswith('IND')))]
    votes_to_reduce[list_div1_INDs + non_IND_non_senates] = 0.0

    votes_to_reduce.loc[:,(votes_to_reduce == 0.0).all()] = SMALL_CONST_FOR_LATENT_IND

    # replace 0 cols with small positive constant to ensure they don't get removed by reduce_candidates_to_set_size!


    if Coalition_double_divs:
        import pdb;pdb.set_trace()
        coalition_proportions = combine_coalition(reduce_candidates_to_set_size(div1, DOP_div_pref_percent_dict, c, by_pp_id=False))[1] # get coalition proportions of div2; DON'T PASS Coalition_double_divs SO IT RETURNS SEPARATED!!!
        votes_to_reduce = separate_coalition(votes_to_reduce, coalition_proportions, by_pp_id=False) # split coalition into 2 again
        import pdb;pdb.set_trace()


    return votes_to_reduce



def full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, c2_dict = {}, votes_to_reduce_dict = False, IND_VOTES_ONLY = False, by_pp_id = True, older_div = ''):
    # input is df with columns corresponding to giver division and receiver division, respectively. Returns a dictionary with new div names as keys, and a df of the full redistributed votes for the original giver's votes for each pp_id as values 
    # all redistribution pairs

    Elimination_order_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Reduce_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, DOP_div_expand_dict, DOP_div_reduce_dict, DOP_div_pref_percent_dict = list_of_DOP_dicts

    Reduce_dict_PP = DOP_By_PP_Reduce_wide_dict if (votes_to_reduce_dict or IND_VOTES_ONLY) else DOP_By_PP_Pref_Percent_wide_dict # if IND_VOTES_ONLY, votes_to_reduce created later
    reduce_dict_whole = DOP_div_reduce_dict if (votes_to_reduce_dict or IND_VOTES_ONLY) else DOP_div_pref_percent_dict
    Expand_dict_PP = DOP_By_PP_Expand_wide_dict
    expand_dict_whole = DOP_div_expand_dict

    if not by_pp_id:
        Reduce_dict_PP = reduce_dict_whole
        Expand_dict_PP = expand_dict_whole
        DOP_By_PP_Pref_Percent_wide_dict = DOP_div_pref_percent_dict


    # If in new_candidates_allocation mode, then main modification is that c2 is no longer obtianed form Elimination order, but from c2_dict, with no expansion allowed.

    First_Prefs_By_PP_Complete_Redistributed_dict = {}

    columns_list = Redistribution_pairs_df.columns.tolist()
    columns_list.extend(['c1_list','c2_list'])

    simplerd, simpleindrd, complexrd = 0,0,0


    for i in range(Redistribution_pairs_df.shape[0]): # 

        div1, div2 = Redistribution_pairs_df.iloc[i].tolist() # get giver and receiver for pair i

        #if div1 not in ['Banks','Barton','Batman','Bendigo']:
        #    continue

        votes_to_reduce = votes_to_reduce_dict[div1] if votes_to_reduce_dict else pd.DataFrame()

        Senate_parties = Senate_parties_by_div.loc[Senate_parties_by_div['div_nm'] == div1,"PartyAbList"].iloc[0]

        new_seat_party_dicts = create_new_seat_party_dicts(data_year)

        if c2_dict:
            # force list_div2 to come from new_candidates!
            list_div1 = Elimination_order_dict[div1].copy()
            list_div2 = c2_dict[div2 if '_' not in div2 else div2.split('_')[0]].copy()
        else:
            # get elimination orders from last election, adjusting if this is new seat. This will always have c2 = m, so no issue that new seats are not a key in all the dicts
            if div2 in new_seats_list:
                print("new seat", div1,div2)
                Elimination_order_dict[div2] = new_seat_party_dicts[div2] #['LP','ALP','GRN','UAPP','ON'] ## Manually generalise to each new seat, naming COAL or LP or NP as appropriate
                list_div2 = Elimination_order_dict[div2]
                list_div1 = Elimination_order_dict[div1].copy()
            else:
                list_div1, list_div2 = Elimination_order_dict[div1].copy(), Elimination_order_dict[div2].copy() # get elimination orders



        # separate out COALNP and COALLP where there are 2: ElimOrder, Expand, PrefPercent (x2)
        Coalition_double_divs = []
        if 'COALNP' in list_div1: # CHECK IF IT IS OK TO REPLACE ORIGINAL Elimination_order_dict[div1] with list_div1
            # we have double div!
            Coalition_double_divs.append(div1)

        if 'COALNP' in list_div2:
            # if there are newly 2 COAL parties (weren't there last time), replace with just COAL, and deal with later!
            if c2_dict and ('COALNP' not in list_div1):
                print("new COAL double div - estimate proportions later!", div1, div2, list_div2)
                for i, party in enumerate(list_div2):
                    if (party=='COALNP') | (party =='COALLP'):
                        list_div2[i] = 'COAL'
                list_div2 = list(set(list_div2)) # reduces duplicate COALs
                
            else:
                # we have double div!
                Coalition_double_divs.append(div2)




        # IF ANY INDS ARE INVOLVED in DIV1, CALCULATE THIER LATENT VOTES STORED IN OTHER PARTIES
        if IND_VOTES_ONLY and any(item.startswith("IND") for item in list_div1):

            # do not yet have votes_to_reduce - need to first reset to Pref_Percent, then later correct back to Reduce
            if votes_to_reduce.empty:
                temp_Reduce_dict = DOP_By_PP_Pref_Percent_wide_dict if by_pp_id else DOP_div_pref_percent_dict
            else:
                temp_Reduce_dict = Reduce_dict_PP

            votes_to_reduce = get_IND_latent_votes(list_div1, div1, Formal_prefs_dict, Expand_dict_PP, temp_Reduce_dict, DOP_By_PP_Pref_Percent_wide_dict,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, votes_to_reduce=votes_to_reduce, by_pp_id=by_pp_id, Coalition_double_divs=Coalition_double_divs)

            #import pdb;pdb.set_trace()


        #import pdb;pdb.set_trace()

        set1,set2 = set(list_div1), set(list_div2)
        common = set1 & set2
        m = len(common) if not c2_dict else len(list_div2) # if new candidates, then list_div2 is this smallest minimum!

        # IF ONE OR BOTH DIVS ARE A COALITION_DOUBLE_DIV, CHECK IF IT'S SIMPLE AT HEART: EITHER ONE COAL IS AN EXTRA, OR WHEN COALS ARE COMBINED, IT IS SIMPLE
        first_COAL_simple, combined_COAL_simple, both_separate_COAL_simple, both_combined_COAL_simple, m_dd, list_div1_COAL_combined, list_div2_COAL_combined, list_div1_first_COAL, list_div2_first_COAL= check_Coalition_double_divs_for_simple_case(div1,div2,list_div1, list_div2, m, Coalition_double_divs, c2_dict)

        # find common candidates, see if simple/complex
        if (set(list_div1[:m]) == set(list_div2[:m])) | (first_COAL_simple + combined_COAL_simple + both_separate_COAL_simple + both_combined_COAL_simple):
            
            #import pdb;pdb.set_trace()


            # SIMPLE REDISTRIBUTION
            print("simple", div1,div2)
            simplerd += 1

            if (set(list_div1[:m]) == set(list_div2[:m])) or both_separate_COAL_simple:
                c1 = len(list_div1)
                c2 = len(list_div2)
                redistributed_votes = simple_redistribution(div1,div2,Reduce_dict_PP, expand_dict_whole, m,c1,c2, votes_to_reduce=votes_to_reduce, by_pp_id = by_pp_id)

            else:
                # determine which it is. If 
                m = m_dd 
                if combined_COAL_simple + both_combined_COAL_simple:
                    # combine COALs together
                    c1, c2 = len(list_div1_COAL_combined), len(list_div2_COAL_combined)
                    redistributed_votes = simple_redistribution(div1,div2,Reduce_dict_PP, expand_dict_whole, m,c1,c2, Coalition_double_divs, votes_to_reduce=votes_to_reduce, by_pp_id = by_pp_id)

                elif first_COAL_simple:
                    # first COAL in elimination order appears as COAL, second as COALNP/LP
                    c1, c2 = len(list_div1_first_COAL), len(list_div2_first_COAL)
                    redistributed_votes = simple_redistribution(div1,div2,Reduce_dict_PP, expand_dict_whole, m,c1,c2, Coalition_double_divs, combine_double_divs = False, votes_to_reduce=votes_to_reduce, by_pp_id = by_pp_id) 
        
        else:

            # try check for simple again after removing any independents

            # rename Coalition parties to COAL
            if Coalition_double_divs:
                list_div1_preserved, list_div2_preserved = list_div1.copy(), list_div2.copy()
                for lst in [list_div1,list_div2]:
                    if 'COALNP' in lst:
                        lst[lst.index('COALNP')] = 'COAL'
                    if 'COALLP' in lst:
                        lst[lst.index('COALLP')] = 'COAL'

                if len(Coalition_double_divs) == 1: # Fix c1,c2 if length == 1; if length == 2, fix later!
                    common = set(list_div1) & set(list_div2) # all COALs are as 'COAL'
                elif len(Coalition_double_divs) == 2:
                    common = set(list_div1) & set(list_div2) 
        
            c1 = find_c1_c2(list_div1, common) # length of relevant subset of list_div1
            c2 = find_c1_c2(list_div2, common) if not c2_dict else len(list_div2)# length of relevant subset of list_div2

            #import pdb;pdb.set_trace()          

            Senate_parties = Senate_parties_by_div.loc[Senate_parties_by_div['div_nm'] == div1,"PartyAbList"].iloc[0] # both div1 and div2 are in the same state so are identical
            list_div1_FP,list_div2_FP, non_senate_parties_div1, non_senate_parties_div2 = remove_non_senate_cands(list_div1[:c1],list_div2[:c2],Senate_parties)

            newset1,newset2 = set(list_div1_FP), set(list_div2_FP)
            newcommon = newset1 & newset2
            m = len(newcommon) if not c2_dict else len(list_div2_FP) # list_div2_FP will always equal list_div2 here because new_parties are excluded!
            # is this correct??? Shouldn't it be identical to previous?
            #import pdb;pdb.set_trace()

            if Coalition_double_divs:
                # restore names to the COAL parties, removing the INDs/non-senates
                list_div1_FP = [p for p in list_div1_preserved[:c1] if p not in non_senate_parties_div1] 
                list_div2_FP = [p for p in list_div2_preserved[:c2] if p not in non_senate_parties_div2]


                if len(Coalition_double_divs)==2: # Should try comparing 1 and 2 treating both parties as separate! --> implications for c1/c2? THIS WILL BE IN PLACE LATER TOO!
                    LPNP_common = set(list_div1_FP) & set(list_div2_FP)
                    m = len(LPNP_common) if not c2_dict else len(list_div2_FP)
                    newcommon = LPNP_common
                    #c1, c2 = find_c1_c2(list_div1_FP, LPNP_common), find_c1_c2(list_div2_FP, LPNP_common) if not c2_dict else len(list_div2) # CHECK THAT REMOVING THIS IS BENIGN!!!

                # now, repeat the process done for the pure simple redistribution case
                first_COAL_simple, combined_COAL_simple, both_separate_COAL_simple, both_combined_COAL_simple, m_dd, list_div1_COAL_combined, list_div2_COAL_combined, list_div1_first_COAL, list_div2_first_COAL = check_Coalition_double_divs_for_simple_case(div1,div2,list_div1_FP, list_div2_FP, m, Coalition_double_divs, c2_dict)


            if (set(list_div1_FP[:m]) == set(list_div2_FP[:m])) | (first_COAL_simple + combined_COAL_simple + both_separate_COAL_simple + both_combined_COAL_simple): # will fail if Coalition_double_divs and 2nd performs well i.e. order is COAL,ALP,COAL,GRN, but succeed if COAL,..., party_m,COAL :)
                
                print("simple non-senate", div1,div2)
                simpleindrd += 1
                #import pdb;pdb.set_trace()

                if (not Coalition_double_divs) or both_separate_COAL_simple: # if latter, m will already be updated above with m = len(LPNP_common)
                    # SIMPLE REDISTRIBUTION WITH INDEPENDENTS/NON-SENATE


                    if non_senate_parties_div1: # Must reduce c1 further to m
                        reduced_votes = independent_redistribution_reduce(div1, Formal_prefs_dict, Expand_dict_PP, Reduce_dict_PP, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP, votes_to_reduce=votes_to_reduce, by_pp_id = by_pp_id)
                    else:
                        reduced_votes = reduce_candidates_to_set_size(div1, Reduce_dict_PP, c1, by_pp_id, votes_to_reduce=votes_to_reduce) # reduce initial to c1 == m
                    

                    if non_senate_parties_div2: # Now, expand from m to c2, or directly to full

                        #staircase_steps, staircase_required, pre_staircase_parties = check_staircase_expansion(non_senate_parties_div2, reduced_votes, list_div2_FP, list_div2)

                        #if not staircase_required:
                        reduced_votes = independent_redistribution_expand(div2, Formal_prefs_dict, expand_dict_whole, reduce_dict_whole, c2,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = reduced_votes)

                        #else:
                        ##    print('I highly doubt that these are common!', list_div2, list_div2_FP)
                        #    import pdb;pdb.set_trace()
                        #    # expand INDs that are before the end of reduced_votes
                        #    curr_c = len(reduced_votes.columns) + len(pre_staircase_parties) # =m?? No, total number of cands to get!
                        #    reduced_votes = independent_redistribution_expand(div2, Formal_prefs_dict, expand_dict_whole, reduce_dict_whole, curr_c,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, list_div2_FP, data_year, votes_to_expand = reduced_votes, cands_to_expand=pre_staircase_parties)
                        #    # these are steps following (m) parties in reduced_votes 
                        #    for step in staircase_steps:
                        #        import pdb;pdb.set_trace()#
                        
                        #        if not step[0]: # ordinary_expand - then expand to len(step[1]) more!
                        #            reduced_votes = expand_candidates_to_set_size(div2, reduced_votes, expand_dict_whole, reduce_dict_whole, curr_c, expanded_c_size = curr_c + len(step[1]))
                        #            curr_c += len(step[1])
                        #        else:  # then expand to step[1] more Non-Senate!
                        #            # reduce to without pre_staircase_parties so that independent_redistribution_expand can proceed
                        #            reduced_votes = independent_redistribution_reduce(div2, Formal_prefs_dict, Expand_dict_PP, Reduce_dict_PP, DOP_By_PP_Pref_Percent_wide_dict, curr_c,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_reduce=reduced_votes)
                        #            curr_c += len(step[1])

                        #            reduced_votes = independent_redistribution_expand(div2, Formal_prefs_dict, expand_dict_whole, reduce_dict_whole, c2,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = reduced_votes, cands_to_expand = step[1])
                        #            pre_staircase_parties.extend(step[1])


                    if not ((div2 in new_seats_list) or c2_dict):
                        redistributed_votes = expand_candidates_to_set_size(div2, reduced_votes, expand_dict_whole, reduce_dict_whole, c2, expanded_c_size = 'full') # expand from c2 to full (irrelevant whether there has been an independent update)
                    else:
                        redistributed_votes = reduced_votes
                
                else:
                    # determine which of the other 3 Coalition_double_div options it is
                    m = m_dd 
                    #if combined_COAL_simple + both_combined_COAL_simple:
                    # combine COALs together unless first_COAL_simple
                    # c1, c2 = len(list_div1_COAL_combined), len(list_div2_COAL_combined) # I think this is already guaranteed as these were calculted with 'COAL' transformed

                    # if combined/both combined, c1 and/or c2 should be updated to one less following combination of COAL parties! NOOOO c1,c2 should stay the same so they are reduced to c1 and c2, before beign combined into c1-1,c2-1 cands
                    #if combined_COAL_simple + both_combined_COAL_simple:
                    #    if div1 in Coalition_double_divs:
                    #        c1 -= 1
                    #    if div2 in Coalition_double_divs:
                    #        c2 -= 1

                    if non_senate_parties_div1: # Must reduce c1 further to m
                        reduced_votes = independent_redistribution_reduce(div1, Formal_prefs_dict, Expand_dict_PP, Reduce_dict_PP, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP, votes_to_expand = None, Coalition_double_divs = Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple, votes_to_reduce=votes_to_reduce, by_pp_id = by_pp_id) # combine unless first_COAL_simple (case where 2nd is dealt with in c1/c2extras)
                    else:
                        reduced_votes = reduce_candidates_to_set_size(div1, Reduce_dict_PP, c1, by_pp_id, Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple, votes_to_reduce=votes_to_reduce) # reduce initial to c1 == m

                    if non_senate_parties_div2: # Now, expand from m to c2, or directly to full

                        reduced_votes = independent_redistribution_expand(div2, Formal_prefs_dict, expand_dict_whole, reduce_dict_whole, c2,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = reduced_votes, Coalition_double_divs = Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple)
                    
                    if not ((div2 in new_seats_list) or c2_dict):
                        redistributed_votes = expand_candidates_to_set_size(div2, reduced_votes, expand_dict_whole, reduce_dict_whole, c2, expanded_c_size = 'full', Coalition_double_divs = Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple) # expand from c2 to full (irrelevant whether there has been an independent update)
                    else:
                        redistributed_votes = reduced_votes
                    #elif first_COAL_simple: # CAN THIS EVER BE USEFUL, AS WE ARE ALREADY RESTRICTED TO C1- AND C2-? No, but in this case other COAL is in c1/c2-extras. Need to take care when expanding/reducing
                    # first COAL in elimination order appears as COAL, second as COALNP/LP
                    #c1, c2 = len(list_div1_first_COAL), len(list_div2_first_COAL)


            else:
                # have already done calcualtions for c1,c2 (only difference is when 2 COALs)
                # if Coalition_double_divs, then list_div1_FP,list_div2_FP will have COALNP/COALLPs
                
                # COMPLEX REDISTRIBUTION (PERHAPS WITH NON-SENATE/IND)

                mlist = [x for x in newcommon]
                list1 = mlist + [x for x in list_div1_FP if x not in newcommon] # order with first m parties, then remaining ci-m parties
                list2 = mlist + [x for x in list_div2_FP if x not in newcommon]


                if len(Coalition_double_divs) == 1: # LONG WAY TO ADD DOUBLE COALS INTO MLIST POSITION, SO THAT THEY WILL BE MERGED TO BE IDENTICAL TO MLIST!
                    if div1 in Coalition_double_divs:
                       
                        hit_else = False  # Track if else clause is triggered
                        mlist_separated_COALS = []
                        for p in mlist:
                            if p in list_div1_FP:
                                mlist_separated_COALS.append(p)
                            elif not hit_else:  # Only trigger once
                                mlist_separated_COALS.extend(["COALLP", "COALNP"])
                                hit_else = True  # Prevent further additions

                        list1 = mlist_separated_COALS + [x for x in list_div1_FP if x not in mlist_separated_COALS] # remainder
                    else:

                        hit_else = False  # Track if else clause is triggered
                        mlist_separated_COALS = []
                        for p in mlist:
                            if p in list_div2_FP:
                                mlist_separated_COALS.append(p)
                            elif not hit_else:  # Only trigger once
                                mlist_separated_COALS.extend(["COALLP", "COALNP"])
                                hit_else = True  # Prevent further additions

                        list2 = mlist_separated_COALS + [x for x in list_div2_FP if x not in mlist_separated_COALS] # remainder


                complex_pair_row = {'old_div': div1, 'new_div': div2, 'c1_list': list1, 'm_list': mlist,'c2_list': list2}

                # before complex redistirbution, we need to have the votes_by_PP of the c1- candidates.
                # If independent involved, start with c1- (i.e. perform independent redistribution reduce)
                # Otherwise, reduce to c1 inside complex_redistribution
                # Complex redistribution transforms from c1 to c2

                #import pdb;pdb.set_trace()

                if non_senate_parties_div1: # Must reduce c1 further
                    do_double_divs = [] if Coalition_double_divs and (div1 not in Coalition_double_divs) else Coalition_double_divs # Only non-empty if div1 is Coalition_double_divs

                    c1_votes = independent_redistribution_reduce(div1, Formal_prefs_dict, Expand_dict_PP, Reduce_dict_PP, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP, Coalition_double_divs=do_double_divs,votes_to_reduce=votes_to_reduce, by_pp_id = by_pp_id)
                    c2_redistributed_votes = complex_redistribution(div1,div2,Reduce_dict_PP, complex_pair_row, c1_votes = c1_votes, by_pp_id = by_pp_id, Coalition_double_divs = Coalition_double_divs)
                else:
                    #do_double_divs = [] if Coalition_double_divs and (div1 not in Coalition_double_divs) else Coalition_double_divs # Only non-empty if div1 is Coalition_double_divs
                    c2_redistributed_votes = complex_redistribution(div1,div2,Reduce_dict_PP, complex_pair_row, c1_votes = None, by_pp_id = by_pp_id, Coalition_double_divs = Coalition_double_divs, votes_to_reduce=votes_to_reduce)

                #import pdb;pdb.set_trace()
                # expand from c2 onwards
                if non_senate_parties_div2:
                    c2_redistributed_votes = independent_redistribution_expand(div2, Formal_prefs_dict, expand_dict_whole, reduce_dict_whole, c2,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = c2_redistributed_votes, Coalition_double_divs = Coalition_double_divs)
                
                if not ((div2 in new_seats_list)or c2_dict):
                    redistributed_votes = expand_candidates_to_set_size(div2, c2_redistributed_votes, expand_dict_whole, reduce_dict_whole, c2, expanded_c_size = 'full', Coalition_double_divs = Coalition_double_divs) # expand from c2 to full (irrelevant whether there has been an independent update)
                else:
                    redistributed_votes = c2_redistributed_votes


                print("complex or independent complex",div1,div2) 
                complexrd += 1   

        #import pdb;pdb.set_trace()

        if not IND_VOTES_ONLY:
            giver_div = div1 if not older_div else older_div
            First_Prefs_By_PP_Complete_Redistributed = transform_to_raw_votes(redistributed_votes, giver_div, name_changes_year_dict, data_year)
        else:
            # get #s of independent votes!
            First_Prefs_By_PP_Complete_Redistributed = redistributed_votes

        # Convert any INDXs back to just INDX
        First_Prefs_By_PP_Complete_Redistributed = First_Prefs_By_PP_Complete_Redistributed.rename(columns = {col: col[:NUM_OF_INDX_LETTERS] for col in First_Prefs_By_PP_Complete_Redistributed.columns if col.startswith('IND')})
        First_Prefs_By_PP_Complete_Redistributed = First_Prefs_By_PP_Complete_Redistributed.rename(columns = {col: col.removesuffix(div2) for col in First_Prefs_By_PP_Complete_Redistributed.columns if col.endswith(div2)})


        if div2.endswith(str(int(data_year)+3)):
            div2 = div1
            #import pdb;pdb.set_trace()

        First_Prefs_By_PP_Complete_Redistributed_dict[(div1,div2)] = First_Prefs_By_PP_Complete_Redistributed.reset_index() # bring back pp_id

        #print(First_Prefs_By_PP_Complete_Redistributed)
        #import pdb;pdb.set_trace()

         

    print(simplerd, simpleindrd, complexrd)
    print(time.time() - start, 'seconds')
    #import pdb;pdb.set_trace()

    #if not c2_dict:
    #    output_folder = f"feather Redistribution pairs {str(int(data_year)+2)}"
    #    os.makedirs(output_folder, exist_ok=True)
    #    for key, sub_df in First_Prefs_By_PP_Complete_Redistributed_dict.items():
    #        filename = f"{output_folder}/{str(int(data_year)+2)}FPBPPRed_{key[0]}_{key[1]}.feather"
    #        sub_df.reset_index(drop=True).to_feather(filename)


    #import pdb;pdb.set_trace()

    

    return First_Prefs_By_PP_Complete_Redistributed_dict







def reduce_to_Omnipresent_parties(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, new_seats_list, Omnipresent_parties, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year):
    # input is df with columns corresponding to giver division and receiver division, respectively. Returns a dictionary with new div names as keys, and a df of the full redistributed votes for the original giver's votes for each pp_id as values 
    Elimination_order_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Reduce_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, DOP_div_expand_dict, DOP_div_reduce_dict, DOP_div_pref_percent_dict = list_of_DOP_dicts
    # get set of parties (the global c2!)

    COAL_party_set = {'LP','NP','LNP','LNQ','CLP','COALLP','COALNP','COAL'}
    Omnipresent_parties = [p if p not in COAL_party_set else 'COAL' for p in Omnipresent_parties]

    Omnipresent_columns = ['div_nm','pp_id'] + Omnipresent_parties
    Omnipresent_parties_df = pd.DataFrame(columns = Omnipresent_columns)

    simplerd, simpleindrd, complexrd = 0,0, 0

    for div1 in sorted(Formal_prefs_dict.keys()): # 

        div2 = 'Hahaha' # hopefully works as placeholder without any errors


        Omnipresent_parties_curr = Omnipresent_parties.copy()



        # temporary - just to do VIC redists:
        #if div_to_state_dict[div1]!= 'VIC':
        #    continue
        #if wait and ((div1 not in ['Gorton']) and (div2 not in ['Hawke'])):
        #    continue
        #else:
        #    wait = 0

        # Ignore new seats - their results can be obtained from the pp_ids of other divs
        #new_seat_party_dicts = create_new_seat_party_dicts(data_year)

        
        # get elimination orders from last election, adjusting if this is new seat. This will always have c2 = m, so no issue that new seats are not a key in all the dicts
        #if div2 in new_seats_list:
            #import pdb;pdb.set_trace()

        #    print("new seat", div1,div2)
        #    Elimination_order_dict[div2] = new_seat_party_dicts[div2] #['LP','ALP','GRN','UAPP','ON'] ## Manually generalise to each new seat, naming COAL or LP or NP as appropriate
        #    list_div2 = Elimination_order_dict[div2]
        #    list_div1 = Elimination_order_dict[div1].copy()
        #else:
        list_div1, list_div2 = Elimination_order_dict[div1].copy(), Omnipresent_parties_curr # get elimination orders



        # separate out COALNP and COALLP where there are 2: ElimOrder, Expand, PrefPercent (x2)

        
        #import pdb;pdb.set_trace()

        # find which COAL party should be in Omnipresent_parties_curr

        Coalition_parties = set(list_div1) & COAL_party_set

        Coalition_double_divs = [div1] if 'COALLP' in Coalition_parties else [] # Only double div if in a Coalition Senate state!!!

        #if not Coalition_double_divs:
            #COAL_party = next(iter(Coalition_parties))
            #list_div2 = [COAL_party if p == 'COAL' else p for p in list_div2] # replace with relevant COAL party

        # whichever party finished higher (if more than 1)!
        COAL_party = next((x for x in Elimination_order_dict[div1] if x in COAL_party_set), None)
        list_div2 = [COAL_party if p == 'COAL' else p for p in list_div2]


        common = set(list_div2)
        m = len(common)



        # m is always 4!


        assert (m==3) & (len(set(list_div2)) == 3)



        # IF ONE OR BOTH DIVS ARE A COALITION_DOUBLE_DIV, CHECK IF IT'S SIMPLE AT HEART: EITHER ONE COAL IS AN EXTRA, OR WHEN COALS ARE COMBINED, IT IS SIMPLE
        first_COAL_simple, combined_COAL_simple, both_separate_COAL_simple, both_combined_COAL_simple, m_dd, list_div1_COAL_combined, list_div2_COAL_combined, list_div1_first_COAL, list_div2_first_COAL= check_Coalition_double_divs_for_simple_case(div1,div2,list_div1, list_div2, m, Coalition_double_divs)
        

        # find common candidates, see if simple/complex
        if (set(list_div1[:m]) == set(list_div2[:m])) | (first_COAL_simple + combined_COAL_simple + both_separate_COAL_simple + both_combined_COAL_simple):
            
            # SIMPLE REDISTRIBUTION
            print("simple", div1,div2)
            simplerd += 1

            if (set(list_div1[:m]) == set(list_div2[:m])) or both_separate_COAL_simple:
                c1 = len(list_div1)
                c2 = len(list_div2)
                redistributed_votes = simple_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict,DOP_div_expand_dict, m,c1,c2)

            else:
                # determine which it is. If 
                m = m_dd 
                if combined_COAL_simple + both_combined_COAL_simple:
                    # combine COALs together
                    c1, c2 = len(list_div1_COAL_combined), len(list_div2_COAL_combined)
                    redistributed_votes = simple_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict,DOP_div_expand_dict, m,c1,c2, Coalition_double_divs)

                elif first_COAL_simple:
                    # first COAL in elimination order appears as COAL, second as COALNP/LP
                    c1, c2 = len(list_div1_first_COAL), len(list_div2_first_COAL)
                    redistributed_votes = simple_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict,DOP_div_expand_dict, m,c1,c2, Coalition_double_divs, combine_double_divs = False) 
            
            assert c2 ==3

        else:

            # rename Coalition parties to COAL
            if Coalition_double_divs:
                list_div1_preserved, list_div2_preserved = list_div1.copy(), list_div2.copy()
                for lst in [list_div1,list_div2]:
                    if 'COALNP' in lst:
                        lst[lst.index('COALNP')] = 'COAL'
                    if 'COALLP' in lst:
                        lst[lst.index('COALLP')] = 'COAL'

                if len(Coalition_double_divs) == 1: # Fix c1,c2 if length == 1; if length == 2, fix later!
                    common = set(list_div1) & set(list_div2) # all COALs are as 'COAL'
        
            c1 = find_c1_c2(list_div1, common) # length of relevant subset of list_div1
            c2 = find_c1_c2(list_div2, common) # length of relevant subset of list_div2

            #import pdb;pdb.set_trace()          


            Senate_parties = Senate_parties_by_div.loc[Senate_parties_by_div['div_nm'] == div1,"PartyAbList"].iloc[0] # both div1 and div2 are in the same state so are identical
            list_div1_FP,list_div2_FP, non_senate_parties_div1, non_senate_parties_div2 = remove_non_senate_cands(list_div1[:c1],list_div2[:c2],Senate_parties)

            newset1,newset2 = set(list_div1_FP), set(list_div2_FP)
            newcommon = newset1 & newset2
            m = len(newcommon) # is this correct??? Shouldn't it be identical to previous?
            #import pdb;pdb.set_trace()

            if Coalition_double_divs:
                # restore names to the COAL parties, removing the INDs/non-senates
                list_div1_FP = [p for p in list_div1_preserved[:c1] if p not in non_senate_parties_div1] 
                list_div2_FP = [p for p in list_div2_preserved[:c2] if p not in non_senate_parties_div2]


                if len(Coalition_double_divs)==2: # Should try comparing 1 and 2 treating both parties as separate! --> implications for c1/c2? THIS WILL BE IN PLACE LATER TOO!
                    LPNP_common = set(list_div1_FP) & set(list_div2_FP)
                    m = len(LPNP_common)
                    newcommon = LPNP_common
                    c1, c2 = find_c1_c2(list_div1_FP, LPNP_common), find_c1_c2(list_div2_FP, LPNP_common)

                # now, repeat the process done for the pure simple redistribution case
                first_COAL_simple, combined_COAL_simple, both_separate_COAL_simple, both_combined_COAL_simple, m_dd, list_div1_COAL_combined, list_div2_COAL_combined, list_div1_first_COAL, list_div2_first_COAL = check_Coalition_double_divs_for_simple_case(div1,div2,list_div1_FP, list_div2_FP, m, Coalition_double_divs)


            if (set(list_div1_FP[:m]) == set(list_div2_FP[:m])) | (first_COAL_simple + combined_COAL_simple + both_separate_COAL_simple + both_combined_COAL_simple): # will fail if Coalition_double_divs and 2nd performs well i.e. order is COAL,ALP,COAL,GRN, but succeed if COAL,..., party_m,COAL :)
                
                print("simple non-senate", div1,div2)
                simpleindrd += 1
                #import pdb;pdb.set_trace()

                if (not Coalition_double_divs) or both_separate_COAL_simple: # if latter, m will already be updated above with m = len(LPNP_common)
                    # SIMPLE REDISTRIBUTION WITH INDEPENDENTS/NON-SENATE

                    #import pdb;pdb.set_trace()


                    if non_senate_parties_div1: # Must reduce c1 further to m
                        reduced_votes = independent_redistribution_reduce(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP)
                    else:
                        reduced_votes = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1,True) # reduce initial to c1 == m
                    #import pdb;pdb.set_trace()
                    if (non_senate_parties_div2) and False: # never expand!  # Now, expand from m to c2, or directly to full
                        reduced_votes = independent_redistribution_expand(div2, Formal_prefs_dict, DOP_div_expand_dict, DOP_div_pref_percent_dict, c2,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = reduced_votes)

                    if (not (div2 in new_seats_list)) and False: 
                        redistributed_votes = expand_candidates_to_set_size(div2, reduced_votes, DOP_div_expand_dict, DOP_div_pref_percent_dict, c2, expanded_c_size = 'full') # expand from c2 to full (irrelevant whether there has been an independent update)
                    else:
                        redistributed_votes = reduced_votes
                
                else:
                    # determine which of the other 3 Coalition_double_div options it is
                    m = m_dd 
                    #if combined_COAL_simple + both_combined_COAL_simple:
                    # combine COALs together unless first_COAL_simple
                    # c1, c2 = len(list_div1_COAL_combined), len(list_div2_COAL_combined) # I think this is already guaranteed as these were calculted with 'COAL' transformed

                    # if combined/both combined, c1 and/or c2 should be updated to one less following combination of COAL parties! NOOOO c1,c2 should stay the same so they are reduced to c1 and c2, before beign combined into c1-1,c2-1 cands
                    #if combined_COAL_simple + both_combined_COAL_simple:
                    #    if div1 in Coalition_double_divs:
                    #        c1 -= 1
                    #    if div2 in Coalition_double_divs:
                    #        c2 -= 1

                    if non_senate_parties_div1: # Must reduce c1 further to m
                        reduced_votes = independent_redistribution_reduce(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP, votes_to_expand = None, Coalition_double_divs = Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple) # combine unless first_COAL_simple (case where 2nd is dealt with in c1/c2extras)
                    else:
                        reduced_votes = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1, True, Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple) # reduce initial to c1 == m
                    #import pdb;pdb.set_trace()

                    if non_senate_parties_div2: # Now, expand from m to c2, or directly to full
                        reduced_votes = independent_redistribution_expand(div2, Formal_prefs_dict, DOP_div_expand_dict, DOP_div_pref_percent_dict, c2,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = reduced_votes, Coalition_double_divs = Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple)
                    
                    if (not (div2 in new_seats_list) and False): # never expand! 
                        redistributed_votes = expand_candidates_to_set_size(div2, reduced_votes, DOP_div_expand_dict, DOP_div_pref_percent_dict, c2, expanded_c_size = 'full', Coalition_double_divs = Coalition_double_divs, combine_double_divs = 1 - first_COAL_simple) # expand from c2 to full (irrelevant whether there has been an independent update)
                    else:
                        redistributed_votes = reduced_votes
                    #elif first_COAL_simple: # CAN THIS EVER BE USEFUL, AS WE ARE ALREADY RESTRICTED TO C1- AND C2-? No, but in this case other COAL is in c1/c2-extras. Need to take care when expanding/reducing
                    # first COAL in elimination order appears as COAL, second as COALNP/LP
                    #c1, c2 = len(list_div1_first_COAL), len(list_div2_first_COAL)


            else:
                # have already done calcualtions for c1,c2 (only difference is when 2 COALs)
                # if Coalition_double_divs, then list_div1_FP,list_div2_FP will have COALNP/COALLPs
                
                # COMPLEX REDISTRIBUTION (PERHAPS WITH NON-SENATE/IND)

                mlist = [x for x in newcommon]
                list1 = mlist + [x for x in list_div1_FP if x not in newcommon] # order with first m parties, then remaining ci-m parties
                list2 = mlist + [x for x in list_div2_FP if x not in newcommon]

                if len(Coalition_double_divs) == 1: # LONG WAY TO ADD DOUBLE COALS INTO MLIST POSITION, SO THAT THEY WILL BE MERGED TO BE IDENTICAL TO MLIST!
                    if div1 in Coalition_double_divs:
                       
                        hit_else = False  # Track if else clause is triggered
                        mlist_separated_COALS = []
                        for p in mlist:
                            if p in list_div1_FP:
                                mlist_separated_COALS.append(p)
                            elif not hit_else:  # Only trigger once
                                mlist_separated_COALS.extend(["COALLP", "COALNP"])
                                hit_else = True  # Prevent further additions

                        list1 = mlist_separated_COALS + [x for x in list_div1_FP if x not in mlist_separated_COALS] # remainder
                    else:

                        hit_else = False  # Track if else clause is triggered
                        mlist_separated_COALS = []
                        for p in mlist:
                            if p in list_div2_FP:
                                mlist_separated_COALS.append(p)
                            elif not hit_else:  # Only trigger once
                                mlist_separated_COALS.extend(["COALLP", "COALNP"])
                                hit_else = True  # Prevent further additions

                        list2 = mlist_separated_COALS + [x for x in list_div2_FP if x not in mlist_separated_COALS] # remainder


                complex_pair_row = {'old_div': div1, 'new_div': div2, 'c1_list': list1, 'm_list': mlist,'c2_list': list2}
                #import pdb;pdb.set_trace()

                # before complex redistirbution, we need to have the votes_by_PP of the c1- candidates.
                # If independent involved, start with c1- (i.e. perform independent redistribution reduce)
                # Otherwise, reduce to c1 inside complex_redistribution
                # Complex redistribution transforms from c1 to c2

                if non_senate_parties_div1: # Must reduce c1 further
                    do_double_divs = [] if Coalition_double_divs and (div1 not in Coalition_double_divs) else Coalition_double_divs # Only non-empty if div1 is Coalition_double_divs

                    c1_votes = independent_redistribution_reduce(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div1_FP, votes_to_expand = None, Coalition_double_divs = do_double_divs)
                    c2_redistributed_votes = complex_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict,complex_pair_row, c1_votes = c1_votes, Coalition_double_divs = Coalition_double_divs)
                else:
                    #do_double_divs = [] if Coalition_double_divs and (div1 not in Coalition_double_divs) else Coalition_double_divs # Only non-empty if div1 is Coalition_double_divs
                    c2_redistributed_votes = complex_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict,complex_pair_row, c1_votes = None, Coalition_double_divs = Coalition_double_divs)

                #import pdb;pdb.set_trace()
                # expand from c2 onwards
                if non_senate_parties_div2:
                    c2_redistributed_votes = independent_redistribution_expand(div2, Formal_prefs_dict, DOP_div_expand_dict, DOP_div_pref_percent_dict, c2,Incumbency_by_div, div_to_state_dict, party_category_dict, FINAL_CANDIDATE_NO, data_year, list_div2_FP, votes_to_expand = c2_redistributed_votes, Coalition_double_divs = Coalition_double_divs)
                
                if (not (div2 in new_seats_list)) and 0: # never expand!
                    redistributed_votes = expand_candidates_to_set_size(div2, c2_redistributed_votes, DOP_div_expand_dict, DOP_div_pref_percent_dict, c2, expanded_c_size = 'full', Coalition_double_divs = Coalition_double_divs) # expand from c2 to full (irrelevant whether there has been an independent update)
                else:
                    redistributed_votes = c2_redistributed_votes


                print("complex or independent complex",div1,div2) 
                complexrd += 1   

        First_Prefs_By_PP_Complete_Redistributed = transform_to_raw_votes(redistributed_votes, div1, name_changes_year_dict, data_year)

        # Convert any INDXs back to just INDX
        #First_Prefs_By_PP_Complete_Redistributed = First_Prefs_By_PP_Complete_Redistributed.rename(columns = {col: col[:NUM_OF_INDX_LETTERS] for col in First_Prefs_By_PP_Complete_Redistributed.columns if col.startswith('IND')})

        # format into df for bootstrapping
        # rename COAL parties into 'COAL'
        
        First_Prefs_By_PP_Complete_Redistributed.rename(columns={col: 'COAL' for col in First_Prefs_By_PP_Complete_Redistributed.columns if col in COAL_party_set}, inplace=True)
        First_Prefs_By_PP_Complete_Redistributed.loc[:,'div_nm'] = div1

        First_Prefs_By_PP_Complete_Redistributed = First_Prefs_By_PP_Complete_Redistributed.reset_index()
        Omnipresent_parties_df = pd.concat([Omnipresent_parties_df,First_Prefs_By_PP_Complete_Redistributed], ignore_index=True)





        #print(div1, First_Prefs_By_PP_Complete_Redistributed)
        #import pdb;pdb.set_trace()

    print(div1, First_Prefs_By_PP_Complete_Redistributed) # last one for general check

    import pdb;pdb.set_trace()



    # add pp_nm to pp_id cols via PP_data
    PP_id_nm = pd.read_csv(f'{data_year}_PP_data.csv', index_col=None)[['div_nm','pp_id','pp_nm']]
    PP_id_nm['div_nm'] = PP_id_nm['div_nm'].replace(name_changes_year_dict)

    Omnipresent_parties_df = Omnipresent_parties_df.merge(PP_id_nm, on = ['div_nm','pp_id'],how = 'left')

    Omnipresent_columns.insert(2,'pp_nm')
    Omnipresent_parties_df = Omnipresent_parties_df[Omnipresent_columns + ['INFORMAL']] # nice order!

    Omnipresent_parties_df.to_csv(f'{data_year}OmnipresentPartiesByPP.csv', index=False)

    print(simplerd, simpleindrd, complexrd)
    print(time.time() - start, 'seconds')

    return Omnipresent_parties_df





def check_house_senate_discrepancies(data_year, name_changes_year_dict):

    #directory = f"C:/Dania/2024/Australian Election/SenateVotesByPP{data_year}"
    directory = Path(f"C:/Dania/2024/Australian Election/SenateVotesByPP{data_year}") if os.name == "nt" else Path.home() / f"Australian Election/SenateVotesByPP{data_year}"
    
    csv_files = sorted(glob.glob(str(f"{directory}/*.csv")))
    senate_votes_full = pd.concat((pd.read_csv(f, skiprows=1)[['DivisionNm','PollingPlaceNm','OrdinaryVotes']].groupby(['DivisionNm','PollingPlaceNm'], as_index=False) \
                                                            .agg({'OrdinaryVotes': 'sum'}) for f in csv_files), ignore_index=True) \
                                                            .rename(columns={'DivisionNm':'div_nm','PollingPlaceNm':'pp_nm','OrdinaryVotes':'senate_votes'})
    #senate_votes_full_aston = senate_votes_full.loc[senate_votes_full['div_nm']=='Aston','senate_votes'].sum()

    # change directory
    base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
    os.chdir(base_dir)

    #add_Other_category

    division_senate_Others = pd.read_csv(f"{data_year}SenateVotesCountedByDivision.csv", skiprows=1).iloc[:,[1,5,6,7,8]].rename(columns={'DivisionNm':'div_nm'})
    division_senate_Others.loc[:,'senate_votes'] = division_senate_Others.iloc[:, 1:].sum(axis=1)
    division_senate_Others_sum = division_senate_Others.iloc[:,[0,-1]]
    division_senate_Others_sum = division_senate_Others_sum.copy()
    division_senate_Others_sum.loc[:,'pp_nm'] = 'Other'
    division_senate_Others_sum = division_senate_Others_sum[['div_nm','pp_nm','senate_votes']]

    senate_votes_full = pd.concat([senate_votes_full,division_senate_Others_sum],axis=0)

    ############ CHANGED THIS - MAKE SURE STILL OK!!!
    #import pdb;pdb.set_trace()
    First_Prefs_By_PP = pd.read_csv(f"{data_year}FirstPrefsByPPComplete.csv",index_col = None)[['pp_nm','div_nm','PartyAb','votes']] 
    house_votes_full = First_Prefs_By_PP.groupby(['div_nm','pp_nm'], as_index=False).agg({"votes":"sum"}).rename(columns={'votes':'house_votes'})


    # The rest is already done when constructing FirstPrefsByPPComplete, so not necessary!!!!!!!!!


    #division_house_Others = pd.read_csv(f"{data_year}HouseVotesCountedByDivision.csv", skiprows=1).iloc[:,[1,5,6,7,8]].rename(columns={'DivisionNm':'div_nm'})

    #division_house_Others.loc[:,'house_votes'] = division_house_Others.iloc[:, 1:].sum(axis=1)
    #division_house_Others_sum = division_house_Others.iloc[:,[0,-1]]
    #division_house_Others_sum = division_house_Others_sum.copy()
    #division_house_Others_sum.loc[:,'pp_nm'] = 'Other'
    #division_house_Others_sum = division_house_Others_sum[['div_nm','pp_nm','house_votes']]


    #house_votes_full = pd.concat([house_votes,division_house_Others_sum],axis=0)

    Other_booth_type_prefixes = ['Remote Mobile', 'Other Mobile','Special Hospital', 'EAV']

    #import pdb;pdb.set_trace()


    # combine Others together
    #house_votes_full.loc[:,"pp_nm"] = house_votes_full.loc[:,"pp_nm"].apply(lambda x: 'Other' if any(x.startswith(prefix) for prefix in Other_booth_type_prefixes) else x)
    senate_votes_full.loc[:,"pp_nm"] = senate_votes_full.loc[:,"pp_nm"].apply(lambda x: 'Other' if any(x.startswith(prefix) for prefix in Other_booth_type_prefixes) else x)

    #house_votes_full = house_votes_full.groupby(["div_nm", "pp_nm"], as_index=False).agg({'house_votes':'sum'})
    senate_votes_full = senate_votes_full.groupby(["div_nm", "pp_nm"], as_index=False).agg({'senate_votes':'sum'})





    formal_senate_full_house_comparison = pd.DataFrame(house_votes_full).merge(pd.DataFrame(senate_votes_full), on = ['div_nm','pp_nm'], how='left')
    formal_senate_full_house_comparison['div_nm'] = formal_senate_full_house_comparison['div_nm'].replace(name_changes_year_dict)
    formal_senate_full_house_comparison.loc[:,'house-sen'] = (formal_senate_full_house_comparison.loc[:,'house_votes'] - formal_senate_full_house_comparison.loc[:,'senate_votes']).values          


    formal_senate_full_house_comparison.loc[:,'house/sen'] = (formal_senate_full_house_comparison.loc[:,'house_votes'] / formal_senate_full_house_comparison.loc[:,'senate_votes']).values

    #print(formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison["house/sen"] < 1) & (formal_senate_full_house_comparison['div_nm']<500),])

    #print(formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison["house/sen"] > 1.2) & (formal_senate_full_house_comparison['div_nm']<500),])

    #import pdb;pdb.set_trace()


    # needs attention if less than 500 votes and difference is stark!
    formal_senate_full_house_comparison.loc[~(formal_senate_full_house_comparison['pp_nm'].str.endswith('PPVC')) & ~(formal_senate_full_house_comparison['pp_nm'].str.endswith('Other')) & (np.abs(formal_senate_full_house_comparison['house-sen'])>100),]       
    formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison['pp_nm'].str.endswith('PPVC')) & ~(formal_senate_full_house_comparison['pp_nm'].str.endswith('Other')) & (np.abs(formal_senate_full_house_comparison['house-sen'])>100),]
    formal_senate_full_house_comparison.loc[~(formal_senate_full_house_comparison['pp_nm'].str.endswith('PPVC')) & (formal_senate_full_house_comparison['senate_votes']<500) & (formal_senate_full_house_comparison['house/sen']>1.05),]
    formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison['pp_nm'].str.startswith('Brisbane North')),]
    formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison['div_nm']=='Lilley'),]


    # For 2019:
    # 1. Sydney(Barton) Sydney BARTON PPVC - solved
    # 2. North Sydney  Artarmon/Central - solved
    # 3. Brisbane City (Lilley) - unsolved - take from Brisbane City (Brisbane) due to proximity!

    # 2016:
    # 1. West Ryde BEROWRA PPVC (from Castle Hill BEROWRA PPVC)
    # 2. Waverley KINGSFORD SMITH PPVC - Wentworth!!!
    # 3. Christies Beach MAYO PPVC - comes from Kingston! Christies Beach KINGSTON PPVC
    # 4. Fairfield WERRIWA PPVC?? - Fairfield BLAXLAND PPVC



    # FOR 2022:

    #QLD/TAS/SA/NT/ACT: Brisbane North (Lilley), Moncrieff Labrador MONCRIEFF PPVC (from runaway bay)


    # to fix: Duggan? Don't worry about it (maxnamara), HOLT, Manjimup East (O'Connor), Sydney
    # PPVCs: McEwen Epping, Footscray MELBOURNE, Haymarket MITCHELL/NORTH SYDNEY/PARRAMATTA/(also BEROWRA PPVC),  Mill Park COOPER/JAGAJAGA/MCEWEN,Newtown SYDNEY,Northcote MELBOURNE,South Yarra MELBOURNE, Sydney GRAYNDLER, The Ponds MITCHELL,  Waverley KINGSFORD SMITH

    # Manjimup East (O'Connor) --> Manjimup PPVC combine/use

    # epping - add Scullin's to Mcewen
    # MELBOURNE ones - all take from Melbourne MELBOURNE!
    # Haymarkets (including BEROWRA!) - take from GRAYNDLER! not neat
    # Mill park - all 3 take from scullin!
    # Newtown - take from GRAYNDLER!
    # Sydney GRAYNDLER - take from Sydney(Sydney) - solves many problems - This is a fun one! Power of deduction!!!
    # The Ponds MITCHELL - take form Greenway, but mystery unsolved!
    # Waverley KINGSFORD SMITH - combine with Randwick KINGSFORD SMITH PPVC - mystery half solved!
    # New England: take Blackville's from Ben Venue!

    # HOLT: Take all that are 0 from Cranbourne East!!!

    # inspect where house or senates are flat out 0: 
    zero_house_df = formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison['house_votes'] == 0 ) ,] 
    zero_senate_df = formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison['senate_votes'] == 0 ) ,]   

    # so far uncovered smaller issues - difference less than 100, but may be high proportion of votes!
    formal_senate_full_house_comparison.loc[(np.abs(formal_senate_full_house_comparison['house-sen'])<100) & (formal_senate_full_house_comparison['house/sen']>1.5),] 
    # (only concerning one in adelaide)
    return formal_senate_full_house_comparison

def amend_Formal_prefs_dict(Formal_prefs_dict, data_year, name_changes_year_dict, all_states = False):
    ### all_states is an argument that dictates if Formal_prefs_dict is defined for all states, or only redistribution ones

    #import pdb;pdb.set_trace()

    h_s_discrepancies = check_house_senate_discrepancies(data_year, name_changes_year_dict)

    if data_year == '2022':

        # 1. New England: take Blackville's from Ben Venue!
        FP_div = Formal_prefs_dict['New England']
        lender = 'Ben Venue'
        borrower = 'Blackville'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        Formal_prefs_dict['New England'] = pd.concat([FP_div,lender_FPs], ignore_index=True)


        # 2. O'Connor: Manjimup East from Manjimup PPVC
        FP_div = Formal_prefs_dict["O'Connor"]
        lender = 'Manjimup PPVC'
        borrower = 'Manjimup East'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        Formal_prefs_dict["O'Connor"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

        #3. Holt!
    
        FP_div = Formal_prefs_dict['Holt']
        lender = 'Cranbourne East'
        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]

        borrower_list = h_s_discrepancies.loc[(h_s_discrepancies['div_nm'] == 'Holt' ) & (h_s_discrepancies['senate_votes']==0),'pp_nm'].tolist()
        for borrower_pp in borrower_list:
            to_add_FP = lender_FPs.copy()
            to_add_FP.loc[:,'pp_nm'] = borrower_pp
            Formal_prefs_dict['Holt'] = pd.concat([Formal_prefs_dict['Holt'],to_add_FP], ignore_index=True)

        # 4. Sydney (Sydney) --> remove from Syndey and add to to Sydney GRAYNDLER
        FP_div = Formal_prefs_dict['Sydney']
        lender = 'Sydney (Sydney)'
        borrower = 'Sydney GRAYNDLER PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Grayndler'
        Formal_prefs_dict['Sydney'] = Formal_prefs_dict['Sydney'].loc[Formal_prefs_dict['Sydney']['pp_nm'] != 'Sydney (Sydney)',]
        Formal_prefs_dict["Grayndler"] = pd.concat([Formal_prefs_dict["Grayndler"],lender_FPs], ignore_index=True)

        # 5. Epping ----- PPVC (add Scullin's to McEwen)
        FP_div = Formal_prefs_dict['Scullin']
        lender = 'Epping SCULLIN PPVC'
        borrower = 'Epping MCEWEN PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'McEwen'
        Formal_prefs_dict["McEwen"] = pd.concat([Formal_prefs_dict["McEwen"],lender_FPs], ignore_index=True)

        # 6. MELBOURNE ones - all take from Melbourne MELBOURNE!
        FP_div = Formal_prefs_dict['Melbourne']
        lender = 'Melbourne MELBOURNE PPVC'
        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]

        borrower_list = h_s_discrepancies.loc[(h_s_discrepancies['div_nm'] == 'Melbourne' ) & (h_s_discrepancies['senate_votes']==0),'pp_nm'].tolist()
        for borrower_pp in borrower_list:
            to_add_FP = lender_FPs.copy()
            to_add_FP.loc[:,'pp_nm'] = borrower_pp
            Formal_prefs_dict['Melbourne'] = pd.concat([Formal_prefs_dict['Melbourne'],to_add_FP], ignore_index=True)

        # 7. Haymarkets (including BEROWRA!) - take from GRAYNDLER! not neat
        FP_div = Formal_prefs_dict['Grayndler']
        lender = 'Haymarket GRAYNDLER PPVC'
        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]

        borrower_list = h_s_discrepancies.loc[(h_s_discrepancies['pp_nm'].str.startswith('Haymarket')) & (h_s_discrepancies['senate_votes']==0),'pp_nm'].tolist()
        div_list = h_s_discrepancies.loc[(h_s_discrepancies['pp_nm'].str.startswith('Haymarket')) & (h_s_discrepancies['senate_votes']==0),'div_nm'].tolist()
        for i, borrower_pp in enumerate(borrower_list):
            to_add_FP = lender_FPs.copy()
            to_add_FP.loc[:,'pp_nm'] = borrower_pp
            to_add_FP.loc[:,'div_nm'] = div_list[i]
            borrower_div = div_list[i] #borrower_pp.split(' ')[-2].capitalize() 
            Formal_prefs_dict[borrower_div] = pd.concat([Formal_prefs_dict[borrower_div],to_add_FP], ignore_index=True)

        # 8. # Mill park - all 3 take from scullin!
        FP_div = Formal_prefs_dict['Scullin']
        lender = 'Mill Park SCULLIN PPVC'
        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        MCEWEN_LIMIT = 30

        borrower_list = h_s_discrepancies.loc[(h_s_discrepancies['pp_nm'].str.startswith('Mill Park')) & (h_s_discrepancies['senate_votes']<MCEWEN_LIMIT),'pp_nm'].tolist()
        div_list = h_s_discrepancies.loc[(h_s_discrepancies['pp_nm'].str.startswith('Mill Park')) & (h_s_discrepancies['senate_votes']<MCEWEN_LIMIT),'div_nm'].tolist()
        for i, borrower_pp in enumerate(borrower_list):
            to_add_FP = lender_FPs.copy()
            to_add_FP.loc[:,'pp_nm'] = borrower_pp
            borrower_div = div_list[i] if div_list[i] != 'Mcewen' else 'McEwen'
            to_add_FP.loc[:,'div_nm'] = borrower_div

            Formal_prefs_dict[borrower_div] = pd.concat([Formal_prefs_dict[borrower_div],to_add_FP], ignore_index=True)

        # 9. Newtown - take from GRAYNDLER!
        FP_div = Formal_prefs_dict['Grayndler']
        lender = 'Newtown GRAYNDLER PPVC'
        borrower = 'Newtown SYDNEY PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Sydney'
        Formal_prefs_dict["Sydney"] = pd.concat([Formal_prefs_dict["Sydney"],lender_FPs], ignore_index=True)

        # 10. The Ponds MITCHELL - take form Greenway, but mystery unsolved!
        FP_div = Formal_prefs_dict['Greenway']
        lender = 'The Ponds GREENWAY PPVC'
        borrower = 'The Ponds MITCHELL PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Mitchell'
        Formal_prefs_dict["Mitchell"] = pd.concat([Formal_prefs_dict["Mitchell"],lender_FPs], ignore_index=True)

        # 11. Waverley KINGSFORD SMITH - combine with Randwick KINGSFORD SMITH PPVC - mystery half solved!
        FP_div = Formal_prefs_dict["Kingsford Smith"]
        lender = 'Randwick KINGSFORD SMITH PPVC'
        borrower = 'Waverley KINGSFORD SMITH PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        Formal_prefs_dict["Kingsford Smith"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

        if all_states:


            # 12. Lilley - best guess
            FP_div = Formal_prefs_dict["Lilley"]
            lender = 'Brisbane Central LILLEY PPVC'
            borrower = 'Brisbane North (Lilley)'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            Formal_prefs_dict["Lilley"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

            # 13. Moncrieff PPVCs - solved!
            FP_div = Formal_prefs_dict["Moncrieff"]
            lender = 'Runaway Bay MONCRIEFF PPVC'
            borrower = 'Labrador MONCRIEFF PPVC'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            Formal_prefs_dict["Moncrieff"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

    # 2019 - Glorious - nothing to adjust, only for all_states!!!!

    elif data_year == '2019':
        

        if all_states:

            # 1. Sydney(Barton) Sydney BARTON PPVC - solved

            FP_div = Formal_prefs_dict["Barton"]
            lender = 'Sydney BARTON PPVC'
            borrower = 'Sydney (Barton)'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            Formal_prefs_dict["Barton"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

            # 2. North Sydney  Artarmon/Central - solved
            FP_div = Formal_prefs_dict["North Sydney"]
            lender = 'Artarmon Central'
            borrower = 'Artarmon'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            Formal_prefs_dict["North Sydney"] = pd.concat([FP_div,lender_FPs], ignore_index=True)


            # 3. Brisbane City (Lilley) - unsolved - take from Brisbane City (Brisbane) due to proximity!
            FP_div = Formal_prefs_dict['Brisbane']
            lender = 'Brisbane City (Brisbane)'
            borrower = 'Brisbane City (Lilley)'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            lender_FPs.loc[:,'div_nm'] = 'Lilley'
            Formal_prefs_dict["Lilley"] = pd.concat([Formal_prefs_dict["Lilley"],lender_FPs], ignore_index=True)

            # 4. Brisbane PETRIE PPVC - nothing obvious, but use Brisbane City PETRIE PPVC
            FP_div = Formal_prefs_dict['Petrie']
            lender = 'Brisbane City PETRIE PPVC'
            borrower = 'Brisbane PETRIE PPVC'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            Formal_prefs_dict["Petrie"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

            # 5. Auburn WATSON PPVC - from Bankstown WATSON PPVC:
            FP_div = Formal_prefs_dict['Watson']
            lender = 'Bankstown WATSON PPVC'
            borrower = 'Auburn WATSON PPVC'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            Formal_prefs_dict["Watson"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

    elif data_year == '2016':
 
        # 1. Christies Beach MAYO PPVC - comes from Kingston! Christies Beach KINGSTON PPVC
        FP_div = Formal_prefs_dict['Kingston']
        lender = 'Christies Beach KINGSTON PPVC'
        borrower = 'Christies Beach MAYO PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Mayo'
        Formal_prefs_dict["Mayo"] = pd.concat([Formal_prefs_dict["Mayo"],lender_FPs], ignore_index=True)

        # 2. Norfold Island Canberra
        FP_div = Formal_prefs_dict["Canberra"]
        lender = 'Norfolk Island'
        borrower = 'Norfolk Island PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        Formal_prefs_dict["Canberra"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

        # 3. Hebert BLV PPVS - not solved but only 5 votes!
        FP_div = Formal_prefs_dict["Herbert"]
        lender = 'Townsville South'
        borrower = 'BLV Herbert PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        Formal_prefs_dict["Herbert"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

        if all_states:

            # 4. West Ryde BEROWRA PPVC (from Castle Hill BEROWRA PPVC)
            FP_div = Formal_prefs_dict["Berowra"]
            lender = 'Castle Hill BEROWRA PPVC'
            borrower = 'West Ryde BEROWRA PPVC'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            Formal_prefs_dict["Berowra"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

            # 5. Waverley KINGSFORD SMITH PPVC - Wentworth!!!
            FP_div = Formal_prefs_dict['Wentworth']
            lender = 'Waverley WENTWORTH PPVC '
            borrower = 'Waverley KINGSFORD SMITH PPVC'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            lender_FPs.loc[:,'div_nm'] = 'Kingsford Smith'
            Formal_prefs_dict["Kingsford Smith"] = pd.concat([Formal_prefs_dict["Kingsford Smith"],lender_FPs], ignore_index=True)


            # 6. Fairfield WERRIWA PPVC?? - Fairfield BLAXLAND PPVC
            FP_div = Formal_prefs_dict['Blaxland']
            lender = 'Fairfield BLAXLAND PPVC'
            borrower = 'Fairfield WERRIWA PPVC'

            lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
            lender_FPs.loc[:,'pp_nm'] = borrower
            lender_FPs.loc[:,'div_nm'] = 'Werriwa'
            Formal_prefs_dict["Werriwa"] = pd.concat([Formal_prefs_dict["Werriwa"],lender_FPs], ignore_index=True)

    elif data_year == '2013':

        FP_div = Formal_prefs_dict['Canberra']
        lender = 'Tuggeranong CANBERRA PPVC'
        borrower = 'Tuggeranong FRASER PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Fenner'
        Formal_prefs_dict["Fener"] = pd.concat([Formal_prefs_dict["Fenner"],lender_FPs], ignore_index=True)

    elif data_year == '2010':
        FP_div = Formal_prefs_dict["Franklin"]
        lender = 'Hobart FRANKLIN PPVC'
        borrower = 'Divisional Office (PREPOLL)'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        Formal_prefs_dict["Franklin"] = pd.concat([FP_div,lender_FPs], ignore_index=True)

    # 2013:
    # Only Tuggeranong FRASER PPVC (now Fenner); replace with Tuggeranong CANBERRA PPVC

    # 2010:
    # Franklin  Divisional Office (PREPOLL)            3           0.0        3.0        inf - arbitrarily replace with Hobart FRANKLIN PPVC
    
    #import pdb;pdb.set_trace()

    return Formal_prefs_dict


Ideo_Categories = ['Left','ALP','Centre','COAL','Right']


def get_matching_average(row, df, Ideo_Categories, year_prev, div_match, num_match): # written by ChatGPT
    # Filtering conditions:
    if year_prev:
        year_condition = df['Year'] == str(int(row['Year']) - 3)  # Year is 3 smaller than current row's Year
    else:
        year_condition = (df['Year'] != str(int(row['Year']) - 3)) & (df['Year'] != row['Year'])

    div_condition = df['div_nm'] == row['div_nm'] if div_match else ((df['div_nm'] != row['div_nm']) & (df['State'] != row['State']))
    num_condition = (np.abs(df['Num_parties'] - row['Num_parties']) <= 1) if num_match else (np.abs(df['Num_parties'] - row['Num_parties']) > 1)

    matching_rows = df[year_condition & div_condition & num_condition & (df['Ideo_Category'] == row['Ideo_Category'])][Ideo_Categories]
    
    # If matching rows exist, return their average (excluding current row)
    if len(matching_rows) > 0:
        #import pdb;pdb.set_trace()

        return matching_rows.mean()  # You can choose to apply this to other columns too
    else:
        return pd.Series(np.nan, index=Ideo_Categories)

def explode_column(weight_df, col, Ideo_Categories):
        exploded_df = pd.DataFrame()
        cols = Ideo_Categories + ['']
        for i in range(6):  # 6 new columns
            exploded_df[f'{cols[i]}{col}'] = weight_df[col]
        return exploded_df

def estimate_donation_proportion(df, new_row, Ideo_Categories):
    ### inputs Ideology donation df and df new_row containing {year, div_nm, State, Ideo_category, Num_parties}


    # add all 8 potential estimates for matching previous year, same div/Num
    for num_match in [1,0]:
        for div_match in [1,0]:
            for year_prev in [1,0]:
                X_num = f'_X{1 + 4*(1-num_match) + 2*(1 - div_match) + (1 - year_prev)}'  # label them as X_1...X_8 according to groupings

                curr_match = new_row.apply(lambda row: get_matching_average(row, df, Ideo_Categories, year_prev, div_match, num_match), axis=1)
                curr_match[X_num] = curr_match.count(axis=1)

                new_row = new_row.join(curr_match, rsuffix = f'{X_num}')

    X_cols = ['_X1','_X2','_X3','_X4','_X5','_X6','_X7','_X8']
    X_weights = [1,0.5,0.5,0.25,0.125,0.0625,0.0625,0.03125] # ad-hoc weights, improve with model in future
    weight_df_ids = (new_row[X_cols]>0)*X_weights
    weight_df = weight_df_ids.div(weight_df_ids.sum(axis=1), axis=0)


    # expand weight df to each set of 6 columns
    exploded_weight_dfs = []
    for col in weight_df.columns:
        exploded_weight_dfs.append(explode_column(weight_df, col, Ideo_Categories))
    exploded_weight_df = pd.concat(exploded_weight_dfs, axis=1)

    # apply weights
    start_weights_index = new_row.columns.get_loc('Num_parties') + 1
    weighted_proportions = new_row.iloc[:,start_weights_index:] * exploded_weight_df
    weighted_proportions.columns = [col[:-3] for col in weighted_proportions.columns]
    estimated_row_proportions = weighted_proportions.T.groupby(weighted_proportions.columns).sum().T[Ideo_Categories] # df of 6 cols, last is ''


    return estimated_row_proportions


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


def incumbency_advantage_change(div, div_to_state_dict, data_year):

    # try adding state/TPP as predictor!

    incumbency_change_adjusted_votes = 1

    Demographic_Classification_State_df = pd.read_csv(f'{data_year}DemographicClassification.csv', index_col=None)
    Demographic = Demographic_Classification_State_df.loc[Demographic_Classification_State_df['div_nm'],'Demographic'].iloc[0]

    elections_won = 1

    # from the linear model PartyCat + elections_won*Demographic
    intercept = 4.7503
    ALP = -1.2831
    LNP = -1.9874
    LP = -3.2023
    elections_won = 0.5673
    Rural = 1.0630
    OuterMetropolitan = 2.2023
    Provincial = 1.5914
    elections_won_Rural = -0.2173
    elections_won_OuterMetropolitan = -0.7526
    elections_won_Provincial = -0.4806

    if Demographic=='Inner Metropolitan':
        elections_won_boost = 0.5673  
    elif Demographic=='Outer Metropolitan':
        elections_won_boost = 0.5673 - 0.7526
    elif Demographic == 'Provincial':
        elections_won_boost = 0.5673 - 0.4806
    else:
        elections_won_boost = 0.5673 - 0.2173

    # need to find these 2 quantities!
    curr_inc_adv = 1
    still_incumbent = 1

    inc_adv_change = elections_won_boost if still_incumbent else -curr_inc_adv

    estimates = {'Intercept':3.9380, 'elections_won':0.8172, 'ALP*elections_won':-0.4875, 'LNP*elections_won':-0.7343,'LP*elections_won':-0.9842}

    # need info - Party, # years incumbent! - just get Final_HS_df for R!


    return incumbency_change_adjusted_votes


def split_into_c2_dict(Div_parties_next_dict, map_new_seats_to_old_seats):

    # make dicts of existing parties and new parties/INDs in each seat
    c2_dict, new_parties_dict = {},{}

    for div in Div_parties_next_dict.keys():

        party_list = Div_parties_next_dict[div].copy()
        new_party_list = []

        # get Senate_party_abvs of new seats using map to old seat
        div_to_get_senate = map_new_seats_to_old_seats[div] if div in new_seats_list else div

        for p in Div_parties_next_dict[div]:

            if data_year in ['2007','2010']:
                print('check what should happen for LNP Coalition in QLD? and how COAL_set process below should evolve')
                # I think that 2010 LNP should just be made into COAL to match the LP+NP COAL in the 2007 senate?
                import pdb;pdb.set_trace()

            if p not in Senate_party_abvs_dict[div_to_get_senate]:
                if not ((p in ['LP','NP']) & ((div_to_state_dict[div_to_get_senate] in ['VIC','NSW']) | ((data_year == '2007') & (div_to_state_dict[div_to_get_senate]=='QLD')))):

                    print(p, div)
                    index = party_list.index(p)
                    new_party = party_list.pop(index)
                    new_party_list.append(new_party)

        # convert COAL double-divs to COAL
        COAL_set = {'NP','LP'}

        if (len(set(party_list) & COAL_set) == 2) & ((div_to_state_dict[div_to_get_senate] in ['VIC','NSW']) | ((data_year == '2007') & (div_to_state_dict[div_to_get_senate]=='QLD'))): # Both members of Coalition in div!
            
            for i, party in enumerate(party_list):
                if (party=='NP') | (party =='LP'):
                    party_list[i] = 'COAL' + party # rename to COALLP

        elif (div_to_state_dict[div_to_get_senate] in ['VIC','NSW']) | ((data_year == '2007') & (div_to_state_dict[div_to_get_senate]=='QLD')):
            # convert LP and NP in VIC/NSW to COAL
            for i, party in enumerate(party_list):
                if (party=='NP') | (party =='LP'):
                    party_list[i] = 'COAL'


        c2_dict[div] = party_list
        new_parties_dict[div] = new_party_list

    return c2_dict, new_parties_dict


def perform_Ideology_donation(Ideology_Donation_df, First_Prefs_By_PP_Complete_Redistributed, new_parties_dict, c2_dict, new_seats_list, div_to_state_dict, map_new_seats_to_old_seats, next_year, Ideo_Categories, party_type = 'new'):
    # partitions keys of First_Prefs dict by new_div, applying new_party expansion through ad-hoc-weighted proportion estimation

    # now add the poor candidates that are new, using New Candidates Allocation

    IND_dict = {}

    for div in new_parties_dict.keys():

        div_to_get_senate = map_new_seats_to_old_seats[div] if div in new_seats_list else div

        if 'IND' in new_parties_dict[div]:
            IND_dict[div] = ['IND']
            new_parties_dict[div] = [p for p in new_parties_dict[div] if p != 'IND']
        else:
            IND_dict[div] = []

        new_party_list = new_parties_dict[div] if party_type=='new' else IND_dict[div]
        if new_party_list:

            num_parties = len(c2_dict[div]) + 1 + len(new_party_list) # ensure that if multiple extra candidates that they split the 'new' vote!
            for new_party in set(new_party_list): # randomised order - essential for unbiasedness

                new_row = pd.DataFrame({'Year':[next_year],'div_nm':[div],'State':[div_to_state_dict[div_to_get_senate]],'Ideo_Category':[party_category_dict[new_party]],'Num_parties':[num_parties]})
                new_row[Ideo_Categories] = np.nan

                estimated_row = estimate_donation_proportion(Ideology_Donation_df, new_row, Ideo_Categories)

                # all pairs with div as 2nd value get the new_candidates adjustment
                current_pairs = [key for key in First_Prefs_By_PP_Complete_Redistributed.keys() if key[1] == div]

                for pair in current_pairs:
                    FP_df = First_Prefs_By_PP_Complete_Redistributed[pair]


                    donated_votes = FP_df.iloc[:,:-1].mul(estimated_row[Ideo_Categories].loc[0, FP_df.columns[:-1].map(party_category_dict)].values, axis=1)
                    FP_df = FP_df.astype(float)
                    FP_df.iloc[:,:-1] = FP_df.iloc[:,:-1] - donated_votes.values
                    FP_df.loc[:,new_party] = donated_votes.sum(axis=1)

                    First_Prefs_By_PP_Complete_Redistributed[pair] = FP_df[[col for col in FP_df.columns if col != 'INFORMAL'] + ['INFORMAL']] # ensure INFORMAL col is always last
    #import pdb;pdb.set_trace()


    return First_Prefs_By_PP_Complete_Redistributed

def finalise_independent_vote_FP_pair(name_changes_year_dict, data_year, pair, FP_pair_df, IND_FP_pair_df):
    # inputs dfs of non-IND votes and latent IND votes, respectively, and combines them to a final df, with transformation.

    FP_pair_df = FP_pair_df.astype(float)
    FP_pair_df = FP_pair_df.subtract(IND_FP_pair_df)
    FP_pair_df.loc[:,'IND'] = IND_FP_pair_df.iloc[:,:-1].sum(axis=1)

    ordered_cols = FP_pair_df.columns.tolist() # reorder cols such that 'INFORMAL' is last
    ordered_cols.remove('INFORMAL')
    ordered_cols.append('INFORMAL')
    FP_pair_df = FP_pair_df[ordered_cols]

    #import pdb;pdb.set_trace()

    # round votes correctly; INFORMAL should already be last!
    FP_pair_df = transform_to_raw_votes(FP_pair_df, pair[0], name_changes_year_dict, data_year, IS_FINAL_TRANSFORMATION = True)

    return FP_pair_df.reset_index('pp_id')







def whole_procedure(Formal_prefs_dict,general_party_df, Senate_party_abvs_dict, list_of_DOP_dicts, new_seats_list, name_changes_year_dict, div_to_state_dict, data_year, x=5):
    
    if data_year not in BTL_ONLY_ELECTIONS:
        Formal_prefs_dict = allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict, general_party_df, Senate_party_abvs_dict)

    print("done", time.time() - start)
    # make list of senate parties for check if they match house ones
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) # currently off

    Incumbent_advantage = 0
    candidate_change_redistribution = 0
    electorate_similarity = 0
    new_candidates_allocation = 1


    if Incumbent_advantage:

        Final_x_House_df = pd.read_csv(f"{data_year}Final_{x}_for_Incumbency.csv")
        #Final_x_House_df = Final_x_House_df.loc[Final_x_House_df['div_nm'].isin(list(Formal_prefs_dict.keys())),] # extra for testing!
        Final_x_House_dict = {name: list(Final_x_House_df.loc[Final_x_House_df['div_nm'] == name, 'PartyAb']) for name in Final_x_House_df['div_nm'].unique()}


        application_dict = Final_x_House_dict

        Final_allocated_pcts_aggregated_dict = allocate_Formal_prefs_by_1234(Formal_prefs_dict, Senate_party_abvs_dict, application_dict)

        Senate_votes = pd.concat([df.melt(id_vars=["div_nm"], value_vars=df.columns[1:], var_name="PartyAb", value_name="Senate_Pct") for df in Final_allocated_pcts_aggregated_dict.values()], ignore_index=True).reset_index(drop=True)

        import pdb;pdb.set_trace()
        Final_x_HS_df = pd.concat([Final_x_House_df, Senate_votes.drop(columns=['div_nm','PartyAb'])], axis=1)[["div_nm","PartyAb","is_incumbent","is_historic_incumbent","elections_won","copied_PartyAb","House_Pct","Senate_Pct"]]
        Final_x_HS_df.loc[:,"Senate_Pct"] = (pd.to_numeric(Final_x_HS_df.loc[:, "Senate_Pct"].astype(str), errors="coerce") * 100).round(2) # fixes issues with string values somehow????

        import pdb;pdb.set_trace()

        Final_x_HS_df.to_csv(f"{data_year}Final_{x}_HS_df.csv", index=False)

    else:
        Incumbency_by_div = pd.read_csv(f"{data_year}Incumbents.csv", index_col = None)
        Incumbency_by_div['div_nm'] = Incumbency_by_div['div_nm'].replace(name_changes_year_dict)
        #import pdb;pdb.set_trace()

        Formal_prefs_dict = amend_Formal_prefs_dict(Formal_prefs_dict, data_year, name_changes_year_dict, all_states= 1 - candidate_change_redistribution) 

        if candidate_change_redistribution:
            Redistribution_pairs_df = pd.read_csv(f"RedistributionPairs{str(int(data_year)+2)}.csv", index_col = None).iloc[1:2,]

            # to test: 1. get Ballarat's initial votes via Pref_Percent dict, transform them into counts, then try a round of redistribution_change
            #import pdb;pdb.set_trace()
            #Redistribution_pairs_df = pd.read_csv(f"RedistributionPairs{str(int(data_year)+2)}.csv", index_col = None).iloc[1:2,:]
            #
            #Ballarat_FP_votes = DOP_By_PP_Pref_Percent_wide_dict['Banks'].loc[DOP_By_PP_Pref_Percent_wide_dict['Banks']['CountNumber']==0,].drop('CountNumber', axis=1).set_index('pp_id')
            #full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year,votes_to_reduce=Ballarat_FP_votes)
            #Ballarat_FP_votes_counts = transform_to_raw_votes(Ballarat_FP_votes, 'Ballarat', name_changes_year_dict, data_year).drop('INFORMAL', axis = 1)
            #import pdb;pdb.set_trace()


            transformed_votes = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year)
            import pdb;pdb.set_trace()

            latent_IND_votes = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, IND_VOTES_ONLY=True)

            
            
            import pdb;pdb.set_trace()
        elif electorate_similarity:

            Omnipresent_parties = ['LP','ALP','GRN'] # Only 3 because Palmer is not in NT Senate in 2022! Aargh

            transformed_votes = reduce_to_Omnipresent_parties(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, new_seats_list, Omnipresent_parties, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year)

        elif new_candidates_allocation:

            Elimination_order_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Reduce_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, DOP_div_expand_dict, DOP_div_reduce_dict, DOP_div_pref_percent_dict = list_of_DOP_dicts


            next_year = str(int(data_year) + 3)
            # 1. Get names of next election's parties in each div for comparison to senate
            DOP_By_Division_next = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'})[["div_nm","PartyAb"]].drop_duplicates()

            DOP_By_Division_next.loc[:,'PartyAb'] = DOP_By_Division_next.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
            Div_parties_next_dict = {div: group['PartyAb'].tolist() for div, group in DOP_By_Division_next.groupby("div_nm")}
            Redistribution_pairs_df = pd.read_csv(f'RedistributionPairs{str(int(data_year)+2)}.csv', index_col = None)

            if new_seats_list:
                map_new_seats_to_old_seats = Redistribution_pairs_df.loc[Redistribution_pairs_df['new_div'].isin(new_seats_list)].iloc[:1,].set_index('new_div')['old_div'].to_dict()

            c2_dict, new_parties_dict = split_into_c2_dict(Div_parties_next_dict, map_new_seats_to_old_seats)

            party_category_dict = make_party_category_dict()
            # Apply incumbency advantage effect!

            # 1. Get full votes from DOP_By_PP
            # 2. Determine Incumbency change by party by redistribution_pair
            # 3. Perform using adjust_c1_c2_for_incumbency_adv


            # This changes input values, meaning full_redistribution_candidate_change needs to be adjusted to perform step-by-step transitions based on DOP_By_PP proportions,
            # not directly reducing! Tricky, but task for tomorrow!
            #import pdb;pdb.set_trace()

            
            # Expand to new parties, as well as INDs if necessary
            Ideology_Donation_df = pd.read_csv('Ideology_Donation_df.csv', index_col = None)
            Ideology_Donation_IND_df = pd.read_csv('Ideology_Donation_IND_df.csv', index_col = None)

            

           # get all IND-> IND non-redistribution divisions
            IND_transition_required = []
            for div in Div_parties_next_dict.keys():
                if div not in new_seats_list:
                    if ('IND' in Div_parties_next_dict[div]) and any(col.startswith('IND') for col in DOP_div_pref_percent_dict[div].columns):
                        IND_transition_required.append(div)

            IND_transition_1_1_pairs_df = pd.DataFrame({'old_div': IND_transition_required, 'new_div': [f"{div}_{next_year}" for div in IND_transition_required]}) #.iloc[2:3,:]
            Transition_First_Prefs_By_PP_Complete_Redistributed = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, IND_transition_1_1_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, c2_dict=c2_dict)
            IND_Transition_First_Prefs_By_PP_Complete_Redistributed = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, IND_transition_1_1_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, c2_dict=c2_dict, IND_VOTES_ONLY=True)
            
            for pair in Transition_First_Prefs_By_PP_Complete_Redistributed.keys():
                vote_sum = Transition_First_Prefs_By_PP_Complete_Redistributed[pair].iloc[:,1:-1].sum(axis=1)
                IND_Transition_First_Prefs_By_PP_Complete_Redistributed[pair].iloc[:,1:] = IND_Transition_First_Prefs_By_PP_Complete_Redistributed[pair].iloc[:,1:].mul(vote_sum, axis=0).fillna(0)/100
                IND_Transition_First_Prefs_By_PP_Complete_Redistributed[pair].loc[:,'INFORMAL'] = 0.0 # so fit structure of perform_Ideology_donation

                Transition_First_Prefs_By_PP_Complete_Redistributed[pair] = Transition_First_Prefs_By_PP_Complete_Redistributed[pair].set_index('pp_id')
                IND_Transition_First_Prefs_By_PP_Complete_Redistributed[pair] = IND_Transition_First_Prefs_By_PP_Complete_Redistributed[pair].set_index('pp_id')


            #import pdb;pdb.set_trace()
            Transition_First_Prefs_By_PP_Complete_Redistributed = perform_Ideology_donation(Ideology_Donation_df, party_category_dict, Transition_First_Prefs_By_PP_Complete_Redistributed, new_parties_dict, c2_dict, new_seats_list, div_to_state_dict, map_new_seats_to_old_seats, next_year, Ideo_Categories)
            IND_Transition_First_Prefs_By_PP_Complete_Redistributed = perform_Ideology_donation(Ideology_Donation_df, party_category_dict, IND_Transition_First_Prefs_By_PP_Complete_Redistributed, new_parties_dict, c2_dict, new_seats_list, div_to_state_dict, map_new_seats_to_old_seats, next_year, Ideo_Categories)

            # get final independent vote, add to Transition_First_Prefs
            for pair in Transition_First_Prefs_By_PP_Complete_Redistributed.keys():

                FP_pair_df = Transition_First_Prefs_By_PP_Complete_Redistributed[pair]
                IND_FP_pair_df = IND_Transition_First_Prefs_By_PP_Complete_Redistributed[pair]

                Transition_First_Prefs_By_PP_Complete_Redistributed[pair] = finalise_independent_vote_FP_pair(name_changes_year_dict, data_year, pair, FP_pair_df, IND_FP_pair_df)
                
                #import pdb;pdb.set_trace()



            # if in Redistribution_pairs and DOP_By_Division_next has IND and DOP_By_Division has IND, first apply a normal full_redistribution_candidate_change
            true_redistribution_required = []
            for new_div in Redistribution_pairs_df['new_div'].unique():
                if ('IND' in Div_parties_next_dict[new_div]) and any(col.startswith('IND') for col in DOP_div_pref_percent_dict[new_div].columns):
                    true_redistribution_required.append(new_div)
            Initial_IND_Redistribution_pairs = Redistribution_pairs_df.loc[Redistribution_pairs_df['new_div'].isin(true_redistribution_required),] #.iloc[:1,]
            #import pdb;pdb.set_trace()
            # Step 1 done:
            Initial_First_Prefs_By_PP_Complete_Redistributed = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Initial_IND_Redistribution_pairs, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year)
            #import pdb;pdb.set_trace()
            # Step 2: submit correct df to redistribute - make a loop over all redistribution pairs
            IND_Initial_First_Prefs_By_PP_Complete_Redistributed = {}

            for pair in Initial_First_Prefs_By_PP_Complete_Redistributed.keys():
                div = pair[1]
                # undo effects of transform_to_raw_votes haha
                curr_pair_PP_df = Initial_First_Prefs_By_PP_Complete_Redistributed[pair].drop('INFORMAL', axis = 1).set_index('pp_id')

                for party in curr_pair_PP_df.columns: # re-add suffix for non-senate parties (including INDX!)
                    if party not in Senate_party_abvs_dict[div]:
                        curr_pair_PP_df.rename(columns = {party: party + div}, inplace = True)

                #import pdb;pdb.set_trace()

                Redistribution_pair_row = pd.DataFrame([[div,f"{div}_{next_year}"]], columns = ['old_div','new_div'])
                votes_to_reduce_dict = {div: curr_pair_PP_df.div(curr_pair_PP_df.sum(axis=1), axis=0)*100}
                # THIS IS NOT BY PP_ID - REDUCE/EXPAND SHOULD BE AS A DIVISION AS A WHOLE
                Initial_First_Prefs_By_PP_Complete_Redistributed[pair] = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pair_row, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, c2_dict=c2_dict, votes_to_reduce_dict = votes_to_reduce_dict, by_pp_id=False, older_div = pair[0])[(div,div)]
                IND_Initial_First_Prefs_By_PP_Complete_Redistributed[pair] = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pair_row, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, c2_dict=c2_dict, votes_to_reduce_dict = votes_to_reduce_dict, IND_VOTES_ONLY=True, by_pp_id=False)[(div,div)]
                
                # make into IND vote counts
                vote_sum = Initial_First_Prefs_By_PP_Complete_Redistributed[pair].iloc[:,1:-1].sum(axis=1)
                IND_Initial_First_Prefs_By_PP_Complete_Redistributed[pair].iloc[:,1:] = IND_Initial_First_Prefs_By_PP_Complete_Redistributed[pair].iloc[:,1:].mul(vote_sum, axis=0)/100
                IND_Initial_First_Prefs_By_PP_Complete_Redistributed[pair].loc[:,'INFORMAL'] = 0.0 # so fit structure of perform_Ideology_donation

                Initial_First_Prefs_By_PP_Complete_Redistributed[pair] = Initial_First_Prefs_By_PP_Complete_Redistributed[pair].set_index('pp_id')
                IND_Initial_First_Prefs_By_PP_Complete_Redistributed[pair] = IND_Initial_First_Prefs_By_PP_Complete_Redistributed[pair].set_index('pp_id')
           
            #import pdb;pdb.set_trace()
            # Next: apply new_divs - both non-IND and IND
            Initial_First_Prefs_By_PP_Complete_Redistributed = perform_Ideology_donation(Ideology_Donation_df, party_category_dict, Initial_First_Prefs_By_PP_Complete_Redistributed, new_parties_dict, c2_dict, new_seats_list, div_to_state_dict, map_new_seats_to_old_seats, next_year, Ideo_Categories)
            IND_Initial_First_Prefs_By_PP_Complete_Redistributed = perform_Ideology_donation(Ideology_Donation_df, party_category_dict, IND_Initial_First_Prefs_By_PP_Complete_Redistributed, new_parties_dict, c2_dict, new_seats_list, div_to_state_dict, map_new_seats_to_old_seats, next_year, Ideo_Categories)
            #import pdb;pdb.set_trace()

            for pair in Initial_First_Prefs_By_PP_Complete_Redistributed.keys():

                FP_pair_df = Initial_First_Prefs_By_PP_Complete_Redistributed[pair]
                IND_FP_pair_df = IND_Initial_First_Prefs_By_PP_Complete_Redistributed[pair]
                Initial_First_Prefs_By_PP_Complete_Redistributed[pair] = finalise_independent_vote_FP_pair(name_changes_year_dict, data_year, pair, FP_pair_df, IND_FP_pair_df)
                #import pdb;pdb.set_trace()

            # Add 1:1 correspondence to redistribution pairs - now every single pairing changed or unchanged has new candidate set; new_div gets next year's suffix to ensure unique div1,div2 for Coalition doubble divs 
            for div in Div_parties_next_dict.keys():
                if (div not in new_seats_list) and div not in IND_transition_required:
                    new_row = pd.DataFrame({'old_div':[div],'new_div':[f'{div}_{next_year}']})
                    Redistribution_pairs_df = pd.concat([Redistribution_pairs_df,new_row], ignore_index=True)
            # Step 3
            # Only keep pairs that are not in Initial_IND_Redistribution_pairs
            #import pdb;pdb.set_trace()
            Redistribution_pairs_df = Redistribution_pairs_df.merge(Initial_IND_Redistribution_pairs, how="left", indicator=True).query('_merge == "left_only"').drop(columns=['_merge']) #.iloc[[0,2,-1],]

            # NOW WE HAVE 3 ALLOCATION DFS: Initial_IND_Redistribution_pairs, IND_transition_1_1_pairs_df, Redistribution_pairs_df.
            First_Prefs_By_PP_Complete_Redistributed = full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, c2_dict=c2_dict)
            #import pdb;pdb.set_trace()

            for pair in First_Prefs_By_PP_Complete_Redistributed:
                First_Prefs_By_PP_Complete_Redistributed[pair] = First_Prefs_By_PP_Complete_Redistributed[pair].set_index('pp_id')

            # first new_parties, then IND
            First_Prefs_By_PP_Complete_Redistributed = perform_Ideology_donation(Ideology_Donation_df, party_category_dict, First_Prefs_By_PP_Complete_Redistributed, new_parties_dict, c2_dict, new_seats_list, div_to_state_dict, map_new_seats_to_old_seats, next_year, Ideo_Categories)
            First_Prefs_By_PP_Complete_Redistributed = perform_Ideology_donation(Ideology_Donation_IND_df, party_category_dict, First_Prefs_By_PP_Complete_Redistributed, new_parties_dict, c2_dict, new_seats_list, div_to_state_dict, map_new_seats_to_old_seats, next_year, Ideo_Categories, party_type='IND')

            for pair in First_Prefs_By_PP_Complete_Redistributed.keys():
                First_Prefs_By_PP_Complete_Redistributed[pair] = transform_to_raw_votes(First_Prefs_By_PP_Complete_Redistributed[pair], pair[0], name_changes_year_dict, data_year, IS_FINAL_TRANSFORMATION = True)


            import pdb;pdb.set_trace()

            First_Prefs_By_PP_Complete_Allocated = {**Initial_First_Prefs_By_PP_Complete_Redistributed, **Transition_First_Prefs_By_PP_Complete_Redistributed, **First_Prefs_By_PP_Complete_Redistributed}

            # IF NEEDS NEW INDEPENDENT TRANSITION, PERFORM IT!
            # 1. Find if last_year has IND (of new_div id redistributed) and new_year has IND

            # 2. Perform latent_IND_votes via full_redistribution_candidate_change(Formal_prefs_dict, Senate_parties_by_div, list_of_DOP_dicts, Incumbency_by_div, Redistribution_pairs_df, new_seats_list, name_changes_year_dict, div_to_state_dict, party_category_dict, data_year, IND_VOTES_ONLY=True)
            # 3. Expand via Ideologies
            # 4. Apply final result to get final estimate

            
            # Expand to new INDs if so
                    

            import pdb;pdb.set_trace()


    return Final_allocated_pcts_aggregated_dict, Final_x_HS_df if Incumbent_advantage else First_Prefs_By_PP_Complete_Allocated




new_seats_list = new_seats_year_dict[data_year]
name_changes_year_dict = name_changes_year_dict[data_year]

list_of_DOP_dicts = [Elimination_order_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Reduce_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, DOP_div_expand_dict, DOP_div_reduce_dict, DOP_div_pref_percent_dict]
Final_allocated_pcts_aggregated_dict, Final_x_HS_df = whole_procedure(Formal_prefs_dict, general_party_df, Senate_party_abvs_dict, list_of_DOP_dicts, new_seats_list, name_changes_year_dict, div_to_state_dict, data_year, x=5)



import pdb;pdb.set_trace()
