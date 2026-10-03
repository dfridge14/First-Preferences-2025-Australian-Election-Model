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
import re

import pickle

import sys
import pdb
import traceback

def exception_handler(type, value, tb):

    print("\a")  # Rings the system bell
    os.system('echo -e "\\a"')  # Extra bell command for reliability
    os.system('tput bel')  # This forces the terminal to beep

    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

# automatic error debugging
sys.excepthook = exception_handler


base_dir = Path.home() / "Australian Election"
os.chdir(base_dir)

data_year = '2025'

FP_ID_COLUMNS = [3,4,5] # remove id columns
START_OF_PREFS = 2 # Prefs begin on the 3th column (after div_nm,pp_nm) - deleted stateab to accomodate 2016 file
BTL_ONLY_ELECTIONS = ['2007','2010','2013']

NUM_OF_INDX_LETTERS = 4
SMALL_CONST_FOR_LATENT_IND = 1e-10

# INCUMBENT_ADVANTAGE_DICT = {5:4.68,4:4.5,3:6}
FINAL_CAND_NO_DICT = {"2025":4,"2022":5, "2019": 4, "2016": 4,"2013": 5, "2010": 3, "2007": 4, "2004": 4,"2001":4}

FINAL_CANDIDATE_NO = FINAL_CAND_NO_DICT[data_year]

NEW_SEATS_YEAR_DICT = {'2025':[],'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner','Gorton']}
NAME_CHANGES_YEAR_DICT = {'2025':{},'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
STATES_TO_REDISTRIBUTE_DICT = {'2022': ['NSW','VIC','WA','NT'],'2019': ['VIC','WA'],'2016':['ACT','NT','QLD','SA','TAS','VIC'],'2013':['ACT','NSW','WA'],'2010':['SA','VIC'],'2007':['NSW','NT','QLD','TAS','WA'],'2004':['ACT','NSW','QLD'],'2001':['QLD','SA','VIC']}
NEW_SEAT_PARTY_DICTS = {'2025':{},'2022':{'Bullwinkel':['LP','ALP','GRN','UAPP','ON']},'2019':{'Hawke':['ALP','COAL','GRN','UAPP']},'2016':{'Bean':['ALP','LP','GRN'],'Fraser':['ALP','COAL','GRN']},'2013':{'Burt':['ALP','LP','GRN']},'2010':{},'2007':{'Wright':['ALP','LNP','GRN']},'2004':{'Flynn':['ALP','COALLLLLL','GRN']},'2001':{'Bonner':['ALP','LP','GRN'],'Gorton':['ALP','LP','GRN']}} # 2007 Flynn is tricky, as both LP and NP contested!!!



IDEO_CATEGORIES = ['Left','ALP','Centre','COAL','Right']
MIXING_RATIO_FOR_IND_ASSOCIATION = 0.5
COMPLEX_CONTESTS = ['Calare','Monash','Moore']




def abbreviate_party_names(party_names_list, general_party_df):
    # handles exceptions to party names

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
                party_abvs_list.append('FUT') # Fixed to FUT to help with 2018 Wentworth by-election
            elif party == 'Australian Sex Party/Marijuana (HEMP) Party':
                party_abvs_list.append('SXHM')
            elif party == 'A.F.N.P.P.':
                party_abvs_list.append('FNPP')
            elif party == 'Gerard Rennick People First - Heart': # 2025 additions
                party_abvs_list.append('GRHE')
            elif party == 'Libertarian / HEART / Gerard Rennick People First': 
                party_abvs_list.append('PFHL')
            elif party == 'HEART/LIBERTARIAN':
                party_abvs_list.append('HETP')
            elif party == "Gerard Rennick People First|Katter's Australian Party": 
                party_abvs_list.append('GRKA')
            elif party == "Great Australian Party & HEART Party": 
                party_abvs_list.append('GAHE')
                
            else:
                if general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].empty:
                    import pdb;pdb.set_trace()
                    continue
                party_abvs_list.append(general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].iloc[0])
        else:
            party_abvs_list.append('')


        #import pdb;pdb.set_trace()

    return party_abvs_list



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
    if not os.path.exists(f"{data_year}Senate_parties_by_div.csv"):
        Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) 

    return Senate_party_abvs_dict



# get state-to-div dict, adjusting for name changes
div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
div_to_state_dict = {NAME_CHANGES_YEAR_DICT[data_year].get(div, div): div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}


general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',"PartyAb"] = 'GRN' # handle exceptions, but think GVIC is the only one


Senate_party_abvs_dict = get_Senate_party_abvs_dict(data_year, div_to_state_dict, to_csv = False)


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
    # computes Transfer Count/Preference Count; more efficient version using vectorised operations
    
    # Pivot table to get values in a single row
    pivot = df.pivot(index=df.columns[:-2].tolist(), columns="CalculationType", values="CalculationValue") # preserve order of parties as BallotPosition is before cand_id

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
    ### appends div_nm onto any INDXs and changes COAL member parties to COAL or COALNP/COALLP for when both LP and NP contest a seat

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
    ### Processes the data - a dictionary of dfs about Elimination Order, Expand, PrefPercent and Reduce - in 4 ways: 
    # 1. Fills Blanks (NAFD) with IND 
    # 2. Given IND numeric label i.e. IND1,IND2 
    # 3. GVIC --> GRN 
    # 4. Processes INDs/COAL names correctly

    ### In future simplify the function by reducing duplication
    
    DOP_table_wide_dict = {}

    if DOP_type == "EliminationOrder":
        # get state-to-div dict

        # print("Parties with no senate comparison:") # prints out party + div


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
                        #print(party + div, 'Double div')
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
                        #print(party + div)

                    

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



######### Candidate Pairs stuff
DOP_By_PP_Expand = pd.read_csv(f"{data_year}DOP_By_PP_Expand.csv", index_col=None)
DOP_By_PP_Pref_Percent = pd.read_csv(f"{data_year}DOP_By_PP_Pref_Percent.csv", index_col=None)
DOP_By_PP_Reduce = pd.read_csv(f"{data_year}DOP_By_PP_Reduce.csv", index_col=None)

# create wide format eliminaation_order_dict
DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)
DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
Div_DOP_dict = {div: group.drop(columns=['div_nm']) for div, group in DOP_By_Division[["div_nm","CountNumber","BallotPosition","cand_id", "PartyAb","CalculationType", "CalculationValue"]].groupby("div_nm")}

Div_DOP_dict = {NAME_CHANGES_YEAR_DICT[data_year].get(key, key): val for key, val in Div_DOP_dict.items()} # adjust for name changes

Elimination_order_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "EliminationOrder")
DOP_div_expand_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "Expand")
DOP_div_pref_percent_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "PrefPercent")
DOP_div_reduce_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, Senate_party_abvs_dict, DOP_type = "Reduce")



DOP_By_PP_Pref_Percent_wide_dict = convert_long_to_wide_format(DOP_By_PP_Pref_Percent, div_to_state_dict, Senate_party_abvs_dict)
DOP_By_PP_Expand_wide_dict = convert_long_to_wide_format(DOP_By_PP_Expand, div_to_state_dict, Senate_party_abvs_dict)
DOP_By_PP_Reduce_wide_dict = convert_long_to_wide_format(DOP_By_PP_Reduce, div_to_state_dict, Senate_party_abvs_dict)




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

    #import pdb; pdb.set_trace()

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

        #import pdb; pdb.set_trace()
        return duplicate_votes_series

def allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict):

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




def check_house_senate_discrepancies(data_year, NAME_CHANGES_YEAR_DICT):

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



    # combine Others together
    #house_votes_full.loc[:,"pp_nm"] = house_votes_full.loc[:,"pp_nm"].apply(lambda x: 'Other' if any(x.startswith(prefix) for prefix in Other_booth_type_prefixes) else x)
    senate_votes_full.loc[:,"pp_nm"] = senate_votes_full.loc[:,"pp_nm"].apply(lambda x: 'Other' if any(x.startswith(prefix) for prefix in Other_booth_type_prefixes) else x)

    #house_votes_full = house_votes_full.groupby(["div_nm", "pp_nm"], as_index=False).agg({'house_votes':'sum'})
    senate_votes_full = senate_votes_full.groupby(["div_nm", "pp_nm"], as_index=False).agg({'senate_votes':'sum'})





    formal_senate_full_house_comparison = pd.DataFrame(house_votes_full).merge(pd.DataFrame(senate_votes_full), on = ['div_nm','pp_nm'], how='left')
    formal_senate_full_house_comparison['div_nm'] = formal_senate_full_house_comparison['div_nm'].replace(NAME_CHANGES_YEAR_DICT)
    formal_senate_full_house_comparison.loc[:,'house-sen'] = (formal_senate_full_house_comparison.loc[:,'house_votes'] - formal_senate_full_house_comparison.loc[:,'senate_votes']).values          


    formal_senate_full_house_comparison.loc[:,'house/sen'] = (formal_senate_full_house_comparison.loc[:,'house_votes'] / formal_senate_full_house_comparison.loc[:,'senate_votes']).values

    #print(formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison["house/sen"] < 1) & (formal_senate_full_house_comparison['house-sen']<500),])

    #print(formal_senate_full_house_comparison.loc[(formal_senate_full_house_comparison["house/sen"] > 1.2) & (formal_senate_full_house_comparison['house-sen']<500),])

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

def amend_Formal_prefs_dict(Formal_prefs_dict, data_year, NAME_CHANGES_YEAR_DICT, all_states = False):
    ### all_states is an argument that dictates if Formal_prefs_dict is defined for all states, or only redistribution ones

    #import pdb;pdb.set_trace()

    h_s_discrepancies = check_house_senate_discrepancies(data_year, NAME_CHANGES_YEAR_DICT)

    if data_year == '2025':
        # ONLY DONE ACT and Farrer so far!
        
        # 1. Parkes (Bean, Canberra, Fenner)
        FP_div = Formal_prefs_dict['Canberra']
        lender = 'Parkes (Canberra)'
        borrower = 'Parkes (Bean)'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Bean'
        Formal_prefs_dict['Bean'] = pd.concat([Formal_prefs_dict['Bean'],lender_FPs], ignore_index=True)

        import pdb; pdb.set_trace()

        FP_div = Formal_prefs_dict['Canberra']
        lender = 'Parkes (Canberra)'
        borrower = 'Parkes (Fenner)'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Fenner'
        Formal_prefs_dict['Fenner'] = pd.concat([Formal_prefs_dict['Fenner'],lender_FPs], ignore_index=True)

        # 2. Parkes PPVC (Bean, Canberra, Fenner)

        # Fenner from Gungahlin
        FP_div = Formal_prefs_dict['Fenner']
        lender = 'Gungahlin FENNER PPVC'
        borrower = 'Parkes FENNER PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        Formal_prefs_dict['Fenner'] = pd.concat([FP_div,lender_FPs], ignore_index=True)

        # Bean from Canberra PPVC
        FP_div = Formal_prefs_dict['Canberra']
        lender = 'Parkes CANBERRA PPVC'
        borrower = 'Parkes BEAN PPVC'

        lender_FPs = FP_div.loc[FP_div['pp_nm'] == lender,]
        lender_FPs.loc[:,'pp_nm'] = borrower
        lender_FPs.loc[:,'div_nm'] = 'Bean'
        Formal_prefs_dict['Bean'] = pd.concat([Formal_prefs_dict['Bean'],lender_FPs], ignore_index=True)



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
            lender = 'Waverley WENTWORTH PPVC'
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
        Formal_prefs_dict["Fenner"] = pd.concat([Formal_prefs_dict["Fenner"],lender_FPs], ignore_index=True)

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



with open(f"Formal_prefs_dict_{data_year}.pkl", "rb") as f:
    Formal_prefs_dict = pickle.load(f)
    
Formal_prefs_dict = amend_Formal_prefs_dict(Formal_prefs_dict, data_year, NAME_CHANGES_YEAR_DICT[data_year], all_states= 0)



# CHECK: can the following be removed, or should Booth_name_pp_id be passed into allocate_formal_preferences_to_allocation_set?

# get correpondence between Booth name and pp_id for given year
Polling_Places_df = pd.read_csv(f"{data_year}GeneralPollingPlaces.csv", index_col = None, skiprows = 1)
Polling_Places_df = Polling_Places_df.iloc[:,2:6].rename(columns={'DivisionNm': 'div_nm','PollingPlaceID': 'pp_id','PollingPlaceNm':'pp_nm'})
Polling_Places_df = Polling_Places_df.loc[Polling_Places_df['PollingPlaceTypeID'].isin([1,5]),].drop('PollingPlaceTypeID', axis=1)

for div in Polling_Places_df['div_nm'].unique().tolist():
    Other_row = pd.DataFrame({"div_nm": [div],"pp_id":[0],"pp_nm":["Other"]})
    Polling_Places_df = pd.concat([Polling_Places_df,Other_row], ignore_index=True)

Booth_name_pp_id = Polling_Places_df

Booth_name_pp_id['div_nm'] = Booth_name_pp_id['div_nm'].replace(NAME_CHANGES_YEAR_DICT[data_year])

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

    zero_columns = reduced_votes_by_PP.columns[(reduced_votes_by_PP.eq(0) | reduced_votes_by_PP.isna()).all()].tolist()


    reduced_votes_by_PP.loc[:,zero_columns] = reduced_votes_by_PP[zero_columns].astype(float) # avoids type warning for 2022 McEwen Casey/Nicholls
    reduced_votes_by_PP.loc[:,zero_columns] = np.nan # convert other cols to nan
    reduced_votes_by_PP = reduced_votes_by_PP.loc[:, ~reduced_votes_by_PP.iloc[0].isna()] # remove nan columns - don't need to store extra c1 candidates!

    if (div in Coalition_double_divs) and combine_double_divs and ('COALLP' in reduced_votes_by_PP.columns) and ('COALNP' in reduced_votes_by_PP.columns):
        #import pdb;pdb.set_trace()
        reduced_votes_by_PP = combine_coalition(reduced_votes_by_PP)[0] # finally, combine 'COALLP' and 'COALNP' into 'COAL'
    elif div in Coalition_double_divs:
        reduced_votes_by_PP = reduced_votes_by_PP.copy().rename(columns={'COALNP':'COAL', 'COALLP':'COAL'})

    #import pdb;pdb.set_trace()

    return reduced_votes_by_PP


def expand_candidates_to_set_size(div, reduced_votes_by_PP, DOP_div_expand_dict, DOP_div_pref_percent_dict, c_size, expanded_c_size, list_div2, Coalition_double_divs = [], combine_double_divs = True):
    ### expands candidate set to expanded_c_size, unfortunately using the DOP of the whole div as opposed to by pp_id
    ### if Coalition_double_divs, expects combined data if combine_double_divs = True

    wide_df_expand = DOP_div_expand_dict[div]

    expanded_votes = reduced_votes_by_PP

    # this should be measured only by whether 'COALLP' and 'COALNP' in columns of expanded_votes
    if (div in Coalition_double_divs) and combine_double_divs:
        #import pdb;pdb.set_trace()
        if ('COAL' in expanded_votes.columns) and ('COALLP' in list_div2) and ('COALNP' in list_div2):
            c_size = max([list_div2.index('COALLP'),list_div2.index('COALNP')])+1 # makes sure that both COALLP and COALNP are there to be proportioned
            coalition_proportions = combine_coalition(reduce_candidates_to_set_size(div, DOP_div_pref_percent_dict, c_size, by_pp_id=False))[1] # get coalition proportions of div2; DON'T PASS Coalition_double_divs SO IT RETURNS SEPARATED!!!
            expanded_votes = separate_coalition(expanded_votes, coalition_proportions, by_pp_id=False) # split coalition into 2 again
        #c_size += 1
        # I THINK THAT NEEDED TO ADD 1 TO C_SIZE, ASSUMING THAT C2 DOES NOT INCLUDE DUPLICATED COAL.
    elif div in Coalition_double_divs:
        # just rename the one that exists!
        import pdb;pdb.set_trace()
        if 'COAL' in expanded_votes.columns:
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


def transform_to_raw_votes(redistributed_votes, giver_div, NAME_CHANGES_YEAR_DICT, data_year, IS_FINAL_TRANSFORMATION = False):

    First_Prefs_By_PP_Complete = pd.read_csv(f"{data_year}FirstPrefsByPPComplete.csv", index_col = None)[['pp_id','div_nm','PartyAb','votes']]
    First_Prefs_By_PP_Complete['div_nm'] = First_Prefs_By_PP_Complete['div_nm'].replace(NAME_CHANGES_YEAR_DICT)
    First_Prefs_By_PP_div = First_Prefs_By_PP_Complete.loc[First_Prefs_By_PP_Complete['div_nm'] == giver_div,].drop(columns = 'div_nm', axis = 1)

    INFORMAL_df = First_Prefs_By_PP_div[First_Prefs_By_PP_div['PartyAb'] == 'INFORMAL'].rename(columns = {'votes':'INFORMAL'}).drop(columns = ['PartyAb'], axis=1).set_index('pp_id').sort_index()
    FORMAL_df = First_Prefs_By_PP_div[First_Prefs_By_PP_div['PartyAb'] != 'INFORMAL']

    grouped_FORMAL = FORMAL_df.groupby('pp_id', as_index=False).agg({'votes': 'sum'}).set_index('pp_id').sort_index()

    # correct for any issues with 0-0 house-senate votes! PERHAPS BETTER TO REMOVE ALTOGETHER???
    grouped_FORMAL = grouped_FORMAL.reindex(redistributed_votes.index, fill_value=0) # ensure no issue when there are 0 votes in House & Senate (excluded from Formal Prefs & First_Prefs_By_PP_div)
    INFORMAL_df = INFORMAL_df.reindex(redistributed_votes.index, fill_value=0)

    mask = redistributed_votes.sum(axis=1) == 0
    redistributed_votes.loc[mask] = redistributed_votes.loc[mask].fillna(0)

    if not (IS_FINAL_TRANSFORMATION | ((s := redistributed_votes.sum(axis=1)).eq(0) | s.between(99, 101)).all()):
        
        #import pdb;pdb.set_trace()
        invalid_rows = ~((s := redistributed_votes.sum(axis=1)).between(99, 101) | s.eq(0))
        redistributed_votes.loc[invalid_rows] = redistributed_votes.loc[invalid_rows].div(s[invalid_rows], axis=0).mul(100)
        print(giver_div, len(redistributed_votes.loc[invalid_rows]), "There is a 0 leak in reducing votes due to small independent. Will just rescale proportionally:")


    # scale redistributed_votes by grouped_FORMAL
    if not IS_FINAL_TRANSFORMATION:
        redistributed_votes_raw = (redistributed_votes / 100).mul(grouped_FORMAL['votes'], axis=0)
    else:
        if (redistributed_votes['INFORMAL'] != INFORMAL_df['INFORMAL']).sum():
            import pdb;pdb.set_trace()

        redistributed_votes_raw = redistributed_votes.drop('INFORMAL', axis=1)

    if redistributed_votes_raw.isna().any().any():
        print(giver_div, 'there are some nan cols')
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

# incumbency advantage calculate
#div = 'Franklin'
#allocation_abvs_list = ['ALP','LP','GRN','ON']
#allocation_set = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div)
#allocate_formal_preferences_to_allocation_set(data_year,  Formal_prefs_dict[div], allocation_set, by_pp_id=False, as_percent = True)

import pdb; pdb.set_trace()
# List of dicts: Elimination_order_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Reduce_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, DOP_div_expand_dict, DOP_div_reduce_dict, DOP_div_pref_percent_dict

# 1. Canberra -> Bean

Canberra_reduced_by_pp_id = reduce_candidates_to_set_size('Canberra', DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size = 3, by_pp_id = True)

div = 'Bean'
allocation_abvs_list = ['ALP','LP','GRN']
allocation_set_Bean = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div)

# allocate proportional to polling booth size when borrowing data from other booths
borrowed_condition = Formal_prefs_dict[div]['pp_nm'].isin(['Parkes (Bean)','Parkes BEAN PPVC'])
native_FPs = Formal_prefs_dict[div].loc[~borrowed_condition,]
borrowed_FPs = Formal_prefs_dict[div].loc[borrowed_condition,]
native_allocation = allocate_formal_preferences_to_allocation_set(data_year, native_FPs, allocation_set_Bean, by_pp_id=False, as_percent = True).set_index('div_nm')
borrowed_allocation = allocate_formal_preferences_to_allocation_set(data_year, borrowed_FPs, allocation_set_Bean, by_pp_id=False, as_percent = True).set_index('div_nm')
borrowed_n = 764 + 2382 # number of total voters for house in Parkes (Bean) and Parkes BEAN PPVC

Major_allocation = (native_allocation * len(native_FPs) + borrowed_allocation * borrowed_n) / (len(native_FPs) + borrowed_n)
Major_allocation.columns = allocation_abvs_list

# incumbency_advantage_change(div,Elimination_order_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, div_to_state_dict, party_category_dict, NAME_CHANGES_YEAR_DICT, Incumbency_by_div, FINAL_CANDIDATE_NO, data_year)

Bean_house_vote = reduce_candidates_to_set_size('Bean', DOP_div_pref_percent_dict, reduced_c_size = 4, by_pp_id = False)
IND_Price_defection = 1 - (Bean_house_vote[['ALP', 'LP', 'GRN']].iloc[0] / (Major_allocation[['ALP', 'LP', 'GRN']].iloc[0]*100))

Canberra_to_Bean = Canberra_reduced_by_pp_id.copy()
Canberra_to_Bean['IND1'] = (Canberra_to_Bean[['ALP', 'LP','GRN']] * IND_Price_defection).sum(axis=1)
Canberra_to_Bean[['ALP', 'LP','GRN']] *= (1 - IND_Price_defection)

Canberra_to_Bean_totals = transform_to_raw_votes(Canberra_to_Bean, 'Canberra', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Canberra_to_Bean_totals.columns = Canberra_to_Bean_totals.columns.str.removesuffix(div)

import pdb; pdb.set_trace()

# 2. Bean -> Canberra

div = 'Bean'

Bean_senate_reduced_by_pp_id = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[div], allocation_set_Bean, by_pp_id=True, as_percent = True)
Bean_senate_reduced_by_pp_id.columns = ['div_nm','pp_id'] + allocation_abvs_list
Bean_senate_reduced_by_pp_id = Bean_senate_reduced_by_pp_id.set_index('pp_id').sort_index().drop(columns=['div_nm'])



# calculate defection percentage by pp_id
Bean_house_by_pp_id = reduce_candidates_to_set_size('Bean', DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size = 4, by_pp_id = True)

senate_votes = Bean_senate_reduced_by_pp_id*100
house_votes = Bean_house_by_pp_id
list_div1_FP = ['ALP','LP','GRN']

Senate_minus_IND_house = senate_votes - house_votes.loc[:,list_div1_FP]
negative_sum = (Senate_minus_IND_house < 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
positive_sum = (Senate_minus_IND_house > 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
positive_sum = positive_sum.replace(0, np.nan) # redundant

proportions = Senate_minus_IND_house.div(positive_sum, axis=0) # negative vals will be damaged, but they will soon be ignored
Proportion_df = Senate_minus_IND_house + proportions.mul(negative_sum, axis=0)
Proportion_df[Proportion_df<0] = 0 # set negatives to 0         

# Defection_percent = Proportion_df.div(senate_votes).replace(np.nan,0)

# Proportion_df is the volume lost to IND. Add it back to the House baseline.
Bean_reduced_by_pp_id = Bean_house_by_pp_id.loc[:, list_div1_FP] + Proportion_df

div = 'Canberra'
Bean_to_Canberra = expand_candidates_to_set_size(div, Bean_reduced_by_pp_id, DOP_div_expand_dict, DOP_div_pref_percent_dict, 3, 6, ['GRN','IMOCanberra','IND1Canberra','LP','ALP','AJP'])
Bean_to_Canberra_totals = transform_to_raw_votes(Bean_to_Canberra, 'Bean', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Bean_to_Canberra_totals.columns = Bean_to_Canberra_totals.columns.str.removesuffix(div)

# 3. Canberra -> Fenner

div = 'Fenner'
Canberra_to_Fenner = expand_candidates_to_set_size(div, Canberra_reduced_by_pp_id, DOP_div_expand_dict, DOP_div_pref_percent_dict, 3, 4, ['ALP','LP','GRN','FFPAFenner'])
Canberra_to_Fenner_totals = transform_to_raw_votes(Canberra_to_Fenner, 'Canberra', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Canberra_to_Fenner_totals.columns = Canberra_to_Fenner_totals.columns.str.removesuffix(div)



# 4. Fenner -> Canberra

div = 'Canberra'
Fenner_reduced_by_pp_id = reduce_candidates_to_set_size('Fenner', DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size = 3, by_pp_id = True)
Fenner_to_Canberra = expand_candidates_to_set_size(div, Fenner_reduced_by_pp_id, DOP_div_expand_dict, DOP_div_pref_percent_dict, 3, 6, ['GRN','IMOCanberra','IND1Canberra','LP','ALP','AJP'])
Fenner_to_Canberra_totals =  transform_to_raw_votes(Fenner_to_Canberra, 'Fenner', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Fenner_to_Canberra_totals.columns = Fenner_to_Canberra_totals.columns.str.removesuffix(div)


# TASMANIA REDISTRIBUTION

list_div1_FP = ['ALP','LP','GRN','ON']
allocation_abvs_list = ['ALP','LP','GRN','ON']
n_senate_parties = 4

div = 'Bass'

# 1. Lyons -> Bass
Lyons_reduced_by_pp_id = reduce_candidates_to_set_size('Lyons', DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size = n_senate_parties, by_pp_id = True)
Lyons_to_Bass = expand_candidates_to_set_size(div, Lyons_reduced_by_pp_id, DOP_div_expand_dict, DOP_div_pref_percent_dict, n_senate_parties, 7, ['GRN','ON','CEC','IND1Bass','LP','ALP','CYA'])
Lyons_to_Bass_totals =  transform_to_raw_votes(Lyons_to_Bass, 'Lyons', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Lyons_to_Bass_totals.columns = Lyons_to_Bass_totals.columns.str.removesuffix(div)

import pdb; pdb.set_trace()

# 2. Clark -> Lyons


div = 'Clark'
allocation_set_Clark = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div)

Clark_senate_reduced_by_pp_id = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[div], allocation_set_Clark, by_pp_id=True, as_percent = True)
Clark_senate_reduced_by_pp_id.columns = ['div_nm','pp_id'] + allocation_abvs_list
Clark_senate_reduced_by_pp_id = Clark_senate_reduced_by_pp_id.set_index('pp_id').sort_index().drop(columns=['div_nm'])


# calculate defection percentage by pp_id
Clark_house_by_pp_id = reduce_candidates_to_set_size('Clark', DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size = 5, by_pp_id = True)

senate_votes = Clark_senate_reduced_by_pp_id*100
house_votes = Clark_house_by_pp_id

Senate_minus_IND_house = senate_votes - house_votes.loc[:,list_div1_FP]
negative_sum = (Senate_minus_IND_house < 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
positive_sum = (Senate_minus_IND_house > 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
positive_sum = positive_sum.replace(0, np.nan) # redundant

proportions = Senate_minus_IND_house.div(positive_sum, axis=0) # negative vals will be damaged, but they will soon be ignored
Proportion_df = Senate_minus_IND_house + proportions.mul(negative_sum, axis=0)
Proportion_df[Proportion_df<0] = 0 # set negatives to 0         

# Defection_percent = Proportion_df.div(senate_votes).replace(np.nan,0)

# Proportion_df is the volume lost to IND. Add it back to the House baseline.
Clark_reduced_by_pp_id = Clark_house_by_pp_id.loc[:, list_div1_FP] + Proportion_df

div = 'Lyons'
Clark_to_Lyons = expand_candidates_to_set_size(div, Clark_reduced_by_pp_id, DOP_div_expand_dict, DOP_div_pref_percent_dict, n_senate_parties, 8, ['GRN','ON','CEC','IND1Lyons','LP','ALP','CYA','ASP'])
Clark_to_Lyons_totals = transform_to_raw_votes(Clark_to_Lyons, 'Clark', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Clark_to_Lyons_totals.columns = Clark_to_Lyons_totals.columns.str.removesuffix(div)

import pdb; pdb.set_trace()

# 3. Franklin -> Clark



# reduce Franklin to senate only

div = 'Franklin'
allocation_set_Franklin = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div)

Franklin_senate_reduced_by_pp_id = allocate_formal_preferences_to_allocation_set(data_year, Formal_prefs_dict[div], allocation_set_Franklin, by_pp_id=True, as_percent = True)
Franklin_senate_reduced_by_pp_id.columns = ['div_nm','pp_id'] + allocation_abvs_list
Franklin_senate_reduced_by_pp_id = Franklin_senate_reduced_by_pp_id.set_index('pp_id').sort_index().drop(columns=['div_nm'])

# calculate defection percentage by pp_id
Franklin_house_by_pp_id = reduce_candidates_to_set_size('Franklin', DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size = 6, by_pp_id = True)

senate_votes = Franklin_senate_reduced_by_pp_id*100
house_votes = Franklin_house_by_pp_id

Senate_minus_IND_house = senate_votes - house_votes.loc[:,list_div1_FP]
negative_sum = (Senate_minus_IND_house < 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
positive_sum = (Senate_minus_IND_house > 0).astype(int).mul(Senate_minus_IND_house).sum(axis=1)
positive_sum = positive_sum.replace(0, np.nan) # redundant

proportions = Senate_minus_IND_house.div(positive_sum, axis=0) # negative vals will be damaged, but they will soon be ignored
Proportion_df = Senate_minus_IND_house + proportions.mul(negative_sum, axis=0)
Proportion_df[Proportion_df<0] = 0 # set negatives to 0         

# Proportion_df is the volume lost to IND. Add it back to the House baseline.
Franklin_reduced_by_pp_id = Franklin_house_by_pp_id.loc[:, list_div1_FP] + Proportion_df



div = 'Clark'
allocation_abvs_list = list_div1_FP
allocation_set_Clark = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div)

Major_allocation = allocate_formal_preferences_to_allocation_set(data_year,  Formal_prefs_dict[div], allocation_set_Clark, by_pp_id=False, as_percent = True).set_index('div_nm')
Major_allocation.columns = allocation_abvs_list

Clark_house_vote = reduce_candidates_to_set_size('Clark', DOP_div_pref_percent_dict, reduced_c_size = 5, by_pp_id = False)
IND_Wilkie_defection = 1 - (Clark_house_vote[list_div1_FP].iloc[0] / (Major_allocation[list_div1_FP].iloc[0]*100))

Franklin_to_Clark = Franklin_reduced_by_pp_id.copy()
Franklin_to_Clark['IND1'] = (Franklin_to_Clark[list_div1_FP] * IND_Wilkie_defection).sum(axis=1)
Franklin_to_Clark[list_div1_FP] *= (1 - IND_Wilkie_defection)

Franklin_to_Clark_totals = transform_to_raw_votes(Franklin_to_Clark, 'Franklin', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Franklin_to_Clark_totals.columns = Franklin_to_Clark_totals.columns.str.removesuffix(div)

import pdb; pdb.set_trace()

# 4. Lyons -> Franklin

div = 'Franklin'

Lyons_reduced_by_pp_id = reduce_candidates_to_set_size('Lyons', DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size = n_senate_parties, by_pp_id = True)


allocation_abvs_list = list_div1_FP
allocation_set_Franklin = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div)


Major_allocation = allocate_formal_preferences_to_allocation_set(data_year,  Formal_prefs_dict[div], allocation_set_Franklin, by_pp_id=False, as_percent = True).set_index('div_nm')
Major_allocation.columns = allocation_abvs_list

Franklin_house_vote = reduce_candidates_to_set_size('Franklin', DOP_div_pref_percent_dict, reduced_c_size = 6, by_pp_id = False)
IND_George_Blomeley_defection = 1 - (Franklin_house_vote[list_div1_FP].iloc[0] / (Major_allocation[list_div1_FP].iloc[0]*100))

Lyons_to_Franklin = Lyons_reduced_by_pp_id.copy()

# Novel: split both IND1 (George) and IND2 (Blomeley) - CEHCK syntax
IND1_ratio = Franklin_house_vote['IND1Franklin'].iloc[0]/(Franklin_house_vote[['IND1Franklin','IND2Franklin']].sum(axis=1).iloc[0])
Lyons_to_Franklin['IND1'] = (Lyons_to_Franklin[list_div1_FP] * IND_George_Blomeley_defection).sum(axis=1) * IND1_ratio
Lyons_to_Franklin['IND2'] = (Lyons_to_Franklin[list_div1_FP] * IND_George_Blomeley_defection).sum(axis=1) * (1-IND1_ratio)
Lyons_to_Franklin[list_div1_FP] *= (1 - IND_George_Blomeley_defection)

Lyons_to_Franklin_totals = transform_to_raw_votes(Lyons_to_Franklin, 'Lyons', NAME_CHANGES_YEAR_DICT[data_year], data_year).reset_index()
Lyons_to_Franklin_totals.columns = Lyons_to_Franklin_totals.columns.str.removesuffix(div)

import pdb; pdb.set_trace()

#First_Prefs_By_PP_Redistributed_dict = {('Bean', 'Canberra'): Bean_to_Canberra_totals, ('Canberra', 'Bean'): Canberra_to_Bean_totals, ('Fenner', 'Canberra'): Fenner_to_Canberra_totals, ('Canberra', 'Fenner'): Canberra_to_Fenner_totals}
First_Prefs_By_PP_Redistributed_dict = {('Lyons', 'Bass'): Lyons_to_Bass_totals, ('Franklin', 'Clark'): Franklin_to_Clark_totals, ('Lyons', 'Franklin'): Lyons_to_Franklin_totals, ('Clark', 'Lyons'): Clark_to_Lyons_totals}

output_folder = f"feather Redistribution pairs {str(int(data_year)+2)}"
os.makedirs(output_folder, exist_ok=True)
for key, sub_df in First_Prefs_By_PP_Redistributed_dict.items():
    filename = f"{output_folder}/{str(int(data_year)+2)}FPBPPRed_{key[0]}_{key[1]}.feather"
    sub_df.reset_index(drop=True).to_feather(filename)