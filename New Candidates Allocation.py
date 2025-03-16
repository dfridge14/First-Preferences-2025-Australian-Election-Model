import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path


# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election" / "New Candidate Files"
os.chdir(base_dir)

election_years = ['2004','2007','2010','2013','2016','2019','2022']

data_year = '2013'
#Party_details = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", index_col=None)


def convert_coalitions(Party_set, state, year):
    # valid for 2010 election onwards

    condition = lambda p: ((state in ['VIC','NSW']) or ((state == 'QLD') and (year == '2007'))) and (p in ['LPNP','LNP','LP','NP'])
    converted_set = {'COAL' if condition(p) else p for p in Party_set}

    converted_set.discard('UNAM') # removes ungrouped label
    converted_set.discard('IND')
    converted_set.discard('NAFD')

    return converted_set


# larger goal - find sets of parties that are new to given election, indicate if they are rebrands, mergers, reinstatements, or completely new


# get dict of years of dict of states of party lists in given election
Senate_parties_by_state_dict = {}
House_parties_by_state_dict = {}
states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']

grand_party_category_df = pd.DataFrame(columns = ['PartyAb','Ideo_Category','HouseYears','SenateYears'])

for year in election_years:
    Senate_parties_by_state_dict[year] = {}
    House_parties_by_state_dict[year] = {}

    Senate_parties_by_state = pd.read_csv(f"{year}SenateFirstPrefsByStateByGroupByVoteType.csv", index_col=None, skiprows=1).iloc[:,:3] # include StateAb,PartyAb/GroupAb,PartyNm/GroupNm
    House_parties_by_state = pd.read_csv(f"{year}HouseFirstPrefsByStateByParty.csv", index_col=None, skiprows=1).iloc[:,:3]

    for state in states:

        Senate_parties_by_state_curr = Senate_parties_by_state.loc[Senate_parties_by_state['StateAb'] == state,]
        House_parties_by_state_curr = House_parties_by_state.loc[House_parties_by_state['StateAb'] == state,]


        Senate_set = set(Senate_parties_by_state_curr['GroupAb'].unique()) # unique set of parties
        Senate_set = convert_coalitions(Senate_set, state, year)
        Senate_parties_by_state_dict[year][state] = Senate_set

        House_set = set(House_parties_by_state_curr['PartyAb'].unique()) # unique set of parties
        House_set = convert_coalitions(House_set, state, year)
        House_parties_by_state_dict[year][state] = House_set

        chambers = [['House','Senate'],['Senate','House']]
        for order in chambers:
            chamber, other_chamber = order
            p_set = House_set if chamber == 'House' else Senate_set 

            for p in p_set:
                if p not in grand_party_category_df['PartyAb'].values:
                    new_row = pd.DataFrame([{'PartyAb':p,'Ideo_Category':'',f"{chamber}Years":[year],f"{other_chamber}Years":[]}])
                    grand_party_category_df = pd.concat([grand_party_category_df,new_row], ignore_index=True)

                elif year not in grand_party_category_df.loc[grand_party_category_df['PartyAb'] == p, f'{chamber}Years'].iloc[0]: # add year if not already there!
                    grand_party_category_df.loc[grand_party_category_df['PartyAb'] == p, f'{chamber}Years'].iloc[0].append(year)



    #import pdb;pdb.set_trace()            

# File will later be modified manually to add Party Ideologies
#grand_party_category_df.to_csv('Grand_Party_Category_df_2004_2022.csv', index = False)
import pdb;pdb.set_trace()

all_parties = pd.read_csv('Grand_Party_Category_df_2004_2022.csv', index_col=None)


def find_house_not_in_previous_senate(House_parties_by_state_dict, Senate_parties_by_state_dict, year, state):

    previous_year = str(int(year) - 3)

    House_set = House_parties_by_state_dict[year][state]
    Senate_set = Senate_parties_by_state_dict[previous_year][state]

    House_set.discard('IND')
    House_set.discard('NAFD')

    house_not_in_senate_set = House_set - Senate_set

    print(year, state, house_not_in_senate_set)


    return house_not_in_senate_set


for year in election_years[1:]:

    house_not_in_senate_set_curr = set()

    for state in states:
        house_not_in_senate_set_curr = house_not_in_senate_set_curr | find_house_not_in_previous_senate(House_parties_by_state_dict, Senate_parties_by_state_dict, year, state)

    



import pdb;pdb.set_trace()



######################################################################################################################################################################
# Use past DOP expansions to estimate shares of each of 5 categories: [Left, ALP, Centre, COAL, Right]

base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)


new_seats_year_dict = {'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner']}
# renaming is not yet updated to before 2016!!!!!!!!
name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'}}
states_to_redistribute_dict = {'2022': ['NSW','VIC','WA','NT'],'2019': ['VIC','WA'],'2016':['ACT','NT','QLD','SA','TAS','VIC'],'2013':['ACT','NSW','WA'],'2010':['SA','VIC'],'2007':['NSW','NT','QLD','TAS','WA'],'2004':['ACT','NSW','QLD'],'2001':['QLD','SA','VIC']}



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

def rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, by_pp_id = False):
    ### apprends div_nm onto any INDXs and changes COAL member parties to COAL or COALNP/COALLP for doubles
    if (DOP_table_wide.columns.isin(COAL_set).sum() == 2) & (div_to_state_dict[div] in ['VIC','NSW']): # Both members of Coalition in div!
        #import pdb;pdb.set_trace()
        for party in DOP_table_wide.columns[1+by_pp_id:]:

            if party.startswith('IND'):
                DOP_table_wide.rename(columns = {party: party + div}, inplace = True) # e.g. IND1Goldstein

            # convert LP and NP to COALLP/COALNP
            if (party=='NP') | (party =='LP'):
                DOP_table_wide.rename(columns = {party: 'COAL' + party}, inplace = True) # rename to COALLP
    
    else:
        for party in DOP_table_wide.columns[1+by_pp_id:]:

            if party.startswith('IND'):
                DOP_table_wide.rename(columns = {party: party + div}, inplace = True) # e.g. IND1Goldstein

            # convert LP and NP in VIC/NSW to COAL
            if div_to_state_dict[div] in ['VIC','NSW']:
                if (party=='NP') | (party =='LP'):
                    DOP_table_wide.rename(columns = {party: 'COAL'}, inplace = True)

    return DOP_table_wide

def create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, DOP_type):
    
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

                    if party.startswith('IND'):
                        Elim_order_list[i] = party + div # e.g. IND1Goldstein

                    # convert LP and NP to COALLP/COALNP
                    if (party=='NP') | (party =='LP'):
                        Elim_order_list[i] = 'COAL' + party # rename to COALLP
            
            else:
                for i, party in enumerate(Elim_order_list):
                    if party.startswith('IND'):
                        Elim_order_list[i] = party + div # e.g. IND1Goldstein

                    # convert LP and NP in VIC/NSW to COAL
                    if div_to_state_dict[div] in ['VIC','NSW']:
                        if (party=='NP') | (party =='LP'):
                            Elim_order_list[i] = 'COAL'

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
            DOP_table_wide = rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, by_pp_id = False)
            

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
            DOP_table_wide = rename_IND_COAL_PartyAbs(div, DOP_table_wide, COAL_set, div_to_state_dict, by_pp_id = False)

            DOP_table_wide_dict[div] = DOP_table_wide
            #import pdb;pdb.set_trace()



    

    return DOP_table_wide_dict


data_year = '2013'

# get state-to-div dict, adjusting for name changes
div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
div_to_state_dict = {name_changes_year_dict[data_year].get(div, div): div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}

# create wide format eliminaation_order_dict
DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)
DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
Div_DOP_dict = {div: group.drop(columns=['div_nm']) for div, group in DOP_By_Division[["div_nm","CountNumber","BallotPosition","cand_id", "PartyAb","CalculationType", "CalculationValue"]].groupby("div_nm")}

Div_DOP_dict = {name_changes_year_dict[data_year].get(key, key): val for key, val in Div_DOP_dict.items()} # adjust for name changes

Elimination_order_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, DOP_type = "EliminationOrder")
DOP_div_expand_dict = create_wide_DOP_dict(Div_DOP_dict, div_to_state_dict, DOP_type = "Expand")

import pdb;pdb.set_trace()


Ideo_Categories = ['Left','ALP','Centre','COAL','Right']
all_parties_house = all_parties.loc[all_parties['Ideo_Category'].notna(),].iloc[:,:2].set_index('PartyAb') # excludes only senates, who don't yet have Ideology written
party_category_dict = all_parties_house.to_dict()['Ideo_Category']
party_category_dict['IND'] = 'Centre'
party_category_dict['COALLP'] = 'COAL'
party_category_dict['COALNP'] = 'COAL'


Left_parties = all_parties.loc[all_parties['Ideo_Category'] == 'Left','PartyAb'].tolist()
ALP_parties = all_parties.loc[all_parties['Ideo_Category'] == 'ALP','PartyAb'].tolist()
Centre_parties = all_parties.loc[all_parties['Ideo_Category'] == 'Centre','PartyAb'].tolist()
COAL_parties = all_parties.loc[all_parties['Ideo_Category'] == 'COAL','PartyAb'].tolist()
Right_parties = all_parties.loc[all_parties['Ideo_Category'] == 'Right','PartyAb'].tolist()

Centre_parties = Centre_parties + ['IND'] # + ['NAFD']??? Or convert all NAFD to IND?


Extra_polled_parties = {'2010':['GRN'],'2013':['GRN','PUP'],'2016':['GRN'],'2019':['GRN','ON','UAPP'],'2022':['GRN','ON','UAPP']}
Polled_parties = COAL_parties + ALP_parties + Extra_polled_parties[data_year]


Ideology_Donation_df = pd.DataFrame(columns = ['Year', 'div_nm','Ideo_Category','Minor_Party','Num_parties'] + Ideo_Categories)     # later add state-by-state rundown

import pdb;pdb.set_trace()

for div in Elimination_order_dict.keys():
    elim_order = Elimination_order_dict[div]
    expand_df = DOP_div_expand_dict[div]

    elim_order = ['IND' if x.startswith('IND') else x for x in elim_order]
    expand_df.columns = expand_df.columns.where(~expand_df.columns.str.startswith("IND"), 'IND')

    for i in reversed(range(3,len(elim_order))): # info not useful for top 3 parties? Or nonsense?
        p = elim_order[i]
        if p not in (Polled_parties + ['IND']):
            

            Cat_p = party_category_dict[p]

            # add expand values to row (average if needed)

            # 1. get correct row of expand df
            expand_row = expand_df.loc[expand_df['CountNumber'] == len(elim_order) - i,]

            new_row = pd.DataFrame([{col: '' if col in ['Year', 'div_nm','Ideo_Category','Minor_Party','Num_parties'] else [] for col in ['Year', 'div_nm','Ideo_Category','Minor_Party','Num_parties'] + Ideo_Categories}])
            new_row['Year'] = data_year
            new_row['div_nm'] = div
            new_row['Ideo_Category'] = Cat_p
            new_row['Num_parties'] = (expand_row.iloc[:,1:] > 0).sum(axis=1).iloc[0] + 1 
            new_row['Minor_Party'] = p

            for don_p in expand_row.columns[1:]:

                expand_prop = expand_row[don_p].iloc[0]

                if expand_prop > 0: # not yet excluded
                    new_row[party_category_dict[don_p]].iloc[0].append(expand_prop)
                    
                #import pdb;pdb.set_trace()

            Ideology_Donation_df = pd.concat([Ideology_Donation_df,new_row], ignore_index=True)
    import pdb;pdb.set_trace()


Ideology_Donation_df.iloc[:, -5:] = Ideology_Donation_df.iloc[:, -5:].applymap(lambda x: sum(x) / len(x) if isinstance(x, list) and x else 0)
import pdb;pdb.set_trace()

### Fix up 2 INDs!!! Aargh! Maybe treat INDS as separates, just adjust whenever indexing from party_cat_dict???