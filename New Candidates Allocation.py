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

#import pdb;pdb.set_trace()

all_parties = pd.read_csv('Grand_Party_Category_df_2004_2022.csv', index_col=None)
all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['CLR'],'Ideo_Category':['ALP'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['NGS'],'Ideo_Category':['Right'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['ARTS'],'Ideo_Category':['Left'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)


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

    



#import pdb;pdb.set_trace()



######################################################################################################################################################################
# Use past DOP expansions to estimate shares of each of 5 categories: [Left, ALP, Centre, COAL, Right]

base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)


new_seats_year_dict = {'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner','Gorton']}
name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
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

Ideo_Categories = ['Left','ALP','Centre','COAL','Right']




def construct_Ideology_Donation_df(all_parties, Ideo_Categories, IND = False):

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

    Extra_polled_parties = {'2004':['DEM','GRN','HAN'],'2007':['GRN'],'2010':['GRN'],'2013':['GRN','PUP'],'2016':['GRN'],'2019':['GRN','ON','UAPP'],'2022':['GRN','ON','UAPP']}

    id_columns = ['Year', 'div_nm','State', 'Ideo_Category','Minor_Party','Num_parties']

    Ideology_Donation_df = pd.DataFrame(columns = id_columns + Ideo_Categories) 

    for data_year in ['2004','2007','2010','2013','2016','2019','2022']:

        Polled_parties = COAL_parties + ALP_parties + Extra_polled_parties[data_year]


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


        for div in Elimination_order_dict.keys():
            elim_order = Elimination_order_dict[div]
            expand_df = DOP_div_expand_dict[div]

            for i in reversed(range(3,len(elim_order))): # info not useful for top 3 parties? Or nonsense?
                p = elim_order[i]

                # check for either exclusively IND or not IND parties that are not polled
                condition = p.startswith('IND') and (p not in Polled_parties) if IND else (p not in (Polled_parties + ['IND'])) and not (p.startswith('IND'))

                if condition:
                    
                    if p not in party_category_dict.keys():
                        if IND and p.startswith('IND'):
                            p = 'IND'
                        else:
                            print(p,div,elim_order, data_year)
                            continue
                    Cat_p = party_category_dict[p]

                    # add expand values to row (average if needed)

                    # 1. get correct row of expand df
                    expand_row = expand_df.loc[expand_df['CountNumber'] == len(elim_order) - i,]

                    new_row = pd.DataFrame([{col: '' if col in id_columns else [] for col in id_columns + Ideo_Categories}])
                    new_row['Year'] = data_year
                    new_row['div_nm'] = div
                    new_row['State'] = div_to_state_dict[div]
                    new_row['Ideo_Category'] = Cat_p
                    new_row['Num_parties'] = (expand_row.iloc[:,1:] > 0).sum(axis=1).iloc[0] + 1 
                    new_row['Minor_Party'] = p

                    for don_p in expand_row.columns[1:]:

                        expand_prop = expand_row[don_p].iloc[0]

                        condition2 = (expand_prop > 0) if IND else (expand_prop > 0 and not don_p.startswith('IND'))            

                        if condition2: # if IND is False, don't infer from INDs - they come last!

                            if don_p not in party_category_dict.keys():
                                if IND and p.startswith('IND'):
                                    don_p = 'IND'
                                else:
                                    print(don_p,div,elim_order, data_year)
                                    continue

                            new_row[party_category_dict[don_p]].iloc[0].append(expand_prop)
                            
                    Ideology_Donation_df = pd.concat([Ideology_Donation_df,new_row], ignore_index=True)

    Ideology_Donation_df.iloc[:, -5:] = Ideology_Donation_df.iloc[:, -5:].map(lambda x: round(sum(x) / len(x),6) if isinstance(x, list) and x else np.nan)
    import pdb;pdb.set_trace()

    Ideology_Donation_df.to_csv(f'Ideology_Donation{'_IND' if IND else ''}_df.csv', index=False)

    return Ideology_Donation_df


#Ideology_Donation_df = construct_Ideology_Donation_df(all_parties, Ideo_Categories, IND = True)
import pdb;pdb.set_trace()

IND = 1

df = pd.read_csv(f'Ideology_Donation_{'IND_' if IND else ''}df.csv', index_col=None)
Ideology_Donation_df = df.copy()


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


def estimate_donation_proportion(df, new_row,Ideo_Categories):
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

    import pdb;pdb.set_trace()
    # apply weights
    start_weights_index = new_row.columns.get_loc('Right') + 1
    weighted_proportions = new_row.iloc[:,start_weights_index:] * exploded_weight_df
    weighted_proportions.columns = [col[:-3] for col in weighted_proportions.columns]
    estimated_row_proportions = weighted_proportions.T.groupby(weighted_proportions.columns).sum().T[Ideo_Categories] # df of 6 cols, last is ''
    import pdb;pdb.set_trace()
    new_row[Ideo_Categories] = estimated_row_proportions

    return estimated_row_proportions




estimate_whole_df = 1

if estimate_whole_df:

    # add all 8 potential estimates for matching previous year, same div/Num
    for num_match in [1,0]:
        for div_match in [1,0]:
            for year_prev in [1,0]:
                X_num = f'_X{1 + 4*(1-num_match) + 2*(1 - div_match) + (1 - year_prev)}'  # label them as X_1...X_8 according to groupings

                curr_match = df.apply(lambda row: get_matching_average(row, df, Ideo_Categories, year_prev, div_match, num_match), axis=1)
                curr_match[X_num] = curr_match.count(axis=1)

                df = df.join(curr_match, rsuffix = f'{X_num}')

    X_cols = ['_X1','_X2','_X3','_X4','_X5','_X6','_X7','_X8']
    X_weights = [1,0.5,0.5,0.25,0.125,0.0625,0.0625,0.03125] # ad-hoc weights, improve with model in future
    weight_df_ids = (df[X_cols]>0)*X_weights
    weight_df = weight_df_ids.div(weight_df_ids.sum(axis=1), axis=0)

    # expand wegiht df to each set of 6 columns
    exploded_weight_dfs = []
    for col in weight_df.columns:
        exploded_weight_dfs.append(explode_column(weight_df, col, Ideo_Categories))
    exploded_weight_df = pd.concat(exploded_weight_dfs, axis=1)

    # apply weights
    weighted_proportions = df.iloc[:,11:] * exploded_weight_df
    weighted_proportions.columns = [col[:-3] for col in weighted_proportions.columns]
    final_proportions = weighted_proportions.T.groupby(weighted_proportions.columns).sum().T[Ideo_Categories] # df of 6 cols, last is ''

    import pdb;pdb.set_trace()

    true_proportions = Ideology_Donation_df.copy()
    true_proportions.iloc[:,-5:] -= final_proportions
    true_proportions.iloc[:,-5:].std()
    true_proportions.iloc[:,-5:].mean()

    Ideology_Donation_Estimate_df = Ideology_Donation_df.join(final_proportions, rsuffix = '_est')
    Ideology_Donation_Estimate_df.to_csv('Ideology_Donation_Estimate_df.csv', index=False)

else:
    new_row = df[['Year','div_nm','State','Ideo_Category','Num_parties'] + Ideo_Categories].iloc[2551:2552,]
    new_row[Ideo_Categories] = np.nan

    estimated_row = estimate_donation_proportion(df, new_row, Ideo_Categories)



import pdb;pdb.set_trace()






# develop weighting by year, div, state, num cands
# 1 to same div last year; 0.5 to same div other year OR same state, last year