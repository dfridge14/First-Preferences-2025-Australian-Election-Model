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

grand_party_category_df = pd.DataFrame(columns = ['PartyAb','Ideo_Category','HouseYears'])

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

        for p in House_set:
            if p not in grand_party_category_df['PartyAb'].values:
                new_row = pd.DataFrame([{'PartyAb':p,'Ideo_Category':'',"HouseYears":[year]}])
                grand_party_category_df = pd.concat([grand_party_category_df,new_row], ignore_index=True)
            elif year not in grand_party_category_df.loc[grand_party_category_df['PartyAb'] == p, 'HouseYears'].iloc[0]: # add year if not already there!
                grand_party_category_df.loc[grand_party_category_df['PartyAb'] == p, 'HouseYears'].iloc[0].append(year)

    import pdb;pdb.set_trace()            

# grand_party_category_df.to_csv('Grand_Party_Category_df_2004+.csv', index = False)


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



import pdb;pdb.set_trace()








