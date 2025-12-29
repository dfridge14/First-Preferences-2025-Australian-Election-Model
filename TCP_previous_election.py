import pandas as pd
import numpy as np
import os, time
import glob
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



base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)
from collections import defaultdict



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

party_category_dict = make_party_category_dict()

name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
new_seats_year_dict = {'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner','Gorton']}

replacement_seats_year_dict = {'2022': {'Hasluck':'Bullwinkel'}, '2019':{'Gorton':'Hawke'}, '2016':{'Canberra':'Bean', 'Maribyrnong':'Fraser'}, '2013':{'Hasluck':'Burt'}}
abolished_divs_dict = {'2022':set(['Higgins','North Sydney']), '2016': set(['Port Adelaide']),'2019':set(['Stirling']),'2013':set(['Charlton'])}






TCP_combination_index = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6,  \
                         ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13}



for data_year in ['2013','2016','2019','2022']:

    next_year = str(int(data_year) + 3)
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

    TPP = ('ALP', 'COAL')


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

    tcp_pairs = (TCP_Preference_Flows.groupby("div_nm")["TCP_Ab"].unique().apply(lambda x: tuple(sorted(set(x)))))  # optional: sort so (ALP, LP) and (LP, ALP) match)

    tcp_coalified = tcp_pairs.apply(lambda tup: tuple('COAL' if x in ['LP', 'NP','LNP','CLP'] else x for x in tup))






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

            tcp_pair = tuple(sorted([party_category_dict[p] if party_category_dict[p] != 'Centre' else 'IND' for p in tcp_pair]))

            if tcp_pair == ('ALP', 'COAL'):
                import pdb;pdb.set_trace()
            if div not in Non_classic_divs[tcp_pair]:
                Non_classic_divs[tcp_pair].append(div)



        first, second = tcp_pair  # alphabetical order
        # We want % transferred to the *first* in alphabetical order
        if tcp == first:
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

    #import pdb;pdb.set_trace()


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

                


    import pdb;pdb.set_trace()
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
    #import pdb;pdb.set_trace()

    # Step 3: Pivot to wide format: one row per division, one column per party
    # Step 2: Create the dictionary of wide DataFrames
    result_dict = {}

    # Iterate over the unique divisions
    for div, group in long_df.groupby('division'):
        div_result = pd.DataFrame(index=range(len(TCP_combination_index)), columns=Preference_flows_dict_with_categories[div].keys())
        
        # Fill the DataFrame with NaN (or 0 if preferred) for all TCP pairs initially
        div_result[:] = None  # Or use np.nan if you prefer NaN

        #if division == 'Melbourne':
           # import pdb;pdb.set_trace()


        # Iterate over each unique tcp_pair for this division
        for tcp_pair, tcp_group in group.groupby('tcp_pair'):

            if tcp_pair not in TCP_combination_index.keys():
                # convert to party category:
                
                tcp_pair = tuple(sorted([party_category_dict[p] if p != 'Centre' else 'IND' for p in tcp_pair]))



            # Map tcp_pair to the corresponding row index
            row_index = TCP_combination_index.get(tcp_pair, None)
            
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


    # Outlier - Canberra 2013 House had no Right parties!
    if data_year == '2016':
        for div in ['Bean','Canberra','Fenner']:
            result_dict[div].loc[:,'Right'] = None
            result_dict[div].loc[0,'Right'] = result_dict['Bean'].loc[0,'LDP']


    import pdb;pdb.set_trace()

    # now, find non-classic divisions and extrapolate

    for tcp_pair in Non_classic_divs.keys():

        if tcp_pair == ('COAL','COAL'):
            continue

        # first get average of all results with said tcp:
        i = TCP_combination_index[tcp_pair]

        series_list = []

        for div in Non_classic_divs[tcp_pair]:
            
            series_list.append(result_dict[div].iloc[i]) # curr reusults
        
        tcp_overall = pd.concat(series_list, axis=1)
        tcp_average = tcp_overall.mean(axis=1).dropna()




        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                filtered_update = tcp_average[result_dict[div].columns.intersection(tcp_average.index)]
                result_dict[div].loc[i, filtered_update.index] = filtered_update

    for tcp_pair in [('IND','IND'),('Left','Left'),('LP','NP'),('Right','Right')]:

        i = TCP_combination_index[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                result_dict[div].loc[i, :] = 50.0

    for tcp_pair in [('IND','Right'),('IND','Left')]:

        # use IND-ALP/COAL i.e. 2 or 1

        i = TCP_combination_index[tcp_pair]

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
                        print(div)
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

        i = TCP_combination_index[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                new_row = result_dict[div].loc[0, :].copy()

                # ALP gets Left's 
                # COAL gets Right's 
                new_row.loc['ALP'] = new_row['Left']
                new_row.loc['COAL'] = new_row['Right']

                result_dict[div].loc[i, :] = new_row

    for tcp_pair in [('ALP','Right'),('COAL','Left')]:
        #import pdb;pdb.set_trace()

        # use ALP/COAL

        i = TCP_combination_index[tcp_pair]

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

    # If RIght is None, get corresponding COAL, smae for left and ALP
    Right_COAL_Right_Preferences = 1 - 0.5839 # ON in 2016/9 Maranoa
    Left_COAL_Left_Preferences = 0.305 # 2016/19 Kooyong/Higgins/Melbourne
    Left_ALP_Left_Preferences = 0.34 # 2016/19 ALP/GRN Seats
    Right_COAL_IND_Preferences = 0.398 # for XEN: 2013 Indi/NE preferences
    Right_COAL_Left_Preferences = 0.4 # for 2016, from 2010 Grayndler/Batman


    for div in result_dict.keys():
        print(div)
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


            # fetch all changes so far!
            row = result_dict[div].loc[i]



            if pd.isna(row['Centre']):
                #import pdb;pdb.set_trace()
                #import pdb;pdb.set_trace()
                if (row['Right']) and (row['Left']):
                    

                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['Left'])/2

                    if pd.isna(result_dict[div].at[i, 'Centre']):
                        import pdb;pdb.set_trace()
                        2
                    
                elif (row['Right']) and (row['ALP']):
                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['ALP'])/2

                elif (row['Left']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['Left'])/2

                elif (row['ALP']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['ALP'])/2

                if pd.isna(result_dict[div].at[i, 'Centre']):
                    print(div, i, row)
                    import pdb;pdb.set_trace()
                    3

                    






            

    for division in result_dict.keys():
        print(division)
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
    for div in result_dict.keys():
        print(div)
        result_dict[div] = expand_and_reorder_duplicate(result_dict[div], Div_parties_next_dict[div]) 


    import pdb;pdb.set_trace()




    # TO DO:
    # Impute parties from Right/Left/Centre
    # Missing Right/Left/Centre - Impute from elsewhere!
    # COAL LP/NP - get proper estimates!



