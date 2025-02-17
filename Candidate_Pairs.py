import pandas as pd
import geopandas as gpd
import numpy as np
import os, time
import ast


os.chdir('C:\\Dania\\2024\\Australian Election')

data_year = '2022'

start = time.time()

DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)

DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
#print(DOP_By_Division)

Div_DOP_dict = {div: group.drop(columns=['div_nm']) for div, group in DOP_By_Division[["div_nm","CountNumber","cand_id", "PartyAb","CalculationType", "CalculationValue"]].groupby("div_nm")}



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

    return pivot_df

def compute_ratio(group):
    ### gets ratio of Transform Count/Preference Count
    count_val = group.loc[group["CalculationType"] == "Preference Count", "CalculationValue"].values[0]
    transfer_val = group.loc[group["CalculationType"] == "Transfer Count", "CalculationValue"].values[0]
    ratio = transfer_val / count_val if count_val > 0 else 0
    return pd.DataFrame({col: group[col].iloc[0] for col in group.columns[:-2]}, index=[0]).assign(CalculationType="Proportion Transferred", CalculationValue=ratio)





def create_wide_DOP_dict(Div_DOP_dict, DOP_type):
    
    DOP_table_wide_dict = {}


    if DOP_type == "VoteCount":
        for div in Div_DOP_dict.keys():

            DOP_table_long = Div_DOP_dict[div].loc[Div_DOP_dict[div]["CalculationType"] == "Preference Count",].reset_index(drop=True)

            # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
            DOP_table_long.loc[:,'PartyAb'] = DOP_table_long['PartyAb'].fillna('IND') 

            # relabel independents in order of ballot appearance if there are multiple
            target = 'IND'
            DOP_table_long.loc[:,'Count'] = (DOP_table_long.groupby('PartyAb').cumcount() + 1)     # Count instances of the target string
            # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
            adjusted_party_names = DOP_table_long.loc[DOP_table_long["CountNumber"] == 0,].apply(
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            )

            num_pref_counts = (DOP_table_long.iloc[-1,0] + 1) # num of final count + original FP count

            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_long.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * num_pref_counts, ignore_index=True).values
            DOP_table_long.loc[DOP_table_long["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!
            #DOP_table_long = DOP_table_long[["CountNumber","PartyAb","CalculationValue"]] # only values needed
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
            DOP_table_wide_dict[div] = DOP_table_wide.astype(int)

    if DOP_type == "EliminationOrder":
        # get state-to-div dict
        div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}

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


            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
            
            # record elimination order
            Elim_order_list_part = DOP_table_wide.iloc[1:,].apply(lambda row: row[row == -100.00].index[0], axis=1).tolist()# Apply the function row-wise to get the column names
            Final_2_Parties = DOP_table_wide.iloc[-1,1:][DOP_table_wide.iloc[-1,] > 0].index.tolist()
            Elim_order_list = Elim_order_list_part + Final_2_Parties

            # give INDs distinct names based on division and convert LP and NP into COAL in Victoria
            for i, party in enumerate(Elim_order_list):
                if party.startswith('IND'):
                    Elim_order_list[i] = party + div # e.g. IND1Goldstein

                # convert LP and NP in VIC/NSW to COAL
                if div_to_state_dict[div] in ['VIC','NSW']:
                    if (party=='NP') | (party =='LP'):
                        Elim_order_list[i] = 'COAL'

            


            DOP_table_wide_dict[div] = Elim_order_list[::-1] # need to still reverse
    
    
    if DOP_type == 'Reduce':
        for div in Div_DOP_dict.keys():
            print(div)
            DOP_table_long = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] > 0) & (Div_DOP_dict[div]["CalculationType"] == "Transfer Percent"),].reset_index(drop=True)

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
            DOP_table_wide_dict[div] = DOP_table_wide.astype(int)





    if DOP_type == 'Expand':
        for div in Div_DOP_dict.keys():
            print(div)

            # get ratio of Transfer Count / Preference Count
            DOP_table_long = Div_DOP_dict[div].groupby(Div_DOP_dict[div].columns[:-2].tolist(), group_keys=False, sort=False).apply(compute_ratio).reset_index(drop=True)
            DOP_table_long = DOP_table_long.loc[DOP_table_long["CountNumber"]>0,]

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

            DOP_table_wide = DOP_table_wide.rename(columns = {"GVIC": "GRN"}) # GVIC issue resolve!
            DOP_table_wide_dict[div] = DOP_table_wide
        
    

    return DOP_table_wide_dict
        






#expand_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "Expand")
#reduce_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "Reduce")
Elimination_order_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "EliminationOrder")
#VoteCount_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "VoteCount")


    
print(time.time() - start, "seconds")
import pdb;pdb.set_trace()


def find_c1_c2(elimination_list, common_set):
    ### finds length of subsection of list upon which all elements of common_set are seen
    
    seen = set()
    for c in range(len(elimination_list)):  # Iterate from the start
        if elimination_list[c] in common_set:
            seen.add(elimination_list[c])  # Track seen elements from the set
        if seen == common_set:  # Stop once all elements have appeared
            return c+1 # inedx+1  
        
    return 0/0 # should never happen!

def redistribution_party_type(div1,div2, Elimination_order_dict, expand_dict, VoteCount_dict):
    # order of div1,div2 important - candidate set change from div1 to div2
    list_div1, list_div2 = Elimination_order_dict[div1], Elimination_order_dict[div2]
    set1,set2 = set(list_div1), set(list_div2)

    common = set1 & set2
    m = len(common)

    if set(list_div1[:m]) == set(list_div2[:m]):
        redistribution_party_type = "simple"
        c1,c2 = len(set1), len(set2) # direct as simple type
        redistributed_votes = simple_redistribution(div1,div2,expand_dict, VoteCount_dict,m,c1,c2)

    else:
        redistribution_party_type = "complex"
        c1 = find_c1_c2(list_div1, common) # length of relevant subset of list_div1
        c2 = find_c1_c2(list_div2, common) # length of relevant subset of list_div2

        import pdb;pdb.set_trace()
        redistributed_votes = complex_redistribution(div1,div2,expand_dict, VoteCount_dict,m,c1,c2,set1,set2)



    return redistributed_votes



#print(redistribution_party_type("Deakin","Aston", Elimination_order_dict, expand_dict, VoteCount_dict))

###### FIND CANDIDATES THAT DO NOT APPEAR IN SENATE & REMOVE
Senate_parties_by_div = pd.read_csv(f"{data_year}Senate_parties_by_div.csv")
Senate_parties_by_div["PartyAbList"] = Senate_parties_by_div["PartyAbList"].apply(ast.literal_eval)

# CURRENTLY IGNORES SEATS WITH BOTH LIBS,NATS BUT ONLY LIBS IN SENATE LIKE IN WESTERN AUSTRALIA


def remove_non_senate_cands_useless(div,list_div,Senate_parties_by_div):

    for party in list_div:
        non_senate_parties = []
        if party not in Senate_parties_by_div[div]:
            non_senate_parties.append(party)
    list_div_in_senate = [x for x in list_div if x not in non_senate_parties]

    return list_div_in_senate # returns parties in elimination list excluding those that are not in the senate



def remove_non_senate_cands(list_div1_c1,list_div2_c2,Senate_parties):
    non_senate_parties = []
    for party in list_div1_c1:
        
        if party not in Senate_parties:
            non_senate_parties.append(party)
    list_div1_FP = [x for x in list_div1_c1 if x not in non_senate_parties]

    non_senate_parties = []
    for party in list_div2_c2:
        if party not in Senate_parties:
            non_senate_parties.append(party)
    list_div2_FP = [x for x in list_div2_c2 if x not in non_senate_parties]

    return list_div1_FP,list_div2_FP # returns parties in elimination lists excluding those that are not in the senate




def get_c1_c2_sets(Redistribution_pairs_df, Elimination_order_dict, Senate_parties_by_div, new_seats_list):
    # input is df with columns corresponding to giver division and receiver division, respectively. Returns the candidate sets of size c1 and c2 for those requiring complex redistr.
    # If independent is involved, this should be dealt with prior to this function.

    # initialise df to input into Formal Preferences Allocation
    columns_list = Redistribution_pairs_df.columns.tolist()
    columns_list.extend(['c1_list','c2_list'])
    Redistribution_pair_complex_c1_c2_lists = pd.DataFrame(columns=Redistribution_pairs_df.columns.tolist())

    redistribution_divs_set = set(Redistribution_pairs_df.iloc[1:,0])|set(Redistribution_pairs_df.iloc[1:,:1]) - {'old_div'} # all relevant divisions
    simplerd, complexrd = 0,0

    # check if double LP/NP contest
    LPNP_double_divs = []
    for div in redistribution_divs_set:
        if Elimination_order_dict[div].count('COAL') == 2:
            LPNP_double_divs.append(div)




    for i in range(Redistribution_pairs_df.shape[0]):
        div1, div2 = Redistribution_pairs_df.iloc[i].tolist() # get giver and receiver for pair i

        # perform different calculations if it double divs are involved
        if (div1 in LPNP_double_divs) or (div2 in LPNP_double_divs):
            print("doubledivs", div1, div2)


        if div2 in new_seats_list:
            print("new seat", div1,div2)
            list_div2 = ['LP','ALP','GRN','UAPP','ON'] ################################### TO DO: generalise to all common parties accounting for COAL or NP or LP for specific election
            list_div1 = Elimination_order_dict[div1]
            
        else:
            list_div1, list_div2 = Elimination_order_dict[div1], Elimination_order_dict[div2] # get elimination orders

        set1,set2 = set(list_div1), set(list_div2)

        common = set1 & set2
        m = len(common)

        # find common candidates, see if simple/complex
        if set(list_div1[:m]) == set(list_div2[:m]):
            redistribution_party_type = "simple"
            print("simple", div1,div2)
            simplerd += 1
        else:
            redistribution_party_type = "non-simple"
            print("not simple", div1,div2)

            c1 = find_c1_c2(list_div1, common) # length of relevant subset of list_div1
            c2 = find_c1_c2(list_div2, common) # length of relevant subset of list_div2

            Senate_parties = Senate_parties_by_div.loc[Senate_parties_by_div['div_nm'] == div1,"PartyAbList"].iloc[0] # both div1 and div2 are in the same state so are identical
            list_div1_FP,list_div2_FP = remove_non_senate_cands(list_div1[:c1],list_div2[:c2],Senate_parties)

            newset1,newset2 = set(list_div1_FP), set(list_div2_FP)
            
            newcommon = newset1 & newset2
            m = len(newcommon)

            if set(list_div1_FP[:m]) == set(list_div2_FP[:m]):
                redistribution_party_type = "now simple"
                print("now simple", div1,div2)
                simplerd += 1
            else:
                print("complex or independent complex",div1,div2) 
                complexrd += 1    

                mlist = [x for x in newcommon]
                list1 = mlist + [x for x in list_div1_FP if x not in newcommon] # order with first m parties, then remaining ci-m parties
                list2 = mlist + [x for x in list_div2_FP if x not in newcommon]

                complex_pair_row = {'old_div': div1, 'new_div': div2, 'c1_list': list1, 'm_list': mlist,'c2_list': list2}
                Redistribution_pair_complex_c1_c2_lists = pd.concat([Redistribution_pair_complex_c1_c2_lists, pd.DataFrame([complex_pair_row])], ignore_index=True)

            

    print(simplerd, complexrd)
    import pdb;pdb.set_trace()

    return Redistribution_pair_complex_c1_c2_lists





def simple_redistribution(div1,div2,expand_dict, VoteCount_dict,m,c1,c2):


    reduced_votes = VoteCount_dict[div1].iloc[c1-m,1:]
    reduced_votes = reduced_votes[reduced_votes>0].sort_index() # sort parties alphabetically
    redistributed_votes = reduced_votes
    for i in reversed(range(c2-m)):
        #         expand_dict[div2].iloc[:c2-m,] s1.sort_index().dot(s2.sort_index())
        expand_div2 = expand_dict[div2].iloc[i,1:]
        common_div2 = expand_div2[expand_div2>0].sort_index()

        lost_votes = round(common_div2*redistributed_votes).astype(int) # alphabetically-sorted 
        redistributed_votes = reduced_votes - lost_votes
        to_expand_party = pd.Series([sum(lost_votes)], index=expand_div2[expand_div2==0].index) # expand_div2 has 0 entry for the party that is being expanded to
        redistributed_votes = pd.concat([redistributed_votes, to_expand_party]).sort_index()
    return redistributed_votes


def complex_redistribution(div1,div2,expand_dict, VoteCount_dict,m,c1,c2,set1,set2):

    reduced_votes = VoteCount_dict[div1].iloc[len(set1)-c1,1:]
    reduced_votes = reduced_votes[reduced_votes>0].sort_index()
    redistributed_votes = reduced_votes

    # Use FP to get senate versions of A) reduced_votes = c1 B) common votes = m C) expanded votes = c2
    # calculate the % shifts in both A>B,B>C
    # apply to reduced_votes --> c2

    for i in reversed(range(len(set2)-c2)):
        expand_div2 = expand_dict[div2].iloc[i,1:]
        common_div2 = expand_div2[expand_div2>0].sort_index()

        lost_votes = round(common_div2*redistributed_votes).astype(int) # alphabetically-sorted 
        redistributed_votes = reduced_votes - lost_votes
        to_expand_party = pd.Series([sum(lost_votes)], index=expand_div2[expand_div2==0].index) # expand_div2 has 0 entry for the party that is being expanded to
        redistributed_votes = pd.concat([redistributed_votes, to_expand_party]).sort_index()

    return redistributed_votes

def independent_redistribution():

    return 1

def complex_independent_redistribution():

    return 1


# reduce dict (key='div_nm') contains transfer percents to each remaining party for each elimination, directly from DOP Transfer Percentage (in wide format)
# expand dict (key='div_nm') contains proportion of party's current vote that was transferred in last count - calculation required (in wide format)
reduce_dict = {}
expand_dict = {}

SA1_By_PP_Votes_2022 = pd.read_csv("2022SA1_By_PP_Votes.csv", index_col=None)

SA1s_with_votes = set(SA1_By_PP_Votes_2022.iloc[:,1])

new_seats_list = ['Bullwinkel']

Redistribution_SA1_changes_2024 = pd.read_csv("Redistribution_SA1_changes2024.csv", index_col=None)
Redistribution_SA1_changes_2024 = Redistribution_SA1_changes_2024.loc[Redistribution_SA1_changes_2024["SA1_CODE21"].isin(SA1s_with_votes),]
Redistribution_SA1_changes_2024_dict = Redistribution_SA1_changes_2024.groupby(['old_div', 'new_div'])['SA1_CODE21'].apply(list).to_dict()
Redistribution_pairs = list(Redistribution_SA1_changes_2024_dict.keys())

Redistribution_pair = {key: [] for key in Redistribution_pairs}

Redistribution_pair_SA1s = Redistribution_SA1_changes_2024.groupby(['old_div', 'new_div'])['SA1_CODE21'].apply(list).reset_index()

recase_map = {'Mcmahon':'McMahon', 'Mcewen':'McEwen','Eden-monaro':'Eden-Monaro',"O'connor": "O'Connor"}

Redistribution_pair_SA1s.iloc[:,:2] = Redistribution_pair_SA1s.iloc[:,:2].replace(recase_map)



Redistribution_pair_c1_c2_lists = get_c1_c2_sets(Redistribution_pair_SA1s.iloc[:,:2], Elimination_order_dict, Senate_parties_by_div, new_seats_list)
Redistribution_pair_c1_c2_lists.to_csv("Redistribution_pair_c1_c2_lists2024.csv", index=False)

import pdb;pdb.set_trace()


# want to get the c1 and c2 for each redistribution pair (unless simple)

# 1. For all pairs, figure if simple, complex, independent (simple needs no further action)
# 2. For all complex/complex independent pairs, figure out c1 and c2
# 3. Store c1 and c2 in a dictionary - to be applied to Formal Preferences!



#for key in Redistribution_pairs:
    
    #redistribution_party_type("Deakin","Aston", Elimination_order_dict, expand_dict, VoteCount_dict)




import pdb;pdb.set_trace()