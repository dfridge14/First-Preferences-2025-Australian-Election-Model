import pandas as pd
import geopandas as gpd
import numpy as np
import os, time


os.chdir('C:\\Dania\\2024\\Australian Election')

data_year = '2022'

start = time.time()

DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)

DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
print(DOP_By_Division)

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
        for div in Div_DOP_dict.keys():
            print(div)
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
#Elimination_order_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "EliminationOrder")
#VoteCount_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "VoteCount")


    
print(time.time() - start, "seconds")
#import pdb;pdb.set_trace()


def find_c1_c2(elimination_list, common_set):
    ### finds length of subsection of list upon which all elements of common_set are seen
    
    seen = set()
    for c in range(len(elimination_list)):  # Iterate from the start
        if elimination_list[c] in common_set:
            seen.add(elimination_list[c])  # Track seen elements from the set
        if seen == common_set:  # Stop once all elements have appeared
            return c  
        
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

Redistribution_SA1_changes_2024 = pd.read_csv("Redistribution_SA1_changes2024.csv", index_col=None)
Redistribution_SA1_changes_2024 = Redistribution_SA1_changes_2024.loc[Redistribution_SA1_changes_2024["SA1_CODE21"].isin(SA1s_with_votes),]
Redistribution_SA1_changes_2024_dict = Redistribution_SA1_changes_2024.groupby(['old_div', 'new_div'])['SA1_CODE21'].apply(list).to_dict()
Redistribution_pairs = list(Redistribution_SA1_changes_2024_dict.keys())

Redistribution_pair = {key: [] for key in Redistribution_pairs}


# want to get the c1 and c2 for each redistribution pair (unless simple)

# 1. For all pairs, figure if simple, complex, independent (simple needs no further action)
# 2. For all complex/complex independent pairs, figure out c1 and c2
# 3. Store c1 and c2 in a dictionary - to be applied to Formal Preferences!

for key in Redistribution_pairs:
    
    redistribution_party_type("Deakin","Aston", Elimination_order_dict, expand_dict, VoteCount_dict)




import pdb;pdb.set_trace()