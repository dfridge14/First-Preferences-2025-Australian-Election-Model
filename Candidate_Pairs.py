import pandas as pd
import numpy as np
import os, time
import ast
from pathlib import Path


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

INCUMBENT_ADVANTAGE = 4
final_cand_no_dict = {"2022":5, "2019": 4, "2016": 4,"2013": 5, "2010": 3, "2007": 4, "2004": 4,"2001":4}
name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}


data_year = '2007'
FINAL_CANDIDATE_NO = final_cand_no_dict[data_year]
NONINCUMBENT_DISADVANTAGE =  INCUMBENT_ADVANTAGE/(FINAL_CANDIDATE_NO-1)





start = time.time()

DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)
DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
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

    if df_type == "DOP_By_PP":

        pivot_df = df.pivot_table(index=['pp_id','CountNumber'], # double index for info across pp_ids
                                columns=['PartyAb'], 
                                values='CalculationValue', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()

    return pivot_df

def compute_ratio(group):
    ### gets ratio of Transform Count/Preference Count
    import pdb;pdb.set_trace()

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
            DOP_table_long.loc[DOP_table_long["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!


            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
            
            # record elimination order
            Elim_order_list_part = DOP_table_wide.iloc[1:,].apply(lambda row: row[row == -100.00].index[0], axis=1).tolist()# Apply the function row-wise to get the column names
            Final_2_Parties = DOP_table_wide.iloc[-1,1:][DOP_table_wide.iloc[-1,] > 0].index.tolist()
            Elim_order_list = Elim_order_list_part + Final_2_Parties

            # give INDs distinct names based on division and convert LP and NP into COAL in Victoria/NSW
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
#Elimination_order_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "EliminationOrder")
#VoteCount_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "VoteCount")



def create_DOP_By_PP_csvs(data_year, name_changes_year_dict):
    DOP_By_PP_year = pd.read_csv(f"{data_year}DOP_By_PP_full.csv", index_col=None).rename(columns={"DivisionNm": 'div_nm', 'PPId': 'pp_id', 'PPNm': 'pp_nm','CountNum': 'CountNumber',"CandidateId":"cand_id"})
    
    DOP_By_PP_year['div_nm'] = DOP_By_PP_year['div_nm'].replace(name_changes_year_dict[data_year]) # adjust for name changes


    DOP_By_PP_year  = DOP_By_PP_year[["div_nm","pp_id","pp_nm","CountNumber","BallotPosition","cand_id", "PartyAb","CalculationType", "CalculationValue"]]

    DOP_By_PP_year = DOP_By_PP_year.loc[~(DOP_By_PP_year['CalculationType'] == 'Transfer Percent'),]
    DOP_By_PP_year.loc[:,'PartyAb'] = DOP_By_PP_year.loc[:,'PartyAb'].fillna('IND') # fix Steve Khouw issue

    # convert relevant pp_nms to pp_id 0

    # combine all the 'Other' ids - deal with %s carefully
    Other_booth_type_prefixes = ['Remote Mobile', 'Other Mobile','Special Hospital','EAV','ABSENT','PROVISIONAL','PRE_POLL','POSTAL']

    Zero_ids = DOP_By_PP_year.loc[DOP_By_PP_year.loc[:,"pp_nm"].astype(str).str.startswith(tuple(Other_booth_type_prefixes)),]
    Zero_ids.loc[:,'pp_id'] = 0
    Zero_ids.loc[:,'pp_nm'] = 'Other'

    grouped_zero_ids = Zero_ids.groupby(["div_nm","pp_id","pp_nm","CountNumber","BallotPosition","cand_id", "PartyAb","CalculationType"], as_index=False).agg("sum") # sum accurate for counts, fatal for %
    # fix Percent values

    df_counts = grouped_zero_ids.loc[grouped_zero_ids["CalculationType"]=='Preference Count',].copy()
    df_totals = df_counts.groupby(['div_nm', 'pp_id', 'CountNumber'], as_index=False)['CalculationValue'].sum() # sum of all counts
    df_totals.rename(columns={'CalculationValue': 'TotalCount'}, inplace=True)
    df_counts = df_counts.merge(df_totals, on=['div_nm', 'pp_id', 'CountNumber'])
    df_counts['CalculationValue'] = round(df_counts['CalculationValue'] / df_counts['TotalCount'] * 100,2)  # Convert to percentage
    df_counts['CalculationType'] = df_counts['CalculationType'].replace({
        'Preference Count': 'Preference Percent',
    })
    df_counts.drop(columns=['TotalCount'], inplace=True)

    # replace with correct percentages
    idx = grouped_zero_ids[grouped_zero_ids["CalculationType"] == "Preference Percent"].index
    grouped_zero_ids.loc[idx, :] = df_counts.values

    

    # Sort to match original structure

    DOP_By_PP_year_formatted = DOP_By_PP_year.loc[~(DOP_By_PP_year.loc[:,"pp_nm"].astype(str).str.startswith(tuple(Other_booth_type_prefixes))),]
    DOP_By_PP_year_formatted = pd.concat([DOP_By_PP_year_formatted, grouped_zero_ids], ignore_index=True)
    DOP_By_PP_year_formatted.sort_values(by=["div_nm", "pp_id", "CountNumber", "BallotPosition"], inplace=True)
    

    #DOP_By_PP_year_formatted = DOP_By_PP_year_formatted.sort_values(["div_nm", "pp_id"], kind="stable") # sort: group all div_nm and pp_nm together

    # calculate proportions using pivot into df for Expanding!
    pivot_df = DOP_By_PP_year_formatted.pivot(index=[col for col in DOP_By_PP_year_formatted.columns if col not in ["CalculationType", "CalculationValue"]], 
                     columns="CalculationType", values="CalculationValue")
    pivot_df["Proportion Transferred"] = pivot_df["Transfer Count"] / pivot_df["Preference Count"]     # Compute ratio directly using vectorized operations
    pivot_df["Proportion Transferred"] = pivot_df["Proportion Transferred"].fillna(0)
    pivot_df["Proportion Transferred"] = pivot_df["Proportion Transferred"].replace([-np.inf], -1) # -1 corresponds to candidate donating their votes


    pivot_df = pivot_df.drop(["Preference Count",'Preference Percent','Transfer Count'], axis=1)
    pivot_df.rename(columns = {"Proportion Transferred":'CalculationValue'}, inplace=True)
    DOP_By_PP_Expand = pivot_df.reset_index()
    DOP_By_PP_Expand.columns.name = None # reset columns index name from 'CalculationType'
    DOP_By_PP_Expand = DOP_By_PP_Expand.drop("BallotPosition", axis = 1)

    DOP_By_PP_Pref_Percent = DOP_By_PP_year_formatted.loc[DOP_By_PP_year_formatted["CalculationType"]=="Preference Percent",].drop(["CalculationType","BallotPosition"], axis=1)
    import pdb;pdb.set_trace()

    DOP_By_PP_Pref_Percent.to_csv(f"{data_year}DOP_By_PP_Pref_Percent.csv", index=False)
    DOP_By_PP_Expand.to_csv(f"{data_year}DOP_By_PP_Expand.csv", index=False)

    return 1

create_DOP_By_PP_csvs(data_year, name_changes_year_dict) # create csv file!

import pdb;pdb.set_trace()

# else, read the existing csv file
DOP_By_PP_Expand = pd.read_csv("2022DOP_By_PP_Expand.csv", index_col=None)
DOP_By_PP_Pref_Percent = pd.read_csv("2022DOP_By_PP_Pref_Percent.csv", index_col=None)


def convert_long_to_wide_format(DOP_table_long):
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

        import pdb;pdb.set_trace()


        DOP_table_wide = convert_to_wide_format(group, "DOP_By_PP")
        DOP_By_PP_dict[div] = DOP_table_wide

    return DOP_By_PP_dict

# create wide format dicts
DOP_By_PP_Pref_Percent_wide_dict = convert_long_to_wide_format(DOP_By_PP_Pref_Percent)
DOP_By_PP_Expand_wide_dict = convert_long_to_wide_format(DOP_By_PP_Expand)

print(time.time() - start, "seconds")

Incumbency_by_div = pd.read_csv(f"{data_year}Incumbents.csv", index_col = None)

def independent_to_c1(div1, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO):
    ### adjusts votes of c1 candidate for incumbency advantage
    # 1. get top 5 cands, 2. reverse incumb.advantage, 3. expand to full vote

    wide_df1 = DOP_By_PP_Pref_Percent_wide_dict[div1]
    wide_df_expand = DOP_By_PP_Expand_wide_dict[div1] # same div here!

    # Reduce to Final 5
    Final_Count_Number = wide_df1.iloc[-1,1] # last index of CountNumber (2nd column)
    reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == Final_Count_Number - (FINAL_CANDIDATE_NO-2),] # when 5 candidates remaining
    reduced_votes_by_PP = reduced_votes_by_PP.drop(columns=[reduced_votes_by_PP.columns[1]]) # remove CountNumber col!
    reduced_votes_by_PP.iloc[:, 1:] = reduced_votes_by_PP.iloc[:, 1:].where(reduced_votes_by_PP.iloc[:, 1:] > 0)
    reduced_votes_by_PP = reduced_votes_by_PP.sort_index()

    top_5_columns = reduced_votes_by_PP.columns[~reduced_votes_by_PP.iloc[0].isna()].tolist()[1:]

    # Reverse incumbency advantage: e.g. -4 to inc, +1 to each non-inc 
    Incumbent_in_div = Incumbency_by_div.loc[Incumbency_by_div['div_nm']==div1,'PartyAb'].tolist()

    for party in Incumbent_in_div:
        reduced_votes_by_PP[top_5_columns] += NONINCUMBENT_DISADVANTAGE
        reduced_votes_by_PP[party] -= (INCUMBENT_ADVANTAGE + NONINCUMBENT_DISADVANTAGE)


    # expand to c1
    expanded_votes = reduced_votes_by_PP

    start_range = 1 + (Final_Count_Number + 2) - c1 # 1 + total num of candidates - c1
    end_range = 1 + Final_Count_Number - (FINAL_CANDIDATE_NO-2)

    for i in reversed(range(start_range, end_range)): #(i.e. from count 4 to count 1, where the difference 4-1=c1-m)
        expand_div = wide_df_expand.loc[wide_df_expand['CountNumber'] == i,].drop('CountNumber', axis = 1)
    
        to_expand_party = expand_div.columns[(expand_div.iloc[0] == -1)].tolist()[0] # party to expand to will have -1 as value

        lost_votes = expanded_votes.set_index("pp_id").multiply(expand_div.set_index("pp_id")).reset_index()

        expanded_votes = expanded_votes.set_index("pp_id").subtract(lost_votes.set_index("pp_id")).reset_index()
        expanded_votes[to_expand_party] = lost_votes.iloc[:,1:].sum(axis=1).values
        #import pdb;pdb.set_trace()

    return expanded_votes

expanded_votes = independent_to_c1("Goldstein",DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, 7, Incumbency_by_div, FINAL_CANDIDATE_NO)
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



def redistribution_votes_by_PP(div1,div2, Elimination_order_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict):
    # order of div1,div2 important - candidate set change from div1 to div2
    list_div1, list_div2 = Elimination_order_dict[div1], Elimination_order_dict[div2]
    set1,set2 = set(list_div1), set(list_div2)

    common = set1 & set2
    m = len(common)

    if set(list_div1[:m]) == set(list_div2[:m]):
        redistribution_party_type = "simple"
        c1,c2 = len(set1), len(set2) # direct as simple type
        redistributed_votes = simple_redistribution(div1,div2,DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict,m,c1,c2)

    else:
        redistribution_party_type = "complex"
        c1 = find_c1_c2(list_div1, common) # length of relevant subset of list_div1
        c2 = find_c1_c2(list_div2, common) # length of relevant subset of list_div2

        import pdb;pdb.set_trace()
        redistributed_votes = complex_redistribution(div1,div2,DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict,m,c1,c2,set1,set2)



    return redistributed_votes








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





def simple_redistribution(div1,div2,DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict,m,c1,c2):

    wide_df1 = DOP_By_PP_Pref_Percent_wide_dict[div1]
    wide_df2 = DOP_By_PP_Expand_wide_dict[div2]

    import pdb;pdb.set_trace()

    reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == c1-m,1:] # c1-m matches the count number!
    reduced_votes_by_PP = reduced_votes_by_PP.drop(columns=[reduced_votes_by_PP.columns[1]]) # remove CountNumber col!
    reduced_votes_by_PP.iloc[:, 1:] = reduced_votes_by_PP.iloc[:, 1:].where(reduced_votes_by_PP.iloc[:, 1:] > 0)
    expanded_votes = reduced_votes_by_PP.sort_index()
    
    for i in reversed(range(c2-m)):
        expand_div2 = wide_df2.loc[wide_df2['CountNumber'] == i,].drop('CountNumber', axis = 1)

        to_expand_party = expand_div2.columns[(expand_div2.iloc[0] == -1)].tolist()[0] # party to expand to will have -1 as value

        lost_votes = expanded_votes.set_index("pp_id").multiply(expand_div2.set_index("pp_id")).reset_index()

        expanded_votes = expanded_votes.set_index("pp_id").subtract(lost_votes.set_index("pp_id")).reset_index()
        expanded_votes[to_expand_party] = lost_votes.iloc[:,1:].sum(axis=1).values


    redistributed_votes = expanded_votes

    return redistributed_votes






def reduce_candidates_to_set_size(div, DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size):
    wide_df1 = DOP_By_PP_Pref_Percent_wide_dict[div]

    import pdb;pdb.set_trace()

    Final_Count_Number = wide_df1.iloc[-1,1] # last index of CountNumber (2nd column)
    reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == (Final_Count_Number+2)-reduced_c_size,1:] # the correct count number!
    reduced_votes_by_PP = reduced_votes_by_PP.drop(columns=[reduced_votes_by_PP.columns[1]]) # remove CountNumber col!
    reduced_votes_by_PP.iloc[:, 1:] = reduced_votes_by_PP.iloc[:, 1:].where(reduced_votes_by_PP.iloc[:, 1:] > 0) # convert other cols to nan
    reduced_votes_by_PP = reduced_votes_by_PP.sort_index() # maybe redundant

    return reduced_votes_by_PP




def expand_candidates_to_set_size(div, reduced_votes_by_PP, DOP_By_PP_Expand_wide_dict, c_size, expanded_c_size):
    
    wide_df_expand = DOP_By_PP_Expand_wide_dict[div]

    expanded_votes = reduced_votes_by_PP

    Final_Count_Number = wide_df_expand.iloc[-1,1]
    start_range = 1 + (Final_Count_Number + 2) - expanded_c_size # 1 + total num of candidates - c1
    end_range = 1 + (Final_Count_Number+2) - c_size

    for i in reversed(range(start_range, end_range)): #(i.e. from count 4 to count 1, where the difference 4-1=c1-m)
        expand_div = wide_df_expand.loc[wide_df_expand['CountNumber'] == i,].drop('CountNumber', axis = 1)
    
        to_expand_party = expand_div.columns[(expand_div.iloc[0] == -1)].tolist()[0] # party to expand to will have -1 as value

        lost_votes = expanded_votes.set_index("pp_id").multiply(expand_div.set_index("pp_id")).reset_index()

        expanded_votes = expanded_votes.set_index("pp_id").subtract(lost_votes.set_index("pp_id")).reset_index()
        expanded_votes[to_expand_party] = lost_votes.iloc[:,1:].sum(axis=1).values
        #import pdb;pdb.set_trace()

    return expanded_votes

def complex_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict, DOP_By_PP_Expand_wide_dict,m,c1,c2,set1,set2):

    
    reduced_votes_by_PP = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1) # want to reduce div1 to c1 candidates
    
    # apply First Preferences allocation

    # Use FP to get senate versions of A) reduced_votes = c1 B) common votes = m C) expanded votes = c2
    # calculate the % shifts in both A>B,B>C
    # apply to reduced_votes --> c2

    
    redistributed_votes = expand_candidates_to_set_size(div2, reduced_votes_by_PP, DOP_By_PP_Expand_wide_dict, c2, expanded_c_size)




    return redistributed_votes



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


Redistribution_pair_SA1s = Redistribution_SA1_changes_2024.groupby(['old_div', 'new_div'])['SA1_CODE21'].apply(list).reset_index()

# this has now been corrected when constructing Redistribution_SA1_changes_2024
#recase_map = {'Mcmahon':'McMahon', 'Mcewen':'McEwen','Eden-monaro':'Eden-Monaro',"O'connor": "O'Connor"}

#Redistribution_pair_SA1s.iloc[:,:2] = Redistribution_pair_SA1s.iloc[:,:2].replace(recase_map)

Redistribution_pairs = Redistribution_pair_SA1s.iloc[:,:2]
Redistribution_pairs.to_csv("RedistributionPairs2024.csv", index = False)



Redistribution_pair_c1_c2_lists = get_c1_c2_sets(Redistribution_pairs, Elimination_order_dict, Senate_parties_by_div, new_seats_list)
Redistribution_pair_c1_c2_lists.to_csv("Redistribution_pair_c1_c2_lists2024.csv", index=False)

import pdb;pdb.set_trace()


# want to get the c1 and c2 for each redistribution pair (unless simple)

# 1. For all pairs, figure if simple, complex, independent (simple needs no further action)
# 2. For all complex/complex independent pairs, figure out c1 and c2
# 3. Store c1 and c2 in a dictionary - to be applied to Formal Preferences!



#for key in Redistribution_pairs:
    
    #redistribution_party_type("Deakin","Aston", Elimination_order_dict, expand_dict, VoteCount_dict)




import pdb;pdb.set_trace()