import pandas as pd
import geopandas as gpd
import numpy as np
import os, time


os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()

DOP_By_Division = pd.read_csv("HouseDOPByDivisionDownload-27966.csv", skiprows=1)
print(DOP_By_Division)

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

Elimination_order_dict = {}
for div in Div_DOP_dict.keys():
    print(div)
    FP_pcts = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] == 0) & (Div_DOP_dict[div]["CalculationType"] == "Preference Percent"),]
    Transfer_pcts = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] > 0) & (Div_DOP_dict[div]["CalculationType"] == "Transfer Percent"),]
    DOP_table_long = pd.concat([FP_pcts, Transfer_pcts], ignore_index=True)

    # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
    DOP_table_long['PartyAb'] = DOP_table_long['PartyAb'].fillna('IND') 

    # relabel independents in order of ballot appearance if there are multiple
    target = 'IND'
    DOP_table_long['Count'] = DOP_table_long.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
    # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
    adjusted_party_names = DOP_table_long.loc[DOP_table_long["CountNumber"] == 0,].apply(
        lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
    )
    num_pref_counts = (DOP_table_long.iloc[-1,0] + 1) # num of final count + original FP count

    DOP_table_long['PartyAb'] = pd.concat([adjusted_party_names] * num_pref_counts, ignore_index=True)


    DOP_table_long = DOP_table_long.drop(columns=['Count'])
    DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
    
    Elim_order_list_part = DOP_table_wide.iloc[1:,].apply(lambda row: row[row == -100.00].index[0], axis=1).tolist()# Apply the function row-wise to get the column names
    Final_2_Parties = DOP_table_wide.iloc[-1,1:][DOP_table_wide.iloc[-1,] > 0].index.tolist()
    Elim_order_list = Elim_order_list_part + Final_2_Parties

    Elimination_order_dict[div] = Elim_order_list[::-1] # need to still reverse
    #import pdb;pdb.set_trace()
    
print(time.time() - start, "seconds")
#import pdb;pdb.set_trace()


def redistribution_party_type(div1,div2):
    list_div1, list_div2 = Elimination_order_dict[div1], Elimination_order_dict[div2]
    set1,set2 = set(list_div1), set(list_div2)
    common = set1 & set2
    m = len(common)

    if set(list_div1[:m]) == set(list_div2[:m]):
        redistribution_party_type = "simple"
    else:
        redistribution_party_type = "complex"

    return redistribution_party_type

print(redistribution_party_type("Deakin","Aston"))

import pdb;pdb.set_trace()