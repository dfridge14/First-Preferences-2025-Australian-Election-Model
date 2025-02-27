import pandas as pd
import numpy as np
import os, time
import ast


os.chdir('C:\\Dania\\2024\\Australian Election')

x = 5

Final_x_HS_df2022 = pd.read_csv(f"2022Final_{x}_HS_df.csv", index_col = None)
Final_x_HS_df2019 = pd.read_csv(f"2019Final_{x}_HS_df.csv", index_col = None)
Final_x_HS_df2016 = pd.read_csv(f"2016Final_{x}_HS_df.csv", index_col = None)

# add year suffixes to differentiate division observation years
Final_x_HS_df2022.loc[:,'div_nm'] = Final_x_HS_df2022.loc[:,'div_nm'] + '22'
Final_x_HS_df2019.loc[:,'div_nm'] = Final_x_HS_df2019.loc[:,'div_nm'] + '19'
Final_x_HS_df2016.loc[:,'div_nm'] = Final_x_HS_df2016.loc[:,'div_nm'] + '16'

combined_HS_df = pd.concat([Final_x_HS_df2022, Final_x_HS_df2019,Final_x_HS_df2016], axis = 0)

# count how many incumbents there are 
incumbent_counts = combined_HS_df.groupby('div_nm')['is_incumbent'].sum().rename('incumbents_in_div')
hist_incumbent_counts = combined_HS_df.groupby('div_nm')['is_historic_incumbent'].sum().rename('hist_incumbents_in_div')

combined_HS_df = combined_HS_df.merge(incumbent_counts, on = 'div_nm',how='left')
combined_HS_df = combined_HS_df.merge(hist_incumbent_counts, on = 'div_nm',how='left')
combined_HS_df.loc[:,"Diff_Pct"] = combined_HS_df.loc[:,"House_Pct"] - combined_HS_df.loc[:,"Senate_Pct"]



# PARTY EFFECTS FOR NON-INCUMBENTS
combined_HS_df.loc[(combined_HS_df["is_historic_incumbent"]!= 1) & (combined_HS_df["incumbents_in_div"]==1) & (combined_HS_df["is_incumbent"] == 0),].groupby("PartyAb")['Diff_Pct'].mean()
# mean and var for non-incumbents
combined_HS_df.loc[(combined_HS_df["is_historic_incumbent"]!= 1)& (combined_HS_df["incumbents_in_div"]==1) & (combined_HS_df["is_incumbent"] == 0),"Diff_Pct"].mean() 
# individual years
combined_HS_df.loc[(combined_HS_df["is_historic_incumbent"]!= 1)& (combined_HS_df["incumbents_in_div"]==1) & (combined_HS_df["is_incumbent"] == 1) & (combined_HS_df["div_nm"].str.endswith('16')),"Diff_Pct"].mean()









import pdb;pdb.set_trace()

Final_x_Incumbency = pd.read_csv("2022Final_x_for_Incumbency.csv", index_col=None)


# INCONSISTENCIES IN ORIGINAL DESCRIPTION, SO WILL USE TEMPORARY MERGED FILE
merged = Final_x_Incumbency.merge(Final_x_HS_df2022.iloc[:,[0,1,-2,-1]], on=["div_nm","PartyAb","House_Pct"], how='left')
merged.loc[:,"Diff_Pct"] = merged.loc[:,"House_Pct"] - merged.loc[:,"Senate_Pct"]




true_counts = merged.groupby('div_nm')['is_incumbent'].sum()

merged = merged.merge(true_counts, on = 'div_nm',how='left')




# Count how many times each true count appears
grouped_counts = true_counts.value_counts().reset_index()
grouped_counts.columns = ['num_of_True', 'count_of_divs']

import pdb;pdb.set_trace()