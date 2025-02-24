import pandas as pd
import numpy as np
import os, time
import ast


os.chdir('C:\\Dania\\2024\\Australian Election')


Final_x_HS_df2022 = pd.read_csv("2022Final_x_HS_df.csv", index_col = None)

Final_x_Incumbency = pd.read_csv("2022Final_x_for_Incumbency.csv", index_col=None)

merged = Final_x_Incumbency.merge(Final_x_HS_df2022.iloc[:,[0,1,-2,-1]], on=["div_nm","PartyAb","House_Pct"], how='left')
merged.loc[:,"Diff_Pct"] = merged.loc[:,"House_Pct"] - merged.loc[:,"Senate_Pct"]



import pdb;pdb.set_trace()

true_counts = merged.groupby('div_nm')['is_incumbent'].sum()

merged = merged.merge(true_counts, on = 'div_nm',how='left')

# Count how many times each true count appears
grouped_counts = true_counts.value_counts().reset_index()
grouped_counts.columns = ['num_of_True', 'count_of_divs']

import pdb;pdb.set_trace()