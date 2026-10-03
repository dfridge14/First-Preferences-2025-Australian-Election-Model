import pandas as pd
import numpy as np
import os, time
import glob
from pathlib import Path


# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler



base_dir = Path.home() / "Australian Election"
os.chdir(base_dir)

# obtain house results by swing
data_year = '2025'
prev_year = str(int(data_year)-3)

Incumbents_prev = pd.read_csv(f"{prev_year}Incumbents.csv")
Incumbency_curr = pd.read_csv(f"{data_year}Incumbents.csv")
Incumbents_prev['data_year'] = prev_year
Incumbency_curr['data_year'] = data_year

TPP_results = pd.read_csv("2025HouseTppByDivision.csv", skiprows = 1)

# exclude IND/GRN incumbents! ish


Incumbency_curr = Incumbency_curr.loc[~Incumbency_curr['div_nm'].isin(['Maribyrnong','Hinkler']),]

import pdb; pdb.set_trace()

mult_inc = Incumbency_curr['div_nm'].value_counts()
mult_inc = mult_inc[mult_inc>1].index
#if mult_inc>1:
#    import pdb; pdb.set_trace() # cowan

# evaluate incumbency delta: was but is no longer. 

# -1: Labor retiring

merged_df = pd.merge(Incumbency_curr, Incumbents_prev, on = 'div_nm',how='outer', suffixes=('', '_prev'))
merged_df = merged_df.loc[~merged_df['div_nm'].isin(['Higgins','North Sydney']),]
ALP_retiring = merged_df.loc[(merged_df['PartyAb_prev'] == 'ALP') & (merged_df['PartyAb'].isna()), ] # (merged_df['elections_won'] != merged_df['elections_won_prev'] + 1),
ALP_to_Oth = merged_df.loc[(merged_df['PartyAb_prev'] == 'ALP') & ~(merged_df['PartyAb'].isin(['LP','CLP','LNP','NP','ALP']) ) & ~(merged_df['PartyAb'].isna()), ]
COAL_retiring = merged_df.loc[(merged_df['PartyAb_prev'].isin(['LP','CLP','LNP','NP'])) & (merged_df['PartyAb'].isna()), ]
COAL_to_Oth = merged_df.loc[(merged_df['PartyAb_prev'].isin(['LP','CLP','LNP','NP'])) & ~(merged_df['PartyAb'].isin(['LP','CLP','LNP','NP','ALP'])) & ~(merged_df['PartyAb'].isna()), ]
COAL_to_ALP =  merged_df.loc[(merged_df['PartyAb_prev'].isin(['LP','CLP','LNP','NP'])) & (merged_df['PartyAb'] == 'ALP'), ]
ALP_gained_inc = merged_df.loc[(merged_df['PartyAb_prev'].isna() | ~merged_df['PartyAb_prev'].isin(['LP','CLP','LNP','NP','ALP'])) & (merged_df['PartyAb'] == 'ALP'),]
COAL_with_inc = merged_df.loc[(merged_df['PartyAb_prev'].isna() | ~merged_df['PartyAb_prev'].isin(['LP','CLP','LNP','NP','ALP'])) & (merged_df['PartyAb'].isin(['LP','CLP','LNP','NP'])),]
Remaining_inc = merged_df.loc[(merged_df['elections_won'] == merged_df['elections_won_prev'] + 1) & (merged_df['PartyAb_prev'] == merged_df['PartyAb']), ]
Byelec_COAL = merged_df.loc[(merged_df['elections_won'] != merged_df['elections_won_prev'] + 1) & (merged_df['PartyAb_prev'] == merged_df['PartyAb']) & (merged_df['PartyAb'].isin(['LP','CLP','LNP','NP'])), ]
Byelec_ALP = merged_df.loc[(merged_df['elections_won'] != merged_df['elections_won_prev'] + 1) & (merged_df['PartyAb_prev'] == merged_df['PartyAb']) & (merged_df['PartyAb'] == 'ALP'), ]

#COAL_to_COAL
#ALP_to_ALP = # 2PP swing vs national average

# -1: ALP -> None/Oth, None/Oth -> COAL
# 0: None/Oth -> None/Oth, ALP -> ALP, COAL-> COAL
# 1: None/Oth -> ALP, COAL -> None/Oth
# 2: COAL -> ALP

div_list = TPP_results['DivisionNm'].to_list()

#for div in div_list:
#    if div not in Incumbency_curr['div_nm']:
#        Incumbency_curr = pd.concat([Incumbency_curr, ])


Inc_shift_dict = {}

for div in COAL_to_ALP['div_nm'].tolist():
    Inc_shift_dict[div] = 2
for div in (ALP_retiring['div_nm'].tolist() + ALP_to_Oth['div_nm'].tolist() + COAL_with_inc['div_nm'].tolist()):
    Inc_shift_dict[div] = -1
for div in( COAL_retiring['div_nm'].tolist() + COAL_to_Oth['div_nm'].tolist() + ALP_gained_inc['div_nm'].tolist()):
    Inc_shift_dict[div] = 1
for div in Remaining_inc['div_nm'].tolist() + Byelec_COAL['div_nm'].tolist() + Byelec_ALP['div_nm'].tolist() + ['Bullwinkel', 'Fowler']:
    Inc_shift_dict[div] = 0


missing_divs = [p for p in div_list if p not in Inc_shift_dict.keys()]
merged_df.loc[merged_df['div_nm'].isin(missing_divs),]



TPP_swing = TPP_results[['DivisionNm','Swing']].rename(columns={'DivisionNm': 'div_nm'})

TPP_swing.loc[TPP_swing['div_nm'] == 'Bendigo','Swing'] = -9.9
TPP_swing.loc[TPP_swing['div_nm'] == 'Brisbane','Swing'] = 4.56
TPP_swing.loc[TPP_swing['div_nm'] == 'Nicholls','Swing'] = -1.87



TPP_swing['Inc_shift_cat'] = TPP_swing['div_nm'].map(Inc_shift_dict)

from collections import Counter
Counter(TPP_swing['Inc_shift_cat'])
TPP_swing.groupby('Inc_shift_cat')['Swing'].mean()

import pdb; pdb.set_trace()

# additional column based on value
pd.concat([])