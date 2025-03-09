import pandas as pd
import numpy as np
import os, time
from pathlib import Path
import re

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

data_year = "2022"
previous_data_year = str(int(data_year)-3)




Omnipresent_parties_df_curr = pd.read_csv(f'{data_year}OmnipresentPartiesByPP.csv', index_col=None)
Omnipresent_parties_df_last = pd.read_csv(f'{previous_data_year}OmnipresentPartiesByPP.csv', index_col=None).drop('pp_id', axis=1)

# make them comparable
Omnipresent_parties_df_curr.loc[:,'pp_nm'] = Omnipresent_parties_df_curr.loc[:,'pp_nm'].str.replace(r'\[.*?\]|\(.*?\)','',regex=True).str.strip()
Omnipresent_parties_df_last.loc[:,'pp_nm'] = Omnipresent_parties_df_last.loc[:,'pp_nm'].str.replace(r'\[.*?\]|\(.*?\)','',regex=True).str.strip()

def remove_second_last_word(s):
    return re.sub(r'(\S+)\s+(\S+)\s+PPVC$', r'\1 PPVC', s)

Omnipresent_parties_df_curr.loc[Omnipresent_parties_df_curr['pp_nm'].str.endswith('PPVC'),'pp_nm'] = Omnipresent_parties_df_curr.loc[Omnipresent_parties_df_curr['pp_nm'].str.endswith('PPVC'),'pp_nm'].apply(remove_second_last_word)
Omnipresent_parties_df_last.loc[Omnipresent_parties_df_last['pp_nm'].str.endswith('PPVC'),'pp_nm'] = Omnipresent_parties_df_last.loc[Omnipresent_parties_df_last['pp_nm'].str.endswith('PPVC'),'pp_nm'].apply(remove_second_last_word)


Redistribution_pairs = pd.read_csv(f"RedistributionPairs{str(int(previous_data_year)+2)}.csv", index_col = None).sort_values(by='new_div')

import pdb;pdb.set_trace()


# 1. Try merge on pp_nm and same div_nm
# 2. For each reciever_div, try add on pp_nm and any of reciever's divisions

Omnipresent_parties_merged = Omnipresent_parties_df_curr.merge(Omnipresent_parties_df_last, on=['pp_nm','div_nm'], how='left', suffixes=(f'_{data_year}', f'_{previous_data_year}'))


df2_redistributed = pd.merge(Omnipresent_parties_df_last, Redistribution_pairs, left_on='div_nm', right_on='old_div')

matches = pd.merge(Omnipresent_parties_df_curr, df2_redistributed.drop('div_nm', axis=1), left_on=['div_nm', 'pp_nm'], right_on=['new_div', 'pp_nm'])


matches = matches.loc[matches['pp_nm']!='Other',]

match_counts = matches.groupby(['new_div', 'pp_nm']).size().reset_index(name='count')

# mostly 1s, occasional 2. Only ~400 disscrepancies, goes a good way
matches['pp_nm'].unique()
Omnipresent_parties_merged.loc[(Omnipresent_parties_merged['ALP_2019'].isna()) & (Omnipresent_parties_merged['pp_nm']!='Other'),'pp_nm'].unique()
Omnipresent_parties_merged.loc[Omnipresent_parties_merged['pp_nm'].isin(set(asd) & set(qwe)),]

import pdb;pdb.set_trace()


# see if last election the pp_nm of old_divs are now the names of pps in new_div
for receiver_div in Redistribution_pairs['new_div'].unique():

    curr_pp_nms = Omnipresent_parties_df_curr.loc[(Omnipresent_parties_df_curr['div_nm']==receiver_div) & (Omnipresent_parties_df_curr['pp_nm']!='Other'),'pp_nm'].unique()

    for giver_div in Redistribution_pairs.loc[Redistribution_pairs['new_div']==receiver_div,'old_div'].unique():

        givers_last_pp_nms = Omnipresent_parties_df_last.loc[(Omnipresent_parties_df_last['div_nm']==giver_div) & (Omnipresent_parties_df_last['pp_nm']!='Other'),'pp_nm'].unique()
        common_pp_nms = set(givers_last_pp_nms) & set(curr_pp_nms)

        import pdb;pdb.set_trace()


# create swings and standardise them
pp_swings = []


divs = Omnipresent_parties_df_curr['div_nm'].unique()
Bootstrapping_dict = {new_div: set(pp_swings['standardised_swing']) for new_div in divs}