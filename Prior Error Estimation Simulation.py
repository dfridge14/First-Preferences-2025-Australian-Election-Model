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



base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)




start = time.time()

data_year = '2013' # predicting next year's election
next_year = '2016'
Actual_results = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", index_col = None).rename(columns={'DivisionNm':'div_nm'})
# COUntNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


# Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
# rename CLR to ALP
# rename IND to INDX by order

target = 'IND'

Actual_results_dict = {}
Fundamentals_results_dict = {}
Fundamentals_estimate_dict = {}

def group_into_Fundamentals_Categories(party_votes_shares_df, div):

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'LP','NP','CLP','LNP','LNQ'}
    GRN_cat = {'GRN'}

    Non_Other_sets = ALP_cat | COAL_cat | GRN_cat  # Union of all sets
    Other_cols = set(party_votes_shares_df.columns) - Non_Other_sets  # Columns in none of the sets

    # Compute the sums
    sum1 = party_votes_shares_df[ALP_cat.intersection(party_votes_shares_df.columns)].sum(axis=1).iloc[0]
    sum2 = party_votes_shares_df[COAL_cat.intersection(party_votes_shares_df.columns)].sum(axis=1).iloc[0]
    sum3 = party_votes_shares_df[GRN_cat.intersection(party_votes_shares_df.columns)].sum(axis=1).iloc[0]
    sum4 = party_votes_shares_df[Other_cols].sum(axis=1).iloc[0]

    Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum4}], index=div)

    return Fundamentals_grouped_df



for div in Actual_results['div_nm'].unique():
    div_results = Actual_results.loc[Actual_results['div_nm'] == div,]

    div_results.loc[:,'Count'] = div_results.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
    # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...
    adjusted_party_names = div_results.loc[div_results["CountNumber"] == 0,].apply(
        lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
    ).reset_index(drop=True)

    import pdb;pdb.set_trace()
    Actual_results.loc[Actual_results['div_nm'] == div,'PartyAb'] = adjusted_party_names

    Actual_results_dict[div] = div_results.pivot(index='div_nm', columns='PartyAb', values='CalculationValue')
    Fundamentals_results_dict[div] = group_into_Fundamentals_Categories(Actual_results_dict[div], div)
    Fundamentals_estimate_dict[div] = group_into_Fundamentals_Categories(Prior_Estimates_dict[div], div)



Fundamentals_dict = {}
# GRN, ALP+CLR, COAL+all the others, Other! - sum each category!


for div in 





