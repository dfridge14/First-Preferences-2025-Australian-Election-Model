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

for div in Actual_results['div_nm'].unique():
    div_results = Actual_results.loc[Actual_results['div_nm'] == div,]

    div_results.loc[:,'Count'] = div_results.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
    # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...
    adjusted_party_names = div_results.loc[div_results["CountNumber"] == 0,].apply(
        lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
    ).reset_index(drop=True)

    import pdb;pdb.set_trace()
    Actual_results.loc[Actual_results['div_nm'] == div,'PartyAb'] = adjusted_party_names



