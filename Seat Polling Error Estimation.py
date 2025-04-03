import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, dirichlet

from scipy.stats import multivariate_t
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize


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

election_date_num = {'2013':1113, '2016':1028, '2019':1050, '2022':1099}



election_years = ['2013','2016','2019','2022']


def get_results_dict(election_year):

    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})

    # Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
    Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
    Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
    Actual_results.loc[Actual_results['PartyAb']=='GVIC','PartyAb'] = 'GRN'
    Actual_results.loc[Actual_results['PartyAb']=='CLR','PartyAb'] = 'ALP'

    Actual_results_dict = {}

    # rename IND to INDX by order
    target = 'IND'

    for div in Actual_results['div_nm'].unique():
        div_results = Actual_results.loc[Actual_results['div_nm'] == div,].copy()

        div_results.loc[:,'Count'] = div_results.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
        # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...
        adjusted_party_names = div_results.apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
        ).reset_index(drop=True)

        div_results_combined = div_results.groupby(['div_nm', 'PartyAb'], as_index=False)['CalculationValue'].sum()

        Actual_results.loc[Actual_results['div_nm'] == div,'PartyAb'] = adjusted_party_names

        Actual_results_dict[div] = div_results_combined.pivot(index='div_nm', columns='PartyAb', values='CalculationValue')

        Actual_results_dict[div] = Actual_results_dict[div].div(Actual_results_dict[div].sum(axis=1), axis=0)

    import pdb;pdb.set_trace()

    return Actual_results_dict


Seat_poll_year_dict = {}
Actual_results_dict_year = {}

for election_year in election_years:

    Seat_polls = pd.read_csv(f"SeatPolls{election_year}Formatted.csv", index_col=None)
    Seat_polls.loc[:,'Days since last election'] = election_date_num[election_year] - Seat_polls.loc[:,'Days since last election']
    Seat_polls.rename(columns={'Days since last election':'Days before election'}, inplace=True)

    Seat_poll_year_dict[election_year] = Seat_polls

    # get actual results 
    Actual_results_dict = get_results_dict(election_year)
    Actual_results_dict_year[election_year] = Actual_results_dict

import pdb;pdb.set_trace()



# get all COAL-ALP-GRN first i.e. all for whom GRN is not 0 and compare!

for election_year in election_years:
    Seat_poll_year_dict[election_year].loc[Seat_poll_year_dict[election_year]['GRN']>0,]







