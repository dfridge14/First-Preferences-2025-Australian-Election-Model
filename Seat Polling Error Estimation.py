import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

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



def group_into_Fundamentals_Categories(party_votes_shares_df, div, is_Other = True):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'COAL','COALNP','COALLP','LP','NP','CLP','LNP','LNQ'}
    GRN_cat = {'GRN'}

    Non_Other_sets = ALP_cat | COAL_cat | GRN_cat  # Union of all sets
    Other_cols = set(party_votes_shares_df.columns) - Non_Other_sets  # Columns in none of the sets

    ALPs = ALP_cat.intersection(party_votes_shares_df.columns)
    COALs = COAL_cat.intersection(party_votes_shares_df.columns)
    GRNs = GRN_cat.intersection(party_votes_shares_df.columns)
    OTHs = Other_cols

    # Compute the sums
    sum1 = party_votes_shares_df[list(next(iter(ALPs)) if len(ALPs) == 1 and isinstance(next(iter(ALPs)), set) else ALPs)].sum(axis=1).iloc[0]
    sum2 = party_votes_shares_df[list(next(iter(COALs)) if len(COALs) == 1 and isinstance(next(iter(COALs)), set) else COALs)].sum(axis=1).iloc[0]
    if is_Other:
        sum3 = party_votes_shares_df[list(next(iter(GRNs)) if len(GRNs) == 1 and isinstance(next(iter(GRNs)), set) else GRNs)].sum(axis=1).iloc[0]
        sum4 = party_votes_shares_df[list(next(iter(OTHs)) if len(OTHs) == 1 and isinstance(next(iter(OTHs)), set) else OTHs)].sum(axis=1).iloc[0]
    else:
        sum3 = party_votes_shares_df[list(next(iter(GRNs)) if len(GRNs) == 1 and isinstance(next(iter(GRNs)), set) else GRNs) + list(next(iter(OTHs)) if len(OTHs) == 1 and isinstance(next(iter(OTHs)), set) else OTHs)].sum(axis=1).iloc[0]

    if is_Other:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum4}], index=[div])
    else:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'Other':sum3}], index=[div])


    return Fundamentals_grouped_df




def get_results_dict(election_year):

    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})

    # Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
    Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
    Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
    Actual_results.loc[Actual_results['PartyAb']=='GVIC','PartyAb'] = 'GRN'
    Actual_results.loc[Actual_results['PartyAb']=='CLR','PartyAb'] = 'ALP'


    Four_party_results_list = []

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

        Four_party_results_list.append(group_into_Fundamentals_Categories(Actual_results_dict[div], div))

    results_df = pd.concat(Four_party_results_list)
    results_df = results_df.div(results_df.sum(axis=1), axis=0).rename(columns={'Other':'OTH'})

    #import pdb;pdb.set_trace()

    return results_df


Seat_poll_year_dict = {}
Actual_results_dict_year = {}

for election_year in election_years:

    Seat_polls = pd.read_csv(f"SeatPolls{election_year}Formatted.csv", index_col=None)
    Seat_polls.loc[:,'Days since last election'] = election_date_num[election_year] - Seat_polls.loc[:,'Days since last election']
    Seat_polls.rename(columns={'Days since last election':'Days before election'}, inplace=True)

    Seat_poll_year_dict[election_year] = Seat_polls

    # get actual results 
    Actual_results_df = get_results_dict(election_year)
    Actual_results_dict_year[election_year] = Actual_results_df


# get all COAL-ALP-GRN first i.e. all for whom GRN is not 0 and compare!

plt.figure(figsize=(10, 6))
x_axis = 'Sample size'
CAGO_abs_diff_list = []


for election_year in election_years:
    CAGO_poll_df = Seat_poll_year_dict[election_year].loc[Seat_poll_year_dict[election_year]['GRN']>0,]

    if not election_year == '2013': # already formatted in 2013

        # Want to reduce to COAL, ALP, GRN, and all the rest into Other
        info_party_cols = CAGO_poll_df.columns[:3].tolist() + ["COAL", "ALP", "GRN"]
        CAGO_poll_df["OTH"] = CAGO_poll_df.drop(columns=info_party_cols).sum(axis=1)     # Create the 'OTH' column by summing all other columns
        CAGO_poll_df = CAGO_poll_df[info_party_cols + ["OTH"]] # Keeps only COAL, ALP, GRN, OTH

    # Compute absolute differences and store them in poll_df
    for party in ["COAL", "ALP", "GRN", "OTH"]:
        CAGO_poll_df[f"{party}_abs_diff"] = (CAGO_poll_df[party] - Actual_results_dict_year[election_year].loc[CAGO_poll_df["Electorate"]].values[:, Actual_results_dict_year[election_year].columns.get_loc(party)]).abs()

    CAGO_abs_diff_list.append(CAGO_poll_df)

    plot_df = CAGO_poll_df.melt(id_vars=[x_axis], 
                        value_vars=["COAL_abs_diff", "ALP_abs_diff", "GRN_abs_diff", "OTH_abs_diff"], 
                        var_name="PartyAb", 
                        value_name="Abs Difference")

    # Map party names to colors
    party_colors = {"COAL": "blue", "ALP": "red", "GRN": "green", "OTH": "gray"}

    # Extract party names for coloring
    plot_df["Party"] = plot_df["PartyAb"].str.replace("_abs_diff", "")  # Remove suffix
    plot_df["Color"] = plot_df["Party"].map(party_colors)


    # Loop through each party and plot separately
    markers = {'2013':'o','2016':'s','2019':'^','2022':'x'}
    for party, color in party_colors.items():
        subset = plot_df[plot_df["Party"] == party]
        plt.scatter(subset[x_axis], subset["Abs Difference"], label=party, color=color, marker = markers[election_year], s = 10, alpha=0.7)

CAGO_abs_diff = pd.concat(CAGO_abs_diff_list)

plt.xlabel(x_axis)
plt.ylabel("Absolute Difference")
plt.title("Absolute Difference Between Polls and Results")
plt.legend()
plt.show()

import pdb;pdb.set_trace()







