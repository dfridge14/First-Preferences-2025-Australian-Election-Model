import numpy as np
import pandas as pd
import arviz as az
import os, time
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm 
import math

import multiprocessing
from collections import defaultdict
from collections import Counter





# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


# Simulated data
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
#az.style.use("seaborn-darkgrid")
az.rcParams['plot.max_subplots'] = 100


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

start_time = time.time()

NO_OF_STATES = 8
NO_OF_ELECTORATES = {'2016':150,'2019':151,'2022':151,'2025':150}
DIM_OF_COV_MATRIX = {'2016':3,'2019':5,'2022':5,'2025':5}

n_simulations = 1000






def group_into_Categories(party_votes_shares_df, div, election_year, is_Other = True):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'COAL','COALNP','COALLP','LP','NP','CLP','LNP','LNQ'}
    GRN_cat = {'GRN'}
    UAPP_cat = {'UAPP','TOP'}
    ON_cat = {'ON'}

    Non_Other_sets = ALP_cat | COAL_cat | GRN_cat # Union of all sets
    if election_year in ['2019','2022','2025']:
        Non_Other_sets = Non_Other_sets | UAPP_cat | ON_cat 
    Other_cols = set(party_votes_shares_df.columns) - Non_Other_sets  # Columns in none of the sets

    ALPs = ALP_cat.intersection(party_votes_shares_df.columns)
    COALs = COAL_cat.intersection(party_votes_shares_df.columns)
    GRNs = GRN_cat.intersection(party_votes_shares_df.columns)
    if election_year in ['2019','2022','2025']:
        ONs =  ON_cat.intersection(party_votes_shares_df.columns)
        UAPPs = UAPP_cat.intersection(party_votes_shares_df.columns)
    OTHs = Other_cols

    # Compute the sums
    sum1 = party_votes_shares_df[list(next(iter(ALPs)) if len(ALPs) == 1 and isinstance(next(iter(ALPs)), set) else ALPs)].sum(axis=1).iloc[0]
    sum2 = party_votes_shares_df[list(next(iter(COALs)) if len(COALs) == 1 and isinstance(next(iter(COALs)), set) else COALs)].sum(axis=1).iloc[0]
    sum3 = party_votes_shares_df[list(next(iter(GRNs)) if len(GRNs) == 1 and isinstance(next(iter(GRNs)), set) else GRNs)].sum(axis=1).iloc[0]
    if election_year in ['2019','2022','2025']:
        sum4 = party_votes_shares_df[list(next(iter(ONs)) if len(ONs) == 1 and isinstance(next(iter(ONs)), set) else ONs)].sum(axis=1).iloc[0]
        sum5 = party_votes_shares_df[list(next(iter(UAPPs)) if len(UAPPs) == 1 and isinstance(next(iter(UAPPs)), set) else UAPPs)].sum(axis=1).iloc[0]
    sum6 = party_votes_shares_df[list(next(iter(OTHs)) if len(OTHs) == 1 and isinstance(next(iter(OTHs)), set) else OTHs)].sum(axis=1).iloc[0]
    if election_year in ['2013','2016']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum6}], index=[div])
    elif election_year in ['2019','2022']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'UAPP':sum5, 'Other':sum6}], index=[div])
    elif election_year == '2025':
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'TOP':sum5, 'Other':sum6}], index=[div])


    return Fundamentals_grouped_df




def get_Prior_estimates_df(election_year, dont_add_ON = False):

    if (election_year == '2016') | dont_add_ON:
        Prior_estimates_df = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv", index_col = None) # ONLY WORKS FOR 2016 - FOR OTHER YEARS WILL REQUIRE Polling_Prior_Votes
    else:
        Prior_estimates_df = pd.read_csv(f"Fundamentals_Votes_ON_add_For_{election_year}.csv", index_col = None)



    Prior_estimates_dict = {
        div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
        for div, group in Prior_estimates_df.groupby("div_nm")
    }

    Prior_estimates_list = []
    for div in Prior_estimates_dict.keys():

        Prior_estimates_list.append(group_into_Categories(Prior_estimates_dict[div], div, election_year))

    Prior_estimates_df = pd.concat(Prior_estimates_list)

    return Prior_estimates_df, Prior_estimates_dict

def remove_ON_back_to_its_country(Prior_estimates_df, election_year):

    # determine transfer ratio of ON votes for those divisions where they are not running

    Prior_estimates_ON_add_df = Prior_estimates_df

    true_prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})

    ON_transfer_percent = {}
    #import pdb;pdb.set_trace()


    # transfer prior %s from ON to original_df parties
    if election_year != '2025':
        for div, proportions in true_prior_estimates_df.iterrows():
            if (proportions['ON']==0):
                curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
                curr_div_True = proportions.to_frame().T

                transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop('ON', axis=1).sum(axis=1).iloc[0]) # This should provide -1 for ON automatically!
                ON_transfer_percent[div] = transfer_proportions

    else:
        # remove both TOP and ON
        for div, proportions in true_prior_estimates_df.iterrows():
            if (proportions['TOP']==0) & (proportions['ON']==0):
                curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
                curr_div_True = proportions.to_frame().T

                transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop(['ON','TOP'], axis=1).sum(axis=1).iloc[0]) # ON and TOP share the -1 between them - careful!
                ON_transfer_percent[div] = transfer_proportions

            elif (proportions['TOP']==0):
                curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
                curr_div_True = proportions.to_frame().T

                transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop('TOP', axis=1).sum(axis=1).iloc[0]) # This should provide -1 for ON automatically!
                ON_transfer_percent[div] = transfer_proportions


    return ON_transfer_percent


election_year = '2025'



Prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON = 0)[0].rename(columns={'Other':'OTH'})
Prior_estimates_no_TOP_df = get_Prior_estimates_df(election_year, dont_add_ON = 1)[0].rename(columns={'Other':'OTH'})

# electorates with no TOP:
No_TOP_electorates = Prior_estimates_no_TOP_df.loc[Prior_estimates_no_TOP_df['TOP']==0,].index


# 1. How much extra would non-TOP seats get? Sum of all divided by sum of only-TOP
Extra_TOP_scaling = Prior_estimates_df['TOP'].sum()/(Prior_estimates_df.loc[~Prior_estimates_df.index.isin(No_TOP_electorates),'TOP'].sum()) - 1 # 1.5654

# 2. Porportions transferred per party: in non-TOP electorates, where do votes come from?

Prior_estimates_df.loc[Prior_estimates_df.index.isin(No_TOP_electorates),]
Prior_estimates_no_TOP_df.loc[Prior_estimates_no_TOP_df.index.isin(No_TOP_electorates),]

TOP_transfers = (Prior_estimates_df.loc[Prior_estimates_df.index.isin(No_TOP_electorates),] - Prior_estimates_no_TOP_df.loc[Prior_estimates_no_TOP_df.index.isin(No_TOP_electorates),]).mean()
TOP_transfers/=TOP_transfers['TOP']

TOP_transfers*= Extra_TOP_scaling
TOP_transfers = TOP_transfers.to_frame().T

import pdb;pdb.set_trace()

National_polls = pd.read_csv("NationalPollsforMGRW2025.csv")
GENERIC_BALLOT_CUTTOFF_DAY = 1059

National_polls.loc[National_polls['Days since last election']>=GENERIC_BALLOT_CUTTOFF_DAY,['COAL','ALP','GRN','ON','TOP','OTH']] +=  National_polls.loc[National_polls['Days since last election'] >= GENERIC_BALLOT_CUTTOFF_DAY, 'TOP'].values[:, None] * TOP_transfers.values

import pdb;pdb.set_trace()

National_polls.to_csv("NationalPollsforMGRW2025.csv", index = False)

