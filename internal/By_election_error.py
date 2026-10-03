import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pingouin import multivariate_normality

from collections import Counter
from itertools import groupby


import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


base_dir = Path.home() / "Necessary CSV Files"
os.chdir(base_dir)






















def group_into_Fundamentals_Categories(party_votes_shares_df, div, is_Other = True):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'COAL','COALNP','COALLP','LP','NP','NAT','CLP','LNP','LNQ'}
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

def group_into_custom_categories(
    party_votes_shares_df,
    div,
    election_year,
    include_sets={"ALP", "COAL", "GRN", "ON", "UAPP"},
    include_other=True
):

    cols = set(party_votes_shares_df.columns)

    # --- category definitions ---
    CAT = {
        "ALP":  {"ALP", "CLR"},
        "COAL": {"COAL","COALNP","COALLP","LP","NP","NAT","CLP","LNP","LNQ"},
        "GRN":  {"GRN"},
        "ON":   {"ON"},
        "UAPP": {"UAPP","TOP"}
    }

    # --- base always included if present ---
    active = {
        k: (v & cols)
        for k, v in CAT.items()
        if k in include_sets and k in ["ALP", "COAL", "GRN"]
    }

    used = set().union(*active.values()) if active else set()

    # --- conditional categories (year-gated but user-controlled) ---
    if election_year in ["2019", "2022", "2025"]:

        if "ON" in include_sets:
            active["ON"] = CAT["ON"] & cols
            used |= active["ON"]

        if "UAPP" in include_sets:
            active["UAPP"] = CAT["UAPP"] & cols
            used |= active["UAPP"]

    # --- compute category sums ---
    out = {}

    for k, v in active.items():
        if len(v) > 0:
            out[k] = party_votes_shares_df[list(v)].sum(axis=1).iloc[0]

    # --- OTHER (residual, always consistent) ---
    if include_other:
        other_cols = list(cols - used)
        out["Other"] = (
            party_votes_shares_df[other_cols].sum(axis=1).iloc[0]
            if len(other_cols) > 0 else 0
        )

    return pd.DataFrame([out], index=[div])



def get_results_df(election_year, to_Fundamentals = True):

    # returns Actual_results_dict with results by party per division, and Fundamentals_results_df


    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    # CountNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best

    if election_year == '2025':
        # concordance with unknown-ab parties
        Actual_results.loc[Actual_results['PartyAb'] == 'CYA','PartyAb'] = 'TOP'
        Actual_results.loc[Actual_results['PartyAb'] == 'GRPF','PartyAb'] = 'PFP'
        Actual_results.loc[Actual_results['PartyAb'] == 'FFPA','PartyAb'] = 'FFP'
        Actual_results.loc[Actual_results['PartyAb'] == 'LTP','PartyAb'] = 'LDP'
        


    # Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
    Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
    Actual_results.loc[Actual_results['PartyAb'].isna(),'PartyAb'] = 'IND'
    Actual_results.loc[Actual_results['PartyAb']=='GVIC','PartyAb'] = 'GRN'
    Actual_results.loc[Actual_results['PartyAb']=='CLR','PartyAb'] = 'ALP'



    # rename IND to INDX by order

    target = 'IND'

    Actual_results_dict = {}
    Fundamentals_results_list = []

    for div in Actual_results['div_nm'].unique():
        div_results = Actual_results.loc[Actual_results['div_nm'] == div,].copy()

        div_results.loc[:,'Count'] = div_results.groupby('PartyAb').cumcount() + 1     # Count instances of the target string

        # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...

        adjusted_party_names = div_results.apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
        ).reset_index(drop=True)


        if to_Fundamentals:
            # keep IND together
            div_results_combined = div_results.groupby(['div_nm', 'PartyAb'], as_index=False)['CalculationValue'].sum()

            Actual_results_dict[div] = div_results_combined.pivot(index='div_nm', columns='PartyAb', values='CalculationValue')


        else:
            # separate independents
            div_results.loc[div_results['div_nm'] == div,'PartyAb'] = adjusted_party_names.values
            div_results_combined = div_results.drop('Count', axis = 1)

            ordered_parties = div_results_combined['PartyAb'].drop_duplicates()

            pivoted = div_results_combined.pivot(index='div_nm', columns='PartyAb', values='CalculationValue')
            
            Actual_results_dict[div] = pivoted.reindex(columns = ordered_parties)


        Fundamentals_results_list.append(group_into_Categories(Actual_results_dict[div], div, election_year))

    Fundamentals_results_df = pd.concat(Fundamentals_results_list)/100

    # Gorton 2016 adjustment: add 0.01 from GRN to Other
    Fundamentals_results_df.loc[Fundamentals_results_df['Other'] == 0.0,['GRN','Other']] += (-0.01,0.01)

    Fundamentals_results_df = Fundamentals_results_df.div(Fundamentals_results_df.sum(axis=1), axis=0).sort_index()

    #Fundamentals_results_df.index = election_year +  Fundamentals_results_df.index


    return Fundamentals_results_df, Actual_results_dict

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

    # Gorton 2016 adjustment: add 0.01 from GRN to Other
    Prior_estimates_df.loc[Prior_estimates_df['Other'] == 0.0,['GRN','Other']] += (-0.01,0.01)

    return Prior_estimates_df, Prior_estimates_dict

def df_to_alr(df, ref_col):
    """Convert a DataFrame of proportions to ALR, dropping ref column."""
    df_alr = np.log(df.drop(columns=[ref_col]).div(df[ref_col], axis=0))
    return df_alr


def alr_to_simplex_vectorized(df, ref_col):
    """Inverse ALR transformation for entire df"""

    # Convert to numpy, apply inverse ALR transformation
    alr_vals = df.values
    exp_vals = np.exp(alr_vals) 

    # Compute the reference category and other values
    ref_vals = 1 / (1 + np.sum(exp_vals, axis=1, keepdims=True))  # Shape: (n_samples, 1)
    simplex_vals = np.concatenate((exp_vals * ref_vals, ref_vals), axis=1)  # Shape: (n_samples, D)
    
    # Return as df with full columns
    new_columns = df.columns.tolist() + [ref_col]
    
    return pd.DataFrame(simplex_vals, columns=new_columns, index=df.index)

def combine_to_CAGO(df):
    cols_to_combine = ['ON', 'UAPP', 'TOP', 'OTH']
    existing_cols = [c for c in cols_to_combine if c in df.columns]

    df['OTH'] = df[existing_cols].sum(axis=1)

    cols_to_drop = [c for c in existing_cols if c != 'OTH']
    df = df.drop(columns=cols_to_drop)

    return df

def get_ALR_swing_df(CAGO_only = 1, ref_col = 'COAL',poll_avg = 0):

    long_ALR_df_list = []

    for year in ['2025','2016','2019','2022']:
        results_df = get_results_df(year)[0].rename(columns={'Other':'OTH'})
        prior_df = get_Prior_estimates_df(year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})

        # combine ON,UAPP,OTH into OTH
        if CAGO_only:
            results_df = combine_to_CAGO(results_df)
            prior_df = combine_to_CAGO(prior_df)
        else:
            if year == '2016':
                continue
            else:
                # use only cases with all parties
                Palmer_cols = {'UAPP','TOP'}
                Palmer_col_curr = [p for p in (set(results_df.columns) & Palmer_cols)][0]

                results_df = results_df.loc[(results_df['ON']>0) & (results_df[Palmer_col_curr]>0),]
                prior_df = prior_df.loc[(prior_df['ON']>0) & (prior_df[Palmer_col_curr]>0),]


        results_ALR_df = df_to_alr(results_df, ref_col)
        prior_ALR_df = df_to_alr(prior_df, ref_col)

        if isinstance(poll_avg, pd.DataFrame):        
            national_prior_alr = df_to_alr(prior_df.mean().to_frame().T, ref_col=ref_col)
            poll_avg_curr = poll_avg.loc[poll_avg['Election'] == int(year),['ALP','GRN','OTH']]
            national_poll_swing = poll_avg_curr.sum() - national_prior_alr
            polls_ALR_df = prior_ALR_df+national_poll_swing.sum()



            prior_df = alr_to_simplex_vectorized(polls_ALR_df, 'COAL')

            ALR_swing = results_ALR_df - polls_ALR_df
        else:
            ALR_swing = results_ALR_df - prior_ALR_df

        #import pdb; pdb.set_trace()


        ALR_swing['div_nm'] = ALR_swing.index
        ALR_swing['election_year'] = year

        prior_df['div_nm'] = prior_df.index
        prior_df['election_year'] = year

      

        # Melt swings
        swing_long = ALR_swing.melt(
            id_vars=['div_nm','election_year'],
            var_name='PartyAb',
            value_name='ALR_swing'
        )

        # Melt previous ALR (baseline)
        prev_long = prior_df.melt(
            id_vars=['div_nm','election_year'],
            var_name='PartyAb',
            value_name='proportion_prev'
        )

        long_ALR_df_list.append(pd.merge(
            prev_long,
            swing_long,
            on=['election_year','div_nm','PartyAb']
        ))

    long_ALR_df = pd.concat(long_ALR_df_list, ignore_index=True)

    return long_ALR_df

def multivariate_t_rvs(mean, cov, df, n_sims):
    """
    Draw from a multivariate t-distribution
    mean : array-like, shape (p,)
    cov  : array-like, shape (p, p)
    df   : degrees of freedom
    n_sims : number of samples
    """
    p = len(mean)
    g = np.random.chisquare(df, n_sims) / df  # shape (n_sims,)
    Z = np.random.multivariate_normal(np.zeros(p), cov, size=n_sims)  # shape (n_sims, p)
    return np.array(mean) + Z / np.sqrt(g)[:, None]  # broadcast sqrt(g) over columns


os.chdir("/home/dania-freidgeim/Australian Election")

ref_col = 'COAL'
CovM = pd.read_csv('PollingErrorALRCovarianceNational2025.csv', index_col = 0)
polled = pd.DataFrame({'ALP':[0.299],'COAL':[0.218],'GRN':[0.126],'ON':[0.253],'UAPP':[0.01],'OTH':[0.094]})

polled = pd.DataFrame({'ALP':[0.20],'COAL':[0.20],'GRN':[0.3],'ON':[0.2],'UAPP':[0.01],'OTH':[0.09]})

#CovM = pd.read_csv('ElectionErrorALRCovarianceNational2025.csv', index_col = 0)
#polled = pd.DataFrame({'ALP':[0.346],'COAL':[0.318],'GRN':[0.122],'ON':[0.064],'UAPP':[0.019],'OTH':[0.131]})

polled_ALR = df_to_alr(polled, ref_col)

n_sims = 1000
#alr_sims = multivariate_t_rvs(mean=polled_ALR.loc[0], cov=CovM, df = 10,n_sims=n_sims)
alr_sims = np.random.multivariate_normal(mean=polled_ALR.loc[0], cov=CovM, size=n_sims)
alr_sims_df = pd.DataFrame(alr_sims, columns=polled_ALR.columns)
prop_sims_df = alr_to_simplex_vectorized(alr_sims_df, ref_col)

np.std(prop_sims_df)


# 2. Apply the asymmetric adjustment to the CovM
# We multiply the whole matrix by the shrinkage, then add the constant to every cell


minor_shrinkage = 0.5     # Tightens GRN, ON, IND, OTH
coal_variance_boost = 0.01  # Expands the COAL reference anchor

minor_shrinkage = 1-coal_variance_boost/0.026
adjusted_CovM = (CovM.values * minor_shrinkage) + coal_variance_boost

alr_sims = np.random.multivariate_normal(mean=polled_ALR.loc[0], cov=adjusted_CovM, size=n_sims)
alr_sims_df = pd.DataFrame(alr_sims, columns=polled_ALR.columns)
prop_sims_df = alr_to_simplex_vectorized(alr_sims_df, ref_col)


import pdb; pdb.set_trace()
# test for symmetric adjustment
polled = pd.DataFrame({'ALP':[0.25],'COAL':[0.25],'GRN':[0.1],'ON':[0.25],'IND':[0.07],'OTH':[0.08]})
coal_variance_boost = 0.015  # Expands the COAL reference anchor

minor_shrinkage = 1-coal_variance_boost/0.362 # 0.026
adjusted_CovM = (CovM * minor_shrinkage) + coal_variance_boost

alr_sims = np.random.multivariate_normal(mean=polled_ALR.loc[0], cov=adjusted_CovM, size=n_sims)
alr_sims_df = pd.DataFrame(alr_sims, columns=polled_ALR.columns)
prop_sims_df = alr_to_simplex_vectorized(alr_sims_df, ref_col)

np.std(prop_sims_df)

plt.figure(figsize=(10,6))
sns.boxplot(data=prop_sims_df)
plt.ylabel("Proportion")
plt.title("Simulated Proportion Variability from ALR Covariance")
plt.show()

import pdb; pdb.set_trace()


get_individual_variability = 0

if get_individual_variability:

    CAGO_only = 1



    if CAGO_only:
        polling_avg = pd.read_csv('National_Day_90_Polls.csv')
        polling_avg['OTH'] = polling_avg[['ON','UAPP','OTH']].sum(axis=1)
        polling_avg = polling_avg[['COAL','ALP','GRN','OTH','Election']]
        polling_avg_ALR = df_to_alr(polling_avg.iloc[:,:-1], ref_col)
        polling_avg_ALR['Election'] = polling_avg['Election']
        
    long_ALR_df = get_ALR_swing_df(CAGO_only = CAGO_only, poll_avg = polling_avg_ALR) # , poll_avg = polling_avg_ALR
    national_mean = long_ALR_df.groupby(['election_year', 'PartyAb'])['ALR_swing'].transform('mean')
    long_ALR_df['ALR_swing_ind'] = long_ALR_df['ALR_swing'] - national_mean # # Seat-level swing = swing minus national mean
    long_ALR_df.loc[long_ALR_df['PartyAb']=='TOP','PartyAb'] = 'UAPP'


    long_ALR_df_swing = get_ALR_swing_df(CAGO_only = CAGO_only, poll_avg = 0) # , poll_avg = polling_avg_ALR
    national_mean_swing = long_ALR_df_swing.groupby(['election_year', 'PartyAb'])['ALR_swing'].transform('mean')
    long_ALR_df_swing['ALR_swing_ind'] = long_ALR_df_swing['ALR_swing'] - national_mean_swing # # Seat-level swing = swing minus national mean
    long_ALR_df_swing.loc[long_ALR_df_swing['PartyAb']=='TOP','PartyAb'] = 'UAPP'

    # correct national residual alr
    # national ALR from actual results (invariant anchor)

    # convert to 
    indiv_seat_alr_swings_CAGO, indiv_seat_alr_swings_ON = [], []

    for year in ['2016','2019','2022','2025']:

        results_df = get_results_df(year)[0].rename(columns={'Other':'OTH'})
        prior_df   = get_Prior_estimates_df(year, dont_add_ON=True)[0].rename(columns={'Other':'OTH'})

        results_df_list, results_df_ON_list = [], []
        prior_df_list, prior_df_ON_list = [], []

        for div in results_df.index:

            # --- CAGO (always) ---
            results_df_list.append(
                group_into_custom_categories(
                    results_df.loc[[div]], div, year,
                    include_sets={'ALP','COAL','GRN'}
                )
            )

            prior_df_list.append(
                group_into_custom_categories(
                    prior_df.loc[[div]], div, year,
                    include_sets={'ALP','COAL','GRN'}
                )
            )

            # --- ON presence check (year + data safe) ---
            ON_present = (
                year in ['2019','2022','2025']
                and 'ON' in results_df.columns
                and results_df.loc[div, 'ON'] > 0
            )

            if ON_present:

                results_df_ON_list.append(
                    group_into_custom_categories(
                        results_df.loc[[div]], div, year,
                        include_sets={'ALP','COAL','GRN','ON'}
                    )
                )

                prior_df_ON_list.append(
                    group_into_custom_categories(
                        prior_df.loc[[div]], div, year,
                        include_sets={'ALP','COAL','GRN','ON'}
                    )
                )

        # =========================
        # SAFE CONCAT (KEY FIX)
        # =========================

        results_df_CAGO = pd.concat(results_df_list)
        prior_df_CAGO   = pd.concat(prior_df_list)

        results_df_ON = (
            pd.concat(results_df_ON_list)
            if len(results_df_ON_list) > 0
            else pd.DataFrame()
        )

        prior_df_ON = (
            pd.concat(prior_df_ON_list)
            if len(prior_df_ON_list) > 0
            else pd.DataFrame()
        )

        # =========================
        # ALR TRANSFORM
        # =========================

        results_ALR_df_CAGO = df_to_alr(results_df_CAGO, ref_col)
        prior_ALR_df_CAGO   = df_to_alr(prior_df_CAGO, ref_col)

        ALR_swing_CAGO = results_ALR_df_CAGO - prior_ALR_df_CAGO
        ALR_swing_CAGO['Election'] = year
        indiv_seat_alr_swings_CAGO.append(ALR_swing_CAGO)

        # --- ON only if exists ---
        if not results_df_ON.empty:

            results_ALR_df_ON = df_to_alr(results_df_ON, ref_col)
            prior_ALR_df_ON   = df_to_alr(prior_df_ON, ref_col)

            ALR_swing_ON = results_ALR_df_ON - prior_ALR_df_ON
            ALR_swing_ON['Election'] = year
            indiv_seat_alr_swings_ON.append(ALR_swing_ON)

    # =========================
    # FINAL CONCAT (SAFE)
    # =========================

    indiv_seat_alr_swings_CAGO = (
        pd.concat(indiv_seat_alr_swings_CAGO)
        .rename(columns={'Other':'OTH'})
    )

    indiv_seat_alr_swings_ON = (
        pd.concat(indiv_seat_alr_swings_ON)
        .rename(columns={'Other':'OTH'})
        if len(indiv_seat_alr_swings_ON) > 0
        else pd.DataFrame()
    )
    import pdb; pdb.set_trace()

    indiv_seat_alr_swings_CAGO.to_csv('Electorate_residual_alr_error_CAGO.csv', index = 'div_nm')
    indiv_seat_alr_swings_ON.to_csv('Electorate_residual_alr_error_CAGOO.csv')


    def plot_variability(long_ALR_df):

        import seaborn as sns
        from statsmodels.nonparametric.smoothers_lowess import lowess

        plt.figure(figsize=(8,6))

        parties = long_ALR_df['PartyAb'].unique()
        colors = {'ALP':'red', 'GRN':'green', 'OTH':'gray','ON':'orange','UAPP':'yellow'}


        for party in parties:
            df_party = long_ALR_df[long_ALR_df['PartyAb'] == party]
            x = df_party['proportion_prev']
            y = df_party['ALR_swing_ind']**2
            smoothed = lowess(y, x, frac=0.3)
            plt.plot(smoothed[:,0], np.sqrt(smoothed[:,1]), color=colors[party], lw=2, label=f'{party} smoothed std')
            plt.scatter(
                x,
                df_party['ALR_swing_ind'],
                alpha=0.2,  # transparency for overplotting
                color=colors[party]
            )
        x = long_ALR_df['proportion_prev']
        y = long_ALR_df['ALR_swing_ind']**2
        smoothed = lowess(y, x, frac=0.3)
        plt.plot(smoothed[:,0], np.sqrt(smoothed[:,1]), color='blue', lw=2, label='overall smoothed std')

        plt.xlabel("Previous proportion")
        plt.ylabel("Seat-level ALR swing residual")
        plt.title("Seat-level ALR swing residuals vs previous proportion")
        plt.legend()
        plt.show()


    plot_variability(long_ALR_df)


    wide = long_ALR_df_swing.pivot_table(index=['div_nm','election_year'],columns='PartyAb',values='ALR_swing_ind')
    Electorate_residual_CovM = np.cov(wide.values, rowvar=False)
    print(Electorate_residual_CovM)











# Election swing error CovM estimation:


def estimate_National_ALR_Covariance_Matrices(ref_col = 'COAL', Day = 90, plot_histogram = False):

    # Old framework that uses single day of polls and removal of years

    REMOVE_ELECTION_YEAR = 1

    # Inputs alr reference column (default Coalition) and Day = {100 - days until election}
    # Uses historical federal and state polling data to estimate ALR-Covariance between main parties after applying reference column

    for curr_election_year in ['2016','2019','2022','2025']:

        curr_election_year = '2025'

        election_year_to_remove = curr_election_year if REMOVE_ELECTION_YEAR else ' '


        for Type in ['Polling','Election_swing']:

            Type = 'Election_swing'


            if Type == 'Polling':
                # 1. Correlation estimate of 3x3 - 6 GRW 2007-2022 + 14 State Elections + 15 State Results
                

                FederalStatePolls = pd.read_csv("StatePollingWeightedAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
                StateElectionsPolls = pd.read_csv("StateElectionsWeightedPollingAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
                OldFederalElectionPollingAverage = pd.read_csv("OldFederalElectionPollingAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
                NationalElectionPollingAveragesGRW = pd.read_csv(f"NationalElectionPollingAveragesGRW_Day_{Day}.csv", index_col = None).set_index('Election')


                FederalStateResults =  pd.read_csv("StateFederalResults.csv", index_col = None).set_index('Election')
                StateElectionResults = pd.read_csv("StateElectionResults.csv", index_col = None).set_index('Election')
                OldFederalElectionResults = pd.read_csv("OldFederalElectionResults.csv", index_col = None).set_index('Election')
                NationalElectionResults = pd.read_csv("NationalElectionResults.csv", index_col = None).set_index('Election')

                NationalElectionPollingAveragesGRW.index = NationalElectionPollingAveragesGRW.index.astype(str)
                NationalElectionResults.index = NationalElectionResults.index.astype(str)
                OldFederalElectionPollingAverage.index = OldFederalElectionPollingAverage.index.astype(str)
                OldFederalElectionResults.index = OldFederalElectionResults.index.astype(str)

                #Arbitrarily selected State per election
                Selected_states = ['2022NSW','2019VIC','2016QLD']
                GRN_OldFederalPolls = OldFederalElectionPollingAverage.rename(columns={'DEM<=1996/GRN':'GRN'})
                GRN_OldFederalResults = OldFederalElectionResults.rename(columns={'DEM<=1996/GRN':'GRN'})


                CAGO_Polling_Avg = pd.concat([FederalStatePolls.iloc[FederalStatePolls.index.isin(Selected_states),], StateElectionsPolls, NationalElectionPollingAveragesGRW, GRN_OldFederalPolls.loc[GRN_OldFederalPolls.index.isin(['2001','2004']),]], ignore_index = False)
                CAGO_Results = pd.concat([FederalStateResults.iloc[FederalStateResults.index.isin(Selected_states),], StateElectionResults, NationalElectionResults, GRN_OldFederalResults.loc[GRN_OldFederalResults.index.isin(['2001','2004']),]], ignore_index = False)

                CAGO_Polling_ALR = np.log(CAGO_Polling_Avg.drop(columns=[ref_col]).div(CAGO_Polling_Avg[ref_col], axis=0))
                CAGO_Results_ALR = np.log(CAGO_Results.drop(columns=[ref_col]).div(CAGO_Results[ref_col], axis=0))

                CAGO_ALR_swings = CAGO_Results_ALR - CAGO_Polling_ALR
                # correct for past polling bias!
                CAGO_ALR_swings_centered = CAGO_ALR_swings - CAGO_ALR_swings.mean()

                if REMOVE_ELECTION_YEAR:
                    CAGO_ALR_swings_centered = CAGO_ALR_swings_centered.loc[~(CAGO_ALR_swings_centered.index.str.startswith(election_year_to_remove)),]

                corr_matrix = np.corrcoef(CAGO_ALR_swings_centered.values, rowvar=False)
                print(corr_matrix)
                #import pdb;pdb.set_trace()



                # VARIANCE ESTIMATION


                CAGO_variance_estimation_polls = pd.concat([CAGO_Polling_Avg,GRN_OldFederalPolls.iloc[:-2]])
                CAGO_variance_estimation_results = pd.concat([CAGO_Results,GRN_OldFederalResults.iloc[:-2]])

                # extract GRN from before 2004 - very different party vote share today to back then!
                CAGO_variance_estimation_polls.loc[CAGO_variance_estimation_polls.index.isin(['1987','1990','1993','1996','1998','2001']),'GRN'] = np.nan
                CAGO_variance_estimation_results.loc[CAGO_variance_estimation_results.index.isin(['1987','1990','1993','1996','1998','2001']),'GRN'] = np.nan


                CAGO_Variance_Polling_ALR = np.log(CAGO_variance_estimation_polls.drop(columns=[ref_col]).div(CAGO_variance_estimation_polls[ref_col], axis=0))
                CAGO_Variance_Results_ALR = np.log(CAGO_variance_estimation_results.drop(columns=[ref_col]).div(CAGO_variance_estimation_results[ref_col], axis=0))



                CAGO_Variance_estimation_swings = CAGO_Variance_Results_ALR - CAGO_Variance_Polling_ALR
                CAGO_Variance_estimation_swings_centered = CAGO_Variance_estimation_swings - CAGO_Variance_estimation_swings.mean()


                # Suggested weighting scheme of polling data - 3 Federal states = 0.3, State election = 0.5, Recent Federal election = 1, 2001/2004 federal: 0.6, Pre-2000 federal: 0.5
                weights = np.array([0.3,0.3,0.3,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,0.6,0.6,0.5,0.5,0.5,0.5,0.5])

                # remove data from current election
                if election_year_to_remove and election_year_to_remove != '2025':
                    if election_year_to_remove == '2016':
                        indices = [4,5,14,15,20,21,22,23,24,25,26,27,28,29]
                    if election_year_to_remove == '2019':
                        indices = [2,3,4,5,7,9,10,14,15,16,19,20,21,22,23,24,25,26,27,28,29]
                    if election_year_to_remove == '2022':
                        indices = [0,2,3,4,5,7,9,10,12,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29]

                    weights = weights[indices]
                    CAGO_Variance_estimation_swings_centered = CAGO_Variance_estimation_swings_centered.iloc[indices]



            elif Type == 'Election_swing':
                Election_swings_df = pd.read_csv("ElectionSwings.csv", index_col=None).set_index("Election")
                Election_swings_df = pd.read_csv("ElectionSwings_updated_2026.csv", index_col=None).set_index("Election")
                Election_results_curr = Election_swings_df.iloc[:,:4]
                Election_results_prev = Election_swings_df.iloc[:,4:]
                Election_results_curr_ALR = np.log(Election_results_curr.drop(columns=[ref_col]).div(Election_results_curr[ref_col], axis=0))
                Election_results_prev_ALR = np.log(Election_results_prev.drop(columns=[ref_col+'_prev']).div(Election_results_prev[ref_col+'_prev'], axis=0))
                Election_swings_ALR = Election_results_curr_ALR - Election_results_prev_ALR.values

                Election_swings_ALR_centered = Election_swings_ALR - Election_swings_ALR.mean()
                Election_swings_ALR_centered_after_1996 = Election_swings_ALR_centered.loc[~(Election_swings_ALR_centered.index.isin(['1987','1990','1993','1996'])),]


                if REMOVE_ELECTION_YEAR:
                    CAGO_ALR_swings_centered = Election_swings_ALR_centered_after_1996.loc[~(Election_swings_ALR_centered_after_1996.index.str.startswith(election_year_to_remove)),]



                corr_matrix_Elec = np.corrcoef(Election_swings_ALR_centered_after_1996.values, rowvar=False)
                print(corr_matrix_Elec)

                Election_swings_ALR_for_Var = Election_swings_ALR.copy()
                Election_swings_ALR_for_Var.loc[Election_swings_ALR_for_Var.index.isin(['1987','1990','1993','1996','1998','2001','SAState2006']),'GRN'] = np.nan
                Election_swings_ALR_for_Var_centered = Election_swings_ALR_for_Var - Election_swings_ALR_for_Var.mean()


                # get weight function decaying in time
                index = Election_swings_ALR_for_Var_centered.index
                years = index.str[-4:].astype(int)
                is_federal = index.str.len() == 4

                current_year = years.max()
                years_ago = current_year - years

                half_life = 8
                lambda_ = np.log(2) / half_life
                w_time = np.exp(-lambda_ * years_ago)
                w_type = np.where(is_federal, 1.0, 0.6)
                weights = w_time * w_type
                weights = weights.values / weights.values.sum()



                # 1 to post-2004 elections, 0.5 to state elections, 0.6 to 2004/2001, 0.5 to 20th century elecs
                if 0:
                    weights = np.array([1,1,1,1,1,1,0.6,0.6] + [0.5]*24)

                    weights = np.array([1,1,1,1,1,1,0.6,0.6] + [0.5]*24 + [1] + [0.5]*4)

                    if election_year_to_remove and election_year_to_remove != '2025':
                        if election_year_to_remove == '2016':
                            indices = [3,4,5,6,7,8,9,10,11,12,15,18,19,22,23,27]
                        if election_year_to_remove == '2019':
                            indices = [2,3,4,5,6,7,8,9,10,11,12,14,15,17,18,19,21,22,23,26,27,29]
                        if election_year_to_remove == '2022':
                            indices = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,18,19,21,22,23,25,26,27,29,31]

                        weights = weights[indices]
                        Election_swings_ALR_for_Var_centered = Election_swings_ALR_for_Var_centered.iloc[indices]






            CAGO_Variance_estimation_swings_centered = CAGO_Variance_estimation_swings_centered if Type == 'Polling' else Election_swings_ALR_for_Var_centered



            def weighted_nanstd(data, weights):
                # estimates variances while respecting np.nan for GRN before 2004
                weighted_std = []

                for i in range(data.shape[1]):
                    col = data[:, i]
                    mask = ~np.isnan(col)
                    w = weights[mask]
                    x = col[mask]
                    w /= w.sum()  # normalize weights
                    mean = np.sum(w * x)
                    var = np.sum(w * (x - mean)**2)
                    weighted_std.append(np.sqrt(var))
                return np.array(weighted_std)


            def weighted_cov_pairwise(data, weights):
                K = data.shape[1]
                Sigma = np.zeros((K, K))
                
                for j in range(K):
                    for k in range(K):
                        col_j = data[:, j]
                        col_k = data[:, k]
                        
                        mask = ~np.isnan(col_j) & ~np.isnan(col_k)
                        
                        if mask.sum() == 0:
                            Sigma[j, k] = np.nan
                            continue
                        
                        w = weights[mask]
                        xj = col_j[mask]
                        xk = col_k[mask]
                        
                        w = w / w.sum()  # normalize
                        
                        mu_j = np.sum(w * xj)
                        mu_k = np.sum(w * xk)
                        
                        Sigma[j, k] = np.sum(w * (xj - mu_j) * (xk - mu_k))
                
                return Sigma
            


            data = Election_swings_ALR_for_Var_centered.values
            cov_matrix = weighted_cov_pairwise(data, weights)

            if not (np.linalg.eigvalsh(cov_matrix)>0).all():
                raise ValueError('CovM not positive semidefinite')
            
            print(cov_matrix)

            #import pdb; pdb.set_trace()





estimate_National_ALR_Covariance_Matrices(ref_col = 'COAL', Day = 90, plot_histogram = False)


ADD_DAYS_SINCE_ELECTION = 0

if ADD_DAYS_SINCE_ELECTION:
    By_election_data = pd.read_csv("Federal_by_election_data.csv")
    ELECTION_DATES_DICT = {'2013': '07.09.13','2016': '02.07.16', '2019': '18.05.19','2022': '21.05.22'}
    By_election_data['days_since_election'] = (pd.to_datetime(By_election_data['byelection_date'], format='%d.%m.%y') - pd.to_datetime(By_election_data['prev_election'].astype(str).map(ELECTION_DATES_DICT), format='%d.%m.%y')).dt.days

    By_election_data.to_csv("Federal_by-election_data.csv", index=False)

By_election_data = pd.read_csv("Federal_by_election_data.csv")
By_election_data['div_nm'] = By_election_data['div_nm'].replace({'Batman':'Cooper'})
Major_sitout_byelections = By_election_data.loc[(~By_election_data['Major_sitouts'].isna()) & (By_election_data['prev_election'].astype(int)>2010),]['div_nm'].to_list()


# concatenate results for by-elections since 2013
By_election_Fundamentals_Results_df_list = []
for data_year in ['2013','2016','2019','2022']:
    By_election_Fundamentals_Results_df_list.append(pd.read_csv(f"By_election_Fundamentals_after_{data_year}.csv"))

By_election_Fundamentals_Results_df = pd.concat(By_election_Fundamentals_Results_df_list)


import pdb; pdb.set_trace()

# copied from group categories function at top
Prior_estimates_dict = {
    div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
    for div, group in By_election_Fundamentals_Results_df.groupby("div_nm")
}
Prior_results_dict = {
    div: pd.DataFrame([group.set_index("PartyAb")["FP_result"].to_dict()])
    for div, group in By_election_Fundamentals_Results_df.groupby("div_nm")
}


Prior_estimates_list, Prior_results_list = [],[]
for div in Prior_estimates_dict.keys():

    Prior_estimates_list.append(group_into_Categories(Prior_estimates_dict[div], div, election_year='2016')) # '2016' coded as CAGO only
    Prior_results_list.append(group_into_Categories(Prior_results_dict[div], div, election_year='2016'))

Prior_estimates_df = pd.concat(Prior_estimates_list)
Prior_results_df = pd.concat(Prior_results_list)

# Convert to ALR, excluding those in Major_sitouts, determine if error increases with days since election

Prior_estimates_ALR = df_to_alr(Prior_estimates_df.loc[~Prior_estimates_df.index.isin(Major_sitout_byelections),], ref_col).rename(columns={'Other':'OTH'})
Prior_results_ALR = df_to_alr(Prior_results_df.loc[~Prior_results_df.index.isin(Major_sitout_byelections),], ref_col).rename(columns={'Other':'OTH'})

ALR_errors_df = (Prior_results_ALR-Prior_estimates_ALR).merge(By_election_data.loc[By_election_data['prev_election'].astype(int)>2010,['div_nm','days_since_election']], left_index = True, right_on = 'div_nm').sort_values(by='days_since_election')
df = ALR_errors_df
df.iloc[:,:3] = df.iloc[:,:3].abs()
# 
plt.scatter(df['days_since_election'], df['ALP'], label='ALP')
plt.scatter(df['days_since_election'], df['GRN'], label='GRN')
plt.scatter(df['days_since_election'], df['OTH'], label='OTH')

plt.xlabel('Days Since Election')
plt.ylabel('ALR Error')
plt.legend()
plt.show()
import pdb; pdb.set_trace()



Byelection_daily_polling_averages = pd.read_csv("Byelection_daily_polling_averages.csv")
Byelection_daily_polling_averages.loc[Byelection_daily_polling_averages['Election']=='Batman2019','Election'] = 'Cooper2019'
prev_election_result_dict = {}

# 1. Get polling shift since last election - national

National_election_results = pd.read_csv("NationalElectionResults.csv")
National_election_results_alr = df_to_alr(National_election_results.iloc[:,:-1], ref_col = 'COAL')
National_election_results_alr['Election'] = National_election_results['Election'].astype(str)
# get alr shift from last election in National_election_results; need to merge things on 'Election' column, which in Byelection_daily_polling_averages is 'Cook2025', where '2025' is 3 years later than election_year
Byelection_daily_polling_averages['prev_election_year'] = (Byelection_daily_polling_averages['Election'].str[-4:].astype(int) - 3).astype(str)
Byelection_daily_polling_averages['div_nm'] = (Byelection_daily_polling_averages['Election'].str[:-4])
Byelection_daily_polling_alr = df_to_alr(Byelection_daily_polling_averages.iloc[:,:4],ref_col='COAL')
Byelection_daily_polling_alr_swing = Byelection_daily_polling_averages.drop(columns=['COAL'])
Byelection_daily_polling_alr_swing.iloc[:,:3] = Byelection_daily_polling_alr

alr_cols = ['ALP', 'GRN', 'OTH']
merged = Byelection_daily_polling_alr_swing.merge(National_election_results_alr,left_on='prev_election_year',right_on='Election',how='left',suffixes=('', '_nat'))
merged[alr_cols] = (merged[alr_cols].values - merged[[c + '_nat' for c in alr_cols]].values)
nat_alr_swing_daily = merged.drop(columns=['Election_nat'] + [c + '_nat' for c in alr_cols])



# 2. Get Fundamentals results converted to CAGO & add polling shift since last election 

nat_alr_swing_daily_with_majors = nat_alr_swing_daily.loc[~nat_alr_swing_daily['div_nm'].isin(Major_sitout_byelections),]


# compare to actual results



merged = nat_alr_swing_daily_with_majors.join(Prior_estimates_ALR,on='div_nm',rsuffix='_prior')
assert not merged[['ALP_prior','GRN_prior','OTH_prior']].isna().any().any()
merged[['ALP','GRN','OTH']] = merged[['ALP','GRN','OTH']].values + merged[['ALP_prior','GRN_prior','OTH_prior']].values


final_polling = merged.loc[merged['Day_index']==-1,alr_cols+['div_nm']].sort_values(by='div_nm').set_index('div_nm')
early_polling = merged.loc[merged['Day_index']==-100,alr_cols+['div_nm']].sort_values(by='div_nm').set_index('div_nm')
month_out_polling = merged.loc[merged['Day_index']==-30,alr_cols+['div_nm']].sort_values(by='div_nm').set_index('div_nm')
print((early_polling - Prior_results_ALR).abs().mean())
print((month_out_polling - Prior_results_ALR).abs().mean()) # 
print((final_polling - Prior_results_ALR).abs().mean()) # final polling averages
print((Prior_estimates_ALR - Prior_results_ALR).abs().mean()) # prior

import pdb; pdb.set_trace()



nat_alr_shift_daily_with_majors = merged[alr_cols]
nat_alr_shift_daily_with_majors.loc[:,['Day_index','Election','Density','prev_election_year','div_nm']] = nat_alr_swing_daily_with_majors[['Day_index','Election','Density','prev_election_year','div_nm']].values

merged2 = nat_alr_shift_daily_with_majors.join(Prior_results_ALR,on='div_nm',rsuffix='_prior')
merged2[alr_cols] = (merged2[alr_cols].values - merged2[[c + '_prior' for c in alr_cols]].values)

nat_alr_prediction_daily_with_majors = nat_alr_shift_daily_with_majors
nat_alr_prediction_daily_with_majors.loc[:,alr_cols] =  (merged2[alr_cols].values - merged2[[c + '_prior' for c in alr_cols]].values)
# direct results
nat_swing_daily_with_majors = alr_to_simplex_vectorized(merged[alr_cols], ref_col)
nat_swing_daily_with_majors[['Day_index','Election','Density','prev_election_year','div_nm']] = nat_alr_swing_daily_with_majors[['Day_index','Election','Density','prev_election_year','div_nm']]



# 3. 
