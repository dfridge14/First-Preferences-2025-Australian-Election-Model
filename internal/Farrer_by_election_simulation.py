import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from collections import defaultdict, Counter
import time
import html
from datetime import date, timedelta

import dill
import pickle
import plotly.graph_objects as go

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

os.chdir('/home/dania-freidgeim/Australian Election')

# Simulated data
import random
ELECTION_SEED = 90526
rng = np.random.default_rng(ELECTION_SEED)


NUM_MAIN_PARTIES = 5
LP_NP_DIR_ALPHA = 4.178317954770676
LP_NP_DIR_ALPHA_POLL  = 12

DIR_epsilon = 5e-3

MAJOR_SITOUT_DIR_ALPHA = 14 # (4 + 7 + 31)/3

BY_ELECTION_SCALING = 1.3# 1.18055492771072
SA_DATA_UNCTY_WGHT = 1.5
LEAPFROG_HELP = 0.7

SEAT_POLL_SCALE_1 = 1.0747964937804577
SEAT_POLL_SCALE_2 = 0.8977871632158285
SEAT_POLL_SCALE_3 = 0.7965537354979125



SEAT_POLL_SCALE = {
    0: 1, # for safety
    1: SEAT_POLL_SCALE_1,
    2: SEAT_POLL_SCALE_2,
    3: SEAT_POLL_SCALE_3
}


# 1. Get nat-poll-adjusted prior_df, for those electorates/election_years that have both priors and seat polls (102+15)



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
    if election_year in ['2013','2016', 'Byelection']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum6}], index=[div])
    elif election_year in ['2019','2022']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'UAPP':sum5, 'Other':sum6}], index=[div])
    elif election_year == '2025':
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'TOP':sum5, 'Other':sum6}], index=[div])


    return Fundamentals_grouped_df



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

    if election_year == 'Byelection':
        Prior_estimates_df = pd.read_csv(f"By_election_Fundamentals_full_added_Majors.csv", index_col = None)
    elif (election_year == '2016') | dont_add_ON:
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



def combine_to_CAGO(df):
    cols_to_combine = ['ON', 'UAPP', 'TOP', 'OTH']
    existing_cols = [c for c in cols_to_combine if c in df.columns]

    df['OTH'] = df[existing_cols].sum(axis=1)

    cols_to_drop = [c for c in existing_cols if c != 'OTH']
    df = df.drop(columns=cols_to_drop)

    return df



CAGO_only = 1
ref_col = 'COAL'


election_years = ['2016','2019','2022','2025','Byelection']
ELECTION_DATE_NUM = {'2013':1113, '2016':1028, '2019':1050, '2022':1099,'2025':1078}

OPT_EXP_DECAY_PARAMETERS = {'OTH':[0.18562381, 0.28067983, 0.0507943],'GRN':[0.15782239, 0.24219055, 0.02555061],'ALP':[0.09403604, 0.20026747, 0.02794699], 'ON': [0.1957, 0.2800, 0.02555]} 



def adjust_covm_for_time(final_day_covm, days_to_election, params_dict, ind_fixed_var=0.0001):
    """
    Takes a base (t=0) 5x5 covariance matrix and scales its variances based on 
    an exponential time-decay curve, preserving the underlying correlations.
    """
    parties = ['ALP', 'GRN', 'ON', 'IND', 'OTH']
    
    # 1. Isolate the pure correlation matrix (R) from the final-day covariance matrix
    # R_ij = Cov_ij / (std_i * std_j)
    stds_base = np.sqrt(np.diag(final_day_covm))
    outer_stds = np.outer(stds_base, stds_base)
    correlation_matrix = final_day_covm / outer_stds
    
    R = correlation_matrix.values if isinstance(correlation_matrix, pd.DataFrame) else correlation_matrix
        
    # 2. Calculate the specific standard deviations for day 't'
    dynamic_stds = []
    for party in parties:
        if party == 'IND':
            # IND is mathematically frozen to its microscopic variance
            dynamic_stds.append(np.sqrt(ind_fixed_var))
        else:
            sigma_start, sigma_end, k = params_dict[party]
            # Exponential decay: tightest at t=0 (sigma_start), widens as t increases
            std_t = sigma_end + (sigma_start - sigma_end) * np.exp(-k * days_to_election)
            dynamic_stds.append(std_t)
            
    # 3. Reconstruct the new covariance matrix (S @ R @ S)
    S = np.diag(dynamic_stds)
    C_t = S @ R @ S
    
    # 4. Guarantee Positive-Definiteness via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(C_t)
    C_clean = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    
    return pd.DataFrame(C_clean, index=parties, columns=parties)




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


def alr_to_simplex_simulation_array(alr_array):
        """
        Perform inverse ALR transformation on a 3D array of shape (S, D, K).
        Returns a 3D array of shape (S, D, K+1) in the probability simplex.
        reference category is the 1st in the simplex
        """
        exp_vals = np.exp(alr_array)  # shape (S, D, K)
        ref_vals = 1 / (1 + np.sum(exp_vals, axis=-1, keepdims=True))  # shape (S, D, 1)
        simplex = np.concatenate([ref_vals, exp_vals * ref_vals], axis=-1)  # shape (S, D, K+1)
        return simplex


def simulate_Polling_Fundamentals_model(
    n_simulations,
    election_year,
    df_t=0,
    Prior_estimates_df=pd.DataFrame(),
    polling_avg = pd.DataFrame(),
    National_prior = pd.DataFrame(),
    National_Polling_error_cov =  pd.DataFrame(),
    Electorate_Residuals_cov =  pd.DataFrame(),
    For_combined_model = False
):
    
    NUM_ACTIVE_ELECTORATES = len(Prior_estimates_df)
    
    # -----------------------------
    # Distribution choice
    # -----------------------------
    dist = "Normal" if df_t == 0 else "t"

    # -----------------------------
    # Volatility grouping (optional, keep only for general elections)
    # -----------------------------

  
    remapped_weights_idx_dict = {0: list(range(NUM_ACTIVE_ELECTORATES))} # base default - all category 0

   
    # -----------------------------
    # NATIONAL POLLING ERROR
    # -----------------------------

    National_Simulated_polling_error = rng.multivariate_normal(
        mean=np.zeros(National_Polling_error_cov.shape[0]),
        cov=National_Polling_error_cov.values,
        size=n_simulations
    )[:, None, :]

    National_Simulated_polling_error_expanded = np.repeat(
        National_Simulated_polling_error,
        NUM_ACTIVE_ELECTORATES,
        axis=1
    )

    # -----------------------------
    # ELECTORATE RESIDUAL ERROR
    # -----------------------------
    
    Scaled_covs = {}    
    
    d = Electorate_Residuals_cov.shape[0]
    OTH_index = -1

    
    
    Scaled_covs[0] = Electorate_Residuals_cov.values # No transformation

    Electorate_Residuals_Simulated_error = np.empty(
        (n_simulations,
        NUM_ACTIVE_ELECTORATES,
        d)
    )

    for scale, indices in remapped_weights_idx_dict.items():

        cov = Scaled_covs[scale]
        n_group = len(indices)

        if n_group == 0:
            continue

        USE_MEAN_SHIFT = 0
        mean = np.zeros(d)
        if (election_year == "Byelection") and USE_MEAN_SHIFT:
            grn_idx = 1
            mean[grn_idx] = -0.28
            print('mean shift used')
            
        group_sims = rng.multivariate_normal(mean=mean,cov=cov,size=n_simulations * n_group)

        if dist == "t":
            g = rng.gamma(df_t / 2., 2. / df_t, size=n_simulations * n_group)
            group_sims = group_sims / np.sqrt(g)[:, None]

        Electorate_Residuals_Simulated_error[:, indices, :] = group_sims.reshape(
            n_simulations,
            n_group,
            d
        )



    # -----------------------------
    # BASE PRIOR STRUCTURE
    # -----------------------------
    ref_col = "COAL"



    polling_alr = np.log(polling_avg.drop(columns=[ref_col]).div(polling_avg[ref_col], axis=0))

    National_prior_alr = df_to_alr(National_prior, ref_col=ref_col)

    Prior_estimates_alr = np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))

    # -----------------------------
    # EXPAND PRIOR
    # -----------------------------
    Prior_estimates_alr_expanded = np.tile(
        Prior_estimates_alr.to_numpy(),
        (n_simulations, 1, 1)
    )

    # -----------------------------
    # NATIONAL SHOCK APPLICATION
    # -----------------------------
    Simulated_national_result_alr = (polling_alr.values+ National_Simulated_polling_error_expanded)

    # -----------------------------
    # ELECTORATE PROJECTION WEIGHTING
    # -----------------------------
    

   
    # BYELECTION: keep national shock, no beta shrinkage
    Projected_Electorate_Polling_Results_ALR = (
        Prior_estimates_alr_expanded
        + (
            Simulated_national_result_alr
            - National_prior_alr.values
        )
    )




    if For_combined_model:
        return Projected_Electorate_Polling_Results_ALR
    
    # -----------------------------
    # ADD ELECTORATE RESIDUALS
    # -----------------------------
    Simulated_Electorate_Polling_Results_ALR = (
        Projected_Electorate_Polling_Results_ALR
        + Electorate_Residuals_Simulated_error
    )

    # -----------------------------
    # CONVERT TO SIMPLEX
    # -----------------------------
    Simulated_Electorate_Polling_Results = alr_to_simplex_simulation_array(
        Simulated_Electorate_Polling_Results_ALR
    )


    return (
        Simulated_Electorate_Polling_Results
    )




def combine_models_alr_precision_weighted(
    n_simulations,
    B,
    nat_alr_sims, # Simulated_Electorate_Polling_Results_ALR for this div (n_sims, 3)
    seat_alr_base, # alr_base from precomputed_dict (3,)
    Nat_poll_covm, # National_Polling_error_cov (3, 3)
    Resid_covm,    # Electorate_Residuals_cov (3, 3)
    Seat_poll_covm # Seat_poll_covm_dict[election_year] (3, 3)
):
    
    # case to deal with no seat polls
    if seat_alr_base is None:
        B = 0  # Force seat poll weight to 0
        seat_alr_base = np.zeros(nat_alr_sims.shape[1])

    # 1. Calculate Total National Covariance = Sum of national-level error and electorate-specific residuals
    Sigma_nat = Nat_poll_covm.values + Resid_covm.values
    Inv_Sigma_nat = np.linalg.inv(Sigma_nat)

    # 2. Calculate Seat Polling Precision, weighted by B
    Sigma_seat = Seat_poll_covm.values
    Inv_Sigma_seat_weighted = B * np.linalg.inv(Sigma_seat)

    # 3. Compute the Combined Covariance (The New Distribution Spread) Sigma_comb = inv(Inv_Nat + B * Inv_Seat)
    Sigma_comb = np.linalg.inv(Inv_Sigma_nat + Inv_Sigma_seat_weighted)

   

    TEST_MIXING = 1
    if TEST_MIXING:
        # Using matrix multiplication (@)
        W_nat = Sigma_comb @ Inv_Sigma_nat
        W_seat = Sigma_comb @ Inv_Sigma_seat_weighted

        # 5. Extract a scalar mixing value (0.0 to 1.0)
        k = W_seat.shape[0] # The number of parties/dimensions in the matrix
        seat_mixing_value = np.trace(W_seat) / k
        nat_mixing_value = np.trace(W_nat) / k

        # To answer "Which one was used more heavily?"
        print(f"National Weight: {nat_mixing_value * 100:.1f}%")
        print(f"Seat Poll Weight: {seat_mixing_value * 100:.1f}%")


        # Seat poll vs nat poll weights

        #nat_weights_per_dim = np.diag(W_nat)
        #seat_weights_per_dim = np.diag(W_seat)

        #print("\n--- Mixing Weights Per ALR Dimension ---")
        #for i, party in enumerate(['GRN','ON','IND','OTH']):
        #    w_n = nat_weights_per_dim[i] * 100
        #    w_s = seat_weights_per_dim[i] * 100
        #    print(f"{party:<6} -> National: {w_n:>4.1f}%  |  Seat Poll: {w_s:>4.1f}%")



    # 4. Compute the Combined Means for every Simulation
    # Because nat_alr_sims varies per simulation, we calculate a unique mu_comb per sim #mu_comb = Sigma_comb * (Inv_Nat * nat_alr_sim + B * Inv_Seat * seat_alr_base)
    
    # Pre-calculate the seat-part of the numerator
    seat_part = Inv_Sigma_seat_weighted @ seat_alr_base # (3,)
    
    # Calculate nat-part for all simulations: (n_sims, 3) @ (3, 3) -> (n_sims, 3)
    nat_part = nat_alr_sims @ Inv_Sigma_nat.T 
    
    # Combine and multiply by Sigma_comb to get the new simulation centers
    combined_mu = (nat_part + seat_part) @ Sigma_comb.T # (n_sims, 3)

    #print(alr_to_simplex_simulation_array(combined_mu))
    #import pdb; pdb.set_trace()

    d = Seat_poll_covm.shape[0]

    # 5. Draw final realizations from the combined distribution
    # This centers the Fundementals around the update while narrowing the variance
    combined_errors = rng.multivariate_normal(
        mean=np.zeros(d),
        cov=Sigma_comb,
        size=n_simulations
    )

    # Final ALR realizations for [ALP, GRN, Other]
    combined_alr = combined_mu + combined_errors

    return combined_alr



def allocate_major_sitout_to_cago(
    nat_sims_cago, 
    cago_names, 
    div, 
    missing_party, 
    mapping_df, 
    proportions_df, 
    prior_row,
    MAJOR_SITOUT_DIR_ALPHA
):
    n_sims = nat_sims_cago.shape[0]
    
    # --- 1. Identify setup and Dirichlet params
    sitout_val_mapping = 'LP' if missing_party == 'COAL' else missing_party
    
    _, has_strong_IND = mapping_df.loc[
        mapping_df['div_nm'] == div, ['Major_sitout', 'has_strong_IND']
    ].drop_duplicates().to_numpy().ravel()
    
    row = proportions_df[
        (proportions_df['Major_sitout'] == sitout_val_mapping) & 
        (proportions_df['has_strong_IND'] == has_strong_IND)
    ].iloc[0]
    
    if has_strong_IND == 1:
        mu = np.array([row['theta_aligned'], row['theta_ind'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'IND', 'other_major', 'other']
    else:
        mu = np.array([row['theta_aligned'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'other_major', 'other']
        
    alpha = mu * MAJOR_SITOUT_DIR_ALPHA
    p_group = rng.dirichlet(alpha, size=n_sims)

    # >>> NEW: Calculate the Aligned Split Ratios from Prior
    aligned_list = ['GRN', 'HMP', 'SPP'] # NEW
    p_aligned_total = prior_row[aligned_list].sum(axis=1).iloc[0] # NEW
    grn_share_of_aligned = prior_row['GRN'].iloc[0] / p_aligned_total # NEW
    oth_share_of_aligned = 1.0 - grn_share_of_aligned # NEW
    
    # --- 2. Extract and remove sitout vote in CAGO matrix
    adjusted_cago = nat_sims_cago.copy()
    sitout_idx = cago_names.index(missing_party)
    V = adjusted_cago[:, sitout_idx].copy()
    adjusted_cago[:, sitout_idx] = 0.0
    
    # --- 3. Allocate to CAGO categories
    sub = mapping_df[
        (mapping_df['div_nm'] == div) & 
        (mapping_df['Major_sitout'] == missing_party) & 
        (mapping_df['has_strong_IND'] == has_strong_IND)
    ]
    
    for g_idx, g in enumerate(groups):
        parties = sub.loc[sub['group'] == g, 'PartyAb'].tolist()



        if not parties:
            continue
            
        group_mass = V * p_group[:, g_idx] # NEW

        if (missing_party == 'ALP') and (g == 'aligned'): # NEW
            # 1. Allocate specific share to the Greens bucket
            if 'GRN' in cago_names: # NEW
                grn_idx = cago_names.index('GRN') # NEW
                adjusted_cago[:, grn_idx] += group_mass * grn_share_of_aligned # NEW
            
            # 2. Allocate the remaining aligned share to the Other bucket
            target_oth = 'OTH' if 'OTH' in cago_names else 'Other' # NEW
            oth_idx = cago_names.index(target_oth) # NEW
            adjusted_cago[:, oth_idx] += group_mass * oth_share_of_aligned # NEW
        else: # NEW
            # Standard logic for IND and Other Major groups
            target_cago = next((p for p in parties if p in cago_names), None)
            
            if target_cago is None:
                if 'IND' in cago_names and any(p.startswith('IND') for p in parties):
                    target_cago = 'IND'
                else:
                    target_cago = 'OTH' if 'OTH' in cago_names else 'Other'
                    
            target_idx = cago_names.index(target_cago)
            adjusted_cago[:, target_idx] += group_mass # NEW




    # --- 4. Delete missing party column
    adjusted_cago = np.delete(adjusted_cago, sitout_idx, axis=1)
    cago_names_reduced = [p for i, p in enumerate(cago_names) if i != sitout_idx]


    
    # Package the reallocation data to ensure perfect synchronicity later
    realloc_data = {
        'p_group': p_group,
        'groups': groups,
        'sub': sub,
        'has_strong_IND': has_strong_IND,
        'missing_party': missing_party,
        'grn_share_of_aligned': grn_share_of_aligned, # NEW
        'oth_share_of_aligned': oth_share_of_aligned  # NEW
    }
    
    return adjusted_cago, cago_names_reduced, realloc_data



def transform_alr_matrix(error_matrix, missing_party):
    # Assumed base variables in 5x5 error_matrix (ref = COAL): 
    # [0: ALP, 1: GRN, 2: ON, 3: IND, 4: OTH]
    
    if missing_party == 'COAL':
        # Shift reference to ALP. 
        # Math: z_i = y_i - y_ALP. (Subtract index 0 from all others)
        A = np.array([
            [-1,  1,  0,  0,  0],  # GRN
            [-1,  0,  1,  0,  0],  # ON
            [-1,  0,  0,  1,  0],  # IND
            [-1,  0,  0,  0,  1]   # OTH
        ])
        new_cols = ['GRN', 'ON', 'IND', 'Other']
        ref_col = 'ALP'
        
    elif missing_party == 'ALP':
        # Reference stays COAL. Drop ALP (index 0).
        A = np.array([
            [ 0,  1,  0,  0,  0],  # GRN
            [ 0,  0,  1,  0,  0],  # ON
            [ 0,  0,  0,  1,  0],  # IND
            [ 0,  0,  0,  0,  1]   # OTH
        ])
        new_cols = ['GRN', 'ON', 'IND', 'Other']
        ref_col = 'COAL'
        
    elif missing_party == 'GRN':
        # Reference stays COAL. Drop GRN (index 1).
        A = np.array([
            [ 1,  0,  0,  0,  0],  # ALP
            [ 0,  0,  1,  0,  0],  # ON
            [ 0,  0,  0,  1,  0],  # IND
            [ 0,  0,  0,  0,  1]   # OTH
        ])
        new_cols = ['ALP', 'ON', 'IND', 'Other']
        ref_col = 'COAL'
        
    elif missing_party == 'ON':
        # Reference stays COAL. Drop ON (index 2).
        A = np.array([
            [ 1,  0,  0,  0,  0],  # ALP
            [ 0,  1,  0,  0,  0],  # GRN
            [ 0,  0,  0,  1,  0],  # IND
            [ 0,  0,  0,  0,  1]   # OTH
        ])
        new_cols = ['ALP', 'GRN', 'IND', 'Other']
        ref_col = 'COAL'
        
    elif missing_party == 'IND':
        # Reference stays COAL. Drop IND (index 3).
        A = np.array([
            [ 1,  0,  0,  0,  0],  # ALP
            [ 0,  1,  0,  0,  0],  # GRN
            [ 0,  0,  1,  0,  0],  # ON
            [ 0,  0,  0,  0,  1]   # OTH
        ])
        new_cols = ['ALP', 'GRN', 'ON', 'Other']
        ref_col = 'COAL'
        
    else:
        raise ValueError(f"Unsupported missing party: {missing_party}")

    # Return the mathematically transformed nx4 matrix
    transformed_matrix =  A @ error_matrix @ A.T
    
    return transformed_matrix, new_cols, ref_col



def simulate_combined_models_CAGO(n_simulations, election_year, prior_df_active_dict, polling_avg_dict, National_prior_dict, byelection_group_structure, seat_alr_bases_dict, n_seat_polls_dict, Nat_poll_covm_dict, Electorate_residual_covm_dict, Seat_poll_covm_dict, Farrer_full_prior, GLOBAL_CSVs, prior_alpha=22, B = 1):
    

    polling_avg = polling_avg_dict[election_year]
    prior_df_active = prior_df_active_dict[election_year]
    National_prior = National_prior_dict[election_year]

    # trial using ALP as reference category
    #A_COAL_ref = np.array([
    #    [-1,  0,  0,  0,  0],  # COAL (This is just -ALP)
    #    [-1,  1,  0,  0,  0],  # GRN  (GRN - ALP)
    #    [-1,  0,  1,  0,  0],  # ON   (ON - ALP)
    #    [-1,  0,  0,  1,  0],  # IND  (IND - ALP)
    #    [-1,  0,  0,  0,  1]   # OTH  (OTH - ALP)
    #])
    #Nat_poll_cov_ALP_ref = pd.DataFrame((A_COAL_ref @ Nat_poll_covm_dict[election_year] @ A_COAL_ref.T).values, columns = ['COAL','GRN','ON','IND','OTH'], index = ['COAL','GRN','ON','IND','OTH'])

    nat_alr_sims = simulate_Polling_Fundamentals_model(
        n_simulations=n_simulations,
        election_year=election_year,
        df_t=0,
        Prior_estimates_df=prior_df_active,
        polling_avg = polling_avg,
        National_prior = National_prior,
        National_Polling_error_cov = Nat_poll_covm_dict[election_year], #  Nat_poll_cov_ALP_ref, #
        Electorate_Residuals_cov =  Electorate_residual_covm_dict[election_year],
        For_combined_model=True
    )

    # print(alr_to_simplex_simulation_array(nat_alr_sims))


    combined_sims = {}
    major_realloc_data = {} # for missing major party in byelections

    for i in range(nat_alr_sims.shape[1]): 

        nat_alr_sims_curr = nat_alr_sims[:,i,:]
        div = prior_df_active_dict[election_year].iloc[[i]].index[0][:]
        


        nat_poll_cov = Nat_poll_covm_dict[election_year].copy()
        resid_poll_cov = Electorate_residual_covm_dict[election_year].copy()
        n_seat_polls = n_seat_polls_dict[election_year][div]
        seat_poll_cov = Seat_poll_covm_dict[election_year].copy() * (SEAT_POLL_SCALE[n_seat_polls])



        assert n_seat_polls <= 2

        #seat_poll_cov = Seat_poll_covm_dict[election_year].copy() *SEAT_POLL_SCALE[3] - test for different # seat polls


        # side-quest: diagnose effect of adjusting covms for COAL symmetry to ALP
        #Sigma_nat_5x5 = nat_poll_cov.values + resid_poll_cov.values
        #Inv_Sigma_nat_5x5 = np.linalg.inv(Sigma_nat_5x5)
        #Sigma_seat_5x5 = seat_poll_cov.values
        #Inv_Sigma_seat_weighted_5x5 = B * np.linalg.inv(Sigma_seat_5x5)
        #Sigma_comb_5x5 = np.linalg.inv(Inv_Sigma_nat_5x5 + Inv_Sigma_seat_weighted_5x5)

        #coal_variance_boost = 0.015
        #minor_shrinkage = 1-coal_variance_boost/ np.diag(Sigma_comb_5x5).sum() # 0.959, denom is 0.362
        #adjusted_CovM = (Sigma_comb_5x5 * minor_shrinkage) + coal_variance_boost



        if election_year == 'Byelection':

            # get missing party if exists
            mapping_df = GLOBAL_CSVs['mapping_df']

            # missing_party logic 
            if div in mapping_df['div_nm'].unique():
                # allocate missing party vote

                missing_party = mapping_df.loc[mapping_df['div_nm']==div,['Major_sitout']].iloc[0,0]

                nat_sims = alr_to_simplex_simulation_array(nat_alr_sims_curr)

                cago_names = ['COAL', 'ALP', 'GRN', 'ON','IND','Other'] # CHANGE

                # 1. Apply major reallocation to CAGO and save state
                nat_sims_adj, new_cago_cols, realloc_data = allocate_major_sitout_to_cago(
                    nat_sims, cago_names, div, missing_party, mapping_df , GLOBAL_CSVs['proportions_df'], Farrer_full_prior, MAJOR_SITOUT_DIR_ALPHA
                )

                # transform ALR covms
                nat_poll_cov, new_cols, ref_col = transform_alr_matrix(nat_poll_cov, missing_party)
                resid_poll_cov, new_cols, ref_col = transform_alr_matrix(resid_poll_cov, missing_party)
                seat_poll_cov, new_cols, ref_col = transform_alr_matrix(seat_poll_cov, missing_party)

                # re-convert to alr
                current_cols = [ref_col] + new_cols # ref_col as ALP/COAL always first
                numerator_idxs = [current_cols.index(c) for c in new_cols]
                denominator_idx = current_cols.index(ref_col)

                nat_alr_sims_curr = np.log(nat_sims_adj[:, numerator_idxs] / nat_sims_adj[:, [denominator_idx]]) 

                major_realloc_data[div] = realloc_data

        # adjust GRN vote for by-election

        USE_MEAN_SHIFT = 1

        if (election_year == "Byelection") and USE_MEAN_SHIFT:
            grn_idx = 0 if missing_party == 'ALP' else 1
            nat_alr_sims_curr[:,grn_idx] -= 0.28

        combined_alr = combine_models_alr_precision_weighted(
                n_simulations,
                B,
                nat_alr_sims_curr, # Simulated_Electorate_Polling_Results_ALR for this div (n_sims, 3)
                seat_alr_bases_dict[election_year][div], # alr_base from precomputed_dict (3,)
                nat_poll_cov, # National_Polling_error_cov (3, 3)
                resid_poll_cov,    # Electorate_Residuals_cov (3, 3)
                seat_poll_cov # Seat_poll_covm_dict[election_year] (3, 3)
            )
    
        combined_simplex = alr_to_simplex_simulation_array(combined_alr)
        combined_sims[div] = combined_simplex

    return combined_sims, major_realloc_data



def adjust_prior_to_matrix(
    prior_row, 
    realloc_data, 
    all_party_names
):
    p_group = realloc_data['p_group']
    groups = realloc_data['groups']
    sub = realloc_data['sub']
    missing_party = realloc_data['missing_party']
    n_sims = p_group.shape[0]

    # --- 1. Identify split ratios and group indices for pooling ---
    grn_share = realloc_data['grn_share_of_aligned'] 
    oth_leak_share = realloc_data['oth_share_of_aligned'] 
    aligned_idx = groups.index('aligned') 
    other_idx = groups.index('other') 
    
    # --- 2. Tile the single prior row into a matrix (n_sims, len(all_party_names)) ---
    prior_row_COALified = prior_row.copy()
    prior_row_COALified.columns = ['COAL' if p in ['LP','NP','CLP','LNP'] else p for p in prior_row.columns]
    prior_matrix = np.tile(prior_row_COALified[all_party_names].values, (n_sims, 1)).astype(float)
    
    # --- 3. Extract and remove sitout vote ---
    all_party_names = [p if p not in ['LP','NP','CLP','COAL'] else 'COAL' for p in all_party_names]
    sitout_idx = all_party_names.index(missing_party)
    V = prior_matrix[:, sitout_idx].copy()
    prior_matrix[:, sitout_idx] = 0.0

    # >>> NEW: POOL THE MASS DESTINED FOR THE 'OTH' BUCKET
    # mass_other_pooled = (Natural Other Group Mass) + (45% of Aligned Group Mass)
    mass_other_pooled = (V * p_group[:, other_idx]) + (V * p_group[:, aligned_idx] * oth_leak_share) # NEW
    
    # --- 4. Track the specific IND addition (Keep skeleton for logic) ---
    ind_surge_dict = {}
    
    # --- 5. Allocate to each minor/major party ---
    for g_idx, g in enumerate(groups):
        parties = sub.loc[sub['group'] == g, 'PartyAb'].tolist()
        
        # --- MODIFIED ALLOCATION LOGIC TO FIX INFLATION ---
        if (missing_party == 'ALP') and (g == 'aligned'): # NEW
            # TARGET ONLY THE GREENS: They get their fixed 55.1% surge
            idx = [all_party_names.index('GRN')] # NEW
            weights = np.ones((n_sims, 1)) # NEW
            allocation = (V * p_group[:, g_idx] * grn_share)[:, None] * weights # NEW
            
        elif (missing_party == 'ALP') and (g == 'other'): # NEW
            # POOLING: Include HMP and SPP in the 'Other' group for proportional splitting
            other_aligned = [p for p in ['HMP', 'SPP'] if p in all_party_names] # NEW
            parties = list(set(parties + other_aligned)) # NEW
            
            mapped_parties = ['IND' if p.startswith('IND') else p for p in parties]
            idx = [all_party_names.index(p) for p in mapped_parties if p in all_party_names]
            
            current = prior_matrix[:, idx]
            row_sums = current.sum(axis=1, keepdims=True)
            weights = np.divide(current, row_sums, out=np.full_like(current, 1/len(idx)), where=row_sums > 0)
            
            # Use the POOLED mass (Other + Aligned Leak) instead of just the 'other' group mass
            allocation = mass_other_pooled[:, None] * weights # NEW

        else:
            # Standard logic for IND and Other Major groups (e.g., SFF, ON, FFPA)
            if not parties: continue
            
            mapped_parties = ['IND' if p.startswith('IND') else p for p in parties]
            idx = list(set([all_party_names.index(p) for p in mapped_parties if p in all_party_names]))
            if not idx: continue
            
            current = prior_matrix[:, idx]
            row_sums = current.sum(axis=1, keepdims=True)
            weights = np.divide(current, row_sums, out=np.full_like(current, 1/len(idx)), where=row_sums > 0)
            
            allocation = (V * p_group[:, g_idx])[:, None] * weights

        # Apply the calculated mass to the matrix
        prior_matrix[:, idx] += allocation

        # Track the specific mass added to the exact IND labels
        if g == 'IND':
            total_ind_surge = (V * p_group[:, g_idx])
            new_total_ind = prior_matrix[:, idx[0]]
            surge_ratio = np.divide(total_ind_surge, new_total_ind, out=np.zeros_like(total_ind_surge), where=new_total_ind > 0)
            
            for p in parties:
                ind_surge_dict[p] = surge_ratio / len(parties)

    # --- 6. Finalize and remove sitout party column ---
    prior_matrix = np.delete(prior_matrix, sitout_idx, axis=1)
    all_party_names_reduced = [p for i, p in enumerate(all_party_names) if i != sitout_idx]
    
    return prior_matrix, all_party_names_reduced, ind_surge_dict

def expand_divisions_using_prior(sim, major_realloc_data, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=100):

    final_sim = {}
    party_name_dict = {} 

    multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
    C200_IND_positions_df = pd.read_csv("C200_IND_positions_df.csv", index_col=None)
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]
    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]

    Complex_contests = {'Calare': 0.4788,'Monash': 0.5002,'Moore': 0.43987} # proportions of split to C200 independent and Incumbent independent (based on % after adding Historical proportions in (div,div) pair)



    Major_parties = ['COAL','LP','NP','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']
    if election_year != '2016':
        if not CAGO_only:
            Major_parties += ['ON','UAPP','TOP']
            if election_year != '2025':
                Polling_parties += ['ON','UAPP']
            else:
                Polling_parties += ['ON','TOP']
    if election_year == 'Byelection':
        Major_parties += ['ON','IND']
        Polling_parties += ['ON','IND']


    # For the split of COAL_double_divs
    NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)

    NP_ratios_curr = NP_ratios_curr.loc[(NP_ratios_curr['election_year']==election_year) ,] # & (NP_ratios_curr['State'].isin(['VIC','NSW']))
        

    for i, div in enumerate(Prior_estimates_dict.keys()): # will be alphabetical
        sim_block = sim[div]            # shape (10000, 4)
        main_parties = sim_block[:, :-1]     # shape (10000, 3)
        other_share = sim_block[:, -1]       # shape (10000,)

        prior_row = Prior_estimates_dict[div]

        prior_row_Other = prior_row[[p for p in prior_row.columns if p not in Major_parties]]
        minor_names = list(prior_row_Other.columns)
        all_party_names = Polling_parties + minor_names


        has_matrix = (major_realloc_data is not None) and (div in major_realloc_data)
        if has_matrix:
            prior_matrix, prior_names, ind_surge_dict = adjust_prior_to_matrix(prior_row, major_realloc_data[div], all_party_names)  
            all_party_names = prior_names  
        else:
            prior_matrix,ind_surge_dict = None,None
        #print(div)

        # Extract prior for this division

        if has_matrix:
            other_idx = [prior_names.index(m) for m in minor_names]
            rel_weights = prior_matrix[:, other_idx]
            rel_weights = rel_weights / rel_weights.sum(axis=1, keepdims=True)
        else:
            rel_weights = prior_row_Other.iloc[0].values
            rel_weights = rel_weights / rel_weights.sum()



        #import pdb;pdb.set_trace()

        # Dirichlet sampling
        if has_matrix:
            alphas = rel_weights * alpha_scalar
            splits = np.array([rng.dirichlet(a) for a in alphas]) 

            K = splits.shape[1] # prevent extremely small compositions
            splits = splits * (1 - K * DIR_epsilon) + DIR_epsilon
        else:
            alpha = rel_weights * alpha_scalar
            splits = rng.dirichlet(alpha, size=sim[div].shape[0])

        # Expand 'Other' proportionally
        other_expanded = splits * other_share[:, None]  # (10000, n_minor)

        # Combine with main parties
        combined = np.concatenate([main_parties, other_expanded], axis=1)


        # if no ON/TOP, demove this column!
        if not CAGO_only:
            ON_index, TOP_index = 3, 4
            to_remove_index = []
            if (election_year in ['2019','2022','2025']) and ('ON' not in prior_row.columns):
                all_party_names = [p for p in all_party_names if p != 'ON']
                to_remove_index.append(ON_index)

            elif (election_year =='2025') and ('TOP' not in prior_row.columns):
                all_party_names = [p for p in all_party_names if p != 'TOP']
                to_remove_index.append(TOP_index)

            combined = np.delete(combined, to_remove_index, axis=1) # does nothing if to_remove_index is empty

        #import pdb;pdb.set_trace()


        if div in NP_ratios_curr['div_nm'].unique():
            COAL_votes = combined[:,0]

            if 'COALLP' in prior_row.columns:
                print('COALLP')
                import pdb;pdb.set_trace()

            if has_matrix and ('NP' in prior_names) and ('LP' in prior_names):
                np_col, lp_col = prior_matrix[:, prior_names.index('NP')], prior_matrix[:, prior_names.index('LP')]
                NP_est = np_col / (lp_col + np_col) # (1000,) Array
            else:

                if 'NP' in prior_row.columns and 'LP' in prior_row.columns:
                    NP_est = (prior_row['NP'] /  prior_row[['LP','NP']].sum(axis=1)).iloc[0]
                else:
                    NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate'].iloc[0]

                # current hack for 2025 Nationals:
                if (election_year == '2025') & (div in ['Bullwinkel','Forrest',"O'Connor"]):
                    #import pdb;pdb.set_trace()
                    NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate'].iloc[0]



            alpha = np.array([1-NP_est,NP_est]) * LP_NP_DIR_ALPHA
            splits = rng.dirichlet(alpha, size=sim[div].shape[0])
            
            K = splits.shape[1] # prevent extremely small compositions
            splits = splits * (1 - K * DIR_epsilon) + DIR_epsilon

            LP_NP_votes = splits * COAL_votes[:, None]

            combined = np.concatenate([combined, LP_NP_votes], axis=1)[:,1:] # Removes 'COAL' from 1st column

            all_party_names = all_party_names[1:] + ['LP','NP'] # correct order - 'COAL' is always first in such case

        # perform independent split as well!
        if 'IND' in Prior_estimates_dict[div]:

            IND_index = all_party_names.index('IND')
            IND_votes = combined[:,IND_index] # should be just one
            
            if div in multiple_INDs_df['div_nm'].unique():
                # split according to C200%, remainder evenly! 
                #import pdb;pdb.set_trace()

                num_INDs_curr = multiple_INDs_df.loc[multiple_INDs_df['div_nm'] == div,'No_of_INDs']
                means_array_sim = np.tile(np.full(num_INDs_curr, 1/num_INDs_curr), (len(IND_votes), 1)) # initialises an even split



                if div in C200_IND_positions_curr['div_nm'].unique():
                    # split according to C200 splits!


                    beta0,beta1 = C200_ratio_lm_params
                    logit_mu = beta0 + beta1 * np.log(IND_votes*100)

                    C200_ratio = np.where(IND_votes < 0.05, 0.9, 1 / (1 + np.exp(-logit_mu)))


                    #C200_ratio = C200_IND_splits.loc[C200_IND_splits['Election'] == f'AverageFor{election_year}','Ratio'].iloc[0]

                    C200_IND_position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm'] == div,'Number'].iloc[0]

                    if (election_year == '2025') and div in Complex_contests.keys():

                        C200_ratio = np.full_like(C200_ratio, Complex_contests[div], dtype=float) # an array col
                    
                    K = num_INDs_curr.iloc[0]
                    non_ind_total = 1 - C200_ratio          # (n_sims,)
                    means_array_sim = np.tile((non_ind_total / (K - 1))[:, None],(1, K))
                    means_array_sim[:, C200_IND_position - 1] = C200_ratio

                #splits = rng.dirichlet(means_array* alpha_scalar, size=sim.shape[0])
                splits = np.array([rng.dirichlet(alpha_scalar * row) for row in means_array_sim])

                K = splits.shape[1] # prevent extremely small compositions
                splits = splits * (1 - K * DIR_epsilon) + DIR_epsilon


                if has_matrix and ind_surge_dict is not None:
                    # Sum ratios across all specific IND keys
                    total_surge_ratio = np.sum(list(ind_surge_dict.values()), axis=0)
                    organic_ind = IND_votes * (1 - total_surge_ratio)

                    All_IND_votes = splits * organic_ind[:, np.newaxis]
                    
                    # Inject 100% of the surge into the primary Independent (C200 position)
                    generated_ind_names = ['IND'+str(j) for j in range(1, num_INDs_curr.iloc[0] + 1)]
                    for idx, label in enumerate(generated_ind_names):
                        if label in ind_surge_dict:
                            # Add the ratio-based surge amount
                            All_IND_votes[:, idx] += (IND_votes * ind_surge_dict[label])
                else:
                    All_IND_votes = splits * IND_votes[:, None]

                combined = np.concatenate([combined, All_IND_votes], axis=1)
                combined = np.delete(combined,IND_index, axis = 1) # remove 'IND' col

                all_party_names = all_party_names + ['IND'+str(i) for i in range(1,num_INDs_curr.iloc[0]+1)]
                all_party_names = [p for p in all_party_names if p != 'IND'] # remove IND

            else:
                all_party_names = [p if p != 'IND' else 'IND1' for p in all_party_names] # renames to IND1 if only single IND


        #assert combined.min() > 1e-40


                            
        # map the order to the final ballot order
        COAL_replacement_list = ['LP','NP','LNP','CLP']
        Ballot_order = Results_dict[div].columns.tolist()



        if 'COAL' in all_party_names:

            COAL_replacement = [p for p in Ballot_order if p in COAL_replacement_list]

            if len(COAL_replacement) > 1:
                print('missed COAL double div')
                import pdb;pdb.set_trace()

            all_party_names = [COAL_replacement[0] if p == 'COAL' else p for p in all_party_names]


        name_to_idx = {name: i for i, name in enumerate(all_party_names)}
        col_indices = [name_to_idx[name] for name in Ballot_order]

        if np.isnan(combined).any():
            import pdb;pdb.set_trace()

        # Store results
        final_sim[div] = combined[:, col_indices]
        party_name_dict[div] = Ballot_order  # add names to avoid confusion in future!

    return final_sim, party_name_dict




def expand_divisions_using_seat_polls(
    sims,
    curr_seat_polls,
    Prior_estimates_dict,
    Results_dict_active,
    election_year,
    prior_alpha=22,
    poll_alpha = 22,
    major_realloc_data = {},
    GLOBAL_CSVs = {},
    n_simulations = 1000
):
    
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]
    C200_IND_positions_df = GLOBAL_CSVs["C200_IND_positions_df"]
    NP_ratios_curr = GLOBAL_CSVs["NP_ratios_curr"]
    NP_ratios_curr = NP_ratios_curr.loc[NP_ratios_curr['election_year'] == election_year]
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]

    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    beta0,beta1 = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]

    Major_parties = ['COAL','LP','NP','NAT','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']

    Major_parties += ['ON','IND']
    Polling_parties += ['ON','IND']


    seat_output_dict = {}
    party_name_dict = {}
    
    for div in sims.keys():

        prior_row = Prior_estimates_dict[div]
        prior_row.index = [div]


        prior_row_Other = prior_row[[p for p in prior_row.columns if p not in Major_parties]]
        minor_names = list(prior_row_Other.columns)
        all_party_names = Polling_parties + minor_names

        has_matrix = (major_realloc_data is not None) and (div in major_realloc_data)
        if has_matrix:
            adjusted_prior = adjust_prior_to_matrix(prior_row, major_realloc_data[div], all_party_names)
            prior_matrix, prior_names,ind_surge_dict = adjusted_prior
        else:
            prior_matrix, prior_names, ind_surge_dict = None, None, None

        missing_party = major_realloc_data[div]['missing_party'] if div in major_realloc_data.keys() else None

        ref_col='COAL' if missing_party != 'COAL' else 'ALP'
        sims_cols = [p for p in ['COAL','ALP','GRN','ON','IND','Other'] if p != missing_party] # [p for p in ['COAL','ALP','GRN','Other'] if p != missing_party]
    
        poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate']==div].set_index('Electorate')
        Results_row = Results_dict_active[div]

        
        multi_ind_c200_case = ((div in multiple_INDs_df['div_nm'].values) and (div in C200_IND_positions_curr['div_nm'].values))
        Position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm']==div,'Number'].iloc[0] if multi_ind_c200_case else 1
        k = int(multiple_INDs_df.loc[multiple_INDs_df['div_nm'] == div,'No_of_INDs'].iloc[0]) if div in multiple_INDs_df['div_nm'].unique() else 1

        # compute CAGO_row
        prior_df = prior_row.copy()
        minor_cols = [p for p in prior_df.columns if p not in Major_parties]

        curr_row = poll_df.iloc[:,2+('byelection_year' in poll_df.columns):]
        curr_row_normalised = curr_row/curr_row.sum(axis=1)[div]
        CAGO_row = group_into_Fundamentals_Categories(curr_row_normalised, div).mean()

        # add alr error and return to prop space, accounting for missing major party
        if missing_party is not None:
            CAGO_row = CAGO_row.drop([missing_party])

        poll_minor_split = curr_row.mean().loc[[p for p in curr_row.mean().index if p not in Major_parties]]     
        poll_minor_split = poll_minor_split[poll_minor_split>0]   
        poll_weights = poll_minor_split / poll_minor_split.sum()
        
        adjusted_cago = sims[div]
        cago_party_names = sims_cols
        oth_idx = cago_party_names.index('Other') # CHECK if it should be 'Other'
        
        macro_oth = adjusted_cago[:, oth_idx]


        alpha_curr = (poll_weights.values * poll_alpha).astype(float)
        poll_draws = rng.dirichlet(alpha_curr, size=n_simulations)

        K = poll_draws.shape[1] # prevent extremely small compositions
        poll_draws = poll_draws * (1 - K * DIR_epsilon) + DIR_epsilon
        
        poll_alloc_matrix = poll_draws * macro_oth[:, np.newaxis]

        poll_names = list(poll_weights.index)
        unpolled_minor_cols = [p for p in minor_cols if p not in poll_names]

        
        

        if has_matrix:
            prior_ind_remainder = prior_matrix[:, prior_names.index('IND')]
        else:
            # Safe access to prior_df; if 'IND' isn't there, default to 0.0
            prior_ind_remainder = prior_df.loc[div, 'IND'] if 'IND' in prior_df.columns else 0.0

        if multi_ind_c200_case:

            if 'IND' in poll_names:
                import pdb; pdb.set_trace()

                ind_idx = poll_names.index('IND')
                
                # CHANGE: Pull IND prior from matrix if available
                if has_matrix:
                    prior_ind_val = prior_matrix[:, prior_names.index('IND')]
                else:
                    prior_ind_val = prior_df.loc[div, 'IND']
                
                # Use vectorized np.where for safe calculation
                safe_ind_val = np.maximum(prior_ind_val * 100, 1e-9)
                logit_mu = beta0 + beta1 * np.log(safe_ind_val)
                c200_ratio = np.where(prior_ind_val < 0.05, 0.9, 1 / (1 + np.exp(-logit_mu)))

                # 3. CHANGE: Instead of updating the DataFrame (which breaks on vectors),
                # we create a local adjusted prior for the unpolled IND expansion logic
                # This represents the 'baseline' or 'minor' IND weight
                prior_ind_remainder = prior_ind_val * (1 - c200_ratio)
                
                # 4. Update the poll names and matrix
                # The polled 'IND' bucket is renamed to the specific C200 candidate (e.g. IND1)
                poll_names[ind_idx] = f'IND{Position}'

                if 'IND' not in unpolled_minor_cols:
                    unpolled_minor_cols.append('IND')


        if 'OTH' in poll_names:
            oth_in_poll_idx = poll_names.index('OTH')
            remaining_oth = poll_alloc_matrix[:, oth_in_poll_idx]
            
            if len(unpolled_minor_cols) > 0 and np.any(remaining_oth > 0):

                if has_matrix:
                    # Build matrix from prior_matrix
                    prior_minors = prior_matrix[:, [prior_names.index(p) for p in unpolled_minor_cols]]
                    # Override the IND column with our 'remainder' logic
                    if 'IND' in unpolled_minor_cols:
                        ind_col_idx = unpolled_minor_cols.index('IND')
                        prior_minors[:, ind_col_idx] = prior_ind_remainder
                else:
                    # Scalar logic
                    prior_minors = prior_df.loc[div, unpolled_minor_cols].copy()
                    if 'IND' in unpolled_minor_cols:
                        prior_minors['IND'] = prior_ind_remainder

                #prior_minors = prior_df.loc[div, unpolled_minor_cols]
                #prior_weights = prior_minors / prior_minors.sum()

                minors_arr = getattr(prior_minors, 'values', prior_minors).astype(float)

                if minors_arr.ndim == 2:
                    row_sums = minors_arr.sum(axis=1, keepdims=True)
                    weights_arr = np.divide(minors_arr, row_sums, out=np.full_like(minors_arr, 1/minors_arr.shape[1]), where=row_sums>0)
                else:
                    row_sum = minors_arr.sum()
                    weights_arr = minors_arr / row_sum if row_sum > 0 else np.full_like(minors_arr, 1/len(minors_arr))

                alpha_prior = weights_arr * prior_alpha
                #alpha_prior = (prior_weights.values * prior_alpha).astype(float)
                
                if has_matrix:
                    draw_prior = np.array([rng.dirichlet(a) for a in alpha_prior])
                    K = draw_prior.shape[1] # prevent extremely small compositions
                    draw_prior = draw_prior * (1 - K * DIR_epsilon) + DIR_epsilon
                else:
                    draw_prior = rng.dirichlet(alpha_prior, size=n_simulations)
                #draw_prior = rng.dirichlet(alpha_prior, size=n_simulations)
                prior_alloc_matrix = draw_prior * remaining_oth[:, np.newaxis]
                
                poll_alloc_no_oth = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
                poll_names_no_oth = [p for p in poll_names if p != 'OTH']
                
                minor_party_block = np.hstack([poll_alloc_no_oth, prior_alloc_matrix])
                all_minor_names = poll_names_no_oth + unpolled_minor_cols
            else:
                minor_party_block = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
                all_minor_names = [p for p in poll_names if p != 'OTH']
        else:
            minor_party_block = poll_alloc_matrix
            all_minor_names = poll_names

        main_party_block = np.delete(adjusted_cago, oth_idx, axis=1)
        main_party_names = [p for p in cago_party_names if p != 'Other']
        #main_party_names =  ['IND1' if p == 'IND' else p for p in cago_party_names] # Farrer specific


        final_array = np.hstack([main_party_block, minor_party_block])
        all_party_names = main_party_names + all_minor_names

        assert np.isclose(final_array.sum(), n_simulations)


        if ('LP' in Results_row.columns) and ('NP' in Results_row.columns):
            NP_LP_split_polled = ('NAT' in poll_df.columns) and poll_df['NAT'].sum() > 0

            if NP_LP_split_polled:
                nat = poll_df.loc[div,'NAT'].astype(float)
                lib = poll_df.loc[div,'COAL'].astype(float)
                
                coal_total = nat + lib
                NP_est = nat / coal_total if coal_total > 0 else 0.5

            else:
                if has_matrix and ('NP' in prior_names) and ('LP' in prior_names):
                    np_col = prior_matrix[:, prior_names.index('NP')]
                    lp_col = prior_matrix[:, prior_names.index('LP')]
                    # Safe division to prevent NaNs
                    NP_est = np.divide(np_col, (lp_col + np_col), 
                                       out=np.full_like(np_col, 0.5), 
                                       where=(lp_col + np_col) > 0)
                
                # --- Standard Prior/Estimate Logic ---
                elif div in NP_ratios_curr['div_nm'].unique():
                    if 'NP' in prior_df.columns and 'LP' in prior_df.columns:
                        denom = prior_df.loc[div, ['LP', 'NP']].sum()
                        NP_est = prior_df.loc[div, 'NP'] / denom if denom > 0 else 0.5
                    else:
                        NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm'] == div, 'final_estimate'].iloc[0]
                else:
                    NP_est = 0.5

            coal_idx = all_party_names.index('COAL')
            coal_votes = final_array[:, coal_idx]

            if isinstance(NP_est, np.ndarray):
                alphas_lp_np = np.vstack([1 - NP_est, NP_est]).T * LP_NP_DIR_ALPHA_POLL# LP_NP_DIR_ALPHA *poll_alpha/prior_alpha # scale by their ratio
                lp_np_splits = np.array([rng.dirichlet(a) for a in alphas_lp_np])

                K = lp_np_splits.shape[1] # prevent extremely small compositions
                lp_np_splits = lp_np_splits * (1 - K * DIR_epsilon) + DIR_epsilon
            else:
                alpha_lp_np = np.array([1 - NP_est, NP_est]) * LP_NP_DIR_ALPHA_POLL # LP_NP_DIR_ALPHA *poll_alpha/prior_alpha
                lp_np_splits = rng.dirichlet(alpha_lp_np, size=n_simulations)

            
            lp_votes = lp_np_splits[:, 0] * coal_votes
            np_votes = lp_np_splits[:, 1] * coal_votes

            final_array = np.delete(final_array, coal_idx, axis=1)
            all_party_names.pop(coal_idx)

            final_array = np.hstack([final_array, lp_votes[:, np.newaxis], np_votes[:, np.newaxis]])
            all_party_names += ['LP', 'NP']



        
        
        if 'IND' in all_party_names: # Farrer specific
            ind_idx = all_party_names.index('IND')
            ind_total = final_array[:, ind_idx]

            if multi_ind_c200_case and div != 'Farrer':
                free_weights = np.full(k - 1, 1 / (k - 1))
                ind_draws = rng.dirichlet(free_weights * prior_alpha, size=n_simulations)

                K = ind_draws.shape[1] # prevent extremely small compositions
                ind_draws = ind_draws * (1 - K * DIR_epsilon) + DIR_epsilon

                ind_split_votes = ind_draws * ind_total[:, np.newaxis]

                final_array = np.delete(final_array, ind_idx, axis=1)
                all_party_names.pop(ind_idx)

                idx = 0
                for i in range(1, k + 1):
                    if i == Position:
                        continue
                    label = f'IND{i if i < Position else i}'
                    final_array = np.hstack([final_array, ind_split_votes[:, [idx]]])
                    all_party_names.append(label)
                    idx += 1

                    
            elif election_year == 'Byelection' and div == 'Wentworth':
                all_party_names[ind_idx] = 'IND3'

            else:
                if div in multiple_INDs_df['div_nm'].unique():
                    means_array = np.full(k, 1 / k)

                    if div in C200_IND_positions_curr['div_nm'].unique():
                        pos = int(C200_IND_positions_curr.loc[
                            C200_IND_positions_curr['div_nm'] == div, 'Number'
                        ].iloc[0]) - 1

                        c200_ratios = np.where(
                            ind_total * 100 < 5, 
                            0.9, 
                            1 / (1 + np.exp(-(beta0 + beta1 * np.log(ind_total * 100))))
                        )

                        means_matrix = np.tile((1 - c200_ratios)[:, np.newaxis] / (k - 1), (1, k))
                        means_matrix[:, pos] = c200_ratios
                        
                    else:
                        means_matrix = np.tile(means_array, (n_simulations, 1))
                        pos = 0 # SAFE FALLBACK: If no C200 candidate exists, default to IND1

                    splits = np.array([rng.dirichlet(m * prior_alpha) for m in means_matrix])

                    K = splits.shape[1] # prevent extremely small compositions
                    splits = splits * (1 - K * DIR_epsilon) + DIR_epsilon
                    

                    ind_split_votes = splits * ind_total[:, np.newaxis] # Farrer specific - just use c200 ratio
                    
                    final_array = np.delete(final_array, ind_idx, axis=1)
                    all_party_names.pop(ind_idx)

                    for i in range(k):
                        final_array = np.hstack([final_array, ind_split_votes[:, [i]]])
                        all_party_names.append(f'IND{i+1}')

                else:
                    all_party_names[ind_idx] = 'IND1'

        #assert final_array.min() > 1e-40

        # Final Coalition Renaming
        if ('COAL' in all_party_names) and (missing_party != 'COAL'):
            coal_idx = all_party_names.index('COAL')
            coal_replacement_list = ['LP', 'NP', 'LNP', 'CLP']
            coal_party = next(p for p in Results_row.columns if p in coal_replacement_list)
            all_party_names[coal_idx] = coal_party

        # Final mapping to match Results_row column order
        ballot_order = Results_row.columns.tolist()
        name_to_idx = {name: i for i, name in enumerate(all_party_names)}
        col_indices = [name_to_idx[name] for name in ballot_order]


        seat_output_dict[div] = final_array[:, col_indices]

        party_name_dict[div] = ballot_order  # add names to avoid confusion in future!
    
    return seat_output_dict, party_name_dict




def Farrer_byelection_adjusted_avg_by_pp_id(election_year, prior_df_active_dict, polling_avg_dict,
        National_prior_dict, Prior_estimates_dict_active, seat_alr_bases_dict):
    
    

    def process_booth_priors(csv_path="Farrer_byelection_Fundamentals_by_pp_id.csv"):
        # Load the DataFrame and set pp_id as the index
        df_booths = pd.read_csv(csv_path)
        
        # Ensure pp_id is the index if it isn't already
        if 'pp_id' in df_booths.columns:
            df_booths = df_booths.set_index('pp_id')
            
        # Drop INFORMAL to isolate the formal vote columns
        if 'INFORMAL' in df_booths.columns:
            formal_df = df_booths.drop(columns=['INFORMAL'])
        else:
            formal_df = df_booths.copy()

        total_formal_votes = formal_df.sum(axis=1)
        props_df = formal_df.div(total_formal_votes, axis=0)

        cago_cols = ['COAL', 'ALP', 'GRN', 'ON', 'IND']
        booth_prior_cago = props_df[cago_cols].copy()

        # Calculate 'OTH' exactly as you did for the macro prior: 1 - sum(Big 5)
        # This vectorizes your `.iloc[0]` logic to apply to all 86 booths simultaneously
        booth_prior_cago['OTH'] = 1.0 - booth_prior_cago.sum(axis=1)
        
        # Rename columns to match 'PartyAb' structure if needed
        booth_prior_cago.columns.name = 'PartyAb'

        return props_df, booth_prior_cago, total_formal_votes

    # get 
    booth_prior_estimates_df, booth_prior_active, saved_booth_totals = process_booth_priors()
    pp_id_list = booth_prior_estimates_df.index

    # calculate poll shift in alr

    polling_alr = df_to_alr(polling_avg_dict[election_year], ref_col='COAL')
    nat_prior_alr = df_to_alr( National_prior_dict[election_year].rename(index = {0:'Farrer'}), ref_col = 'COAL')
    nat_poll_shift = polling_alr - nat_prior_alr

    prior_active_alr = df_to_alr(prior_df_active_dict[election_year], ref_col='COAL')

    poll_shift_simplex = alr_to_simplex_vectorized(prior_active_alr + nat_poll_shift, ref_col='COAL')


    Prior_active_alr = df_to_alr(prior_df_active_dict[election_year], ref_col='COAL')
    prior_full = Prior_estimates_dict_active['Farrer']
    prior_GRN_prop_aligned = (prior_full['GRN']/(prior_full[['SPP','GRN','HMP']].sum(axis=1)) ).iloc[0]
    prior_ON_prop_other = (poll_shift_simplex['ON']/( poll_shift_simplex[['ON','OTH']].sum(axis=1)) ).iloc[0]

    def allocate_ALP_Farrer(composition, prior_full):

        # calculate seat poll shift in alr - first remove ALP from fundamentals result
        ALR_reallocation_df = pd.read_csv('Missing_major_reallocation_proportions.csv').iloc[1,:]
        prior_GRN_prop_aligned = (prior_full['GRN']/(prior_full[['SPP','GRN','HMP']].sum(axis=1)) ).iloc[0]
        prior_ON_prop_other = (poll_shift_simplex['ON']/( poll_shift_simplex[['ON','OTH']].sum(axis=1)) ).iloc[0]

        ALP_vote = composition.loc['Farrer','ALP']
        composition['GRN'] += ALR_reallocation_df['theta_aligned']*prior_GRN_prop_aligned*ALP_vote
        composition['IND'] += ALR_reallocation_df['theta_ind']*ALP_vote
        composition['COAL'] += ALR_reallocation_df['theta_other_major']*ALP_vote
        composition['ON'] += ALR_reallocation_df['theta_other']*ALP_vote * prior_ON_prop_other
        composition['OTH'] += ALR_reallocation_df['theta_other']*ALP_vote * (1-prior_ON_prop_other) + ALR_reallocation_df['theta_aligned']*(1-prior_GRN_prop_aligned)*ALP_vote
        composition = composition.drop(columns=['ALP'])
        assert np.isclose(composition.sum(axis=1).iloc[0],1)

        return composition

    prior_seat_result = allocate_ALP_Farrer(prior_df_active_dict[election_year], prior_full)
    

    prior_seat_alr = df_to_alr(prior_seat_result, ref_col='COAL')
    seat_poll_alr = pd.DataFrame( [seat_alr_bases_dict[election_year]['Farrer']], columns = ['GRN','ON','IND','OTH'], index = ['Farrer'])
    seat_poll_shift = seat_poll_alr - prior_seat_alr
    


    # 
    average_per_pp_id = []
    for pp_id in pp_id_list:
        Prior_estimates_df = booth_prior_estimates_df.loc[booth_prior_estimates_df.index==pp_id]
        booth_prior_active_df =booth_prior_active.loc[booth_prior_active.index==pp_id]

        #CHECK = 0

        # fix 0 GRN vote:
        if booth_prior_active_df.min(axis=1).iloc[0] == 0:
            zero_cols = booth_prior_active_df.loc[:,booth_prior_active_df.iloc[0] == 0]
            for col in zero_cols:
                booth_prior_active_df.loc[:,col] = 0.001
                if col == 'OTH':
                    import pdb; pdb.set_trace()
                booth_prior_active_df.loc[:,'OTH'] -= 0.001
            #CHECK = 1
            #import pdb; pdb.set_trace()


        # apply nat poll shift in 6d alr, and seat poll shift in 5d alr
        prior_alr = df_to_alr(booth_prior_active_df, ref_col='COAL')
        prior_without_ALP = allocate_ALP_Farrer(booth_prior_active_df.rename(index={pp_id: 'Farrer'}), prior_full)
        prior_alr_without_ALP = df_to_alr(prior_without_ALP, ref_col='COAL')

        seat_poll_CGOIO = alr_to_simplex_vectorized(prior_alr_without_ALP.values + seat_poll_shift, ref_col='COAL')
        
        nat_poll_adjustment = pd.Series(0.0, index=prior_alr.columns)
        nat_poll_adjustment['GRN'] = -0.28
        nat_poll_CAGOIO = alr_to_simplex_vectorized(prior_alr.values + nat_poll_shift + nat_poll_adjustment, ref_col='COAL')

        
        nat_poll_CGOIO = allocate_ALP_Farrer(nat_poll_CAGOIO, prior_full)

        # In 5d alr, weight them according to the parties' weighting
        party_seat_poll_weights = {'GRN': 0.544,'ON':0.452,'IND':0.634,'OTH':0.610}
        w_seat = pd.Series(party_seat_poll_weights)
        w_nat = 1.0 - w_seat
        alr_cols = w_seat.index 

        SEAT_ONLY = 1

        if SEAT_ONLY:
            w_seat = 1.0 - pd.DataFrame([np.array([0,0,0,0])], columns = ['GRN','ON','IND','OTH']).mean()
            w_nat = 1.0 - w_seat

        nat_poll_CGOIO_alr = df_to_alr(nat_poll_CGOIO, ref_col='COAL')
        seat_poll_CGOIO_alr = df_to_alr(seat_poll_CGOIO, ref_col='COAL')

        weighted_CAGOIO_alr = (seat_poll_CGOIO_alr[alr_cols] * w_seat) + (nat_poll_CGOIO_alr[alr_cols] * w_nat)
        weighted_CAGOIO_props = alr_to_simplex_vectorized(weighted_CAGOIO_alr, ref_col='COAL')[['COAL','GRN','ON','IND','OTH']]

        # apply expansion logic - prior
        row = weighted_CAGOIO_props.mean().copy()

        C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
        C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]

        beta0,beta1 = C200_ratio_lm_params
        logit_mu = beta0 + beta1 * np.log(row['IND']*100)

        C200_ratio = 1 / (1 + np.exp(-logit_mu))

        row['IND2'] = row['IND'] * (1-C200_ratio)/2
        row['IND3'] = row['IND'] * (1-C200_ratio)/2
        row['IND1'] = row['IND'] * C200_ratio
        row = row.drop('IND')

        # --- Split OTH across remaining minor parties (excluding majors + ON + IND) ---
        prior = Prior_estimates_df.mean()
        exclude = ['ALP', 'COAL', 'GRN', 'ON', 'IND']
        oth_parties = [p for p in prior.index if p not in exclude]

        oth_weights = prior[oth_parties] / prior[oth_parties].sum()
        for p in oth_parties:
            row[p] = row.get(p, 0) + row['OTH'] * oth_weights[p]
        row = row.drop('OTH')

        np_share = 0.220224 # --- Split COAL into LP and NP ---


        if SEAT_ONLY:
            np_share = 0.44 # actual results # DELETE


        row['NP'] = row['COAL'] * np_share
        row['LP'] = row['COAL'] * (1 - np_share)
        row = row.drop('COAL')

        result_prior_alloc = row.to_frame().T # --- Final result as DataFrame ---


        # allocation using seat poll as primary evidence
        row = weighted_CAGOIO_props.mean().copy()

        row['IND2'] = row['IND'] * (1-C200_ratio)/2
        row['IND3'] = row['IND'] * (1-C200_ratio)/2
        row['IND1'] = row['IND'] * C200_ratio
        row = row.drop('IND')

        # --- Split OTH across remaining minor parties - first by whether they're polled, then by original

        macro_oth = row['OTH']
        row = row.drop('OTH').copy()

        poll_result = pd.read_csv('SeatPollAverageFarrerByelectionFormatted.csv')

        poll_minors = poll_result[['SPP','FFPA','OTH']].mean()

        # 1. Allocate the CAGO 'OTH' mass to the minors specifically named in the poll
        poll_weights = poll_minors / poll_minors.sum()
        for p in poll_weights.index:
            row[p] = macro_oth * poll_weights[p]

        # 2. If the poll left a generic 'OTH' bucket, expand it using the unpolled prior minors
        if 'OTH' in row:
            unpolled_mass = row['OTH']
            row = row.drop('OTH')
            
            # Find minors in the prior that weren't specifically polled
            exclude = ['ALP', 'COAL', 'GRN', 'ON', 'IND'] + list(row.index)
            unpolled_cols = [p for p in prior.index if p not in exclude]

            prior = Prior_estimates_df.mean()

            if unpolled_cols and prior[unpolled_cols].sum() > 0:
                prior_weights = prior[unpolled_cols] / prior[unpolled_cols].sum()
                for p in unpolled_cols:
                    # .get(p, 0) is a safe-catch, though exclude should prevent overlaps
                    row[p] = row.get(p, 0) + (unpolled_mass * prior_weights[p])


        np_poll = poll_result['NAT'].iloc[0]
        lp_poll = poll_result['COAL'].iloc[0]

        np_share = np_poll/(np_poll+lp_poll)

        if SEAT_ONLY:
            np_share = 0.44 # Final result # DELETE

        row['NP'] = row['COAL'] * np_share
        row['LP'] = row['COAL'] * (1 - np_share)
        row = row.drop('COAL')

        result_poll_alloc = row.to_frame().T 


        final_average = result_poll_alloc * 0.52 + result_prior_alloc * 0.48

        average_per_pp_id.append(final_average.rename(index = {0:pp_id}))

        #if CHECK:
        #    print(pp_id)
        #    print(Prior_estimates_df[['ASP','FFPA','GRN','GRPF','HMP','IND','COAL','ON','SPP','ALP']])
        #    print(final_average)
        #    import pdb; pdb.set_trace()

    avg_df = pd.concat(average_per_pp_id)



    weighted_avg = (avg_df.mul(saved_booth_totals, axis=0).sum(axis=0) / saved_booth_totals.sum()) # for computing adjustments

    import pdb; pdb.set_trace()

    non_low_grn_rows = avg_df.loc[avg_df['GRN']>=0.01,].index

    if not SEAT_ONLY:

        avg_df.loc[non_low_grn_rows,'GRN']-=0.002 # DISPARITY BETWEEN RESULTS
        avg_df.loc[non_low_grn_rows,'LP']-=0.003
        avg_df.loc[non_low_grn_rows,'IND1']-=0.002
        avg_df.loc[non_low_grn_rows,'ON']+=0.007
    else:
        avg_df.loc[non_low_grn_rows,'GRN']-=0.001 # DISPARITY BETWEEN RESULTS
        avg_df.loc[non_low_grn_rows,'HMP']+=0.005
        avg_df.loc[non_low_grn_rows,'GRPF']-=0.004
        avg_df.loc[non_low_grn_rows,'IND1']-=0.003
        avg_df.loc[non_low_grn_rows,'ON']+=0.003

    # comparison to prior
    #formal_df = pd.read_csv("Farrer_byelection_Fundamentals_by_pp_id.csv").set_index('pp_id').drop(columns=['INFORMAL'])
    #total_formal_votes = formal_df.sum(axis=1)
    #props_df = formal_df.div(total_formal_votes, axis=0)
    

    import folium

    # make the map!

    PP_data = pd.read_csv("2025_PP_data.csv")
    PP_data_Farrer = PP_data.loc[PP_data['div_nm']=='Farrer',]
    PP_data_Farrer.loc[PP_data_Farrer['pp_nm']=='Henty',['Lat','Long']] = -35.51695524516439, 147.03113745176006
    
    



    # 1. Execute your merge
    merged_df = avg_df.merge(PP_data_Farrer[['pp_id', 'pp_nm', 'Lat', 'Long']], left_index=True, right_on='pp_id').set_index('pp_id')

    merged_df = merged_df.merge(saved_booth_totals.rename('Total votes'), left_index=True, right_index=True)

    def make_Farrer_booth_map(merged_df):

        colour_df = pd.read_csv("Party Colours.csv")
        colour_df.loc[:,'Party'] = colour_df['Party'].str.replace("’", "'", regex=False)
        colors = colour_df.set_index("Party")["Colour"].to_dict()
        colors['Ind. Milthorpe'] = colors['Independent']

        party_name_dict = {
            'NP': 'National', 'COAL': 'Coalition', 'ON' : 'One Nation',
            'GRN' : 'Greens', 'HMP' : 'Legalise Cannabis', 'FFPA' : 'Family First',
            'ASP' : 'Shooters Fishers & Farmers', 'LP' : 'Liberal' , 'IND1': 'Ind. Milthorpe', 'OTH': 'Other'
        }

        raw_2025_df = pd.read_csv('2025HouseStateFirstPrefsByPollingPlace-NSW.csv', skiprows=1).rename(columns={'PollingPlaceID':'pp_id'})
        raw_2025_df = raw_2025_df.loc[raw_2025_df['DivisionNm']=='Farrer',][['pp_id','PartyAb','OrdinaryVotes']]

        # get 2025 results per booth:
        df_formal = raw_2025_df[raw_2025_df['PartyAb'].notna()].copy()

        # get combined other FP votes
        Other_votes_df = pd.read_csv("2025FirstPrefsByPPComplete.csv")
        Other_votes_df = Other_votes_df.loc[(Other_votes_df['div_nm']=='Farrer') & (Other_votes_df['Booth_type']=='Other'),['pp_id','PartyAb','votes']].iloc[:-1,].rename(columns={'votes':'OrdinaryVotes'})
        df_formal = pd.concat([Other_votes_df, df_formal], ignore_index=True)

        # 2. Ensure we have a clean integer pp_id column
        # If your column is literally "746 Albury", extract just the number:
        # df_formal['pp_id'] = df_formal['PollingPlace'].str.extract(r'(\d+)').astype(int)
        # If it's already a separate ID column, just ensure it's an int:
        # 3. Pivot the table so parties are columns and booths are rows
        df_2025_votes = df_formal.pivot_table(index='pp_id',columns='PartyAb',values='OrdinaryVotes', aggfunc='sum', fill_value=0)

        
        # 4. Standardize the IND and Minor Party labels to match your projection dataframe
        if 'IND' in df_2025_votes.columns:
            df_2025_votes = df_2025_votes.rename(columns={'IND': 'IND1'})
        if 'FFP' in df_2025_votes.columns:
            df_2025_votes = df_2025_votes.rename(columns={'FFP': 'FFPA'}) # Match your previous map_df

        # 5. Convert raw votes to formal percentages per booth
        df_2025_pct = df_2025_votes.div(df_2025_votes.sum(axis=1), axis=0)

        # 6. Add a suffix to prevent column name collisions with your projections!
        df_2025_pct = df_2025_pct.add_suffix('_2025')

        os.chdir("/home/dania-freidgeim/Australian Election/Making Maps")

        ced_25 = gpd.read_file(
            "ASGS_Ed3_Non_ABS_Structures_GDA2020_updated_2025.gpkg",
            layer="CED_2025_AUST_GDA2020"
        )

        # Isolate Farrer
        farrer_boundary = ced_25[ced_25["CED_NAME_2025"].str.contains("Farrer", case=False, na=False)]

        # CRITICAL FOR FOLIUM: Convert from GDA2020 to standard web WGS84 (Lat/Lon)
        farrer_boundary = farrer_boundary.to_crs(epsg=4326)


        # 2. Clean the spatial data 
        # Drop 'Other' (pp_id 0) or any booth missing coordinates so Folium doesn't crash
        #map_df = merged_df.dropna(subset=['Lat', 'Long']).copy()

        # Instead, add coordinates for 'Other' ball
        merged_df.loc[merged_df.index==0,['Lat','Long']] = -33.268889, 146.905374
        map_df = merged_df.join(df_2025_pct)

        # 3. Initialize the Map
        # Center it dynamically based on the average coordinates of all Farrer booths
        center_lat = map_df['Lat'].mean()
        center_lon = map_df['Long'].mean()

        m = folium.Map(tiles="cartodbpositron", zoom_snap=0.1, zoom_delta = 0.5)
        # Get the bounding box of the Farrer electorate
        # total_bounds returns [min_lon, min_lat, max_lon, max_lat]
        bounds = farrer_boundary.total_bounds 

        # Folium expects [[min_lat, min_lon], [max_lat, max_lon]]
        m.fit_bounds([[bounds[1]-0.3, bounds[0]], [bounds[3], bounds[2]]])



        # Add Farrer boundaries
        folium.GeoJson(
            farrer_boundary,
            name="Farrer 2025 Boundary",
            style_function=lambda feature: {
                'color': 'black',     # Make the outline bold black
                'weight': 3,          # Thickness of the line
                'fillOpacity': 0,     # Keep the inside completely transparent so you can see towns
            }
        ).add_to(m)

        # Optional: Add your LGA boundaries if you have the GeoDataFrame loaded from earlier
        # try:
        #     folium.GeoJson(farrer_lgas.to_crs(epsg=4326), name="LGAs").add_to(m)
        # except NameError:
        #     pass # Skips if farrer_lgas isn't defined yet

        # 4. Define the parties for the tooltip
        # We will show these explicitly, and sum the rest into 'OTH'
        top_parties = ['ON', 'IND1', 'LP', 'NP', 'GRN']
        # Define parties to list explicitly
        all_parties = [col for col in map_df.columns if col not in ['pp_nm', 'Lat', 'Long', 'Total votes']]
        minor_parties = [p for p in all_parties if ((p not in top_parties) and not p.endswith('2025'))]

        for idx, row in map_df.iterrows():
    
            # 1. ISOLATE PROJECTION COLUMNS (This stops the black dot bug!)
            proj_parties = [
                col for col in map_df.columns 
                if not str(col).endswith('_2025') 
                and col not in ['pp_nm', 'Lat', 'Long', 'Total votes']
            ]
            
            # 2. DETERMINE THE WINNER USING ONLY PROJECTION COLUMNS
            winning_PartyAb = row[proj_parties].astype(float).idxmax()
            bubble_color = colors.get(party_name_dict.get(winning_PartyAb, winning_PartyAb), '#000000')
            
            # --- 2. CALCULATE BUBBLE SIZE ---
            # Square root scaling ensures visual area matches the vote proportion.
            # The divisor (e.g., 6.0) scales the overall map. Adjust this number if bubbles are too big/small.
            # The '+ 2' ensures even tiny booths have a minimum visible size.
            total_v = row.get('Total votes', 0)
            scaled_radius = (np.sqrt(total_v) / 4.5) + 2 

            # get opacity based on lead
            top_2_pcts = row[proj_parties].astype(float).nlargest(2).values
            
            # Calculate the lead (margin)
            if len(top_2_pcts) >= 2:
                margin = top_2_pcts[0] - top_2_pcts[1]
            else:
                margin = 1.0 # Fallback 
                
            # Scale the margin to an opacity between 0.4 and 1.0
            if margin >= 0.15:
                dynamic_opacity = 0.8
            elif margin <= 0.02:
                dynamic_opacity = 0.4
            else:
                # Interpolate: maps the 0.02-0.15 range proportionally to the 0.4-1.0 range
                dynamic_opacity = 0.4 + ((margin - 0.02) / 0.13) * 0.4

            # --- 3. BUILD THE TABULAR HTML TOOLTIP ---
            # Increased min-width to 240px to comfortably fit three columns
            tooltip_html = f"""
            <div style="min-width: 240px;">
                <b style="font-size: 14px;">{row['pp_nm']}</b><br>
                <hr style='margin:4px 0;'>
                <table style="width:100%; font-size: 13px; border-collapse: collapse;">
                    <tr>
                        <th style="text-align:left; border-bottom: 1px solid #ccc; padding-bottom: 2px;">Party</th>
                        <th style="text-align:right; border-bottom: 1px solid #ccc; padding-bottom: 2px;">Average<br>Projection</th>
                        <th style="text-align:right; border-bottom: 1px solid #ccc; padding-bottom: 2px;">2025<br>Result</th>
                    </tr>
            """
            
            # Add top parties as table rows
            for party in top_parties:
                if party in row and not pd.isna(row[party]):
                    pct = row[party] * 100
                    p_color = colors.get(party_name_dict.get(party, party), '#000000')
                    
                    # Safely fetch the 2025 actuals
                    col_2025 = f"{party}_2025"
                    if col_2025 in row and not pd.isna(row[col_2025]):
                        pct_2025 = f"{row[col_2025] * 100:.1f}%"
                    else:
                        pct_2025 = "-"
                    
                    # <tr> defines a row, <td> defines a cell
                    # The color: black on the 3rd <td> overrides the colored <tr>
                    tooltip_html += f"""
                    <tr style="color: {p_color};">
                        <td><b>{party_name_dict.get(party, party)}</b></td>
                        <td style="text-align:right;">{pct:.1f}%</td>
                        <td style="text-align:right; color: black;">{pct_2025}</td>
                    </tr>
                    """
                    
            # Add OTH row at the bottom
            oth_prop = row[minor_parties].sum() if len(minor_parties) > 0 else 0

            
            # Safely calculate 2025 OTH using matching minor party columns
            minor_cols_2025 = ['ASP_2025','CYA_2025','FFPA_2025','GRPF_2025']
            oth_prop_2025 = row[minor_cols_2025].sum() if len(minor_cols_2025) > 0 else 0
            pct_2025_oth_str = f"{(oth_prop_2025 * 100):.1f}%" if oth_prop_2025 > 0 else "-"

            # Only show the row if there's an 'Other' vote in either projection or 2025
            if oth_prop > 0 or oth_prop_2025 > 0:
                tooltip_html += f"""
                <tr style="color: black;">
                    <td><b>Other</b></td>
                    <td style="text-align:right;">{(oth_prop * 100):.1f}%</td>
                    <td style="text-align:right; color: black;">{pct_2025_oth_str}</td>
                </tr>
                """
            # --- ADD LABOR ROW ---
            if 'ALP_2025' in row and not pd.isna(row['ALP_2025']):
                pct_2025_alp_str = f"{(row['ALP_2025'] * 100):.1f}%"
            else:
                pct_2025_alp_str = "-"

            tooltip_html += f"""
            <tr style="color: black;">
                <td>(Labor)</td>
                <td style="text-align:right;">-</td>
                <td style="text-align:right;">{pct_2025_alp_str}</td>
            </tr>
            """

            # Close the table, add a divider, and append the 2025 Formal Votes line
            # int() handles any floats safely, and {:,} formats it nicely with commas (e.g., 1,234)
            safe_total_v = int(total_v) if not pd.isna(total_v) else 0
            tooltip_html += f"""
                </table>
                <hr style='margin:4px 0;'>
                <div style="font-size: 13px; margin-top: 4px; color: black; text-align: right;">
                    <b>2025 Formal Votes:</b> {safe_total_v:,}
                </div>
            </div>
            """

            # --- 4. DRAW THE MARKER ---
            folium.CircleMarker(
                location=[row['Lat'], row['Long']],
                radius=scaled_radius,
                color='white',                # Thin white outline to make overlapping bubbles distinct
                weight=1,
                fill=True,
                fill_color=bubble_color,      # Filled with the winning party's color
                fill_opacity=dynamic_opacity,            # Slightly transparent
                popup=folium.Popup(tooltip_html, max_width=250),
                tooltip=row['pp_nm']
            ).add_to(m)

        from folium.features import DivIcon

        # 1. Find the exact center of the Farrer boundary
        # (Using the farrer_boundary GeoDataFrame you loaded earlier)
        farrer_center_lat = farrer_boundary.geometry.centroid.y.iloc[0]
        farrer_center_lon = farrer_boundary.geometry.centroid.x.iloc[0]

        # 2. Apply a slight offset for "top left"
        # Farrer is massive, so an offset of 0.3 degrees is roughly 30km. 
        # Tweak these numbers if it needs to move more!
        label_lat = farrer_center_lat + 0.3  # Positive moves North (Up)
        label_lon = farrer_center_lon - 0.4  # Negative moves West (Left)

        # 3. Add the styled text using DivIcon
        folium.Marker(
            location=[label_lat, label_lon],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(75, 18), # Centers the text box on the coordinate
                html=f"""
                    <div style="
                        font-size: 36px; 
                        font-weight: bold; 
                        color: #933eea; 
                        font-family: Arial, sans-serif;
                        white-space: nowrap;
                        /* Adding a subtle white glow/shadow ensures it is readable over any map lines */
                        text-shadow: 2px 2px 4px rgba(255,255,255,0.9), -2px -2px 4px rgba(255,255,255,0.9);
                        letter-spacing: 2px;
                    ">
                        FARRER
                    </div>
                """
            )
        ).add_to(m)

        # Save and view
        os.chdir("/home/dania-freidgeim/Australian Election/")
        m.save("Farrer_Booth_Projection_Map.html")
        print("Map saved to Farrer_Booth_Projection_Map.html")



        

        # 1. Extract the raw HTML string from your Folium map
        map_html = m.get_root().render()

        # 2. Escape the HTML so it can be safely injected into an iframe srcdoc
        escaped_map_html = html.escape(map_html)

        # 3. Build the Master HTML wrapper with your exact accordion CSS
        # Note: I increased max-width from 700px to 850px so the map has room to breathe, 
        # but you can change it right back to 700px if you prefer!
        accordion_master_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap" rel="stylesheet">
            <style>
                /* Main Accordion Wrapper: Pale Purple Background */
                details.tcp-details {{ 
                    background: #faf5ff; 
                    border: 2px solid #e9d5ff; 
                    border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(147, 51, 234, 0.05); 
                    overflow: hidden; 
                }}
                
                /* Banner */
                details.tcp-details summary {{ 
                    background: #f3e8ff; 
                    padding: 16px 24px; 
                    font-weight: 800; 
                    color: #6419a8; 
                    cursor: pointer; 
                    list-style: none; 
                    display: flex; 
                    align-items: center; 
                    justify-content: space-between; 
                    font-size: clamp(0.85rem, 4vw, 1.1rem); 
                    border-bottom: 2px solid #e9d5ff; 
                    transition: background 0.2s; 
                }}
                
                details.tcp-details summary:hover {{ background: #e9d5ff; }}
                details.tcp-details summary::-webkit-details-marker {{ display: none; }}
                
                /* Custom Dropdown Arrow */
                details.tcp-details summary::after {{ content: "▼"; font-size: 0.9rem; color: #9333ea; }}
                details.tcp-details[open] summary::after {{ content: "▲"; }}
                
                /* Map Container inside Accordion */
                .map-container {{
                    width: 100%;
                    height: 650px; /* Height of the map */
                    padding: 0;
                    margin: 0;
                    background: #ffffff;
                }}
            </style>
        </head>
        <body style="background-color: #ffffff; padding: 20px;">

            <!-- YOUR ACCORDION -->
            <div class="tcp-methodology-accordion" style="font-family: 'Poppins', sans-serif; max-width: 850px; margin: 2rem auto;">
                <details class="tcp-details" id="map-details" open>
                    <summary>Show Polling Place Projection Map</summary>
                    
                    <div class="map-container">
                        <!-- IFRAME EMBED: Isolates the Folium Leaflet CSS from your page CSS -->
                        <iframe 
                            srcdoc="{escaped_map_html}" 
                            width="100%" 
                            height="100%" 
                            style="border:none;"
                            allowfullscreen>
                        </iframe>
                    </div>
                    
                </details>
            </div>

        </body>
        </html>
        """

       

    

    make_Farrer_booth_map(merged_df)

    import pdb; pdb.set_trace()
    return 1







def make_single_electorate_violin_plot(combined_df, tcp_pair_array=None, winning_tcp_array=None, winning_p_array=None, to_plot=False, group_OTH=False, div_results_dict = {}):

    colour_df = pd.read_csv("Party Colours.csv")
    colour_df.loc[:,'Party'] = colour_df['Party'].str.replace("’", "'", regex=False)
    colors = colour_df.set_index("Party")["Colour"].to_dict()

    party_name_dict = {
        'ALP':'Labor', 'NP': 'National', 'COAL': 'Coalition', 'ON' : 'One Nation',
        'GRN' : 'Greens', 'HMP' : 'Legalise Cannabis', 'FFPA' : 'Family First',
        'ASP' : 'Shooters Fishers & Farmers', 'LP' : 'Liberal' , 'TOP' : 'Trumpet of Patriots' ,
        'IND' : 'Independent' , 'AJP' : 'Animal Justice', 'SOPA' :'FUSION' ,
        'LDP' : 'Libertarian' , 'AUD' : 'Australian Democrats', 'CEC' : 'Citizens Party',
        'VNS' : 'Victorian Socialists', 'IMO' : 'HEART', 'GRPF' : "People's First",
        'AUC' : 'Christians', 'SAL' : 'Socialist Alliance', 'IAP' : 'Indigenous-Aboriginal',
        'GAP' : 'Great Australian Party', 'KAP' : "Katter's Australian",
        'CLP' : 'Country Liberal', 'LNP' : 'Liberal National', 'XEN' : 'Centre Alliance',
        'IND1' : 'Independent 1', 'IND2' : 'Independent 2', 'IND3' : 'Independent 3',
        'IND4' : 'Independent 4', 'IND5' : 'Independent 5', 'OTH' : 'Other',
        'IND1Goldstein': 'Ind. Zoe Daniel', 'IND1Calare': 'Ind. Kate Hook',
        'IND2Calare': 'Ind. Andrew Gee', 'IND1Kooyong': 'Ind. Monique Ryan',
        'IND3Mackellar': 'Ind. Sophie Scamps', 'IND1Moore': 'Ind. Ian Goodenough',
        'IND1McPherson': 'Ind. Erchana Murray-Barlett', 'IND1Wannon': 'Ind. Alex Dyson',
        'IND1Curtin' : 'Ind. Kate Chaney', 'IND1Indi' : 'Ind. Helen Haines',
        'IND1Clark' : 'Ind. Andrew Wilkie', 'IND1Moncrieff' : 'Ind. Nicole Arrowsmith',
        'IND2Moore' : 'Ind. Nathan Barton', 'IND2Bradfield': 'Ind. Nicolette Boele',
        'IND2Berowra' : 'Ind. Tina Smith', 'IND1Forrest' : 'Ind. Sue Chapman',
        'IND1Sturt' : 'Ind. Verity Cooper', 'IND1Gilmore' : 'Ind. Kate Dezarnaulds',
        'IND1Casey' : 'Ind. Claire Miles', 'IND1Franklin' : 'Ind. Peter George',
        'IND2Cowper' : 'Ind. Caz Heise', 'IND1Fremantle' : 'Ind. Kate Hulett',
        'IND1Fisher' : 'Ind. Kerryn Jones', 'IND1Grey' : 'Ind. Anita Kuss',
        'IND1Monash' : 'Ind. Russell Broadbent', 'IND2Monash' : 'Ind. Deb Leonard',
        'IND1Lyne' : 'Ind. Jeremy Miller', 'IND1Farrer' : 'Ind. Milthorpe',
        'IND1Deakin' : 'Ind. Jess Ness', 'IND1Bean' : 'Ind. Jessie Price',
        'IND5Riverina' : 'Ind. Jenny Rolfe', 'IND1Solomon' : 'Ind. Phil Scott',
        'IND1Flinders' : 'Ind. Ben Smith', 'IND1Dickson' : 'Ind. Ellie Smith',
        'IND1Fairfax' : 'Ind. Francine Wiig', 'IND1Fowler' : 'Ind. Dai Le',
        'IND2Wentworth' : 'Ind. Allegra Spender', 'IND1Groom' : 'Ind. Suzie Holt',
        'IND1Warringah' : 'Ind. Zali Steggall', 'CYA': 'Trumpet of Patriots',
        'GRPF': "People's First", 'FFPA': 'Family First', 'LTP': 'Libertarian',
        'SPP': 'Sustainable Australia','IND2Farrer': 'Ind. Woodward', 'IND3Farrer': 'Ind. Pappin'
    }

    site_abbreviation_dict = {
        "Labor": "LAB", "Liberal": "LIB", "National" : 'NAT', "Coalition" : 'COAL',
        "Other" : 'OTH', "Greens": "GRN", "One Nation" : 'ON', 'Legalise Cannabis': 'LCP',
        'Family First': 'FFP', 'Shooters Fishers & Farmers': 'SFF', 'Trumpet of Patriots': 'TOP',
        'Independent': 'IND', 'Animal Justice': 'AJP', 'FUSION': 'FUSN' ,
        'Libertarian':'LBT' , 'Australian Democrats': 'AUD', 'Citizens Party': 'CIT',
        'Victorian Socialists':'VSOC', 'HEART':'HRT', "People's First":'PPF',
        'Christians':'AUC', 'Socialist Alliance':'SA', 'Indigenous-Aboriginal': 'IAP',
        'Great Australian Party':'GAP', "Katter's Australian": 'KAP', 'Country Liberal': 'CLP',
        'Liberal National': 'LNP', 'Centre Alliance': 'CA', 'Independent 1' : 'IND1',
        'Independent 2' : 'IND2', 'Independent 3' : 'IND3', 'Independent 4' : 'IND4',
        'Independent 5' : 'IND5', 'Sustainable Australia': 'SAP'
    }

    Independents = {key: value for key, value in party_name_dict.items() if key.startswith('IND') and len(key) > 5}
    div = 'Farrer'
    for ind in Independents.keys():
        IND_full_name = Independents[ind]
        surname = IND_full_name.split(' ')[-1]
        abbreviation = f"IND {surname[:3].title()}."
        site_abbreviation_dict[IND_full_name] = abbreviation

    # --- 0. ATTACH TCP DATA ---
    combined_df = combined_df.copy()
    if tcp_pair_array is not None and winning_tcp_array is not None and winning_p_array is not None:
        combined_df['Winner'] = [party_name_dict.get(w, w) for w in winning_p_array]
        
        tcp_strings = []
        for w, pair, pct in zip(winning_p_array, tcp_pair_array, winning_tcp_array):
            if pct < 1.1: pct *= 100  
            lose_pct = 100.0 - pct
            
            if isinstance(pair, (list, tuple, np.ndarray)):
                loser = pair[1] if str(pair[0]).strip() == str(w).strip() else pair[0]
            else:
                parts = str(pair).replace('v', ' ').replace(',', ' ').split()
                loser = parts[1] if parts[0] == str(w) and len(parts)>1 else parts[0]
            
            def get_abbr(p):
                full = party_name_dict.get(p, p)
                return site_abbreviation_dict.get(full, p)

            # Get exact hex colors
            win_color = colors.get(party_name_dict.get(w, w), '#333333')
            lose_color = colors.get(party_name_dict.get(loser, loser), '#333333')
            
            win_abbr = get_abbr(w)
            lose_abbr = get_abbr(loser)
            
            # Format HTML Matrix (Single line to keep JSON clean)
            matrix_html = f"<div class='tcp-matrix'><div class='tcp-matrix-header'>{win_abbr}</div><div class='tcp-matrix-header'>{lose_abbr}</div><div class='tcp-matrix-val' style='color: {win_color};'>{pct:.1f}</div><div class='tcp-matrix-val' style='color: {lose_color};'>{lose_pct:.1f}</div></div>"
                
            tcp_strings.append(matrix_html)
            
        combined_df['TCP_String'] = tcp_strings
    else:
        combined_df['Winner'] = "Unknown"
        combined_df['TCP_String'] = ""

    # --- 1. Apply Custom Party Name Logic to Columns ---
    df_plot = combined_df.copy()

    if len(df_plot) > 1000:
        df_plot = df_plot.sample(n=1000, random_state=42)
    
    # Hide TCP text columns before grouping or doing math
    tcp_data = df_plot[['Winner', 'TCP_String']].copy()
    df_plot.drop(columns=['Winner', 'TCP_String'], inplace=True)
    
    if group_OTH:
        keep_cols = ['ON', 'IND1', 'LP', 'NP','GRN']
        cols_to_group = [c for c in df_plot.columns if c not in keep_cols and c != 'OTH']
        
        if cols_to_group:
            if 'OTH' in df_plot.columns:
                df_plot['OTH'] += df_plot[cols_to_group].sum(axis=1)
            else:
                df_plot['OTH'] = df_plot[cols_to_group].sum(axis=1)
            df_plot.drop(columns=cols_to_group, inplace=True)

    df_plot.columns = [str(c) + div if str(c).startswith('IND') else str(c) for c in df_plot.columns]
    df_plot.columns = df_plot.columns.to_series().replace(party_name_dict).values
    df_plot.columns = df_plot.columns.str.replace(r'^(IND\d).*', r'\1', regex=True)
    df_plot.columns = df_plot.columns.to_series().replace(party_name_dict).values

    # Convert proportions to percentages
    df_pct = df_plot * 100
    
    # Re-attach TCP Data
    df_pct['Winner'] = tcp_data['Winner']
    df_pct['TCP_String'] = tcp_data['TCP_String']
   
    # Determine party order (ignoring strings)
    numeric_cols = df_pct.select_dtypes(include=[np.number])
    party_order = numeric_cols.mean().sort_values(ascending=False).index.tolist()

    target_last = 'Other'
    if target_last in party_order:
        party_order.remove(target_last)
        party_order.append(target_last)

    summary_stats = {
        party: {
            'mean': df_pct[party].mean(),
            'lower_95': np.percentile(df_pct[party], 2.5),
            'upper_95': np.percentile(df_pct[party], 97.5)
        }
        for party in party_order
    }

    fig = go.Figure()

    for party in party_order:
        party_data = df_pct[party].to_numpy()
        stats = summary_stats[party]
        
        display_name = party_name_dict.get(party, party)
        x_label = site_abbreviation_dict.get(party, party)
        party_color = colors.get(party, 'gray')

        dynamic_bw = 0.7 if stats['mean'] > 4.0 else 0.2 
        dynamic_bw = 2 if stats['mean'] > 10 else dynamic_bw

        fig.add_trace(go.Violin(
            x=[x_label] * len(party_data),
            y=party_data,
            name=display_name,
            hoveron='violins',
            hoverinfo='none',
            hovertemplate=(
                f"<b>{display_name}</b><br>"
                f"Mean: {stats['mean']:.1f}%<br>"
                f"2.5% CI: {stats['lower_95']:.1f}%<br>"
                f"97.5% CI: {stats['upper_95']:.1f}%<extra></extra>"
            ),
            box_visible=False,
            meanline_visible=True,
            meanline=dict(color='#5d3fd3'),
            line_color='rgba(0,0,0,0)',
            fillcolor=party_color,
            opacity=0.7,
            width=0.7,
            bandwidth=dynamic_bw,
            points=False
        ))

        if div_results_dict and party in div_results_dict:
            result_value = div_results_dict[party] 
            
            fig.add_trace(go.Scatter(
                x=[x_label], 
                y=[result_value],  
                mode='markers',
                name=f'Result::{party}', 
                # zorder removed so it respects insertion order!
                marker=dict(
                    color='rgba(0, 0, 0, 0.6)', 
                    size=12,
                ),
                hovertemplate=(
                    f"<b>{display_name}</b><br>"
                    "Result: %{y:.2f}%<extra></extra>"
                ),
                showlegend=False
            ))

        sample_size = 200  
        sorted_data = np.sort(party_data)  
        lowest_40 = sorted_data[:40]
        highest_30 = sorted_data[-30:]
        middle_values = sorted_data[40:-30]
        
        middle_130 = rng.choice(middle_values, 130, replace=False)
        hover_y = np.concatenate([lowest_40, highest_30, middle_130])

        sim_templates = []
        sim_bgs = []
        sim_fonts = []

        if div_results_dict and party in div_results_dict:
            for y in hover_y:
                if abs(y - result_value) <= 1.0:
                    # Overlapping dot: mimic the actual result's template and default colors
                    sim_templates.append(f"<b>{display_name}</b><br>Result: {result_value:.2f}%<extra></extra>")
                    sim_bgs.append('rgba(0, 0, 0, 0.6)') 
                    sim_fonts.append('white')
                else:
                    # Normal dot: standard simulation template and colors
                    sim_templates.append(f"<b>{display_name}</b><br>%{{y:.1f}}%<extra></extra>")
                    sim_bgs.append('white')
                    sim_fonts.append(party_color)
        else:
            sim_templates = [f"<b>{display_name}</b><br>%{{y:.1f}}%<extra></extra>"] * sample_size
            sim_bgs = ['white'] * sample_size
            sim_fonts = [party_color] * sample_size

        fig.add_trace(go.Scatter(
            x=[x_label] * sample_size,
            y=hover_y,
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=12), 
            hovertemplate=sim_templates,
            hoverlabel=dict(bgcolor=sim_bgs, font=dict(color=sim_fonts)),
            name=f"{display_name} sims",
            showlegend=False
        ))


    fig.update_layout(
        title=None,          
        showlegend=True,
        hovermode="closest",
        legend=dict(
            font=dict(size=26),
            x=0.7,  
            y=1,  
            traceorder='normal',
            bgcolor='rgba(255, 255, 255, 0)',  
            borderwidth=2,  
            itemwidth=30,  
        ),
        violinmode="group",
        violingap=0.1,
        margin=dict(l=30, r=30, t=20, b=40),
        xaxis_tickangle=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(
        tick0=0,
        dtick=5,
        tickformat='.0f',
        ticksuffix='%',
        tickfont=dict(size=16),  
        range=[-1, 61]
    )

    filename = "Farrer_byelection_violin_grouped.json" if group_OTH else "Farrer_byelection_violin.json"
    with open(filename, "w") as f:
        f.write(fig.to_json())

    if not group_OTH:
        df_sample = df_pct.sample(n=100) 
        df_sample.to_json("Farrer_simulations_sample.json", orient="records") 

    if to_plot:
        fig.show()

    return fig



def make_tcp_violin_plot(per_electorate_tcp_pair_vote):

    # --- 1. YOUR SETUP CODE ---
    colour_df = pd.read_csv("Party Colours.csv")
    colour_df.loc[:,'Party'] = colour_df['Party'].str.replace("’", "'", regex=False)
    colors = colour_df.set_index("Party")["Colour"].to_dict()

    party_name_dict = {
        'ALP':'Labor', 'NP': 'National', 'COAL': 'Coalition', 'ON' : 'One Nation',
        'GRN' : 'Greens', 'HMP' : 'Legalise Cannabis', 'FFPA' : 'Family First',
        'ASP' : 'Shooters Fishers & Farmers', 'LP' : 'Liberal', 'TOP' : 'Trumpet of Patriots',
        'IND' : 'Independent', 'AJP' : 'Animal Justice', 'SOPA' :'FUSION',
        'LDP' : 'Libertarian', 'AUD' : 'Australian Democrats', 'CEC' : 'Citizens Party',
        'VNS' : 'Victorian Socialists', 'IMO' : 'HEART', 'GRPF' : "People's First",
        'AUC' : 'Christians', 'SAL' : 'Socialist Alliance', 'IAP' : 'Indigenous-Aboriginal',
        'GAP' : 'Great Australian Party', 'KAP' : "Katter's Australian",
        'CLP' : 'Country Liberal', 'LNP' : 'Liberal National', 'XEN' : 'Centre Alliance',
        'IND1' : 'Independent 1', 'IND2' : 'Independent 2', 'IND3' : 'Independent 3',
        'IND4' : 'Independent 4', 'IND5' : 'Independent 5', 'OTH' : 'Other',
        'IND1Farrer' : 'Ind. Milthorpe', # Target Ind
        # ... (other specific independents included here)
    }

    site_abbreviation_dict = {
        "Labor": "LAB", "Liberal": "LIB", "National": "NAT", "Coalition": "COAL", 
        "Other": "OTH", "Greens": "GRN", "One Nation": "ON", "Legalise Cannabis": "LCP",
        "Independent": "IND", "Independent 1": "IND1" # ...
    }

    div = 'Farrer'
    total_sims = 10000

    # --- 2. PLOTLY GENERATION ---
    fig = go.Figure()

    # Assume per_electorate_tcp_pair_vote['Farrer'] exists and is loaded
    tcp_data = per_electorate_tcp_pair_vote[div]

    for (party1, party2), votes in tcp_data.items():
        vote_array = np.array(votes)
        if len(vote_array) == 0:
            continue
            
        # --- A. Name Resolution Logic ---
        # Check if the party is an IND that needs an electorate-specific mapping (e.g., IND1 -> IND1Farrer)
        p1_key = f"{party1}{div}" if party1.startswith('IND') else party1
        p2_key = f"{party2}{div}" if party2.startswith('IND') else party2
        
        # Get the full readable name
        p1_full_name = party_name_dict.get(p1_key, party_name_dict.get(party1, party1))
        p2_full_name = party_name_dict.get(p2_key, party_name_dict.get(party2, party2))
        
        # Get short abbreviations for clean axes (fallback to full name if not in dict)
        p1_display = site_abbreviation_dict.get(p1_full_name, p1_full_name)
        p2_display = site_abbreviation_dict.get(p2_full_name, p2_full_name)
        
        # --- B. Color Resolution ---
        # Attempt to grab the exact color from your CSV dict using the full name, default to gray if missing
        p1_color = colors.get(p1_full_name, colors.get(party1, '#64748b'))
        
        # --- C. Frequency Calculation ---
        frequency = len(vote_array) / total_sims
        freq_percent = frequency * 100
        
        # --- D. Formatting the Label ---
        # Bakes the matchup text and the occurrence frequency straight into the x-axis tick
        x_label = f"<b>{p1_display}</b> vs {p2_display}<br><span style='font-size:11px; color:#64748b;'>Occurs: {freq_percent:.1f}%</span>"
        
        # Add the violin trace shaded purely by Party 1's color
        fig.add_trace(go.Violin(
            y=vote_array,
            x=[x_label] * len(vote_array),
            name=f"{party1} vs {party2}",
            line_color=p1_color,
            fillcolor=p1_color,
            opacity=0.8,
            box_visible=True,
            meanline_visible=True,
            showlegend=False
        ))

    # --- 3. THE 50% WIN THRESHOLD ---
    fig.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="#1e293b", 
        line_width=2,
        annotation_text="50% Win Threshold", 
        annotation_position="top left",
        annotation_font=dict(color="#1e293b", size=12, family="Poppins")
    )

    # --- 4. LAYOUT STYLING ---
    fig.update_layout(
        font_family="Poppins",
        yaxis_title="TCP Vote Share (First Party)",
        yaxis=dict(
            tickformat=".0%", 
            range=[0.35, 0.65], # Locks the view to highlight marginal differences around 50%
            gridcolor="#e2e8f0"
        ), 
        xaxis=dict(
            tickangle=0 # Keeps the labels flat and readable
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=60, l=60, r=20)
    )

    # Export the figure to your web assets folder
    # fig.write_json(f"Farrer_tcp_violins.json")
    fig.show()

    

def make_horizontal_bars(div_list, win_proportions_df, to_json = 0):


    colour_df = pd.read_csv("Party Colours.csv")

    colour_df.loc[:,'Party'] = colour_df['Party'].str.replace("’", "'", regex=False)

    colors = colour_df.set_index("Party")["Colour"].to_dict()


    party_name_dict = {
        'ALP':'Labor',
        'NP': 'National',
        'COAL': 'Coalition',
        'ON' : 'One Nation',
        'GRN' : 'Greens',
        'HMP' : 'Legalise Cannabis',
        'FFPA' : 'Family First',
        'ASP' : 'Shooters Fishers & Farmers',
        'LP' : 'Liberal' ,
        'TOP' : 'Trumpet of Patriots' ,
        'IND' : 'Independent' ,
        'AJP' : 'Animal Justice',
        'SOPA' :'FUSION' ,
        'LDP' : 'Libertarian' ,
        'AUD' : 'Australian Democrats',
        'CEC' : 'Citizens Party',
        'VNS' : 'Victorian Socialists',
        'IMO' : 'HEART',
        'GRPF' : "People's First",
        'AUC' : 'Christians',
        'SAL' : 'Socialist Alliance',
        'IAP' : 'Indigenous-Aboriginal',
        'GAP' : 'Great Australian Party',
        'KAP' : "Katter's Australian",
        'CLP' : 'Country Liberal',
        'LNP' : 'Liberal National',
        'XEN' : 'Centre Alliance',
        'IND1' : 'Independent 1',
        'IND2' : 'Independent 2',
        'IND3' : 'Independent 3',
        'IND4' : 'Independent 4',
        'IND5' : 'Independent 5',
        'OTH' : 'Other',
        'IND1Goldstein': 'Ind. Zoe Daniel',
        'IND1Calare': 'Ind. Kate Hook',
        'IND2Calare': 'Ind. Andrew Gee',
        'IND1Kooyong': 'Ind. Monique Ryan',
        'IND3Mackellar': 'Ind. Sophie Scamps',
        'IND1Moore': 'Ind. Ian Goodenough',
        'IND1McPherson': 'Ind. Erchana Murray-Barlett',
        'IND1Wannon': 'Ind. Alex Dyson',
        'IND1Curtin' : 'Ind. Kate Chaney',
        'IND1Indi' : 'Ind. Helen Haines',
        'IND1Clark' : 'Ind. Andrew Wilkie',
        'IND1Moncrieff' : 'Ind. Nicole Arrowsmith',
        'IND2Moore' : 'Ind. Nathan Barton',
        'IND2Bradfield': 'Ind. Nicolette Boele',
        'IND2Berowra' : 'Ind. Tina Smith',
        'IND1Forrest' : 'Ind. Sue Chapman',
        'IND1Sturt' : 'Ind. Verity Cooper',
        'IND1Gilmore' : 'Ind. Kate Dezarnaulds',
        'IND1Casey' : 'Ind. Claire Miles',
        'IND1Franklin' : 'Ind. Peter George',
        'IND2Cowper' : 'Ind. Caz Heise',
        'IND1Fremantle' : 'Ind. Kate Hulett',
        'IND1Fisher' : 'Ind. Kerryn Jones',
        'IND1Grey' : 'Ind. Anita Kuss',
        'IND1Monash' : 'Ind. Russell Broadbent',
        'IND2Monash' : 'Ind. Deb Leonard',
        'IND1Lyne' : 'Ind. Jeremy Miller',
        'IND1Farrer' : 'Ind. Milthorpe',
        'IND1Deakin' : 'Ind. Jess Ness',
        'IND1Bean' : 'Ind. Jessie Price',
        'IND5Riverina' : 'Ind. Jenny Rolfe',
        'IND1Solomon' : 'Ind. Phil Scott',
        'IND1Flinders' : 'Ind. Ben Smith',
        'IND1Dickson' : 'Ind. Ellie Smith',
        'IND1Fairfax' : 'Ind. Francine Wiig',
        'IND1Fowler' : 'Ind. Dai Le',
        'IND2Wentworth' : 'Ind. Allegra Spender',
        'IND1Groom' : 'Ind. Suzie Holt',
        'IND1Warringah' : 'Ind. Zali Steggall',
        'CYA': 'Trumpet of Patriots',
        'GRPF': "People's First",
        'FFPA': 'Family First',
        'LTP': 'Libertarian',
        'SPP': 'Sustainable Australia'
    }

    site_abbreviation_dict = {
        "Labor": "LAB",
        "Liberal": "LIB",
        "National" : 'NAT',
        "Coalition" : 'COAL',
        "Other" : 'OTH',
        "Greens": "GRN",
        "One Nation" : 'ON',
        'Legalise Cannabis': 'LCP',
        'Family First': 'FFP',
        'Shooters Fishers & Farmers': 'SFF',
        'Trumpet of Patriots': 'TOP',
        'Independent': 'IND',
        'Animal Justice': 'AJP',
        'FUSION': 'FUSN' ,
        'Libertarian':'LBT' ,
        'Australian Democrats': 'AUD',
        'Citizens Party': 'CIT',
        'Victorian Socialists':'VSOC',
        'HEART':'HRT',
        "People's First":'PPF',
        'Christians':'AUC',
        'Socialist Alliance':'SA',
        'Indigenous-Aboriginal': 'IAP',
        'Great Australian Party':'GAP',
        "Katter's Australian": 'KAP',
        'Country Liberal': 'CLP',
        'Liberal National': 'LNP',
        'Centre Alliance': 'CA',
        'Independent 1' : 'IND1',
        'Independent 2' : 'IND2',
        'Independent 3' : 'IND3',
        'Independent 4' : 'IND4',
        'Independent 5' : 'IND5',
        'Sustainable Australia': 'SAP'
    }

    


    import plotly.graph_objects as go


    for div in div_list:
        print(div)

        df = win_proportions_df

        # adjust IND winners
        df.loc[df['Party'].str.startswith('IND'),'Party'] = df.loc[df['Party'].str.startswith('IND'),'Party'] + div
        df['Party'] = df['Party'].replace(party_name_dict) # change names for main independents
        df.loc[(df['Party'].str.startswith('IND')) & (df['Party'].str.match(r'^IND\d')),'Party'] = df['Party'].str[:4] # unchanged Independents
        df['Party'] = df['Party'].replace(party_name_dict)

        fig = go.Figure()

        df = df.sort_values('Win Percentage', ascending=False)

        fig = go.Figure()
        #import pdb;pdb.set_trace()


        cumulative = 0

        for idx, row in df.iterrows():
            party = row['Party']
            pct = row['Win Percentage']
            color = colors.get(party, colors['Independent'])

            fig.add_trace(go.Bar(
                y=[div],  # The electorate name appears as the bar label
                x=[pct],         # Width of the party's slice
                name=party,
                orientation='h',
                marker=dict(color=color),
                hovertemplate=f"{party}: {pct:.1%}<extra></extra>",
                text=f"{party} {pct:.0%}" if pct >= 0.05 else None,
                textposition="inside" if pct > 0.05 else None,
                insidetextanchor="middle",
                textfont=dict(color="white", size=16, weight=600),
                width=0.6  # Makes the bar thicker
            ))

        fig.update_layout(
            barmode='stack',  # STACK, not relative
            title="Win Probabilities",
            title_x=0.05,  # Align title to the left
            title_y = 0.8,
            title_font=dict(size=20, family="Arial, sans-serif", color="black"), 
            height=150,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                range=[0, 1],
                fixedrange=True
            ),
            yaxis=dict(showticklabels=False),
            showlegend=False,
            plot_bgcolor='white'
        )

        fig.show()
        #import pdb;pdb.set_trace()

        if to_json:

            # Save as JSON string
            fig_json = fig.to_json()

            with open(f"Farrer_byelection_bar.json", "w") as f:
                f.write(fig_json)
    return 1



def make_party_category_dict():

        all_parties = pd.read_csv('Grand_Party_Category_df_2004_2022.csv', index_col=None)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['CLR'],'Ideo_Category':['ALP'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['NGS'],'Ideo_Category':['Right'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['ARTS'],'Ideo_Category':['Left'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties_house = all_parties.loc[all_parties['Ideo_Category'].notna(),].iloc[:,:2].set_index('PartyAb') # excludes only senates, who don't yet have Ideology written
        party_category_dict = all_parties_house.to_dict()['Ideo_Category']
        party_category_dict['IND'] = 'Centre'
        party_category_dict['COALLP'] = 'COAL'
        party_category_dict['COALNP'] = 'COAL'

        return party_category_dict


def get_winner_from_top_2(sim_votes, proportions_with_joint_noise, curr_parties, top2_parties, top2_indices, party_to_category_centered_IND, TCP_COMBINATION_INDEX):


    # Step 2: Determine the category (alphabetically)
    cat_index_pairs = [
        (party_to_category_centered_IND.get(party, 'IND'), idx)
        for party, idx in zip(top2_parties, top2_indices)
    ]
    #print("cat pairs: ", cat_index_pairs)
    #import pdb;pdb.set_trace()


    sorted_cats_with_indices = sorted(cat_index_pairs, key=lambda x: x[0])  # [('GRN', 4), ('LNP', 1)]
    top2_category = tuple(cat for cat, _ in sorted_cats_with_indices)  # ('GRN', 'LNP')
    
    #top2_category = tuple(sorted([party_category_dict.get(p, 'IND') for p in top2_parties])) # Centre if party is IND1,IND2,IND3 etc.
    #first_idx = [i for i in top2_indices if party_to_category[i] == first_cat][0] # FIXXXX
    #row_index = TCP_COMBINATION_INDEX[tuple(sorted(top2_category))]


    #first_cat = top2_category[0]
    first_idx, second_idx = sorted_cats_with_indices[0][1], sorted_cats_with_indices[1][1]
    #print("first_idx: ", first_idx)
    
    # Step 3: Fetch transfer proportions
    row_index = TCP_COMBINATION_INDEX[top2_category]
    transfer_proportions = proportions_with_joint_noise.iloc[row_index].values # shape: (n_parties,)


    #transfer_proportions = transfer_proportions.copy()  # don’t overwrite source!
    #transfer_proportions[top2_indices] = 0
    #print("transfer_proportions: ", transfer_proportions)
    #import pdb;pdb.set_trace()


    # Step 6: Compute 2PP Allocation for the alphabetically-first party 
    non_top2_indices = [i for i in range(len(sim_votes)) if i not in top2_indices]

    # Total transfer to the first top-2 party is the dot product
    transferred_to_P1 = np.dot(sim_votes[non_top2_indices], transfer_proportions[non_top2_indices])

    # Add this to the original primary vote of P1
    P1_2PP_vote = sim_votes[first_idx] + transferred_to_P1/100 # as decimal
    #print("2PP: ", P1_2PP_vote)
    #import pdb;pdb.set_trace()



    if P1_2PP_vote > 0.5:
        winner_idx = first_idx
    else:
        winner_idx = second_idx

    winner_party = curr_parties[winner_idx]
    runner_up = curr_parties[second_idx if winner_idx == first_idx else first_idx]
    tcp_pair = (curr_parties[first_idx], curr_parties[second_idx])
    winning_tcp = P1_2PP_vote if winner_idx == first_idx else 1-P1_2PP_vote

    return winner_party,runner_up,tcp_pair,winning_tcp

def distribution_to_top_2(final_simulated_votes, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind, allow_leapfrog = False, chosen_top_2_tcps = []):

    TCP_COMBINATION_INDEX = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6, ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14}

    electorate_names = Results_dict.keys()
    n_simulations = len(final_simulated_votes['Farrer'])


    # --- ATTEMPT AT RIGOROUS ROW SCALING (N+1 Predictive Variance) ---
    by_election_multiplier = np.sqrt(BY_ELECTION_SCALING)  # Bump for Farrer 2026 by-election volatility
    N_6 = 1.5  # 1 Fed 2025 electorate + partial weight from SA data
    N_9 = 1.0  # 1 SA electorate only
    state_transfer_penalty = SA_DATA_UNCTY_WGHT  # Additional uncertainty for moving SA data to NSW

    # Initialize all 15 rows with the baseline independent noise
    row_joint_sigmas = np.full(15, sigma_joint)

    # Row 1: High N, but apply by-election multiplier
    row_joint_sigmas[1] = sigma_joint * by_election_multiplier

    # Row 6: Scale by sqrt((N+1)/N)
    row_joint_sigmas[6] = sigma_joint * np.sqrt((N_6 + 1) / N_6)

    # Row 9: N=1 scaling + state transfer penalty
    row_joint_sigmas[9] = sigma_joint * np.sqrt((N_9 + 1) / N_9) * state_transfer_penalty
    # ------------------------------------------------------

    # Your original joint noise line remains unchanged
    joint_noise = rng.normal(0, row_joint_sigmas, size=(n_simulations, 15))

    
    
    from collections import defaultdict, Counter

    # For goal 1
    per_electorate_winners = defaultdict(list)  # {'Electorate A': ['ALP', 'ALP', 'LP', ...]} # care about IND1,IND2
    per_electorate_runners_up = defaultdict(list)
    per_electorate_tcp_pair = defaultdict(list)
    per_electorate_winning_tcp = defaultdict(list)
    per_electorate_leapfrogs = defaultdict(list)
    per_electorate_tcp_pair_vote = defaultdict(lambda: defaultdict(list))

    # For goal 2
    per_simulation_winners = [ [] for _ in range(n_simulations) ]  # [[ALP, LP, ...], [LP, IND, ...], ...] # don't care about IND1/IND2 - show for COAL, ALP, GRN, ON, CA, XEN, IND, OTH


    for i, electorate in enumerate(electorate_names):
        #print(electorate)
        #electorate = 'Kennedy'
        # Get the simulations for this electorate
        sims = final_simulated_votes[electorate]  # shape: (10000, n_parties)
        proportions_df = proportions_transferred_to_first[electorate]  # shape: (15, n_parties)

        curr_parties = Results_dict[electorate].columns
        
        for sim_id in range(n_simulations):
            sim_votes = sims[sim_id]  # shape: (n_parties,)
            #print(sim_votes)

            # shape the new proportions_df:
            proportions_with_joint_noise = proportions_df.copy()


            proportions_with_joint_noise.iloc[:] += joint_noise[sim_id][:, np.newaxis] + rng.normal(0, sigma_ind, size=sim_votes.shape)
            #import pdb;pdb.set_trace()




            
            # Step 1: Get top 2 parties by vote share
            #top2_indices = np.argsort(sim_votes)[-2:][::-1]

            # Step 1: Get top 2 parties by vote share
            top2_indices = np.argsort(sim_votes)[-2:][::-1].copy()
            
            # --- COALITION LEAPFROG CLAUSE ---
            if allow_leapfrog and ('LP' in curr_parties and 'NP' in curr_parties):
                lp_idx = list(curr_parties).index('LP')
                np_idx = list(curr_parties).index('NP')
                
                # Determine who is the 'Lead' and who is the 'Trailer'
                if sim_votes[lp_idx] >= sim_votes[np_idx]:
                    coal_lead_idx, coal_sub_idx = lp_idx, np_idx
                else:
                    coal_lead_idx, coal_sub_idx = np_idx, lp_idx
                    
                # ONLY proceed if the Lead partner is NOT already in the top 2
                if coal_lead_idx not in top2_indices:
                    current_2nd_idx = top2_indices[1]
                    
                    # Leapfrog check: Lead + 80% of Trailer vs current 2nd place
                    leapfrog_vote = sim_votes[coal_lead_idx] + (LEAPFROG_HELP * sim_votes[coal_sub_idx])
                    
                    if leapfrog_vote > sim_votes[current_2nd_idx]:
                        # Swap the 2nd place index for the Lead Coalition partner
                        top2_indices[1] = coal_lead_idx
                        per_electorate_leapfrogs[electorate].append(sim_id)
            # --- END CLAUSE ---

            top2_parties = [curr_parties[i] for i in top2_indices]
            top2_parties = [Results_dict[electorate].columns[i] for i in top2_indices]  # depends on how parties are ordered

            winner_party,runner_up,tcp_pair,winning_tcp = get_winner_from_top_2(sim_votes, proportions_with_joint_noise, curr_parties, top2_parties, top2_indices, party_to_category_centered_IND, TCP_COMBINATION_INDEX)

               
            if chosen_top_2_tcps:
                for tcp in chosen_top_2_tcps: # [[ON,IND1], [ON,LP], [ON,NP], [IND1,LP], [IND1,NP],['LP','NP']]

                    top2_parties = [p for p in tcp]
                    top2_indices = [curr_parties.get_loc(p) for p in top2_parties]
                    top2_parties = [Results_dict[electorate].columns[i] for i in top2_indices]

                    winner_curr_party,_,_,winning_curr_tcp = get_winner_from_top_2(sim_votes, proportions_with_joint_noise, curr_parties, top2_parties, top2_indices, party_to_category_centered_IND, TCP_COMBINATION_INDEX)
                    per_electorate_tcp_pair_vote[electorate][tcp].append(winning_curr_tcp if winner_curr_party == top2_parties[0] else 1 - winning_curr_tcp)

            per_electorate_winners[electorate].append(winner_party) # vertical lists
            per_simulation_winners[sim_id].append(winner_party) # horizontal lists
            per_electorate_tcp_pair[electorate].append(tcp_pair)
            per_electorate_winning_tcp[electorate].append(winning_tcp)
            per_electorate_runners_up[electorate].append(runner_up)
            #print("winner: ", winner_party)
            #import pdb;pdb.set_trace()


    return (per_electorate_winners,  per_simulation_winners, per_electorate_tcp_pair, per_electorate_winning_tcp, per_electorate_runners_up, per_electorate_leapfrogs, per_electorate_tcp_pair_vote)  # dict[str, dict[str, int]] — useful for percentages ; list[dict[str, int]] or pd.DataFrame



def calculate_estimated_tcp(tcp_pair_array, per_electorate_winners, winning_tcp_array, p1,p2):

    mask_tcp = ((tcp_pair_array[:, 0] == p1) & (tcp_pair_array[:, 1] == p2)) + ((tcp_pair_array[:, 0] == p2) & (tcp_pair_array[:, 1] == p1))

    idx = np.where(mask_tcp)[0]
    idx_p1_win =  np.where(np.array(per_electorate_winners['Farrer'])==p1)[0]
    ind_p2_win = np.where(np.array(per_electorate_winners['Farrer'])==p2)[0]

    p1_win_tcp = winning_tcp_array[[p for p in idx if p in idx_p1_win ]]
    p1_lose_tcp = 1-winning_tcp_array[[p for p in idx if p in ind_p2_win ]]

    return np.concatenate((p1_win_tcp, p1_lose_tcp)).mean()


def calculate_mean_sitout_donations(direct_result, div, Major_sitout, mapping_df, proportions_df, prior_row):
    """
    Calculates the exact expected value of the sit-out redistribution 
    without running random Dirichlet simulations.
    
    direct_result: pd.Series of party vote shares for the single electorate
    """
    # 1. Isolate the Sitout Vote
    if Major_sitout not in direct_result:
        return pd.Series(0.0, index=direct_result.index)
        
    V_sitout = direct_result[Major_sitout]
    
    # 2. Get division properties from mapping
    div_mapping = mapping_df[(mapping_df['div_nm'] == div) & (mapping_df['Major_sitout'] == Major_sitout)]
    if div_mapping.empty:
        raise ValueError(f"No mapping found for {div} with sitout {Major_sitout}")
        
    has_strong_IND = div_mapping['has_strong_IND'].iloc[0]

    # >>> NEW: Pre-calculate the Aligned Split Weights from the Prior Row
    aligned_list = ['GRN', 'HMP', 'SPP'] # NEW
    p_aligned_total = prior_row[aligned_list].sum(axis=1).iloc[0] # NEW
    aligned_fixed_weights = {p: prior_row[p].iloc[0] / p_aligned_total for p in aligned_list} # NEW
    
    # 3. Get exact group expected means (theta)
    prop_lookup_party = 'LP' if Major_sitout == 'COAL' else Major_sitout
    row = proportions_df[(proportions_df['Major_sitout'] == prop_lookup_party) & 
                         (proportions_df['has_strong_IND'] == has_strong_IND)].iloc[0]
    
    if has_strong_IND == 1:
        groups = {
            'aligned': row['theta_aligned'], 
            'IND': row['theta_ind'], 
            'other_major': row['theta_other_major'], 
            'other': row['theta_other']
        }
    else:
        groups = {
            'aligned': row['theta_aligned'], 
            'other_major': row['theta_other_major'], 
            'other': row['theta_other']
        }
        
    # 4. Initialize the donations tracking Series (all zeros)
    donations = pd.Series(0.0, index=direct_result.index)
    
    # 5. Deterministically allocate to each group based on proportional weights
    for group_name, theta in groups.items():
        # Look up which parties belong to this group in this electorate
        group_parties = div_mapping.loc[div_mapping['group'] == group_name, 'PartyAb'].tolist()

        if group_parties == ['COAL']:
            valid_parties =['LP','NP']
        else:
            # Filter down to parties actually present in our data row
            valid_parties = [p for p in group_parties if p in direct_result.index and p != Major_sitout]


            
        # Calculate current share sum for proportional weighting
        if not valid_parties: # NEW
            import pdb; pdb.set_trace()
            continue # NEW

        # >>> CHANGE: Handle Aligned group with Fixed Weights, others with Proportional
        if group_name == 'aligned': # NEW
            # Use the fixed ratios from the prior row (GRN ~55%, HMP ~40%, SPP ~5%)
            weights = pd.Series([aligned_fixed_weights.get(p, 0.0) for p in valid_parties], index=valid_parties) # NEW
        else: # NEW
            # Standard Proportional weighting (IND, Other Major) based on current results
            current_shares = direct_result[valid_parties]
            group_total = current_shares.sum()
            
            if group_total > 0:
                weights = current_shares / group_total
            else:
                weights = pd.Series(1.0 / len(valid_parties), index=valid_parties)
            
        # The expected donation math: V * expected_group_share * party_weight
        donations[valid_parties] = V_sitout * theta * weights
        
    # Drop the sitout party from the output series so it represents purely the *gains*
    return donations.drop(labels=[Major_sitout])


def generate_tcp_accordion_html(tcp_vote_dict, matchup_list, total_sims=10000):
    # Dictionaries to swap Python codes to Display Names and Colors
    display_names = {
        'ON': 'ON', 'IND1': 'IND', 'LP': 'LIB', 
        'NP': 'NAT', 'ALP': 'ALP', 'GRN': 'GRN'
    }
    colors = {
        'ON': '#f46b09', 'IND1': '#64748b', 'LP': '#0509b3', 
        'NP': '#057e0d', 'ALP': '#FF0000', 'GRN': '#4daf4a'
    }
    
    pair_counts = Counter(matchup_list)

    matchup_indices = defaultdict(list)

    for i, pair in enumerate(matchup_list):
        p1, p2 = pair
        # Standardize key upfront
        dict_key = (p1, p2)
        matchup_indices[dict_key].append(i)
    
    # HTML Base with Scoped CSS
    html = [
        '<div class="tcp-methodology-accordion" style="font-family: \'Poppins\', sans-serif; max-width: 700px; margin: 2rem auto;">',
        '<style>',
        '  /* Main Accordion Wrapper: Pale Purple Background */',
        '  details.tcp-details { background: #faf5ff; border: 2px solid #e9d5ff; border-radius: 12px; box-shadow: 0 4px 12px rgba(147, 51, 234, 0.05); overflow: hidden; }',
        '  /* Banner */',
        '  details.tcp-details summary { background: #f3e8ff; padding: 16px 24px; font-weight: 800; color: #6419a8; cursor: pointer; list-style: none; display: flex; align-items: center; justify-content: space-between; font-size: clamp(0.85rem, 4vw, 1.1rem); border-bottom: 2px solid #e9d5ff; transition: background 0.2s; }',
        '  details.tcp-details summary:hover { background: #e9d5ff; }',
        '  details.tcp-details summary::-webkit-details-marker { display: none; }',
        '  /* Custom Dropdown Arrow */',
        '  details.tcp-details summary::after { content: "▼"; font-size: 0.9rem; color: #9333ea; }',
        '  details.tcp-details[open] summary::after { content: "▲"; }',
        '  /* Responsive 4-Column Grid - Adjusted for desktop word fit */',
        '  .tcp-grid { display: grid; grid-template-columns: 1.45fr 0.75fr 1.2fr 1.0fr; gap: 10px; padding: 16px 24px; align-items: center; border-bottom: 1px solid #e9d5ff; }',
        '  .tcp-grid:last-child { border-bottom: none; }',
        '  /* Increased text sizes for Desktop */',
        '  .tcp-header { font-size: 0.85rem; font-weight: 800; color: #6419a8; text-align: center; text-transform: uppercase; letter-spacing: 0.5px; }',
        '  .tcp-row { font-size: 1.05rem; font-weight: 700; color: #334155; text-align: center; }',
        '  .tcp-pill { padding: 6px 8px; border-radius: 6px; color: white; font-weight: 700; font-size: 0.9rem; display: inline-block; min-width: 50px; text-align: center; line-height: 1.2; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '  /* Note Box Trapped in Accordion */',
        '  .tcp-accordion-note { width: calc(100% - 48px); margin: 16px auto 24px auto; padding: 12px 16px; background: #faf5ff; border-left: 4px solid #c084fc; border-radius: 4px 8px 8px 4px; font-size: 0.85rem; color: #475569; line-height: 1.5; box-sizing: border-box; }',
        '  /* Mobile/Desktop Text Toggling */',
        '  .mobile-text { display: none; }',
        '  /* Mobile Squish Rules */',
        '  @media (max-width: 600px) {',
        '    .tcp-grid { grid-template-columns: 1.2fr 55px 85px 75px; padding: 12px 8px; gap: 4px; }',
        '    .desktop-text { display: none; }',
        '    .mobile-text { display: inline; }',
        '    .tcp-pill { min-width: 35px; font-size: 0.75rem; padding: 4px 4px; }',
        '    .tcp-header { font-size: 0.65rem; }',
        '    .tcp-row { font-size: 0.85rem; }',
        '    details.tcp-details summary { padding: 12px 14px; }',
        '    .tcp-accordion-note { width: calc(100% - 16px); margin: 12px auto 16px auto; padding: 10px 12px; }',
        '  }',
        '</style>',
        '<details class="tcp-details">',
        '  <summary>View Details of TCP Matchups</summary>',
        '  <div class="tcp-grid" style="background: #f3e8ff; border-bottom: 2px solid #e9d5ff; padding-bottom: 12px; padding-top: 12px;">',
        '    <div class="tcp-header" style="text-align: left;">TCP Matchup</div>',
        '    <div class="tcp-header" style="text-align: center;"><span class="desktop-text">Frequency</span><span class="mobile-text">FREQ.</span></div>',
        '    <div class="tcp-header" style="text-align: center;">Average TCP</div>',
        '    <div class="tcp-header" style="text-align: center;"><span class="desktop-text">Proportion Won</span><span class="mobile-text">PROP. WON</span></div>',
        '  </div>'
    ]
    
    for pair, count in pair_counts.most_common():
        p1, p2 = pair
        
        # Standardize key retrieval
        dict_key = (p1, p2) if (p1, p2) in tcp_vote_dict else (p2, p1)
        if dict_key not in tcp_vote_dict:
            continue
            
        party_a, party_b = dict_key
        votes = np.array(tcp_vote_dict[dict_key])
        
        # Determine frequency
        freq_pct = (count / total_sims) * 100
        
        # Calculate Conditional TCP and Proportion Won
        curr_matchup_indices = matchup_indices.get(pair, []) + matchup_indices.get(pair[::-1], [])
        
        if len(curr_matchup_indices) > 0:
            conditional_votes = votes[curr_matchup_indices]
            
            # Conditional Average TCP
            cond_overall_a = np.mean(conditional_votes) * 100
            cond_overall_b = 100.0 - cond_overall_a
            
            # Proportion Won (Conditional)
            win_prob_a = np.mean(conditional_votes > 0.5) * 100
            
            # Identify the victor to grab their designated color
            if win_prob_a >= 50:
                win_color = colors.get(party_a, '#94a3b8')
                win_text = f"{win_prob_a:.0f}%"
            else:
                win_color = colors.get(party_b, '#94a3b8')
                win_text = f"{(100 - win_prob_a):.0f}%"
        else:
            # Safe fallback
            cond_overall_a, cond_overall_b = 0.0, 0.0
            win_color = "#94a3b8"
            win_text = "N/A"
        
        # Mappings for pills
        name_a = display_names.get(party_a, party_a)
        name_b = display_names.get(party_b, party_b)
        color_a = colors.get(party_a, '#94a3b8')
        color_b = colors.get(party_b, '#94a3b8')
        
        # The Matchup Column (Pill UI)
        matchup_ui = f"""
        <div style="display: flex; align-items: center; justify-content: flex-start; gap: 8px;">
            <div class="tcp-pill" style="background: {color_a};">{name_a}</div>
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 800;">v</span>
            <div class="tcp-pill" style="background: {color_b};">{name_b}</div>
        </div>
        """
        
        # The Conditional TCP Column
        tcp_colored = f'<span style="color: {color_a};">{cond_overall_a:.1f}</span> <span style="color: #475569; font-weight: 800; margin: 0 4px;">-</span> <span style="color: {color_b};">{cond_overall_b:.1f}</span>'
        
        # The Proportion Won Column
        win_ui = f'<span style="color: {win_color}; font-weight: 800;">{win_text}</span>'
        
        # Append Row
        html.append(f'  <div class="tcp-grid">')
        html.append(f'    <div>{matchup_ui}</div>')
        html.append(f'    <div class="tcp-row" style="color: #9333ea; text-align: center;">{freq_pct:.1f}%</div>')
        html.append(f'    <div class="tcp-row" style="text-align: center;">{tcp_colored}</div>')
        html.append(f'    <div class="tcp-row" style="text-align: center;">{win_ui}</div>')
        html.append(f'  </div>')
    
    # Append the Note Block at the bottom, trapped inside the <details> tag
    note_html = """
    <div class="tcp-accordion-note">
        <strong style="color: #6419a8;">Note on Average TCP:</strong> 
        The 'Average TCP' column above calculates the <em>conditional</em> Two-Candidate Preferred vote: the average TCP in the simulations where this specific matchup occurs. This excludes cases where either candidate was eliminated early.
    </div>
    """
    html.append(note_html)
        
    html.append('</details>')
    html.append('</div>')
    
    return "\n".join(html)


def generate_election_flow_html(flow_df):


    colour_df = pd.read_csv("Party Colours.csv")

    colour_df.loc[:,'Party'] = colour_df['Party'].str.replace("’", "'", regex=False)

    colors = colour_df.set_index("Party")["Colour"].to_dict()


    party_name_dict = {
        'ALP':'Labor',
        'NP': 'National',
        'COAL': 'Coalition',
        'ON' : 'One Nation',
        'GRN' : 'Greens',
        'HMP' : 'Legalise Cannabis',
        'FFPA' : 'Family First',
        'ASP' : 'Shooters Fishers & Farmers',
        'LP' : 'Liberal' ,
        'TOP' : 'Trumpet of Patriots' ,
        'IND' : 'Independent' ,
        'AJP' : 'Animal Justice',
        'SOPA' :'FUSION' ,
        'LDP' : 'Libertarian' ,
        'AUD' : 'Australian Democrats',
        'CEC' : 'Citizens Party',
        'VNS' : 'Victorian Socialists',
        'IMO' : 'HEART',
        'GRPF' : "People's First",
        'AUC' : 'Christians',
        'SAL' : 'Socialist Alliance',
        'IAP' : 'Indigenous-Aboriginal',
        'GAP' : 'Great Australian Party',
        'KAP' : "Katter's Australian",
        'CLP' : 'Country Liberal',
        'LNP' : 'Liberal National',
        'XEN' : 'Centre Alliance',
        'IND1' : 'Independent 1',
        'IND2' : 'Independent 2',
        'IND3' : 'Independent 3',
        'IND4' : 'Independent 4',
        'IND5' : 'Independent 5',
        'OTH' : 'Other',
        'IND1Goldstein': 'Ind. Zoe Daniel',
        'IND1Calare': 'Ind. Kate Hook',
        'IND2Calare': 'Ind. Andrew Gee',
        'IND1Kooyong': 'Ind. Monique Ryan',
        'IND3Mackellar': 'Ind. Sophie Scamps',
        'IND1Moore': 'Ind. Ian Goodenough',
        'IND1McPherson': 'Ind. Erchana Murray-Barlett',
        'IND1Wannon': 'Ind. Alex Dyson',
        'IND1Curtin' : 'Ind. Kate Chaney',
        'IND1Indi' : 'Ind. Helen Haines',
        'IND1Clark' : 'Ind. Andrew Wilkie',
        'IND1Moncrieff' : 'Ind. Nicole Arrowsmith',
        'IND2Moore' : 'Ind. Nathan Barton',
        'IND2Bradfield': 'Ind. Nicolette Boele',
        'IND2Berowra' : 'Ind. Tina Smith',
        'IND1Forrest' : 'Ind. Sue Chapman',
        'IND1Sturt' : 'Ind. Verity Cooper',
        'IND1Gilmore' : 'Ind. Kate Dezarnaulds',
        'IND1Casey' : 'Ind. Claire Miles',
        'IND1Franklin' : 'Ind. Peter George',
        'IND2Cowper' : 'Ind. Caz Heise',
        'IND1Fremantle' : 'Ind. Kate Hulett',
        'IND1Fisher' : 'Ind. Kerryn Jones',
        'IND1Grey' : 'Ind. Anita Kuss',
        'IND1Monash' : 'Ind. Russell Broadbent',
        'IND2Monash' : 'Ind. Deb Leonard',
        'IND1Lyne' : 'Ind. Jeremy Miller',
        'IND1Farrer' : 'Ind. Michelle Milthorpe',
        'IND1Deakin' : 'Ind. Jess Ness',
        'IND1Bean' : 'Ind. Jessie Price',
        'IND5Riverina' : 'Ind. Jenny Rolfe',
        'IND1Solomon' : 'Ind. Phil Scott',
        'IND1Flinders' : 'Ind. Ben Smith',
        'IND1Dickson' : 'Ind. Ellie Smith',
        'IND1Fairfax' : 'Ind. Francine Wiig',
        'IND1Fowler' : 'Ind. Dai Le',
        'IND2Wentworth' : 'Ind. Allegra Spender',
        'IND1Groom' : 'Ind. Suzie Holt',
        'IND1Warringah' : 'Ind. Zali Steggall',
        'CYA': 'Trumpet of Patriots',
        'GRPF': "People's First",
        'FFPA': 'Family First',
        'LTP': 'Libertarian',
        'SPP': 'Sustainable Australia'
    }

    site_abbreviation_dict = {
        "Labor": "LAB",
        "Liberal": "LIB",
        "National" : 'NAT',
        "Coalition" : 'COAL',
        "Other" : 'OTH',
        "Greens": "GRN",
        "One Nation" : 'ON',
        'Legalise Cannabis': 'LCP',
        'Family First': 'FFP',
        'Shooters Fishers & Farmers': 'SFF',
        'Trumpet of Patriots': 'TOP',
        'Independent': 'IND',
        'Animal Justice': 'AJP',
        'FUSION': 'FUSN' ,
        'Libertarian':'LBT' ,
        'Australian Democrats': 'AUD',
        'Citizens Party': 'CIT',
        'Victorian Socialists':'VSOC',
        'HEART':'HRT',
        "People's First":'PPF',
        'Christians':'AUC',
        'Socialist Alliance':'SA',
        'Indigenous-Aboriginal': 'IAP',
        'Great Australian Party':'GAP',
        "Katter's Australian": 'KAP',
        'Country Liberal': 'CLP',
        'Liberal National': 'LNP',
        'Centre Alliance': 'CA',
        'Independent 1' : 'IND1',
        'Independent 2' : 'IND2',
        'Independent 3' : 'IND3',
        'Independent 4' : 'IND4',
        'Independent 5' : 'IND5',
        'Sustainable Australia': 'SAP'
    }

    Independents = {key: value for key, value in party_name_dict.items() if key.startswith('IND') and len(key) > 5}

    div = 'Farrer'

    for ind in Independents.keys():
        IND_full_name = Independents[ind]
        site_abbreviation_dict[IND_full_name] = IND_full_name[:6].upper().replace('.','') + '.' + IND_full_name.split(' ')[-1][0] + '.' # initials


    
    # Configuration for the 8 data columns
    # Background shadings removed. 'class' added to manage flow spacing.
    
    columns_config = [
        {'title': '2025<br>Result', 'id_key': 'Baseline', 'style': 'color: #0f172a;', 'class': 'spacer-right', 'type': 'abs', 'val': 'Baseline_2025', 'prev': None, 'border': ''},
        {'title': 'Incumbency<br>Adjust.', 'id_key': 'Incumbency', 'style': '', 'class': '', 'type': 'delta', 'val': 'Incumbency_Adj', 'prev': 'Baseline_2025', 'border': ''},
        {'title': 'Parties<br>Contesting', 'id_key': 'Field', 'style': '', 'class': '', 'type': 'delta_new', 'val': 'Field_Changes', 'prev': 'Incumbency_Adj', 'border': ''},
        {'title': 'National<br>Poll Swing', 'id_key': 'ALR', 'style': '', 'class': '', 'type': 'delta', 'val': 'National_ALR', 'prev': 'Field_Changes', 'border': ''},
        {'title': 'Labor<br>Realloc.', 'id_key': 'Realloc', 'style': '', 'class': 'spacer-right', 'type': 'delta', 'val': 'Labor_Redist', 'prev': 'National_ALR', 'border': ''},
        {'title': 'Model<br>Average', 'id_key': 'ModelAvg', 'style': 'color: #0f172a;', 'class': '', 'type': 'abs', 'val': 'Labor_Redist', 'prev': None, 'border': 'border: 2px solid #000000;'},
        {'title': 'Seat Poll<br>Avg', 'id_key': 'SeatPoll', 'style': 'color: #0f172a;', 'class': '', 'type': 'abs', 'val': 'Seat_Polling', 'prev': None, 'border': 'border: 2px solid #ffffff;'},
        {'title': 'Final<br>Average', 'id_key': 'Final', 'style': 'color: #9333ea;', 'class': '', 'type': 'final_abs', 'val': 'Final_Average', 'prev': None, 'border': 'border: 2px solid #9333ea;'}
    ]

    html = """
        <div style="max-width: 1250px; margin: 2rem auto 1rem auto; font-family: 'Poppins', sans-serif;"> <h2 style="color: #6419a8; font-weight: 800; font-size: 2rem; margin-bottom: 8px;">Deconstructing the Model</h2> <div style="display: inline-block; padding: 6px 16px; border-radius: 6px; color: #475569; font-size: 0.9rem; font-weight: 500; letter-spacing: 0.2px;"> Tracking how incumbency, party changes, and polling swings shape each party's final projected vote. </div>
    <div class="model-flow-container">
        <div class="model-flow-grid">
            <div class="flow-header party-label-header" style="color: #334155;">Party</div>"""
            
    for col in columns_config:
        html += f'\n            <div class="flow-header {col["class"]}" style="{col["style"]}">{col["title"]}</div>'

    html += "\n"

    sort_weights = flow_df['Final_Average']
    flow_df = flow_df.loc[sort_weights.sort_values(ascending=False).index]

    for party, row in flow_df.iterrows():
        # 1. Grab the true name and lookup the color FIRST
        true_full_name = party_name_dict.get(party, party)
        party_hex = colors.get(true_full_name, colors.get(party, '#64748b'))
        
        # 2. NOW override the display name for the UI
        if party == 'IND1': display_name = 'Ind. Milthorpe'
        elif party == 'IND2': display_name = 'Ind. Woodward'
        elif party == 'IND3': display_name = 'Ind. Pappin'
        elif party in ['ASP', 'SFF']: display_name = 'Shooters F&F'
        else: display_name = true_full_name
        
        pill_full = f"background-color: {party_hex}; color: white;"
        pill_light = f"background-color: {party_hex}; color: white; opacity: 0.6;"
        
        label_opacity = 'opacity: 0.8;' if party == 'ALP' else 'opacity: 1;'
        html += f'\n            <div class="party-label" style="color: {party_hex}; {label_opacity}">{display_name}</div>'

        for col in columns_config:
            val = row.get(col['val'], 0.0)
            cell_class = f"data-cell {col['class']}"

            pill_id = f'id="cell-{party}-{col["id_key"]}"'
            
            if col['type'] in ['abs', 'final_abs']:
                if val == 0:
                    html += f'\n            <div class="{cell_class}"><div {pill_id} class="vote-pill pill-ghost">0.0%</div></div>'
                else:
                    html += f'\n            <div class="{cell_class}"><div {pill_id} class="vote-pill" style="{pill_full} {col["border"]}">{val:.1f}%</div></div>'
            else:
                prev_val = row.get(col['prev'], 0.0)
                delta = val - prev_val
                
                if col['type'] == 'delta_new' and prev_val == 0 and val > 0:
                    html += f'\n            <div class="{cell_class}"><div {pill_id} class="vote-pill" style="{pill_full} {col["border"]}">{val:.1f}%</div></div>'
                elif val == 0 and prev_val == 0:
                    html += f'\n            <div class="{cell_class}"><div {pill_id} class="vote-pill pill-ghost">-</div></div>'
                elif abs(delta) < 0.05:
                    html += f'\n            <div class="{cell_class}"><div {pill_id} class="vote-pill" style="{pill_light} opacity: 0.4; {col["border"]}">-</div></div>'
                else:
                    html += f'\n            <div class="{cell_class}"><div {pill_id} class="vote-pill" style="{pill_light} {col["border"]}">{delta:+.1f}</div></div>'

    html += """
        </div>
    </div>
    """
    return html


def generate_output_tables_plots(combined_FP_df, per_electorate_winners, per_electorate_tcp_pair, per_electorate_winning_tcp, per_electorate_tcp_pair_vote, election_year, National_prior_dict, polling_avg_dict, prior_df_active_dict, prior_df_active, Farrer_full_prior, poll_df, Prior_estimates_dict_active, seat_alr_bases_dict, GLOBAL_CSVs, n_simulations, div_results_dict = {}):


    tcp_table_html = generate_tcp_accordion_html(per_electorate_tcp_pair_vote['Farrer'], per_electorate_tcp_pair['Farrer'], total_sims=n_simulations)

    final_export = tcp_table_html
    with open("TCP_table.html", "w", encoding="utf-8") as f:
        f.write(final_export)

    


    tcp_pair_array = np.array(per_electorate_tcp_pair['Farrer'])
    winning_tcp_array = np.array(per_electorate_winning_tcp['Farrer'])
    winning_p_array = np.array(per_electorate_winners['Farrer'])


    calculate_estimated_tcp(tcp_pair_array, per_electorate_winners, winning_tcp_array, 'ON','IND1')


    winner_counts = Counter(per_electorate_winners['Farrer'])
    
    win_proportions_df = pd.DataFrame(winner_counts.items(), columns=['Party', 'Count']).assign(**{'Win Percentage': lambda x: x['Count']/x['Count'].sum()}).drop(columns='Count').sort_values('Win Percentage', ascending=False).reset_index(drop=True)

    make_horizontal_bars(['Farrer'], win_proportions_df, to_json = 1)


    make_single_electorate_violin_plot(combined_FP_df, tcp_pair_array, winning_tcp_array, winning_p_array, group_OTH=False, to_plot = True, div_results_dict=div_results_dict)
    #make_single_electorate_violin_plot(combined_FP_df, to_plot=True, group_OTH=True)


    total = sum(winner_counts.values())

    on_win_pct  = winner_counts['ON']  / total*100
    ind_win_pct = winner_counts['IND1'] / total*100
    lib_win_pct = winner_counts['LP']  / total*100
    nat_win_pct = winner_counts['NP']  / total*100


    summary_html = f"""
   <div style="max-width: 850px; margin: 0 auto 40px auto; text-align: center; color: #334155; line-height: 1.6; font-size: clamp(0.9rem, 3vw, 1.1rem);padding: 0 10px;">
        <strong style="color: #f46b09;">One Nation</strong> is the current favorite, winning <strong style="color: #f46b09;">{on_win_pct:.0f}%</strong> of the 10,000 simulations. 
        The <strong style="color: #64748b;">Independent</strong> Michelle Milthorpe is the second favourite, winning <strong>{ind_win_pct:.0f}%</strong>, 
        while the <strong style="color: #0509b3;">Liberals </strong> ({lib_win_pct:.0f}%) and <strong style="color: #057e0d;">Nationals </strong> ({nat_win_pct:.1f}%)
        are the only other parties with statistically plausible paths to victory.
    </div>
    """

    final_export = summary_html
    with open("Summary.html", "w", encoding="utf-8") as f:
        f.write(final_export)

    
    National_prior_alr = df_to_alr( National_prior_dict[election_year].rename(index = {0:'Farrer'}), ref_col = 'COAL')
    polling_alr = df_to_alr(polling_avg_dict[election_year], ref_col='COAL')
    Prior_estimates_alr = df_to_alr(prior_df_active, ref_col='COAL')

    # calculate 3 swings
    prior_polling_votes = pd.Series( {'COAL':0.3843,'ALP':0.1582,'GRN':0.0453,'ON':0.0778,'IND':0.2256,'OTH':0.1088}) # Fundamentals votes
    polling_avg -  National_prior_dict[election_year].rename(index = {0:'Farrer'}) # Uniform National
    polling_avg/National_prior_dict[election_year].rename(index = {0:'Farrer'})*prior_df_active - prior_polling_votes # proportional 
    alr_to_simplex_vectorized( Prior_estimates_alr + polling_alr - National_prior_alr.rename(index = {0:'Farrer'}), ref_col = 'COAL') - prior_polling_votes # alr

    import pdb; pdb.set_trace()
    # DETOUR: calculate polling shift
    Prior_estimates_alr + polling_alr - National_prior_alr.rename(index = {0:'Farrer'})
    Prior_estimates_alr_GRN_penalty = Prior_estimates_alr.copy()
    Prior_estimates_alr_GRN_penalty['GRN'] -= 0.28

    Poll_adjusted = alr_to_simplex_vectorized( Prior_estimates_alr + polling_alr - National_prior_alr.rename(index = {0:'Farrer'}), ref_col = 'COAL')
    Poll_adjusted_GRN_penalty = alr_to_simplex_vectorized( Prior_estimates_alr_GRN_penalty + polling_alr - National_prior_alr.rename(index = {0:'Farrer'}), ref_col = 'COAL')


    prior = Farrer_full_prior.mean()
    penalty_result = 0
    direct_result = 0

    for penalty in [0,1]:
        row = Poll_adjusted.loc['Farrer'].copy() if not penalty else Poll_adjusted_GRN_penalty.loc['Farrer'].copy()

        # --- Split IND across detailed IND-type parties ---

        C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
        C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]

        beta0,beta1 = C200_ratio_lm_params
        logit_mu = beta0 + beta1 * np.log(row['IND']*100)

        C200_ratio = 1 / (1 + np.exp(-logit_mu))

        row['IND2'] = row['IND'] * (1-C200_ratio)/2
        row['IND3'] = row['IND'] * (1-C200_ratio)/2
        row['IND1'] = row['IND'] * C200_ratio
        row = row.drop('IND')

        # --- Split OTH across remaining minor parties (excluding majors + ON + IND) ---
        exclude = ['ALP', 'COAL', 'GRN', 'ON', 'IND']
        oth_parties = [p for p in prior.index if p not in exclude]

        oth_weights = prior[oth_parties] / prior[oth_parties].sum()
        for p in oth_parties:
            row[p] = row.get(p, 0) + row['OTH'] * oth_weights[p]
        row = row.drop('OTH')

        # --- Split COAL into LP and NP ---
        np_share = 0.220224
        row['NP'] = row['COAL'] * np_share
        row['LP'] = row['COAL'] * (1 - np_share)
        row = row.drop('COAL')

        # --- Final result as DataFrame ---
        result = row.to_frame().T

        mapping_df = GLOBAL_CSVs['mapping_df']
        proportions_df = GLOBAL_CSVs['proportions_df']

        alp_donations = calculate_mean_sitout_donations(
            direct_result=result.mean(),
            div='Farrer',
            Major_sitout='ALP',
            mapping_df=mapping_df,
            proportions_df=proportions_df,
            prior_row = Farrer_full_prior
        )

        # 2. Add them to the baseline to get the new total
        post_redistribution_shares = result.mean().drop(labels=['ALP']) + alp_donations
        post_redistribution_shares['ALP'] = 0.0 # ALP is now physically empty

        if penalty:
            penalty_result = post_redistribution_shares
        else:
            direct_result = post_redistribution_shares


    # seat poll data
    seat_polled = poll_df.iloc[:,3:].mean()
    seat_polled = seat_polled[seat_polled>0].rename(index={'COAL':'LP','IND':'IND1','NAT':'NP'})

    remaining_unpolled = [p for p in penalty_result[penalty_result>0].index if p not in seat_polled.index]

    prior_unpolled = prior.loc[[p for p in remaining_unpolled if p in prior.index]]
    prior_unpolled['IND2'] = 0.0094
    prior_unpolled['IND3'] = 0.0094

    prior_unpolled_norm = prior_unpolled/prior_unpolled.sum()
    
    prior_unpolled_est = prior_unpolled_norm* seat_polled['OTH']

    seat_polled = seat_polled.drop('OTH')

    seat_full = pd.concat([seat_polled,prior_unpolled_est[remaining_unpolled]])

    import pdb; pdb.set_trace()



    # The immutable Farrer constants (scaled to percentages)
    farrer_static_data = {
        'PartyAb': [
            'LP', 'ALP', 'IND1', 'GRN', 'ON', 'ASP', 
            'GRPF', 'FFPA', 'CYA', 'HMP', 'SPP', 'IND2', 'IND3','NP'
        ],
        # 1. The 2025 Result
        'Baseline_2025': [
            43.41, 15.09, 19.96, 4.93, 6.60, 3.47, 
            2.02, 2.15, 2.37, 0.0, 0.0, 0.0, 0.0,0.0
        ],
        # 2. Sussan Ley's personal vote removed
        'Incumbency_Adj': [
            38.61, 16.10, 21.37, 5.27, 7.72, 3.74, 
            2.17, 2.44, 2.57, 0.0, 0.0, 0.0, 0.0,0.0
        ],
        # 3. New candidate field (CYA leaves, minors enter)
        'Field_Changes': [
            29.97, 15.82, 20.68, 4.53, 7.78, 3.07, 
            1.97, 2.17, 0.0, 3.29, 0.39, 0.94, 0.94, 8.46
        ] # NP takes 0.2202235 of COAL vote
    }

    # Create the static dataframe
    static_df = pd.DataFrame(farrer_static_data).set_index('PartyAb')

    dynamic_data = pd.DataFrame({
        'National_ALR': result.mean(), 
        'Labor_Redist': post_redistribution_shares.drop('ALP'),        
        'Seat_Polling': seat_full,    
        'Final_Average': combined_FP_df.mean()    
    })

    # Staple the immutable history to the dynamic simulation
    flow_df = static_df.join(dynamic_data*100, how='outer').fillna(0.0)

    # Generate the HTML using the function we built earlier!
    html_table = generate_election_flow_html(flow_df)

    css_block = """
    <style>
        /* Main Container: Restored generous max-width for desktop */
        .model-flow-container { font-family: 'Poppins', sans-serif; max-width: 1250px; margin: 2rem auto; overflow-x: auto; padding-bottom: 10px; }
        
        /* Force visible horizontal scrollbar */
        .model-flow-container::-webkit-scrollbar {
            height: 8px;
            -webkit-appearance: none;
        }
        /* 1. The Container: Needs to be a flex parent to center its child */
        .model-flow-container { 
            font-family: 'Poppins', sans-serif; 
            max-width: 1250px; 
            margin: 2rem auto; /* Centers the container itself */
            overflow-x: auto; 
            padding-bottom: 10px;
            display: flex; 
            flex-direction: column;
            align-items: center; /* Centers the grid horizontally if there is extra space */
        }

        /* 2. The Grid: Needs to know it shouldn't try to be wider than necessary */
        .model-flow-grid { 
            display: grid; 
            grid-template-columns: 150px repeat(8, minmax(85px, 1fr)); 
            gap: 6px; 
            min-width: 1000px; 
            margin: 0 auto; /* Ensures the grid itself stays centered in the scroll area */
            align-items: center; 
        }
        /* Grid Layout Desktop: Restored generous spacing with minmax(85px, 1fr) */
        .model-flow-grid { display: grid; grid-template-columns: 150px repeat(8, minmax(85px, 1fr)); gap: 6px; min-width: 1000px; align-items: center; justify-content: start; }
        
        /* Headers Desktop: Restored larger, more readable font size */
        .flow-header { font-weight: 700; color: #475569; font-size: 0.85rem; text-align: center; padding-bottom: 10px; border-bottom: 2px solid #cbd5e1; margin-bottom: 10px; line-height: 1.25; }
        /* --- STICKY FIRST COLUMN --- */
        .party-label-header, .party-label { 
            position: sticky; 
            left: 0; 
            background-color: #ece4f6; /* CHANGE THIS to your site's background color */
            z-index: 10; /* Ensures it stays on top of everything */
            
            /* These lines fix the "gaps" */
            height: 100%;
            margin: 0 !important;
            padding: 1px 1px; 
            display: flex;
            align-items: center;
            box-sizing: border-box;
            border-right: 1px solid #e2e8f0; /* Optional: adds a thin line to separate the frozen part */
        }
        
        /* Updated shadow to look cleaner */
        .party-label-header { 
            border-bottom: 2px solid #94a3b8 !important; 
            font-size: 0.95rem; 
            font-weight: 700;
        }
        
        .party-label { 
            font-weight: 700; 
            font-size: 0.95rem; 
            line-height: 1.2;
        }
        
        /* Cells & Data Display Desktop */
        .data-cell { display: flex; justify-content: center; }
        .vote-pill { box-sizing: border-box; border-radius: 8px; padding: 6px 12px; font-weight: 600; font-size: 0.9rem; width: 70px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .pill-ghost { background-color: #f1f5f9; color: #94a3b8; box-shadow: none; border: 1px dashed #cbd5e1; }
        
        /* Visual Breaks Desktop */
        .spacer-right { margin-right: 16px; }

        /* --- MOBILE RESPONSIVENESS --- */
        @media (max-width: 768px) {
            /* 1. Add the tiny margin to the container so it doesn't touch the screen edge */
            .model-flow-container { margin: 1.5rem 10px; }
            
            /* 2. Shrink the party column to 85px, and squeeze data columns down to 52px */
            .model-flow-grid { grid-template-columns: 85px repeat(8, minmax(52px, 60px)); min-width: 550px; gap: 2px; }
            
            /* 3. Tighter, smaller bubbles */
            .vote-pill { width: 50px; padding: 3px 4px; font-size: 0.75rem; border-radius: 6px; }
            
            /* Aggressive scale down for small screens */
            .flow-header { font-size: 0.65rem; padding-bottom: 4px; letter-spacing: -0.2px; line-height: 1.1; margin-bottom: 6px; }
            .party-label { font-size: 0.8rem; }
            .spacer-right { margin-right: 8px; }
        }
    </style>
    """

    # Export
    final_export = css_block + html_table
    with open("Farrer_Flow_Table.html", "w", encoding="utf-8") as f:
        f.write(final_export)

    



    print(combined_FP_df.mean())
    Farrer_byelection_adjusted_avg_by_pp_id(election_year, prior_df_active_dict, polling_avg_dict,National_prior_dict, Prior_estimates_dict_active,seat_alr_bases_dict)


    return 1

def simulate_electorate(n_simulations, Days_to_election, polling_avg, seat_polls_df, election_year = 'Byelection', w = 0.54, B = 1.2, prior_alpha_fixed = 11, poll_alpha_fixed = 20, plot_table_outputs = True, div_results_dict = {}):

    elections = ['2016', '2019', '2022','2025','Byelection']

    
    div_order_dict = {}
    prior_df_active_dict = {}
    polling_avg_dict = {}
    National_prior_dict = {}

    Prior_estimates_dict_active_per_election = {}
    Results_dict_active_per_election = {}

    seat_alr_bases_dict = {}
    n_seat_polls_dict = {}

    byelection_group_structure = []

    GLOBAL_CSVs = {
        "multiple_INDs_df": {election_year: pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None) for election_year in elections},
        "C200_IND_splits": pd.read_csv("Independent_splits_multiple.csv", index_col=None),
        "C200_IND_positions_df": pd.read_csv("C200_IND_positions_df.csv", index_col=None),
        "NP_ratios_curr": pd.read_csv("NP_ratio_estimated_df.csv", index_col=None),
        "proportions_df": pd.read_csv('Missing_major_reallocation_proportions.csv'),
        "mapping_df": pd.read_csv('Missing_major_allocation.csv')
    }



    with open("Farrer_byelection_covms.pkl", "rb") as f:
        data = pickle.load(f)
    
    Nat_poll_covm = adjust_covm_for_time(
        final_day_covm=data["Nat_poll_covm"], 
        days_to_election=Days_to_election, 
        params_dict=OPT_EXP_DECAY_PARAMETERS
    )   


    Seat_poll_covm_dict = {election_year: data["Seat_poll_covm"]}
    Electorate_residual_covm_dict = {election_year: data["Electorate_residual_covm"]}
    Nat_poll_covm_dict = {election_year: Nat_poll_covm}


    

    Seat_poll_year_dict = {election_year: seat_polls_df}       


    # 1. Extract active electorate(s) and map them to dictionaries
    div_order = ['Farrer']

    Farrer_full_prior = pd.read_csv('By_election_Fundamentals_after_2025.csv')
    Farrer_full_prior = Farrer_full_prior.set_index('PartyAb')['FP_Votes'].to_frame().T
    Farrer_prior = Farrer_full_prior[['COAL','ALP','GRN','ON','IND']]
    Farrer_prior = Farrer_prior.assign(OTH=(1 - Farrer_prior.sum(axis=1)).iloc[0])
    Farrer_prior = Farrer_prior.reset_index(drop=True)
    Farrer_prior.index =['Farrer']
    prior_df_active = Farrer_prior

    div_order_dict[election_year] = div_order
    prior_df_active_dict[election_year] = prior_df_active

    # 2. Setup Polling Average
    # Directly indexing 'Farrer2026' and storing it in the polling dictionary like a general election
    #polling_avg = pd.DataFrame([[0.226,0.297,0.125,0.236,0.072,0.044]], columns=['COAL','ALP','GRN','ON','IND','OTH'], index=['Farrer']) # Day 9
    # Day 14: [0.222,0.296,0.126,0.238,0.072,0.046] # 3.9 -> 4.6
    # Day 10: [0.222,0.297,0.124,0.242,0.072,0.043] # 3 -> 4.3
    # Day 9:  [0.226,0.297,0.125,0.236,0.072,0.044] # 3.4 -> 4.4
    # Day 7:  [0.222,0.298,0.126,0.237,0.072,0.045] # 3.6 -> 4.5
    # pd.DataFrame([[0.2228,0.3071,0.1259,0.227,0.072,0.0452]], columns=['COAL','ALP','GRN','ON','IND','OTH'], index=['Farrer'])


    polling_avg_dict[election_year] = polling_avg

    # 3. Setup National Prior
    National_prior_dict[election_year] = pd.DataFrame([[0.3456,0.3182,0.122,0.064,0.072,0.0782]], columns = ['ALP','COAL','GRN','ON','IND','OTH']) # for Farrer

    # 4. Process Results

    Farrer_candidates = pd.read_csv('Farrer_candidates.csv', index_col = ['PartyAb'])['FP_Votes'].to_frame().T
    Results_dict = {'Farrer': Farrer_candidates}

    # 5. Retrieve Prior Estimates
    Prior_estimates_dict = {'Farrer': Farrer_full_prior}

    # 6. Filter to Active Electorate and finalize storage
    Prior_estimates_dict_active = {div: Prior_estimates_dict[div] for div in div_order}
    Results_dict_active = {div: Results_dict.get(div) for div in div_order}

    Prior_estimates_dict_active_per_election[election_year] = Prior_estimates_dict_active
    Results_dict_active_per_election[election_year] = Results_dict_active


    # seat alr base - prepared deterministically outside simulation loop!
    # Assumes div_order matches the order of electorates in nat_alr_sims

    n_seat_polls_dict[election_year] = {}
    seat_alr_bases_dict[election_year] = {}

    curr_seat_polls = Seat_poll_year_dict[election_year]
    
    for div in div_order:

        # 1. Isolate the seat poll for this division
        poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate'] == div].set_index('Electorate')

        if poll_df.empty:
            n_seat_polls_dict[election_year][div] = 0
            seat_alr_bases_dict[election_year][div] = None # This triggers the zero-weighting of seat polls in function combine_models_alr_precision_weighted
            continue
        
        # 2. Replicate your logic: Normalize and Group
        curr_row = poll_df.iloc[:, 2 + ('byelection_year' in poll_df.columns):]
        curr_row_normalised = curr_row.div(curr_row.sum(axis=1), axis=0)
        
        # Take the mean of polls for this seat (if multiple polls exist)
        CAGOIO_row = curr_row_normalised[['COAL','ALP','GRN','ON','IND']].copy()
        CAGOIO_row.loc[:,'COAL'] +=  curr_row_normalised.loc[:,'NAT'].iloc[0]
        CAGOIO_row.loc[:,'Other'] = 1 - CAGOIO_row.sum(axis=1).iloc[[0]]
        CAGOIO_row = CAGOIO_row.loc[div]


        # adjust IND so that it is a global IND category to match national poll model
        C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
        C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]
        beta0,beta1 = C200_ratio_lm_params
        logit_mu = beta0 + beta1 * np.log(CAGOIO_row['IND']*100)
        C200_ratio = 1 / (1 + np.exp(-logit_mu))
        diff = (CAGOIO_row['IND'] / C200_ratio) - CAGOIO_row['IND']
        CAGOIO_row['Other'] -= diff
        CAGOIO_row['IND'] /= C200_ratio

        missing_party = 'ALP' # Farrer

        # 3. Drop missing party if necessary
        if missing_party is not None and missing_party in CAGOIO_row.index:
            CAGOIO_row = CAGOIO_row.drop([missing_party])
        
        ref_col = 'ALP' if missing_party == 'COAL' else 'COAL'
        
        # 4. Convert to ALR: [ALP, GRN, Other] / COAL
        # Resulting Series index: ['ALP', 'GRN', 'Other']
        alr_base = np.log(CAGOIO_row.drop(ref_col) / CAGOIO_row[ref_col])

        n_seat_polls_dict[election_year][div] = poll_df['n_polls'].iloc[0] # 1,2,3

        seat_alr_bases_dict[election_year][div] = alr_base.values



    combined = simulate_combined_models_CAGO(
        n_simulations, election_year, prior_df_active_dict, polling_avg_dict,
        National_prior_dict, byelection_group_structure,
        seat_alr_bases_dict, n_seat_polls_dict,
        Nat_poll_covm_dict, Electorate_residual_covm_dict, 
        Seat_poll_covm_dict, Farrer_full_prior, GLOBAL_CSVs, prior_alpha=prior_alpha_fixed, B=B
    )
    
    # 2. Generate both prior and seat-poll expansions
   
    prior_expand = expand_divisions_using_prior(
                combined[0],
                combined[1],
                Prior_estimates_dict_active_per_election[election_year],
                Results_dict_active_per_election[election_year],
                election_year=election_year,
                alpha_scalar=prior_alpha_fixed
            )[0]
    if not seat_polls_df.empty:
        seat_expand = expand_divisions_using_seat_polls(
                    combined[0],
                    Seat_poll_year_dict[election_year],
                    Prior_estimates_dict_active_per_election[election_year],
                    Results_dict_active_per_election[election_year],
                    election_year,
                    prior_alpha=prior_alpha_fixed,
                    poll_alpha = poll_alpha_fixed,
                    major_realloc_data = combined[1],
                    GLOBAL_CSVs = GLOBAL_CSVs,
                    n_simulations = n_simulations
                )[0]
    
    # 3. Apply best_w to create the final heldout ensemble
    n_seat = int(w* n_simulations) 

    if not seat_polls_df.empty:
        final_sims = np.vstack([seat_expand[div][:n_seat], prior_expand[div][n_seat:]])
    else:
        final_sims = prior_expand[div]




    div = 'Farrer'
    Ballot_order = Results_dict_active['Farrer'].columns

    combined_FP_df = pd.DataFrame(final_sims, columns = Ballot_order)




    sigma_joint, sigma_ind = 4.2, 0.5 + 4 # Additional scaling TBD

    # adjust sigma_joint 

    with open(f"TCP_pair_category_dict_for_2028.pkl", "rb") as f:
        proportions_transferred_to_first = pickle.load(f)
    party_category_dict = make_party_category_dict()
    party_to_category_centered_IND = {k: ('IND' if v == 'Centre' else v) for k, v in party_category_dict.items()}


    df = proportions_transferred_to_first['Farrer']


    df.loc[1,'HMP'] = 37 # taken from Indi - unfortunately not Senate-adjusted (no IND in Senate!)
    df.loc[1,'LP'] = 85
    df.loc[1,'NP'] = 85 # increasing competition with ON
    #df.loc[1,'ON'] = 58
    df.loc[6,'IND'] = 75 # Flinders SA 2026
    df.loc[9,'LP'] = 33 # Stuart SA 2026
    df.loc[9,'NP'] = 33 # Stuart SA 2026
    df.loc[9,'HMP'] = 63

    # try adjust minor parties as well
    #df.loc[9,'GRPF'] = 40
    #df.loc[9,'FFPA'] = 40
    #df.loc[9,'ASP'] = 45
    proportions_transferred_to_first['Farrer'] = df

    chosen_top_2_tcps = [('ON','IND1'), ('ON','LP'), ('ON','NP'), ('IND1','LP'), ('IND1','NP'),('LP','NP')]

    per_electorate_winners, per_simulation_winners, per_electorate_tcp_pair, per_electorate_winning_tcp, per_electorate_runners_up, per_electorate_leapfrogs, per_electorate_tcp_pair_vote = distribution_to_top_2({div:final_sims}, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind, allow_leapfrog = True, chosen_top_2_tcps=chosen_top_2_tcps)
    
    #make_tcp_violin_plot(per_electorate_tcp_pair_vote)

    print(Counter(per_electorate_winners['Farrer']))
    #print(Counter(per_electorate_tcp_pair['Farrer']))

    
    if plot_table_outputs:
        generate_output_tables_plots(combined_FP_df, per_electorate_winners, per_electorate_tcp_pair, per_electorate_winning_tcp, per_electorate_tcp_pair_vote, election_year, National_prior_dict, polling_avg_dict, prior_df_active_dict, prior_df_active, Farrer_full_prior, poll_df, Prior_estimates_dict_active, seat_alr_bases_dict, GLOBAL_CSVs, n_simulations, div_results_dict)

    
    model_outputs = {'FP_votes': combined_FP_df,'winners': per_electorate_winners,'tcp_pairs': per_electorate_tcp_pair,'winning_tcp': per_electorate_winning_tcp,'tcp_pair_vote': per_electorate_tcp_pair_vote}

    election_date = date(2026, 5, 9)
    date_string = (election_date - timedelta(days=Days_to_election)).strftime("%d.%m.%y")
    print(f"{date_string}:", Counter(per_electorate_winners['Farrer']))

    with open(f"farrer_model_data_{date_string}.pkl", 'wb') as f:
        dill.dump(model_outputs, f)

    return model_outputs


polling_avg = pd.DataFrame([[0.222,0.298,0.126,0.237,0.072,0.045]], columns=['COAL','ALP','GRN','ON','IND','OTH'], index=['Farrer']) 
#polling_avg = pd.DataFrame([[0.188,0.278,0.089,0.328,0.072,0.045]], columns=['COAL','ALP','GRN','ON','IND','OTH'], index=['Farrer'])
#polling_avg = pd.DataFrame([[0.221,0.295,0.122,0.24,0.072,0.05]], columns=['COAL','ALP','GRN','ON','IND','OTH'], index=['Farrer']) 
Days_to_election = 7
Seat_polls = pd.read_csv('SeatPollAverageFarrerByelectionFormatted.csv')

# Seat_polls.loc[:,['COAL','IND','GRN','ON','NAT','SPP','FFPA','OTH']] = 0.124,0.284,0.023,0.395,0.097,0.006,0.012,0.059 # actual result
div_results_dict = {
        'Liberal': 12.41,
        'Ind. Milthorpe': 28.08,
        "People's First": 0.70,
        'National': 9.78,
        'Legalise Cannabis': 2.33,
        'Greens': 2.32,
        'Ind. Woodward': 0.32,
        'One Nation': 39.53,
        'Family First': 1.22,
        'Sustainable Australia': 0.59,
        'Ind. Pappin': 0.78,
        'Shooters Fishers & Farmers': 1.96
    }

sims = simulate_electorate(n_simulations=100000, Days_to_election = Days_to_election, polling_avg =polling_avg, seat_polls_df=Seat_polls, election_year='Byelection', w = 0.52, B = 1.1, prior_alpha_fixed = 10, poll_alpha_fixed = 15, plot_table_outputs = True, div_results_dict = div_results_dict)
# sims = simulate_electorate(n_simulations=10000, Days_to_election = Days_to_election, polling_avg =polling_avg, seat_polls_df=Seat_polls, election_year='Byelection', w = 0.52, B = 0.01, prior_alpha_fixed = 10, poll_alpha_fixed = 15, plot_table_outputs = True)


def get_simulated_elections_series():

    adjusted_polling_averages_df = pd.read_csv("Farrer_byelection_Kalman_daily_avg_adjusted.csv")

    start_time = time.time()

    for day in adjusted_polling_averages_df['Day_index'].tolist():

        # get current day's polling average

        polling_avg = adjusted_polling_averages_df.loc[adjusted_polling_averages_df['Day_index']==day,['COAL','ALP','GRN','ON','IND','OTH']]
        polling_avg.index = ['Farrer']
        Days_to_election = day * (-1)

        # get seat polls (where they exist)
        if Days_to_election > 65:
            Seat_polls = pd.read_csv('SeatPollAverageFarrerByelectionFormatted.csv')
            Seat_polls = Seat_polls.loc[Seat_polls['Days since last election']==0,]
        elif Days_to_election > 30:
            Seat_polls = pd.read_csv("SeatPollsFarrerByelectionFormatted.csv").iloc[[0],]
            Seat_polls = Seat_polls.rename(columns={'Sample size':'n_polls'})
            Seat_polls['n_polls'] = 1
        else:
            Seat_polls = pd.read_csv('SeatPollAverageFarrerByelectionFormatted.csv')


        sims = simulate_electorate(n_simulations=100000, Days_to_election = Days_to_election, polling_avg =polling_avg, seat_polls_df=Seat_polls, election_year='Byelection', w = 0.52, B = 1.1, prior_alpha_fixed = 10, poll_alpha_fixed = 15, plot_table_outputs = False)
        

    print(time.time() - start_time)
    import pdb; pdb.set_trace()

def probabilities_stacked_bar_chart():

    import glob
    from datetime import datetime

    election_date = date(2026, 5, 9)
    start_date = date(2026, 1, 6)
    folder_path = "/home/dania-freidgeim/Australian Election/farrer_model_data_series" # Change if your dills are in a specific subfolder
    file_pattern = os.path.join(folder_path, "farrer_model_data_*.pkl")

    # --- 2. COLOUR & NAME LOGIC ---
    def slick_color(hex_code, factor=0.15):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = [int(c + (255 - c) * factor) for c in rgb]
        return f'rgb({new_rgb[0]}, {new_rgb[1]}, {new_rgb[2]})'

    colour_df = pd.read_csv("Party Colours.csv")
    colour_df.loc[:,'Party'] = colour_df['Party'].str.replace("’", "'", regex=False)
    base_colors = colour_df.set_index("Party")["Colour"].to_dict()
    base_colors['Ind. Milthorpe'] = base_colors.get('Independent', '#008080')

    party_name_dict = {
        'NP': 'National', 'COAL': 'Coalition', 'ON' : 'One Nation',
        'GRN' : 'Greens', 'HMP' : 'Legalise Cannabis', 'FFPA' : 'Family First',
        'ASP' : 'Shooters Fishers & Farmers', 'LP' : 'Liberal' , 
        'IND1': 'Ind. Milthorpe', 'OTH': 'Other'
    }

    # --- 3. DATA HARVESTING ---
    all_files = glob.glob(file_pattern)
    raw_data = {}
    actual_poll_dates = []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        try:
            date_str = fname.replace("farrer_model_data_", "").replace(".pkl", "")
            dt = datetime.strptime(date_str, "%d.%m.%y").date()
            with open(fpath, 'rb') as f:
                model_outputs = dill.load(f)
                winners = model_outputs['winners']['Farrer']
                total = len(winners)
                counts = Counter(winners)
                daily_probs = {party_name_dict.get(k, k): (v / total) * 100 for k, v in counts.items()}
                raw_data[dt] = daily_probs
                actual_poll_dates.append(dt)
        except Exception: continue

    # --- 4. DATA WRANGLING (Refined Logic) ---
    df = pd.DataFrame.from_dict(raw_data, orient='index').fillna(0)
    df.index = pd.to_datetime(df.index)
    full_range = pd.date_range(start=start_date, end=election_date)
    df = df.reindex(full_range).ffill() 



    # Grouping and Ordering
    # Stack order (Bottom to Top): Other -> Milthorpe -> Liberal -> National -> One Nation
    # Legend order: One Nation -> National -> Liberal -> Milthorpe -> Other -> Updates
    stack_order = ['Other', 'Ind. Milthorpe', 'Liberal', 'National', 'One Nation']
    legend_ranks = {'One Nation': 1, 'National': 2, 'Liberal': 3, 'Ind. Milthorpe': 4, 'Other': 5, 'National Poll Update': 6, 'Seat Poll Update': 7}

    existing_cols = df.columns.tolist()
    named_cols = [c for c in stack_order if c in existing_cols]
    other_cols = [c for c in existing_cols if c not in named_cols]

    if other_cols:
        df['Other'] = df[other_cols].sum(axis=1)
        df = df.drop(columns=other_cols)

    final_stack = [c for c in stack_order if c in df.columns]

    # --- 5. FIGURE CONSTRUCTION ---
    fig = go.Figure()

    # 1. ACTUAL BARS (Hidden from Legend)
    final_stack = ['Other', 'Ind. Milthorpe', 'Liberal', 'National', 'One Nation']

    for party in final_stack:
        is_other = (party == 'Other')
        color = "#333333" if is_other else slick_color(base_colors.get(party, "#CCCCCC"))
        hover_fmt = ":.3f" if is_other else ":.1f"
        
        fig.add_trace(go.Bar(
            name=party,
            x=df.index,
            y=df[party],
            marker=dict(color=color, line=dict(width=1.0, color='rgba(255, 255, 255, 0.3)')), 
            width=1000*3600*24,
            showlegend=False, 
            hovertemplate=f"<b>{party}</b>: %{{y{hover_fmt}}}%<extra></extra>"
        ))

    # --- 6. ACTUAL TRIANGLES (Hidden from Legend) ---
    seat_poll_days = [election_date - timedelta(days=65), election_date - timedelta(days=30)]
    other_poll_dates = [d for d in actual_poll_dates if d not in seat_poll_days]

    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(d) - timedelta(hours=12) for d in other_poll_dates],
        y=[0.5] * len(other_poll_dates),
        mode='markers',
        marker=dict(symbol='triangle-up', size=11, color='white', line=dict(width=1, color='black')),
        name='National Poll Update',
        showlegend=False, 
        hoverinfo='skip' 
    ))

    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(d) - timedelta(hours=12) for d in seat_poll_days],
        y=[0.5] * len(seat_poll_days),
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='black'),
        name='Seat Poll Update',
        showlegend=False,
        hoverinfo='skip'
    ))

    # --- 7. DUMMY LEGEND ARCHITECTURE ---
    # We add 4 non-breaking spaces (&nbsp;) to every name. 
    # This physically prevents Chrome from squishing the text.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='triangle-up', size=12, color='black'), name='Seat Poll Update&nbsp;&nbsp;&nbsp;&nbsp;', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='triangle-up', size=11, color='white', line=dict(width=1, color='black')), name='National Poll Update&nbsp;&nbsp;&nbsp;&nbsp;', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=slick_color(base_colors.get('Ind. Milthorpe', "#CCCCCC")), line=dict(width=1.0, color='rgba(255, 255, 255, 0.3)')), name='Ind. Milthorpe&nbsp;&nbsp;&nbsp;&nbsp;', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=slick_color(base_colors.get('Liberal', "#CCCCCC")), line=dict(width=1.0, color='rgba(255, 255, 255, 0.3)')), name='Liberal&nbsp;&nbsp;&nbsp;&nbsp;', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=slick_color(base_colors.get('National', "#CCCCCC")), line=dict(width=1.0, color='rgba(255, 255, 255, 0.3)')), name='National&nbsp;&nbsp;&nbsp;&nbsp;', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=slick_color(base_colors.get('One Nation', "#CCCCCC")), line=dict(width=1.0, color='rgba(255, 255, 255, 0.3)')), name='One Nation&nbsp;&nbsp;&nbsp;&nbsp;', showlegend=True, hoverinfo='skip'))

    # --- 8. ANNOTATIONS (Brute Force Anchor Fix) ---
    purple = "#933eea"
    milestones = [
        (85, "Ley announces retirement"),     
        (65, "By-election called"),    
        (25, "Candidates declared")      
    ]

    for days, label in milestones:
        target_dt = election_date - timedelta(days=days)
        line_x = pd.to_datetime(target_dt) - timedelta(hours=12)
        
        fig.add_shape(type="line", x0=line_x, y0=0, x1=line_x, y1=100,
                    line=dict(color=purple, width=1.5, dash="dot"), xref="x", yref="y")
        
        fig.add_annotation(
            x=line_x, 
            y=1.0,          
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(color=purple, size=11, family="Arial, sans-serif"), 
            textangle=15,    
            # FIX: Chrome fails 'right', so we 'center' it and then 
            # physically shift it left by 75 pixels to fake a right-alignment.
            xanchor="center", 
            yanchor="bottom",
            xshift=-50,      
            yshift=15        
        )

    # --- 9. FINAL LAYOUT (Legend & Zoom Fix) ---
    fig.update_layout(
        dragmode=False, # KILL the zoom-drag box completely
        barmode='stack',
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        
        yaxis=dict(
            range=[0, 100], 
            ticksuffix="%", 
            showgrid=True,
            automargin=True,
            fixedrange=True
        ),
        xaxis=dict(
            range=[start_date, election_date + timedelta(days=1)],
            showspikes=False,
            automargin=True,
            fixedrange=True
        ),
        bargap=0,
        legend=dict(
            orientation="h", 
            yanchor="top", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=10, family="Arial, sans-serif"),
            traceorder="reversed",
            # FIX: Force massive pixel gaps between items
            entrywidth=150, 
            entrywidthmode="pixels",
            valign="middle"
        ),
        hovermode="x unified",
        margin=dict(t=100, b=50, l=10, r=10), 
    )

    # --- 10. THE RESPONSIVE JAVASCRIPT (Final Cleanup) ---
    js_script = f"""
    var plotElement = document.getElementsByClassName('plotly-graph-div')[0];
    if (plotElement) {{
        window.addEventListener('resize', function() {{
            var width = window.innerWidth;
            var triangleSize = width < 600 ? 5 : 11;
            var outlineWidth = width < 600 ? 0 : 1;
            
            // On mobile, we push the legend way down to prevent overlap
            var newMarginB = width < 600 ? 120 : 50; 
            var newLegendY = width < 600 ? -0.35 : -0.15;
            
            Plotly.relayout(plotElement, {{
                'margin.b': newMarginB,
                'legend.y': newLegendY
            }});
            
            var barIndices = [0, 1, 2, 3, 4, 9, 10, 11, 12];
            var scatterIndices = [5, 6, 7, 8];
            
            Plotly.restyle(plotElement, {{'marker.line.width': outlineWidth}}, barIndices);
            Plotly.restyle(plotElement, {{'marker.size': triangleSize}}, scatterIndices);
        }});
        window.dispatchEvent(new Event('resize'));
    }}
    """

    config = {'displayModeBar': False, 'responsive': True, 'doubleClick': False}
    fig.write_html("farrer_trend_final.html", config=config, post_script=js_script)
    fig.show(config=config)

#probabilities_stacked_bar_chart()


def analyse_results(FP_sim):
    FP_results = {'LP': 0.1241, 'IND1':0.2808,'GRPF':0.007,'NP':0.0978,'HMP':0.0233,'GRN':0.0232,'IND2':0.0032,'ON':0.3959,'FFPA':0.0122,'SPP':0.0059,''
    'IND3':0.0078,'ASP':0.0196}

    prob_list = []
    
    for p in FP_sim.columns:
        prob_extreme = min([(FP_sim[p] > FP_results[p]).mean(), (FP_sim[p] < FP_results[p]).mean()])*2
        print(p,prob_extreme)
        prob_df = pd.DataFrame([prob_extreme], index = [p])
        prob_list.append(prob_df)

    # model gave an event as extreme as the result a {prob_extreme} probability
    prob_extreme_df = pd.concat(prob_list)

    print(prob_extreme_df)


    import pdb; pdb.set_trace()

analyse_results(sims['FP_votes'])
import pdb; pdb.set_trace()















with open("covms_all.pkl", "rb") as f:
        data = pickle.load(f)


CovM = data["Seat_poll_covm_dict"]['Byelection']
polled = pd.DataFrame({'ALP':[0.25],'COAL':[0.25],'GRN':[0.1],'ON':[0.2],'IND':[0.15],'OTH':[0.05]})
#polled = pd.DataFrame({'ALP':[0.30],'COAL':[0.30],'GRN':[0.1],'OTH':[0.2]})
                       
#CovM = pd.read_csv('ElectionErrorALRCovarianceNational2025.csv', index_col = 0)
#polled = pd.DataFrame({'ALP':[0.346],'COAL':[0.318],'GRN':[0.122],'ON':[0.064],'UAPP':[0.019],'OTH':[0.131]})

polled_ALR = df_to_alr(polled, ref_col)

n_sims = 1000
alr_sims = rng.multivariate_normal(mean=polled_ALR.loc[0], cov=CovM, size=n_sims)
#alr_sims = multivariate_t_rvs(mean=polled_ALR.loc[0], cov=CovM, df = 10,n_sims=n_sims)
alr_sims_df = pd.DataFrame(alr_sims, columns=polled_ALR.columns)
prop_sims_df = alr_to_simplex_vectorized(alr_sims_df, ref_col)

np.std(prop_sims_df)


# 2. Apply the asymmetric adjustment to the CovM
# We multiply the whole matrix by the shrinkage, then add the constant to every cell


avg_var = np.diag(CovM).mean()
  
coal_variance_boost = 0.004  # Expands the COAL reference anchor


minor_shrinkage = 1-coal_variance_boost/avg_var  # Tightens GRN, ON, IND, OTH
adjusted_CovM = (CovM.values * minor_shrinkage) + coal_variance_boost

alr_sims = rng.multivariate_normal(mean=polled_ALR.loc[0], cov=adjusted_CovM, size=n_sims)
alr_sims_df = pd.DataFrame(alr_sims, columns=polled_ALR.columns)
prop_sims_df = alr_to_simplex_vectorized(alr_sims_df, ref_col)

np.std(prop_sims_df)

from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.boxplot(data=prop_sims_df)
plt.ylabel("Proportion")
plt.title("Simulated Proportion Variability from ALR Covariance")
plt.show()



import pdb; pdb.set_trace()

