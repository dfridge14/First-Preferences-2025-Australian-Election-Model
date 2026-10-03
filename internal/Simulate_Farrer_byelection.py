import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import time
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
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


NUM_MAIN_PARTIES = 5
LP_NP_DIR_ALPHA = 4.178317954770676
MAJOR_SITOUT_DIR_ALPHA = 14 # (4 + 7 + 31)/3


SEAT_POLL_SCALE_2 = 0.8977871632158285






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

election_year = 'Byelection'

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

def full_weight_vector(all_divisions, High_Others_df, others_column="OTH", beta=0.5, k=10, start=0.2):
    import numpy as np

    # align OTH values to all_divisions
    oth = High_Others_df.reindex(all_divisions)[others_column].astype(float).fillna(0.0).values

    # logistic-style downweighting (same spirit as original)
    # higher OTH → lower weight
    x = (oth - start) * k
    w = 1 / (1 + np.exp(x))

    # apply beta scaling
    w = 1 - beta * (1 - w)

    return w


def simulate_Polling_Fundamentals_model(
    n_simulations,
    election_year,
    df_t=0,
    Prior_estimates_df=pd.DataFrame(),
    polling_avg = pd.DataFrame(),
    National_prior = pd.DataFrame(),
    National_Polling_error_cov =  pd.DataFrame(),
    Electorate_Residuals_cov =  pd.DataFrame()
):
    
    use_volatility_weights = (election_year != 'Byelection')

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

    National_Simulated_polling_error = np.random.multivariate_normal(
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

        MEAN_SHIFT = 1
        mean = np.zeros(d)
        if (election_year == "Byelection") and MEAN_SHIFT:
            grn_idx = 1
            mean[grn_idx] = -0.28
            #import pdb; pdb.set_trace()
            
        group_sims = np.random.multivariate_normal(mean=mean,cov=cov,size=n_simulations * n_group)

        if dist == "t":
            g = np.random.gamma(df_t / 2., 2. / df_t, size=n_simulations * n_group)
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


import numpy as np

def redistribute_major_sitout(
    combined,
    all_party_names,
    div,
    mapping_df
    ):
    
    proportions_df = pd.read_csv('Missing_major_reallocation_proportions.csv')
    Major_sitout, has_strong_IND = mapping_df.loc[mapping_df['div_nm']==div,['Major_sitout','has_strong_IND']].drop_duplicates().to_numpy().ravel()
    
    combined = combined.copy()
    n_sims = combined.shape[0]
    
    # --- 1. index of sitout party
    sitout_idx = all_party_names.index(Major_sitout)
    
    # --- 2. extract and remove sitout vote
    V = combined[:, sitout_idx].copy()
    combined[:, sitout_idx] = 0.0
    
    # --- 3. get group means (μ)
    row = proportions_df[(proportions_df['Major_sitout'] ==  ('LP' if Major_sitout == 'COAL' else Major_sitout) ) & (proportions_df['has_strong_IND'] == has_strong_IND)].iloc[0]
    

    if has_strong_IND == 1:
        mu = np.array([row['theta_aligned'], row['theta_ind'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'IND', 'other_major', 'other']
    else:
        mu = np.array([row['theta_aligned'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'other_major', 'other']
    
    # --- 4. Dirichlet parameters
    alpha = mu * MAJOR_SITOUT_DIR_ALPHA
    
    # --- 5. sample group shares
    p_group = np.random.dirichlet(alpha, size=n_sims)
    
    # --- 6. mapping for this division + regime
    sub = mapping_df[(mapping_df['div_nm'] == div) &(mapping_df['Major_sitout'] == Major_sitout) &(mapping_df['has_strong_IND'] == has_strong_IND)]
        
    # --- 7. allocate to each group
    for g_idx, g in enumerate(groups):
        
        parties = sub.loc[sub['group'] == g, 'PartyAb'].tolist()
        
        if not parties:
            continue
        
        # map to indices in combined
        idx = [all_party_names.index(p) for p in parties if p in all_party_names]
        
        if not idx:
            continue
        
        current = combined[:, idx]
        row_sums = current.sum(axis=1, keepdims=True)
        
        # proportional weights (fallback = equal split)
        weights = np.divide(
            current,
            row_sums,
            out=np.full_like(current, 1/len(idx)),
            where=row_sums > 0
        )
        
        allocation = (V * p_group[:, g_idx])[:, None] * weights
        
        combined[:, idx] += allocation

    combined = np.delete(combined, sitout_idx, axis=1)
    all_party_names = [p for i,p in enumerate(all_party_names) if i != sitout_idx]
    
    return combined, all_party_names

def expand_all_divisions_from_prior_df(sim, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=100):

    final_sim = {}
    party_name_dict = {} 

    multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
    C200_IND_splits = pd.read_csv(f"Independent_splits_multiple.csv", index_col=None)
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
        sim_block = sim[:, i, :]            # shape (10000, 5)
        main_parties = sim_block[:, :NUM_MAIN_PARTIES]     # shape (10000, 4)
        other_share = sim_block[:, NUM_MAIN_PARTIES]       # shape (10000,)

        #print(div)

        # Extract prior for this division

        prior_row = Prior_estimates_dict[div]
        prior_row_Other = prior_row[[p for p in prior_row.columns if p not in Major_parties]]
        minor_names = list(prior_row_Other.columns)
        rel_weights = prior_row_Other.iloc[0].values
        rel_weights = rel_weights / rel_weights.sum()

        


        # Dirichlet sampling
        alpha = rel_weights * alpha_scalar
        splits = np.random.dirichlet(alpha, size=sim.shape[0])  # shape (10000, n_minor)

        # Expand 'Other' proportionally
        other_expanded = splits * other_share[:, None]  # (10000, n_minor)

        # Combine with main parties
        combined = np.concatenate([main_parties, other_expanded], axis=1)

        all_party_names = Polling_parties + minor_names


        # if no ON/TOP, remove this column!
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


            if 'NP' in prior_row.columns and 'LP' in prior_row.columns:
                NP_est = (prior_row['NP'] /  prior_row[['LP','NP']].sum(axis=1)).iloc[0]
            else:
                NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate'].iloc[0]

            # current hack for 2025 Nationals:
            if (election_year == '2025') & (div in ['Bullwinkel','Forrest',"O'Connor"]):
                #import pdb;pdb.set_trace()
                NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate'].iloc[0]



            alpha = np.array([1-NP_est,NP_est]) * LP_NP_DIR_ALPHA
            splits = np.random.dirichlet(alpha, size=sim.shape[0])
            LP_NP_votes = splits * COAL_votes[:, None]

            combined = np.concatenate([combined, LP_NP_votes], axis=1)[:,1:] # Removes 'COAL' from 1st column

            all_party_names = all_party_names[1:] + ['LP','NP'] # correct order

        # perform independent split as well!
        if 'IND' in Prior_estimates_dict[div]:
            #import pdb;pdb.set_trace()

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
                    C200_ratio = 1 / (1 + np.exp(-logit_mu))

                    #C200_ratio = C200_IND_splits.loc[C200_IND_splits['Election'] == f'AverageFor{election_year}','Ratio'].iloc[0]




                    C200_IND_position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm'] == div,'Number'].iloc[0]

                    if (election_year == '2025') and div in Complex_contests.keys():

                        C200_ratio = np.full_like(C200_ratio, Complex_contests[div], dtype=float) # an array col
                    
                    K = num_INDs_curr.iloc[0]
                    non_ind_total = 1 - C200_ratio          # (n_sims,)
                    means_array_sim = np.tile((non_ind_total / (K - 1))[:, None],(1, K))
                    means_array_sim[:, C200_IND_position - 1] = C200_ratio

                #splits = np.random.dirichlet(means_array* alpha_scalar, size=sim.shape[0])
                splits = np.array([np.random.dirichlet(alpha_scalar * row) for row in means_array_sim])

                All_IND_votes = splits * IND_votes[:, None]

                combined = np.concatenate([combined, All_IND_votes], axis=1)
                combined = np.delete(combined,IND_index, axis = 1) # remove 'IND' col

                all_party_names = all_party_names + ['IND'+str(i) for i in range(1,num_INDs_curr.iloc[0]+1)]
                all_party_names = [p for p in all_party_names if p != 'IND'] # remove IND

            else:
                all_party_names = [p if p != 'IND' else 'IND1' for p in all_party_names] # renames to IND1 if only single IND


        # If missing_major, perform final reallocation using estimated model
        if election_year == 'Byelection':
            mapping_df = pd.read_csv('Missing_major_allocation.csv')

            if div in mapping_df['div_nm'].unique():
                combined, all_party_names = redistribute_major_sitout(combined,all_party_names,div, mapping_df)
                            
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




def simulate_nat_polling_full_model(n_simulations, election_year, prior_df_active_dict, polling_avg_dict, National_prior_dict, Prior_estimates_dict_active_per_election, Results_dict_active_per_election, byelection_group_structure, Nat_poll_covm_dict, Electorate_residual_covm_dict):

    polling_avg = polling_avg_dict[election_year]
    prior_df_active = prior_df_active_dict[election_year]
    National_prior = National_prior_dict[election_year]

    sim = simulate_Polling_Fundamentals_model(
        n_simulations=n_simulations,
        election_year=election_year,
        df_t=0,
        Prior_estimates_df=prior_df_active,
        polling_avg = polling_avg,
        National_prior = National_prior,
        National_Polling_error_cov =  Nat_poll_covm_dict[election_year],
        Electorate_Residuals_cov =  Electorate_residual_covm_dict[election_year]
    )

    Prior_estimates_dict_active = Prior_estimates_dict_active_per_election[election_year]
    Results_dict_active = Results_dict_active_per_election[election_year]


    final_sim, party_name_dict = expand_all_divisions_from_prior_df(sim, Prior_estimates_dict_active, Results_dict_active, election_year, alpha_scalar=22)


    return final_sim, party_name_dict





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
    transformed_matrix = error_matrix @ A.T
    
    return transformed_matrix, new_cols, ref_col

def calculate_rigorous_split(poll_minors, cago_other):
    minor_cols = [c for c in poll_minors.columns if c != 'OTH']
    naive_means = poll_minors[minor_cols].replace(0, np.nan).mean().fillna(0)
    
    w_oth = poll_minors['OTH'].replace(0, np.nan).min() or 0.01 
    missing_mask = (poll_minors[minor_cols] == 0)
    row_weights = missing_mask.dot(naive_means)
    
    alloc_factor = np.where(
        (poll_minors['OTH'] > 0) & (row_weights > 0),
        poll_minors['OTH'] / (row_weights + w_oth),
        0
    )
    
    imputed = poll_minors.copy()
    for col in minor_cols:
        imputed[col] += missing_mask[col] * alloc_factor * naive_means[col]
    
    final_split = imputed[minor_cols].mean()
    final_split['OTH'] = cago_other - final_split.sum()
    
    return final_split


def precompute_division(poll_df,
    prior_df,
    div,
    election_year,
    missing_party = None,
    ref_col='COAL',
    GLOBAL_CSVs = {}
):


    # compute CAGO_row
    minor_cols = [p for p in prior_df.columns if p not in ['ALP','COAL','LP','NP','LNP','CLP','GRN'] + ['IND','ON']]
    # if GRN vote share not reported, adjust GRN/OTH split (Longman, Bennelong, 2019 Flinders, Cowan, 2022 Mackellar, Norht Sydney, 2025 Bullwinkel)
    if (poll_df['GRN']==0).any():
        poll_df_reset = poll_df.reset_index()
        mask = poll_df_reset['GRN'] == 0  # only apply to GRN = 0 polls
        grn_nonzero = poll_df_reset.loc[~mask, 'GRN'] 

        if len(grn_nonzero) > 0: # try use other poll average
            grn_val = grn_nonzero.mean()
        else:
            denom = prior_df.loc[div, minor_cols].sum() + prior_df.loc[div, 'GRN'] # otherwise use prior proportion of OTH
            grn_val = poll_df.loc[div, 'OTH'] * (prior_df.loc[div, 'GRN'] / denom)

        poll_df_reset.loc[mask, 'GRN'] = grn_val
        poll_df_reset.loc[mask, 'OTH'] -= grn_val

        poll_df = poll_df_reset.set_index('Electorate')
    

    # reduce to CAGO categories, combine for poll average
    CAGO_rows = []
    for i in range(len(poll_df)):
        curr_row = poll_df.iloc[i:i+1,].iloc[:,2+('byelection_year' in poll_df.columns):]
        curr_row_normalised = curr_row/curr_row.sum(axis=1)[div]
        curr_row_normalised['COAL'] = curr_row_normalised['COAL'] + curr_row_normalised['NAT']
        curr_row_normalised = curr_row_normalised[['COAL','ALP','GRN','ON','IND']]
        curr_row_normalised.loc[:,'Other'] = (1 - curr_row_normalised.sum(axis=1)).iloc[0]

        CAGO_rows.append(curr_row_normalised) # extra col for byelecitons
    CAGO_row = pd.concat(CAGO_rows).mean()


    # add alr error and return to prop space, accounting for missing major party
    if missing_party is not None:
        CAGO_row = CAGO_row.drop([missing_party])


     # reallocate minor parties - first according to poll proportions, then for remaining according to prior proportions
    # 1. Polled party 'OTH' reallocation
    poll_minors = poll_df.reindex(columns = minor_cols+ ['OTH']).astype(float).fillna(0)
    poll_minor_means = poll_minors.replace(0, np.nan).mean() # mean shares of any minor parties polled
    poll_minor_split = poll_minor_means.dropna()
    poll_minor_split['OTH'] = CAGO_row['Other'] - poll_minor_split.loc[poll_minor_split.index !='OTH'].sum() # remaining OTH vote after minor polled party averages accounted for (safe when polled minors differ from poll to poll)
    


    if set(minor_cols).issubset(poll_minor_split.index):
        if not np.isclose(poll_minor_split['OTH'],0):
            import pdb; pdb.set_trace() # just check in case fails 
        poll_minor_split = poll_minor_split.loc[poll_minor_split.clip(lower = 0) > 0] # remove 'OTH' if OTH == 0
        

    if 'OTH' in poll_minor_split and poll_minor_split['OTH'] <= 0:
        poll_minor_split = calculate_rigorous_split(poll_minors, CAGO_row['Other'])

    poll_minor_split = poll_minor_split.loc[poll_minor_split.clip(lower = 0) > 0] # remove 'OTH' if OTH == 0

    poll_weights = poll_minor_split / poll_minor_split.sum()

    
    import pdb; pdb.set_trace()

    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]
    C200_IND_splits = GLOBAL_CSVs["C200_IND_splits"]
    C200_IND_positions_df = GLOBAL_CSVs["C200_IND_positions_df"]
    NP_ratios_curr = GLOBAL_CSVs["NP_ratios_curr"]
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]

    multi_ind_c200_case = ((div in multiple_INDs_df['div_nm'].values) and (div in C200_IND_positions_curr['div_nm'].values))
    Position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm']==div,'Number'].iloc[0] if multi_ind_c200_case else 1

    k = int(multiple_INDs_df.loc[multiple_INDs_df['div_nm'] == div,'No_of_INDs'].iloc[0]) if div in multiple_INDs_df['div_nm'].unique() else 1

    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]

    NP_ratios_curr = NP_ratios_curr.loc[NP_ratios_curr['election_year'] == election_year]

    
    return {
        "alr_base": np.log(CAGO_row.drop(ref_col) / CAGO_row[ref_col]),
        "poll_weights": poll_weights,
        "minor_cols": minor_cols,
        "multi_ind_c200_case": multi_ind_c200_case,
        "Position": Position if multi_ind_c200_case else None,
        "k": k,
        'C200_ratio_lm_params': C200_ratio_lm_params,
        "C200_IND_positions_curr":C200_IND_positions_curr,
        "NP_ratios_curr": NP_ratios_curr
    }

def process_division(
    poll_df,
    prior_row,
    Results_row,
    div,
    election_year,
    alr_error_row,
    missing_party = None,
    ref_col='COAL',
    prior_alpha=22,
    poll_alpha = 22,
    precomputed_dict = {},
    GLOBAL_CSVs = {}
):
    
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]
    

    alr_base = precomputed_dict['alr_base']
    poll_weights = precomputed_dict['poll_weights']
    minor_cols = precomputed_dict['minor_cols']
    multi_ind_c200_case = precomputed_dict['multi_ind_c200_case'] # currently C200 INDs comprise all such cases
    Position = precomputed_dict['Position']
    k = precomputed_dict['k']
    beta0,beta1 = precomputed_dict['C200_ratio_lm_params']
    C200_IND_positions_curr = precomputed_dict['C200_IND_positions_curr']
    NP_ratios_curr = precomputed_dict['NP_ratios_curr']

    prior_df = prior_row.copy()

    alr_adj = alr_base + alr_error_row
    adjusted = alr_to_simplex_vectorized(alr_adj, ref_col).rename(columns={'IND':'IND1'}) # Farrer-specific

    macro_oth = adjusted['Other'].iloc[0]


    poll_alloc = pd.Series(0.0, index=minor_cols + ['OTH'])
    alpha_curr = (poll_weights.values*poll_alpha).astype(float)  # or scaled version if you want concentration
    draw = np.random.dirichlet(alpha_curr)
    poll_alloc.loc[poll_weights.index] = macro_oth * draw

    # 2. Prior reallocation for remaining minor parties - only if poll is incomplete
    remaining_oth = poll_alloc['OTH']
    unpolled_minor_cols = poll_alloc[poll_alloc == 0].index.tolist()


    if multi_ind_c200_case:
        
        logit_mu = beta0 + beta1 * np.log(prior_df.loc[div, 'IND'])
        C200_ratio = 1 / (1 + np.exp(-logit_mu))

        # scale down prior IND mass to residual space only, artificially
        prior_df.loc[div, 'IND'] = prior_df.loc[div, 'IND'] * (1 - C200_ratio)

        if ('IND' in poll_alloc):
            #  rename allocated IND to correct INDX
            poll_alloc.loc[f'IND{Position}'] = poll_alloc.loc['IND']

        # reset for remaiing INDs
        poll_alloc.loc['IND'] = 0

        unpolled_minor_cols += ['IND']

    if election_year == 'Byelection' and div == 'Wentworth': # special by-election case of multi_ind_c200_case. ADD: Farrer.
        # Heath gets 6.435 (polling average), Phelps: 20.974
        WENTWORTH_IND_RATIO = 20.974/(6.435 + 20.974)

        poll_alloc.loc['IND1'] = poll_alloc.loc['IND'] * WENTWORTH_IND_RATIO
        poll_alloc.loc['IND2'] = poll_alloc.loc['IND'] * (1 - WENTWORTH_IND_RATIO)

        poll_alloc.loc['IND'] = 0
        unpolled_minor_cols += ['IND']



    if len(unpolled_minor_cols) > 0 and remaining_oth > 0:
        prior_minors = prior_df.loc[div, unpolled_minor_cols]
        prior_weights = prior_minors / prior_minors.sum()
        alpha_prior = prior_weights.values * prior_alpha
        draw_prior = np.random.dirichlet(alpha_prior)

        poll_alloc.loc[prior_weights.index] = remaining_oth * draw_prior
        poll_alloc = poll_alloc.drop('OTH')
    elif remaining_oth == 0:
        poll_alloc = poll_alloc.drop('OTH')

    assert np.isclose(poll_alloc.sum(), macro_oth) # oth vote share correctly distirbuted

    final_row = pd.concat([adjusted.drop(columns=['Other']),poll_alloc.to_frame().T], axis=1)



    #### LP-NP split
    # --- Determine NP share (Dirichlet centre) ---

    COAL_double_div = False

    if ('LP' in Results_row) and ('NP' in Results_row):
        COAL_double_div = True

        NP_LP_split_polled = ('NAT' in poll_df.columns) and poll_df['NAT'].sum() > 0

        if NP_LP_split_polled:
            # Use poll-implied ratio
            nat = poll_df['NAT'].astype(float)
            coal = poll_df['COAL'].astype(float)

            nat_mean = nat.replace(0, np.nan).mean()
            adj_np = nat.where(nat > 0, nat_mean) # Only fill NAT where missing
            adj_lp = coal.where(nat > 0, coal - nat_mean).clip(lower=0) # Only subtract from COAL where NAT was missing

            total = adj_np.mean() + adj_lp.mean()
            if total > 0:
                NP_est = adj_np.mean() / total
            else:
                NP_LP_split = False # fallback if degenerate
            
        if not NP_LP_split_polled:
            # use prior-based split
            
            if div in NP_ratios_curr['div_nm'].unique():
                if 'NP' in prior_df.columns and 'LP' in prior_df.columns:
                    NP_est = (prior_df['NP'] / prior_df[['LP','NP']].sum(axis=1)).iloc[0]
                else:
                    NP_est = NP_ratios_curr.loc[
                        NP_ratios_curr['div_nm'] == div, 'final_estimate'
                    ].iloc[0]

                # 2025 override
                if (election_year == '2025') & (div in ['Bullwinkel','Forrest',"O'Connor"]):
                    NP_est = NP_ratios_curr.loc[
                        NP_ratios_curr['div_nm'] == div, 'final_estimate'
                    ].iloc[0]


        # dirichlet split of COAL vote into NP and LP
        alpha = np.array([1 - NP_est, NP_est]) * LP_NP_DIR_ALPHA
        splits = np.random.dirichlet(alpha)
        LP_NP_votes = splits * final_row['COAL'].iloc[0]  # (n_sim=1, 2)
        LP_mean = LP_NP_votes[0].mean()
        NP_mean = LP_NP_votes[1].mean()

        # remove COAL, replace with LP/NP
        final_row = final_row.drop(columns=['COAL'])
        final_row[['LP', 'NP']] = [LP_mean, NP_mean]


    

    if 'IND' in final_row.columns:

        # split IND (or any that remains)
        ind_total = final_row['IND'].iloc[0]

        if multi_ind_c200_case:
             
            free_weights = np.full(k - 1, 1 / (k - 1))
            IND_draw = np.random.dirichlet(free_weights * prior_alpha)

            ind_means = IND_draw * ind_total

            final_row = final_row.drop(columns=['IND'])

            idx = 0
            for i in range(1, k + 1):
                if i == Position:
                    continue
                final_row[f'IND{i if i < Position else i}'] = ind_means[idx]
                idx += 1
                
        elif election_year == 'Byelection' and div == 'Wentworth':
            final_row = final_row.rename(columns={'IND':'IND3'}) # manually set

        else:

            if div in multiple_INDs_df['div_nm'].unique():

                means_array = np.full(k, 1 / k)

                # --- C200 structured split ---
                if div in C200_IND_positions_curr['div_nm'].unique():

                    pos = int(C200_IND_positions_curr.loc[
                        C200_IND_positions_curr['div_nm'] == div,
                        'Number'
                    ].iloc[0]) - 1

                            
                    logit_mu = beta0 + beta1 * np.log(ind_total)
                    C200_ratio = 1 / (1 + np.exp(-logit_mu))

                    import pdb; pdb.set_trace()

                    rest = (1 - C200_ratio) / (k - 1)
                    means_array[:] = rest
                    means_array[pos] = C200_ratio

                # --- Dirichlet split ---
                splits = np.random.dirichlet(means_array * prior_alpha)

                ind_means = splits * ind_total

                # replace IND with IND1..INDk
                final_row = final_row.drop(columns=['IND'])

                for i in range(k):
                    final_row[f'IND{i+1}'] = ind_means[i]

            else:
                # single IND case → just rename
                final_row = final_row.rename(columns={'IND': 'IND1'})

        
    if (not COAL_double_div) and not (missing_party == 'COAL'):
        COAL_replacement_list = ['LP', 'NP', 'LNP', 'CLP']

        coal_party = next(p for p in Results_row.columns if p in COAL_replacement_list)
        if 'COAL' in final_row.columns:
            final_row = final_row.rename(columns={'COAL': coal_party})

    assert set(final_row.columns) == set(Results_row.columns)
    
    return final_row


def simulate_seat_polling_full_model(n_simulations, election_year, Seat_poll_year_dict, Seat_poll_covm_dict, Results_dict_active , prior_alpha = 22, poll_alpha = 22, GLOBAL_CSVs = {}):

    # currently unvectorised, simulates for single seat at a time

    seat_poll_sims = {}

    prior_long = pd.read_csv(f"By_election_Fundamentals_after_2025.csv")

    # perform for each electorate with seat polling
    curr_seat_polls = Seat_poll_year_dict[election_year]
    div = 'Farrer'

    prior_row = prior_long.loc[prior_long['div_nm']==div,][['PartyAb','FP_Votes']].set_index('PartyAb').T.rename(index = {'FP_Votes':div}) * 100 #[['COAL','ALP','GRN','ON','IND']]
    #prior_row.loc[:,'OTH'] = 1-prior_row.sum(axis=1)
    poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate']==div].set_index('Electorate')

    
    

    results = []

    d = Seat_poll_covm_dict[election_year].shape[0]

    alr_error_matrix = np.random.multivariate_normal(
        mean=np.zeros(d),
        cov=Seat_poll_covm_dict[election_year],
        size=n_simulations
    )

    missing_party = 'ALP'
    alr_error_matrix, new_cols, ref_col = transform_alr_matrix(alr_error_matrix, missing_party)


    Results_row = Results_dict_active[election_year][div]

    precomputed_dict = precompute_division(poll_df,
        prior_row,
        div,
        election_year,
        missing_party,
        ref_col,
        GLOBAL_CSVs
    )

    for s in range(n_simulations):
        alr_error_row = pd.DataFrame([alr_error_matrix[s]], columns=new_cols)

        out = process_division(
            poll_df,
            prior_row,
            Results_row,
            div,
            election_year,
            alr_error_row=alr_error_row,
            missing_party=missing_party,
            ref_col=ref_col,
            prior_alpha=prior_alpha,
            poll_alpha = poll_alpha,
            precomputed_dict=precomputed_dict,
            GLOBAL_CSVs=GLOBAL_CSVs
        )

        results.append(out)

    seat_poll_sims[div] = pd.concat(results)

    # return numpy array in order of Ballot appearance
    Ballot_order = Results_dict_active[election_year][div].columns.tolist()
    seat_poll_sims[div] = seat_poll_sims[div][Ballot_order].to_numpy()

    return seat_poll_sims





def make_single_electorate_violin_plot(combined_df):

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

    # --- 1. Apply Custom Party Name Logic to Columns ---
    # Convert to a copy so we don't accidentally modify the parent DataFrame in memory
    df_plot = combined_df.copy()
    
    # Step 1: Append div name to INDs (e.g., 'IND1' -> 'IND1Farrer')
    df_plot.columns = [str(c) + div if str(c).startswith('IND') else str(c) for c in df_plot.columns]
    
    # Step 2: Replace known specific local independents
    df_plot.columns = df_plot.columns.to_series().replace(party_name_dict).values
    
    # Step 3: Strip div suffix from unmapped standard INDs (e.g., 'IND2Farrer' back to 'IND2')
    df_plot.columns = df_plot.columns.str.replace(r'^(IND\d).*', r'\1', regex=True)
    
    # Step 4: Replace standard INDx with generic names (e.g., 'IND2' -> 'Independent 2')
    df_plot.columns = df_plot.columns.to_series().replace(party_name_dict).values

    # Convert proportions to percentages
    df_pct = df_plot * 100
   

    # 2. Determine party order by descending mean vote share
    party_order = df_pct.mean().sort_values(ascending=False).index.tolist()

    # 3. Calculate summary stats instantly across all columns
    summary_stats = {
        party: {
            'mean': df_pct[party].mean(),
            'lower_95': np.percentile(df_pct[party], 2.5),
            'upper_95': np.percentile(df_pct[party], 97.5)
        }
        for party in party_order
    }


    # 4. Build Figure
    fig = go.Figure()

    for party in party_order:
        party_data = df_pct[party].to_numpy()
        stats = summary_stats[party]
        
        # Extract mapped names (fallback to original column name if missing from dicts)
        display_name = party_name_dict.get(party, party)
        x_label = site_abbreviation_dict.get(party, party)
        party_color = colors.get(party, 'gray')

        dynamic_bw = 0.7 if stats['mean'] > 5.0 else 0.2 
        dynamic_bw = 1.2 if stats['mean'] > 10 else dynamic_bw

        # Add violin trace
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
            width=0.5,
            bandwidth=dynamic_bw,
            points=False   # hide default points
        ))

    # Optional layout adjustments for clean formatting
    fig.update_layout(
        yaxis_title="First Preference Vote (%)",
        showlegend=False,
        plot_bgcolor='white'
    )
    # 2) Add downsampled invisible scatter for hover linking
    #    randomly pick `sample_size` indices

    #sample_size = 100
    #hover_y = np.random.choice(party_data, sample_size, replace=False)

    sample_size = 100  # total size
    sorted_data = np.sort(party_data)  # sort from smallest to largest
    lowest_20 = sorted_data[:20]
    highest_15 = sorted_data[-15:]
    middle_values = sorted_data[15:-15]
    middle_65 = np.random.choice(middle_values, 65, replace=False)
    hover_y = np.concatenate([lowest_20, highest_15, middle_65])

    

    fig.add_trace(go.Scatter(
        x=[site_abbreviation_dict.get(party, party)] * sample_size,
        y=hover_y,
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=4),
        #customdata=pick,          # simulation index
        hoverinfo='none',
        hovertemplate=(
            f"<b>{party}</b><br>"
            + "%{y:.1f}%<br>"
            + "<extra></extra>"
        ),
        name=f"{party} sims",
        showlegend=False
    ))

    # --- Layout adjustments ---
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        title=dict(
        text="Distribution of First Preference Votes",
        x=0.01,  # far left
        xanchor='left',
        yanchor='top',
        y=0.99,
        font=dict(size=26)
    ),
        legend=dict(
        font=dict(
            size=26,  # Increase legend font size
        ),
        x=0.7,  # Adjust legend position horizontally
        y=1,  # Adjust legend position vertically
        traceorder='normal',
        bgcolor='rgba(255, 255, 255, 0)',  # Transparent background for legend
        borderwidth=2,  # Border around the legend
        itemwidth=30,  # Set width of legend items (if needed for large labels)
    ),
        #yaxis_title="First Preference Vote in 1,000 Simulations",
        violinmode="group",
        violingap=0.1,
        margin=dict(l=30, r=30, t=60, b=100),
        xaxis_tickangle=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(
        tick0=0,
        dtick=5,
        tickformat='.0f',
        ticksuffix='%',
        tickfont=dict(size=16),  # make y-axis numbers larger
        rangemode="tozero"
    )

    fig.show()

    return fig





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



def distribution_to_top_2(final_simulated_votes, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind):

    TCP_COMBINATION_INDEX = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6, ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14}

    electorate_names = Results_dict.keys()
    n_simulations = len(final_simulated_votes['Farrer'])
    
    joint_noise = np.random.normal(0, sigma_joint, size=(n_simulations, 15))

    from collections import defaultdict, Counter

    # For goal 1
    per_electorate_winners = defaultdict(list)  # {'Electorate A': ['ALP', 'ALP', 'LP', ...]} # care about IND1,IND2

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


            proportions_with_joint_noise.iloc[:] += joint_noise[sim_id][:, np.newaxis] + np.random.normal(0, sigma_ind, size=sim_votes.shape)
            #import pdb;pdb.set_trace()




            
            # Step 1: Get top 2 parties by vote share
            top2_indices = np.argsort(sim_votes)[-2:][::-1]
            #top2_votes = sim_votes[top2_indices]
            top2_parties = [Results_dict[electorate].columns[i] for i in top2_indices]  # depends on how parties are ordered
            
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
            transfer_proportions = proportions_df.iloc[row_index].values # shape: (n_parties,)

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

            per_electorate_winners[electorate].append(winner_party) # vertical lists
            per_simulation_winners[sim_id].append(winner_party) # horizontal lists
            #print("winner: ", winner_party)
            #import pdb;pdb.set_trace()


    return (per_electorate_winners,  per_simulation_winners)  # dict[str, dict[str, int]] — useful for percentages ; list[dict[str, int]] or pd.DataFrame



def simulate_Farrer_byelection(n_simulations, Days_to_election, poll_alpha = 44, w = 0.3):

    elections = ['Byelection']
    election_year = 'Byelection'

    div_order_dict = {}
    prior_df_active_dict = {}
    polling_avg_dict = {}
    National_prior_dict = {}

    Prior_estimates_dict_active_per_election = {}
    Results_dict_active_per_election = {}

    byelection_group_structure = []

    GLOBAL_CSVs = {
        "multiple_INDs_df": {election_year: pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None) for election_year in elections},
        "C200_IND_splits": pd.read_csv("Independent_splits_multiple.csv", index_col=None),
        "C200_IND_positions_df": pd.read_csv("C200_IND_positions_df.csv", index_col=None),
        "NP_ratios_curr": pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)
    }


    with open("Farrer_byelection_covms.pkl", "rb") as f:
        data = pickle.load(f)



    
    Nat_poll_covm = adjust_covm_for_time(
        final_day_covm=data["Nat_poll_covm"], 
        days_to_election=Days_to_election, 
        params_dict=OPT_EXP_DECAY_PARAMETERS
    )   

    Seat_poll_covm_dict = {election_year: data["Seat_poll_covm"] * SEAT_POLL_SCALE_2}
    Electorate_residual_covm_dict = {election_year: Nat_poll_covm}
    Nat_poll_covm_dict = {election_year: data["Nat_poll_covm"]}







    Seat_polls =  pd.read_csv(f"SeatPollsByelectionFormatted.csv", index_col=None)
    Seat_polls = Seat_polls.loc[Seat_polls['Electorate']=='Farrer',]

    Seat_poll_year_dict = {election_year: Seat_polls}       

    

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
    polling_avg = pd.DataFrame([[0.2228,0.3071,0.1259,0.227,0.072,0.0452]], columns=['COAL','ALP','GRN','ON','IND','OTH'], index=['Farrer'])
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



    nat_sim = simulate_nat_polling_full_model(
            n_simulations,
            election_year,
            prior_df_active_dict,
            polling_avg_dict,
            National_prior_dict,
            Prior_estimates_dict_active_per_election,
            Results_dict_active_per_election,
            byelection_group_structure,
            Nat_poll_covm_dict,
            Electorate_residual_covm_dict
        )[0]
    seat_sim = simulate_seat_polling_full_model(
            n_simulations,
            election_year,
            Seat_poll_year_dict,
            Seat_poll_covm_dict,
            Results_dict_active_per_election,
            prior_alpha=22,
            poll_alpha=poll_alpha,
            GLOBAL_CSVs=GLOBAL_CSVs
        )
    


    base_idx = np.random.permutation(n_simulations)
    n_nat = int(w * n_simulations)
    idx_nat = base_idx[:n_nat]
    idx_seat = base_idx[n_nat:]

    div = 'Farrer'
    Ballot_order = Results_dict_active['Farrer'].columns
    combined_samples = np.concatenate((nat_sim[div][idx_nat], seat_sim[div][idx_seat]),axis=0)

    seat_df = pd.DataFrame(seat_sim['Farrer'], columns = Ballot_order)
    nat_df = pd.DataFrame(nat_sim['Farrer'], columns = Ballot_order)
    combined_df = pd.DataFrame(combined_samples, columns = Ballot_order)
    
    
    #make_single_electorate_violin_plot(combined_df)
    #make_single_electorate_violin_plot(seat_df)
    #make_single_electorate_violin_plot(nat_df)


    sigma_joint, sigma_ind = 4.2, 0.5 + 4 # Additional scaling TBD

    # adjust sigma_joint 

    with open(f"TCP_pair_category_dict_for_2028.pkl", "rb") as f:
        proportions_transferred_to_first = pickle.load(f)
    party_category_dict = make_party_category_dict()
    party_to_category_centered_IND = {k: ('IND' if v == 'Centre' else v) for k, v in party_category_dict.items()}


    df = proportions_transferred_to_first['Farrer']

    df.loc[1,'LP'] = 90
    df.loc[1,'NP'] = 90
    #df.loc[1,'ON'] = 49
    df.loc[6,'IND'] = 73
    df.loc[9,'LP'] = 33
    df.loc[9,'NP'] = 33
    proportions_transferred_to_first['Farrer'] = df

    per_electorate_winners, per_simulation_winners = distribution_to_top_2({div:combined_samples}, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind)
    
    from collections import Counter
    print(Counter(per_electorate_winners['Farrer']))

    import pdb; pdb.set_trace()


First_Preference_model = simulate_Farrer_byelection(n_simulations=1000, Days_to_election=15,poll_alpha = 44, w = 0.3)








import pdb; pdb.set_trace()

