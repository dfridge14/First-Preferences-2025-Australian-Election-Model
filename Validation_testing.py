import pymc as pm
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

def get_results_df(election_year, to_Fundamentals = True):
    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    # CountNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


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



# test, get data for 2025!
A, B = get_results_df('2025', to_Fundamentals = False)

import pdb; pdb.set_trace()


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


def get_National_State_Prior_estimates(election_year, new_vote_totals_states, dont_add_ON = False, adjust_No_OTHs = True):

    Prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON)[0].rename(columns={'Other':'OTH'}) # adds ON to every seat if no ON (for 2019 and 2022)

    merged_totals = Prior_estimates_df.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
    weights = merged_totals['new_vote_totals']/merged_totals['new_vote_totals'].sum()
    National_prior = (merged_totals.iloc[:,:-1] * weights.values[:,None]).sum().to_frame().T

    merged_totals_states = Prior_estimates_df.merge(new_vote_totals_states.set_index('div_nm'), left_index=True, right_index=True)

    State_prior_df_list = [] 
    for state in sorted(merged_totals_states['StateAb'].unique()):
        merged_totals_curr_state = merged_totals_states.loc[merged_totals_states['StateAb']==state,].drop('StateAb', axis = 1) # no longer need StateAb
        curr_weights = merged_totals_curr_state['new_vote_totals']/merged_totals_curr_state['new_vote_totals'].sum()
        State_prior = (merged_totals_curr_state.iloc[:,:-1] * curr_weights.values[:,None]).sum().to_frame().T
        State_prior.index = [state]
        State_prior_df_list.append(State_prior)
    State_prior_df = pd.concat(State_prior_df_list)



    # add 0.001 to Gorton Other in 2016, or 0 others in 2019/2022 (use ON votes as they are higher --> less distortion)!
    No_OTH_divisions = []

    if election_year == '2016':
        Prior_estimates_df.loc[Prior_estimates_df.index=='Gorton',['GRN','OTH']] += (-0.005,+0.005)
    elif adjust_No_OTHs:
        No_OTH_divisions = Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,].index
        Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,['GRN','OTH']] += (-0.005,+0.005)


    return Prior_estimates_df, National_prior, State_prior_df, No_OTH_divisions


def sigmoid(x, midpoint=1.0, steepness=2.0):
    """Smooth saturation from 0 to 1 centered at midpoint."""
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def compute_effective_state_weights(P_s_y, steepness=2.0):
    """
    Compute effective weights for state polling using a sigmoid of relative precision.
    
    Parameters:
    - P_s_y: pd.Series or np.ndarray of state polling precision per state/year.
    - s: Global trust level in state polling (between 0 and 1).
    - steepness: Controls how rapidly weight saturates toward 1.

    Returns:
    - effective_s: np.ndarray of weights between 0 and s (smoothly scaled).
    """
    # Compute average precision (per-year if needed)
    avg_precision = np.mean(P_s_y)
    relative_precision = P_s_y / avg_precision

    # Apply sigmoid to smooth-saturate the weighting
    scaled = sigmoid(relative_precision, midpoint=1.0, steepness=steepness)
    return scaled


# scale state weights nicely
Average_Precisions = pd.read_csv("State_Polling_Average_Precisions.csv")
Scaled_Precisions =  compute_effective_state_weights(Average_Precisions['Mean_precision'], steepness=2.0)
Average_Precisions.loc[:,"Scaled_Precisions"] = Scaled_Precisions
Average_Precisions.set_index('Year').to_csv("State_Polling_Scaled_Precisions.csv", index = True)
#import pdb;pdb.set_trace()



def non_uniform_swing_weight_exp(O_series, beta=0.5, start=0.2, k=10):

    O = O_series.values
    weight = np.ones_like(O)

    mask = O > start
    decay = np.exp(-k * (O[mask] - start))
    weight[mask] = (1 - beta) * decay + beta

    return pd.Series(weight, index=O_series.index)

def full_weight_vector(all_divisions, high_others_df, others_column="OTH", division_state_indices=None, beta=0.5, k=10, start=0.2):
    """
    Parameters:
        all_divisions: list or Index of all 150 electorates
        high_others_df: DF with OTH column and index = electorates with high Others
        division_state_map: Series mapping each electorate to its state index
    """

    # Step 1: Exponential weights for high Others
    high_weights = non_uniform_swing_weight_exp(high_others_df[others_column], beta=beta, k=k, start=start)

    # Step 2: Build full weight vector (init with NaNs)
    full_weights = pd.Series(index=all_divisions, dtype=float)
    full_weights.loc[high_weights.index] = high_weights

    # Step 3: Fill in low Others weights to preserve state average
    for state_idx in np.unique(division_state_indices):
        # Get all divisions in this state
        div_mask = division_state_indices == state_idx
        state_divisions = np.array(all_divisions)[div_mask]
        state_weights = full_weights.loc[state_divisions]

        n_known = state_weights.notna().sum()
        sum_known = state_weights.dropna().sum()
        n_total = len(state_weights)
        n_unknown = n_total - n_known

        if n_unknown > 0:
            required_total = n_total * 1.0
            constant = (required_total - sum_known) / n_unknown
            full_weights.loc[state_weights[state_weights.isna()].index] = constant
            #print(constant)
        else:
            # All weights known — nothing to fill
            continue

        #import pdb;pdb.set_trace()

    return full_weights


states = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']
state_to_index = {state: i for i, state in enumerate(states)}
election_years = ['2016','2019','2022','2025']
beta_list = [r for r in np.arange(0,1.01,0.1)]

weights_by_beta = {}

for election_year in election_years:

    if election_year != '2025':
        div_to_state = pd.read_csv(f"{election_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'}).set_index('div_nm')

    else:
        div_to_state = pd.read_csv(f"2022HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        div_to_state.loc[div_to_state['div_nm'] == 'North Sydney',] = 'Bullwinkel', 'WA'
        div_to_state = div_to_state.loc[~(div_to_state['div_nm'] == 'Higgins'),]
        div_to_state = div_to_state.set_index('div_nm')

        # re-order
        Electorate_order_2025 = pd.read_csv('Electorate_names_2025.csv')['post_title']

        div_to_state = div_to_state.loc[Electorate_order_2025.values]


    division_state_indices = div_to_state['StateAb'].map(state_to_index).values 

    all_divisions = div_to_state.index

    High_Others_df = pd.read_csv(f"High_Prior_OTH_Electorates_{election_year}.csv", index_col = 0)

    #import pdb;pdb.set_trace()


    # Pre-compute for each election_year and beta value: 0 to 1, intervals 0.1
    weights_by_beta[election_year] = {
        np.round(beta_val,2): full_weight_vector(
            all_divisions, High_Others_df,
            others_column="OTH",
            division_state_indices=division_state_indices,
            beta=beta_val,
            k=10,
            start=0.2
        )
        for beta_val in beta_list 
    }

#import pdb;pdb.set_trace()



def simulate_Polling_Fundamentals_model(n_simulations, election_year, df_t = 0, v = 0.1, s = 0.6, beta = 0.5):

    dist = "Normal" if df_t == 0 else "t"
    #print(dist)
    Volatility_cat = pd.read_csv(f"Volatility_weights_df_{election_year}.csv", index_col= None)

    weights_idx_dict = defaultdict(list)
    for idx, scale in enumerate(Volatility_cat['Volatility_weights']):
        weights_idx_dict[scale].append(idx)

    weights_idx_dict = dict(weights_idx_dict)
    


    National_Polling_error_ALR_cov = pd.read_csv(f"PollingErrorALRCovarianceNational{election_year}.csv", index_col=0)
    National_Simulated_polling_error = np.random.multivariate_normal(mean = np.zeros(len(National_Polling_error_ALR_cov)), cov = National_Polling_error_ALR_cov.values, size=n_simulations)[:, None, :] 
    # Broadcast national polling results across 10000 simulations and 150 electorates
    National_Simulated_polling_error_expanded = np.repeat(National_Simulated_polling_error, NO_OF_ELECTORATES[election_year], axis=1)


    National_Election_error_ALR_cov = pd.read_csv(f"ElectionErrorALRCovarianceNational{election_year}.csv", index_col=0)
    National_Simulated_election_error = np.random.multivariate_normal(mean = np.zeros(len(National_Election_error_ALR_cov)), cov = National_Election_error_ALR_cov.values, size=n_simulations)[:, None, :] 
    National_Simulated_election_error_expanded = np.repeat(National_Simulated_election_error, NO_OF_ELECTORATES[election_year], axis=1)


    # state errors
    State_Polling_error_ALR_cov = pd.read_csv(f"PollingErrorALRCovarianceStateDeviation{election_year}.csv", index_col=0) 
    State_Simulated_polling_error = np.random.multivariate_normal(mean = np.zeros(len(State_Polling_error_ALR_cov)), cov = State_Polling_error_ALR_cov.values, size=n_simulations*NO_OF_STATES)
    State_Simulated_polling_error = State_Simulated_polling_error.reshape(n_simulations, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])

    State_Election_error_ALR_cov = pd.read_csv(f"ElectionErrorALRCovarianceStateDeviation{election_year}.csv", index_col=0)  
    State_Simulated_election_error = np.random.multivariate_normal(mean = np.zeros(len(State_Election_error_ALR_cov)), cov = State_Election_error_ALR_cov.values, size=n_simulations*NO_OF_STATES)
    State_Simulated_election_error = State_Simulated_election_error.reshape(n_simulations, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])


    #import pdb;pdb.set_trace()

    # this should be tuned:
    Electorate_Residuals_cov = pd.read_csv(f"ElectorateResidualALRCovariance{election_year}.csv", index_col=0) 

    Scaled_covs = {}
    category_weights = {0:0.95, 1:1.25,2:1.5,3:4}

    OTH_index = -1
    for cat, weight in category_weights.items():
        scaling = (1 + (weight-1) * v) if (weight > 1) else (1 - (1-weight)*v) # proper variability of Others
        
        # Adjust the OTH variance and covariances
        cov_adj = Electorate_Residuals_cov.values.copy()
        
        # Amount to add to the OTH row and column
        current_var = Electorate_Residuals_cov.values[OTH_index, OTH_index]
        target_var = scaling**2 * current_var
        delta = target_var - current_var
        
        # Add delta to variance
        cov_adj[OTH_index, OTH_index] += delta
        
        # Add delta to covariances with OTH, assuming proportional increase
        for i in range(DIM_OF_COV_MATRIX[election_year]):
            if i != OTH_index:
                cov_adj[OTH_index, i] *= scaling
                cov_adj[i, OTH_index] = cov_adj[OTH_index, i]  # keep symmetry
        
        Scaled_covs[cat] = cov_adj
        #import pdb;pdb.set_trace()


    #Electorate_Residuals_Simulated_error = np.random.multivariate_normal(mean = np.zeros(len(Electorate_Residuals_cov)), cov = Electorate_Residuals_cov.values, size=n_simulations*NO_OF_ELECTORATES[election_year])
    #Electorate_Residuals_Simulated_error = Electorate_Residuals_Simulated_error.reshape(n_simulations, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year])



    Electorate_Residuals_Simulated_error = np.empty((n_simulations, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year]))

    d = len(Electorate_Residuals_cov)
    n = n_simulations * NO_OF_ELECTORATES[election_year]
    cov = Electorate_Residuals_cov.values

    # Batch simulate by category:
    for scale, indices in weights_idx_dict.items():
        cov = Scaled_covs[scale]  # Use the scaled covariance matrix
        n_group = len(indices)

        if n_group == 0:
            continue  # skip empty groups

        # Generate multivariate normal samples for this group - # 2. Standard multivariate normal samples # shape: (n, d)

        group_sims = np.random.multivariate_normal(
            mean=np.zeros(d),
            cov=cov,
            size=(n_simulations * n_group)
        )

        if dist == 't':
        
            g = np.random.gamma(df_t / 2., 2. / df_t, size=n_simulations *n_group)   # 1. Gamma samples (for scaling) ; shape: (n,)
            group_sims = group_sims / np.sqrt(g)[:, None]  # shape: (n, d) # 3. Scale by sqrt(gamma) to simulate from t-distribution


        # Place the group simulations into the correct positions
        Electorate_Residuals_Simulated_error[:, indices, :] = group_sims.reshape(n_simulations, n_group, DIM_OF_COV_MATRIX[election_year]) # 4. Reshape to (n_simulations, electorates, dim)



    # weights of each state and division:
    Div_relative_weights_dict = {}
    State_relative_weights_dict = {}
    for year in ['2016','2019','2022','2025']:
        last_election_year = str(int(year) - 3)

        Enrolment_by_Div_prev = pd.read_csv(f"{last_election_year}GeneralEnrolmentByDivision.csv",index_col=None, skiprows=1).rename(columns={'DivisionNm':'old_div','StateAb':'State'})[['old_div','State','Enrolment']]
        # adjust for redistribution
        Correspondence_old_new = pd.read_csv(f"Correspondence_CED_{str(int(year) - 4)}_{str(int(year) - 1)}.csv")
        merged = Correspondence_old_new.merge(Enrolment_by_Div_prev, on='old_div')
        merged['Enrolment'] = merged['Enrolment'] * merged['RATIO_FROM_TO']
        Enrolment_by_Div = merged.groupby(['new_div','State'])['Enrolment'].sum().reset_index().rename(columns={'new_div':'div_nm'})

        Enrolment_by_State_prev = Enrolment_by_Div.groupby('State')['Enrolment'].sum()
        State_relative_weights = Enrolment_by_State_prev / Enrolment_by_State_prev.sum()
        Div_relative_weights = Enrolment_by_Div.iloc[:,:2]
        Div_relative_weights.loc[:,'Relative weights'] = Enrolment_by_Div['Enrolment'] / Enrolment_by_Div.groupby('State')['Enrolment'].transform('sum')
        Div_relative_weights = Div_relative_weights.set_index('div_nm')

        Div_relative_weights_dict[year] = Div_relative_weights
        State_relative_weights_dict[year] = State_relative_weights


    Div_relative_weights = Div_relative_weights_dict[election_year]
    State_relative_weights = State_relative_weights_dict[election_year]

    # centre state polling/swing deviations to weighted sum of 0!
    w = State_relative_weights.values.reshape(1,NO_OF_STATES,1)
    weighted_means = np.sum(State_Simulated_polling_error * w, axis=1, keepdims=True)  # Sum over states
    State_Simulated_polling_error_centered = State_Simulated_polling_error - weighted_means # Subtract the weighted mean from each state: shape still (10000, 8, 5)

    weighted_means = np.sum(State_Simulated_election_error * w, axis=1, keepdims=True)  # Sum over states
    State_Simulated_election_error_centered = State_Simulated_election_error - weighted_means # Subtract the weighted mean from each state: shape still (10000, 8, 5)

    Scaled_precisions_curr = pd.read_csv("State_Polling_Scaled_Precisions.csv", index_col = 0).loc[int(election_year)]
    relative_state_precisions = Scaled_precisions_curr.set_index('Scope', drop = True).drop('Mean_precision', axis = 1)

    s_i = s * relative_state_precisions

    #print("s_i", v, s)

    # weight the State poll deviaiton and electorate poll deviation by s_i per state
    s_i_reshaped = s_i.values.reshape(1,NO_OF_STATES,1)

    State_Simulated_polling_error_centered = s_i_reshaped * State_Simulated_polling_error_centered + (1-s_i_reshaped) * State_Simulated_election_error_centered


    #import pdb;pdb.set_trace()


    # reshape State simulations correctly - map to correct div_nms
    states = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']
    state_to_index = {state: i for i, state in enumerate(states)}
    division_state_indices = Div_relative_weights['State'].map(state_to_index).values  # shape (150,)

    State_Simulated_polling_error_centered_expanded = State_Simulated_polling_error_centered[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]
    State_Simulated_election_error_centered_expanded = State_Simulated_election_error_centered[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]

    #import pdb;pdb.set_trace()


    # Centre Electorate Residuals to 0!
    w_div = Div_relative_weights.iloc[:,1:].values.reshape(1,NO_OF_ELECTORATES[election_year],1)
    weighted_means = np.sum(Electorate_Residuals_Simulated_error * w_div, axis=1, keepdims=True)

    centered_residuals = np.empty_like(Electorate_Residuals_Simulated_error)

    for state_idx in range(len(states)):  # 8 states

        div_indices = np.where(division_state_indices == state_idx)[0] #  Find divisions that belong to this state
        weights = Div_relative_weights.iloc[div_indices]['Relative weights'].values.reshape(1, -1, 1) # Get relative weights for these divisions
        residuals = Electorate_Residuals_Simulated_error[:, div_indices, :] # Get the residuals for these divisions
        weighted_mean = np.sum(residuals * weights, axis=1, keepdims=True)  # Compute weighted state mean
        centered_residuals[:, div_indices, :] = residuals - weighted_mean # Center the residuals

    Electorate_Residuals_Simulated_error_centered = centered_residuals






    if election_year in ['2016','2019','2022','2025']:
        last_election_vote_totals = pd.read_csv(f"{last_election_year}HouseVotesCountedByDivision.csv", skiprows=1, index_col=None).rename(columns={'DivisionNm':'old_div'})[['old_div', 'TotalVotes']]
        redistribution_df = pd.read_csv(f'Correspondence_CED_{str(int(election_year)-4)}_{str(int(election_year)-1)}.csv', index_col = None)

        merged_df = redistribution_df.merge(last_election_vote_totals, on="old_div")
        merged_df["new_vote_totals"] = merged_df["TotalVotes"] * merged_df["RATIO_FROM_TO"]
        new_vote_totals = merged_df.groupby("new_div")["new_vote_totals"].sum().reset_index().rename(columns={'new_div':'div_nm'})

        if election_year == '2025':
            div_to_state = pd.read_csv(f"2022HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
            div_to_state.loc[div_to_state['div_nm'] == 'North Sydney',] = 'Bullwinkel', 'WA'
            div_to_state = div_to_state.loc[~(div_to_state['div_nm'] == 'Higgins'),]
        else:
            div_to_state = pd.read_csv(f"{election_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        new_vote_totals_states = new_vote_totals.merge(div_to_state, on = 'div_nm', how='left')


    if election_year in ['2016','2019','2022','2025']:
        Prior_estimates_df, National_prior, State_prior_df, No_OTH_divisions = get_National_State_Prior_estimates(election_year, new_vote_totals_states)
        #import pdb;pdb.set_trace()
        #Prior_estimates_df.loc[Prior_estimates_df['OTH']>0.2,['OTH']].to_csv(f"High_Prior_OTH_Electorates_{election_year}.csv", index = True)


    def alr_to_simplex_simulation_array(alr_array):
        """
        Perform inverse ALR transformation on a 3D array of shape (S, D, K).
        Returns a 3D array of shape (S, D, K+1) in the probability simplex.
        """
        exp_vals = np.exp(alr_array)  # shape (S, D, K)
        ref_vals = 1 / (1 + np.sum(exp_vals, axis=-1, keepdims=True))  # shape (S, D, 1)
        simplex = np.concatenate([ref_vals, exp_vals * ref_vals], axis=-1)  # shape (S, D, K+1)
        return simplex





    # test for variability of seat results corrected for state swings - Obtain

    def test_variability_of_Electorate_Residuals(new_vote_totals_states, year_to_remove):

        State_Results_2016_2022 = pd.read_csv('StateResults2016_2022.csv', index_col=None)

        CAGO = 1 if year_to_remove == '2016' else 0

        Electorate_residuals_list = []

        for year in [p for p in ['2016','2019','2022'] if p != year_to_remove]: # ['2016','2019','2022']:
            NUM_DIMS = DIM_OF_COV_MATRIX[year_to_remove] if year != '2016' else 3

            Prior_estimates_df1, National_prior1, State_prior_df1, No_OTH_divisions1 = get_National_State_Prior_estimates(year, new_vote_totals_states, dont_add_ON = True, adjust_No_OTHs=False)
            State_Results_curr = State_Results_2016_2022.loc[State_Results_2016_2022['Election']==int(year),]
            Results_df = get_results_df(year)[0].rename(columns={'Other':'OTH'})
            
            if CAGO and (year in ['2019','2022']):
                Prior_estimates_df1.loc[:,'OTH'] = Prior_estimates_df1.iloc[:,-3:].sum(axis=1)
                Prior_estimates_df1 = Prior_estimates_df1.drop(columns=['ON','UAPP'])
                State_prior_df1.loc[:,'OTH'] = State_prior_df1.iloc[:,3:6].sum(axis=1)
                State_prior_df1 = State_prior_df1.drop(columns=['ON','UAPP'])
                State_Results_curr.loc[:,'OTH'] = State_Results_curr.iloc[:,3:6].sum(axis=1)
                State_Results_curr = State_Results_curr.drop(columns=['ON','UAPP'])
                Results_df.loc[:,'OTH'] = Results_df.iloc[:,3:6].sum(axis=1)
                Results_df = Results_df.drop(columns=['ON','UAPP'])

            elif year == '2016':
                State_Results_curr = State_Results_curr.drop(columns=['ON','UAPP'])

            else: # not CAGO and year is 2019/2022 - replace 0s with nan!
                Prior_estimates_df1 = Prior_estimates_df1.replace(0,np.nan)
                State_prior_df1 = State_prior_df1.replace(0,np.nan)
                State_Results_curr = State_Results_curr.replace(0,np.nan)
                Results_df = Results_df.replace(0,np.nan)

            State_Results_curr = State_Results_curr.drop('Election', axis=1).set_index('State')
            
            # convert both prior and results to ALR
            ref_col = 'COAL'
            State_Results_ALR = np.log(State_Results_curr.drop(columns=[ref_col]).div(State_Results_curr[ref_col], axis=0))
            State_Prior_ALR = np.log(State_prior_df1.drop(columns=[ref_col]).div(State_prior_df1[ref_col], axis=0))
            Prior_estimates_ALR_df1 = np.log(Prior_estimates_df1.drop(columns=[ref_col]).div(Prior_estimates_df1[ref_col], axis=0))
            #import pdb;pdb.set_trace()

            Div_relative_weights = Div_relative_weights_dict[year]

            # add to corresponding divisions in states
            True_State_ALR_swings = State_Results_ALR - State_Prior_ALR
            Prior_estimates_ALR_df1.loc[:,'State'] = Div_relative_weights['State'].values
            merged = pd.merge(Prior_estimates_ALR_df1, True_State_ALR_swings, left_on = 'State',right_index = True, suffixes = ('','_state_swing'))
            #import pdb;pdb.set_trace()
            merged.iloc[:,:NUM_DIMS] += merged.iloc[:,-NUM_DIMS:].values
            State_swing_ALR = merged.iloc[:,:NUM_DIMS]
            #

            # get actual results for 4/6 parties
            Results_df_ALR = np.log(Results_df.drop(columns=[ref_col]).div(Results_df[ref_col], axis=0))
            Electorate_residuals = Results_df_ALR - State_swing_ALR

            Electorate_residuals.index = year + Electorate_residuals.index

            if (year == '2016') and (year_to_remove != '2016'):
                Electorate_residuals.loc[:,['ON','UAPP']] = np.nan,np.nan

            Electorate_residuals_list.append(Electorate_residuals)


        Electorate_residuals_ALR_df = pd.concat(Electorate_residuals_list)
        Electorate_residuals_ALR_df = Electorate_residuals_ALR_df.loc[~(Electorate_residuals_ALR_df.index.str.startswith(year_to_remove)),]

        if year_to_remove != '2016':
            Electorate_residuals_ALR_df = Electorate_residuals_ALR_df[['ALP','GRN','ON','UAPP','OTH']]



        #import pdb;pdb.set_trace()


        return Electorate_residuals_ALR_df.cov(min_periods=1)

    test = 0

    if test:
        Electorate_residuals_covMs = {}
        for year_to_remove in ['2016','2019','2022','2025']:
            Electorate_residuals_covMs[year_to_remove] = test_variability_of_Electorate_Residuals(new_vote_totals_states, year_to_remove)
            import pdb;pdb.set_trace()

            Electorate_residuals_covMs[year_to_remove].to_csv(f"ElectorateResidualALRCovariance{year_to_remove}.csv", index = True)

        #import pdb;pdb.set_trace()

    #print(National_prior)



    Day = 90

    day_z_polling_avg_df = pd.read_csv(f"National_Day_{Day}_Polls.csv")

    day_z_polling_avg = day_z_polling_avg_df.loc[day_z_polling_avg_df['Election'] == int(election_year),].drop('Election', axis = 1).reset_index(drop=True)

    if election_year == '2016':
        day_z_polling_avg = day_z_polling_avg[['COAL','ALP','GRN','OTH']]
    elif election_year == '2025':
        day_z_polling_avg = day_z_polling_avg.rename(columns = {'UAPP':'TOP'})
        #day_z_polling_avg.iloc[0,:] = 0.39,0.305,0.12,0.07,0.01,0.105

    #print(day_z_polling_avg)




    #day_80_polling_avg_dict = {'2016': pd.DataFrame([[0.412262, 0.351972, 0.105693, 0.130074]], columns = ['COAL','ALP','GRN','OTH']), \
    #                    '2019':pd.DataFrame([[0.384782, 0.36451, 0.095751, 0.035, 0.031, 0.088957]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
    #                    '2022':pd.DataFrame([[0.355905, 0.362643, 0.118432, 0.0383,0.0244,0.10032,]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
    #                    '2025':pd.DataFrame([[0.346862,  0.319669,  0.128995,  0.071243,  0.016797,  0.116434]], columns = ['COAL','ALP','GRN','ON','TOP','OTH'])}
    
                        # 0.352028, 0.3153, 0.127548, 0.07171, 0.011641, 0.121772

                        # 0.346862  0.319669  0.128995  0.071243  0.016797  0.116434

    #import pdb;pdb.set_trace()

    
    state_poll_dev_alr = pd.read_csv("State_Polling_Deviations_from_National.csv", index_col=None)
    state_poll_dev_alr_2025 = pd.read_csv(f"2025_State_Polling_Deviations_from_National_Day_{Day}.csv", index_col=None)


    State_Polls_Deviations_from_National_df_dict = {'2016': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2016,].drop(['ON','Election_year'], axis=1), \
                                                    '2019': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2019,].drop(['Election_year'], axis=1).fillna(0), \
                                                    '2022': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2022,].drop(['Election_year'], axis=1).fillna(0), \
                                                    '2025': state_poll_dev_alr_2025.drop(['Election_year'], axis=1).fillna(0),}



    # get alr values of all quantities
    ref_col = 'COAL'
    polling_alr = np.log(day_z_polling_avg.drop(columns=[ref_col]).div(day_z_polling_avg[ref_col], axis=0))
    National_prior_alr = np.log(National_prior.drop(columns=[ref_col]).div(National_prior[ref_col], axis=0))

    State_prior_alr =  np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))
    State_prior_expanded = np.tile(State_prior_alr.to_numpy(), (n_simulations, 1, 1)).reshape(n_simulations, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])
    State_prior_expanded = State_prior_expanded[np.arange(n_simulations)[:, None], division_state_indices[None, :], :] # Then use advanced indexing to map divisions to their state across all samples

    #import pdb;pdb.set_trace()
    # get initial state deviations and expand
    State_prior_deviation_alr_expanded = State_prior_expanded - National_prior_alr.values.flatten()








    Prior_estimates_alr =  np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))
    Prior_estimates_alr_expanded = np.tile(Prior_estimates_alr.to_numpy(), (n_simulations, 1, 1))

    # get State deviations into a (10000, 8, 5) array
    State_polling_deviation_alr = State_Polls_Deviations_from_National_df_dict[election_year].set_index('State')
    if election_year in ['2019','2022']:
        State_polling_deviation_alr.loc[:,'UAPP'] = 0.0 # add 0 deviation from National if no state polling!
        State_polling_deviation_alr = State_polling_deviation_alr[['ALP','GRN','ON','UAPP','OTH']]
    elif election_year == '2025':
        State_polling_deviation_alr.loc[:,'TOP'] = 0.0 # add 0 deviation from National if no state polling!
        State_polling_deviation_alr = State_polling_deviation_alr[['ALP','GRN','ON','TOP','OTH']]

    # scale the state polling deviations, based on relative polling precision (sample size etc.)

    #relative_state_precisions = relative_state_precisions['Scaled_Precisions']/relative_state_precisions['Scaled_Precisions'].mean()

    
    State_polling_deviation_alr = State_polling_deviation_alr.mul(s_i.values, axis = 0) # now scaled based on state precision


    #import pdb;pdb.set_trace()


    State_polling_deviation_alr_matrix = State_polling_deviation_alr.values  # Convert to numpy array for easy broadcasting
    State_polling_deviation_alr_matrix = np.expand_dims(State_polling_deviation_alr_matrix, axis=0)  # Add an extra dimension for broadcasting
    State_polling_deviation_alr_matrix_expanded = np.repeat(State_polling_deviation_alr_matrix, n_simulations, axis=0)  
    State_polling_deviation_alr_matrix_expanded = State_polling_deviation_alr_matrix_expanded[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]

    #State_prior_alr = np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))

    # apply National Polling error
    Simulated_national_result_alr = National_Simulated_polling_error_expanded + polling_alr.values  # shape: [1M, 5]

    # apply State Polling error
    Simulated_state_polling_deviation = State_polling_deviation_alr_matrix_expanded + State_Simulated_polling_error_centered_expanded

    # weight State Polling, relative to National: use linear combination of last year's 
    #import pdb;pdb.set_trace()
    
    #state_weights = s * Year_state_polling_weights.loc[int(election_year)].iloc[0]

    # get 1 - s_i per state

    Scaled_national_prior_deviations = (State_prior_alr - National_prior_alr.values).mul(1-s_i.values, axis = 0)
    Scaled_national_prior_deviations.index = Scaled_national_prior_deviations.index.map(state_to_index)
    seat_state_alr = Scaled_national_prior_deviations.iloc[division_state_indices]
    Scaled_national_prior_deviations_expanded = np.broadcast_to(seat_state_alr, (n_simulations, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year])).copy()
    
    #Scaled_national_polling_deviations_array = np.expand_dims(Scaled_national_polling_deviations, axis=0)  # Add an extra dimension for broadcasting
    #Scaled_national_polling_deviations_expanded = np.repeat(Scaled_national_polling_deviations_array, n_simulations, axis=0)  
    #Scaled_national_polling_deviations_expanded = Scaled_national_polling_deviations_expanded[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]


    # get High Others votes and index set of all divisions
   
    Non_Uniformity_weights = weights_by_beta[election_year][np.round(beta,2)].values[None,:,None]


    Combined_state_mean = polling_alr.values + Scaled_national_prior_deviations_expanded + State_polling_deviation_alr_matrix_expanded # already scaled!
    Combined_state_errors = National_Simulated_polling_error_expanded + State_Simulated_polling_error_centered_expanded

    Simulated_State_Weighted_Polling_Results = Combined_state_mean + Combined_state_errors

    #Simulated_State_Polling_Results = Simulated_national_result_alr + (1 - state_weights) * (National_prior_alr.values - State_prior_expanded) + (state_weights) * Simulated_state_polling_deviation

    #import pdb;pdb.set_trace()

    Projected_Electorate_Results = Prior_estimates_alr_expanded + Non_Uniformity_weights*(Simulated_State_Weighted_Polling_Results - State_prior_expanded)

    Simulated_Electorate_Polling_Results_ALR = Projected_Electorate_Results + Electorate_Residuals_Simulated_error_centered

    Simulated_Electorate_Polling_Results = alr_to_simplex_simulation_array(Simulated_Electorate_Polling_Results_ALR)


    # Now, do the same for Fundamentals:

    # 1. GET NATIONAL SWING (centered at 0)
    # 2. START WITH PRIOR STATE DEVIATIONS ( from state_prior) - essentially from last election
    # 3. ADD SIMULATED STATE DEVIATIONS
    # 4. ADD THIS TO NATIONAL SWING RESULT
    # 5. ADD TO ALL ELECTORATES
    # 6. ADD ELECTORATE ERROR

    Simulated_national_swing_alr = National_Simulated_election_error_expanded + National_prior_alr.values # 1
    Simulated_state_election_deviation = State_prior_deviation_alr_expanded + State_Simulated_election_error_centered_expanded # 2
    Simulated_State_election_Results = Simulated_national_swing_alr + Simulated_state_election_deviation

    Projected_Electorate_Swing_Results = Prior_estimates_alr_expanded + Non_Uniformity_weights*(Simulated_State_election_Results - State_prior_expanded)

    Simulated_Electorate_Swing_Results_ALR = Projected_Electorate_Swing_Results + Electorate_Residuals_Simulated_error_centered

    Simulated_Electorate_Swing_Results = alr_to_simplex_simulation_array(Simulated_Electorate_Swing_Results_ALR)


    #import pdb;pdb.set_trace()




    # Adjust how uniform the seats swing based on difference to national polling:
    




    # Post-processing to remove artificial additions (includding ON in 2019/22)!
    if election_year == '2016':
        div_idx = Div_relative_weights.index.get_loc('Gorton')
        OTH = Simulated_Electorate_Polling_Results[:, div_idx, 3]  # shape (10000,)
        # Apply the shift: increase GRN, decrease OTH
        Simulated_Electorate_Polling_Results[:, div_idx, 2] += 1.0 * OTH  # GRN (index 2)
        Simulated_Electorate_Polling_Results[:, div_idx, 3] -= 1.0 * OTH  # OTH (index 3)

        #import pdb;pdb.set_trace()

    def shift_share(sim, div_idx, from_party_idx, to_party_idx, proportion=1.0):
        """Shifts a proportion of vote share from one party to another in one division, across all simulations."""
        shift_amount = proportion * sim[:, div_idx, from_party_idx]
        sim[:, div_idx, to_party_idx] += shift_amount
        sim[:, div_idx, from_party_idx] -= shift_amount

        return 1

    def redistribute_ON_votes(sim, division_names, party_names, ON_transfer_dict, election_year):
        party_index_map = {name: idx for idx, name in enumerate(party_names)}

        if election_year == '2025':
            ON_idx = party_index_map['TOP'] # confusing - quick adaptation to 2025, where TOP is running fewer candidates
        else:
            ON_idx = party_index_map['ON']

        for div, transfer_row in ON_transfer_dict.items():
            div_idx = division_names.index(div)

            if (election_year == '2025') and (div in ['Canberra', 'Fenner', 'Bean']):
                #import pdb;pdb.set_trace()

                ON_votes = sim[:, div_idx, [3,4]]

                sim[:, div_idx, :] += np.outer(ON_votes.sum(axis = -1), transfer_row.values)

                # Zero out ON votes
                sim[:, div_idx, [3,4]] = 0.0

            else:

                # Get ON vote array for this division
                ON_votes = sim[:, div_idx, ON_idx]  # shape (10000,)

                sim[:, div_idx, :] += np.outer(ON_votes, transfer_row.values) # add proportions scaled by ON_votes to df

                # Zero out ON votes
                sim[:, div_idx, ON_idx] = 0.0

        return 1  # confirmation


    if election_year in ['2019','2022','2025']:
        for div_nm in No_OTH_divisions:
            div_idx = Div_relative_weights.index.get_loc(div_nm)
            shift_share(Simulated_Electorate_Polling_Results, div_idx, 5, 2, proportion=1.0) # always from OTH to UAPP (maybe different in 2025?)
            shift_share(Simulated_Electorate_Swing_Results, div_idx, 5, 2, proportion=1.0)

            # return ON back to its country in divs it did not run in!
            ON_transfer_dict = remove_ON_back_to_its_country(Prior_estimates_df, election_year)

            Polled_parties = ['COAL','ALP','GRN','ON','UAPP','OTH'] if  election_year != '2025' else ['COAL','ALP','GRN','ON','TOP','OTH']
            redistribute_ON_votes(Simulated_Electorate_Polling_Results, Div_relative_weights.index.tolist(), Polled_parties, ON_transfer_dict, election_year)
            redistribute_ON_votes(Simulated_Electorate_Swing_Results, Div_relative_weights.index.tolist(), Polled_parties, ON_transfer_dict, election_year)






    return Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results



def expand_all_divisions_from_prior_df(sim, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=100):

    LP_NP_VOLATILITY_FACTOR = 2

    final_sim = {}
    party_name_dict = {}
    NUM_MAIN_PARTIES = DIM_OF_COV_MATRIX[election_year]

    multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
    C200_IND_splits = pd.read_csv(f"Independent_splits_multiple.csv", index_col=None)
    C200_IND_positions_df = pd.read_csv("C200_IND_positions_df.csv", index_col=None)
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == int(election_year),]

    Complex_contests = {'Calare': 0.4788,'Monash': 0.5002,'Moore': 0.43987} # proportions of split to C200 independent and Incumbent independent (based on % after adding Historical proportions in (div,div) pair)



    Major_parties = ['COAL','LP','NP','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']
    if election_year != '2016':
        Major_parties += ['ON','UAPP','TOP']
        if election_year != '2025':
            Polling_parties += ['ON','UAPP']
        else:
            Polling_parties += ['ON','TOP']


    # For the split of COAL_double_divs
    if election_year == '2025':
        NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df_2025.csv", index_col=None)
    else:
        NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)

    NP_ratios_curr = NP_ratios_curr.loc[(NP_ratios_curr['election_year']==int(election_year)) ,] # & (NP_ratios_curr['State'].isin(['VIC','NSW']))
        

    for i, div in enumerate(Prior_estimates_dict.keys()): # will be alphabetical
        sim_block = sim[:, i, :]            # shape (10000, 4)
        main_parties = sim_block[:, :NUM_MAIN_PARTIES]     # shape (10000, 3)
        other_share = sim_block[:, NUM_MAIN_PARTIES]       # shape (10000,)

        #print(div)

        # Extract prior for this division

        prior_row = Prior_estimates_dict[div]
        prior_row_Other = prior_row[[p for p in prior_row.columns if p not in Major_parties]]
        minor_names = list(prior_row_Other.columns)
        rel_weights = prior_row_Other.iloc[0].values
        rel_weights = rel_weights / rel_weights.sum()

        #import pdb;pdb.set_trace()

        # Dirichlet sampling
        alpha = rel_weights * alpha_scalar
        splits = np.random.dirichlet(alpha, size=sim.shape[0])  # shape (10000, n_minor)

        # Expand 'Other' proportionally
        other_expanded = splits * other_share[:, None]  # (10000, n_minor)

        # Combine with main parties
        combined = np.concatenate([main_parties, other_expanded], axis=1)

        all_party_names = Polling_parties + minor_names


        # if no ON/TOP, demove this column!
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



            alpha = np.array([1-NP_est,NP_est]) * alpha_scalar/LP_NP_VOLATILITY_FACTOR
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
                means_array = np.full(num_INDs_curr, 1/num_INDs_curr) # initialises an even split



                if div in C200_IND_positions_curr['div_nm'].unique():
                    # split according to C200 splits!

                    C200_ratio = C200_IND_splits.loc[C200_IND_splits['Election'] == f'AverageFor{election_year}','Ratio'].iloc[0]

                    C200_IND_position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm'] == div,'Number'].iloc[0]

                    if (election_year == '2025') and div in Complex_contests.keys():

                        C200_ratio = Complex_contests[div]
                    

                    rest_value = (1 - C200_ratio) / (len(means_array) - 1)
                    means_array[:] = rest_value  # fill all
                    means_array[C200_IND_position - 1] = C200_ratio 

                elif (election_year == '2025') and (div == 'Groom'):
                    import pdb;pdb.set_trace()
                    # mix between C200 and last split! - or ignore? TBD


                splits = np.random.dirichlet(means_array* alpha_scalar, size=sim.shape[0])
                All_IND_votes = splits * IND_votes[:, None]

                combined = np.concatenate([combined, All_IND_votes], axis=1)
                combined = np.delete(combined,IND_index, axis = 1) # remove 'IND' col

                all_party_names = all_party_names + ['IND'+str(i) for i in range(1,num_INDs_curr.iloc[0]+1)]
                all_party_names = [p for p in all_party_names if p != 'IND'] # remove IND


            else:
                all_party_names = [p if p != 'IND' else 'IND1' for p in all_party_names] # renames to IND1 if only single IND


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





def get_election_MAE(combined_samples, Prior_estimates_dict, Results_dict, election_year, coverage_level, alpha):

    final_simulated_votes = expand_all_divisions_from_prior_df(combined_samples, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=alpha)[0]

    all_abs_diffs = []

    coverage_hits = 0
    total_predictions = 0

    lower_percentile = (1 - coverage_level) / 2 * 100
    upper_percentile = (1 + coverage_level) / 2 * 100


    for div in final_simulated_votes.keys():
        pred = final_simulated_votes[div] * 100 # working in percentages finally
        actual = Results_dict[div].iloc[0].values

        # Broadcast subtraction: (n_sim, n_parties) - (n_parties,) => (n_sim, n_parties)
        abs_diff = np.abs(pred - actual)
        all_abs_diffs.append(abs_diff)

        # Compute prediction intervals
        lower_bounds = np.percentile(pred, lower_percentile, axis=0)
        upper_bounds = np.percentile(pred, upper_percentile, axis=0)

        # Check if actual values fall within the prediction intervals
        within_bounds = (actual >= lower_bounds) & (actual <= upper_bounds)
        coverage_hits += np.sum(within_bounds)
        total_predictions += len(actual)

    combined_abs_diffs = np.concatenate(all_abs_diffs, axis=1)
    mae_per_simulation = combined_abs_diffs.mean(axis=1) # Average over parties for each simulation (axis=1)
    average_mae = np.mean(mae_per_simulation)

    coverage_probability = coverage_hits / total_predictions

    return average_mae, coverage_probability




def perform_validation_testing(n_simulations, coverage_level, coverage_weight = 5):




    weights = np.array([0.4,0.6])#np.linspace(0, 1, 21)
    alphas = np.array([100,1000])#np.logspace(1, 3, 5)
    df_ts = np.array([5,0])#np.array([2.5,3,5,10,20,0]) # 0 --> normal!
    vs = np.array([0.25,0.5])#np.linspace(0, 2, 11)




    best_params = {}

    elections = ['2016','2019','2022']

    Prior_estimates_dict_per_election = {}
    Results_dict_per_election = {}

    for election_year in elections:

        # get these only once

        Prior_estimates_dict_per_election[election_year] = get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # single row df for each div_nm
        Results_dict_per_election[election_year] = get_results_df(election_year, to_Fundamentals=False)[1]



    for heldout in elections:
        # Get the other two as training
        train_elections = [e for e in elections if e != heldout]
        
        results = []


        for df_t in df_ts:

            for v in vs:

                Polling_simulations_dict = {}
                Election_swing_simulations_dict = {}


                for election_year in train_elections:
                    Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_simulations, election_year, df_t=df_t, v=v)

                    Polling_simulations_dict[election_year] = Simulated_Electorate_Polling_Results
                    Election_swing_simulations_dict[election_year] = Simulated_Electorate_Swing_Results

                    print("Done 20000 simulation processing:", time.time() - start_time, "seconds")

                
                for w in weights:
                    n_polling_samples = int(w * n_simulations)

                    indices_poll = np.random.choice(n_simulations, n_polling_samples, replace=False)
                    indices_fund = np.random.choice(n_simulations, n_simulations - n_polling_samples, replace=False)

                    for alpha in alphas:
                        print("weight = ", w, "alpha = ", alpha, "df_t =", df_t, "v = ", v, "heldout = ", heldout)

                        val_scores = []
                        for election in train_elections: # choose from both elections

                            combined_samples = np.concatenate((Polling_simulations_dict[election_year][indices_poll],  Election_swing_simulations_dict[election_year][indices_fund]), axis=0) 
                            
                            
                            mae, coverage = get_election_MAE(combined_samples, Prior_estimates_dict_per_election[election_year], Results_dict_per_election[election_year], election_year, coverage_level, alpha)
                            
                            penalty = max(0, coverage_level - coverage)
                            val_score = mae + coverage_weight * penalty

                            val_scores.append(val_score)

                        avg_score = np.mean(val_scores)

                        results.append(((df_t, v, w, alpha), avg_score))
                            

                
        best_w_alpha, best_val_score = min(results, key=lambda x: x[1])

        # Save all best parameters
        best_params[heldout] = {
            'w': best_w_alpha[2],
            'alpha': best_w_alpha[3],
            'df_t': best_w_alpha[0],
            'v': best_w_alpha[1],
            'val_score': best_val_score
        }

        print("Done estimation:", time.time() - start_time, "seconds")

        # Now evaluate on heldout set using best parameters
        best_df_t, best_v, best_w, best_alpha = best_w_alpha

        Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_simulations, heldout, df_t=best_df_t, v=best_v)
        
        # Now evaluate on heldout
        n_polling_samples = int(best_w*n_simulations)
        indices_poll = np.random.choice(n_simulations, n_polling_samples, replace=False)
        indices_fund = np.random.choice(n_simulations, n_simulations - n_polling_samples, replace=False)
        combined_samples = np.concatenate((Simulated_Electorate_Polling_Results[indices_poll],  Simulated_Electorate_Swing_Results[indices_fund]), axis=0)

        heldout_mae, heldout_coverage = get_election_MAE(combined_samples, Prior_estimates_dict_per_election[heldout], Results_dict_per_election[heldout], heldout, coverage_level, alpha)
        best_params[heldout]['test_mae'] = heldout_mae
        best_params[heldout]['test_coverage'] = heldout_coverage

    return best_params





def worker(job):

    coverage_weight = 5
    
    weights = np.linspace(0, 1, 21) # np.array([0.4,0.6])# np.linspace(0, 1, 21) # 
    alphas = np.linspace(10,60, 6) # np.array([100,1000])# 
   

    elections = ['2016','2019','2022']
    heldout, s, v, beta = job
    train_elections = [e for e in elections if e != heldout]

    Polling_simulations_dict = {}
    Election_swing_simulations_dict = {}

    Prior_estimates_dict_per_election = {}
    Results_dict_per_election = {}

    for election_year in train_elections:
        polling, swing = simulate_Polling_Fundamentals_model(n_simulations, election_year, s = s, v=v, beta=beta)
        Polling_simulations_dict[election_year] = polling
        Election_swing_simulations_dict[election_year] = swing

        Prior_estimates_dict_per_election[election_year] = get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # single row df for each div_nm
        Results_dict_per_election[election_year] = get_results_df(election_year, to_Fundamentals=False)[1]

    results = []

    for w in weights:
        n_polling_samples = int(w * n_simulations)
        indices_poll = np.random.choice(n_simulations, n_polling_samples, replace=False)
        indices_fund = np.random.choice(n_simulations, n_simulations - n_polling_samples, replace=False)

        for alpha in alphas:
            val_scores = []
            #print("weight = ", w, "alpha = ", alpha, "s =", s, "v = ", v, "heldout = ", heldout)
            for election_year in train_elections:
                combined_samples = np.concatenate(
                    (
                        Polling_simulations_dict[election_year][indices_poll],
                        Election_swing_simulations_dict[election_year][indices_fund]
                    ),
                    axis=0
                )
                mae, coverage = mae, coverage = get_election_MAE(combined_samples, Prior_estimates_dict_per_election[election_year], Results_dict_per_election[election_year], election_year, coverage_level, alpha)
                            
                penalty = max(0, coverage_level - coverage)
                val_score = mae + coverage_weight * penalty

                val_scores.append(val_score)

            avg_score = np.mean(val_scores)

            results.append((s, v, beta, w, alpha, avg_score))

    return results


def parallelise_validation_simulation(coverage_level):
    elections = ['2016','2019','2022']
    s_s = np.linspace(0.5,1,11) # np.array([5,0])#  0 --> normal! 
    vs = np.linspace(0.05, 0.3, 6) # np.array([0.25,0.5])# 
    betas = np.linspace(0.2,1,5)

    jobs = list(product(elections, s_s, vs, betas))

    with ProcessPoolExecutor(max_workers=24) as executor:
        results_nested = list(tqdm(executor.map(worker, jobs), total=len(jobs), desc="Processing jobs"))

    NUM_PARAMS = 5

    # Flatten the nested list of results into a single list
    results_flat = []

    for i, results in enumerate(results_nested):
        heldout = jobs[i][0]  # Get the heldout value from jobs (index i)
        if not isinstance(results, list):
            print(f"[ERROR] Unexpected type at job {i}: {type(results)} - {results}")
            continue

        for result in results:
            if len(result) != NUM_PARAMS + 1:
                print(f"[ERROR] Malformed result at job {i}: {result}")
                continue
            s, v, beta, w, alpha, avg_score = result
            results_flat.append((heldout, s, v, beta, w, alpha, avg_score))

    # Create DataFrame
    df_results = pd.DataFrame(results_flat, columns=['heldout', 's', 'v', 'beta','w', 'alpha', 'avg_score'])

    best_params = {}

    for heldout in df_results['heldout'].unique():
        df_heldout = df_results[df_results['heldout'] == heldout]
        
        # Get the row with minimum validation MAE
        best_row = df_heldout.loc[df_heldout['avg_score'].idxmin()]

        # Extract best params
        best_s = best_row['s']
        best_v = best_row['v']
        best_beta = best_row['beta']
        best_w = best_row['w']
        best_alpha = best_row['alpha']
        best_val_score = best_row['avg_score']

        best_params[heldout] = {
            'w': best_w,
            'alpha': best_alpha,
            's': best_s,
            'v': best_v ,
            'beta': best_beta ,
            'val_score': best_val_score
        }

        Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_simulations, heldout, s=best_s, v=best_v, beta=best_beta)

        Prior_estimates_dict = get_Prior_estimates_df(heldout, dont_add_ON = True)[1] # single row df for each div_nm
        Results_dict = get_results_df(heldout, to_Fundamentals=False)[1]

        
        # Now evaluate on heldout
        n_polling_samples = int(best_w*n_simulations)
        indices_poll = np.random.choice(n_simulations, n_polling_samples, replace=False)
        indices_fund = np.random.choice(n_simulations, n_simulations - n_polling_samples, replace=False)
        combined_samples = np.concatenate((Simulated_Electorate_Polling_Results[indices_poll],  Simulated_Electorate_Swing_Results[indices_fund]), axis=0)

        heldout_mae, heldout_coverage = get_election_MAE(combined_samples, Prior_estimates_dict, Results_dict, heldout, coverage_level, best_alpha)
        best_params[heldout]['test_mae'] = heldout_mae
        best_params[heldout]['test_coverage'] = heldout_coverage

    import pdb;pdb.set_trace()


    return best_params









def get_topn_sets_per_electorate(data_dict, n = 2, threshold=0.15):
    result = {}

    for electorate, arr in data_dict.items():
        # arr shape: (n_simulations, n_parties)
        topn_indices = np.argsort(arr, axis=1)[:, -n:]  # shape: (n_simulations, 3)
        topn_values = np.take_along_axis(arr, topn_indices, axis=1)

        unique_sets = set()
        for i in range(arr.shape[0]):
            if topn_values[i, 0] > threshold:
                party_tuple = tuple(sorted(topn_indices[i]))  # consistent order
                unique_sets.add(party_tuple)

        result[electorate] = list(unique_sets)

    return result

def index_tuples_to_candidate_names(topn_by_div, colnames_by_div):
    result = {}

    for div, topn_tuples in topn_by_div.items():
        candidate_names = colnames_by_div[div].columns.tolist()
        name_lists = [
            [candidate_names[i] for i in tup]  # convert indices to names
            for tup in topn_tuples
        ]
        result[div] = name_lists

    return result



def First_Preference_Model_Simulation(election_year, w, alpha, v, s, beta, n_simulations = 1000):


    # Simulate votes for ALP, COAL, GRN, ON, TOP/UAPP, OTH (No ON/TOP/UAPP in 2016)
    Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_simulations, election_year, v = v, s=s, beta=beta)

    #import pdb;pdb.set_trace()
    print(f"Simulated {n_simulations*2} MVNs in", time.time() - start_time, "seconds")

    Prior_estimates_dict = get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # single row df for each div_nm


    # Get dictionary results as reference for the candidate order
    if election_year != '2025':
        Results_dict = get_results_df(election_year, to_Fundamentals=False)[1]
    
    else:

        Candidates_2025 = pd.read_csv("2025Candidates_By_Division.csv", index_col = None)

        # Make a faux-Results_dict
        Results_dict = {}


        target = 'IND'
        for div, sub_df in Candidates_2025.groupby('div_nm', sort=False):

            div_parties = sub_df
            div_parties.loc[:,'Count'] = div_parties.groupby('PartyAb').cumcount() + 1     # Count instances of the target string

            # To distinguish 'IND', Replace duplicates of 'IND' with increasing strings IND1, IND2, IND3, ...
            adjusted_party_names = div_parties.apply(
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            ).reset_index(drop=True)

            # separate independents
            div_parties.loc[div_parties['div_nm'] == div,'PartyAb'] = adjusted_party_names.values
            div_results_combined = div_parties.drop('Count', axis = 1)

            ordered_parties = div_results_combined['PartyAb'].drop_duplicates()

            # convert to wide format to match existing Results_dict format
            pivoted = div_results_combined.pivot(index='div_nm', columns='PartyAb')
            Results_dict[div] = pivoted.reindex(columns = ordered_parties)

    # randomly mix Polling Simulations and Fundamentals simulations wiht weight w
    indices = np.random.permutation(n_simulations)

    Weighted_Simulations = np.concatenate((Simulated_Electorate_Polling_Results[indices[:int(w*n_simulations)]],  Simulated_Electorate_Swing_Results[indices[int(w*n_simulations):]]), axis=0)

    # Expand to full candidate size, expanding 'COAL', 'OTH' and the 'IND' categories using Dirichlet distribution with parameter alpha
    final_simulated_votes = expand_all_divisions_from_prior_df(Weighted_Simulations, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=alpha)[0]

    return final_simulated_votes, Results_dict

Method = None # 'Validation' 'Simulation'
coverage_level = 0.95


Day = 90

if Day == 80:
    w = 0.85
    alpha = 20
    v = 0.2
    s = 0.75
    beta = 0.6

elif Day == 90:
    FP_validation_results_Day_90 = {'2016': {'w': 0.8500000000000001, 'alpha': 20.0, 's': 0.7, 'v': 0.05, 'beta': 0.8, 'val_score': 3.1949988090775223, 'test_mae': 3.476870499000724, 'test_coverage': 0.9295774647887324}, '2019': {'w': 0.9, 'alpha': 20.0, 's': 0.65, 'v': 0.2, 'beta': 0.6000000000000001, 'val_score': 3.3244865269228403, 'test_mae': 3.228332417566434, 'test_coverage': 0.9384469696969697}, '2022': {'w': 0.9500000000000001, 'alpha': 30.0, 's': 0.9, 'v': 0.05, 'beta': 0.4, 'val_score': 3.3928479934061113, 'test_mae': 3.041729217178223, 'test_coverage': 0.9118869492934331}}
    # Another: close results! {'2016': {'w': 0.8500000000000001, 'alpha': 20.0, 's': 0.75, 'v': 0.1, 'beta': 0.8, 'val_score': 3.1886505715847515, 'test_mae': 3.4538902844200066, 'test_coverage': 0.9275653923541247}, '2019': {'w': 0.9, 'alpha': 20.0, 's': 0.75, 'v': 0.1, 'beta': 0.6000000000000001, 'val_score': 3.3291967680158114, 'test_mae': 3.2397858737616434, 'test_coverage': 0.9356060606060606}, '2022': {'w': 0.9, 'alpha': 20.0, 's': 0.8, 'v': 0.05, 'beta': 0.4, 'val_score': 3.3894457195184042, 'test_mae': 3.087975018296332, 'test_coverage': 0.9251870324189526}}
    w = 0.9
    alpha = 22
    s = 0.75
    beta = 0.5
    v = 0.15
    
if Method == 'Validation':
    validation = {}
    #for coverage_weight in [2,3,5,10]:
    best_params = parallelise_validation_simulation(coverage_level)
    #validation[coverage_weight] = best_params
    print(best_params)
    print(time.time() - start_time)
    import pdb;pdb.set_trace()

elif Method =='Simulation':

    election_year = '2025'
        
    if election_year == '2022':
        1
        #w, alpha, s, v, beta =0.95,30,0.9,0.5,0.4
    elif election_year == '2019':
        w, alpha, s, v, beta = 0.9,20,0.65,0.2,0.6

    elif election_year == '2016':
        w, alpha, s, v, beta = 0.85,20,0.7,0.05,0.8
    final_simulated_votes, Results_dict =  First_Preference_Model_Simulation(election_year = election_year, w = w, alpha = alpha, v = v, s = s, beta = beta, n_simulations = n_simulations)

    



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

party_category_dict = make_party_category_dict()

party_to_category_centered_IND = {
    k: ('IND' if v == 'Centre' else v)
    for k, v in party_category_dict.items()
}

TCP_combination_index = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6,  \
                            ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14}





def make_TCP_pair_category_dict(election_year):

    from collections import defaultdict


    data_year = str(int(election_year) - 3)
    

    name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
    new_seats_year_dict = {'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner','Gorton']}

    replacement_seats_year_dict = {'2022': {'Hasluck':'Bullwinkel'}, '2019':{'Gorton':'Hawke'}, '2016':{'Canberra':'Bean', 'Maribyrnong':'Fraser'}, '2013':{'Hasluck':'Burt'}}
    abolished_divs_dict = {'2022':set(['Higgins','North Sydney']), '2016': set(['Port Adelaide']),'2019':set(['Stirling']),'2013':set(['Charlton'])}






    TCP_combination_index = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6,  \
                            ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13,('COAL','COAL'):14}

    data_years = [str(int(year) - 3) for year in election_years]


    next_year = election_year
    # 1. Get names of next election's parties in each div for comparison to senate
    if next_year != '2025':

        DOP_By_Division_next = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'})[["div_nm","PartyAb"]].drop_duplicates()

    else:
        DOP_By_Division_next = pd.read_csv("2025Candidates_By_Division.csv", index_col = None)
    
    DOP_By_Division_next.loc[:,'PartyAb'] = DOP_By_Division_next.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
    Div_parties_next_dict = {div: group['PartyAb'].tolist() for div, group in DOP_By_Division_next.groupby("div_nm")}

    Div_parties_next_dict_COAL = {div: ['COAL' if p in ['LP', 'NP','CLP','LNP'] else p for p in Div_parties_next_dict[div]] for div in Div_parties_next_dict.keys()}


    TCP_Preference_Flows = pd.read_csv(f"{data_year}HouseTCPFlowByDivision.csv", skiprows = 1, index_col = None).rename(columns = {'DivisionNm':'div_nm','FromCandidatePartyAb':'PartyAb', \
                                                'FromCandidateBallotPosition':'Ballot_Position','ToCandidatePartyAb':'TCP_Ab','ToCandidateBallotPosition':'TCP_Ballot_Position'})
    TCP_Preference_Flows = TCP_Preference_Flows[['div_nm','PartyAb','Ballot_Position','TCP_Ab','TCP_Ballot_Position','TransferPercentage']]
    TCP_Preference_Flows = TCP_Preference_Flows.loc[TCP_Preference_Flows['Ballot_Position']>0,]
    TCP_Preference_Flows = TCP_Preference_Flows[['div_nm','PartyAb','TCP_Ab','TransferPercentage']]
    TCP_Preference_Flows['div_nm'] = TCP_Preference_Flows['div_nm'].replace(name_changes_year_dict[data_year])

    TPP = ('ALP', 'COAL')


    TPP_by_State = pd.read_csv(f"{data_year}HouseTPPFlowByStateByParty.csv", skiprows = 1, index_col = None)
    TPP_by_State = TPP_by_State[['StateAb', 'PartyAb', 'Australian Labor Party Transfer Percentage']]
    TPP_by_State = TPP_by_State.loc[(TPP_by_State['PartyAb'].notna()) & (TPP_by_State['PartyAb'] != 'NAFD'), ].rename(columns={'Australian Labor Party Transfer Percentage':'ALP%'})

    TPP_nationally = pd.read_csv(f"{data_year}HouseTPPFlowByParty.csv", skiprows = 1, index_col = None)[['PartyAb', 'Australian Labor Party Transfer Percentage']].rename(columns={'Australian Labor Party Transfer Percentage':'ALP%'})
    
    if next_year != '2025':
        div_to_state = pd.read_csv(f"{next_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})

    else:
        div_to_state = pd.read_csv(f"2022HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        div_to_state.loc[div_to_state['div_nm'] == 'North Sydney',] = 'Bullwinkel', 'WA'
        div_to_state = div_to_state.loc[~(div_to_state['div_nm'] == 'Higgins'),]

    div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}




    # 1. Get PartyAb: 1st alphabetically for each pair:

    tcp_pairs = (TCP_Preference_Flows.groupby("div_nm")["TCP_Ab"].unique().apply(lambda x: tuple(sorted(set(x)))))  # optional: sort so (ALP, LP) and (LP, ALP) match)

    tcp_coalified = tcp_pairs.apply(lambda tup: tuple('COAL' if x in ['LP', 'NP','LNP','CLP'] else x for x in tup))






    # add TCP to dict!

    Preference_flows_dict = {}
    Non_classic_divs = defaultdict(list)


    def normalize_party(p):
        return 'COAL' if p in ['LP', 'NP'] else p

    def sorted_tcp_pair(tcp1, tcp2):
        return tuple(sorted([normalize_party(tcp1), normalize_party(tcp2)]))


    # Step 1: Precompute the sorted and COALified 2CP pair per division
    tcp_pairs_by_div = (
        TCP_Preference_Flows.groupby("div_nm")["TCP_Ab"]
        .unique()
        .apply(lambda x: sorted_tcp_pair(*x))
    )

    # Step 2: Build the desired structure
    for _, row in TCP_Preference_Flows.iterrows():
        div = row['div_nm']
        party = row['PartyAb']
        tcp = normalize_party(row['TCP_Ab'])
        pct = row['TransferPercentage']

        if div in abolished_divs_dict[data_year]:
            continue
        
        tcp_pair = tuple(sorted([p if p not in ['LP','NP','LNP','CLP'] else 'COAL' for p in tcp_pairs_by_div[div]]))

        if tcp_pair != ('ALP', 'COAL'):
            # we have a Non-classic contest!

            tcp_pair = tuple(sorted([party_category_dict[p] if party_category_dict[p] != 'Centre' else 'IND' for p in tcp_pair]))

            if tcp_pair == ('ALP', 'COAL'):
                import pdb;pdb.set_trace() # should not happen

            if div not in Non_classic_divs[tcp_pair]:
                Non_classic_divs[tcp_pair].append(div)



        first, second = tcp_pair  # alphabetical order
        #if div == 'Brisbane':
        #    import pdb;pdb.set_trace() 
        # We want % transferred to the *first* in alphabetical order
        if party_category_dict[tcp] == first:
            transfer_pct = pct
        else:
            transfer_pct = 100 - pct

        Preference_flows_dict.setdefault(div, {})

        party = party if party not in ['LP','NP','LNP','CLP'] else 'COAL'

        if party in Div_parties_next_dict_COAL[div]:
            Preference_flows_dict[div][party] = {tcp_pair: transfer_pct}

        if div in replacement_seats_year_dict[data_year].keys():
            new_div = replacement_seats_year_dict[data_year][div]
            Preference_flows_dict.setdefault(new_div, {})
            if party in Div_parties_next_dict_COAL[new_div]:
                Preference_flows_dict[new_div][party] = {tcp_pair: transfer_pct}



    #import pdb;pdb.set_trace()

    from collections import defaultdict

    # Assume `result` and `party_categories` are already defined

    # Make a new version of result with category rollups
    Preference_flows_dict_with_categories = {}

    for div, party_dict in Preference_flows_dict.items():
        category_values = defaultdict(list)
        div_result = dict(party_dict)  # copy original party entries

        # Get the TCP pair from any of the entries (they all have the same)
        tcp_pair = next(iter(next(iter(party_dict.values())).keys()))

        for party, tcp_data in party_dict.items():
            category = party_category_dict.get(party)
            if category in ['Left', 'Right', 'Centre']:
                pct = tcp_data[tcp_pair]
                category_values[category].append(pct)

        

        # Compute category averages and add them to the division result
        for category, values in category_values.items():
            avg_pct = sum(values) / len(values)
            div_result[category] = {tcp_pair: avg_pct}

        Preference_flows_dict_with_categories[div] = div_result

    #import pdb;pdb.set_trace()


    ######### Now 2PP values if missing - extend to all categories!

    # STEP 1 — Fill in missing PARTY entries by div
    for div, state in div_to_state_dict.items():
        if div not in Preference_flows_dict_with_categories: # should be redundant!
            Preference_flows_dict_with_categories[div] = {}
        
        for _, row in TPP_by_State[TPP_by_State['StateAb'] == state].iterrows():
            party = row['PartyAb']
            percent_to_alp = row['ALP%']

            if party in ['LP','NP']:
                continue
            
            # Add the party only if not already present
            if party in Div_parties_next_dict_COAL[div]:
                if party not in Preference_flows_dict_with_categories[div]:
                    Preference_flows_dict_with_categories[div][party] = {}
                #import pdb;pdb.set_trace()

                if ('ALP', 'COAL') not in Preference_flows_dict_with_categories[div][party]:
                    Preference_flows_dict_with_categories[div][party][('ALP', 'COAL')] = percent_to_alp





    state_category_values = defaultdict(lambda: defaultdict(list))

    # STEP 2 — Fill in missing PARTY entries by div
    for _, row in TPP_by_State.iterrows():
        state = row['StateAb']
        party = row['PartyAb']
        percent_to_alp = row['ALP%']
        
        category = party_category_dict.get(party)
        if category and category not in ['ALP', 'COAL']:  # Skip major parties
            state_category_values[state][category].append(percent_to_alp)

    # STEP 3 — Fill in missing category averages per div
    for div, state in div_to_state_dict.items():
        #if div not in Preference_flows_dict_with_categories:
        #    continue

        subdict = Preference_flows_dict_with_categories[div]

        for category in ['Left', 'Right', 'Centre']:
            if category not in subdict:
                values = state_category_values[state].get(category)
                if values:
                    median_value = np.median(values)
                    subdict.setdefault(category, {})[('ALP', 'COAL')] = median_value
            else:
                if ('ALP','COAL') not in subdict[category]:
                    values = state_category_values[state].get(category)
                    if values:
                        median_value = np.median(values)

                        subdict[category][('ALP','COAL')] = median_value

    for div in Preference_flows_dict_with_categories.keys():
        for _, row in TPP_nationally.iterrows():

            party = row['PartyAb']
            percent_to_alp = row['ALP%']

            if party in ['LP','NP']:
                    continue
                
            # Add the party only if not already present
            if party in Div_parties_next_dict_COAL[div]:
                if party not in Preference_flows_dict_with_categories[div]:
                    Preference_flows_dict_with_categories[div][party] = {}
                #import pdb;pdb.set_trace()

                if ('ALP', 'COAL') not in Preference_flows_dict_with_categories[div][party]:
                    Preference_flows_dict_with_categories[div][party][('ALP', 'COAL')] = percent_to_alp

                


    #import pdb;pdb.set_trace()
    # reformat into dict of dicts of dfs:


    rows = []

    for division, party_dict in Preference_flows_dict_with_categories.items():
        for party, tcp_dict in party_dict.items():
            for tcp_pair, percent in tcp_dict.items():
                rows.append({
                    'division': division,
                    'tcp_pair': tcp_pair,
                    'party': party,
                    'percent': percent
                })



    # Step 2: Convert to DataFrame and pivot
    long_df = pd.DataFrame(rows)
    #import pdb;pdb.set_trace()

    # Step 3: Pivot to wide format: one row per division, one column per party
    # Step 2: Create the dictionary of wide DataFrames
    result_dict = {}

    # Iterate over the unique divisions
    for div, group in long_df.groupby('division'):
        div_result = pd.DataFrame(index=range(len(TCP_combination_index)), columns=Preference_flows_dict_with_categories[div].keys())
        
        # Fill the DataFrame with NaN (or 0 if preferred) for all TCP pairs initially
        div_result[:] = None  # Or use np.nan if you prefer NaN

        #if division == 'Melbourne':
        # import pdb;pdb.set_trace()


        # Iterate over each unique tcp_pair for this division
        for tcp_pair, tcp_group in group.groupby('tcp_pair'):

            if tcp_pair not in TCP_combination_index.keys():
                # convert to party category:
                
                tcp_pair = tuple(sorted([party_category_dict[p] if p != 'Centre' else 'IND' for p in tcp_pair]))



            # Map tcp_pair to the corresponding row index
            row_index = TCP_combination_index.get(tcp_pair, None)
            
            # If the index is not found in mapping, skip (shouldn't happen if mapping is correct)
            if row_index is None:
                #import pdb;pdb.set_trace()
                continue
            
            # Pivot the tcp_group into a wide format (party as columns, percent as values)
            for _, row in tcp_group.iterrows():
                div_result.at[row_index, row['party']] = row['percent']
        
        # Store the result for this division in the final dictionary
        result_dict[div] = div_result
        if 'ALP' not in result_dict[div]:
            result_dict[div].loc[:,'ALP'] = None
        if 'COAL' not in result_dict[div]:
            result_dict[div].loc[:,'COAL'] = None

        #if div == 'Macnamara':
        #    import pdb;pdb.set_trace()



    # Outlier - Canberra 2013 House had no Right parties!
    if data_year == '2016':
        for div in ['Bean','Canberra','Fenner']:
            result_dict[div].loc[:,'Right'] = None
            result_dict[div].loc[0,'Right'] = result_dict['Bean'].loc[0,'LDP']

        #import pdb;pdb.set_trace()


    #import pdb;pdb.set_trace()

    # now, find non-classic divisions and extrapolate

    for tcp_pair in Non_classic_divs.keys():

        if tcp_pair == ('COAL','COAL'):
            continue

        # first get average of all results with said tcp:
        i = TCP_combination_index[tcp_pair]

        series_list = []

        for div in Non_classic_divs[tcp_pair]:
            
            series_list.append(result_dict[div].iloc[i]) # curr reusults
        
        tcp_overall = pd.concat(series_list, axis=1)
        tcp_average = tcp_overall.mean(axis=1).dropna()




        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                filtered_update = tcp_average[result_dict[div].columns.intersection(tcp_average.index)]
                result_dict[div].loc[i, filtered_update.index] = filtered_update


        #if tcp_pair == ('COAL','Left'):
        #    import pdb;pdb.set_trace()

    for tcp_pair in [('IND','IND'),('Left','Left'),('LP','NP'),('Right','Right'),('COAL','COAL')]:

        i = TCP_combination_index[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                result_dict[div].loc[i, :] = 50.0

    for tcp_pair in [('IND','Right'),('IND','Left')]:

        # use IND-ALP/COAL i.e. 2 or 1

        i = TCP_combination_index[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                # symmetrically apply!

                source_row = result_dict[div].loc[i - 8, :].copy()

                # Invert each value ONLY if it's not None
                new_row = source_row.apply(lambda x: 100 - x if x is not None else None).astype('object')
                new_row = new_row.where(new_row.notna(), None)
                #new_row = 100 - result_dict[div].loc[i-8, :].copy() # correct indexing!
                if tcp_pair == ('IND', 'Right'):

                    if 'Right' not in new_row.index:
                        #print(div)
                        import pdb;pdb.set_trace()

                    new_row['COAL'] = new_row['Right'] 
                    
                elif tcp_pair == ('IND', 'Left'):
                    new_row['ALP'] = new_row['Left'] 

                # new_row.index.map(lambda x: 'ALP' if x == 'Left' else x)

                #result_dict[div].iloc[i,:] = new_row

                for col in result_dict[div].columns:
                    if new_row[col] is not None:  # Only overwrite if the new value isn't None
                        result_dict[div].at[i, col] = new_row[col]

            #import pdb;pdb.set_trace()
            #4


    for tcp_pair in [('Left','Right')]:
        #import pdb;pdb.set_trace()

        # use ALP/COAL

        i = TCP_combination_index[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                new_row = result_dict[div].loc[0, :].copy()

                # ALP gets Left's 
                # COAL gets Right's 
                new_row.loc['ALP'] = new_row['Left']
                new_row.loc['COAL'] = new_row['Right']

                result_dict[div].loc[i, :] = new_row

    for tcp_pair in [('ALP','Right'),('COAL','Left')]:
        #import pdb;pdb.set_trace()

        # use ALP/COAL

        i = TCP_combination_index[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():

                # ALP gets Left's 
                # COAL gets Right's 

                if tcp_pair == ('ALP','Right'):

                    new_row = result_dict[div].loc[0, :].copy()

                    new_row.loc['COAL'] = new_row['Right']

                    result_dict[div].loc[i, :] = new_row


                elif tcp_pair == ('COAL','Left'):

                    source_row = result_dict[div].loc[0, :].copy()
                    # switch order
                    new_row = source_row.apply(lambda x: 100 - x if x is not None else None).astype('object')
                    new_row = new_row.where(new_row.notna(), None)

                    new_row.loc['ALP'] = new_row['Left']              

                    for col in result_dict[div].columns:
                        if new_row[col] is not None:  # Only overwrite if the new value isn't None
                            result_dict[div].at[i, col] = new_row[col]      

                    #import pdb;pdb.set_trace()          


    #import pdb;pdb.set_trace()

    # If RIght is None, get corresponding COAL, smae for left and ALP
    Right_COAL_Right_Preferences = 100 - 58.39 # ON in 2016/9 Maranoa
    Left_COAL_Left_Preferences = 30.5 # 2016/19 Kooyong/Higgins/Melbourne
    Left_ALP_Left_Preferences = 34.0 # 2016/19 ALP/GRN Seats
    Right_COAL_IND_Preferences = 39.8 # for XEN: 2013 Indi/NE preferences
    Right_COAL_Left_Preferences = 40.0 # for 2016, from 2010 Grayndler/Batman
    Right_COAL_COAL_Preferences = 50.0


    for div in result_dict.keys():
        #print(div)
        div = div
        for i, row in result_dict[div].iterrows():
            if pd.isna(row['Right']):
                #import pdb;pdb.set_trace()
                result_dict[div].at[i, 'Right'] = row['COAL']
            if pd.isna(row['Left']):
                #import pdb;pdb.set_trace()
                result_dict[div].at[i, 'Left'] = row['ALP']




            # Fix no Right for RIght-COAL contests:
            if (data_year == '2022') and (i == 6) and (pd.isna(result_dict[div].loc[6,'Right'])):
                result_dict[div] = result_dict[div].copy()
                result_dict[div].at[6,'Right'] = Right_COAL_Right_Preferences

            
            if (data_year == '2022') and (i == 5) and (pd.isna(result_dict[div].loc[5,'Left'])):
                result_dict[div].at[5, 'Left'] = Left_COAL_Left_Preferences

            if (data_year == '2022') and (i == 3) and (pd.isna(result_dict[div].loc[3,'Left'])):
                result_dict[div].at[3, 'Left'] = Left_ALP_Left_Preferences

            if (data_year == '2019') and (i == 3) and (pd.isna(result_dict[div].loc[3,'Left'])):
                result_dict[div].at[3, 'Left'] = Left_ALP_Left_Preferences

            if (data_year == '2016') and (i == 1) and (pd.isna(result_dict[div].loc[1,'Right'])):
                result_dict[div].at[1, 'Right'] = Right_COAL_IND_Preferences

            if (data_year == '2016') and (i == 9) and (pd.isna(result_dict[div].loc[9,'Right'])):
                result_dict[div].at[9, 'Right'] = 1 - Right_COAL_IND_Preferences

            if (data_year == '2016') and (i == 5) and (pd.isna(result_dict[div].loc[5,'Right'])):
                result_dict[div].at[5, 'Right'] = Right_COAL_Left_Preferences

            if (data_year == '2016') and (i == 6) and (pd.isna(result_dict[div].loc[6,'Right'])):
                result_dict[div].at[6, 'Right'] = Right_COAL_Right_Preferences # cheating from future - but quick fix!

            if (data_year == '2016') and (i == 14) and (pd.isna(result_dict[div].loc[14,'Right'])):
                result_dict[div].at[14, 'Right'] = Right_COAL_COAL_Preferences


            # fetch all changes so far!
            row = result_dict[div].loc[i]



            if pd.isna(row['Centre']):
                #import pdb;pdb.set_trace()
                #import pdb;pdb.set_trace()
                if (row['Right']) and (row['Left']):
                    

                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['Left'])/2

                    if pd.isna(result_dict[div].at[i, 'Centre']):
                        import pdb;pdb.set_trace()
                        2
                    
                elif (row['Right']) and (row['ALP']):
                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['ALP'])/2

                elif (row['Left']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['Left'])/2

                elif (row['ALP']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['ALP'])/2

                if pd.isna(result_dict[div].at[i, 'Centre']):
                    print(div, i, row)
                    import pdb;pdb.set_trace()
                    3

                    






            

    for division in result_dict.keys():
        #print(division)
        if result_dict[division]['Centre'].isna().sum():
            import pdb;pdb.set_trace()
            1
        if result_dict[division]['Right'].isna().sum():
            import pdb;pdb.set_trace()
            6
        if result_dict[division]['Left'].isna().sum():
            import pdb;pdb.set_trace()
            5


            

    #import pdb;pdb.set_trace()


    # find COALITION_double_divs_last_year
    DOP_By_Division_curr = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'})[["div_nm","PartyAb"]].drop_duplicates()    
    DOP_By_Division_curr.loc[:,'PartyAb'] = DOP_By_Division_curr.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
    DOP_By_Division_curr = {div: group['PartyAb'].tolist() for div, group in DOP_By_Division_curr.groupby("div_nm")}

    COAL_double_divs_curr = []
    for div in DOP_By_Division_curr.keys():
        if ('LP' in DOP_By_Division_curr[div]) and ('NP' in DOP_By_Division_curr[div]):
            COAL_double_divs_curr.append(div)


    COAL_double_div_transfers = TCP_Preference_Flows.loc[((TCP_Preference_Flows['PartyAb']=='NP') & (TCP_Preference_Flows['TCP_Ab']=='LP')) | ((TCP_Preference_Flows['PartyAb']=='LP') & (TCP_Preference_Flows['TCP_Ab']=='NP')),].rename(columns = {'TransferPercentage':'COAL%'})[['div_nm','COAL%']]
    COAL_double_div_transfers = COAL_double_div_transfers.set_index('div_nm')
    COAL_double_div_transfers.loc['Average'] = COAL_double_div_transfers['COAL%'].mean()
    #import pdb;pdb.set_trace()

    COAL_double_divs_next = []
    for div in Div_parties_next_dict.keys():
        if ('LP' in Div_parties_next_dict[div]) and ('NP' in Div_parties_next_dict[div]):
            COAL_double_divs_next.append(div)



    COAL_list = ['LP','NP','CLP','LNP']


    for div in result_dict.keys():

        # convert all None to nan
        result_dict[div] = result_dict[div].where(pd.notna(result_dict[div] ), None)  # no-op if already NaN
        result_dict[div]  = result_dict[div] .astype(float)


        for i, party in enumerate(Div_parties_next_dict_COAL[div]):

            # new parties
            if party not in result_dict[div].columns:
                result_dict[div].loc[:,party] =  result_dict[div][party_category_dict[party]] # replace with the category!

            else: # fill in missing vals for old parties
                result_dict[div][party] = result_dict[div][party].combine_first(result_dict[div][party_category_dict[party]])
                #import pdb;pdb.set_trace()

            if party == 'COAL':
                PartyAb = Div_parties_next_dict[div][i]
                result_dict[div].loc[:,PartyAb] = result_dict[div]['COAL']


        # Finally, add COAL -> COAL support if COAL_double_div

        if div in COAL_double_divs_next:

            if div in COAL_double_div_transfers.index:
                COAL_Pct = COAL_double_div_transfers.loc[div].iloc[0]
            else:
                COAL_Pct = COAL_double_div_transfers.loc['Average'].iloc[0]

            for col in ['LP','NP']:
                row_indexer = [0,1,5,6]
                result_dict[div].loc[row_indexer,col] = [100 - COAL_Pct, COAL_Pct, COAL_Pct, COAL_Pct]

        #import pdb;pdb.set_trace()

        # fill in ALP/COAL rows with 0


            

            

        # Remove Centre, LeftLRight, "COAL"
        result_dict[div] = result_dict[div].drop(['Left','Centre','Right','COAL'], axis = 1)


    #import pdb;pdb.set_trace()
    cols_to_drop = ['ALP', 'LNP', 'LP', 'NP', 'CLP']

    for div in result_dict.keys():

        df_dropped = result_dict[div].drop(columns=[col for col in cols_to_drop if col in result_dict[div].columns])

        # Now check for NaN values in the remaining columns
        if df_dropped.isna().any().any():
            import pdb;pdb.set_trace()

                

    def expand_and_reorder_duplicate(df, new_names):
        return pd.concat([df[col] for col in new_names], axis=1, keys=new_names)


    

    # get them in Ballot order: 

    # Div_parties_next_dict groups all the INDs together - must use Div_Ballot_Order_next_dict instead to get all INDs in Ballot order
    if next_year != '2025':
        DOP_by_div_full = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'}).rename(columns = {'DivisionNm':'div_nm'}) 
        DOP_by_div_full = DOP_by_div_full.loc[(DOP_by_div_full['CountNumber']==0) & (DOP_by_div_full['CalculationType'] == 'Preference Count'),['div_nm', 'PartyAb']]
        DOP_by_div_full.loc[:,'PartyAb'] = DOP_by_div_full.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
        Div_Ballot_Order_next_dict = DOP_by_div_full.groupby('div_nm')['PartyAb'].apply(list).to_dict()
    else:
        Div_Ballot_Order_next_dict = Div_parties_next_dict

    for div in result_dict.keys():

        result_dict[div] = expand_and_reorder_duplicate(result_dict[div], Div_Ballot_Order_next_dict[div]) 


    #import pdb;pdb.set_trace()

    # adjust preferences of Defecting Independents in 2022 to be more favourable to their original parties (~ like ON)

    if data_year == '2022':
        Defected_INDs_Ballot_order = {'Moore':1, 'Monash':3,'Calare':7}
        for div in ['Monash','Moore','Calare']:
            idx = Defected_INDs_Ballot_order[div] - 1
            result_dict[div].iloc[:,idx] = result_dict[div]['ON']

        # Macnamara - Josh Burns Open Preference in 2025
        result_dict['Macnamara'].loc[[5],'ALP'] += 6.6

        #import pdb;pdb.set_trace()


    TCP_pair_category_dict = result_dict



    return TCP_pair_category_dict




def add_randomness_to_proportions(proportions_transferred_to_first, sigma_joint, sigma_ind):


    global_noise = np.random.normal(0, sigma_joint, (15,))  # 15 values for the 15 rows

    # Iterate through each division (electorate)
    for div, proportions_df in proportions_transferred_to_first.items():

        proportions_transferred_to_first[div].fillna(0.5, inplace=True)

        # Add global noise (sigma_joint) to the DataFrame across all rows
        for i in range(15):
            proportions_transferred_to_first[div].iloc[i, :] += global_noise[i]  # Add the global noise for each row
        
        # Add independent noise (sigma_ind) - random noise for each element
        independent_noise = np.random.normal(0, sigma_ind, proportions_df.shape)  # Independent noise for each element
        proportions_transferred_to_first[div] += independent_noise  # Add independent noise element-wise


        # Step 4: Replace values below 5 with 5, and values above 95 with 95
        proportions_transferred_to_first[div] = proportions_transferred_to_first[div].clip(lower=5, upper=95)

    return proportions_transferred_to_first


def distribution_to_top_2(final_simulated_votes, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind):

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
            #row_index = TCP_combination_index[tuple(sorted(top2_category))]


            #first_cat = top2_category[0]
            first_idx, second_idx = sorted_cats_with_indices[0][1], sorted_cats_with_indices[1][1]
            #print("first_idx: ", first_idx)
            
            # Step 3: Fetch transfer proportions
            row_index = TCP_combination_index[top2_category]
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

def get_actual_winners(election_year):

    Members_elected = pd.read_csv(f"{election_year}HouseMembersElected.csv", skiprows=1).rename(columns = {'DivisionNm':'div_nm'})[['div_nm', 'PartyAb']]
    Members_elected.loc[Members_elected["PartyAb"] == "GVIC","PartyAb"] = 'GRN'


    # add numbering to 'IND' to match IND Ballot order!
    IND_elected_divs = Members_elected.loc[Members_elected['PartyAb']=='IND','div_nm'].tolist()
    #print(IND_elected_divs)

    First_Prefs_by_div = pd.read_csv(f"{election_year}HouseFirstPrefsByCandidateByVoteType.csv", skiprows=1).rename(columns = {'DivisionNm':'div_nm'})[['div_nm', 'PartyAb','Elected']]
    First_Prefs_by_div = First_Prefs_by_div.loc[First_Prefs_by_div['div_nm'].isin(IND_elected_divs),]

    #import pdb;pdb.set_trace()

    First_Prefs_by_div.loc[:,'PartyAb'] = First_Prefs_by_div['PartyAb'].fillna('IND') 
    # relabel independents in order of ballot appearance if there are multiple
    target = 'IND'
    for div in IND_elected_divs:

        curr_div_FP = First_Prefs_by_div.loc[First_Prefs_by_div['div_nm']==div,].copy()

        curr_div_FP.loc[:,'Count'] = (curr_div_FP.groupby('PartyAb').cumcount() + 1)     # Count instances of the target string

        adjusted_party_names = curr_div_FP.apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
        )
        

        curr_div_FP.loc[:,'PartyAb'] = adjusted_party_names.values


        # add Ballot enumaration to 'IND'
        winner_IND = curr_div_FP.loc[curr_div_FP['Elected']=='Y','PartyAb'] # winner has a Y
        Members_elected.loc[Members_elected['div_nm']==div,'PartyAb'] = winner_IND.iloc[0]


    #import pdb;pdb.set_trace()


    actual_winners_dict = Members_elected.set_index('div_nm')['PartyAb'].to_dict()


    return actual_winners_dict



def true_coverage(per_electorate_winners, actual_winners_dict, n_simulations=1000, coverage_intervals=[50, 70, 90, 95, 99]):
    """
    Computes true coverage: how often the model's favorite wins, at different confidence levels.

    Arguments:
      - per_electorate_winners: dict {electorate: list of simulated winners}
      - actual_winners_dict: dict {electorate: actual winner}
      - n_simulations: number of simulations per electorate
      - coverage_intervals: list of coverage intervals to check (in %)

    Returns:
      - dict {coverage level -> observed success rate}
    """

    # Step 1: Compute favorite party + probability for each electorate
    fav_probs = {}
    fav_parties = {}
    for electorate, sims in per_electorate_winners.items():
        counts = Counter(sims)
        most_common_party, count = counts.most_common(1)[0]
        fav_probs[electorate] = count / n_simulations
        fav_parties[electorate] = most_common_party

    # Step 2: Build data for each electorate: (confidence, correct or not)
    results = []
    for electorate in fav_probs:
        confidence = fav_probs[electorate]
        prediction = fav_parties[electorate]
        actual = actual_winners_dict[electorate]
        correct = (prediction == actual)
        results.append((confidence, correct))

    # Step 3: For each coverage level, compute success rate
    coverage_results = {}
    for coverage in coverage_intervals:
        threshold = coverage / 100
        # Electorates where the favorite probability >= threshold
        filtered = [(conf, corr) for conf, corr in results if conf >= threshold]
        if filtered:
            success_rate = sum(corr for _, corr in filtered) / len(filtered)
        else:
            success_rate = float('nan')  # No electorates with enough confidence
        coverage_results[coverage] = success_rate

    return coverage_results

def objective_function(actual_coverage,
                       coverage_intervals=[50, 70, 90, 95, 99],
                       weights=None):
    """
    actual_coverage: dict {coverage_pct → observed_fraction}
    Returns the sum of squared (observed – nominal) deviations.
    """
    total_penalty = 0.0
    for c in coverage_intervals:
        observed = actual_coverage.get(c, 0.0)
        nominal  = c / 100.0
        w        = weights.get(c, 1.0) if weights else 1.0
        total_penalty += w * (observed - nominal) ** 2
    return total_penalty





def evaluate_grid_point_old(args):
    sigma_ind, sigma_joint, year = args
    
    # --- Load relevant data for the given year ---
    final_simulated_votes = ALL_final_simulated_votes[year]
    proportions_transferred_to_first = ALL_proportions_transferred_to_first[year]
    Results_dict = ALL_results_dict[year]
    actual_winners = ALL_winners_dict[year]

    # --- Simulate winner predictions using vote distributions and transfer proportions ---
    per_electorate_winners, _ = distribution_to_top_2(
        final_simulated_votes,
        proportions_transferred_to_first,
        Results_dict,
        party_to_category_centered_IND,
        sigma_ind,
        sigma_joint
    )

    # --- Compare simulated results with actual winners ---
    coverage_results = true_coverage(per_electorate_winners, actual_winners) # coverage_probability(per_electorate_winners, actual_winners, n_simulations)
    obj_score = objective_function(coverage_results)

    return {
        'sigma_ind': sigma_ind,
        'sigma_joint': sigma_joint,
        'year': year,
        'obj_score': obj_score,
        'coverage_results': coverage_results,  # Optional: remove if not needed
    }

def calibration_curve(per_electorate_winners, actual_winners_dict, n_simulations=1000, bins=[(40,60), (60,70), (70,80), (80,90), (90,95), (95,99), (99,100)]):
    """
    Compute calibration: accuracy inside bins of predicted probability.
    """
    results = []
    for electorate, sims in per_electorate_winners.items():
        counts = Counter(sims)
        most_common_party, count = counts.most_common(1)[0]
        prob = count / n_simulations
        actual = actual_winners_dict[electorate]
        correct = (most_common_party == actual)
        results.append((prob, correct))
    
    calibration = {}
    for low, high in bins:
        selected = [(p, c) for (p, c) in results if low/100 <= p < high/100]
        if selected:
            calibration[(low, high)] = sum(c for _, c in selected) / len(selected)
        else:
            calibration[(low, high)] = float('nan')
    
    return calibration

def calibration_objective(calibration, weights=None):
    """
    calibration: dict {(low, high): observed_correct_rate}
    Returns sum of squared differences between predicted probability and observed frequency.
    """
    penalty = 0.0
    for (low, high), observed in calibration.items():
        mid = (low + high) / 2 / 100  # midpoint as predicted confidence
        w = weights.get((low, high), 1.0) if weights else 1.0
        if not math.isnan(observed):
            penalty += w * (observed - mid) ** 2
    return penalty


def evaluate_grid_point(args):
    sigma_ind, sigma_joint, heldout_year = args

    # --- Select data for this year ---
    final_simulated_votes = ALL_final_simulated_votes[heldout_year]
    proportions_transferred_to_first = ALL_proportions_transferred_to_first[heldout_year]
    Results_dict = ALL_results_dict[heldout_year]
    winners_dict = ALL_winners_dict[heldout_year]

    # --- Call your core function ---
    per_electorate_winners,_ = distribution_to_top_2(
        final_simulated_votes,
        proportions_transferred_to_first,
        Results_dict,
        party_to_category_centered_IND,
        sigma_ind,
        sigma_joint
    )


    #coverage_results = calibration_curve(per_electorate_winners, winners_dict) # coverage_probability(per_simulation_winners, actual_winners)
    #obj_score = objective_function(coverage_results, winners_dict)

    calibration = calibration_curve(per_electorate_winners, winners_dict, n_simulations)
    obj_score = calibration_objective(calibration)


    return {
        'sigma_ind': sigma_ind,
        'sigma_joint': sigma_joint,
        'heldout_year': heldout_year,
        'obj_score': obj_score,
        'calibration_curve': calibration #coverage_results
        #'calibration_curve': calibration  # 🎯 NEW
    }


def plot_calibration_curve(calibration_dict, title='Calibration Curve'):
    # Filter out bins with NaN
    bins = []
    accuracies = []
    for (low, high), acc in calibration_dict.items():
        if not np.isnan(acc):
            bins.append((low + high) / 2)  # Midpoint of the bin
            accuracies.append(acc)

    plt.figure(figsize=(8,6))
    plt.plot(bins, accuracies, 'o-', label='Model Calibration', color='blue')
    plt.plot([30, 100], [0.3, 1.0], 'k--', label='Perfect Calibration')  # from 30%-100%
    plt.xlabel('Predicted Probability (%)')
    plt.ylabel('Observed Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xlim(30, 100)
    plt.show()



TPP_Method = None # 'Validation' 'Simulation'

if TPP_Method == 'Validation':



    # Global dictionaries
    ALL_final_simulated_votes = {}
    ALL_proportions_transferred_to_first = {}
    ALL_results_dict = {}
    ALL_winners_dict = {}


    # Preload before starting the pool
    for election_year in ['2016', '2019', '2022']:
        # ensure we use parameters corresponding to correct years
        params =  FP_validation_results_Day_90[election_year] # from FP model validation
        w,alpha,s,v,beta = params['w'], params['alpha'], params['s'], params['v'], params['beta']
        ALL_final_simulated_votes[election_year],  ALL_results_dict[election_year] = First_Preference_Model_Simulation(election_year = election_year, w = w, alpha=alpha, v = v, s = s, beta = beta, n_simulations=n_simulations)

        ALL_proportions_transferred_to_first[election_year] = make_TCP_pair_category_dict(election_year)
        ALL_winners_dict[election_year] = get_actual_winners(election_year)


    #import pdb;pdb.set_trace()
    
    test_election_years = ['2016','2019','2022']
    sigma_ind_params = [i for i in np.arange(0.5,2.51,0.5)]
    sigma_joint_params = [i for i in np.arange(2,8.1,0.5)]
    #all_sigma = [i for i in np.arange(1,8.5,0.5)] # 0 to 10


    full_results = []

    for heldout_year in test_election_years:
        training_years = [y for y in test_election_years if y != heldout_year]

        # List of all parameter combinations and training years
        train_param_grid = [
            (sigma_ind, sigma_joint, train_year)
            for sigma_ind in sigma_ind_params
            for sigma_joint in sigma_joint_params
            for train_year in training_years
        ]

        # Parallel evaluation across training years
        with multiprocessing.Pool(processes=24) as pool:
            training_results = list(tqdm(
                pool.imap_unordered(evaluate_grid_point, train_param_grid, chunksize=1),
                total=len(train_param_grid)
            ))

        # Group results by (sigma_ind, sigma_joint)
        param_to_scores = defaultdict(list)
        for result in training_results:
            key = (result['sigma_ind'], result['sigma_joint'])
            param_to_scores[key].append(result['obj_score'])

        # Average the training scores
        param_to_mean_score = {
            param: sum(scores) / len(scores)
            for param, scores in param_to_scores.items()
        }

        # Pick best parameters
        best_params = min(param_to_mean_score.items(), key=lambda x: x[1])[0]
        best_sigma_ind, best_sigma_joint = best_params

        # Evaluate best parameters on the held-out year
        heldout_result = evaluate_grid_point((best_sigma_ind, best_sigma_joint, heldout_year))

        # Save final results
        full_results.append({
            'heldout_year': heldout_year,
            'best_sigma_ind': best_sigma_ind,
            'best_sigma_joint': best_sigma_joint,
            'test_obj_score': heldout_result['obj_score'],
            'test_calibration': heldout_result['calibration_curve']
        })


    print(full_results)

    # Grid: [2:12]^2 : [{'heldout_year': '2016', 'best_sigma_ind': 1, 'best_sigma_joint': 1, 'test_obj_score': 0.4148815591389979, 'test_calibration': {(30, 40): 0.0, (40, 50): 0.0, (50, 60): 0.35714285714285715, (60, 70): 0.8421052631578947, (70, 80): 0.7272727272727273, (80, 90): 0.9473684210526315, (90, 95): 1.0, (95, 99): 0.9583333333333334, (99, 100): 1.0}}, {'heldout_year': '2019', 'best_sigma_ind': 1, 'best_sigma_joint': 1, 'test_obj_score': 0.2574570834361045, 'test_calibration': {(30, 40): nan, (40, 50): 0.0, (50, 60): 0.75, (60, 70): 0.7272727272727273, (70, 80): 0.7777777777777778, (80, 90): 0.8235294117647058, (90, 95): 1.0, (95, 99): 0.926829268292683, (99, 100): 1.0}}, {'heldout_year': '2022', 'best_sigma_ind': 1, 'best_sigma_joint': 1, 'test_obj_score': 0.26535128532321933, 'test_calibration': {(30, 40): nan, (40, 50): 0.0, (50, 60): 0.3333333333333333, (60, 70): 0.7692307692307693, (70, 80): 0.7142857142857143, (80, 90): 0.8518518518518519, (90, 95): 0.9444444444444444, (95, 99): 0.972972972972973, (99, 100): 1.0}}]
    # [{'heldout_year': '2016', 'best_sigma_ind': 0.5, 'best_sigma_joint': 6.5, 'test_obj_score': 0.0431912886651867, 'test_calibration': {(40, 60): 0.42857142857142855, (60, 70): 0.7894736842105263, (70, 80): 0.6956521739130435, (80, 90): 0.95, (90, 95): 1.0, (95, 99): 0.9642857142857143, (99, 100): 1.0}}, {'heldout_year': '2019', 'best_sigma_ind': 0.5, 'best_sigma_joint': 2.0, 'test_obj_score': 0.05617569509522556, 'test_calibration': {(40, 60): 0.7272727272727273, (60, 70): 0.7, (70, 80): 0.75, (80, 90): 0.8666666666666667, (90, 95): 0.9230769230769231, (95, 99): 0.9285714285714286, (99, 100): 1.0}}, {'heldout_year': '2022', 'best_sigma_ind': 0.5, 'best_sigma_joint': 6.0, 'test_obj_score': 0.020847464107653906, 'test_calibration': {(40, 60): 0.4166666666666667, (60, 70): 0.7272727272727273, (70, 80): 0.7142857142857143, (80, 90): 0.7692307692307693, (90, 95): 0.9333333333333333, (95, 99): 0.9761904761904762, (99, 100): 1.0}}]
    
    # Final: {'2016': {'w': 0.8500000000000001, 'alpha': 20.0, 's': 0.75, 'v': 0.1, 'beta': 0.8, 'val_score': 3.1886505715847515, 'test_mae': 3.4538902844200066, 'test_coverage': 0.9275653923541247}, '2019': {'w': 0.9, 'alpha': 20.0, 's': 0.75, 'v': 0.1, 'beta': 0.6000000000000001, 'val_score': 3.3291967680158114, 'test_mae': 3.2397858737616434, 'test_coverage': 0.9356060606060606}, '2022': {'w': 0.9, 'alpha': 20.0, 's': 0.8, 'v': 0.05, 'beta': 0.4, 'val_score': 3.3894457195184042, 'test_mae': 3.087975018296332, 'test_coverage': 0.9251870324189526}}
    import pdb;pdb.set_trace()

elif TPP_Method == 'Simulation':

    sigma_joint, sigma_ind = 3.5, 0.5 # 4.2, 0.5


    election_year = '2022'

    final_simulated_votes, Results_dict =  First_Preference_Model_Simulation(election_year, w, alpha, v, s, beta, n_simulations)

    proportions_transferred_to_first = make_TCP_pair_category_dict(election_year)
    actual_winners_dict = get_actual_winners(election_year)
    per_electorate_winners, _ = distribution_to_top_2(final_simulated_votes, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind)


    calibration_dict = calibration_curve(per_electorate_winners, actual_winners_dict, n_simulations=1000, bins=[(30,40), (40,50), (50,60), (60,70), (70,80), (80,90), (90,95), (95,99), (99,100)])


    plot_calibration_curve(calibration_dict, title=f"Calibration Curve {election_year}")

    import pdb;pdb.set_trace()











# for 2025:

if TPP_Method:

    election_year = '2025'

    proportions_transferred_to_first = make_TCP_pair_category_dict(election_year = election_year)


    #import pdb;pdb.set_trace()

    sigma_joint = 4.2 #4.2
    sigma_ind = 0.5


    per_electorate_winners, per_simulation_winners = distribution_to_top_2(final_simulated_votes, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind)



    COAL_PARTIES = {"LNP", "LP", "NP", "CLP"}
    ALP_NAME = "ALP"
    IND_PARTIES = {'IND1','IND2','IND3','IND4','IND5'}

    # Initialize counters
    alp_majority_count = 0
    coal_majority_count = 0
    alp_lead_count = 0
    coal_lead_count = 0
    hung_count = 0

    alp_avg = 0
    coal_avg = 0
    grn_avg = 0
    ind_avg = 0

    alp_seat_list = []
    coal_seat_list = []


    # Loop over simulations
    for i, sim_results in enumerate(per_simulation_winners):
        alp_seats = sum(1 for winner in sim_results if winner == ALP_NAME)
        coal_seats = sum(1 for winner in sim_results if winner in COAL_PARTIES)
        ind_seats = sum(1 for winner in sim_results if winner in IND_PARTIES)
        grn_seats = sum(1 for winner in sim_results if winner == 'GRN')

        coal_seat_list.append(coal_seats)
        alp_seat_list.append(alp_seats)

        alp_avg += alp_seats
        coal_avg += coal_seats
        grn_avg += grn_seats
        ind_avg += ind_seats
        
        if alp_seats >= 76:
            alp_majority_count += 1
        elif coal_seats >= 76:
            coal_majority_count += 1
        elif alp_seats > coal_seats:
            alp_lead_count += 1
        elif coal_seats > alp_seats:
            coal_lead_count += 1
        else:
            hung_count += 1

        print(coal_seats, alp_seats)


    # Compute probabilities
    total_simulations = len(per_simulation_winners)
    alp_prob = alp_majority_count / total_simulations
    coal_prob = coal_majority_count / total_simulations
    alp_lead_prob = alp_lead_count / total_simulations
    coal_lead_prob = coal_lead_count / total_simulations
    hung_prob = hung_count / total_simulations

    print(f"ALP majority probability: {alp_prob:.3f}")
    print(f"COAL majority probability: {coal_prob:.3f}")

import pdb;pdb.set_trace()



def export_simulations_to_csv(sim_dict, Results_dict, output_dir="2025_Election_FP_Practice"):
    """
    sim_dict: dict with keys = electorate names (str), values = (10000 x n_parties) arrays
    party_names: list of party names in column order (length = n_parties)
    output_dir: directory to save each electorate's CSV file
    """

    os.makedirs(output_dir, exist_ok=True)

    for div, sim_array in sim_dict.items():
        n_sims, n_parties = sim_array.shape
        df = pd.DataFrame(sim_array, columns=Results_dict[div].columns)
        df['sim_no'] = df.index
        long_df = df.melt(id_vars='sim_no', var_name='Party', value_name='First Preference Votes')

        csv_path = os.path.join(output_dir, f"{div.replace(' ', '_')}.csv")
        long_df.to_csv(csv_path, index=False)

    print(f"✅ Exported {len(sim_dict)} electorates to {output_dir}/")



def export_winner_proportions_to_csv(winner_dict, output_dir="2025_Election_Winner_Proportions"):
    """
    winner_dict: dict with keys = electorate names (str), values = list of winner party names (length = n_sims)
    output_dir: directory to save each electorate's CSV file

    For each electorate, outputs a CSV with columns:
    - Party
    - Win Percentage
    """

    os.makedirs(output_dir, exist_ok=True)

    for div, winners in winner_dict.items():
        counts = Counter(winners)
        total = sum(counts.values())
        proportions = {party: count / total for party, count in counts.items()}

        df = pd.DataFrame({
            'Party': list(proportions.keys()),
            'Win Percentage': list(proportions.values())
        })

        csv_path = os.path.join(output_dir, f"{div.replace(' ', '_')}.csv")
        df.to_csv(csv_path, index=False)

    print(f"✅ Exported winner proportions for {len(winner_dict)} electorates to {output_dir}/")



def export_simulation_seat_counts(per_simulation_winners, output_csv="2025_Simulated_seat_counts.csv"):
    """
    Export seat counts per simulation, reduced to 8 meta-party categories.
    Each row is a simulation; each column is the seat count for a meta-party.
    """
    # Define party groupings
    coal_parties = {'LP', 'NP', 'LNP', 'CLP'}
    ind_prefixes = {'IND'}  # We'll catch things like IND1, IND2, etc.
    parties_out = ['ALP', 'COAL', 'GRN', 'IND', 'OTH'] # ['ALP', 'COAL', 'GRN', 'IND', 'ON', 'KAP', 'XEN', 'OTH']
    
    all_rows = []

    for sim_winners in per_simulation_winners:
        counts = Counter()

        for party in sim_winners:
            if party in coal_parties:
                counts['COAL'] += 1
            elif any(party.startswith(prefix) for prefix in ind_prefixes):
                counts['IND'] += 1
            elif party in parties_out:
                counts[party] += 1
            else:
                counts['OTH'] += 1

        row = [counts.get(party, 0) for party in parties_out]
        all_rows.append(row)

    import pdb;pdb.set_trace()


    # Create DataFrame and export
    df = pd.DataFrame(all_rows, columns=parties_out)
    df.to_csv(output_csv, index=False)

    print(f"✅ Exported seat counts for {len(per_simulation_winners)} simulations to {output_csv}/")

import pdb;pdb.set_trace()

export = 0

if export:
    export_simulations_to_csv(final_simulated_votes, Results_dict, output_dir="2025_Election_FP_Practice")
    export_winner_proportions_to_csv(per_electorate_winners, output_dir="2025_Election_Winner_Proportions")
    export_simulation_seat_counts(per_simulation_winners, output_csv="2025_Simulated_seat_counts.csv")


import pdb;pdb.set_trace()









# check that the aggregated results match the national polling average
#merged_totals_polling = Polling_estimates.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
#weights_polling = merged_totals_polling['new_vote_totals']/merged_totals_polling['new_vote_totals'].sum()
#weighted_national_polling = (merged_totals_polling.iloc[:,:-1] * weights_polling.values[:,None]).sum().to_frame().T
import pdb;pdb.set_trace()



# Polling_estimates_from_National.to_csv(f"National_Polling_Estimates_{election_year}_Day_{day_of_interest}.csv", index=True)




