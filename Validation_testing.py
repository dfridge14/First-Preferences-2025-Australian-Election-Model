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

from collections import defaultdict



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

n_samples = 100






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
    # COUntNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


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




def simulate_Polling_Fundamentals_model(n_samples, election_year, df_t = 0, v = 1, s = 0.1):

    dist = "Normal" if df_t == 0 else "t"
    #print(dist)
    Volatility_cat = pd.read_csv(f"Volatility_weights_df_{election_year}.csv", index_col= None)

    weights_idx_dict = defaultdict(list)
    for idx, scale in enumerate(Volatility_cat['Volatility_weights']):
        weights_idx_dict[scale].append(idx)

    weights_idx_dict = dict(weights_idx_dict)
    


    National_Polling_error_ALR_cov = pd.read_csv(f"PollingErrorALRCovarianceNational{election_year}.csv", index_col=0)
    National_Simulated_polling_error = np.random.multivariate_normal(mean = np.zeros(len(National_Polling_error_ALR_cov)), cov = National_Polling_error_ALR_cov.values, size=n_samples)[:, None, :] 
    # Broadcast national polling results across 10000 simulations and 150 electorates
    National_Simulated_polling_error_expanded = np.repeat(National_Simulated_polling_error, NO_OF_ELECTORATES[election_year], axis=1)


    National_Election_error_ALR_cov = pd.read_csv(f"ElectionErrorALRCovarianceNational{election_year}.csv", index_col=0)
    National_Simulated_election_error = np.random.multivariate_normal(mean = np.zeros(len(National_Election_error_ALR_cov)), cov = National_Election_error_ALR_cov.values, size=n_samples)[:, None, :] 
    National_Simulated_election_error_expanded = np.repeat(National_Simulated_election_error, NO_OF_ELECTORATES[election_year], axis=1)


    # state errors
    State_Polling_error_ALR_cov = pd.read_csv(f"PollingErrorALRCovarianceStateDeviation{election_year}.csv", index_col=0) 
    State_Simulated_polling_error = np.random.multivariate_normal(mean = np.zeros(len(State_Polling_error_ALR_cov)), cov = State_Polling_error_ALR_cov.values, size=n_samples*NO_OF_STATES)
    State_Simulated_polling_error = State_Simulated_polling_error.reshape(n_samples, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])

    State_Election_error_ALR_cov = pd.read_csv(f"ElectionErrorALRCovarianceStateDeviation{election_year}.csv", index_col=0)  
    State_Simulated_election_error = np.random.multivariate_normal(mean = np.zeros(len(State_Election_error_ALR_cov)), cov = State_Election_error_ALR_cov.values, size=n_samples*NO_OF_STATES)
    State_Simulated_election_error = State_Simulated_election_error.reshape(n_samples, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])


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


    #Electorate_Residuals_Simulated_error = np.random.multivariate_normal(mean = np.zeros(len(Electorate_Residuals_cov)), cov = Electorate_Residuals_cov.values, size=n_samples*NO_OF_ELECTORATES[election_year])
    #Electorate_Residuals_Simulated_error = Electorate_Residuals_Simulated_error.reshape(n_samples, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year])



    Electorate_Residuals_Simulated_error = np.empty((n_samples, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year]))

    d = len(Electorate_Residuals_cov)
    n = n_samples * NO_OF_ELECTORATES[election_year]
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
            size=(n_samples * n_group)
        )

        if dist == 't':
        
            g = np.random.gamma(df_t / 2., 2. / df_t, size=n_samples *n_group)   # 1. Gamma samples (for scaling) ; shape: (n,)
            group_sims = group_sims / np.sqrt(g)[:, None]  # shape: (n, d) # 3. Scale by sqrt(gamma) to simulate from t-distribution


        # Place the group simulations into the correct positions
        Electorate_Residuals_Simulated_error[:, indices, :] = group_sims.reshape(n_samples, n_group, DIM_OF_COV_MATRIX[election_year]) # 4. Reshape to (n_samples, electorates, dim)



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




    # reshape State simulations correctly - map to correct div_nms
    states = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']
    state_to_index = {state: i for i, state in enumerate(states)}
    division_state_indices = Div_relative_weights['State'].map(state_to_index).values  # shape (150,)

    State_Simulated_polling_error_centered_expanded = State_Simulated_polling_error_centered[np.arange(n_samples)[:, None], division_state_indices[None, :], :]
    State_Simulated_election_error_centered_expanded = State_Simulated_election_error_centered[np.arange(n_samples)[:, None], division_state_indices[None, :], :]




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

    #import pdb;pdb.set_trace()





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



        import pdb;pdb.set_trace()


        return Electorate_residuals_ALR_df.cov(min_periods=1)

    test = 0

    if test:
        Electorate_residuals_covMs = {}
        for year_to_remove in ['2016','2019','2022','2025']:
            Electorate_residuals_covMs[year_to_remove] = test_variability_of_Electorate_Residuals(new_vote_totals_states, year_to_remove)
            #import pdb;pdb.set_trace()

            Electorate_residuals_covMs[year_to_remove].to_csv(f"ElectorateResidualALRCovariance{year_to_remove}.csv", index = True)

        #import pdb;pdb.set_trace()

    #print(National_prior)






    day_80_polling_avg_dict = {'2016': pd.DataFrame([[0.412262, 0.351972, 0.105693, 0.130074]], columns = ['COAL','ALP','GRN','OTH']), \
                        '2019':pd.DataFrame([[0.384782, 0.36451, 0.095751, 0.035, 0.031, 0.088957]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
                        '2022':pd.DataFrame([[0.355905, 0.362643, 0.118432, 0.0383,0.0244,0.10032,]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
                        '2025':pd.DataFrame([[0.34757, 0.312214, 0.125763, 0.07127, 0.011843, 0.131341]], columns = ['COAL','ALP','GRN','ON','TOP','OTH'])}

    
    state_poll_dev_alr = pd.read_csv("State_Polling_Deviations_from_National.csv", index_col=None)
    state_poll_dev_alr_2025 = pd.read_csv("2025_State_Polling_Deviations_from_National.csv", index_col=None)


    State_Polls_Deviations_from_National_df_dict = {'2016': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2016,].drop(['ON','Election_year'], axis=1), \
                                                    '2019': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2019,].drop(['Election_year'], axis=1).fillna(0), \
                                                    '2022': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2022,].drop(['Election_year'], axis=1).fillna(0), \
                                                    '2025': state_poll_dev_alr_2025.drop(['Election_year'], axis=1).fillna(0),}


    day_80_polling_avg = day_80_polling_avg_dict[election_year]/ day_80_polling_avg_dict[election_year].sum(axis=1)[0]

        







    # get alr values of all quantities
    ref_col = 'COAL'
    polling_alr = np.log(day_80_polling_avg.drop(columns=[ref_col]).div(day_80_polling_avg[ref_col], axis=0))
    National_prior_alr = np.log(National_prior.drop(columns=[ref_col]).div(National_prior[ref_col], axis=0))

    State_prior_alr =  np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))
    State_prior_expanded = np.tile(State_prior_alr.to_numpy(), (n_samples, 1, 1)).reshape(n_samples, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])
    State_prior_expanded = State_prior_expanded[np.arange(n_samples)[:, None], division_state_indices[None, :], :] # Then use advanced indexing to map divisions to their state across all samples

    #import pdb;pdb.set_trace()
    # get initial state deviations and expand
    State_prior_deviation_alr_expanded = State_prior_expanded - National_prior_alr.values.flatten()





    Prior_estimates_alr =  np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))
    Prior_estimates_alr_expanded = np.tile(Prior_estimates_alr.to_numpy(), (n_samples, 1, 1))

    # get State deviations into a (10000, 8, 5) array
    State_polling_deviation_alr = State_Polls_Deviations_from_National_df_dict[election_year].set_index('State')
    if election_year in ['2019','2022']:
        State_polling_deviation_alr.loc[:,'UAPP'] = 0.0 # add 0 deviation from National if no state polling!
        State_polling_deviation_alr = State_polling_deviation_alr[['ALP','GRN','ON','UAPP','OTH']]
    elif election_year == '2025':
        State_polling_deviation_alr.loc[:,'TOP'] = 0.0 # add 0 deviation from National if no state polling!
        State_polling_deviation_alr = State_polling_deviation_alr[['ALP','GRN','ON','TOP','OTH']]

    # scale the state polling deviations, based on relative polling precision (sample size etc.)
    Scaled_precisions_curr = pd.read_csv("State_Polling_Scaled_Precisions.csv", index_col = 0).loc[int(election_year)]
    relative_state_precisions = Scaled_precisions_curr.set_index('Scope', drop = True).drop('Mean_precision', axis = 1)
    #relative_state_precisions = relative_state_precisions['Scaled_Precisions']/relative_state_precisions['Scaled_Precisions'].mean()

    #import pdb;pdb.set_trace()
    
    State_polling_deviation_alr = State_polling_deviation_alr.mul(s*relative_state_precisions.values, axis = 0) # now scaled based on state precision

    s_i = s * relative_state_precisions

    print("s_i", v, s)

    #import pdb;pdb.set_trace()


    State_polling_deviation_alr_matrix = State_polling_deviation_alr.values  # Convert to numpy array for easy broadcasting
    State_polling_deviation_alr_matrix = np.expand_dims(State_polling_deviation_alr_matrix, axis=0)  # Add an extra dimension for broadcasting
    State_polling_deviation_alr_matrix_expanded = np.repeat(State_polling_deviation_alr_matrix, n_samples, axis=0)  
    State_polling_deviation_alr_matrix_expanded = State_polling_deviation_alr_matrix_expanded[np.arange(n_samples)[:, None], division_state_indices[None, :], :]

    #State_prior_alr = np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))

    # apply National Polling error
    Simulated_national_result_alr = National_Simulated_polling_error_expanded + polling_alr.values  # shape: [1M, 5]

    # apply State Polling error
    Simulated_state_polling_deviation = State_polling_deviation_alr_matrix_expanded + State_Simulated_polling_error_centered_expanded

    # weight State Polling, relative to National: use linear combination of last year's 
    #import pdb;pdb.set_trace()
    
    #state_weights = s * Year_state_polling_weights.loc[int(election_year)].iloc[0]

    # get 1 - s_i per state

    Scaled_national_polling_deviations = (National_prior_alr.values - State_prior_alr).mul(1-s_i.values, axis = 0)
    Scaled_national_polling_deviations.index = Scaled_national_polling_deviations.index.map(state_to_index)
    seat_state_alr = Scaled_national_polling_deviations.iloc[division_state_indices]
    Scaled_national_polling_deviations_expanded = np.broadcast_to(seat_state_alr, (n_samples, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year])).copy()
    
    #Scaled_national_polling_deviations_array = np.expand_dims(Scaled_national_polling_deviations, axis=0)  # Add an extra dimension for broadcasting
    #Scaled_national_polling_deviations_expanded = np.repeat(Scaled_national_polling_deviations_array, n_samples, axis=0)  
    #Scaled_national_polling_deviations_expanded = Scaled_national_polling_deviations_expanded[np.arange(n_samples)[:, None], division_state_indices[None, :], :]


    Combined_state_mean = polling_alr.values + Scaled_national_polling_deviations_expanded + State_polling_deviation_alr_matrix_expanded # already scaled!
    Combined_state_errors = National_Simulated_polling_error_expanded + State_Simulated_polling_error_centered_expanded

    Simulated_State_Weighted_Polling_Results = Combined_state_mean + Combined_state_errors

    #Simulated_State_Polling_Results = Simulated_national_result_alr + (1 - state_weights) * (National_prior_alr.values - State_prior_expanded) + (state_weights) * Simulated_state_polling_deviation



    Projected_Electorate_Results = Prior_estimates_alr_expanded + (Simulated_State_Weighted_Polling_Results - State_prior_expanded)

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

    Projected_Electorate_Swing_Results = Prior_estimates_alr_expanded + (Simulated_State_election_Results - State_prior_expanded)

    Simulated_Electorate_Swing_Results_ALR = Projected_Electorate_Swing_Results + Electorate_Residuals_Simulated_error_centered

    Simulated_Electorate_Swing_Results = alr_to_simplex_simulation_array(Simulated_Electorate_Swing_Results_ALR)




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
            #import pdb;pdb.set_trace()
            COAL_votes = combined[:,0]

            if 'COALLP' in prior_row.columns:
                print('COALLP')
                import pdb;pdb.set_trace()


            if 'NP' in prior_row.columns and 'LP' in prior_row.columns:
                NP_est = (prior_row['NP'] /  prior_row[['LP','NP']].sum(axis=1)).iloc[0]
            else:
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




def perform_validation_testing(n_samples, coverage_level, coverage_weight = 5):




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
                    Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_samples, election_year, df_t=df_t, v=v)

                    Polling_simulations_dict[election_year] = Simulated_Electorate_Polling_Results
                    Election_swing_simulations_dict[election_year] = Simulated_Electorate_Swing_Results

                    print("Done 20000 simulation processing:", time.time() - start_time, "seconds")

                
                for w in weights:
                    n_polling_samples = int(w * n_samples)

                    indices_poll = np.random.choice(n_samples, n_polling_samples, replace=False)
                    indices_fund = np.random.choice(n_samples, n_samples - n_polling_samples, replace=False)

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

        Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_samples, heldout, df_t=best_df_t, v=best_v)
        
        # Now evaluate on heldout
        n_polling_samples = int(best_w*n_samples)
        indices_poll = np.random.choice(n_samples, n_polling_samples, replace=False)
        indices_fund = np.random.choice(n_samples, n_samples - n_polling_samples, replace=False)
        combined_samples = np.concatenate((Simulated_Electorate_Polling_Results[indices_poll],  Simulated_Electorate_Swing_Results[indices_fund]), axis=0)

        heldout_mae, heldout_coverage = get_election_MAE(combined_samples, Prior_estimates_dict_per_election[heldout], Results_dict_per_election[heldout], heldout, coverage_level, alpha)
        best_params[heldout]['test_mae'] = heldout_mae
        best_params[heldout]['test_coverage'] = heldout_coverage

    return best_params





def worker(job):

    coverage_weight = 5
    
    weights = np.linspace(0, 1, 21) # np.array([0.4,0.6])# np.linspace(0, 1, 21) # 
    alphas = np.linspace(20,60, 9) # np.array([100,1000])# 
   

    elections = ['2016','2019','2022']
    heldout, s, v = job
    train_elections = [e for e in elections if e != heldout]

    Polling_simulations_dict = {}
    Election_swing_simulations_dict = {}

    Prior_estimates_dict_per_election = {}
    Results_dict_per_election = {}

    for election_year in train_elections:
        polling, swing = simulate_Polling_Fundamentals_model(n_samples, election_year, s = s, v=v)
        Polling_simulations_dict[election_year] = polling
        Election_swing_simulations_dict[election_year] = swing

        Prior_estimates_dict_per_election[election_year] = get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # single row df for each div_nm
        Results_dict_per_election[election_year] = get_results_df(election_year, to_Fundamentals=False)[1]

    results = []

    for w in weights:
        n_polling_samples = int(w * n_samples)
        indices_poll = np.random.choice(n_samples, n_polling_samples, replace=False)
        indices_fund = np.random.choice(n_samples, n_samples - n_polling_samples, replace=False)

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

            results.append((s, v, w, alpha, avg_score))

    return results


def parallelise_validation_simulation(coverage_level, coverage_weight):
    elections = ['2016','2019','2022']
    s_s = np.linspace(0,0,1) # np.array([5,0])#  0 --> normal! 
    vs = np.linspace(0.05, 0.5, 10) # np.array([0.25,0.5])# 

    jobs = list(product(elections, s_s, vs))

    with ProcessPoolExecutor(max_workers=24) as executor:
        results_nested = list(tqdm(executor.map(worker, jobs), total=len(jobs), desc="Processing jobs"))


    # Flatten the nested list of results into a single list
    results_flat = []

    for i, results in enumerate(results_nested):
        heldout = jobs[i][0]  # Get the heldout value from jobs (index i)
        if not isinstance(results, list):
            print(f"[ERROR] Unexpected type at job {i}: {type(results)} - {results}")
            continue

        for result in results:
            if len(result) != 5:
                print(f"[ERROR] Malformed result at job {i}: {result}")
                continue
            s, v, w, alpha, avg_score = result
            results_flat.append((heldout, s, v, w, alpha, avg_score))

    # Create DataFrame
    df_results = pd.DataFrame(results_flat, columns=['heldout', 's', 'v', 'w', 'alpha', 'avg_score'])

    best_params = {}

    for heldout in df_results['heldout'].unique():
        df_heldout = df_results[df_results['heldout'] == heldout]
        
        # Get the row with minimum validation MAE
        best_row = df_heldout.loc[df_heldout['avg_score'].idxmin()]

        # Extract best params
        best_s = best_row['s']
        best_v = best_row['v']
        best_w = best_row['w']
        best_alpha = best_row['alpha']
        best_val_score = best_row['avg_score']

        best_params[heldout] = {
            'w': best_w,
            'alpha': best_alpha,
            's': best_s,
            'v': best_v ,
            'val_score': best_val_score
        }

        Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_samples, heldout, s=best_s, v=best_v)

        Prior_estimates_dict = get_Prior_estimates_df(heldout, dont_add_ON = True)[1] # single row df for each div_nm
        Results_dict = get_results_df(heldout, to_Fundamentals=False)[1]

        
        # Now evaluate on heldout
        n_polling_samples = int(best_w*n_samples)
        indices_poll = np.random.choice(n_samples, n_polling_samples, replace=False)
        indices_fund = np.random.choice(n_samples, n_samples - n_polling_samples, replace=False)
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



Method = 'Validation' # 'Validation'
coverage_level = 0.95

    
if Method == 'Validation':
    validation = {}
    #for coverage_weight in [2,3,5,10]:
    best_params = parallelise_validation_simulation(coverage_level, 5)
    #validation[coverage_weight] = best_params
    print(best_params)
    print(time.time() - start_time)
    import pdb;pdb.set_trace()

elif Method =='Simulation':

    w = 0.8
    alpha = 30
    v = 0.30

    election_year = '2016'

    Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_samples, election_year, v = v)

    import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()

    print(f"Done {n_samples*2}simulation processing:", time.time() - start_time, "seconds")

    Prior_estimates_dict = get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # single row df for each div_nm

    if election_year != '2025':
        Results_dict = get_results_df(election_year, to_Fundamentals=False)[1]
    else:
        Candidates_2025 = pd.read_csv("2025Candidates_By_Division.csv", index_col = None)

        Results_dict = {}

        # Make a faux-results dict

        target = 'IND'
        for div, sub_df in Candidates_2025.groupby('div_nm', sort=False):

            div_parties = sub_df

            div_parties.loc[:,'Count'] = div_parties.groupby('PartyAb').cumcount() + 1     # Count instances of the target string

            # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...

            adjusted_party_names = div_parties.apply(
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            ).reset_index(drop=True)

            # separate independents
            div_parties.loc[div_parties['div_nm'] == div,'PartyAb'] = adjusted_party_names.values
            div_results_combined = div_parties.drop('Count', axis = 1)

            ordered_parties = div_results_combined['PartyAb'].drop_duplicates()

            pivoted = div_results_combined.pivot(index='div_nm', columns='PartyAb')
            
            Results_dict[div] = pivoted.reindex(columns = ordered_parties)

        #import pdb;pdb.set_trace()



    indices = np.random.permutation(n_samples)


    Weighted_Simulations = np.concatenate((Simulated_Electorate_Polling_Results[indices[:int(w*n_samples)]],  Simulated_Electorate_Swing_Results[indices[int(w*n_samples):]]), axis=0)


    final_simulated_votes = expand_all_divisions_from_prior_df(Weighted_Simulations, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=alpha)[0]


    top_n_dict = get_topn_sets_per_electorate(final_simulated_votes, n=2, threshold=0.15)

    top_n_names_dict = index_tuples_to_candidate_names(top_n_dict, Results_dict)

    all_names = set(
        name
        for name_lists in top_n_names_dict.values()
        for name_list in name_lists
        for name in name_list
    )

    import pandas as pd


def redistribute():
    top2_idx = np.argsort(sim_votes, axis=1)[:, -2:]
    top2_names = np.sort(np.array(party_names)[top2_idx])

    top2_votes = np.take_along_axis(sim_votes, top2_idx, axis=1).astype(float)

    for i, party in enumerate(party_names):
        if party in top2_names:  # skip if party is a top 2 in any sim
            continue
        if party not in prefs:
            continue

    for (p1, p2), share_p1 in prefs[party].items():
        pair = tuple(sorted([p1, p2]))
        match_mask = np.all(top2_names == pair, axis=1)

    winner_idx = np.argmax(top2_votes, axis=1)
    winners = top2_names[np.arange(n_sim), winner_idx]

    unique, counts = np.unique(winners, return_counts=True)
    proportions = dict(zip(unique, counts / n_sim))

    return proportions








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





export_simulations_to_csv(final_simulated_votes, Results_dict, output_dir="2025_Election_FP_Practice")


import pdb;pdb.set_trace()









# check that the aggregated results match the national polling average
#merged_totals_polling = Polling_estimates.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
#weights_polling = merged_totals_polling['new_vote_totals']/merged_totals_polling['new_vote_totals'].sum()
#weighted_national_polling = (merged_totals_polling.iloc[:,:-1] * weights_polling.values[:,None]).sum().to_frame().T
import pdb;pdb.set_trace()



# Polling_estimates_from_National.to_csv(f"National_Polling_Estimates_{election_year}_Day_{day_of_interest}.csv", index=True)




