import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import os
from pathlib import Path
import matplotlib.pyplot as plt

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

NO_OF_STATES = 8
NO_OF_ELECTORATES = {'2016':150,'2019':151,'2022':151,'2025':150}
DIM_OF_COV_MATRIX = {'2016':3,'2019':5,'2022':5,'2025':5}

election_year = '2019'

n_samples = 10000
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

Electorate_Residuals_cov = pd.read_csv(f"ElectorateResidualALRCovariance{election_year}.csv", index_col=0) 
Electorate_Residuals_Simulated_error = np.random.multivariate_normal(mean = np.zeros(len(Electorate_Residuals_cov)), cov = Electorate_Residuals_cov.values, size=n_samples*NO_OF_ELECTORATES[election_year])
Electorate_Residuals_Simulated_error = Electorate_Residuals_Simulated_error.reshape(n_samples, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year])



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





if election_year in ['2016','2019','2022']:
    last_election_vote_totals = pd.read_csv(f"{last_election_year}HouseVotesCountedByDivision.csv", skiprows=1, index_col=None).rename(columns={'DivisionNm':'old_div'})[['old_div', 'TotalVotes']]
    redistribution_df = pd.read_csv(f'Correspondence_CED_{str(int(election_year)-4)}_{str(int(election_year)-1)}.csv', index_col = None)

    merged_df = redistribution_df.merge(last_election_vote_totals, on="old_div")
    merged_df["new_vote_totals"] = merged_df["TotalVotes"] * merged_df["RATIO_FROM_TO"]
    new_vote_totals = merged_df.groupby("new_div")["new_vote_totals"].sum().reset_index().rename(columns={'new_div':'div_nm'})

    div_to_state = pd.read_csv(f"{election_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
    new_vote_totals_states = new_vote_totals.merge(div_to_state, on = 'div_nm', how='left')



def group_into_Categories(party_votes_shares_df, div, election_year, is_Other = True):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'COAL','COALNP','COALLP','LP','NP','CLP','LNP','LNQ'}
    GRN_cat = {'GRN'}
    UAPP_cat = {'UAPP'}
    ON_cat = {'ON'}

    Non_Other_sets = ALP_cat | COAL_cat | GRN_cat # Union of all sets
    if election_year in ['2019','2022']:
        Non_Other_sets = Non_Other_sets | UAPP_cat | ON_cat 
    Other_cols = set(party_votes_shares_df.columns) - Non_Other_sets  # Columns in none of the sets

    ALPs = ALP_cat.intersection(party_votes_shares_df.columns)
    COALs = COAL_cat.intersection(party_votes_shares_df.columns)
    GRNs = GRN_cat.intersection(party_votes_shares_df.columns)
    if election_year in ['2019','2022']:
        ONs =  ON_cat.intersection(party_votes_shares_df.columns)
        UAPPs = UAPP_cat.intersection(party_votes_shares_df.columns)
    OTHs = Other_cols

    # Compute the sums
    sum1 = party_votes_shares_df[list(next(iter(ALPs)) if len(ALPs) == 1 and isinstance(next(iter(ALPs)), set) else ALPs)].sum(axis=1).iloc[0]
    sum2 = party_votes_shares_df[list(next(iter(COALs)) if len(COALs) == 1 and isinstance(next(iter(COALs)), set) else COALs)].sum(axis=1).iloc[0]
    sum3 = party_votes_shares_df[list(next(iter(GRNs)) if len(GRNs) == 1 and isinstance(next(iter(GRNs)), set) else GRNs)].sum(axis=1).iloc[0]
    if election_year in ['2019','2022']:
        sum4 = party_votes_shares_df[list(next(iter(ONs)) if len(ONs) == 1 and isinstance(next(iter(ONs)), set) else ONs)].sum(axis=1).iloc[0]
        sum5 = party_votes_shares_df[list(next(iter(UAPPs)) if len(UAPPs) == 1 and isinstance(next(iter(UAPPs)), set) else UAPPs)].sum(axis=1).iloc[0]
    sum6 = party_votes_shares_df[list(next(iter(OTHs)) if len(OTHs) == 1 and isinstance(next(iter(OTHs)), set) else OTHs)].sum(axis=1).iloc[0]
    if election_year == '2016':
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum6}], index=[div])
    elif election_year in ['2019','2022']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'UAPP':sum5, 'Other':sum6}], index=[div])

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

def get_results_df(election_year):
    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    # COUntNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


    # Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
    Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
    Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
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

        div_results_combined = div_results.groupby(['div_nm', 'PartyAb'], as_index=False)['CalculationValue'].sum()

        Actual_results.loc[Actual_results['div_nm'] == div,'PartyAb'] = adjusted_party_names

        Actual_results_dict[div] = div_results_combined.pivot(index='div_nm', columns='PartyAb', values='CalculationValue')
        #Fundamentals_results_dict[div] = group_into_Fundamentals_Categories(Actual_results_dict[div], div)
        #Fundamentals_estimate_dict[div] = group_into_Fundamentals_Categories(Prior_estimates_dict[div], div)

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


    # transfer prior %s from ON to original_df parties
    for div, proportions in true_prior_estimates_df.iterrows():
        if proportions['ON']==0:
            curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
            curr_div_True = proportions.to_frame().T

            transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop('ON', axis=1).sum(axis=1).iloc[0]) # This should provide -1 for ON automatically!
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
        Prior_estimates_df.loc[Prior_estimates_df.index=='Gorton',['GRN','OTH']] += (-0.01,+0.01)
    elif adjust_No_OTHs:
        No_OTH_divisions = Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,].index
        Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,['GRN','OTH']] += (-0.005,+0.005)


    return Prior_estimates_df, National_prior, State_prior_df, No_OTH_divisions



if election_year in ['2016','2019','2022']:
    Prior_estimates_df, National_prior, State_prior_df, No_OTH_divisions = get_National_State_Prior_estimates(election_year, new_vote_totals_states)


def alr_to_simplex_vectorized(df, ref_col):
    """Inverse ALR transformation for an entire DataFrame in a vectorized way."""
    # Convert the DataFrame to a numpy array for vectorized operations
    alr_vals = df.values
    
    # Apply the inverse ALR transformation across all values
    exp_vals = np.exp(alr_vals)

    # Compute the reference category correctly
    ref_vals = 1 / (1 + np.sum(exp_vals, axis=1, keepdims=True))  # Shape: (n_samples, 1)

    # Compute all components
    simplex_vals = np.concatenate((exp_vals * ref_vals, ref_vals), axis=1)  # Shape: (n_samples, D)
    
    # Create new column names, appending a reference category
    new_columns = df.columns.tolist() + [ref_col]
    
    # Return as a DataFrame with the original indices and new columns
    return pd.DataFrame(simplex_vals, columns=new_columns, index=df.index)

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
    #import pdb;pdb.set_trace()

    return Electorate_residuals_ALR_df.cov(min_periods=1)

test = 0

if test:
    Electorate_residuals_covMs = {}
    for year_to_remove in ['2016','2019','2022','2025']:
        Electorate_residuals_covMs[year_to_remove] = test_variability_of_Electorate_Residuals(new_vote_totals_states, year_to_remove)
        import pdb;pdb.set_trace()

        Electorate_residuals_covMs[year_to_remove].to_csv(f"ElectorateResidualALRCovariance{year_to_remove}.csv", index = True)

    import pdb;pdb.set_trace()






day_80_polling_avg_dict = {'2016': pd.DataFrame([[0.412262, 0.351972, 0.105693, 0.130074]], columns = ['COAL','ALP','GRN','OTH']), \
                      '2019':pd.DataFrame([[0.384782, 0.36451, 0.095751, 0.035, 0.031, 0.088957]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
                      '2022':pd.DataFrame([[0.355905, 0.362643, 0.118432, 0.0383,0.0244,0.10032,]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
                      '2025':[]}

state_poll_dev_alr = pd.read_csv("State_Polling_Deviations_from_National.csv", index_col=None)
State_Polls_Deviations_from_National_df_dict = {'2016': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2016,].drop(['ON','Election_year'], axis=1), \
                                                '2019': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2019,].drop(['Election_year'], axis=1).fillna(0), \
                                                '2022': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2022,].drop(['Election_year'], axis=1).fillna(0), \
                                                '2025': []}


day_80_polling_avg = day_80_polling_avg_dict[election_year]/ day_80_polling_avg_dict[election_year].sum(axis=1)[0]



# get alr values of all quantities
ref_col = 'COAL'
polling_alr = np.log(day_80_polling_avg.drop(columns=[ref_col]).div(day_80_polling_avg[ref_col], axis=0))
National_prior_alr = np.log(National_prior.drop(columns=[ref_col]).div(National_prior[ref_col], axis=0))

State_prior_alr =  np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))
State_prior_expanded = np.tile(State_prior_alr.to_numpy(), (n_samples, 1, 1)).reshape(n_samples, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])
State_prior_expanded = State_prior_expanded[np.arange(n_samples)[:, None], division_state_indices[None, :], :] # Then use advanced indexing to map divisions to their state across all samples

import pdb;pdb.set_trace()
# get initial state deviations and expand
State_prior_deviation_alr_expanded = State_prior_expanded - National_prior_alr.values.flatten()







Prior_estimates_alr =  np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))
Prior_estimates_alr_expanded = np.tile(Prior_estimates_alr.to_numpy(), (n_samples, 1, 1))

# get State deviations into a (10000, 8, 5) array
State_polling_deviation_alr = State_Polls_Deviations_from_National_df_dict[election_year].set_index('State')
if election_year in ['2019','2022','2025']:
    State_polling_deviation_alr.loc[:,'UAPP'] = 0.0 # add 0 deviation from National if no state polling!
    State_polling_deviation_alr = State_polling_deviation_alr[['ALP','GRN','ON','UAPP','OTH']]
State_polling_deviation_alr_matrix = State_polling_deviation_alr.values  # Convert to numpy array for easy broadcasting
State_polling_deviation_alr_matrix = np.expand_dims(State_polling_deviation_alr_matrix, axis=0)  # Add an extra dimension for broadcasting
State_polling_deviation_alr_matrix_expanded = np.repeat(State_polling_deviation_alr_matrix, n_samples, axis=0)  
State_polling_deviation_alr_matrix_expanded = State_polling_deviation_alr_matrix_expanded[np.arange(n_samples)[:, None], division_state_indices[None, :], :]

#State_prior_alr = np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))

# apply National Polling error
Simulated_national_result_alr = National_Simulated_polling_error_expanded + polling_alr.values  # shape: [1M, 5]

# apply State Polling error
Simulated_state_polling_deviation = State_polling_deviation_alr_matrix_expanded + State_Simulated_polling_error_centered_expanded
Simulated_State_Polling_Results = Simulated_national_result_alr + Simulated_state_polling_deviation

Projected_Electorate_Results = Prior_estimates_alr_expanded + (Simulated_State_Polling_Results - State_prior_expanded)

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




#import pdb;pdb.set_trace()


# Post-processing to remove artificial additions (includding ON in 2019/22)!
if election_year == '2016':
    div_idx = Div_relative_weights.index.get_loc('Gorton')
    OTH = Simulated_Electorate_Polling_Results[:, div_idx, 3]  # shape (10000,)
    # Apply the shift: increase GRN, decrease OTH
    Simulated_Electorate_Polling_Results[:, div_idx, 2] += 1.0 * OTH  # GRN (index 2)
    Simulated_Electorate_Polling_Results[:, div_idx, 3] -= 1.0 * OTH  # OTH (index 3)

    import pdb;pdb.set_trace()

def shift_share(sim, div_idx, from_party_idx, to_party_idx, proportion=1.0):
    """Shifts a proportion of vote share from one party to another in one division, across all simulations."""
    shift_amount = proportion * sim[:, div_idx, from_party_idx]
    sim[:, div_idx, to_party_idx] += shift_amount
    sim[:, div_idx, from_party_idx] -= shift_amount

    return 1

def redistribute_ON_votes(sim, division_names, party_names, ON_transfer_dict):
    party_index_map = {name: idx for idx, name in enumerate(party_names)}
    ON_idx = party_index_map['ON']

    for div_nm, transfer_row in ON_transfer_dict.items():
        div_idx = division_names.index(div_nm)

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

        # return ON back to its country in divs it did not run in!
        ON_transfer_dict = remove_ON_back_to_its_country(Prior_estimates_df, election_year)

        Polled_parties = ['COAL','ALP','GRN','ON','UAPP','OTH'] if  election_year != '2025' else ['COAL','ALP','GRN','ON','TOP','OTH']
        redistribute_ON_votes(Simulated_Electorate_Polling_Results, Div_relative_weights.index.tolist(), Polled_parties, ON_transfer_dict)




#import pdb;pdb.set_trace()

Prior_estimates_dict = get_Prior_estimates_df(election_year, dont_add_ON = False)[1] # single row df for each div_nm
Results_dict = get_results_df(election_year)[1]




def expand_all_divisions_from_prior_df(sim, Prior_estimates_dict, election_year, multiple_INDs_dict, alpha_scalar=100):
    final_sim = {}
    party_name_dict = {}
    NUM_MAIN_PARTIES = DIM_OF_COV_MATRIX[election_year]

    Major_parties = ['COAL','LP','NP','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']
    if election_year != '2016':
        Major_parties += ['ON','UAPP','TOP']
        if election_year != '2025':
            Polling_parties += ['ON','UAPP']
        else:
            Polling_parties += ['ON','TOP']


    # For the split of COAL_double_divs
    NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)
    NP_ratios_curr = NP_ratios_curr.loc[(NP_ratios_curr['election_year']==election_year) & (NP_ratios_curr['State'].isin(['VIC','NSW'])),]
        

    for i, div in enumerate(Prior_estimates_dict.keys()): # will be alphabetical
        sim_block = sim[:, i, :]            # shape (10000, 4)
        main_parties = sim_block[:, :NUM_MAIN_PARTIES]     # shape (10000, 3)
        other_share = sim_block[:, NUM_MAIN_PARTIES]       # shape (10000,)

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


        if div in NP_ratios_curr['div_nm'].unique():
            COAL_votes = combined[:,0]
            NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate']
            alpha = np.array([1-NP_est,NP_est]) * alpha_scalar
            splits = np.random.dirichlet(alpha, size=sim.shape[0])
            LP_NP_votes = splits * COAL_votes[:, None]

            combined = np.concatenate([combined, LP_NP_votes], axis=1)[:,1:] # Removes 'COAL' from 1st column

            all_party_names = Polling_parties[1:] + minor_names + ['LP','NP'] # correct order

        # perform independent split as well!
        if 'IND' in Prior_estimates_dict[div]:
            
            if div in multiple_INDs:

            else:
                all_party_names = [p if p != 'IND' else 'IND1' for p in all_party_names]


        

        # Store results
        final_sim[div] = combined 
        party_name_dict[div] = all_party_names  # add names to avoid confusion in future!




    return final_sim, party_name_dict


final_sim, party_name_dict = expand_all_divisions_from_prior_df(Simulated_Electorate_Polling_Results, Prior_estimates_dict, election_year, alpha_scalar=100)




import pdb;pdb.set_trace()








# check that the aggregated results match the national polling average
merged_totals_polling = Polling_estimates.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
weights_polling = merged_totals_polling['new_vote_totals']/merged_totals_polling['new_vote_totals'].sum()
weighted_national_polling = (merged_totals_polling.iloc[:,:-1] * weights_polling.values[:,None]).sum().to_frame().T
import pdb;pdb.set_trace()



import pdb;pdb.set_trace()
# Polling_estimates_from_National.to_csv(f"National_Polling_Estimates_{election_year}_Day_{day_of_interest}.csv", index=True)




