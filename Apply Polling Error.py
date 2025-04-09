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

election_year = '2016'

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




# weights of each state and division:
Div_relative_weights_dict = {}
State_relative_weights_dict = {}
for year in ['2016','2019','2022','2025']:
    last_election_year = str(int(year) - 3)

    Enrolment_by_Div_prev = pd.read_csv(f"{last_election_year}GeneralEnrolmentByDivision.csv",index_col=None, skiprows=1).rename(columns={'DivisionNm':'old_div','StateAb':'State'})[['old_div','State','Enrolment']]
    # adjust for redistribution
    Correspondence_old_new = pd.read_csv(f"Correspondence_CED_{str(int(election_year) - 4)}_{str(int(election_year) - 1)}.csv")
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

    return Prior_estimates_df

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


    return Fundamentals_results_df

def remove_ON_back_to_its_country(Prior_estimates_df, Polling_estimates, election_year):


    # determine transfer ratio of ON votes

    Prior_estimates_ON_add_df = Prior_estimates_df

    true_prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON = True).rename(columns={'Other':'OTH'})

    ON_transfer_percent = {}

    # transfer prior %s from ON to original_df parties
    for div, proportions in true_prior_estimates_df.iterrows():
        if proportions['ON']==0:
            curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
            curr_div_True = proportions.to_frame().T

            transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop('ON', axis=1).sum(axis=1).iloc[0]) # This should provide -1 for ON automatically!
            ON_transfer_percent[div] = transfer_proportions

    #import pdb;pdb.set_trace()

    # distribute these ON votes to actual 
    for div in ON_transfer_percent.keys():
        new_row = (Polling_estimates.loc[div] + Polling_estimates.loc[div]['ON'] * ON_transfer_percent[div])
        new_row['ON'] = 0.0
        Polling_estimates.loc[div] =  new_row.iloc[0]


    #import pdb;pdb.set_trace()

    return Polling_estimates


def get_National_State_Prior_estimates(election_year, new_vote_totals_states, dont_add_ON = False):

    Prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON).rename(columns={'Other':'OTH'}) # adds ON to every seat if no ON (for 2019 and 2022)

    merged_totals = Prior_estimates_df.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
    weights = merged_totals['new_vote_totals']/merged_totals['new_vote_totals'].sum()
    National_prior = (merged_totals.iloc[:,:-1] * weights.values[:,None]).sum().to_frame().T

    merged_totals_states = Prior_estimates_df.merge(new_vote_totals_states.set_index('div_nm'), left_index=True, right_index=True)

    State_prior_df = pd.DataFrame(columns=Prior_estimates_df.columns)
    for state in sorted(merged_totals_states['StateAb'].unique()):
        merged_totals_curr_state = merged_totals_states.loc[merged_totals_states['StateAb']==state,].drop('StateAb', axis = 1) # no longer need StateAb
        curr_weights = merged_totals_curr_state['new_vote_totals']/merged_totals_curr_state['new_vote_totals'].sum()
        State_prior = (merged_totals_curr_state.iloc[:,:-1] * curr_weights.values[:,None]).sum().to_frame().T
        State_prior.index = [state]
        State_prior_df = pd.concat([State_prior_df, State_prior])



    # add 0.001 to Gorton Other in 2016, or 0 others in 2019/2022 (use ON votes as they are higher --> less distortion)!
    if election_year == '2016':
        Prior_estimates_df.loc[Prior_estimates_df.index=='Gorton',['GRN','OTH']] += (-0.01,+0.01)
        No_OTH_divisions = []
    else:
        No_OTH_divisions = Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,].index
        Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,['ON','OTH']] += (-0.005,+0.005)

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






# test for variability of seat results corrected for state swings.
State_Results_2016_2022 = pd.read_csv('StateResults2016_2022.csv', index_col=None)
CAGO = 1

for year in ['2016','2019','2022']: # ['2016','2019','2022']:
    Prior_estimates_df1, National_prior1, State_prior_df1, No_OTH_divisions1 = get_National_State_Prior_estimates(year, new_vote_totals_states, dont_add_ON = True)
    State_Results_curr = State_Results_2016_2022.loc[State_Results_2016_2022['Election']==int(year),]
    if CAGO:
        if year in ['2019','2022']:
            Prior_estimates_df1.loc[:,'OTH'] = Prior_estimates_df1.iloc[:,-3:].sum(axis=1)
            Prior_estimates_df1 = Prior_estimates_df1.drop(columns=['ON','UAPP'])
            State_prior_df1.loc[:,'OTH'] = State_prior_df1.iloc[:,3:6].sum(axis=1)
            State_prior_df1 = State_prior_df1.drop(columns=['ON','UAPP'])
            State_Results_curr.loc[:,'OTH'] = State_Results_curr.iloc[:,3:6].sum(axis=1)
            State_Results_curr = State_Results_curr.drop(columns=['ON','UAPP'])
    State_Results_curr = State_Results_curr.drop('Election', axis=1).set_index('State')
    
    # convert both prior and results to ALR
    ref_col = 'COAL'
    State_Results_ALR = np.log(State_Results_curr.drop(columns=[ref_col]).div(State_Results_curr[ref_col], axis=0))
    State_Prior_ALR = np.log(State_prior_df1.drop(columns=[ref_col]).div(State_prior_df1[ref_col], axis=0))
    Prior_estimates_ALR_df1 = np.log(Prior_estimates_df1.drop(columns=[ref_col]).div(Prior_estimates_df1[ref_col], axis=0))
    import pdb;pdb.set_trace()

    Div_relative_weights = Div_relative_weights_dict[year]

    # add to corresponding divisions in states
    True_State_ALR_swings = State_Results_ALR - State_Prior_ALR
    Prior_estimates_ALR_df1.loc[:,'State'] = Div_relative_weights['State'].values
    merged = pd.merge(Prior_estimates_ALR_df1, True_State_ALR_swings, left_on = 'State',right_index = True, suffixes = ('','_state_swing'))
    merged.iloc[:,:3] += merged.iloc[:,-3:].values
    State_swing_ALR = merged.iloc[:,:3]

    Results_df = get_results_df(year).rename(columns={'Other':'OTH'})
    Results_df_ALR = np.log(Results_df.drop(columns=[ref_col]).div(Results_df[ref_col], axis=0))

    Electorate_residuals = Results_df_ALR - State_swing_ALR

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



# get alr-swing of
ref_col = 'COAL'
polling_alr = np.log(day_80_polling_avg.drop(columns=[ref_col]).div(day_80_polling_avg[ref_col], axis=0))
National_prior_alr = np.log(National_prior.drop(columns=[ref_col]).div(National_prior[ref_col], axis=0))

# get State deviations into a (10000, 8, 5) array
State_deviation_alr = State_Polls_Deviations_from_National_df_dict[election_year].set_index('State')
if election_year in ['2019','2022','2025']:
    State_deviation_alr.loc[:,'UAPP'] = 0.0 # add 0 deviation from National if no state polling!
    State_deviation_alr = State_deviation_alr[['ALP','GRN','ON','UAPP','OTH']]
State_deviation_alr_matrix = State_deviation_alr.values  # Convert to numpy array for easy broadcasting
State_deviation_alr_matrix = np.expand_dims(State_deviation_alr_matrix, axis=0)  # Add an extra dimension for broadcasting
State_deviation_alr_matrix_expanded = np.repeat(State_deviation_alr_matrix, n_samples, axis=0)  
State_deviation_alr_matrix_expanded = State_deviation_alr_matrix_expanded[np.arange(n_samples)[:, None], division_state_indices[None, :], :]

#State_prior_alr = np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))

# apply National Polling error
Simulated_national_result_alr = National_Simulated_election_error_expanded + polling_alr.values  # shape: [1M, 5]

# apply State Polling error
Simulated_state_polling_deviation = State_deviation_alr_matrix_expanded + State_Simulated_polling_error_centered_expanded
Simulated_State_Polling_Results = Simulated_national_result_alr + Simulated_state_polling_deviation


national_alr_swing = Simulated_national_result_alr - National_prior_alr.values  # shape: [1M, 5]


import pdb;pdb.set_trace()












# 2. Broadcast to all 8 states
# Shape: [1M, 8, 5] = [1M, 1, 5] + [1, 8, 5]
#State_prior_alr_array = State_prior_alr.to_numpy()
#uniform_state_alr = national_alr_swing[:, np.newaxis, :] + State_prior_alr_array[np.newaxis, :, :]

#national_alr_swing = Simulated_national_result_alr - National_prior_alr # new national swing - to be applied to all states!
#unform_state_alr = State_prior_alr + national_alr_swing.values # states adjusted by uniform swing!
#alr_to_simplex_vectorized(unform_state_alr,ref_col)[National_prior.columns.tolist()]


# Add state polling with state errors
#alr_to_simplex_vectorized(unform_state_alr,ref_col)[National_prior.columns.tolist()]



# transform prior votes to ALR, apply swing, back-transform
#Prior_estimates_alr = np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))

#Polling_estimates_alr = Prior_estimates_alr.add(national_alr_swing.iloc[0], axis=1)



#Polling_estimates = alr_to_simplex_vectorized(Polling_estimates_alr,ref_col)[National_prior.columns.tolist()]





# check that the aggregated results match the national polling average
merged_totals_polling = Polling_estimates.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
weights_polling = merged_totals_polling['new_vote_totals']/merged_totals_polling['new_vote_totals'].sum()
weighted_national_polling = (merged_totals_polling.iloc[:,:-1] * weights_polling.values[:,None]).sum().to_frame().T
import pdb;pdb.set_trace()

if election_year == '2016':
    Polling_estimates.loc[Polling_estimates.index=='Gorton',['GRN','OTH']] += np.array([1.0,-1.0]) * Polling_estimates.loc[Polling_estimates.index=='Gorton','OTH'].iloc[0]
    Polling_estimates_from_National = Polling_estimates
    import pdb;pdb.set_trace()


if election_year in ['2019','2022']:
    OTH_values = Polling_estimates.loc[Polling_estimates.index.isin(No_OTH_divisions),'OTH'].values
    Polling_estimates.loc[Polling_estimates.index.isin(No_OTH_divisions),['ON','OTH']] += np.array([OTH_values, - OTH_values]).T
    Prior_estimates_df.loc[Prior_estimates_df.index.isin(No_OTH_divisions),['ON','OTH']] += (0.005,-0.005)
    import pdb;pdb.set_trace()

    Polling_estimates_from_National = remove_ON_back_to_its_country(Prior_estimates_df, Polling_estimates, election_year)


import pdb;pdb.set_trace()
# Polling_estimates_from_National.to_csv(f"National_Polling_Estimates_{election_year}_Day_{day_of_interest}.csv", index=True)




