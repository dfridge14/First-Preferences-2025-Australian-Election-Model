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

election_year = '2019'

n_samples = 1000000
National_Polling_error_ALR_cov = pd.read_csv(f"PollingErrorALRCovarianceNational{election_year}.csv", index_col=0)
National_Simulated_polling_error = np.random.multivariate_normal(mean = np.zeros(len(National_Polling_error_ALR_cov)), cov = National_Polling_error_ALR_cov.values, size=n_samples)

National_Election_error_ALR_cov = pd.read_csv(f"ElectionErrorALRCovarianceNational{election_year}.csv", index_col=0)
National_Simulated_election_error = np.random.multivariate_normal(mean = np.zeros(len(National_Election_error_ALR_cov)), cov = National_Election_error_ALR_cov.values, size=n_samples)


last_election_year = str(int(election_year) - 3)



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



if election_year in ['2016','2019','2022']:
    Prior_estimates_df = get_Prior_estimates_df(election_year).rename(columns={'Other':'OTH'}) # adds ON to every seat if no ON (for 2019 and 2022)




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
    else:
        No_OTH_divisions = Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,].index
        Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,['ON','OTH']] += (-0.005,+0.005)




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



day_80_polling_avg_dict = {'2016': pd.DataFrame([[0.412262, 0.351972, 0.105693, 0.130074]], columns = ['COAL','ALP','GRN','OTH']), \
                      '2019':pd.DataFrame([[0.384782, 0.36451, 0.095751, 0.035, 0.031, 0.088957]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
                      '2022':pd.DataFrame([[0.355905, 0.362643, 0.118432, 0.0383,0.0244,0.10032,]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
                      '2025':[]}

day_80_polling_avg = day_80_polling_avg_dict[election_year]/ day_80_polling_avg_dict[election_year].sum(axis=1)[0]



# get alr-swing of
ref_col = 'COAL'
polling_alr = np.log(day_80_polling_avg.drop(columns=[ref_col]).div(day_80_polling_avg[ref_col], axis=0))
National_prior_alr = np.log(National_prior.drop(columns=[ref_col]).div(National_prior[ref_col], axis=0))
State_prior_alr = np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))



# apply National Polling error
Simulated_national_result_alr = polling_alr.values + National_Simulated_polling_error  # shape: [1M, 5]
national_alr_swing = Simulated_national_result_alr - National_prior_alr.values  # shape: [1M, 5]

# 2. Broadcast to all 8 states
# Shape: [1M, 8, 5] = [1M, 1, 5] + [1, 8, 5]
State_prior_alr_array = State_prior_alr.to_numpy()
uniform_state_alr = national_alr_swing[:, np.newaxis, :] + State_prior_alr_array[np.newaxis, :, :]

#national_alr_swing = Simulated_national_result_alr - National_prior_alr # new national swing - to be applied to all states!
#unform_state_alr = State_prior_alr + national_alr_swing.values # states adjusted by uniform swing!
#alr_to_simplex_vectorized(unform_state_alr,ref_col)[National_prior.columns.tolist()]


# Add state polling with state errors
#alr_to_simplex_vectorized(unform_state_alr,ref_col)[National_prior.columns.tolist()]



# transform prior votes to ALR, apply swing, back-transform
#Prior_estimates_alr = np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))

#Polling_estimates_alr = Prior_estimates_alr.add(national_alr_swing.iloc[0], axis=1)



#Polling_estimates = alr_to_simplex_vectorized(Polling_estimates_alr,ref_col)[National_prior.columns.tolist()]

import pdb;pdb.set_trace()




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




