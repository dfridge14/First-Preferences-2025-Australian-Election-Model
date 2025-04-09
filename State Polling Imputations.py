import pandas as pd
import numpy as np
from itertools import product
import os,time
from datetime import datetime
from pathlib import Path

from scipy.special import logit


# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):

    print("\a")  # Rings the system bell
    os.system('echo -e "\\a"')  # Extra bell command for reliability
    os.system('tput bel')  # This forces the terminal to beep

    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler



base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)



StatePolling = pd.read_csv("StatePollingWeightedAverage_rel_to_Nat.csv", index_col=None)

StatePolling.loc[:,'Election_year'] = StatePolling['Election'].str.extract(r'(\d{4})')[0]
StatePolling.loc[:,'State'] = StatePolling['Election'].str.extract(r'\d{4}([A-Z]+)')[0]
StatePolling = StatePolling.drop('Election', axis = 1)


State_Polling_combined = StatePolling.copy()

State_Polling_combined.loc[State_Polling_combined['ON'].isna(),'ON'] = 0.0
State_Polling_combined.loc[:,'OTH'] += State_Polling_combined.loc[:,'ON'] # combine ON with Others
State_Polling_CAGO = State_Polling_combined.drop(['ON','ON_stds'], axis = 1)


# Get ALR values
def get_ALR_deviations_from_National(State_Polling_CAGO, ref_col, include_ON = 0):
    ref_col = 'COAL'

    if include_ON:
        State_Polling_CAGO_ALR = State_Polling_CAGO[['COAL','ALP','GRN','ON','OTH','Election_year','State']].drop(ref_col, axis = 1)
    else:
        State_Polling_CAGO_ALR = State_Polling_CAGO[['COAL','ALP','GRN','OTH','Election_year','State']].drop(ref_col, axis = 1)

    State_Polling_CAGO_ALR.iloc[:,:3+include_ON] = np.log(State_Polling_CAGO.iloc[:,:4+include_ON].drop(columns=[ref_col]).div(State_Polling_CAGO.iloc[:,:4+include_ON][ref_col], axis=0)) 

    National_poll_ALR = State_Polling_CAGO_ALR.loc[State_Polling_CAGO_ALR['State']== 'NAT',].reset_index(drop=True)
    State_poll_ALR = State_Polling_CAGO_ALR.loc[State_Polling_CAGO_ALR['State']!= 'NAT',].reset_index(drop=True)

    State_NAT_poll_merged = pd.merge(State_poll_ALR, National_poll_ALR, on='Election_year', how='left', suffixes=('','_Nat'))

    # get Nat - State differences
    alr_cols = [col for col in ['COAL','ALP','GRN','OTH'] if col != ref_col] if not include_ON else [col for col in ['COAL','ALP','GRN','ON','OTH'] if col != ref_col]
    for col in alr_cols:
        State_NAT_poll_merged[col + '_rel_to_Nat'] = State_NAT_poll_merged[col] - State_NAT_poll_merged[col + '_Nat']

    return State_NAT_poll_merged, State_Polling_CAGO_ALR

ref_col = 'COAL'
State_NAT_poll_merged, State_Polling_CAGO_ALR =  get_ALR_deviations_from_National(State_Polling_CAGO, ref_col)

for election_year in ['2016','2019','2022']:
    State_Correlation_Matrix = pd.read_csv(f"State_Correlation_Matrix_{election_year}.csv", index_col=0)

    # Get ALR Swings of existing polls
    National_State_ALR_df = pd.read_csv("National_State_ALR_df.csv", index_col = None)
    National_State_ALR_df_last = National_State_ALR_df.loc[National_State_ALR_df['Election_year'] == int(election_year) - 3,]
    National_State_ALR_df_last = National_State_ALR_df_last.loc[National_State_ALR_df_last['State']!='NAT',]

    State_Polling_CAGO_ALR_curr = State_Polling_CAGO_ALR.loc[State_Polling_CAGO['Election_year'] == election_year,]
    State_Polling_CAGO_ALR_curr = State_Polling_CAGO_ALR_curr.loc[State_Polling_CAGO_ALR_curr['State']!='NAT',]

    State_Polling_CAGO_ALR_swing = State_Polling_CAGO_ALR_curr.merge(National_State_ALR_df_last, on = ['State'], how = 'left', suffixes = ('','_prev'))
    State_Polling_CAGO_ALR_swing.iloc[:,:3] = State_Polling_CAGO_ALR_swing.iloc[:,:3] - State_Polling_CAGO_ALR_swing.iloc[:,-3:].values
    State_Polling_CAGO_ALR_swing = State_Polling_CAGO_ALR_swing.iloc[:,:5]

    def impute_missing_polling(missing_states, State_Correlation_Matrix, State_Polling_CAGO_ALR_swing, National_State_ALR_df_last):
        """
        Impute the missing polling based on swings and correlations with other states.
        
        state_name: The state for which polling needs to be imputed.
        correlation_matrix: The state-to-state correlation matrix.
        State_Polling_CAGO: The DataFrame containing state polling and swings.
        
        Returns the imputed polling value.
        """

        imputed_pollings = []
        other_states_swings = State_Polling_CAGO_ALR_swing.set_index('State').iloc[:,:3]


        for state in missing_states:
            # Get the correlations between this state and other states
            state_corr = State_Correlation_Matrix.loc[state]
            weights = state_corr[state_corr.index.isin(other_states_swings.index)]  # exclude self

            # Get the swing values from other states
            contributing_swings = other_states_swings.loc[weights.index]  # shape: (n_other_states, 3)

            # Weight the swing vectors by their correlation
            weighted_swings = contributing_swings.multiply(weights.values[:, np.newaxis])  # elementwise multiply

            # Normalize by total weight
            imputed_swing = weighted_swings.sum(axis=0) / weights.sum()

            # Get last election ALR for this state
            last_poll = National_State_ALR_df_last[National_State_ALR_df_last['State'] == state].iloc[:, 2:]

            # Add imputed swing to last election result
            imputed_polling = last_poll.values.flatten() + imputed_swing.values

            imputed_pollings.append(pd.DataFrame([imputed_polling], index = [state], columns = State_Polling_CAGO_ALR_swing.columns[:3]))

        # Combine results into a DataFrame
        imputed_polling_df = pd.concat(imputed_pollings)

        
        return imputed_polling_df


    missing_states = {'2016': ['NT','ACT'],'2019':['NT','ACT','TAS'],'2022':['NT','ACT','TAS'],'2025':['NT','ACT','TAS']}

    imputed_polling_ALR = impute_missing_polling(missing_states[election_year], State_Correlation_Matrix, State_Polling_CAGO_ALR_swing, National_State_ALR_df_last)
    existing_polling_ALR = State_Polling_CAGO_ALR_curr.set_index('State').iloc[:,:3]
    combined_polling_ALR = pd.concat([existing_polling_ALR,imputed_polling_ALR])

    #import pdb;pdb.set_trace()



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

    imputed_state_polling = alr_to_simplex_vectorized(imputed_polling_ALR, 'COAL').reset_index(names='State')
    imputed_state_polling.loc[:,'Election_year'] = election_year

    StatePolling = pd.concat((StatePolling, imputed_state_polling), ignore_index=True)

StatePolling.to_csv("Full_State_Inputed_Polling.csv", index=False)

State_Polling_Deviations_from_National = get_ALR_deviations_from_National(StatePolling, ref_col, include_ON = 1)[0]
State_Polling_Deviations_from_National.iloc[:,:4] = State_Polling_Deviations_from_National.iloc[:,-4:].values
State_Polling_Deviations_from_National = State_Polling_Deviations_from_National.iloc[:,:6].sort_values(by=['Election_year','State'])
State_Polling_Deviations_from_National.loc[:,'Election_year'] = State_Polling_Deviations_from_National.loc[:,'Election_year'].astype(str)

import pdb;pdb.set_trace()

State_Polling_Deviations_from_National.to_csv("State_Polling_Deviations_from_National.csv", index=False)
