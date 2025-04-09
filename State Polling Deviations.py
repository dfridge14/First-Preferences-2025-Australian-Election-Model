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




Type = 'Election_swing' # 'Election_swing' 'Polling'
ref_col = 'COAL'

remove_election_year = 1



def extend_corr_matrix_to_5x5(corr_matrix_3x3, var4, var5, cov_matrix_3x3, ref_col):
    # Extends to 5x5 corr and covMs through rickety imputation, checking for positive definiteness and adding column names, reutrning 5x5 df


    R5 = np.pad(corr_matrix_3x3, ((0, 2), (0, 2)), mode='constant', constant_values=0)

    ON_UAPP_CORRELATION = 0.5
    ALP_GRN_CORRELATION_LOSS = 0.3

    # ON/ALP; UAPP/ALP
    R5[0,3] = R5[0,2] - ALP_GRN_CORRELATION_LOSS
    R5[3,0] = R5[0,2] - ALP_GRN_CORRELATION_LOSS
    R5[0,4] = R5[0,2] - ALP_GRN_CORRELATION_LOSS
    R5[4,0] = R5[0,2] - ALP_GRN_CORRELATION_LOSS

    # ON/GRN; UAPP/GRN
    R5[1,3] = R5[1,2] - ALP_GRN_CORRELATION_LOSS
    R5[3,1] = R5[1,2] - ALP_GRN_CORRELATION_LOSS
    R5[1,4] = R5[1,2] - ALP_GRN_CORRELATION_LOSS
    R5[4,1] = R5[1,2] - ALP_GRN_CORRELATION_LOSS

    # ON/OTH; UAPP/OTH
    R5[2,3] = R5[1,2]
    R5[3,2] = R5[1,2]
    R5[2,4] = R5[1,2]
    R5[4,2] = R5[1,2]

    # ON/UAPP
    R5[4,3] = ON_UAPP_CORRELATION
    R5[3,4] = ON_UAPP_CORRELATION

    R5[3,3] = 1
    R5[4,4] = 1

    # construct CovM from 3x3CovM, correlation matrix and variances of 4/5

    std_devs = np.zeros(5)
    std_devs[:3] = np.sqrt(np.diag(cov_matrix_3x3))
    std_devs[3] = np.sqrt(var4)
    std_devs[4] = np.sqrt(var5)

    S = np.diag(std_devs)
    C5 = S @ R5 @ S

    idx = [0, 1, 3, 4, 2]  # new order: move index 2 to the end
    R5 = R5[np.ix_(idx, idx)]
    C5 = C5[np.ix_(idx, idx)]

    cols = [p for p in ['COAL','ALP', 'GRN', 'ON', 'UAPP','OTH'] if p != ref_col]

    # Create labeled DataFrame
    R5 = pd.DataFrame(R5, index=cols, columns=cols)
    C5 = pd.DataFrame(C5, index=cols, columns=cols)


    # eignenvalue check!
    assert np.all(np.linalg.eigvalsh(C5) > 0)



    return R5, C5

# impute results about ON/UAPP form relative general polling error
National_Polling_covm_2019 = pd.read_csv("PollingErrorALRCovarianceNational2019.csv", index_col=0)
National_Polling_covm_2022 = pd.read_csv("PollingErrorALRCovarianceNational2022.csv", index_col=0)
National_Polling_covm_2025 = pd.read_csv("PollingErrorALRCovarianceNational2025.csv", index_col=0)
National_Election_covm_2019 = pd.read_csv("ElectionErrorALRCovarianceNational2019.csv", index_col=0)
National_Election_covm_2022 = pd.read_csv("ElectionErrorALRCovarianceNational2022.csv", index_col=0)
National_Election_covm_2025 = pd.read_csv("ElectionErrorALRCovarianceNational2025.csv", index_col=0)

ON_OTH_VAR_ratio_Polling = {'2019': National_Polling_covm_2019.iloc[3,3]/National_Polling_covm_2019.iloc[4,4], \
                    '2022': National_Polling_covm_2022.iloc[3,3]/National_Polling_covm_2022.iloc[4,4], \
                    '2025': National_Polling_covm_2025.iloc[3,3]/National_Polling_covm_2025.iloc[4,4]}

ON_OTH_VAR_ratio_Election = {'2019': National_Election_covm_2019.iloc[3,3]/National_Election_covm_2019.iloc[4,4], \
                    '2022': National_Election_covm_2022.iloc[3,3]/National_Election_covm_2022.iloc[4,4], \
                    '2025': National_Election_covm_2025.iloc[3,3]/National_Election_covm_2025.iloc[4,4]}



if Type == 'Election_swing':


    party_map = {
        'ALP': 'ALP',
        'CLR': 'ALP',
        'LP': 'COAL',
        'NP': 'COAL',
        'LNP': 'COAL',
        'LNQ': 'COAL',
        'CLP': 'COAL',
        'GRN': 'GRN',
        'GVIC': 'GRN'
    }


    # first, get the state results for each election


    election_state_result_list = []
    for election_year in [2004,2007,2010,2013,2016,2019,2022]:
        #for state in ['NSW','VIC','QLD','WA','SA','TAS','NT','ACT']:
        StateFirstPrefs = pd.read_csv(f"{election_year}HouseFirstPrefsByStateByParty.csv", skiprows=1, index_col=None).rename(columns={'StateAb':'State'})
        StateFirstPrefs = StateFirstPrefs.loc[StateFirstPrefs['PartyAb'].isin(['ALP','CLR','LP','NP','CLP','LNP','LNQ','GRN','GVIC']),['State','PartyAb','TotalPercentage']]
        StateFirstPrefs['PartyAb'] = StateFirstPrefs['PartyAb'].replace(party_map)

        grouped = StateFirstPrefs.groupby(['State', 'PartyAb'], as_index=False)['TotalPercentage'].sum()
        pivoted = grouped.pivot(index='State', columns='PartyAb', values='TotalPercentage').fillna(0).reset_index()
        pivoted.loc[:,'Election_year'] = election_year
        election_state_result_list.append(pivoted)

    Federal_State_1996_01 = pd.read_csv("OldFederalStateResults.csv", index_col=None)
    Nat_polls_2007_22 = pd.read_csv("NationalElectionResults.csv", index_col=None).drop('OTH', axis=1).rename(columns={'Election':'Election_year'})
    Nat_polls_2007_22.loc[:,'State'] = 'NAT'
    Nat_polls_2007_22.iloc[:,:3] *= 100 # same format as percentage

    National_State_df = pd.concat(election_state_result_list + [Federal_State_1996_01,Nat_polls_2007_22], ignore_index=True)
    National_State_df.loc[:,['COAL','ALP','GRN']] /= 100 # convert to proportions
    National_State_df.loc[:,'OTH'] = 1-National_State_df.loc[:,['COAL','ALP','GRN']].sum(axis=1)
    National_State_df = National_State_df[['Election_year','State','COAL','ALP','GRN','OTH']].sort_values(by=['Election_year','State'])
    import pdb;pdb.set_trace()


    # convert to ALR
    National_State_ALR_df = National_State_df.copy().drop(ref_col, axis=1)
    National_State_ALR_df.iloc[:,2:] = np.log(National_State_df.iloc[:,2:].drop(columns=[ref_col]).div(National_State_df.iloc[:,2:][ref_col], axis=0))

    # Save as csv file (Note: currently, years are INT type)
    National_State_ALR_df.to_csv("National_State_ALR_df.csv", index = False)

    National_ALR_df = National_State_ALR_df.loc[National_State_ALR_df['State']=='NAT',] 
    National_ALR_df.loc[:,'Election_year'] = National_ALR_df['Election_year']
    State_ALR_df =  National_State_ALR_df.loc[National_State_ALR_df['State']!='NAT',] 
    State_NAT_merged = pd.merge(State_ALR_df, National_ALR_df, on='Election_year', how='left', suffixes=('','_Nat'))

    # Take 1 difference between elections - equivalent to calculating change in swing

    alr_cols = [col for col in ['COAL','ALP','GRN','OTH'] if col != ref_col]
    # get Nat - State differences
    for col in alr_cols:
        State_NAT_merged[col + '_rel_to_Nat'] = State_NAT_merged[col] - State_NAT_merged[col + '_Nat']

    ALR_deviation_cols = [col + '_rel_to_Nat' for col in alr_cols]
    State_NAT_merged = State_NAT_merged.sort_values(by=['State', 'Election_year'])

    differenced_df = State_NAT_merged.groupby('State')[['Election_year'] + ALR_deviation_cols].apply(lambda group: group.set_index('Election_year').diff().dropna()).reset_index()

    differenced_df_cleaned = differenced_df.loc[~((np.abs(differenced_df['GRN_rel_to_Nat'])>0.5) & (differenced_df['Election_year']<2004)),]
    differenced_df_cleaned = differenced_df_cleaned.loc[~((np.abs(differenced_df_cleaned['OTH_rel_to_Nat'])>0.9) ),] # & (differenced_df_cleaned['State']!= 'SA') - remove due to no longer such a state effect in play!


    import pdb;pdb.set_trace()

    # Now, remove current and all previous years for 2016,2019,2022,2025 from estimation for validation

    for curr_election_year in ['2016','2019','2022','2025']:
        election_year_to_remove = curr_election_year if remove_election_year else ' '
        differenced_df_cleaned_curr = differenced_df_cleaned.loc[differenced_df_cleaned['Election_year']<int(curr_election_year),].iloc[:,2:]

        curr_year_Election_swing_covM = differenced_df_cleaned_curr.cov()
        if curr_election_year == '2016':
            curr_year_Election_swing_covM = pd.DataFrame(curr_year_Election_swing_covM.values, index = ['ALP','GRN','OTH'], columns = ['ALP','GRN','OTH'])
        else:
            cov_curr = differenced_df_cleaned_curr.cov()
            corr_curr = differenced_df_cleaned_curr.corr()
            ON_var = ON_OTH_VAR_ratio_Election[curr_election_year] * cov_curr.iloc[-1,-1]
            curr_year_Election_swing_covM = extend_corr_matrix_to_5x5(corr_curr, ON_var, ON_var, cov_curr, ref_col)[1]

        

        import pdb;pdb.set_trace()

        curr_year_Election_swing_covM.to_csv(f'ElectionErrorALRCovarianceStateDeviation{curr_election_year}.csv', index = True)





elif Type == 'Polling':

    State_Polling = pd.read_csv("StatePollingWeightedAverage_rel_to_Nat.csv", index_col=None)
    State_Results = pd.read_csv("StateFederalResults_rel_to_Nat.csv", index_col=None)

    # get separate cols for Year and State to make it easier to distirbute
    State_Polling.loc[:,'Year'] = State_Polling['Election'].str.extract(r'(\d{4})')
    State_Polling.loc[:,'Scope'] = State_Polling['Election'].str.extract(r'(\D+)$')
    State_Results.loc[:,'Year'] = State_Results['Election'].str.extract(r'(\d{4})')
    State_Results.loc[:,'Scope'] = State_Results['Election'].str.extract(r'(\D+)$')


    # combine ON into OTH for correlation matrix

    State_Polling_combined = State_Polling.copy()
    State_Results_combined = State_Results.copy()

    State_Polling_combined.loc[State_Polling_combined['ON'].isna(),'ON'] = 0.0
    State_Results_combined.loc[State_Results_combined['ON'].isna(),'ON'] = 0.0


    State_Polling_combined.loc[:,'OTH'] += State_Polling_combined.loc[:,'ON'] # combine ON with Others
    State_Polling_CAGO = State_Polling_combined[['Year','Scope','COAL','ALP','GRN','OTH']]

    State_Results_combined.loc[:,'OTH'] += State_Results_combined.loc[:,'ON'] # combine ON with Others
    State_Results_CAGO = State_Results_combined[['Year','Scope','COAL','ALP','GRN','OTH']]


    # convert to ALR:
    State_Polling_CAGO_ALR = State_Results_CAGO.copy().drop(ref_col, axis=1)
    State_Results_CAGO_ALR = State_Results_CAGO.copy().drop(ref_col, axis=1)

    State_Polling_CAGO_ALR.iloc[:,2:] = np.log(State_Polling_CAGO.iloc[:,2:].drop(columns=[ref_col]).div(State_Polling_CAGO.iloc[:,2:][ref_col], axis=0))
    State_Results_CAGO_ALR.iloc[:,2:] = np.log(State_Results_CAGO.iloc[:,2:].drop(columns=[ref_col]).div(State_Results_CAGO.iloc[:,2:][ref_col], axis=0))


    # separate national from states
    National_poll_ALR = State_Polling_CAGO_ALR.loc[State_Polling_CAGO_ALR['Scope']== 'NAT',].reset_index(drop=True)
    State_poll_ALR = State_Polling_CAGO_ALR.loc[State_Polling_CAGO_ALR['Scope']!= 'NAT',].reset_index(drop=True)

    National_result_ALR = State_Results_CAGO_ALR.loc[State_Results_CAGO_ALR['Scope']== 'NAT',].reset_index(drop=True)
    State_result_ALR = State_Results_CAGO_ALR.loc[State_Results_CAGO_ALR['Scope']!= 'NAT',].reset_index(drop=True)


    # get state - national, for each of Poll and results
    State_NAT_poll_merged = pd.merge(State_poll_ALR, National_poll_ALR, on='Year', how='left', suffixes=('','_Nat'))
    State_NAT_result_merged = pd.merge(State_result_ALR, National_result_ALR, on='Year', how='left', suffixes=('','_Nat'))

    # get Nat - State differences
    alr_cols = [col for col in ['COAL','ALP','GRN','OTH'] if col != ref_col]
    for col in alr_cols:
        State_NAT_poll_merged[col + '_rel_to_Nat'] = State_NAT_poll_merged[col] - State_NAT_poll_merged[col + '_Nat']
        State_NAT_result_merged[col + '_rel_to_Nat'] = State_NAT_result_merged[col] - State_NAT_result_merged[col + '_Nat']

    

    #import pdb;pdb.set_trace()


    # weights based on sample sizes of state polls:
    std_weights_df = State_Polling_combined.loc[State_Polling_combined['Scope']!= 'NAT',].iloc[:,[6,7,8,10,11]] #stds, combining ON and OTH
    means_df = State_Polling_combined.loc[State_Polling_combined['Scope']!= 'NAT',][['COAL','ALP','GRN','OTH','Year']]

    std_weights_df.iloc[-6:-1,:4] =  std_weights_df.iloc[:5,:4]*1.5 # arbitrary scaling of 2016 results (taken off Pollbludger's May 2016 aggregate, so more variable!)
    std_weights_df.iloc[-1:,:4] = std_weights_df.iloc[-2:-1,:4] * 2 # TAS weighting


    def vectorized_log_ratio_precision(means_df, tau_array, ref_col='COAL', numerator_indices=[1,2,3], numerator_list=['ALP', 'GRN', 'OTH'], rho=0.0): # ChatGPT written
        """
        Vectorized calculation of log-ratio precisions for each row of means and tau dataframes.
        
        Args:
            means_df: DataFrame of shape (n, d) — means of proportions
            tau_df:   DataFrame of shape (n, d) — precisions (1/variance)
            denominator: str — name of the denominator party (e.g., 'COAL')
            numerator_list: list of str — numerator parties
            rho: assumed correlation (scalar or array of same shape as output, default 0)
        
        Returns:
            A DataFrame of shape (n, len(numerator_list)) with log-ratio precisions
        """
        mu_x = means_df[numerator_list].values       # shape (n, k)
        mu_y = means_df[ref_col].values[:, None] # shape (n, 1)
        
        tau_x = tau_array[:,numerator_indices]       # shape (n, k)
        tau_y = tau_array[:,[0]]  # shape (n, 1)
        
        term1 = 1 / (mu_x**2 * tau_x)
        term2 = 1 / (mu_y**2 * tau_y)
        term3 = 2 * rho / (mu_x * mu_y * np.sqrt(tau_x * tau_y))

        var_log_ratio = term1 + term2 - term3
        precision_log_ratio = 1 / var_log_ratio
        
        return pd.DataFrame(precision_log_ratio, columns=[f"{p}_rel_to_Nat" for p in numerator_list], index=means_df.index)




    for curr_election_year in ['2016','2019','2022','2025']:
        election_year_to_remove = curr_election_year if remove_election_year else ' '

        # remove data from current election
        State_NAT_result_merged_curr = State_NAT_result_merged.loc[State_NAT_result_merged['Year']!=curr_election_year,].iloc[:,-3:].reset_index(drop=True)
        State_NAT_poll_merged_curr = State_NAT_poll_merged.loc[State_NAT_poll_merged['Year']!=curr_election_year,].iloc[:,-3:].reset_index(drop=True)


        means_df_curr = means_df.loc[means_df['Year']!= curr_election_year,].drop('Year', axis = 1)
        std_weights_df_curr = std_weights_df.loc[std_weights_df['Year']!= curr_election_year,].drop('Year', axis = 1)
        std_weights_array = 1/std_weights_df_curr.values

        # convert precisions to ALR, using formula 1 / (prec_alp * mu_alp**2) + 1 / (prec_coal * mu_coal**2) - Delta Method approximation (for simplicity, use independence)
        precision_weights = vectorized_log_ratio_precision(means_df_curr, std_weights_array, ref_col='COAL', numerator_list=['ALP', 'GRN', 'OTH'], rho=0.0)

        # normalise weights
        observation_precision_weights = precision_weights.mean(axis=1).reset_index(drop=True)
        weight_sum = observation_precision_weights.sum()
        observation_precision_weights /= weight_sum

        # Take 1 difference between state polls and results' relative to the nation
        Deviation_swings = State_NAT_result_merged_curr - State_NAT_poll_merged_curr
        Deviation_swings_centered = Deviation_swings - Deviation_swings.mean()
        weighted_Deviation_swings_centered = Deviation_swings_centered.mul(np.sqrt(observation_precision_weights), axis=0)

        # DoF adjustment: 
        cov_matrix = (weighted_Deviation_swings_centered.T @ weighted_Deviation_swings_centered)
        correction = 1.0 / (1.0 - np.sum(observation_precision_weights**2))
        cov_matrix *= correction

        # Get Correlation matrix
        stddev = np.sqrt(np.diag(cov_matrix))
        outer_stddev = np.outer(stddev, stddev) # Outer product of stddevs
        corr_matrix = cov_matrix / outer_stddev            

        if curr_election_year == '2016':
            curr_year_Polling_swing_covM = pd.DataFrame(cov_matrix.values, index = ['ALP','GRN','OTH'], columns = ['ALP','GRN','OTH'])
        else:
            cov_curr = cov_matrix
            corr_curr = corr_matrix
            ON_var = ON_OTH_VAR_ratio_Polling[curr_election_year] * cov_curr.iloc[-1,-1]
            curr_year_Polling_swing_covM = extend_corr_matrix_to_5x5(corr_curr, ON_var, ON_var, cov_curr, ref_col)[1]

        import pdb;pdb.set_trace()

        curr_year_Polling_swing_covM.to_csv(f'PollingErrorALRCovarianceStateDeviation{curr_election_year}.csv', index = True)



    import pdb;pdb.set_trace()




import pdb;pdb.set_trace()



# I get this correlation structure! Small correlations!
#                ALP_rel_to_Nat  GRN_rel_to_Nat  OTH_rel_to_Nat
#ALP_rel_to_Nat        1.000000        0.080057        0.119370
#GRN_rel_to_Nat        0.080057        1.000000        0.052068
#OTH_rel_to_Nat        0.119370        0.052068        1.000000


#                ALP_rel_to_Nat  GRN_rel_to_Nat  OTH_rel_to_Nat
#ALP_rel_to_Nat        0.014288        0.001556        0.004268
#GRN_rel_to_Nat        0.001556        0.026433        0.002532
#OTH_rel_to_Nat        0.004268        0.002532        0.089487



import pdb;pdb.set_trace()
