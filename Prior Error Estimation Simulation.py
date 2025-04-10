import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gammaln  # Correct import for the gamma function
from scipy.stats import multivariate_t, multivariate_normal, dirichlet



# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler

base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)





start = time.time()

def group_into_Fundamentals_Categories(party_votes_shares_df, div, is_Other = True):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'COAL','COALNP','COALLP','LP','NP','CLP','LNP','LNQ'}
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

def get_prior_and_results_df(data_year):
    next_year = str(int(data_year)+3)
    Actual_results = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    # COUntNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


    # Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
    Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
    Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
    Actual_results.loc[Actual_results['PartyAb']=='GVIC','PartyAb'] = 'GRN'
    Actual_results.loc[Actual_results['PartyAb']=='CLR','PartyAb'] = 'ALP'

    # rename IND to INDX by order

    target = 'IND'

    Actual_results_dict = {}

    Prior_estimates_df = pd.read_csv(f"Fundamentals_Votes_For_{next_year}.csv", index_col = None)



    Prior_estimates_dict = {
        div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
        for div, group in Prior_estimates_df.groupby("div_nm")
    }
    import pdb;pdb.set_trace()

    Fundamentals_results_list = []
    Fundamentals_estimate_list = []


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

        Fundamentals_results_list.append(group_into_Fundamentals_Categories(Actual_results_dict[div], div))
        Fundamentals_estimate_list.append(group_into_Fundamentals_Categories(Prior_estimates_dict[div], div))



    Fundamentals_results_df = pd.concat(Fundamentals_results_list)/100
    Fundamentals_estimate_df = pd.concat(Fundamentals_estimate_list)

    # Gorton 2016 adjustment: add 0.001 from GRN to Other
    Fundamentals_results_df.loc[Fundamentals_results_df['Other'] == 0.0,['GRN','Other']] += (-0.01,0.01)
    Fundamentals_estimate_df.loc[Fundamentals_estimate_df['Other'] == 0.0,['GRN','Other']] += (-0.01,0.01)


    Fundamentals_results_df = Fundamentals_results_df.div(Fundamentals_results_df.sum(axis=1), axis=0).sort_index()
    Fundamentals_estimate_df = Fundamentals_estimate_df.div(Fundamentals_estimate_df.sum(axis=1), axis=0).sort_index()  

    Fundamentals_results_df.index = str(int(data_year)+3) +  Fundamentals_results_df.index
    Fundamentals_estimate_df.index = str(int(data_year)+3) +  Fundamentals_estimate_df.index


    return Fundamentals_results_df, Fundamentals_estimate_df

def get_Fundamentals_alr_swings_and_cov(ref_col):

    Fundamentals_results_list = []
    Fundamentals_estimate_list = []


    for data_year in ['2013','2016','2019']:
        Fundamentals_results_df,Fundamentals_estimate_df = get_prior_and_results_df(data_year)
        Fundamentals_results_list.append(Fundamentals_results_df)
        Fundamentals_estimate_list.append(Fundamentals_estimate_df)

    full_Fundamentals_results_df = pd.concat(Fundamentals_results_list)
    full_Fundamentals_estimate_df = pd.concat(Fundamentals_estimate_list)

    # center the natural swings to avoid bias - we assume swings are generally unbiased - that we cannot predict direction 3 years prior!
    swings = full_Fundamentals_results_df - full_Fundamentals_estimate_df
    #swings_centered = (swings - swings.mean()).sum(axis=1)
    full_Fundamentals_estimate_df = full_Fundamentals_estimate_df + swings.mean()

    import pdb; pdb.set_trace()

    results_alr = np.log(full_Fundamentals_results_df.drop(columns=[ref_col]).div(full_Fundamentals_results_df[ref_col], axis=0))
    estimate_alr = np.log(full_Fundamentals_estimate_df.drop(columns=[ref_col]).div(full_Fundamentals_estimate_df[ref_col], axis=0))
    alr_swing = results_alr - estimate_alr

    alr_swing_cov = alr_swing.cov()


    print(alr_swing.cov())
    print((full_Fundamentals_results_df - full_Fundamentals_estimate_df).mean()) # should be 0 due to centralisation adjustment

    return alr_swing, alr_swing_cov, full_Fundamentals_estimate_df, estimate_alr

def get_Polling_alr_swings_and_cov(ref_col, full_Fundamentals_results_df, ON_and_UAPP = False):
    ### converts polling estimes df into alr swings between alr of polling estimates and results df

    Polling_estimate_list = []

    if ON_and_UAPP:
        # estimates for only 2019 and 2022

        for election_year in ['2019','2022']:
            Polling_estimates = pd.read_csv(f"National_Polling_Estimates_{election_year}_Day_80.csv", index_col = 0)
            Polling_estimate_list.append(Polling_estimates)

    else:
        for election_year in ['2016','2019','2022']:

            # reduce to set of 4 to examine polling error
            Polling_estimates = pd.read_csv(f"National_Polling_Estimates_{election_year}_Day_80.csv", index_col = 0)
            if election_year != '2016':
                Polling_estimates.loc[:,'OTH'] = Polling_estimates.iloc[:,3:].sum(axis=1)
                Polling_estimates = Polling_estimates.drop(['ON','UAPP'], axis = 1)

            Polling_estimates.index = election_year + Polling_estimates.index
            Polling_estimate_list.append(Polling_estimates)


    

    full_Polling_estimate_df = pd.concat(Polling_estimate_list)

    # center the natural swings to avoid bias - we assume swings are generally unbiased - that we cannot predict direction 3 years prior!

    full_Fundamentals_results_df.rename(columns={'Other':'OTH'}, inplace = True)

    swings = full_Fundamentals_results_df.rename(columns={'Other':'OTH'}) - full_Polling_estimate_df
    #swings_centered = (swings - swings.mean()).sum(axis=1)
    full_Polling_estimate_df = full_Polling_estimate_df + swings.mean()

    results_alr = np.log(full_Fundamentals_results_df.drop(columns=[ref_col]).div(full_Fundamentals_results_df[ref_col], axis=0))
    estimate_alr = np.log(full_Polling_estimate_df.drop(columns=[ref_col]).div(full_Polling_estimate_df[ref_col], axis=0))
    alr_swing = results_alr - estimate_alr

    alr_swing_cov = alr_swing.cov()


    print(alr_swing.cov())
    print((full_Fundamentals_results_df - full_Fundamentals_estimate_df).mean()) # should be 0 due to centralisation adjustment

    return alr_swing, alr_swing_cov, full_Polling_estimate_df, estimate_alr


Fundamentals_results_list = []
for data_year in ['2013','2016','2019']:
    Fundamentals_results_df = get_prior_and_results_df(data_year)[0]
    Fundamentals_results_list.append(Fundamentals_results_df)
full_Fundamentals_results_df = pd.concat(Fundamentals_results_list)


ref_col = 'COAL'
ON_and_UAPP = False
alr_swing, alr_swing_cov, full_Fundamentals_estimate_df, estimate_alr = get_Fundamentals_alr_swings_and_cov(ref_col)
alr_swing_poll, alr_swing_poll_cov, full_Polling_estimate_df, estimate_alr = get_Polling_alr_swings_and_cov(ref_col, full_Fundamentals_results_df, ON_and_UAPP)

#simulated_alr_swings = multivariate_normal.rvs(mean=np.zeros(alr_swing.shape[1]), cov=alr_swing_cov, size=451)

#predicted_alr = estimate_alr + simulated_alr_swings



###################################################################################################### PLAY AROUND WITH R_ELECTORATES CORRELATION MATRIX#######################


import scipy.linalg
from scipy.linalg import eigh
from scipy.linalg import sqrtm, inv

# Simulated inputs: Replace these with your actual data
np.random.seed(42)

# 3x3 ALR covariance matrix (estimate from data)
ALR_cov = alr_swing_cov




R_2016 = pd.read_csv(f"Electorate_Correlation_Matrix_2016.csv", index_col = 0)
R_2019 = pd.read_csv(f"Electorate_Correlation_Matrix_2019.csv", index_col = 0)
R_2022 = pd.read_csv(f"Electorate_Correlation_Matrix_2022.csv", index_col = 0)






def regularize_correlation_matrix(R, alpha=0, beta=0.3, min_eigen=0.05, max_eigen=15):
    """
    Regularizes a correlation matrix by:
    - Shrinking eigenvalues to avoid extreme variance domination.
    - Moving extreme correlations toward 0.5.
    
    Parameters:
    - R (np.array): 150x150 correlation matrix.
    - alpha (float): Strength of shrinkage toward identity.
    - beta (float): Strength of push toward 0.5.
    - min_eigen (float): Minimum eigenvalue threshold.
    - max_eigen (float): Maximum eigenvalue threshold.

    Returns:
    - np.array: Regularized correlation matrix.
    """
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    
    # Clip eigenvalues within reasonable bounds
    eigvals = np.clip(eigvals, min_eigen, max_eigen)
    
    # Reconstruct the matrix
    R_shrinked = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Shrink towards identity (reducing dominance of strong correlations)
    R_regularized = (1 - alpha) * R_shrinked + alpha * np.eye(R.shape[0])
    
    # Push correlations toward 0.5 to avoid extremes
    R_final = R_regularized + beta * (0.5 - R_regularized)
    
    # Ensure symmetry and valid range
    R_final = np.clip((R_final + R_final.T) / 2, -1, 1)

    np.fill_diagonal(R_final, 1.0)

    return R_final


# Mahalanobis distance check - does R fit the data point reasonably (in ALR)?
x = alr_swing.loc[alr_swing.index.str.startswith('2016'),]
y = alr_swing.loc[alr_swing.index.str.startswith('2019'),]
z = alr_swing.loc[alr_swing.index.str.startswith('2022'),]

# example heuristic tests
# x.to_numpy().flatten() @ inv(np.kron(R_electorates_regularized,alr_swing_cov)) @ x.to_numpy().flatten()
#arr = regularize_correlation_matrix(R_2022, alpha=0, beta=0, min_eigen=0.05, max_eigen=10.0)
# arr[arr<1].max()
#0.5538609162965131
# arr = regularize_correlation_matrix(R_2022, alpha=0, beta=0, min_eigen=0.05, max_eigen=20.0)
# arr[arr<1].max()
#0.5925031228978586
# z1.to_numpy().flatten() @ inv(np.kron(regularize_correlation_matrix(R_2022, alpha=0, beta=0, min_eigen=0.05, max_eigen=20.0),alr_swing_poll_cov)) @ z1.to_numpy().flatten()
#618.6528729063027
# z1.to_numpy().flatten() @ inv(np.kron(regularize_correlation_matrix(R_2022, alpha=0, beta=0, min_eigen=0.05, max_eigen=10.0),alr_swing_poll_cov)) @ z1.to_numpy().flatten()
#553.3188666765652
# z1.to_numpy().flatten() @ inv(np.kron(regularize_correlation_matrix(R_2022, alpha=0, beta=0, min_eigen=0.05, max_eigen=20.0),alr_swing_poll_cov)) @ z1.to_numpy().flatten()
# 618.6528729063027

# Want distance not too far from 450; 600 is acceptable (use discretion as only 6 swing data points - fundamentals/polls each of 3 years, and even these are correlated)
# try using alpha = 0, beta = 0.3, min_eigen = 0.05, max_eigen = 15

# Example usage with your 150x150 matrix
R_electorates_regularized_dict = {}
for election_year in ['2016','2019','2022']:
    R_electorates = pd.read_csv(f"Electorate_Correlation_Matrix_{election_year}.csv", index_col = 0)
    R_electorates_regularized = regularize_correlation_matrix(R_electorates)
    R_electorates_regularized_dict[election_year] = pd.DataFrame(R_electorates_regularized, index = R_electorates.index, columns=R_electorates.columns)





#import pdb;pdb.set_trace()

# 150x150 electorate correlation matrix (estimated from data)
year = '2016'
R_electorates = pd.read_csv(f"Electorate_Correlation_Matrix_{year}.csv", index_col = 0)

# --- Independent Model (Block Diagonal 450x450) ---
independent_cov = np.kron(np.eye(150), ALR_cov)  # Block diagonal (no cross-electorate correlation)
independent_samples = np.random.multivariate_normal(np.zeros(450), independent_cov, size=10000)

# --- Correlated Model (Kronecker Product 450x450) ---
#kronecker_cov = np.kron(R_electorates_regularized, ALR_cov)  # Symmetric Kronecker product
kronecker_cov = np.kron(R_electorates, alr_swing_cov)
correlated_samples = np.random.multivariate_normal(np.zeros(450), kronecker_cov, size=10000)

# Compute variances
independent_variance = np.var(independent_samples, axis=0).reshape(150, 3).mean(axis=0)
correlated_variance = np.var(correlated_samples, axis=0).reshape(150, 3).mean(axis=0)

# Display results
#print("Variance of swings under different models:")
##print("Independent Model:", independent_variance)
#print("Correlated Model (Kronecker Product):", correlated_variance)

# Ratio of variance reduction
variance_ratio = correlated_variance / independent_variance
#print("Variance ratio (Correlated / Independent):", variance_ratio)








################################################### SIMULATE POLLING ERROR ###########################################################################################


import pdb;pdb.set_trace()




def split_vote_share_dirichlet(total_vote_share, Others_proportions, alpha_scale=50, n_samples=1): # chatgpt written
    """
    Splits a given total vote share across m parties using a Dirichlet distribution.
    
    Parameters:
    - total_vote_share (float): The total percentage of votes to split (e.g., 0.185 for 18.5%).
    - Others proportions (list of float): A list of m proportions summing to 1 (used as mean proportions).
    - alpha_scale (float): Scaling factor for the Dirichlet concentration parameter (higher = lower variance).
    - n_samples (int): Number of samples to generate.
    
    Returns:
    - np.ndarray: An array of shape (n_samples, m) with sampled vote shares.
    """
    assert np.isclose(sum(Others_proportions), 1), "Proportions must sum to 1."
    alpha = np.array(Others_proportions) * alpha_scale  # Convert mean proportions into Dirichlet parameters
    samples = np.random.dirichlet(alpha, size=n_samples) * total_vote_share
    return samples

def expand_Others_votes(election_year, simulated_votes_df, alpha_scale=50):


    Prior_df_long = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv", index_col = None)

    Prior_estimates_dict = {
        div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
        for div, group in Prior_df_long.groupby("div_nm")
    }

    expanded_Others_dict = {div:simulated_votes_df.loc[div] for div in simulated_votes_df.index} # series for each div

    # FIX IF THERE IS A 0 I.E. GORTON 2016/SOME 2019/2022 NO OTHERS

    ################## WHAT IF THERE IS A NATIONAL LURKING?????


    # Get all div_nms where Others were simulated
    for div, row in simulated_votes_df.iterrows():

        simulated_others_share = row['OTH']
        major_parties = ['ALP','COAL','GRN'] if election_year == '2016' else ['ALP','COAL','GRN','ON','UAPP']

        # Filter prior_df to just small parties in this division
        prior_row_df = Prior_estimates_dict.get(div)
        if prior_row_df is None or simulated_others_share == 0:
            continue

        prior_row = prior_row_df.iloc[0] # make into series (for each div)

        Other_parties = [p for p in prior_row.index if p not in major_parties]

        COAL_check = {'LP','NP','COALLP','COALNP'}
        if COAL_check & set(Other_parties):
            import pdb;pdb.set_trace()  # deal with NP properly! i..e. fix what was before? Or just let it be an 'other'

        if not Other_parties:
            continue

        prior_Others_votes = prior_row[Other_parties]

        # Calculate prior proportions (relative only to Others)
        rel_proportions = (prior_Others_votes / prior_Others_votes.sum()).values

        # Sample using Dirichlet
        sampled_votes = split_vote_share_dirichlet(simulated_others_share, rel_proportions, alpha_scale, n_samples=1)[0]

        # Combine major parties and sampled others
        combined = {party: row.get(party, 0.0) for party in major_parties}
        combined.update({party: share for party, share in zip(Other_parties, sampled_votes)})

        expanded_Others_dict[div] = pd.Series({k: combined.get(k, 0.0) for k in prior_row.columns})
        

    # Return as DataFrame
    return expanded_Others_dict

def expand_COAL_double_divs_votes(election_year, expanded_Others_dict, simulated_votes_df, alpha_scale = 50):

    # not yet adapted for 2025!

    # check if they are already separate! Probably only for VIC,NSW,2007QLD

    NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)
    NP_ratios_curr = NP_ratios_curr.loc[(NP_ratios_curr['election_year']==election_year) & (NP_ratios_curr['State'].isin(['VIC','NSW'])),]

    for div in NP_ratios_curr['div_nm'].unique():
        COAL_simulated_prop = simulated_votes_df.loc[simulated_votes_df['div_nm']==div,'COAL']
        NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate']
        COAL_mean_vector = [COAL_simulated_prop*(1-NP_est),COAL_simulated_prop*NP_est]
        LP_NP_votes = split_vote_share_dirichlet(COAL_simulated_prop, COAL_mean_vector, alpha_scale, n_samples=1)[0]

        expanded_Others_dict[div].loc[:,['LP','NP']] = LP_NP_votes[0], LP_NP_votes[0]
        expanded_Others_dict[div].drop('COAL', axis=1)

    return expanded_Others_dict


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

def split_category_in_simplex(simplex_probs, split_index, num_splits, dirichlet_alpha): # chatgpt written
    """Splits a category in probability space into `num_splits` subcategories using a Dirichlet distribution."""
    category_to_split = simplex_probs[:, split_index]
    split_proportions = dirichlet.rvs(dirichlet_alpha, size=len(simplex_probs))
    new_split_probs = category_to_split[:, None] * split_proportions

    # Construct new probability simplex
    new_simplex_probs = np.delete(simplex_probs, split_index, axis=1)
    new_simplex_probs = np.hstack([
        new_simplex_probs[:, :split_index], 
        new_split_probs, 
        new_simplex_probs[:, split_index:]
    ]) 
    return new_simplex_probs

def simulate_swings(alr_swing, alr_swing_cov, estimate_alr, num_simulations, dist, df_t, ref_col, full_Fundamentals_estimate_df):

    all_simulated_samples = np.zeros((num_simulations, alr_swing.shape[0], alr_swing.shape[1]+1))
    alr_simulated_samples = np.zeros((num_simulations, alr_swing.shape[0], alr_swing.shape[1]))

    prediction = np.zeros((alr_swing.shape[0], alr_swing.shape[1]+1))

    dimensions = {'2016':150,'2019':151,'2022':151,'2025':150}

    if dist == 't':
        cov_t_corrected = ((df_t - 2) / df_t) * alr_swing_cov  # Adjust scale matrix
        CovM =  np.kron(np.eye(dimensions[year]), cov_t_corrected)
    elif dist == 'Normal':
        CovM =  np.kron(np.eye(dimensions[year]), alr_swing_cov)
    
    for i in range(num_simulations):


        predicted_alr_year_list = []

        for year in ['2016','2019','2022']:

        
            if dist == 't':
                # Simulate swings using the corrected covariance
                simulated_alr_swings = multivariate_t.rvs(loc=np.zeros(CovM.shape[0]), shape=CovM, df=df_t).reshape(dimensions[year], 3)
            elif dist == 'Normal':
                simulated_alr_swings = multivariate_normal.rvs(mean=np.zeros(CovM.shape[0]), cov=alr_swing_cov).reshape(dimensions[year], 3)

            predicted_alr = estimate_alr + simulated_alr_swings
            predicted_alr_year_list.append(predicted_alr) # add current year's values

        concatenated_predicted_alr = pd.concat(predicted_alr_year_list) 



        curr_prediction = alr_to_simplex_vectorized(concatenated_predicted_alr, ref_col)[['ALP','COAL','GRN','OTH']]
        prediction += curr_prediction

        all_simulated_samples[i] = curr_prediction
        alr_simulated_samples[i] = concatenated_predicted_alr


    prediction /= num_simulations

    simulated_variance = np.var(all_simulated_samples, axis=0)

    # Compute the standard deviation as the square root of the variance
    simulated_std_dev = np.sqrt(simulated_variance)

    return prediction, pd.DataFrame(simulated_std_dev, columns=full_Fundamentals_estimate_df.columns, index=full_Fundamentals_estimate_df.index), all_simulated_samples, alr_simulated_samples


def simulate_correlated_swings(alr_swing, alr_swing_cov, num_simulations, dist, df_t, R_electorates_regularized_dict):

    all_simulated_samples = np.zeros((num_simulations, alr_swing.shape[0], alr_swing.shape[1]+1))
    alr_simulated_samples = np.zeros((num_simulations, alr_swing.shape[0], alr_swing.shape[1]))

    prediction =  np.zeros((alr_swing.shape[0], alr_swing.shape[1]+1))

    dimensions = {'2016':150,'2019':151,'2022':151,'2025':150}


    Sigma_total_by_year = {}
    Choleskys_by_year = {}
    for year in ['2016','2019','2022']:
        R_electorates_regularized = R_electorates_regularized_dict[year]
        if dist == 't':
            # Rescale covariance for the multivariate t-distribution
            Cov_t_corrected = ((df_t - 2) / df_t) * alr_swing_cov  # Adjust scale matrix
            Sigma_total = np.kron(R_electorates_regularized, Cov_t_corrected)
        elif dist == 'Normal':
            Sigma_total = np.kron(R_electorates_regularized, alr_swing_cov)

        Choleskys_by_year[year] = np.linalg.cholesky(Sigma_total)
        Sigma_total_by_year[year] = Sigma_total

    for i in range(num_simulations):

        predicted_alr_year_list = []

        for year in ['2016','2019','2022']:
            Sigma_total = Sigma_total_by_year[year]


            if dist == 't':
                # Simulate swings using the corrected covariance
                simulated_alr_swings = multivariate_t.rvs(loc=np.zeros(Sigma_total.shape[0]), shape=Sigma_total, df=df_t).reshape(dimensions[year], 3)
            
            elif dist == 'Normal':
                simulated_alr_swings =  np.dot(np.random.normal(size=(dimensions[year]*3,)), Choleskys_by_year[year]).reshape((dimensions[year], 3))
                #multivariate_normal.rvs(mean=np.zeros(Sigma_total.shape[0]), cov=Sigma_total).reshape(dimensions[year], 3)

            predicted_alr = pd.concat([estimate_alr.loc[estimate_alr.index.str.startswith(year),] + simulated_alr_swings]) # sum of two values

            predicted_alr_year_list.append(predicted_alr) # add current year's values

        concatenated_predicted_alr = pd.concat(predicted_alr_year_list) 

        curr_prediction = alr_to_simplex_vectorized(concatenated_predicted_alr, ref_col)[['ALP','COAL','GRN','OTH']]
        prediction += curr_prediction

        all_simulated_samples[i] = curr_prediction
        alr_simulated_samples[i] = concatenated_predicted_alr

    import pdb;pdb.set_trace()


    prediction /= num_simulations

    simulated_variance = np.var(all_simulated_samples, axis=0)

    # Compute the standard deviation as the square root of the variance
    simulated_std_dev = np.sqrt(simulated_variance)

    return prediction, pd.DataFrame(simulated_std_dev, columns=full_Fundamentals_estimate_df.columns, index=full_Fundamentals_estimate_df.index), all_simulated_samples, alr_simulated_samples


use_correlation = 1

df_t = 5
num_simulations = 100

method = 'Polls'
swing = alr_swing if method == 'Fundamentals' else alr_swing_poll
cov = alr_swing_cov if method == 'Fundamentals' else alr_swing_poll_cov

import pdb;pdb.set_trace()
if use_correlation:
    t_means, t_stds, all_simulated_samples_t, alr_simulated_samples_t = simulate_correlated_swings(swing, cov, num_simulations, 't', df_t, R_electorates_regularized_dict)

    N_means, N_stds, all_simulated_samples_normal, alr_simulated_samples_normal = simulate_correlated_swings(swing, cov, num_simulations, 'Normal', df_t, R_electorates_regularized_dict)
else:
    t_means, t_stds, all_simulated_samples_t, alr_simulated_samples_t = simulate_swings(swing, cov, estimate_alr, num_simulations, 't', df_t, ref_col, full_Fundamentals_estimate_df)

    N_means, N_stds, all_simulated_samples_normal, alr_simulated_samples_normal = simulate_swings(swing, cov, estimate_alr, num_simulations, 'Normal', df_t, ref_col, full_Fundamentals_estimate_df)


sns.kdeplot(alr_simulated_samples_normal[:, 0], label="Normal", color="blue")
sns.kdeplot(alr_simulated_samples_t[:, 0], label="T-distribution", color="red", linestyle="dashed")
plt.legend()
plt.title("Comparison of Simulated Swings (ALR Component 1)")
plt.show()



import pdb;pdb.set_trace()



# Load your ALR-transformed swings (shape: 150 electorates × 3 ALR components)

# Step 1: Fit a Multivariate Normal Distribution
mu_norm = np.mean(alr_swing, axis=0)
cov_norm = np.cov(alr_swing, rowvar=False)  # Covariance of ALR swings
log_likelihood_norm = np.sum(stats.multivariate_normal.logpdf(alr_swing, mean=mu_norm, cov=cov_norm))

# Step 2: Compute the log-likelihood for a multivariate t-distribution
def multivariate_t_logpdf(x, df, mu, cov):
    """
    Compute the log-density of a multivariate t-distribution at x.
    """
    d = len(mu)  # Number of dimensions (should be 3)
    x_mu = x - mu  # Center the data
    inv_cov = np.linalg.inv(cov)  # Inverse covariance matrix
    quad_form = np.sum(x_mu @ inv_cov * x_mu, axis=1)  # Quadratic form (Mahalanobis distance)

    log_det_cov = np.linalg.slogdet(cov)[1]  # Log determinant of covariance matrix
    log_const = (
        gammaln((df + d) / 2)  # Corrected gamma function
        - gammaln(df / 2)
        - (d / 2) * np.log(df)
        - (d / 2) * np.log(np.pi)
        - 0.5 * log_det_cov
    )
    
    log_pdf = log_const - ((df + d) / 2) * np.log(1 + quad_form / df)
    return log_pdf

def multivariate_t_log_likelihood(params):
    """ Negative log-likelihood function for a multivariate t-distribution. """
    df, *mu_flat, cov_flat = params[0], params[1:4], params[4:]
    mu = np.array(mu_flat)
    cov = np.reshape(cov_flat, (3, 3))
    
    if df <= 2:  # Degrees of freedom must be >2 for a valid covariance
        return np.inf

    log_likelihood = np.sum(multivariate_t_logpdf(alr_swing, df, mu, cov))
    return -log_likelihood  # Minimize negative log-likelihood

# Step 3: Optimize parameters (df=5, mean=normal mean, covariance=normal covariance)
init_params = np.hstack([5, mu_norm, cov_norm.flatten()])
bounds = [(2.01, None)] + [(None, None)] * 3 + [(None, None)] * 9  # Ensure df > 2

res = minimize(multivariate_t_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B")

df_t, mu_t, cov_t = res.x[0], res.x[1:4], np.reshape(res.x[4:], (3, 3))
log_likelihood_t = -res.fun

# Step 4: Compare Log-Likelihoods
print(f"Log-likelihood (Normal): {log_likelihood_norm}")
print(f"Log-likelihood (T): {log_likelihood_t}")
print(f"Estimated Degrees of Freedom (T): {df_t:.2f}")

if log_likelihood_t > log_likelihood_norm:
    print("Multivariate t-distribution fits better!")
else:
    print("Multivariate normal fits better!")

import pdb;pdb.set_trace()



