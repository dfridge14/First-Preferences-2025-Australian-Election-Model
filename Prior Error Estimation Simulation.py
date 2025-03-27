import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, dirichlet



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

data_year = '2013' # predicting next year's election
next_year = '2016'
Actual_results = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", index_col = None).rename(columns={'DivisionNm':'div_nm'})
# COUntNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


# Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
# rename CLR to ALP
# rename IND to INDX by order

target = 'IND'

Actual_results_dict = {}
Fundamentals_results_dict = {}
Fundamentals_estimate_dict = {}

Prior_estimates_df = pd.read_csv(f"Fundamentals_Votes_Following_{data_year}.csv", index_col = None)

Prior_estimates_dict = {
    div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
    for div, group in Prior_estimates_df.groupby("div_nm")
}

def group_into_Fundamentals_Categories(party_votes_shares_df, div):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'LP','NP','CLP','LNP','LNQ'}
    GRN_cat = {'GRN'}

    Non_Other_sets = ALP_cat | COAL_cat | GRN_cat  # Union of all sets
    Other_cols = set(party_votes_shares_df.columns) - Non_Other_sets  # Columns in none of the sets

    # Compute the sums
    sum1 = party_votes_shares_df[ALP_cat.intersection(party_votes_shares_df.columns)].sum(axis=1).iloc[0]
    sum2 = party_votes_shares_df[COAL_cat.intersection(party_votes_shares_df.columns)].sum(axis=1).iloc[0]
    sum3 = party_votes_shares_df[GRN_cat.intersection(party_votes_shares_df.columns)].sum(axis=1).iloc[0]
    sum4 = party_votes_shares_df[Other_cols].sum(axis=1).iloc[0]

    Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum4}], index=div)

    return Fundamentals_grouped_df

Fundamentals_results_list = []
Fundamentals_estimate_list = []


for div in Actual_results['div_nm'].unique():
    div_results = Actual_results.loc[Actual_results['div_nm'] == div,]

    div_results.loc[:,'Count'] = div_results.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
    # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...
    adjusted_party_names = div_results.loc[div_results["CountNumber"] == 0,].apply(
        lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
    ).reset_index(drop=True)

    import pdb;pdb.set_trace()
    Actual_results.loc[Actual_results['div_nm'] == div,'PartyAb'] = adjusted_party_names

    Actual_results_dict[div] = div_results.pivot(index='div_nm', columns='PartyAb', values='CalculationValue')
    #Fundamentals_results_dict[div] = group_into_Fundamentals_Categories(Actual_results_dict[div], div)
    #Fundamentals_estimate_dict[div] = group_into_Fundamentals_Categories(Prior_estimates_dict[div], div)

    Fundamentals_results_list.append(group_into_Fundamentals_Categories(Actual_results_dict[div], div))
    Fundamentals_estimate_list.append(group_into_Fundamentals_Categories(Prior_estimates_dict[div], div))



Fundamentals_results_df = pd.concat(Fundamentals_results_list)
Fundamentals_estimate_df = pd.concat(Fundamentals_results_list)


# Step 1: Compute Swing, as a proportion
swing = Fundamentals_results_df.div(Fundamentals_results_df.sum(axis=1), axis=0) - Fundamentals_estimate_df.div(Fundamentals_estimate_df.sum(axis=1), axis=0)  # Swing is just the difference

# Step 2: ALR Transformation (Drop last column 'D')
ref_col = 'COAL'  # Reference category to remove
alr_swing = np.log(swing.drop(columns=[ref_col]).div(swing[ref_col], axis=0))

# Step 3: Compute Covariance Matrix
cov_matrix = alr_swing.cov()



def split_vote_share_dirichlet(total_vote_share, Others_proportions, alpha_scale=50, n_samples=1): # chatgpt written
    """
    Splits a given total vote share across m parties using a Dirichlet distribution.
    
    Parameters:
    - total_vote_share (float): The total percentage of votes to split (e.g., 0.185 for 18.5%).
    - proportions (list of float): A list of m proportions summing to 1 (used as mean proportions).
    - alpha_scale (float): Scaling factor for the Dirichlet concentration parameter (higher = lower variance).
    - n_samples (int): Number of samples to generate.
    
    Returns:
    - np.ndarray: An array of shape (n_samples, m) with sampled vote shares.
    """
    assert np.isclose(sum(Others_proportions), 1), "Proportions must sum to 1."
    alpha = np.array(Others_proportions) * alpha_scale  # Convert mean proportions into Dirichlet parameters
    samples = np.random.dirichlet(alpha, size=n_samples) * total_vote_share
    return samples



def alr_to_simplex(alr_samples): # chatgpt written
    """Convert ALR-transformed samples back to simplex space."""
    exp_values = np.exp(alr_samples)
    denominator = 1 + np.sum(exp_values, axis=-1, keepdims=True)
    base_category = 1 / denominator
    return np.hstack([exp_values / denominator, base_category])

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





# simulate from 750-dim covariance matrix:
dim = 750  # Number of dimensions
n_samples = 1000  # Number of samples

# Example: Generate a random positive semi-definite covariance matrix
np.random.seed(42)  # For reproducibility
A = np.random.randn(dim, dim)
cov_matrix = A @ A.T  # Ensure it's positive semi-definite

# Example: Set mean vector (750-dimensional)
mean_vector = np.random.randn(dim)

# Sample from the MVN distribution
samples = multivariate_normal.rvs(mean=mean_vector, cov=cov_matrix, size=n_samples)

print(samples.shape)  # Should be (1000, 750)