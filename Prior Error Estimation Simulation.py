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



def get_prior_ALR_covariance(data_year):
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
    Fundamentals_results_dict = {}
    Fundamentals_estimate_dict = {}

    Prior_estimates_df = pd.read_csv(f"Fundamentals_Votes_For_{next_year}.csv", index_col = None)

    Prior_estimates_dict = {
        div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
        for div, group in Prior_estimates_df.groupby("div_nm")
    }

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

    Fundamentals_results_df = Fundamentals_results_df[Fundamentals_results_df['Other'] != 0.0]
    Fundamentals_estimate_df = Fundamentals_estimate_df[Fundamentals_estimate_df['Other'] != 0.0]

    Fundamentals_results_df = Fundamentals_results_df.div(Fundamentals_results_df.sum(axis=1), axis=0)
    Fundamentals_estimate_df = Fundamentals_estimate_df.div(Fundamentals_estimate_df.sum(axis=1), axis=0)  

    return Fundamentals_results_df, Fundamentals_estimate_df

Fundamentals_results_list = []
Fundamentals_estimate_list = []


for data_year in ['2013','2016','2019']:
    Fundamentals_results_df,Fundamentals_estimate_df = get_prior_ALR_covariance(data_year)
    Fundamentals_results_list.append(Fundamentals_results_df)
    Fundamentals_estimate_list.append(Fundamentals_estimate_df)

full_Fundamentals_results_df = pd.concat(Fundamentals_results_list)
full_Fundamentals_estimate_df = pd.concat(Fundamentals_estimate_list)

# center the natural swings to avoid bias - we assume swings are generally unbiased - that we cannot predict direction 3 years prior!
swings = full_Fundamentals_results_df - full_Fundamentals_estimate_df
#swings_centered = (swings - swings.mean()).sum(axis=1)
full_Fundamentals_estimate_df = full_Fundamentals_estimate_df + swings.mean()

import pdb;pdb.set_trace()


ref_col = 'COAL'  # Reference category to remove

results_alr = np.log(full_Fundamentals_results_df.drop(columns=[ref_col]).div(full_Fundamentals_results_df[ref_col], axis=0))
estimate_alr = np.log(full_Fundamentals_estimate_df.drop(columns=[ref_col]).div(full_Fundamentals_estimate_df[ref_col], axis=0))
alr_swing = results_alr - estimate_alr

alr_swing_cov = alr_swing.cov()


print(alr_swing.cov())
print((full_Fundamentals_results_df - full_Fundamentals_estimate_df).mean()) # should be 0 due to centralisation adjustment

import pdb;pdb.set_trace()




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







import pdb;pdb.set_trace()




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
samples = multivariate_normal.rvs(mean=mean_vector, cov=alr_swing_cov, size=n_samples)

print(samples.shape)  # Should be (1000, 750)