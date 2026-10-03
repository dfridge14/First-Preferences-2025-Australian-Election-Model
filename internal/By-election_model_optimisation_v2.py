import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import time

import sys
import pdb
import traceback




def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


base_dir = Path.home() / "Necessary CSV Files"
os.chdir(base_dir)

os.chdir('/home/dania-freidgeim/Australian Election')

# Simulated data
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


NUM_MAIN_PARTIES = 3
LP_NP_DIR_ALPHA = 4.178317954770676
LP_NP_DIR_ALPHA_POLL  = 12
MAJOR_SITOUT_DIR_ALPHA = 14 # (4 + 7 + 31)/3


DIR_epsilon = 5e-3

SEAT_POLL_SCALE_1 = 1.0747964937804577
SEAT_POLL_SCALE_2 = 0.8977871632158285
SEAT_POLL_SCALE_3 = 0.7965537354979125

SEAT_POLL_SCALE = {
    1: SEAT_POLL_SCALE_1,
    2: SEAT_POLL_SCALE_2,
    3: SEAT_POLL_SCALE_3
}


# 1. Get nat-poll-adjusted prior_df, for those electorates/election_years that have both priors and seat polls (102+15)



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
    if election_year in ['2013','2016', 'Byelection']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum6}], index=[div])
    elif election_year in ['2019','2022']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'UAPP':sum5, 'Other':sum6}], index=[div])
    elif election_year == '2025':
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'TOP':sum5, 'Other':sum6}], index=[div])


    return Fundamentals_grouped_df



def group_into_Fundamentals_Categories(party_votes_shares_df, div, is_Other = True):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'COAL','COALNP','COALLP','LP','NP','NAT','CLP','LNP','LNQ'}
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




def get_results_df(election_year, to_Fundamentals = True):

    # returns Actual_results_dict with results by party per division, and Fundamentals_results_df


    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    # CountNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best

    if election_year == '2025':
        # concordance with unknown-ab parties
        Actual_results.loc[Actual_results['PartyAb'] == 'CYA','PartyAb'] = 'TOP'
        Actual_results.loc[Actual_results['PartyAb'] == 'GRPF','PartyAb'] = 'PFP'
        Actual_results.loc[Actual_results['PartyAb'] == 'FFPA','PartyAb'] = 'FFP'
        Actual_results.loc[Actual_results['PartyAb'] == 'LTP','PartyAb'] = 'LDP'
        


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

def get_Prior_estimates_df(election_year, dont_add_ON = False):

    if election_year == 'Byelection':
        Prior_estimates_df = pd.read_csv(f"By_election_Fundamentals_full_added_Majors.csv", index_col = None)
    elif (election_year == '2016') | dont_add_ON:
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

    # Gorton 2016 adjustment: add 0.01 from GRN to Other
    Prior_estimates_df.loc[Prior_estimates_df['Other'] == 0.0,['GRN','Other']] += (-0.01,0.01)

    return Prior_estimates_df, Prior_estimates_dict

def df_to_alr(df, ref_col):
    """Convert a DataFrame of proportions to ALR, dropping ref column."""
    df_alr = np.log(df.drop(columns=[ref_col]).div(df[ref_col], axis=0))
    return df_alr


def alr_to_simplex_vectorised(alr_matrix, ref_col, col_names):
    exp_vals = np.exp(alr_matrix) 

    # Compute reference share: 1 / (1 + sum(exp(alr)))
    # Shape: (n_simulations, 1)
    ref_vals = 1 / (1 + np.sum(exp_vals, axis=1, keepdims=True))  
    
    # Concatenate ref_vals FIRST, then the expanded values
    # Shape: (n_simulations, D)
    simplex_vals = np.concatenate((ref_vals, exp_vals * ref_vals), axis=1)  
    
    # Update column names to match the new order
    new_columns = [ref_col] + col_names
    
    return simplex_vals, new_columns

def combine_to_CAGO(df):
    cols_to_combine = ['ON', 'UAPP', 'TOP', 'OTH']
    existing_cols = [c for c in cols_to_combine if c in df.columns]

    df['OTH'] = df[existing_cols].sum(axis=1)

    cols_to_drop = [c for c in existing_cols if c != 'OTH']
    df = df.drop(columns=cols_to_drop)

    return df



CAGO_only = 1
ref_col = 'COAL'

prior_df_dict = {}
National_prior_dict = {}
National_prior_dict['2026'] = pd.DataFrame([[0.3456,0.3182,0.122,0.2142]], columns = ['ALP','COAL','GRN','OTH'])

for year in ['2016','2019','2022','2025','Byelection']:

    prior_df = get_Prior_estimates_df(year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})

    # combine ON,UAPP,OTH into OTH
    if CAGO_only:
        prior_df = combine_to_CAGO(prior_df)
    else:
        if year == '2016':
            continue
        else:
            # use only cases with all parties
            Palmer_cols = {'UAPP','TOP'}
            Palmer_col_curr = [p for p in (set(prior_df.columns) & Palmer_cols)][0]

            prior_df = prior_df.loc[(prior_df['ON']>0) & (prior_df[Palmer_col_curr]>0),]


    prior_df_dict[year] = prior_df

    if year != 'Byelection':
        National_prior_dict[year] = prior_df.mean().to_frame().T

    # for each byleection get its own prior mean - FIX

# FIX - byelection prior should involve all majors!                                                     DONE
# FIX - byeleciton national prior should use previous years! USE By_eleciton_error for this! 


# 2. Get seat poll

election_years = ['2016','2019','2022','2025','Byelection']
ELECTION_DATE_NUM = {'2013':1113, '2016':1028, '2019':1050, '2022':1099,'2025':1078}
Seat_poll_year_dict = {}

for election_year in election_years:

    Seat_poll_year_dict[election_year] = pd.read_csv(f"SeatPollAverage{election_year}Formatted.csv", index_col=None)

    if 0:

        if election_year != 'Byelection':

            Seat_polls = pd.read_csv(f"SeatPollAverage{election_year}Formatted.csv", index_col=None)
            Seat_polls.loc[:,'Days since last election'] = ELECTION_DATE_NUM[election_year] - Seat_polls.loc[:,'Days since last election']
            Seat_polls.rename(columns={'Days since last election':'Days before election'}, inplace=True)

            Seat_poll_year_dict[election_year] = Seat_polls
        else:

            # add by_election seat polls/results to dicts
            By_election_data = pd.read_csv("Federal_by_election_data.csv")
            Seat_polls =  pd.read_csv(f"SeatPollAverageByelectionFormatted.csv", index_col=None)

            # format first col into days before byelection via merge with By_election_data:
            By_election_data['byelection_date'] = pd.to_datetime(By_election_data['byelection_date'],format='%d.%m.%y')
            By_election_data['byelection_year'] = By_election_data['byelection_date'].dt.year
            By_election_data.loc[By_election_data['div_nm']=='Batman','div_nm'] = 'Cooper' # manual adjustment

            Seat_polls = Seat_polls.merge(By_election_data[['days_since_election','byelection_year','div_nm']], left_on = ['byelection_year','Electorate'], right_on = ['byelection_year','div_nm']).drop(columns=['div_nm'])
            Seat_polls.loc[:,'Days since last election'] = Seat_polls.loc[:,'days_since_election'] - Seat_polls.loc[:,'Days since last election']
            Seat_polls = Seat_polls.rename(columns={'Days since last election':'Days before election'}).drop(columns=['days_since_election'])

            Seat_polls = Seat_polls[Seat_polls['Electorate'] != 'Canberra']
            Seat_poll_year_dict['Byelection'] = Seat_polls




# get by-eleciton national priors

Byelection_daily_polling_averages = pd.read_csv("Byelection_daily_polling_averages.csv")
Byelection_daily_polling_averages.loc[Byelection_daily_polling_averages["Election"] == "Batman2019", "Election"] = "Cooper2019"
Byelection_final_polling_averages = Byelection_daily_polling_averages.loc[Byelection_daily_polling_averages["Day_index"] == -1].copy()
Byelection_final_polling_averages["Election"] = Byelection_final_polling_averages["Election"].astype(str)
Byelection_final_polling_averages["prev_election_year"] = Byelection_final_polling_averages["Election"].str[-4:].astype(int) - 3
Fundamentals_Votes_For_Byelection = pd.read_csv("Fundamentals_Votes_For_Byelection.csv")
Byelection_final_polling_averages["div_nm"] = Byelection_final_polling_averages["Election"].str.extract(r"([A-Za-z\- ]+)")[0]
Fundamentals_Votes_For_Byelection = Fundamentals_Votes_For_Byelection.drop_duplicates(subset="div_nm")
Byelection_final_polling_averages = Byelection_final_polling_averages.merge(Fundamentals_Votes_For_Byelection[["div_nm", "byelection_year"]], on="div_nm", how="left", validate="many_to_one")
Byelection_final_polling_averages["Election"] = Byelection_final_polling_averages["div_nm"] + Byelection_final_polling_averages["byelection_year"].astype(int).astype(str)

National_election_results = pd.read_csv("NationalElectionResults.csv")
cols = ["COAL", "ALP", "GRN", "OTH"]
nat_prior_df = Byelection_final_polling_averages[["prev_election_year"]].join(National_election_results.set_index("Election")[cols],on="prev_election_year").drop(columns=['prev_election_year'])

Byelection_final_polling_averages = Byelection_final_polling_averages.set_index("Election").drop(columns=["Density", "prev_election_year", "div_nm", "byelection_year",'Day_index'])
nat_prior_df.index = Byelection_final_polling_averages.index







# Covariance matrices

import pickle

with open("covms_all.pkl", "rb") as f:
    data = pickle.load(f)

Seat_poll_covm_dict = data["Seat_poll_covm_dict"]
Electorate_residual_covm_dict = data["Electorate_residual_covm_dict"]
Nat_poll_covm_dict = data["Nat_poll_covm_dict"]


# 3. Simulate polling error

#                                       GENERAL    BYELECTION
# 1. Prior_estimates_df
# 2. day_z_polling_avg                  DONE        
# 3. Naitonal prior?                    DONE
# 4. Number of polled electorates
#NUM_ACTIVE_ELECTORATES = {year: Seat_poll_year_dict[year]['Electorate'].nunique() for year in Seat_poll_year_dict}

def alr_to_simplex_simulation_array(alr_array):
        """
        Perform inverse ALR transformation on a 3D array of shape (S, D, K).
        Returns a 3D array of shape (S, D, K+1) in the probability simplex.
        reference category is the 1st in the simplex
        """
        exp_vals = np.exp(alr_array)  # shape (S, D, K)
        ref_vals = 1 / (1 + np.sum(exp_vals, axis=-1, keepdims=True))  # shape (S, D, 1)
        simplex = np.concatenate([ref_vals, exp_vals * ref_vals], axis=-1)  # shape (S, D, K+1)
        return simplex

def full_weight_vector(all_divisions, High_Others_df, others_column="OTH", beta=0.5, k=10, start=0.2):
    import numpy as np

    # align OTH values to all_divisions
    oth = High_Others_df.reindex(all_divisions)[others_column].astype(float).fillna(0.0).values

    # logistic-style downweighting (same spirit as original)
    # higher OTH → lower weight
    x = (oth - start) * k
    w = 1 / (1 + np.exp(x))

    # apply beta scaling
    w = 1 - beta * (1 - w)

    return w


def get_beta_weights(election_year, prior_df_dict, beta=0.5):
    High_Others_df = pd.read_csv(f"High_Prior_OTH_Electorates_{election_year}.csv", index_col=0)
    all_divs = prior_df_dict[election_year].index
    w_all = full_weight_vector(all_divisions=all_divs, High_Others_df=High_Others_df, beta=beta)
    w_series = pd.Series(w_all, index=all_divs)
    active_divs = list(Seat_poll_year_dict[election_year]['Electorate'].unique())
    return w_series.loc[active_divs].to_frame('weight')

def simulate_Polling_Fundamentals_model(
    n_simulations,
    election_year,
    df_t=0,
    v=0.15,
    beta=0.5,
    Prior_estimates_df=pd.DataFrame(),
    polling_avg = pd.DataFrame(),
    National_prior = pd.DataFrame(),
    National_Polling_error_cov =  pd.DataFrame(),
    Electorate_Residuals_cov =  pd.DataFrame(),
    For_combined_model = False
):
    
    use_volatility_weights = (election_year != 'Byelection')

    NUM_ACTIVE_ELECTORATES = len(Prior_estimates_df)
    
    # -----------------------------
    # Distribution choice
    # -----------------------------
    dist = "Normal" if df_t == 0 else "t"

    # -----------------------------
    # Volatility grouping (optional, keep only for general elections)
    # -----------------------------

    if use_volatility_weights:
    
        Volatility_cat = pd.read_csv(
            f"Volatility_weights_df_{election_year}.csv", index_col=None
        )
        # keep track of correct indices - remapped_weights_idx_dict is a dict with category: indices of electorate within Prior_estimates_df subset
        full_index_to_name = Volatility_cat['Electorate'].tolist()
        subset_names = Prior_estimates_df.index.tolist()
        name_to_subset_index = {name: i for i, name in enumerate(subset_names)}

        weights_idx_dict = defaultdict(list)
        for idx, scale in enumerate(Volatility_cat['Volatility_weights']):
            weights_idx_dict[scale].append(idx)
        weights_idx_dict = dict(weights_idx_dict)
        
        remapped_weights_idx_dict = defaultdict(list)
        for scale, idx_list in weights_idx_dict.items():
            for full_idx in idx_list:
                electorate_name = full_index_to_name[full_idx]
                if electorate_name in name_to_subset_index:
                    remapped_weights_idx_dict[scale].append(name_to_subset_index[electorate_name])
    else:
        remapped_weights_idx_dict = {0: list(range(NUM_ACTIVE_ELECTORATES))} # base default - all category 0

   
    # -----------------------------
    # NATIONAL POLLING ERROR
    # -----------------------------
    National_Polling_error_cov = Nat_poll_covm_dict[election_year]

    National_Simulated_polling_error = np.random.multivariate_normal(
        mean=np.zeros(National_Polling_error_cov.shape[0]),
        cov=National_Polling_error_cov.values,
        size=n_simulations
    )[:, None, :]

    National_Simulated_polling_error_expanded = np.repeat(
        National_Simulated_polling_error,
        NUM_ACTIVE_ELECTORATES,
        axis=1
    )

    # -----------------------------
    # ELECTORATE RESIDUAL ERROR
    # -----------------------------
    Electorate_Residuals_cov = Electorate_residual_covm_dict[election_year]
    
    Scaled_covs = {}    
    
    d = Electorate_Residuals_cov.shape[0]
    OTH_index = -1

    if use_volatility_weights:
        category_weights = {0: 0.95, 1: 1.25, 2: 1.5, 3: 4}
        for cat, weight in category_weights.items():

            scaling = (1 + (weight - 1) * v) if weight > 1 else (1 - (1 - weight) * v)

            cov_adj = Electorate_Residuals_cov.copy().values

            current_var = cov_adj[OTH_index, OTH_index]
            target_var = scaling**2 * current_var
            delta = target_var - current_var

            cov_adj[OTH_index, OTH_index] += delta

            for i in range(d):
                if i != OTH_index:
                    cov_adj[OTH_index, i] *= scaling
                    cov_adj[i, OTH_index] = cov_adj[OTH_index, i]

            Scaled_covs[cat] = cov_adj
    else:
        Scaled_covs[0] = Electorate_Residuals_cov.values # No transformation

    Electorate_Residuals_Simulated_error = np.empty(
        (n_simulations,
        NUM_ACTIVE_ELECTORATES,
        d)
    )

    for scale, indices in remapped_weights_idx_dict.items():

        cov = Scaled_covs[scale]
        n_group = len(indices)

        if n_group == 0:
            continue

        MEAN_SHIFT = 1
        mean = np.zeros(d)
        if (election_year == "Byelection") and MEAN_SHIFT:
            grn_idx = 1
            mean[grn_idx] = -0.27
            
        group_sims = np.random.multivariate_normal(mean=mean,cov=cov,size=n_simulations * n_group)

        if dist == "t":
            g = np.random.gamma(df_t / 2., 2. / df_t, size=n_simulations * n_group)
            group_sims = group_sims / np.sqrt(g)[:, None]

        Electorate_Residuals_Simulated_error[:, indices, :] = group_sims.reshape(
            n_simulations,
            n_group,
            d
        )



    # -----------------------------
    # BASE PRIOR STRUCTURE
    # -----------------------------
    ref_col = "COAL"



    polling_alr = np.log(polling_avg.drop(columns=[ref_col]).div(polling_avg[ref_col], axis=0))

    National_prior_alr = df_to_alr(National_prior, ref_col=ref_col)

    Prior_estimates_alr = np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))

    # -----------------------------
    # EXPAND PRIOR
    # -----------------------------
    Prior_estimates_alr_expanded = np.tile(
        Prior_estimates_alr.to_numpy(),
        (n_simulations, 1, 1)
    )

    # -----------------------------
    # NATIONAL SHOCK APPLICATION
    # -----------------------------
    Simulated_national_result_alr = (polling_alr.values+ National_Simulated_polling_error_expanded)

    # -----------------------------
    # ELECTORATE PROJECTION WEIGHTING
    # -----------------------------
    

    if use_volatility_weights:
        weights_by_beta = get_beta_weights(election_year, prior_df_dict, beta=0.5)
        Non_Uniformity_weights = weights_by_beta['weight'].values[None, :, None]
        
        Projected_Electorate_Polling_Results_ALR = (
            Prior_estimates_alr_expanded
            + Non_Uniformity_weights * (
                Simulated_national_result_alr
                - National_prior_alr.values
            )
        )
    else:
        # BYELECTION: keep national shock, no beta shrinkage
        Projected_Electorate_Polling_Results_ALR = (
            Prior_estimates_alr_expanded
            + (
                Simulated_national_result_alr
                - National_prior_alr.values
            )
        )


    if For_combined_model:
        return Projected_Electorate_Polling_Results_ALR
    
    # -----------------------------
    # ADD ELECTORATE RESIDUALS
    # -----------------------------
    Simulated_Electorate_Polling_Results_ALR = (
        Projected_Electorate_Polling_Results_ALR
        + Electorate_Residuals_Simulated_error
    )

    # -----------------------------
    # CONVERT TO SIMPLEX
    # -----------------------------
    Simulated_Electorate_Polling_Results = alr_to_simplex_simulation_array(
        Simulated_Electorate_Polling_Results_ALR
    )


    return (
        Simulated_Electorate_Polling_Results
    )



def combine_models_alr_precision_weighted(
    n_simulations,
    B,
    nat_alr_sims, # Simulated_Electorate_Polling_Results_ALR for this div (n_sims, 3)
    seat_alr_base, # alr_base from precomputed_dict (3,)
    Nat_poll_covm, # National_Polling_error_cov (3, 3)
    Resid_covm,    # Electorate_Residuals_cov (3, 3)
    Seat_poll_covm # Seat_poll_covm_dict[election_year] (3, 3)
):
    

    # 1. Calculate Total National Covariance = Sum of national-level error and electorate-specific residuals
    Sigma_nat = Nat_poll_covm.values + Resid_covm.values
    Inv_Sigma_nat = np.linalg.inv(Sigma_nat)

    # 2. Calculate Seat Polling Precision, weighted by B
    Sigma_seat = Seat_poll_covm.values
    Inv_Sigma_seat_weighted = B * np.linalg.inv(Sigma_seat)

    # 3. Compute the Combined Covariance (The New Distribution Spread) Sigma_comb = inv(Inv_Nat + B * Inv_Seat)
    Sigma_comb = np.linalg.inv(Inv_Sigma_nat + Inv_Sigma_seat_weighted)

    # 4. Compute the Combined Means for every Simulation
    # Because nat_alr_sims varies per simulation, we calculate a unique mu_comb per sim #mu_comb = Sigma_comb * (Inv_Nat * nat_alr_sim + B * Inv_Seat * seat_alr_base)
    
    # Pre-calculate the seat-part of the numerator
    seat_part = Inv_Sigma_seat_weighted @ seat_alr_base # (3,)
    
    # Calculate nat-part for all simulations: (n_sims, 3) @ (3, 3) -> (n_sims, 3)
    nat_part = nat_alr_sims @ Inv_Sigma_nat.T 
    
    # Combine and multiply by Sigma_comb to get the new simulation centers
    combined_mu = (nat_part + seat_part) @ Sigma_comb.T # (n_sims, 3)

    d = Seat_poll_covm.shape[0]

    # 5. Draw final realizations from the combined distribution
    # This centers the Fundementals around the update while narrowing the variance
    combined_errors = np.random.multivariate_normal(
        mean=np.zeros(d),
        cov=Sigma_comb,
        size=n_simulations
    )

    # Final ALR realizations for [ALP, GRN, Other]
    combined_alr = combined_mu + combined_errors

    return combined_alr



def redistribute_major_sitout(
    combined,
    all_party_names,
    div,
    mapping_df
    ):
    
    proportions_df = pd.read_csv('Missing_major_reallocation_proportions.csv')
    Major_sitout, has_strong_IND = mapping_df.loc[mapping_df['div_nm']==div,['Major_sitout','has_strong_IND']].drop_duplicates().to_numpy().ravel()
    
    combined = combined.copy()
    n_sims = combined.shape[0]
    
    # --- 1. index of sitout party
    sitout_idx = all_party_names.index(Major_sitout)
    
    # --- 2. extract and remove sitout vote
    V = combined[:, sitout_idx].copy()
    combined[:, sitout_idx] = 0.0
    
    # --- 3. get group means (μ)
    row = proportions_df[(proportions_df['Major_sitout'] ==  ('LP' if Major_sitout == 'COAL' else Major_sitout) ) & (proportions_df['has_strong_IND'] == has_strong_IND)].iloc[0]
    

    if has_strong_IND == 1:
        mu = np.array([row['theta_aligned'], row['theta_ind'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'IND', 'other_major', 'other']
    else:
        mu = np.array([row['theta_aligned'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'other_major', 'other']
    
    # --- 4. Dirichlet parameters
    alpha = mu * MAJOR_SITOUT_DIR_ALPHA
    
    # --- 5. sample group shares
    p_group = np.random.dirichlet(alpha, size=n_sims)
    
    # --- 6. mapping for this division + regime
    sub = mapping_df[(mapping_df['div_nm'] == div) &(mapping_df['Major_sitout'] == Major_sitout) &(mapping_df['has_strong_IND'] == has_strong_IND)]
        
    # --- 7. allocate to each group
    for g_idx, g in enumerate(groups):
        
        parties = sub.loc[sub['group'] == g, 'PartyAb'].tolist()
        
        if not parties:
            continue
        
        # map to indices in combined
        idx = [all_party_names.index(p) for p in parties if p in all_party_names]
        
        if not idx:
            continue
        
        current = combined[:, idx]
        row_sums = current.sum(axis=1, keepdims=True)
        
        # proportional weights (fallback = equal split)
        weights = np.divide(
            current,
            row_sums,
            out=np.full_like(current, 1/len(idx)),
            where=row_sums > 0
        )
        
        allocation = (V * p_group[:, g_idx])[:, None] * weights
        
        combined[:, idx] += allocation

    combined = np.delete(combined, sitout_idx, axis=1)
    all_party_names = [p for i,p in enumerate(all_party_names) if i != sitout_idx]
    
    return combined, all_party_names


def adjust_prior_to_matrix(
    prior_row, 
    realloc_data, 
    all_party_names
):
    p_group = realloc_data['p_group']
    groups = realloc_data['groups']
    sub = realloc_data['sub']
    missing_party = realloc_data['missing_party']
    n_sims = p_group.shape[0]
    
    # --- 1. Tile the single prior row into a matrix (n_sims, len(all_party_names))
    prior_row_COALified = prior_row.copy()
    prior_row_COALified.columns = ['COAL' if p in['LP','NP','CLP','LNP'] else p for p in prior_row.columns]
    prior_matrix = np.tile(prior_row_COALified[all_party_names].values, (n_sims, 1)).astype(float)
    
    # --- 2. Extract and remove sitout vote
    all_party_names = [p if p not in ['LP','NP','CLP','COAL'] else 'COAL' for p in all_party_names]
    sitout_idx = all_party_names.index(missing_party)
    V = prior_matrix[:, sitout_idx].copy()
    prior_matrix[:, sitout_idx] = 0.0
    
    # --- 3. Track the specific IND addition
    ind_surge_dict = {}
    
    # --- 4. Allocate to each minor/major party proportionally
    for g_idx, g in enumerate(groups):
        parties = sub.loc[sub['group'] == g, 'PartyAb'].tolist()
        mapped_parties = ['IND' if p.startswith('IND') else p for p in parties]

        if not parties:
            import pdb; pdb.set_trace()
            continue
            
        idx = list(set([all_party_names.index(p) for p in mapped_parties if p in all_party_names]))
        if not idx:
            import pdb; pdb.set_trace() 
            continue
            
        current = prior_matrix[:, idx]
        row_sums = current.sum(axis=1, keepdims=True)
        
        weights = np.divide(
            current,
            row_sums,
            out=np.full_like(current, 1/len(idx)),
            where=row_sums > 0
        )
        
        allocation = (V * p_group[:, g_idx])[:, None] * weights
        prior_matrix[:, idx] += allocation

        
        # Track the specific mass added to the exact IND labels
        if g == 'IND':
            total_ind_surge = (V * p_group[:, g_idx])

            new_total_ind = prior_matrix[:, idx[0]]
            surge_ratio = np.divide(total_ind_surge, new_total_ind, out=np.zeros_like(total_ind_surge), where=new_total_ind > 0)
            
            # Split the surge equally among however many specific INDs are mapped
            # (Usually there is only one, e.g., ['IND1'], so it gets 100%)
            for p in parties:
                ind_surge_dict[p] = surge_ratio / len(parties)


    # --- 5. Remove missing party
    prior_matrix = np.delete(prior_matrix, sitout_idx, axis=1)
    all_party_names_reduced = [p for i, p in enumerate(all_party_names) if i != sitout_idx]
    
    return prior_matrix, all_party_names_reduced, ind_surge_dict

def expand_all_divisions_from_prior_df(sim, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=100):

    final_sim = {}
    party_name_dict = {} 

    multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
    C200_IND_positions_df = pd.read_csv("C200_IND_positions_df.csv", index_col=None)
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]
    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]

    Complex_contests = {'Calare': 0.4788,'Monash': 0.5002,'Moore': 0.43987} # proportions of split to C200 independent and Incumbent independent (based on % after adding Historical proportions in (div,div) pair)



    Major_parties = ['COAL','LP','NP','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']
    if election_year != '2016':
        if not CAGO_only:
            Major_parties += ['ON','UAPP','TOP']
            if election_year != '2025':
                Polling_parties += ['ON','UAPP']
            else:
                Polling_parties += ['ON','TOP']


    # For the split of COAL_double_divs
    NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)

    NP_ratios_curr = NP_ratios_curr.loc[(NP_ratios_curr['election_year']==election_year) ,] # & (NP_ratios_curr['State'].isin(['VIC','NSW']))
        

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
        if not CAGO_only:
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



            alpha = np.array([1-NP_est,NP_est]) * LP_NP_DIR_ALPHA
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
                means_array_sim = np.tile(np.full(num_INDs_curr, 1/num_INDs_curr), (len(IND_votes), 1)) # initialises an even split



                if div in C200_IND_positions_curr['div_nm'].unique():
                    # split according to C200 splits!

                    beta0,beta1 = C200_ratio_lm_params
                    logit_mu = beta0 + beta1 * np.log(IND_votes*100)
                    C200_ratio = np.where(IND_votes < 0.05, 0.9, 1 / (1 + np.exp(-logit_mu)))


                    #C200_ratio = C200_IND_splits.loc[C200_IND_splits['Election'] == f'AverageFor{election_year}','Ratio'].iloc[0]

                    C200_IND_position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm'] == div,'Number'].iloc[0]

                    if (election_year == '2025') and div in Complex_contests.keys():

                        C200_ratio = np.full_like(C200_ratio, Complex_contests[div], dtype=float) # an array col
                    
                    K = num_INDs_curr.iloc[0]
                    non_ind_total = 1 - C200_ratio          # (n_sims,)
                    means_array_sim = np.tile((non_ind_total / (K - 1))[:, None],(1, K))
                    means_array_sim[:, C200_IND_position - 1] = C200_ratio

                #splits = np.random.dirichlet(means_array* alpha_scalar, size=sim.shape[0])
                splits = np.array([np.random.dirichlet(alpha_scalar * row) for row in means_array_sim])

                All_IND_votes = splits * IND_votes[:, None]

                combined = np.concatenate([combined, All_IND_votes], axis=1)
                combined = np.delete(combined,IND_index, axis = 1) # remove 'IND' col

                all_party_names = all_party_names + ['IND'+str(i) for i in range(1,num_INDs_curr.iloc[0]+1)]
                all_party_names = [p for p in all_party_names if p != 'IND'] # remove IND

            else:
                all_party_names = [p if p != 'IND' else 'IND1' for p in all_party_names] # renames to IND1 if only single IND


        # If missing_major, perform final reallocation using estimated model
        if election_year == 'Byelection':
            mapping_df = pd.read_csv('Missing_major_allocation.csv')

            if div in mapping_df['div_nm'].unique():
                combined, all_party_names = redistribute_major_sitout(combined,all_party_names,div, mapping_df)
                            
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




def expand_divisions_using_prior(sim, major_realloc_data, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=100):

    final_sim = {}
    party_name_dict = {} 

    multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
    C200_IND_positions_df = pd.read_csv("C200_IND_positions_df.csv", index_col=None)
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]
    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]

    Complex_contests = {'Calare': 0.4788,'Monash': 0.5002,'Moore': 0.43987} # proportions of split to C200 independent and Incumbent independent (based on % after adding Historical proportions in (div,div) pair)



    Major_parties = ['COAL','LP','NP','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']
    if election_year != '2016':
        if not CAGO_only:
            Major_parties += ['ON','UAPP','TOP']
            if election_year != '2025':
                Polling_parties += ['ON','UAPP']
            else:
                Polling_parties += ['ON','TOP']


    # For the split of COAL_double_divs
    NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)

    NP_ratios_curr = NP_ratios_curr.loc[(NP_ratios_curr['election_year']==election_year) ,] # & (NP_ratios_curr['State'].isin(['VIC','NSW']))
        

    for i, div in enumerate(Prior_estimates_dict.keys()): # will be alphabetical
        sim_block = sim[div]            # shape (10000, 4)
        main_parties = sim_block[:, :-1]     # shape (10000, 3)
        other_share = sim_block[:, -1]       # shape (10000,)

        prior_row = Prior_estimates_dict[div]

        prior_row_Other = prior_row[[p for p in prior_row.columns if p not in Major_parties]]
        minor_names = list(prior_row_Other.columns)
        all_party_names = Polling_parties + minor_names


        has_matrix = (major_realloc_data is not None) and (div in major_realloc_data)
        if has_matrix:
            prior_matrix, prior_names, ind_surge_dict = adjust_prior_to_matrix(prior_row, major_realloc_data[div], all_party_names)  
            all_party_names = prior_names  
        else:
            prior_matrix,ind_surge_dict = None,None
        #print(div)

        # Extract prior for this division

        if has_matrix:
            other_idx = [prior_names.index(m) for m in minor_names]
            rel_weights = prior_matrix[:, other_idx]
            rel_weights = rel_weights / rel_weights.sum(axis=1, keepdims=True)
        else:
            rel_weights = prior_row_Other.iloc[0].values
            rel_weights = rel_weights / rel_weights.sum()



        #import pdb;pdb.set_trace()

        # Dirichlet sampling
        if has_matrix:
            alphas = rel_weights * alpha_scalar
            splits = np.array([np.random.dirichlet(a) for a in alphas]) 
        else:
            alpha = rel_weights * alpha_scalar
            splits = np.random.dirichlet(alpha, size=sim[div].shape[0])

        # Expand 'Other' proportionally
        other_expanded = splits * other_share[:, None]  # (10000, n_minor)

        # Combine with main parties
        combined = np.concatenate([main_parties, other_expanded], axis=1)


        # if no ON/TOP, demove this column!
        if not CAGO_only:
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

            if has_matrix and ('NP' in prior_names) and ('LP' in prior_names):
                np_col, lp_col = prior_matrix[:, prior_names.index('NP')], prior_matrix[:, prior_names.index('LP')]
                NP_est = np_col / (lp_col + np_col) # (1000,) Array
            else:

                if 'NP' in prior_row.columns and 'LP' in prior_row.columns:
                    NP_est = (prior_row['NP'] /  prior_row[['LP','NP']].sum(axis=1)).iloc[0]
                else:
                    NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate'].iloc[0]

                # current hack for 2025 Nationals:
                if (election_year == '2025') & (div in ['Bullwinkel','Forrest',"O'Connor"]):
                    #import pdb;pdb.set_trace()
                    NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm']==div,'final_estimate'].iloc[0]



            alpha = np.array([1-NP_est,NP_est]) * LP_NP_DIR_ALPHA
            splits = np.random.dirichlet(alpha, size=sim[div].shape[0])
            LP_NP_votes = splits * COAL_votes[:, None]

            combined = np.concatenate([combined, LP_NP_votes], axis=1)[:,1:] # Removes 'COAL' from 1st column

            all_party_names = all_party_names[1:] + ['LP','NP'] # correct order - 'COAL' is always first in such case

        # perform independent split as well!
        if 'IND' in Prior_estimates_dict[div]:

            IND_index = all_party_names.index('IND')
            IND_votes = combined[:,IND_index] # should be just one
            
            if div in multiple_INDs_df['div_nm'].unique():
                # split according to C200%, remainder evenly! 
                #import pdb;pdb.set_trace()

                num_INDs_curr = multiple_INDs_df.loc[multiple_INDs_df['div_nm'] == div,'No_of_INDs']
                means_array_sim = np.tile(np.full(num_INDs_curr, 1/num_INDs_curr), (len(IND_votes), 1)) # initialises an even split



                if div in C200_IND_positions_curr['div_nm'].unique():
                    # split according to C200 splits!


                    beta0,beta1 = C200_ratio_lm_params
                    logit_mu = beta0 + beta1 * np.log(IND_votes*100)

                    C200_ratio = np.where(IND_votes < 0.05, 0.9, 1 / (1 + np.exp(-logit_mu)))


                    #C200_ratio = C200_IND_splits.loc[C200_IND_splits['Election'] == f'AverageFor{election_year}','Ratio'].iloc[0]

                    C200_IND_position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm'] == div,'Number'].iloc[0]

                    if (election_year == '2025') and div in Complex_contests.keys():

                        C200_ratio = np.full_like(C200_ratio, Complex_contests[div], dtype=float) # an array col
                    
                    K = num_INDs_curr.iloc[0]
                    non_ind_total = 1 - C200_ratio          # (n_sims,)
                    means_array_sim = np.tile((non_ind_total / (K - 1))[:, None],(1, K))
                    means_array_sim[:, C200_IND_position - 1] = C200_ratio

                #splits = np.random.dirichlet(means_array* alpha_scalar, size=sim.shape[0])
                splits = np.array([np.random.dirichlet(alpha_scalar * row) for row in means_array_sim])


                if has_matrix and ind_surge_dict is not None:
                    # Sum ratios across all specific IND keys
                    total_surge_ratio = np.sum(list(ind_surge_dict.values()), axis=0)
                    organic_ind = IND_votes * (1 - total_surge_ratio)

                    All_IND_votes = splits * organic_ind[:, np.newaxis]
                    
                    # Inject 100% of the surge into the primary Independent (C200 position)
                    generated_ind_names = ['IND'+str(j) for j in range(1, num_INDs_curr.iloc[0] + 1)]
                    for idx, label in enumerate(generated_ind_names):
                        if label in ind_surge_dict:
                            # Add the ratio-based surge amount
                            All_IND_votes[:, idx] += (IND_votes * ind_surge_dict[label])
                else:
                    All_IND_votes = splits * IND_votes[:, None]

                combined = np.concatenate([combined, All_IND_votes], axis=1)
                combined = np.delete(combined,IND_index, axis = 1) # remove 'IND' col

                all_party_names = all_party_names + ['IND'+str(i) for i in range(1,num_INDs_curr.iloc[0]+1)]
                all_party_names = [p for p in all_party_names if p != 'IND'] # remove IND

            else:
                all_party_names = [p if p != 'IND' else 'IND1' for p in all_party_names] # renames to IND1 if only single IND


        #assert combined.min() > 1e-40


                            
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


def allocate_major_sitout_to_cago(
    nat_sims_cago, 
    cago_names, 
    div, 
    missing_party, 
    mapping_df, 
    proportions_df, 
    MAJOR_SITOUT_DIR_ALPHA
):
    n_sims = nat_sims_cago.shape[0]
    
    # --- 1. Identify setup and Dirichlet params
    sitout_val_mapping = 'LP' if missing_party == 'COAL' else missing_party
    
    _, has_strong_IND = mapping_df.loc[
        mapping_df['div_nm'] == div, ['Major_sitout', 'has_strong_IND']
    ].drop_duplicates().to_numpy().ravel()
    
    row = proportions_df[
        (proportions_df['Major_sitout'] == sitout_val_mapping) & 
        (proportions_df['has_strong_IND'] == has_strong_IND)
    ].iloc[0]
    
    if has_strong_IND == 1:
        mu = np.array([row['theta_aligned'], row['theta_ind'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'IND', 'other_major', 'other']
    else:
        mu = np.array([row['theta_aligned'], row['theta_other_major'], row['theta_other']])
        groups = ['aligned', 'other_major', 'other']
        
    alpha = mu * MAJOR_SITOUT_DIR_ALPHA
    p_group = np.random.dirichlet(alpha, size=n_sims)
    
    # --- 2. Extract and remove sitout vote in CAGO matrix
    adjusted_cago = nat_sims_cago.copy()
    sitout_idx = cago_names.index(missing_party)
    V = adjusted_cago[:, sitout_idx].copy()
    adjusted_cago[:, sitout_idx] = 0.0
    
    # --- 3. Allocate to CAGO categories
    sub = mapping_df[
        (mapping_df['div_nm'] == div) & 
        (mapping_df['Major_sitout'] == missing_party) & 
        (mapping_df['has_strong_IND'] == has_strong_IND)
    ]
    
    for g_idx, g in enumerate(groups):
        parties = sub.loc[sub['group'] == g, 'PartyAb'].tolist()
        if not parties:
            continue
            
        # Map specific parties to their CAGO bucket
        # If the group contains a main party (e.g., GRN), it targets that. Otherwise, 'Other'.
        target_cago = next((p for p in parties if p in cago_names), 'Other')
        target_idx = cago_names.index(target_cago)
        
        adjusted_cago[:, target_idx] += V * p_group[:, g_idx]

    # --- 4. Delete missing party column
    adjusted_cago = np.delete(adjusted_cago, sitout_idx, axis=1)
    cago_names_reduced = [p for i, p in enumerate(cago_names) if i != sitout_idx]
    
    # Package the reallocation data to ensure perfect synchronicity later
    realloc_data = {
        'p_group': p_group,
        'groups': groups,
        'sub': sub,
        'has_strong_IND': has_strong_IND,
        'missing_party': missing_party
    }
    
    return adjusted_cago, cago_names_reduced, realloc_data



def transform_alr_matrix(error_matrix, missing_party):
    if missing_party == 'COAL':
        A = np.array([[-1, 1, 0],
                      [-1, 0, 1]])
        new_cols = ['GRN','Other']
        ref_col = 'ALP'
    elif missing_party == 'GRN':
        A = np.array([[1, 0, 0],
                      [0, 0, 1]])
        new_cols = ['ALP','Other']
        ref_col = 'COAL'
    elif missing_party == 'ALP':
        A = np.array([[0, 1, 0],
                      [0, 0, 1]])
        new_cols = ['GRN','Other']
        ref_col = 'COAL'
    else:
        raise ValueError("Unsupported party")

    # error_matrix shape: (3, 3)
    new_error_matrix = A @ error_matrix @ A.T  # → (2, 2)

    return new_error_matrix, new_cols, ref_col

def simulate_combined_models_CAGO(n_simulations, election_year, prior_df_active_dict, polling_avg_dict, National_prior_dict, byelection_group_structure, seat_alr_bases_dict, n_seat_polls_dict, Nat_poll_covm_dict, Electorate_residual_covm_dict, Seat_poll_covm_dict, GLOBAL_CSVs, prior_alpha=22, B = 1):
    
    if election_year == 'Byelection':
        # for byleeciton, take it group by group, ensuring they are correctly ordered

        byelection_group_sims = []


        for i in range(len(byelection_group_structure)):

            item = byelection_group_structure[i]

            group, polling_avg, prior_df_curr, National_prior = (
                item['group'],
                item['polling_avg'],
                item['prior_df_active'],   # rename happens here
                item['National_prior']
            )
            
            sim = simulate_Polling_Fundamentals_model(
                n_simulations=n_simulations,
                election_year=election_year,
                df_t=0,
                v=0.15,
                beta=0.5,
                Prior_estimates_df=prior_df_curr,
                polling_avg = polling_avg,
                National_prior = National_prior,
                National_Polling_error_cov = Nat_poll_covm_dict[election_year],
                Electorate_Residuals_cov = Electorate_residual_covm_dict[election_year],
                For_combined_model=True
            )

            byelection_group_sims.append(sim)
        
        nat_alr_sims = np.concatenate(byelection_group_sims, axis=1)

    else:
        polling_avg = polling_avg_dict[election_year]
        prior_df_active = prior_df_active_dict[election_year]
        National_prior = National_prior_dict[election_year]

        nat_alr_sims = simulate_Polling_Fundamentals_model(
            n_simulations=n_simulations,
            election_year=election_year,
            df_t=0,
            v=0.15,
            beta=0.5,
            Prior_estimates_df=prior_df_active,
            polling_avg = polling_avg,
            National_prior = National_prior,
            National_Polling_error_cov = Nat_poll_covm_dict[election_year],
            Electorate_Residuals_cov = Electorate_residual_covm_dict[election_year],
            For_combined_model=True
        )



    combined_sims = {}
    major_realloc_data = {} # for missing major party in byelections

    for i in range(nat_alr_sims.shape[1]): 

        nat_alr_sims_curr = nat_alr_sims[:,i,:]
        div = prior_df_active_dict[election_year].iloc[[i]].index[0][:]
        


        nat_poll_cov = Nat_poll_covm_dict[election_year].copy()
        resid_poll_cov = Electorate_residual_covm_dict[election_year].copy()
        n_seat_polls = n_seat_polls_dict[election_year][div]
        seat_poll_cov = Seat_poll_covm_dict[election_year].copy() * SEAT_POLL_SCALE[n_seat_polls]


        if election_year == 'Byelection':

            # get missing party if exists
            mapping_df = GLOBAL_CSVs['mapping_df']

        

            # missing_party logic 
            if div in mapping_df['div_nm'].unique():
                # allocate missing party vote

                missing_party = mapping_df.loc[mapping_df['div_nm']==div,['Major_sitout']].iloc[0,0]

                nat_sims = alr_to_simplex_simulation_array(nat_alr_sims_curr)

                cago_names = ['COAL', 'ALP', 'GRN', 'Other']
                # 1. Apply major reallocation to CAGO and save state
                nat_sims_adj, new_cago_cols, realloc_data = allocate_major_sitout_to_cago(
                    nat_sims, cago_names, div, missing_party, mapping_df , GLOBAL_CSVs['proportions_df'], MAJOR_SITOUT_DIR_ALPHA
                )

                # transform ALR covms
                nat_poll_cov, new_cols, ref_col = transform_alr_matrix(nat_poll_cov, missing_party)
                resid_poll_cov, new_cols, ref_col = transform_alr_matrix(resid_poll_cov, missing_party)
                seat_poll_cov, new_cols, ref_col = transform_alr_matrix(seat_poll_cov, missing_party)

                # re-convert to alr
                current_cols = [ref_col] + new_cols # ref_col as ALP/COAL always first
                numerator_idxs = [current_cols.index(c) for c in new_cols]
                denominator_idx = current_cols.index(ref_col)

                nat_alr_sims_curr = np.log(nat_sims_adj[:, numerator_idxs] / nat_sims_adj[:, [denominator_idx]]) 

                major_realloc_data[div] = realloc_data

        combined_alr = combine_models_alr_precision_weighted(
                n_simulations,
                B,
                nat_alr_sims_curr, # Simulated_Electorate_Polling_Results_ALR for this div (n_sims, 3)
                seat_alr_bases_dict[election_year][div], # alr_base from precomputed_dict (3,)
                nat_poll_cov, # National_Polling_error_cov (3, 3)
                resid_poll_cov,    # Electorate_Residuals_cov (3, 3)
                seat_poll_cov # Seat_poll_covm_dict[election_year] (3, 3)
            )
    
        combined_simplex = alr_to_simplex_simulation_array(combined_alr)
        combined_sims[div] = combined_simplex


    return combined_sims, major_realloc_data





def simulate_nat_polling_full_model(n_simulations, election_year, prior_df_active_dict, polling_avg_dict, National_prior_dict, Prior_estimates_dict_active_per_election, Results_dict_active_per_election, byelection_group_structure, Nat_poll_covm_dict, Electorate_residual_covm_dict, prior_alpha=22):



    if election_year == 'Byelection':
        # for byleeciton, take it group by group, ensuring they are correctly ordered

        byelection_group_sims = []


        for i in range(len(byelection_group_structure)):

            item = byelection_group_structure[i]

            group, polling_avg, prior_df_curr, National_prior = (
                item['group'],
                item['polling_avg'],
                item['prior_df_active'],   # rename happens here
                item['National_prior']
            )
            
            sim = simulate_Polling_Fundamentals_model(
                n_simulations=n_simulations,
                election_year=election_year,
                df_t=0,
                v=0.15,
                beta=0.5,
                Prior_estimates_df=prior_df_curr,
                polling_avg = polling_avg,
                National_prior = National_prior,
                National_Polling_error_cov = Nat_poll_covm_dict[election_year],
                Electorate_Residuals_cov = Electorate_residual_covm_dict[election_year]
            )

            byelection_group_sims.append(sim)
        
        sim = np.concatenate(byelection_group_sims, axis=1)

    else:
        polling_avg = polling_avg_dict[election_year]
        prior_df_active = prior_df_active_dict[election_year]
        National_prior = National_prior_dict[election_year]

        sim = simulate_Polling_Fundamentals_model(
            n_simulations=n_simulations,
            election_year=election_year,
            df_t=0,
            v=0.15,
            beta=0.5,
            Prior_estimates_df=prior_df_active,
            polling_avg = polling_avg,
            National_prior = National_prior
        )


    Prior_estimates_dict_active = Prior_estimates_dict_active_per_election[election_year]
    Results_dict_active = Results_dict_active_per_election[election_year]


    final_sim, party_name_dict = expand_all_divisions_from_prior_df(sim, Prior_estimates_dict_active, Results_dict_active, election_year, alpha_scalar=prior_alpha)


    return final_sim, party_name_dict





# seat polling simulate



def transform_alr_array(error_matrix, missing_party):
    if missing_party == 'COAL':
        A = np.array([[-1, 1, 0],
                      [-1, 0, 1]])
        new_cols = ['GRN','Other']
        ref_col = 'ALP'
    elif missing_party == 'GRN':
        A = np.array([[1, 0, 0],
                      [0, 0, 1]])
        new_cols = ['ALP','Other']
        ref_col = 'COAL'
    elif missing_party == 'ALP':
        A = np.array([[0, 1, 0],
                      [0, 0, 1]])
        new_cols = ['GRN','Other']
        ref_col = 'COAL'
    else:
        raise ValueError("Unsupported party")

    # error_matrix shape: (3, 3)
    new_error_matrix = error_matrix @ A.T  # → (2, 2)

    return new_error_matrix, new_cols, ref_col



def calculate_rigorous_split(poll_minors, cago_other):
    minor_cols = [c for c in poll_minors.columns if c != 'OTH']
    naive_means = poll_minors[minor_cols].replace(0, np.nan).mean().fillna(0)
    
    w_oth = poll_minors['OTH'].replace(0, np.nan).min() or 0.01 
    missing_mask = (poll_minors[minor_cols] == 0)
    row_weights = missing_mask.dot(naive_means)
    
    alloc_factor = np.where(
        (poll_minors['OTH'] > 0) & (row_weights > 0),
        poll_minors['OTH'] / (row_weights + w_oth),
        0
    )
    
    imputed = poll_minors.copy()
    for col in minor_cols:
        imputed[col] += missing_mask[col] * alloc_factor * naive_means[col]
    
    final_split = imputed[minor_cols].mean()
    final_split['OTH'] = cago_other - final_split.sum()
    
    return final_split


def precompute_division(poll_df,
    prior_df,
    div,
    election_year,
    missing_party = None,
    ref_col='COAL',
    GLOBAL_CSVs = {}
):

    # compute CAGO_row
    minor_cols = [p for p in prior_df.columns if p not in ['ALP','COAL','LP','NP','NAT','LNP','CLP','GRN']]

    curr_row = poll_df.iloc[:,2+('byelection_year' in poll_df.columns):]
    curr_row_normalised = curr_row/curr_row.sum(axis=1)[div]
    CAGO_row = group_into_Fundamentals_Categories(curr_row_normalised, div).mean()

    # add alr error and return to prop space, accounting for missing major party
    if missing_party is not None:
        CAGO_row = CAGO_row.drop([missing_party])

    poll_minor_split = curr_row.mean().loc[[p for p in curr_row.mean().index if p not in ['ALP','COAL','LP','NP','NAT','LNP','CLP','GRN']]]     
    poll_minor_split = poll_minor_split[poll_minor_split>0]   
    poll_weights = poll_minor_split / poll_minor_split.sum()

    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]
    C200_IND_splits = GLOBAL_CSVs["C200_IND_splits"]
    C200_IND_positions_df = GLOBAL_CSVs["C200_IND_positions_df"]
    NP_ratios_curr = GLOBAL_CSVs["NP_ratios_curr"]
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]

    multi_ind_c200_case = ((div in multiple_INDs_df['div_nm'].values) and (div in C200_IND_positions_curr['div_nm'].values))
    Position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm']==div,'Number'].iloc[0] if multi_ind_c200_case else 1

    k = int(multiple_INDs_df.loc[multiple_INDs_df['div_nm'] == div,'No_of_INDs'].iloc[0]) if div in multiple_INDs_df['div_nm'].unique() else 1

    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    C200_ratio_lm_params = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]

    NP_ratios_curr = NP_ratios_curr.loc[NP_ratios_curr['election_year'] == election_year]

    
    return {
        "alr_base": np.log(CAGO_row.drop(ref_col) / CAGO_row[ref_col]),
        "poll_weights": (poll_weights).rename({'Other':'OTH'}),
        "minor_cols": minor_cols,
        "multi_ind_c200_case": multi_ind_c200_case,
        "Position": Position if multi_ind_c200_case else None,
        "k": k,
        'C200_ratio_lm_params': C200_ratio_lm_params,
        "C200_IND_positions_curr":C200_IND_positions_curr,
        "NP_ratios_curr": NP_ratios_curr
    }

def process_division(
    poll_df,
    prior_row,
    Results_row,
    div,
    election_year,
    missing_party = None,
    ref_col='COAL',
    prior_alpha=22,
    poll_alpha = 22,
    precomputed_dict = {},
    GLOBAL_CSVs = {},
    n_simulations = 1000
):
    
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]
    

    alr_base = precomputed_dict['alr_base']
    poll_weights = precomputed_dict['poll_weights']
    minor_cols = precomputed_dict['minor_cols']
    multi_ind_c200_case = precomputed_dict['multi_ind_c200_case'] # currently C200 INDs comprise all such cases
    Position = precomputed_dict['Position']
    k = precomputed_dict['k']
    beta0,beta1 = precomputed_dict['C200_ratio_lm_params']
    C200_IND_positions_curr = precomputed_dict['C200_IND_positions_curr']
    NP_ratios_curr = precomputed_dict['NP_ratios_curr']

    prior_df = prior_row.copy()

   

    alr_error_matrix = np.random.multivariate_normal(
        mean=np.zeros(3),
        cov=Seat_poll_covm_dict[election_year],
        size=n_simulations
    )

    if election_year == 'Byelection' and missing_party is not None:
        if poll_df[missing_party].sum() == 0:
            alr_error_matrix, new_cols, ref_col = transform_alr_array(alr_error_matrix, missing_party)
    else:
        new_cols, ref_col = ['ALP','GRN','Other'], 'COAL'

    alr_adj = alr_base.values + alr_error_matrix
    adjusted_cago = alr_to_simplex_vectorised(alr_adj, ref_col, new_cols)[0]
    cago_party_names = [ref_col] + new_cols
    oth_idx = cago_party_names.index('Other')

    macro_oth = adjusted_cago[:, oth_idx] 


    alpha_curr = (poll_weights.values * poll_alpha).astype(float)
    poll_draws = np.random.dirichlet(alpha_curr, size=n_simulations)
    poll_alloc_matrix = poll_draws * macro_oth[:, np.newaxis]

    poll_names = list(poll_weights.index)
    unpolled_minor_cols = [p for p in minor_cols if p not in poll_names]

    if multi_ind_c200_case:
        ind_idx = poll_names.index('IND')
        
        prior_ind_val = prior_df.loc[div, 'IND']
        if prior_ind_val < 0.05:
            c200_ratio = 0.9
        else:
            logit_mu = beta0 + beta1 * np.log(prior_ind_val * 100)
            c200_ratio = 1 / (1 + np.exp(-logit_mu))

        prior_df.loc[div, 'IND'] = prior_ind_val * (1 - c200_ratio)
        
        poll_names[ind_idx] = f'IND{Position}'
        
        ind_reset_col = np.zeros((n_simulations, 1))
        poll_alloc_matrix = np.hstack([poll_alloc_matrix, ind_reset_col])
        poll_names.append('IND')
        unpolled_minor_cols += ['IND']


    if election_year == 'Byelection' and div == 'Wentworth':
        ind_idx = poll_names.index('IND')
        ind_shares = poll_alloc_matrix[:, ind_idx]
        
        WENTWORTH_IND_RATIO = 20.974 / (6.435 + 20.974)

        ind1_col = (ind_shares * WENTWORTH_IND_RATIO)[:, np.newaxis]
        ind2_col = (ind_shares * (1 - WENTWORTH_IND_RATIO))[:, np.newaxis]
        ind_reset_col = np.zeros((n_simulations, 1))

        poll_alloc_matrix = np.delete(poll_alloc_matrix, ind_idx, axis=1)
        poll_names.pop(ind_idx)

        poll_alloc_matrix = np.hstack([poll_alloc_matrix, ind1_col, ind2_col, ind_reset_col])
        poll_names += ['IND1', 'IND2', 'IND']
        unpolled_minor_cols += ['IND']



    if 'OTH' in poll_names:
        oth_in_poll_idx = poll_names.index('OTH')
        remaining_oth = poll_alloc_matrix[:, oth_in_poll_idx]
        
        if len(unpolled_minor_cols) > 0 and np.any(remaining_oth > 0):
            prior_minors = prior_df.loc[div, unpolled_minor_cols]
            prior_weights = prior_minors / prior_minors.sum()
            alpha_prior = (prior_weights.values * prior_alpha).astype(float)
            
            draw_prior = np.random.dirichlet(alpha_prior, size=n_simulations)
            prior_alloc_matrix = draw_prior * remaining_oth[:, np.newaxis]
            
            poll_alloc_no_oth = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
            poll_names_no_oth = [p for p in poll_names if p != 'OTH']
            
            minor_party_block = np.hstack([poll_alloc_no_oth, prior_alloc_matrix])
            all_minor_names = poll_names_no_oth + unpolled_minor_cols
        else:
            minor_party_block = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
            all_minor_names = [p for p in poll_names if p != 'OTH']
    else:
        minor_party_block = poll_alloc_matrix
        all_minor_names = poll_names

    main_party_block = np.delete(adjusted_cago, oth_idx, axis=1)
    main_party_names = [p for p in cago_party_names if p != 'Other']

    final_array = np.hstack([main_party_block, minor_party_block])
    all_party_names = main_party_names + all_minor_names

    assert np.isclose(final_array.sum(), n_simulations)


    if ('LP' in Results_row.columns) and ('NP' in Results_row.columns):
        NP_LP_split_polled = ('NAT' in poll_df.columns) and poll_df['NAT'].sum() > 0

        if NP_LP_split_polled:
            nat = poll_df.loc[div,'NAT'].astype(float)
            coal = poll_df.loc[div,'COAL'].astype(float)

            
            total = nat + coal
            NP_est = nat / total if total > 0 else 0.5

        else:
            if div in NP_ratios_curr['div_nm'].unique():
                if 'NP' in prior_df.columns and 'LP' in prior_df.columns:
                    NP_est = (prior_df.loc[div, 'NP'] / prior_df.loc[div, ['LP', 'NP']].sum())
                else:
                    NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm'] == div, 'final_estimate'].iloc[0]

                if (election_year == '2025') and (div in ['Bullwinkel', 'Forrest', "O'Connor"]):
                    NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm'] == div, 'final_estimate'].iloc[0]
            else:
                NP_est = 0.5

        coal_idx = all_party_names.index('COAL')
        coal_votes = final_array[:, coal_idx]

        alpha_lp_np = np.array([1 - NP_est, NP_est]) * LP_NP_DIR_ALPHA_POLL # Fix - polling probably more accurate!
        lp_np_splits = np.random.dirichlet(alpha_lp_np, size=n_simulations)
        
        lp_votes = lp_np_splits[:, 0] * coal_votes
        np_votes = lp_np_splits[:, 1] * coal_votes

        final_array = np.delete(final_array, coal_idx, axis=1)
        all_party_names.pop(coal_idx)

        final_array = np.hstack([final_array, lp_votes[:, np.newaxis], np_votes[:, np.newaxis]])
        all_party_names += ['LP', 'NP']



    if 'IND' in all_party_names:
        ind_idx = all_party_names.index('IND')
        ind_total = final_array[:, ind_idx]

        if multi_ind_c200_case:
            free_weights = np.full(k - 1, 1 / (k - 1))
            ind_draws = np.random.dirichlet(free_weights * prior_alpha, size=n_simulations)
            ind_split_votes = ind_draws * ind_total[:, np.newaxis]

            final_array = np.delete(final_array, ind_idx, axis=1)
            all_party_names.pop(ind_idx)

            idx = 0
            for i in range(1, k + 1):
                if i == Position:
                    continue
                label = f'IND{i if i < Position else i}'
                final_array = np.hstack([final_array, ind_split_votes[:, [idx]]])
                all_party_names.append(label)
                idx += 1
                
        elif election_year == 'Byelection' and div == 'Wentworth':
            all_party_names[ind_idx] = 'IND3'

        else:
            if div in multiple_INDs_df['div_nm'].unique():
                means_array = np.full(k, 1 / k)

                if div in C200_IND_positions_curr['div_nm'].unique():
                    pos = int(C200_IND_positions_curr.loc[
                        C200_IND_positions_curr['div_nm'] == div, 'Number'
                    ].iloc[0]) - 1

                    # Vectorized ratio calculation
                    c200_ratios = np.where(
                        ind_total * 100 < 5, 
                        0.9, 
                        1 / (1 + np.exp(-(beta0 + beta1 * np.log(ind_total * 100))))
                    )

                    # Build mean matrix for Dirichlet (n_sims, k)
                    means_matrix = np.tile((1 - c200_ratios)[:, np.newaxis] / (k - 1), (1, k))
                    means_matrix[:, pos] = c200_ratios
                else:
                    means_matrix = np.tile(means_array, (n_simulations, 1))

                # Draw for each simulation. Note: np.random.dirichlet is not fully 
                # vectorized for unique alpha rows, so we use a list comprehension or loop

                splits = np.array([np.random.dirichlet(m * prior_alpha) for m in means_matrix])
                ind_split_votes = splits * ind_total[:, np.newaxis]

                final_array = np.delete(final_array, ind_idx, axis=1)
                all_party_names.pop(ind_idx)

                for i in range(k):
                    final_array = np.hstack([final_array, ind_split_votes[:, [i]]])
                    all_party_names.append(f'IND{i+1}')

            else:
                all_party_names[ind_idx] = 'IND1'

    # Final Coalition Renaming
    if ('COAL' in all_party_names) and (missing_party != 'COAL'):
        coal_idx = all_party_names.index('COAL')
        coal_replacement_list = ['LP', 'NP', 'LNP', 'CLP']
        coal_party = next(p for p in Results_row.columns if p in coal_replacement_list)
        all_party_names[coal_idx] = coal_party

    # Final mapping to match Results_row column order
    ballot_order = Results_row.columns.tolist()
    name_to_idx = {name: i for i, name in enumerate(all_party_names)}
    col_indices = [name_to_idx[name] for name in ballot_order]
    
    return final_array[:, col_indices]


def expand_divisions_using_seat_polls_old(
    sims,
    curr_seat_polls,
    Results_dict_active,
    election_year,
    prior_alpha=22,
    poll_alpha = 22,
    GLOBAL_CSVs = {},
    n_simulations = 1000
):
    
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]
    C200_IND_positions_df = GLOBAL_CSVs["C200_IND_positions_df"]
    NP_ratios_curr = GLOBAL_CSVs["NP_ratios_curr"]
    NP_ratios_curr = NP_ratios_curr.loc[NP_ratios_curr['election_year'] == election_year]
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]


    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    beta0,beta1 = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]

    prior_long = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv")

    seat_output_dict = {}

    
    for div in sims.keys():

        if prior_long.loc[prior_long['div_nm']==div,].empty: # e.g., Canberra 1995
            continue
        prior_row = prior_long.loc[prior_long['div_nm']==div,][['PartyAb','FP_Votes']].set_index('PartyAb').T.rename(index = {'FP_Votes':div})
        
        missing_party = next((p for p in ['COAL','ALP','GRN'] if prior_long.loc[div, p] == 0),None)

        ref_col='COAL' if missing_party != 'COAL' else 'ALP'
        sims_cols = [p for p in ['COAL','ALP','GRN','OTH'] if p != missing_party]
    
        poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate']==div].set_index('Electorate')
        Results_row = Results_dict_active[election_year][div]

        
        multi_ind_c200_case = ((div in multiple_INDs_df['div_nm'].values) and (div in C200_IND_positions_curr['div_nm'].values))
        Position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm']==div,'Number'].iloc[0] if multi_ind_c200_case else 1
        k = int(multiple_INDs_df.loc[multiple_INDs_df['div_nm'] == div,'No_of_INDs'].iloc[0]) if div in multiple_INDs_df['div_nm'].unique() else 1


        # compute CAGO_row
        minor_cols = [p for p in prior_df.columns if p not in ['ALP','COAL','LP','NP','NAT','LNP','CLP','GRN']]

        curr_row = poll_df.iloc[:,2+('byelection_year' in poll_df.columns):]
        curr_row_normalised = curr_row/curr_row.sum(axis=1)[div]
        CAGO_row = group_into_Fundamentals_Categories(curr_row_normalised, div).mean()

        # add alr error and return to prop space, accounting for missing major party
        if missing_party is not None:
            CAGO_row = CAGO_row.drop([missing_party])

        poll_minor_split = curr_row.mean().loc[[p for p in curr_row.mean().index if p not in ['ALP','COAL','LP','NP','NAT','LNP','CLP','GRN']]]     
        poll_minor_split = poll_minor_split[poll_minor_split>0]   
        poll_weights = poll_minor_split / poll_minor_split.sum()

        prior_df = prior_row.copy()
    


        adjusted_cago = sims[div]
        cago_party_names = [ref_col] + sims_cols
        oth_idx = cago_party_names.index('Other')

        macro_oth = adjusted_cago[:, oth_idx]


        alpha_curr = (poll_weights.values * poll_alpha).astype(float)
        poll_draws = np.random.dirichlet(alpha_curr, size=n_simulations)
        poll_alloc_matrix = poll_draws * macro_oth[:, np.newaxis]

        poll_names = list(poll_weights.index)
        unpolled_minor_cols = [p for p in minor_cols if p not in poll_names]

        if multi_ind_c200_case:
            ind_idx = poll_names.index('IND')
            
            prior_ind_val = prior_df.loc[div, 'IND']
            if prior_ind_val < 0.05:
                c200_ratio = 0.9
            else:
                logit_mu = beta0 + beta1 * np.log(prior_ind_val * 100)
                c200_ratio = 1 / (1 + np.exp(-logit_mu))

            c200_ratio = np.where(prior_ind_val < 0.05, 0.9, 1 / (1 + np.exp(-logit_mu)))


            prior_df.loc[div, 'IND'] = prior_ind_val * (1 - c200_ratio)
            
            poll_names[ind_idx] = f'IND{Position}'
            
            ind_reset_col = np.zeros((n_simulations, 1))
            poll_alloc_matrix = np.hstack([poll_alloc_matrix, ind_reset_col])
            poll_names.append('IND')
            unpolled_minor_cols += ['IND']


        if election_year == 'Byelection' and div == 'Wentworth':
            ind_idx = poll_names.index('IND')
            ind_shares = poll_alloc_matrix[:, ind_idx]
            
            WENTWORTH_IND_RATIO = 20.974 / (6.435 + 20.974)

            ind1_col = (ind_shares * WENTWORTH_IND_RATIO)[:, np.newaxis]
            ind2_col = (ind_shares * (1 - WENTWORTH_IND_RATIO))[:, np.newaxis]
            ind_reset_col = np.zeros((n_simulations, 1))

            poll_alloc_matrix = np.delete(poll_alloc_matrix, ind_idx, axis=1)
            poll_names.pop(ind_idx)

            poll_alloc_matrix = np.hstack([poll_alloc_matrix, ind1_col, ind2_col, ind_reset_col])
            poll_names += ['IND1', 'IND2', 'IND']
            unpolled_minor_cols += ['IND']



        if 'OTH' in poll_names:
            oth_in_poll_idx = poll_names.index('OTH')
            remaining_oth = poll_alloc_matrix[:, oth_in_poll_idx]
            
            if len(unpolled_minor_cols) > 0 and np.any(remaining_oth > 0):
                prior_minors = prior_df.loc[div, unpolled_minor_cols]
                prior_weights = prior_minors / prior_minors.sum()
                alpha_prior = (prior_weights.values * prior_alpha).astype(float)
                
                draw_prior = np.random.dirichlet(alpha_prior, size=n_simulations)
                prior_alloc_matrix = draw_prior * remaining_oth[:, np.newaxis]
                
                poll_alloc_no_oth = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
                poll_names_no_oth = [p for p in poll_names if p != 'OTH']
                
                minor_party_block = np.hstack([poll_alloc_no_oth, prior_alloc_matrix])
                all_minor_names = poll_names_no_oth + unpolled_minor_cols
            else:
                minor_party_block = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
                all_minor_names = [p for p in poll_names if p != 'OTH']
        else:
            minor_party_block = poll_alloc_matrix
            all_minor_names = poll_names

        main_party_block = np.delete(adjusted_cago, oth_idx, axis=1)
        main_party_names = [p for p in cago_party_names if p != 'Other']

        final_array = np.hstack([main_party_block, minor_party_block])
        all_party_names = main_party_names + all_minor_names

        assert np.isclose(final_array.sum(), n_simulations)


        if ('LP' in Results_row.columns) and ('NP' in Results_row.columns):
            NP_LP_split_polled = ('NAT' in poll_df.columns) and poll_df['NAT'].sum() > 0

            if NP_LP_split_polled:
                nat = poll_df.loc[div,'NAT'].astype(float)
                lib = poll_df.loc[div,'COAL'].astype(float)
                
                coal_total = nat + lib
                NP_est = nat / coal_total if coal_total > 0 else 0.5
            else:
                if div in NP_ratios_curr['div_nm'].unique():
                    if 'NP' in prior_df.columns and 'LP' in prior_df.columns:
                        NP_est = (prior_df.loc[div, 'NP'] / prior_df.loc[div, ['LP', 'NP']].sum())
                    else:
                        NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm'] == div, 'final_estimate'].iloc[0]

                    if (election_year == '2025') and (div in ['Bullwinkel', 'Forrest', "O'Connor"]):
                        NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm'] == div, 'final_estimate'].iloc[0]
                else:
                    NP_est = 0.5

            coal_idx = all_party_names.index('COAL')
            coal_votes = final_array[:, coal_idx]

            alpha_lp_np = np.array([1 - NP_est, NP_est]) * LP_NP_DIR_ALPHA_POLL # Fix - polling probably more accurate!
            lp_np_splits = np.random.dirichlet(alpha_lp_np, size=n_simulations)
            
            lp_votes = lp_np_splits[:, 0] * coal_votes
            np_votes = lp_np_splits[:, 1] * coal_votes

            final_array = np.delete(final_array, coal_idx, axis=1)
            all_party_names.pop(coal_idx)

            final_array = np.hstack([final_array, lp_votes[:, np.newaxis], np_votes[:, np.newaxis]])
            all_party_names += ['LP', 'NP']



        if 'IND' in all_party_names:
            ind_idx = all_party_names.index('IND')
            ind_total = final_array[:, ind_idx]

            if multi_ind_c200_case:
                free_weights = np.full(k - 1, 1 / (k - 1))
                ind_draws = np.random.dirichlet(free_weights * prior_alpha, size=n_simulations)
                ind_split_votes = ind_draws * ind_total[:, np.newaxis]

                final_array = np.delete(final_array, ind_idx, axis=1)
                all_party_names.pop(ind_idx)

                idx = 0
                for i in range(1, k + 1):
                    if i == Position:
                        continue
                    label = f'IND{i if i < Position else i}'
                    final_array = np.hstack([final_array, ind_split_votes[:, [idx]]])
                    all_party_names.append(label)
                    idx += 1
                    
            elif election_year == 'Byelection' and div == 'Wentworth':
                all_party_names[ind_idx] = 'IND3'

            else:
                if div in multiple_INDs_df['div_nm'].unique():
                    means_array = np.full(k, 1 / k)

                    if div in C200_IND_positions_curr['div_nm'].unique():
                        pos = int(C200_IND_positions_curr.loc[
                            C200_IND_positions_curr['div_nm'] == div, 'Number'
                        ].iloc[0]) - 1

                        # Vectorized ratio calculation
                        c200_ratios = np.where(
                            ind_total * 100 < 5, 
                            0.9, 
                            1 / (1 + np.exp(-(beta0 + beta1 * np.log(ind_total * 100))))
                        )

                        # Build mean matrix for Dirichlet (n_sims, k)
                        means_matrix = np.tile((1 - c200_ratios)[:, np.newaxis] / (k - 1), (1, k))
                        means_matrix[:, pos] = c200_ratios
                    else:
                        means_matrix = np.tile(means_array, (n_simulations, 1))

                    # Draw for each simulation. Note: np.random.dirichlet is not fully 
                    # vectorized for unique alpha rows, so we use a list comprehension or loop
                    splits = np.array([np.random.dirichlet(m * prior_alpha) for m in means_matrix])
                    ind_split_votes = splits * ind_total[:, np.newaxis]

                    final_array = np.delete(final_array, ind_idx, axis=1)
                    all_party_names.pop(ind_idx)

                    for i in range(k):
                        final_array = np.hstack([final_array, ind_split_votes[:, [i]]])
                        all_party_names.append(f'IND{i+1}')

                else:
                    all_party_names[ind_idx] = 'IND1'

        # Final Coalition Renaming
        if ('COAL' in all_party_names) and (missing_party != 'COAL'):
            coal_idx = all_party_names.index('COAL')
            coal_replacement_list = ['LP', 'NP', 'LNP', 'CLP']
            coal_party = next(p for p in Results_row.columns if p in coal_replacement_list)
            all_party_names[coal_idx] = coal_party

        # Final mapping to match Results_row column order
        ballot_order = Results_row.columns.tolist()
        name_to_idx = {name: i for i, name in enumerate(all_party_names)}
        col_indices = [name_to_idx[name] for name in ballot_order]


        seat_output_dict[div] = final_array[:, col_indices]
    
    return seat_output_dict




def expand_divisions_using_seat_polls(
    sims,
    curr_seat_polls,
    Prior_estimates_dict,
    Results_dict_active,
    election_year,
    prior_alpha=22,
    poll_alpha = 22,
    major_realloc_data = {},
    GLOBAL_CSVs = {},
    n_simulations = 1000
):
    
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]
    C200_IND_positions_df = GLOBAL_CSVs["C200_IND_positions_df"]
    NP_ratios_curr = GLOBAL_CSVs["NP_ratios_curr"]
    NP_ratios_curr = NP_ratios_curr.loc[NP_ratios_curr['election_year'] == election_year]
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == election_year,]

    C200_ratio_lm_params = pd.read_csv("C200_ratio_lm_params.csv")
    beta0,beta1 = C200_ratio_lm_params.loc[C200_ratio_lm_params['election_year']==election_year,['beta0','beta1']].values[0]
    multiple_INDs_df = GLOBAL_CSVs["multiple_INDs_df"][election_year]

    Major_parties = ['COAL','LP','NP','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']


    seat_output_dict = {}
    party_name_dict = {}
    
    for div in sims.keys():

        prior_row = Prior_estimates_dict[div]
        prior_row.index = [div]


        prior_row_Other = prior_row[[p for p in prior_row.columns if p not in Major_parties]]
        minor_names = list(prior_row_Other.columns)
        all_party_names = Polling_parties + minor_names

        has_matrix = (major_realloc_data is not None) and (div in major_realloc_data)
        if has_matrix:
            adjusted_prior = adjust_prior_to_matrix(prior_row, major_realloc_data[div], all_party_names)
            prior_matrix, prior_names,ind_surge_dict = adjusted_prior
        else:
            prior_matrix, prior_names, ind_surge_dict = None, None, None

        missing_party = major_realloc_data[div]['missing_party'] if div in major_realloc_data.keys() else None

        ref_col='COAL' if missing_party != 'COAL' else 'ALP'
        sims_cols = [p for p in ['COAL','ALP','GRN','Other'] if p != missing_party]
    
        poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate']==div].set_index('Electorate')
        Results_row = Results_dict_active[div]

        
        multi_ind_c200_case = ((div in multiple_INDs_df['div_nm'].values) and (div in C200_IND_positions_curr['div_nm'].values))
        Position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm']==div,'Number'].iloc[0] if multi_ind_c200_case else 1
        k = int(multiple_INDs_df.loc[multiple_INDs_df['div_nm'] == div,'No_of_INDs'].iloc[0]) if div in multiple_INDs_df['div_nm'].unique() else 1

        # compute CAGO_row
        prior_df = prior_row.copy()
        minor_cols = [p for p in prior_df.columns if p not in Major_parties]

        curr_row = poll_df.iloc[:,2+('byelection_year' in poll_df.columns):]
        curr_row_normalised = curr_row/curr_row.sum(axis=1)[div]
        CAGO_row = group_into_Fundamentals_Categories(curr_row_normalised, div).mean()

        # add alr error and return to prop space, accounting for missing major party
        if missing_party is not None:
            CAGO_row = CAGO_row.drop([missing_party])

        poll_minor_split = curr_row.mean().loc[[p for p in curr_row.mean().index if p not in Major_parties]]     
        poll_minor_split = poll_minor_split[poll_minor_split>0]   
        poll_weights = poll_minor_split / poll_minor_split.sum()
        
        adjusted_cago = sims[div]
        cago_party_names = sims_cols
        oth_idx = cago_party_names.index('Other') # CHECK if it should be 'Other'
        
        macro_oth = adjusted_cago[:, oth_idx]


        alpha_curr = (poll_weights.values * poll_alpha).astype(float)
        poll_draws = np.random.dirichlet(alpha_curr, size=n_simulations)
        poll_alloc_matrix = poll_draws * macro_oth[:, np.newaxis]

        poll_names = list(poll_weights.index)
        unpolled_minor_cols = [p for p in minor_cols if p not in poll_names]

        if has_matrix:
            prior_ind_remainder = prior_matrix[:, prior_names.index('IND')]
        else:
            # Safe access to prior_df; if 'IND' isn't there, default to 0.0
            prior_ind_remainder = prior_df.loc[div, 'IND'] if 'IND' in prior_df.columns else 0.0

        if multi_ind_c200_case:
            ind_idx = poll_names.index('IND')
            
            # CHANGE: Pull IND prior from matrix if available
            if has_matrix:
                prior_ind_val = prior_matrix[:, prior_names.index('IND')]
            else:
                prior_ind_val = prior_df.loc[div, 'IND']
            
            # Use vectorized np.where for safe calculation
            safe_ind_val = np.maximum(prior_ind_val * 100, 1e-9)
            logit_mu = beta0 + beta1 * np.log(safe_ind_val)
            c200_ratio = np.where(prior_ind_val < 0.05, 0.9, 1 / (1 + np.exp(-logit_mu)))

            # 3. CHANGE: Instead of updating the DataFrame (which breaks on vectors),
            # we create a local adjusted prior for the unpolled IND expansion logic
            # This represents the 'baseline' or 'minor' IND weight
            prior_ind_remainder = prior_ind_val * (1 - c200_ratio)
            
            # 4. Update the poll names and matrix
            # The polled 'IND' bucket is renamed to the specific C200 candidate (e.g. IND1)
            poll_names[ind_idx] = f'IND{Position}'
            
            # Add a placeholder 'IND' column to poll_alloc_matrix to receive 
            # the 'remainder' (minor INDs) later in the unpolled_minor_cols logic
            
            #ind_reset_col = np.zeros((n_simulations, 1))
            #poll_alloc_matrix = np.hstack([poll_alloc_matrix, ind_reset_col])
            #poll_names.append('IND')
            #unpolled_minor_cols += ['IND']

            if 'IND' not in unpolled_minor_cols:
                unpolled_minor_cols.append('IND')


        if election_year == 'Byelection' and div == 'Wentworth':
            ind_idx = poll_names.index('IND')
            ind_shares = poll_alloc_matrix[:, ind_idx]
            
            WENTWORTH_IND_RATIO = 20.974 / (6.435 + 20.974)

            ind1_col = (ind_shares * WENTWORTH_IND_RATIO)[:, np.newaxis]
            ind2_col = (ind_shares * (1 - WENTWORTH_IND_RATIO))[:, np.newaxis]

            poll_alloc_matrix = np.delete(poll_alloc_matrix, ind_idx, axis=1)
            poll_names.pop(ind_idx)

            poll_alloc_matrix = np.hstack([poll_alloc_matrix, ind1_col, ind2_col])
            poll_names += ['IND1', 'IND2']

            if 'IND' not in unpolled_minor_cols:
                unpolled_minor_cols.append('IND')



        if 'OTH' in poll_names:
            oth_in_poll_idx = poll_names.index('OTH')
            remaining_oth = poll_alloc_matrix[:, oth_in_poll_idx]
            
            if len(unpolled_minor_cols) > 0 and np.any(remaining_oth > 0):

                if has_matrix:
                    # Build matrix from prior_matrix
                    prior_minors = prior_matrix[:, [prior_names.index(p) for p in unpolled_minor_cols]]
                    # Override the IND column with our 'remainder' logic
                    if 'IND' in unpolled_minor_cols:
                        ind_col_idx = unpolled_minor_cols.index('IND')
                        prior_minors[:, ind_col_idx] = prior_ind_remainder
                else:
                    # Scalar logic
                    prior_minors = prior_df.loc[div, unpolled_minor_cols].copy()
                    if 'IND' in unpolled_minor_cols:
                        prior_minors['IND'] = prior_ind_remainder

                #prior_minors = prior_df.loc[div, unpolled_minor_cols]
                #prior_weights = prior_minors / prior_minors.sum()

                minors_arr = getattr(prior_minors, 'values', prior_minors).astype(float)

                if minors_arr.ndim == 2:
                    row_sums = minors_arr.sum(axis=1, keepdims=True)
                    weights_arr = np.divide(minors_arr, row_sums, out=np.full_like(minors_arr, 1/minors_arr.shape[1]), where=row_sums>0)
                else:
                    row_sum = minors_arr.sum()
                    weights_arr = minors_arr / row_sum if row_sum > 0 else np.full_like(minors_arr, 1/len(minors_arr))

                alpha_prior = weights_arr * prior_alpha
                #alpha_prior = (prior_weights.values * prior_alpha).astype(float)
                
                if has_matrix:
                    draw_prior = np.array([np.random.dirichlet(a) for a in alpha_prior])
                else:
                    draw_prior = np.random.dirichlet(alpha_prior, size=n_simulations)
                #draw_prior = np.random.dirichlet(alpha_prior, size=n_simulations)
                prior_alloc_matrix = draw_prior * remaining_oth[:, np.newaxis]
                
                poll_alloc_no_oth = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
                poll_names_no_oth = [p for p in poll_names if p != 'OTH']
                
                minor_party_block = np.hstack([poll_alloc_no_oth, prior_alloc_matrix])
                all_minor_names = poll_names_no_oth + unpolled_minor_cols
            else:
                minor_party_block = np.delete(poll_alloc_matrix, oth_in_poll_idx, axis=1)
                all_minor_names = [p for p in poll_names if p != 'OTH']
        else:
            minor_party_block = poll_alloc_matrix
            all_minor_names = poll_names

        main_party_block = np.delete(adjusted_cago, oth_idx, axis=1)
        main_party_names = [p for p in cago_party_names if p != 'Other']

        final_array = np.hstack([main_party_block, minor_party_block])
        all_party_names = main_party_names + all_minor_names

        assert np.isclose(final_array.sum(), n_simulations)


        if ('LP' in Results_row.columns) and ('NP' in Results_row.columns):
            NP_LP_split_polled = ('NAT' in poll_df.columns) and poll_df['NAT'].sum() > 0

            if NP_LP_split_polled:
                nat = poll_df.loc[div,'NAT'].astype(float)
                lib = poll_df.loc[div,'COAL'].astype(float)
                
                coal_total = nat + lib
                NP_est = nat / coal_total if coal_total > 0 else 0.5
            else:
                if has_matrix and ('NP' in prior_names) and ('LP' in prior_names):
                    np_col = prior_matrix[:, prior_names.index('NP')]
                    lp_col = prior_matrix[:, prior_names.index('LP')]
                    # Safe division to prevent NaNs
                    NP_est = np.divide(np_col, (lp_col + np_col), 
                                       out=np.full_like(np_col, 0.5), 
                                       where=(lp_col + np_col) > 0)
                
                # --- Standard Prior/Estimate Logic ---
                elif div in NP_ratios_curr['div_nm'].unique():
                    if 'NP' in prior_df.columns and 'LP' in prior_df.columns:
                        denom = prior_df.loc[div, ['LP', 'NP']].sum()
                        NP_est = prior_df.loc[div, 'NP'] / denom if denom > 0 else 0.5
                    else:
                        NP_est = NP_ratios_curr.loc[NP_ratios_curr['div_nm'] == div, 'final_estimate'].iloc[0]
                else:
                    NP_est = 0.5

            coal_idx = all_party_names.index('COAL')
            coal_votes = final_array[:, coal_idx]

            if isinstance(NP_est, np.ndarray):
                alphas_lp_np = np.vstack([1 - NP_est, NP_est]).T *  LP_NP_DIR_ALPHA_POLL # Fix - polling probably more accurate!
                lp_np_splits = np.array([np.random.dirichlet(a) for a in alphas_lp_np])
            else:
                alpha_lp_np = np.array([1 - NP_est, NP_est]) * LP_NP_DIR_ALPHA_POLL
                lp_np_splits = np.random.dirichlet(alpha_lp_np, size=n_simulations)

            
            lp_votes = lp_np_splits[:, 0] * coal_votes
            np_votes = lp_np_splits[:, 1] * coal_votes

            final_array = np.delete(final_array, coal_idx, axis=1)
            all_party_names.pop(coal_idx)

            final_array = np.hstack([final_array, lp_votes[:, np.newaxis], np_votes[:, np.newaxis]])
            all_party_names += ['LP', 'NP']



        if 'IND' in all_party_names:
            ind_idx = all_party_names.index('IND')
            ind_total = final_array[:, ind_idx]

            if multi_ind_c200_case:
                free_weights = np.full(k - 1, 1 / (k - 1))
                ind_draws = np.random.dirichlet(free_weights * prior_alpha, size=n_simulations)
                ind_split_votes = ind_draws * ind_total[:, np.newaxis]

                final_array = np.delete(final_array, ind_idx, axis=1)
                all_party_names.pop(ind_idx)

                idx = 0
                for i in range(1, k + 1):
                    if i == Position:
                        continue
                    label = f'IND{i if i < Position else i}'
                    final_array = np.hstack([final_array, ind_split_votes[:, [idx]]])
                    all_party_names.append(label)
                    idx += 1
                    
            elif election_year == 'Byelection' and div == 'Wentworth':
                all_party_names[ind_idx] = 'IND3'

            else:
                if div in multiple_INDs_df['div_nm'].unique():
                    means_array = np.full(k, 1 / k)

                    if div in C200_IND_positions_curr['div_nm'].unique():
                        pos = int(C200_IND_positions_curr.loc[
                            C200_IND_positions_curr['div_nm'] == div, 'Number'
                        ].iloc[0]) - 1

                        c200_ratios = np.where(
                            ind_total * 100 < 5, 
                            0.9, 
                            1 / (1 + np.exp(-(beta0 + beta1 * np.log(ind_total * 100))))
                        )

                        means_matrix = np.tile((1 - c200_ratios)[:, np.newaxis] / (k - 1), (1, k))
                        means_matrix[:, pos] = c200_ratios
                    else:
                        means_matrix = np.tile(means_array, (n_simulations, 1))
                        pos = 0 # SAFE FALLBACK: If no C200 candidate exists, default to IND1

                    splits = np.array([np.random.dirichlet(m * prior_alpha) for m in means_matrix])
                    
                    if has_matrix and ind_surge_dict is not None:
                        # 1. Total surge across all IND candidates in this division
                        total_surge_ratio = np.sum(list(ind_surge_dict.values()), axis=0)
                        
                        # 2. Subtract surge from the total (usually poll-derived) to get organic base
                        organic_ind = ind_total * (1 - total_surge_ratio)
                        ind_split_votes = splits * organic_ind[:, np.newaxis]
                        
                        # 3. Map specific surges back to the numbered IND columns
                        for j in range(k):
                            label = f'IND{j+1}'
                            if label in ind_surge_dict:
                                ind_split_votes[:, j] += (ind_total * ind_surge_dict[label])
                    else:
                        ind_split_votes = splits * ind_total[:, np.newaxis]


                    final_array = np.delete(final_array, ind_idx, axis=1)
                    all_party_names.pop(ind_idx)

                    for i in range(k):
                        final_array = np.hstack([final_array, ind_split_votes[:, [i]]])
                        all_party_names.append(f'IND{i+1}')

                else:
                    all_party_names[ind_idx] = 'IND1'

        #assert final_array.min() > 1e-40

        # Final Coalition Renaming
        if ('COAL' in all_party_names) and (missing_party != 'COAL'):
            coal_idx = all_party_names.index('COAL')
            coal_replacement_list = ['LP', 'NP', 'LNP', 'CLP']
            coal_party = next(p for p in Results_row.columns if p in coal_replacement_list)
            all_party_names[coal_idx] = coal_party

        # Final mapping to match Results_row column order
        ballot_order = Results_row.columns.tolist()
        name_to_idx = {name: i for i, name in enumerate(all_party_names)}
        col_indices = [name_to_idx[name] for name in ballot_order]


        seat_output_dict[div] = final_array[:, col_indices]

        party_name_dict[div] = ballot_order  # add names to avoid confusion in future!
    
    return seat_output_dict, party_name_dict



def simulate_seat_polling_full_model(n_simulations, election_year, Seat_poll_year_dict, Seat_poll_covm_dict, Results_dict_active , prior_alpha = 22, poll_alpha = 22, GLOBAL_CSVs = {}):

    # currently unvectorised, simulates for single seat at a time

    seat_poll_sims = {}

    prior_long = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv")

    # perform for each electorate with seat polling
    curr_seat_polls = Seat_poll_year_dict[election_year]
    for div in curr_seat_polls['Electorate'].unique():

        if prior_long.loc[prior_long['div_nm']==div,].empty: # e.g., Canberra 1995
            continue

        prior_row = prior_long.loc[prior_long['div_nm']==div,][['PartyAb','FP_Votes']].set_index('PartyAb').T.rename(index = {'FP_Votes':div})
        poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate']==div].set_index('Electorate')
        prior_CAGO =  group_into_Fundamentals_Categories(prior_row, div, is_Other = True)


        results = []

        alr_error_matrix = np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=Seat_poll_covm_dict[election_year],
            size=n_simulations
        )

        missing_party = next((p for p in ['COAL','ALP','GRN'] if prior_CAGO.loc[div, p] == 0),None)
        if election_year == 'Byelection' and missing_party is not None:
            if poll_df[missing_party].sum() == 0:
                alr_error_matrix, new_cols, ref_col = transform_alr_array(alr_error_matrix, missing_party)
        else:
            new_cols, ref_col = ['ALP','GRN','Other'], 'COAL'





        Results_row = Results_dict_active[election_year][div]

        precomputed_dict = precompute_division(poll_df,
            prior_row,
            div,
            election_year,
            missing_party,
            ref_col,
            GLOBAL_CSVs
        )



        results = process_division(
            poll_df,
            prior_row,
            Results_row,
            div,
            election_year,
            missing_party=missing_party,
            ref_col=ref_col,
            prior_alpha=prior_alpha,
            poll_alpha = poll_alpha,
            precomputed_dict=precomputed_dict,
            GLOBAL_CSVs=GLOBAL_CSVs,
            n_simulations=n_simulations
        )

        seat_poll_sims[div] = results

    return seat_poll_sims




#seat_model_sim = simulate_seat_polling_full_model(n_simulations, election_year, Seat_poll_year_dict, Seat_poll_covm_dict, Results_dict_active)

#import pdb; pdb.set_trace()




def get_election_MAE(combined_samples, Results_dict, coverage_level):

    final_simulated_votes = combined_samples

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


def get_election_metrics(combined_samples, Results_dict, coverage_levels):
    
    final_simulated_votes = combined_samples
    all_abs_diffs = []

    # Track hits for each coverage level in the schedule
    coverage_hits = {c: 0 for c in coverage_levels}
    total_predictions = 0

    for div in final_simulated_votes.keys():
        pred = final_simulated_votes[div] * 100 # working in percentages
        actual = Results_dict[div].iloc[0].values

        # 1. MAE Calculation (Calculated once)
        abs_diff = np.abs(pred - actual)
        all_abs_diffs.append(abs_diff)
        
        total_predictions += len(actual)

        # 2. Coverage Calculation (Calculated for each target level)
        for c in coverage_levels:
            lower_percentile = (1 - c) / 2 * 100
            upper_percentile = (1 + c) / 2 * 100

            lower_bounds = np.percentile(pred, lower_percentile, axis=0)
            upper_bounds = np.percentile(pred, upper_percentile, axis=0)

            within_bounds = (actual >= lower_bounds) & (actual <= upper_bounds)
            coverage_hits[c] += np.sum(within_bounds)

    # Finalize MAE
    combined_abs_diffs = np.concatenate(all_abs_diffs, axis=1)
    mae_per_simulation = combined_abs_diffs.mean(axis=1) 
    average_mae = np.mean(mae_per_simulation)

    # Finalize Coverages
    empirical_coverages = {c: hits / total_predictions for c, hits in coverage_hits.items()}

    return average_mae, empirical_coverages


def perform_validation_testing(n_simulations, coverage_level, coverage_weight=5):

    weights = np.linspace(0, 1, 11)
    poll_alphas = np.linspace(20, 100, 5)

    elections = ['2016', '2019', '2022','2025','Byelection']

    best_params = {}

    
    div_order_dict = {}
    prior_df_active_dict = {}
    polling_avg_dict = {}
    National_prior_dict = {}

    Prior_estimates_dict_active_per_election = {}
    Results_dict_active_per_election = {}

    byelection_group_structure = []

    GLOBAL_CSVs = {
        "multiple_INDs_df": {election_year: pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None) for election_year in elections},
        "C200_IND_splits": pd.read_csv("Independent_splits_multiple.csv", index_col=None),
        "C200_IND_positions_df": pd.read_csv("C200_IND_positions_df.csv", index_col=None),
        "NP_ratios_curr": pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)
    }



    import pickle

    with open("covms_all.pkl", "rb") as f:
        data = pickle.load(f)

    Seat_poll_covm_dict = data["Seat_poll_covm_dict"]
    Electorate_residual_covm_dict = data["Electorate_residual_covm_dict"]
    Nat_poll_covm_dict = data["Nat_poll_covm_dict"]

    #National_prior_dict['2026'] = pd.DataFrame([[0.3456,0.3182,0.122,0.2142]], columns = ['ALP','COAL','GRN','OTH']) # for Farrer

    for election_year in elections:

        div_order = Seat_poll_year_dict[election_year]['Electorate'].drop_duplicates().tolist()
        prior_df_active = prior_df_dict[election_year].loc[div_order]

        div_order_dict[election_year] = div_order
        prior_df_active_dict[election_year] = prior_df_active

        if election_year == 'Byelection':
            # for byleeciton, take it group by group, ensuring they are correctly ordered

            order_map={k:i for i,k in enumerate(div_order)}
            by_election_groups=(Byelection_final_polling_averages.reset_index().loc[lambda d:d['Election'].str[:-4].isin(prior_df_active.index)].assign(div=lambda d:d['Election'].str[:-4]).sort_values('div',key=lambda s:s.map(order_map)).groupby(['COAL','ALP','GRN','OTH'])['Election'].apply(list).tolist())
            by_election_groups=sorted(by_election_groups,key=lambda g:min(order_map[e[:-4]] for e in g))


            for group in by_election_groups:

                polling_avg = Byelection_final_polling_averages.loc[Byelection_final_polling_averages.index.isin(group)].iloc[[0],:]
                National_prior = National_prior_dict[str(min(y for y in map(int, National_prior_dict.keys()) if y > int(polling_avg.index[0][-4:])))]
                # Narrow down prior_df_active to group
                prior_df_curr = prior_df_active.loc[[elec[:-4] for elec in group]]

                byelection_group_structure.append({
                    "group": group,
                    "polling_avg": polling_avg,
                    "prior_df_active": prior_df_curr,
                    "National_prior": National_prior
                })
                
            # Get Results_dict concatenate results for by-elections since 2013
            By_elections_Results = pd.read_csv('By-election_results.csv').iloc[:,:4]
            By_elections_Results.loc[By_elections_Results['div_nm']=='Batman','div_nm'] = 'Cooper'
            By_elections_Results = By_elections_Results[By_elections_Results['byelection_year']>=2013]
            Results_dict =  {div: g.assign(FP=g['FirstPreferencePercent']).groupby('PartyAb')['FP'].sum().to_frame().T.reset_index(drop=True) for div, g in By_elections_Results.groupby('div_nm')}
            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # different approach for byelection
        else:
            polling_avg = pd.read_csv("Final_poll_averages_CAGO.csv")
            polling_avg = polling_avg.loc[polling_avg["Election"] == election_year,["COAL", "ALP", "GRN", "OTH"]]
            polling_avg_dict[election_year] = polling_avg


            prior_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})
            prior_df = combine_to_CAGO(prior_df)

            prior_df_dict[election_year] = prior_df

            National_prior_dict[election_year] = prior_df.mean().to_frame().T

            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1]
            Results_dict = get_results_df(election_year, to_Fundamentals=False)[1]


        Prior_estimates_dict_active = {div: Prior_estimates_dict[div] for div in div_order}
        Results_dict_active = {div: Results_dict[div] for div in div_order}

        Prior_estimates_dict_active_per_election[election_year] = Prior_estimates_dict_active
        Results_dict_active_per_election[election_year] = Results_dict_active

    # =========================================================
    # OUTER LOOP: LOO-CV
    # =========================================================
    for heldout in elections:

        train_elections = [e for e in elections if e != heldout]

        results = []

        # -----------------------------
        # NAT: compute once (heldout-independent of alpha/w)
        # -----------------------------
        start_time = time.time()

        Nat_polling_simulations_dict = {
            election: simulate_nat_polling_full_model(
                n_simulations,
                election,
                prior_df_active_dict,
                polling_avg_dict,
                National_prior_dict,
                Prior_estimates_dict_active_per_election,
                Results_dict_active_per_election,
                byelection_group_structure,
                Nat_poll_covm_dict,
                Electorate_residual_covm_dict,
                prior_alpha = 22
            )
            for election in train_elections
        }

        print(f"nat,heldout = {heldout}", time.time() - start_time)

        # =====================================================
        # PRECOMPUTE SEAT MODELS ONCE PER poll_alpha
        # =====================================================

        start_time = time.time()

        Seat_polling_simulations_dict_by_alpha = {}

        for poll_alpha in poll_alphas:

            Seat_polling_simulations_dict_by_alpha[poll_alpha] = {
                election: simulate_seat_polling_full_model(
                    n_simulations,
                    election,
                    Seat_poll_year_dict,
                    Seat_poll_covm_dict,
                    Results_dict_active_per_election,
                    prior_alpha=22,
                    poll_alpha=poll_alpha,
                    GLOBAL_CSVs=GLOBAL_CSVs
                )
                for election in train_elections
            }

        print(f"seat_alphas,heldout = {heldout}", time.time() - start_time)

        # =====================================================
        # GRID SEARCH OVER w × poll_alpha
        # =====================================================
        for w in weights:

            base_idx = np.random.permutation(n_simulations)
            n_nat = int(w * n_simulations)
            idx_nat = base_idx[:n_nat]
            idx_seat = base_idx[n_nat:]

            for poll_alpha in poll_alphas:

                Seat_polling_simulations_dict = Seat_polling_simulations_dict_by_alpha[poll_alpha]

                val_scores = []

                for election in train_elections:

                    nat_sim = Nat_polling_simulations_dict[election][0]
                    seat_sim = Seat_polling_simulations_dict[election]

                    combined_samples = {
                        div: np.concatenate(
                            (nat_sim[div][idx_nat], seat_sim[div][idx_seat]),
                            axis=0
                        )
                        for div in nat_sim.keys()
                    }

                    mae, coverage = get_election_MAE(
                        combined_samples,
                        Results_dict_active_per_election[election],
                        coverage_level,
                    )

                    penalty = max(0, coverage_level - coverage)
                    val_scores.append(mae + coverage_weight * penalty)

                results.append((w, poll_alpha, np.mean(val_scores)))

        # =====================================================
        # SELECT BEST
        # =====================================================
        best_w, best_alpha, best_val_score = min(results, key=lambda x: x[2])

        print('Evaluation step')

        # =====================================================
        # HELDOUT EVAL
        # =====================================================

        

        nat_sim = simulate_nat_polling_full_model(
            n_simulations,
            heldout,
            prior_df_active_dict,
            polling_avg_dict,
            National_prior_dict,
            Prior_estimates_dict_active_per_election,
            Results_dict_active_per_election,
            byelection_group_structure,
            Nat_poll_covm_dict,
            Electorate_residual_covm_dict,
            prior_alpha = 22
        )[0]

        seat_sim = simulate_seat_polling_full_model(
            n_simulations,
            heldout,
            Seat_poll_year_dict,
            Seat_poll_covm_dict,
            Results_dict_active_per_election,
            prior_alpha=22,
            poll_alpha=best_alpha,
            GLOBAL_CSVs=GLOBAL_CSVs
        )

        base_idx = np.random.permutation(n_simulations)
        n_nat = int(best_w * n_simulations)

        idx_nat = base_idx[:n_nat]
        idx_seat = base_idx[n_nat:]

        combined_samples = {
            div: np.concatenate(
                (nat_sim[div][idx_nat], seat_sim[div][idx_seat]),
                axis=0
            )
            for div in nat_sim.keys()
        }

        heldout_mae, heldout_coverage = get_election_MAE(
            combined_samples,
            Results_dict_active_per_election[heldout],
            coverage_level
        )

        best_params[heldout] = {
            "w": best_w,
            "poll_alpha": best_alpha,
            "val_score": best_val_score,
            "test_mae": heldout_mae,
            "test_coverage": heldout_coverage
        }
 
    import pdb; pdb.set_trace()

    return best_params




def perform_validation_testing_new(n_simulations, target_coverage_schedule, coverage_weight=5, poll_alpha_fixed = 20, prior_alpha_fixed = 12):

    expansion_weights = np.linspace(0.44, 0.7, 14)
    B_values = np.linspace(0.8, 1.7,10)

    elections = ['2016', '2019', '2022','2025','Byelection']

    best_params = {}

    
    div_order_dict = {}
    prior_df_active_dict = {}
    polling_avg_dict = {}
    National_prior_dict = {}

    Prior_estimates_dict_active_per_election = {}
    Results_dict_active_per_election = {}

    seat_alr_bases_dict = {}
    n_seat_polls_dict = {}

    byelection_group_structure = []

    GLOBAL_CSVs = {
        "multiple_INDs_df": {election_year: pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None) for election_year in elections},
        "C200_IND_splits": pd.read_csv("Independent_splits_multiple.csv", index_col=None),
        "C200_IND_positions_df": pd.read_csv("C200_IND_positions_df.csv", index_col=None),
        "NP_ratios_curr": pd.read_csv("NP_ratio_estimated_df.csv", index_col=None),
        "proportions_df": pd.read_csv('Missing_major_reallocation_proportions.csv'),
        "mapping_df": pd.read_csv('Missing_major_allocation.csv')
    }



    import pickle

    with open("covms_all.pkl", "rb") as f:
        data = pickle.load(f)

    Seat_poll_covm_dict = data["Seat_poll_covm_dict"]
    Electorate_residual_covm_dict = data["Electorate_residual_covm_dict"]
    Nat_poll_covm_dict = data["Nat_poll_covm_dict"]

    #National_prior_dict['2026'] = pd.DataFrame([[0.3456,0.3182,0.122,0.2142]], columns = ['ALP','COAL','GRN','OTH']) # for Farrer

    for election_year in elections:

        div_order = Seat_poll_year_dict[election_year]['Electorate'].drop_duplicates().tolist()
        prior_df_active = prior_df_dict[election_year].loc[div_order]

        div_order_dict[election_year] = div_order
        prior_df_active_dict[election_year] = prior_df_active

        if election_year == 'Byelection':
            # for byleeciton, take it group by group, ensuring they are correctly ordered

            order_map={k:i for i,k in enumerate(div_order)}
            by_election_groups=(Byelection_final_polling_averages.reset_index().loc[lambda d:d['Election'].str[:-4].isin(prior_df_active.index)].assign(div=lambda d:d['Election'].str[:-4]).sort_values('div',key=lambda s:s.map(order_map)).groupby(['COAL','ALP','GRN','OTH'])['Election'].apply(list).tolist())
            by_election_groups=sorted(by_election_groups,key=lambda g:min(order_map[e[:-4]] for e in g))


            for group in by_election_groups:

                polling_avg = Byelection_final_polling_averages.loc[Byelection_final_polling_averages.index.isin(group)].iloc[[0],:]
                National_prior = National_prior_dict[str(min(y for y in map(int, National_prior_dict.keys()) if y > int(polling_avg.index[0][-4:])))]
                # Narrow down prior_df_active to group
                prior_df_curr = prior_df_active.loc[[elec[:-4] for elec in group]]

                byelection_group_structure.append({
                    "group": group,
                    "polling_avg": polling_avg,
                    "prior_df_active": prior_df_curr,
                    "National_prior": National_prior
                })
                
            # Get Results_dict concatenate results for by-elections since 2013
            By_elections_Results = pd.read_csv('By-election_results.csv').iloc[:,:4]
            By_elections_Results.loc[By_elections_Results['div_nm']=='Batman','div_nm'] = 'Cooper'
            By_elections_Results = By_elections_Results[By_elections_Results['byelection_year']>=2013]
            Results_dict =  {div: g.assign(FP=g['FirstPreferencePercent']).groupby('PartyAb')['FP'].sum().to_frame().T.reset_index(drop=True) for div, g in By_elections_Results.groupby('div_nm')}
            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # different approach for byelection
        else:
            polling_avg = pd.read_csv("Final_poll_averages_CAGO.csv")
            polling_avg = polling_avg.loc[polling_avg["Election"] == election_year,["COAL", "ALP", "GRN", "OTH"]]
            polling_avg_dict[election_year] = polling_avg


            prior_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})
            prior_df = combine_to_CAGO(prior_df)

            prior_df_dict[election_year] = prior_df

            National_prior_dict[election_year] = prior_df.mean().to_frame().T

            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1]
            Results_dict = get_results_df(election_year, to_Fundamentals=False)[1]



        # seat alr base - should be done deterministically outside the loop!
        # Assuming div_order matches the order of electorates in nat_alr_sims

        n_seat_polls_dict[election_year] = {}
        seat_alr_bases_dict[election_year] = {}

        curr_seat_polls = Seat_poll_year_dict[election_year]
        prior_long = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv")
        
        for div in div_order:

            # 1. Isolate the seat poll for this division
            poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate'] == div].set_index('Electorate')
            
            # 2. Replicate your logic: Normalize and Group
            curr_row = poll_df.iloc[:, 2 + ('byelection_year' in poll_df.columns):]
            curr_row_normalised = curr_row.div(curr_row.sum(axis=1), axis=0)
            
            # Take the mean of polls for this seat (if multiple polls exist)
            CAGO_row = group_into_Fundamentals_Categories(curr_row_normalised, div).mean()

            prior_row = prior_long.loc[prior_long['div_nm']==div,][['PartyAb','FP_Votes']].set_index('PartyAb').T.rename(index = {'FP_Votes':div})
            prior_CAGO =  group_into_Fundamentals_Categories(prior_row, div, is_Other = True)
            missing_party = next((p for p in ['COAL','ALP','GRN'] if prior_CAGO.loc[div, p] == 0),None)


            # 3. Drop missing party if necessary
            if missing_party is not None and missing_party in CAGO_row.index:
                CAGO_row = CAGO_row.drop([missing_party])
            
            ref_col = 'ALP' if missing_party == 'COAL' else 'COAL'
            
            # 4. Convert to ALR: [ALP, GRN, Other] / COAL
            # Resulting Series index: ['ALP', 'GRN', 'Other']
            alr_base = np.log(CAGO_row.drop(ref_col) / CAGO_row[ref_col])

            n_seat_polls_dict[election_year][div] = poll_df['n_polls'].iloc[0] # 1,2,3

            seat_alr_bases_dict[election_year][div] = alr_base.values


        Prior_estimates_dict_active = {div: Prior_estimates_dict[div] for div in div_order}
        Results_dict_active = {div: Results_dict[div] for div in div_order}

        Prior_estimates_dict_active_per_election[election_year] = Prior_estimates_dict_active
        Results_dict_active_per_election[election_year] = Results_dict_active

    # =====================================================
    # OUTER LOOP: LOO-CV
    # =====================================================
    for heldout in elections:
        train_elections = [e for e in elections if e != heldout]
        results = []

        start  = time.time()

        # =====================================================
        # GRID SEARCH: B x expansion_w
        # =====================================================
        for B in B_values:
            # 1. ALR FUSION: Compute combined CAGO shares once per B
            # Returns {election: {div: (n_sims, 4)}}
            combined_CAGO_dict = {
                election: simulate_combined_models_CAGO(
                    n_simulations, election, prior_df_active_dict, polling_avg_dict,
                    National_prior_dict,byelection_group_structure,
                    seat_alr_bases_dict, n_seat_polls_dict,
                    Nat_poll_covm_dict, Electorate_residual_covm_dict, 
                    Seat_poll_covm_dict, GLOBAL_CSVs, prior_alpha=prior_alpha_fixed, B=B
                )
                for election in train_elections
            }

            # 2. MICRO-EXPANSION: Generate the two alternative universes for 'OTH'
            # These are computed ONCE per B since poll_alpha is fixed.
            seat_expanded_universe = {
                election: expand_divisions_using_seat_polls(
                    combined_CAGO_dict[election][0], # simulation set
                    Seat_poll_year_dict[election],
                    Prior_estimates_dict_active_per_election[election],
                    Results_dict_active_per_election[election],
                    election,
                    prior_alpha=prior_alpha_fixed,
                    poll_alpha = poll_alpha_fixed,
                    major_realloc_data = combined_CAGO_dict[election][1], # major_realloc_data
                    GLOBAL_CSVs = GLOBAL_CSVs,
                    n_simulations = n_simulations
                )
                for election in train_elections
            }

            
            prior_expanded_universe = {
                election: expand_divisions_using_prior(
                    combined_CAGO_dict[election][0],
                    combined_CAGO_dict[election][1],# major_realloc_data
                    Prior_estimates_dict_active_per_election[election],
                    Results_dict_active_per_election[election],
                    election_year=election,
                    alpha_scalar=prior_alpha_fixed
                )
                for election in train_elections
            }

            # assertion checks that order is all good
            for year in train_elections:

                assert seat_expanded_universe[year][1].keys() == prior_expanded_universe[year][1].keys()

                for div in seat_expanded_universe[year][1].keys():
                    assert seat_expanded_universe[year][1][div] == prior_expanded_universe[year][1][div]

                    # not too small proportions
                    #assert np.min(seat_expanded_universe[year][0][div]>1e-30)
                    #assert np.min(prior_expanded_universe[year][0][div]>1e-40)

            # 3. OPTIMIZE w: Slice between the expansion strategies
            for w in expansion_weights:
                val_scores = []
                
                n_seat = int(w * n_simulations)
                idx_seat = np.arange(n_seat)
                idx_prior = np.arange(n_seat, n_simulations)

                for election in train_elections:
                    combined_samples = {}
                    for div in combined_CAGO_dict[election][0].keys():
                        # Assemble the ensemble: w% from Poll-Expansion, (1-w)% from Prior-Expansion
                        combined_samples[div] = np.vstack([
                            seat_expanded_universe[election][0][div][idx_seat],
                            prior_expanded_universe[election][0][div][idx_prior]
                        ])

                    mae, empirical_coverages = get_election_metrics(
                        combined_samples,
                        Results_dict_active_per_election[election],
                        target_coverage_schedule
                    )

                    # Calculate the mean penalty across the coverage schedule
                    total_penalty = 0
                    for target in target_coverage_schedule:
                        empirical = empirical_coverages[target]
                        total_penalty += max(0, target - empirical)

                    avg_penalty = total_penalty / len(target_coverage_schedule)
                    val_scores.append(mae + coverage_weight * avg_penalty)

                    
                    # OLD: only mae
                    #penalty = max(0, coverage_level - coverage)
                    #val_scores.append(mae + coverage_weight * penalty)

                results.append((w, B, np.mean(val_scores)))


        print(heldout, "completed in", time.time()-start)
        # =====================================================
        # SELECT BEST & EVALUATE HELDOUT
        # =====================================================
        best_w, best_B, best_val_score = min(results, key=lambda x: x[2])
        
        # 1. Final Fused CAGO for heldout
        heldout_combined = simulate_combined_models_CAGO(
            n_simulations, heldout, prior_df_active_dict, polling_avg_dict,
            National_prior_dict, byelection_group_structure,
            seat_alr_bases_dict, n_seat_polls_dict,
            Nat_poll_covm_dict, Electorate_residual_covm_dict, 
            Seat_poll_covm_dict, GLOBAL_CSVs, prior_alpha=prior_alpha_fixed, B=best_B
        )
        
        # 2. Generate both expansions for the heldout election
        h_seat_exp = expand_divisions_using_seat_polls(
                    heldout_combined[0],
                    Seat_poll_year_dict[heldout],
                    Prior_estimates_dict_active_per_election[heldout],
                    Results_dict_active_per_election[heldout],
                    heldout,
                    prior_alpha=prior_alpha_fixed,
                    poll_alpha = poll_alpha_fixed,
                    major_realloc_data = heldout_combined[1],
                    GLOBAL_CSVs = GLOBAL_CSVs,
                    n_simulations = n_simulations
                )[0]
        h_prior_exp = expand_divisions_using_prior(
                    heldout_combined[0],
                    heldout_combined[1],
                    Prior_estimates_dict_active_per_election[heldout],
                    Results_dict_active_per_election[heldout],
                    election_year=heldout,
                    alpha_scalar=prior_alpha_fixed
                )[0]
        
        # 3. Apply best_w to create the final heldout ensemble
        n_seat = int(best_w * n_simulations)
        final_heldout_samples = {
            div: np.vstack([h_seat_exp[div][:n_seat], h_prior_exp[div][n_seat:]])
            for div in heldout_combined[0].keys()
        }

        heldout_mae, heldout_coverages = get_election_metrics(
            final_heldout_samples, 
            Results_dict_active_per_election[heldout], 
            target_coverage_schedule
        )

        best_params[heldout] = {
            "w": best_w, 
            "B": best_B, 
            "test_mae": heldout_mae, 
            "test_coverage": heldout_coverages
        }

    return best_params



def validate_prior_alpha(n_simulations, target_coverage_schedule, coverage_weight=5):
    """
    Optimized LOO-CV for poll_alpha in the seat model.
    Precomputes all simulations to avoid O(N^2) complexity.
    """

    prior_alphas = np.linspace(5, 25, 21)

    elections = ['2016', '2019', '2022','2025','Byelection']

    best_params = {}

    div_order_dict = {}
    prior_df_active_dict = {}
    polling_avg_dict = {}
    National_prior_dict = {}

    Prior_estimates_dict_active_per_election = {}
    Results_dict_active_per_election = {}

    byelection_group_structure = []

    GLOBAL_CSVs = {
        "multiple_INDs_df": {election_year: pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None) for election_year in elections},
        "C200_IND_splits": pd.read_csv("Independent_splits_multiple.csv", index_col=None),
        "C200_IND_positions_df": pd.read_csv("C200_IND_positions_df.csv", index_col=None),
        "NP_ratios_curr": pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)
    }

    import pickle

    with open("covms_all.pkl", "rb") as f:
        data = pickle.load(f)

    Seat_poll_covm_dict = data["Seat_poll_covm_dict"]
    Electorate_residual_covm_dict = data["Electorate_residual_covm_dict"]
    Nat_poll_covm_dict = data["Nat_poll_covm_dict"]

    #National_prior_dict['2026'] = pd.DataFrame([[0.3456,0.3182,0.122,0.2142]], columns = ['ALP','COAL','GRN','OTH']) # for Farrer

    for election_year in elections:

        div_order = Seat_poll_year_dict[election_year]['Electorate'].drop_duplicates().tolist()
        prior_df_active = prior_df_dict[election_year].loc[div_order]

        div_order_dict[election_year] = div_order
        prior_df_active_dict[election_year] = prior_df_active

        if election_year == 'Byelection':
            # for byleeciton, take it group by group, ensuring they are correctly ordered

            order_map={k:i for i,k in enumerate(div_order)}
            by_election_groups=(Byelection_final_polling_averages.reset_index().loc[lambda d:d['Election'].str[:-4].isin(prior_df_active.index)].assign(div=lambda d:d['Election'].str[:-4]).sort_values('div',key=lambda s:s.map(order_map)).groupby(['COAL','ALP','GRN','OTH'])['Election'].apply(list).tolist())
            by_election_groups=sorted(by_election_groups,key=lambda g:min(order_map[e[:-4]] for e in g))


            for group in by_election_groups:

                polling_avg = Byelection_final_polling_averages.loc[Byelection_final_polling_averages.index.isin(group)].iloc[[0],:]
                National_prior = National_prior_dict[str(min(y for y in map(int, National_prior_dict.keys()) if y > int(polling_avg.index[0][-4:])))]
                # Narrow down prior_df_active to group
                prior_df_curr = prior_df_active.loc[[elec[:-4] for elec in group]]

                byelection_group_structure.append({
                    "group": group,
                    "polling_avg": polling_avg,
                    "prior_df_active": prior_df_curr,
                    "National_prior": National_prior
                })
                
            # Get Results_dict concatenate results for by-elections since 2013
            By_elections_Results = pd.read_csv('By-election_results.csv').iloc[:,:4]
            By_elections_Results.loc[By_elections_Results['div_nm']=='Batman','div_nm'] = 'Cooper'
            By_elections_Results = By_elections_Results[By_elections_Results['byelection_year']>=2013]
            Results_dict =  {div: g.assign(FP=g['FirstPreferencePercent']).groupby('PartyAb')['FP'].sum().to_frame().T.reset_index(drop=True) for div, g in By_elections_Results.groupby('div_nm')}
            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # different approach for byelection
        else:
            polling_avg = pd.read_csv("Final_poll_averages_CAGO.csv")
            polling_avg = polling_avg.loc[polling_avg["Election"] == election_year,["COAL", "ALP", "GRN", "OTH"]]
            polling_avg_dict[election_year] = polling_avg


            prior_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})
            prior_df = combine_to_CAGO(prior_df)

            prior_df_dict[election_year] = prior_df

            National_prior_dict[election_year] = prior_df.mean().to_frame().T

            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1]
            Results_dict = get_results_df(election_year, to_Fundamentals=False)[1]


        Prior_estimates_dict_active = {div: Prior_estimates_dict[div] for div in div_order}
        Results_dict_active = {div: Results_dict[div] for div in div_order}

        Prior_estimates_dict_active_per_election[election_year] = Prior_estimates_dict_active
        Results_dict_active_per_election[election_year] = Results_dict_active
    # =========================================================
    # PHASE 1: PRECOMPUTE ALL SIMULATIONS
    # =========================================================
    # This is the "Efficiency Engine": We run every simulation once.
    # Structure: all_sims[alpha][election_year]
    print(f"Starting precomputation for {len(prior_alphas)} alphas across {len(elections)} elections...")
    start_precompute = time.time()


    
    
    all_sims = {}
    for alpha in prior_alphas:
        all_sims[alpha] = {}
        for election in elections:
            all_sims[alpha][election] = simulate_nat_polling_full_model(
                n_simulations,
                election,
                prior_df_active_dict,
                polling_avg_dict,
                National_prior_dict,
                Prior_estimates_dict_active_per_election,
                Results_dict_active_per_election,
                byelection_group_structure,
                Nat_poll_covm_dict,
                Electorate_residual_covm_dict,
                prior_alpha=alpha
            )[0]
            
    print(f"Precomputation complete in {time.time() - start_precompute:.2f} seconds.")

    # =========================================================
    # PHASE 2: OUTER LOOP: LOO-CV
    # =========================================================
    for heldout in elections:
        print(f"Evaluating Held-out: {heldout}")
        train_elections = [e for e in elections if e != heldout]
        results = []

        # GRID SEARCH OVER poll_alpha
        for alpha in prior_alphas:
            val_scores = []

            for election in train_elections:
                # Lookup precomputed sim
                nat_sim = all_sims[alpha][election]

                mae, empirical_coverages = get_election_metrics(
                        nat_sim,
                        Results_dict_active_per_election[election],
                        target_coverage_schedule
                    )

                total_penalty = 0
                for target in target_coverage_schedule:
                    empirical = empirical_coverages[target]
                    total_penalty += max(0, target - empirical)

                avg_penalty = total_penalty / len(target_coverage_schedule)
                val_scores.append(mae + coverage_weight * avg_penalty)

            results.append((alpha, np.mean(val_scores)))

            

                    # Calculate the mean penalty across the coverage schedule
                    

        # =====================================================
        # SELECT BEST (Fixed index to [1] to match (alpha, score))
        # =====================================================
        best_alpha, best_val_score = min(results, key=lambda x: x[1])

        # =====================================================
        # HELDOUT EVAL (Lookup, no re-simulation needed)
        # =====================================================
        heldout_sim = all_sims[best_alpha][heldout]

        heldout_mae, heldout_coverages = get_election_metrics(
                        heldout_sim,
                        Results_dict_active_per_election[heldout],
                        target_coverage_schedule
                    )

        best_params[heldout] = {
            "poll_alpha": best_alpha,
            "val_score": best_val_score,
            "test_mae": heldout_mae,
            "test_coverage": heldout_coverages
        }

        #
        
        print(f"  Best Alpha: {best_alpha} | Test MAE: {heldout_mae:.4f} | Test Coverage: {heldout_coverages}")
    import pdb; pdb.set_trace()
    return best_params




def validate_seat_poll_model(n_simulations, target_coverage_schedule, coverage_weight=5, prior_alpha = 10):
    """
    Optimized LOO-CV for poll_alpha in the seat model.
    Precomputes all simulations to avoid O(N^2) complexity.
    """

    poll_alphas = np.linspace(11, 30, 20)

    elections = ['2016', '2019', '2022','2025','Byelection']

    best_params = {}

    div_order_dict = {}
    prior_df_active_dict = {}
    polling_avg_dict = {}
    National_prior_dict = {}

    Prior_estimates_dict_active_per_election = {}
    Results_dict_active_per_election = {}

    byelection_group_structure = []

    GLOBAL_CSVs = {
        "multiple_INDs_df": {election_year: pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None) for election_year in elections},
        "C200_IND_splits": pd.read_csv("Independent_splits_multiple.csv", index_col=None),
        "C200_IND_positions_df": pd.read_csv("C200_IND_positions_df.csv", index_col=None),
        "NP_ratios_curr": pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)
    }

    import pickle

    with open("covms_all.pkl", "rb") as f:
        data = pickle.load(f)

    Seat_poll_covm_dict = data["Seat_poll_covm_dict"]
    Electorate_residual_covm_dict = data["Electorate_residual_covm_dict"]
    Nat_poll_covm_dict = data["Nat_poll_covm_dict"]

    #National_prior_dict['2026'] = pd.DataFrame([[0.3456,0.3182,0.122,0.2142]], columns = ['ALP','COAL','GRN','OTH']) # for Farrer

    for election_year in elections:

        div_order = Seat_poll_year_dict[election_year]['Electorate'].drop_duplicates().tolist()
        prior_df_active = prior_df_dict[election_year].loc[div_order]

        div_order_dict[election_year] = div_order
        prior_df_active_dict[election_year] = prior_df_active

        if election_year == 'Byelection':
            # for byleeciton, take it group by group, ensuring they are correctly ordered

            order_map={k:i for i,k in enumerate(div_order)}
            by_election_groups=(Byelection_final_polling_averages.reset_index().loc[lambda d:d['Election'].str[:-4].isin(prior_df_active.index)].assign(div=lambda d:d['Election'].str[:-4]).sort_values('div',key=lambda s:s.map(order_map)).groupby(['COAL','ALP','GRN','OTH'])['Election'].apply(list).tolist())
            by_election_groups=sorted(by_election_groups,key=lambda g:min(order_map[e[:-4]] for e in g))


            for group in by_election_groups:

                polling_avg = Byelection_final_polling_averages.loc[Byelection_final_polling_averages.index.isin(group)].iloc[[0],:]
                National_prior = National_prior_dict[str(min(y for y in map(int, National_prior_dict.keys()) if y > int(polling_avg.index[0][-4:])))]
                # Narrow down prior_df_active to group
                prior_df_curr = prior_df_active.loc[[elec[:-4] for elec in group]]

                byelection_group_structure.append({
                    "group": group,
                    "polling_avg": polling_avg,
                    "prior_df_active": prior_df_curr,
                    "National_prior": National_prior
                })
                
            # Get Results_dict concatenate results for by-elections since 2013
            By_elections_Results = pd.read_csv('By-election_results.csv').iloc[:,:4]
            By_elections_Results.loc[By_elections_Results['div_nm']=='Batman','div_nm'] = 'Cooper'
            By_elections_Results = By_elections_Results[By_elections_Results['byelection_year']>=2013]
            Results_dict =  {div: g.assign(FP=g['FirstPreferencePercent']).groupby('PartyAb')['FP'].sum().to_frame().T.reset_index(drop=True) for div, g in By_elections_Results.groupby('div_nm')}
            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # different approach for byelection
        else:
            polling_avg = pd.read_csv("Final_poll_averages_CAGO.csv")
            polling_avg = polling_avg.loc[polling_avg["Election"] == election_year,["COAL", "ALP", "GRN", "OTH"]]
            polling_avg_dict[election_year] = polling_avg


            prior_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})
            prior_df = combine_to_CAGO(prior_df)

            prior_df_dict[election_year] = prior_df

            National_prior_dict[election_year] = prior_df.mean().to_frame().T

            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1]
            Results_dict = get_results_df(election_year, to_Fundamentals=False)[1]


        Prior_estimates_dict_active = {div: Prior_estimates_dict[div] for div in div_order}
        Results_dict_active = {div: Results_dict[div] for div in div_order}

        Prior_estimates_dict_active_per_election[election_year] = Prior_estimates_dict_active
        Results_dict_active_per_election[election_year] = Results_dict_active
    # =========================================================
    # PHASE 1: PRECOMPUTE ALL SIMULATIONS
    # =========================================================
    # This is the "Efficiency Engine": We run every simulation once.
    # Structure: all_sims[alpha][election_year]
    print(f"Starting precomputation for {len(poll_alphas)} alphas across {len(elections)} elections...")
    start_precompute = time.time()
    
    all_sims = {}
    for alpha in poll_alphas:
        all_sims[alpha] = {}
        for election in elections:

            #asd = time.time()
            all_sims[alpha][election] = simulate_seat_polling_full_model(
                n_simulations,
                election,
                Seat_poll_year_dict,
                Seat_poll_covm_dict,
                Results_dict_active_per_election,
                prior_alpha=prior_alpha,
                poll_alpha=alpha,
                GLOBAL_CSVs=GLOBAL_CSVs
            )
            #print(time.time()-asd)
            
    print(f"Precomputation complete in {time.time() - start_precompute:.2f} seconds.")

    # =========================================================
    # PHASE 2: OUTER LOOP: LOO-CV
    # =========================================================
    for heldout in elections:
        print(f"Evaluating Held-out: {heldout}")
        train_elections = [e for e in elections if e != heldout]
        results = []

        # GRID SEARCH OVER poll_alpha
        for alpha in poll_alphas:
            val_scores = []

            for election in train_elections:
                # Lookup precomputed sim
                seat_sim = all_sims[alpha][election]

                mae, empirical_coverages = get_election_metrics(
                        seat_sim,
                        Results_dict_active_per_election[election],
                        target_coverage_schedule
                    )

                total_penalty = 0
                for target in target_coverage_schedule:
                    empirical = empirical_coverages[target]
                    total_penalty += max(0, target - empirical)

                avg_penalty = total_penalty / len(target_coverage_schedule)
                val_scores.append(mae + coverage_weight * avg_penalty)

            results.append((alpha, np.mean(val_scores)))

        # =====================================================
        # SELECT BEST (Fixed index to [1] to match (alpha, score))
        # =====================================================
        best_alpha, best_val_score = min(results, key=lambda x: x[1])

        # =====================================================
        # HELDOUT EVAL (Lookup, no re-simulation needed)
        # =====================================================
        heldout_sim = all_sims[best_alpha][heldout]

        heldout_mae, heldout_coverages = get_election_metrics(
            heldout_sim,
            Results_dict_active_per_election[heldout],
            target_coverage_schedule
        )

       

        best_params[heldout] = {
            "poll_alpha": best_alpha,
            "val_score": best_val_score,
            "test_mae": heldout_mae,
            "test_coverage": heldout_coverages
        }
        
        print(f"  Best Alpha: {best_alpha} | Test MAE: {heldout_mae:.4f} Test Coverage {heldout_coverages}")

    import pdb; pdb.set_trace()

    return best_params




#results_1 = validate_prior_alpha(n_simulations=10000, target_coverage_schedule= [0.8,0.85,0.9,0.95], coverage_weight=10)

#import pdb; pdb.set_trace()


#results = validate_seat_poll_model(n_simulations=10000, target_coverage_schedule= [0.8,0.85,0.9,0.95], coverage_weight=8, prior_alpha=11)

#results = perform_validation_testing(n_simulations=100,coverage_level=0.9,coverage_weight=10)


#print(results['2016'])
#print(results['2019'])
#print(results['2022'])
#print(results['2025'])
#print(results['Byelection'])



final_testing = perform_validation_testing_new(n_simulations = 10000, target_coverage_schedule= [0.8,0.85,0.9,0.95], coverage_weight=40, poll_alpha_fixed = 15, prior_alpha_fixed = 10)

print(final_testing['2016'])
print(final_testing['2019'])
print(final_testing['2022'])
print(final_testing['2025'])
print(final_testing['Byelection'])

# TO DO
import pdb; pdb.set_trace()


# make prominent-seat-poll-IND's share variable via dirichlet! - maybe unsafe, as it could eat into OTH...? FIX LATER
# transformed Linear model for C200 averages - not fixed! plt.scatter(C200_IND_splits['IND'].iloc[:-4], C200_IND_splits['Ratio'].iloc[:-4]) DONE

# By-eleciton specific: 
# ensure by-elections get appropriate priors for IND. (Wentworth only?) DONE 
# check main prior thread logic