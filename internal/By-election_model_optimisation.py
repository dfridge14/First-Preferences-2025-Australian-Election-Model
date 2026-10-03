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
MAJOR_SITOUT_DIR_ALPHA = 14 # (4 + 7 + 31)/3

SEAT_POLL_SCALE_1 = 1.0747964937804577
SEAT_POLL_SCALE_2 = 0.8977871632158285
SEAT_POLL_SCALE_3 = 0.7965537354979125


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


def alr_to_simplex_vectorized(df, ref_col):
    """Inverse ALR transformation for entire df"""

    # Convert to numpy, apply inverse ALR transformation
    alr_vals = df.values
    exp_vals = np.exp(alr_vals) 

    # Compute the reference category and other values
    ref_vals = 1 / (1 + np.sum(exp_vals, axis=1, keepdims=True))  # Shape: (n_samples, 1)
    simplex_vals = np.concatenate((exp_vals * ref_vals, ref_vals), axis=1)  # Shape: (n_samples, D)
    
    # Return as df with full columns
    new_columns = df.columns.tolist() + [ref_col]
    
    return pd.DataFrame(simplex_vals, columns=new_columns, index=df.index)

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
    Electorate_Residuals_cov =  pd.DataFrame()
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


import numpy as np

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

def expand_all_divisions_from_prior_df(sim, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=100):

    final_sim = {}
    party_name_dict = {} 

    multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
    C200_IND_splits = pd.read_csv(f"Independent_splits_multiple.csv", index_col=None)
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

                    if IND_votes<0.05: # misspecified prior for IND
                        C200_ratio = 0.9
                    else:
                        beta0,beta1 = C200_ratio_lm_params
                        logit_mu = beta0 + beta1 * np.log(IND_votes*100)
                        C200_ratio = 1 / (1 + np.exp(-logit_mu))


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


def REDUNDANT():
    election_year = 'Byelection'
    n_simulations = 10

    # get election-specific data
        
    div_order = Seat_poll_year_dict[election_year]['Electorate'].drop_duplicates().tolist()
    prior_df_active = prior_df_dict[election_year].loc[div_order] 


    if election_year == 'Byelection':
        # for byleeciton, take it group by group, ensuring they are correctly ordered

        by_election_groups = Byelection_final_polling_averages.reset_index().loc[lambda d: d['Election'].str[:-4].isin(prior_df_active.index)].groupby(['COAL','ALP','GRN','OTH'])['Election'].apply(list).tolist()
        order_map={k:i for i,k in enumerate(div_order)}
        by_election_groups=(Byelection_final_polling_averages.reset_index().loc[lambda d:d['Election'].str[:-4].isin(prior_df_active.index)].assign(div=lambda d:d['Election'].str[:-4]).sort_values('div',key=lambda s:s.map(order_map)).groupby(['COAL','ALP','GRN','OTH'])['Election'].apply(list).tolist())
        by_election_groups=sorted(by_election_groups,key=lambda g:min(order_map[e[:-4]] for e in g))
        
        byelection_group_sims = []
        group_sizes = [] # for safe appendage
        group_order = []


        for group in by_election_groups:

            polling_avg = Byelection_final_polling_averages.loc[Byelection_final_polling_averages.index.isin(group)].iloc[[0],:]
            National_prior = National_prior_dict[str(min(y for y in map(int, National_prior_dict.keys()) if y > int(polling_avg.index[0][-4:])))]
                

            # Narrow down prior_df_active to group
            prior_df_curr = prior_df_active.loc[[elec[:-4] for elec in group]]
            
            sim = simulate_Polling_Fundamentals_model(
                n_simulations=n_simulations,
                election_year=election_year,
                df_t=0,
                v=0.15,
                beta=0.5,
                Prior_estimates_df=prior_df_curr,
                polling_avg = polling_avg,
                National_prior = National_prior
            )

            assert sim.ndim == 3
            assert sim.shape[0] == n_simulations
            assert sim.shape[2] == 4

            byelection_group_sims.append(sim)
            group_sizes.append(sim.shape[1])
            group_order.append(group)
            
            # concatenate results for by-elections since 2013
            By_elections_Results = pd.read_csv('By-election_results.csv').iloc[:,:4]
            By_elections_Results.loc[By_elections_Results['div_nm']=='Batman','div_nm'] = 'Cooper'
            By_elections_Results = By_elections_Results[By_elections_Results['byelection_year']>=2013]

            Results_dict =  {div: g.assign(FP=g['FirstPreferencePercent']/100).groupby('PartyAb')['FP'].sum().to_frame().T.reset_index(drop=True) for div, g in By_elections_Results.groupby('div_nm')}
            Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # different approach for byelection
        
        sim = np.concatenate(byelection_group_sims, axis=1)

    else:
        polling_avg = pd.read_csv("Final_poll_averages_CAGO.csv")
        polling_avg = polling_avg.loc[polling_avg["Election"] == election_year,["COAL", "ALP", "GRN", "OTH"]]

        sim = simulate_Polling_Fundamentals_model(
            n_simulations=n_simulations,
            election_year=election_year,
            df_t=0,
            v=0.15,
            beta=0.5,
            Prior_estimates_df=prior_df_active,
            polling_avg = polling_avg,
            National_prior = National_prior_dict[election_year]
        )

        Prior_estimates_dict =  get_Prior_estimates_df(election_year, dont_add_ON = True)[1]
        Results_dict = get_results_df(election_year, to_Fundamentals=False)[1] if election_year != 'Byelection' else 1


    Prior_estimates_dict_active = {div: Prior_estimates_dict[div] for div in div_order}
    Results_dict_active = {div: Results_dict[div] for div in div_order}


    final_sim, party_name_dict = expand_all_divisions_from_prior_df(sim, Prior_estimates_dict_active, Results_dict_active, election_year, alpha_scalar=22)


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

    # error_matrix shape: (n_sim, 3)
    new_error_matrix = error_matrix @ A.T  # → (n_sim, 2)

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
    # if GRN vote share not reported, adjust GRN/OTH split (Longman, Bennelong, 2019 Flinders, Cowan, 2022 Mackellar, Norht Sydney, 2025 Bullwinkel)

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
    alr_error_row,
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

    alr_adj = alr_base + alr_error_row
    adjusted = alr_to_simplex_vectorized(alr_adj, ref_col)

    macro_oth = adjusted['Other'].iloc[0]

    alr_error_matrix = np.random.multivariate_normal(
        mean=np.zeros(3),
        cov=Seat_poll_covm_dict[election_year],
        size=n_simulations
    )

    if election_year == 'Byelection' and missing_party is not None:
        if poll_df[missing_party].sum() == 0:
            alr_error_matrix, new_cols, ref_col = transform_alr_matrix(alr_error_matrix, missing_party)
    else:
        new_cols, ref_col = ['ALP','GRN','Other'], 'COAL'

    alr_adj = alr_base.values + alr_error_matrix
    adjusted_cago = alr_to_simplex_vectorized_2d(alr_adj, ref_col, new_cols)
    cago_party_names = [ref_col] + new_cols
    oth_idx = cago_party_names.index('Other')

    macro_oth = adjusted_cago[:, oth_idx] 


    alpha_curr = (poll_weights.values * poll_alpha).astype(float)
    poll_draws = np.random.dirichlet(alpha_curr, size=n_simulations)
    poll_alloc_matrix = poll_draws * macro_oth[:, np.newaxis]

    poll_names = list(poll_weights.index)
    unpolled_minor_cols = [p for p in minor_cols if p not in poll_names]





    poll_alloc = pd.Series(0.0, index=minor_cols + ['OTH'])
    alpha_curr = (poll_weights.values*poll_alpha).astype(float)  # or scaled version if you want concentration
    draw = np.random.dirichlet(alpha_curr)
    poll_alloc.loc[poll_weights.index] = macro_oth * draw

    # 2. Prior reallocation for remaining minor parties - only if poll is incomplete
    remaining_oth = poll_alloc['OTH']
    unpolled_minor_cols = poll_alloc[poll_alloc == 0].index.tolist()


    if multi_ind_c200_case:

        if prior_df.loc[div, 'IND']<0.05: # misspecified prior for IND
            C200_ratio = 0.9
        else:
            logit_mu = beta0 + beta1 * np.log(prior_df.loc[div, 'IND']*100)
            C200_ratio = 1 / (1 + np.exp(-logit_mu))

        # scale down prior IND mass to residual space only, artificially
        prior_df.loc[div, 'IND'] = prior_df.loc[div, 'IND'] * (1 - C200_ratio)
        
        #  rename allocated IND to correct INDX
        poll_alloc.loc[f'IND{Position}'] = poll_alloc.loc['IND']

        # reset for remaiing INDs
        poll_alloc.loc['IND'] = 0
        unpolled_minor_cols += ['IND']

    if election_year == 'Byelection' and div == 'Wentworth': # special by-election case of multi_ind_c200_case. ADD: Farrer.
        # Heath gets 6.435 (polling average), Phelps: 20.974
        WENTWORTH_IND_RATIO = 20.974/(6.435 + 20.974)

        poll_alloc.loc['IND1'] = poll_alloc.loc['IND'] * WENTWORTH_IND_RATIO
        poll_alloc.loc['IND2'] = poll_alloc.loc['IND'] * (1 - WENTWORTH_IND_RATIO)

        poll_alloc.loc['IND'] = 0
        unpolled_minor_cols += ['IND']



    if len(unpolled_minor_cols) > 0 and remaining_oth > 0:
        prior_minors = prior_df.loc[div, unpolled_minor_cols]
        prior_weights = prior_minors / prior_minors.sum()
        alpha_prior = prior_weights.values * prior_alpha
        draw_prior = np.random.dirichlet(alpha_prior)

        poll_alloc.loc[prior_weights.index] = remaining_oth * draw_prior
        poll_alloc = poll_alloc.drop('OTH')
    elif remaining_oth == 0:
        poll_alloc = poll_alloc.drop('OTH')

    assert np.isclose(poll_alloc.sum(), macro_oth) # oth vote share correctly distirbuted

    final_row = pd.concat([adjusted.drop(columns=['Other']),poll_alloc.to_frame().T], axis=1)

    #### LP-NP split
    # --- Determine NP share (Dirichlet centre) ---

    COAL_double_div = False

    if ('LP' in Results_row) and ('NP' in Results_row):
        COAL_double_div = True

        NP_LP_split_polled = ('NAT' in poll_df.columns) and poll_df['NAT'].sum() > 0

        if NP_LP_split_polled:
            # Use poll-implied ratio
            nat = poll_df['NAT'].astype(float)
            coal = poll_df['COAL'].astype(float)

            nat_mean = nat.replace(0, np.nan).mean()
            adj_np = nat.where(nat > 0, nat_mean) # Only fill NAT where missing
            adj_lp = coal.where(nat > 0, coal - nat_mean).clip(lower=0) # Only subtract from COAL where NAT was missing

            total = adj_np.mean() + adj_lp.mean()
            if total > 0:
                NP_est = adj_np.mean() / total
            else:
                NP_LP_split = False # fallback if degenerate
            
        if not NP_LP_split_polled:
            # use prior-based split
            
            if div in NP_ratios_curr['div_nm'].unique():
                if 'NP' in prior_df.columns and 'LP' in prior_df.columns:
                    NP_est = (prior_df['NP'] / prior_df[['LP','NP']].sum(axis=1)).iloc[0]
                else:
                    NP_est = NP_ratios_curr.loc[
                        NP_ratios_curr['div_nm'] == div, 'final_estimate'
                    ].iloc[0]

                # 2025 override
                if (election_year == '2025') & (div in ['Bullwinkel','Forrest',"O'Connor"]):
                    NP_est = NP_ratios_curr.loc[
                        NP_ratios_curr['div_nm'] == div, 'final_estimate'
                    ].iloc[0]


        # dirichlet split of COAL vote into NP and LP
        alpha = np.array([1 - NP_est, NP_est]) * LP_NP_DIR_ALPHA
        splits = np.random.dirichlet(alpha)
        LP_NP_votes = splits * final_row['COAL'].iloc[0]  # (n_sim=1, 2)
        LP_mean = LP_NP_votes[0].mean()
        NP_mean = LP_NP_votes[1].mean()

        # remove COAL, replace with LP/NP
        final_row = final_row.drop(columns=['COAL'])
        final_row[['LP', 'NP']] = [LP_mean, NP_mean]


    

    if 'IND' in final_row.columns:

        # split IND (or any that remains)
        ind_total = final_row['IND'].iloc[0]

        if multi_ind_c200_case:
             
            free_weights = np.full(k - 1, 1 / (k - 1))
            IND_draw = np.random.dirichlet(free_weights * prior_alpha)

            ind_means = IND_draw * ind_total

            final_row = final_row.drop(columns=['IND'])

            idx = 0
            for i in range(1, k + 1):
                if i == Position:
                    continue
                final_row[f'IND{i if i < Position else i}'] = ind_means[idx]
                idx += 1
                
        elif election_year == 'Byelection' and div == 'Wentworth':
            final_row = final_row.rename(columns={'IND':'IND3'}) # manually set

        else:

            if div in multiple_INDs_df['div_nm'].unique():

                means_array = np.full(k, 1 / k)

                # --- C200 structured split ---
                if div in C200_IND_positions_curr['div_nm'].unique():

                    pos = int(C200_IND_positions_curr.loc[
                        C200_IND_positions_curr['div_nm'] == div,
                        'Number'
                    ].iloc[0]) - 1

                    if ind_total<5: # misspecified prior for IND
                        C200_ratio = 0.9
                    else:
                        logit_mu = beta0 + beta1 * np.log(ind_total)
                        C200_ratio = 1 / (1 + np.exp(-logit_mu))

                    import pdb; pdb.set_trace()

                    rest = (1 - C200_ratio) / (k - 1)
                    means_array[:] = rest
                    means_array[pos] = C200_ratio

                # --- Dirichlet split ---
                splits = np.random.dirichlet(means_array * prior_alpha)

                ind_means = splits * ind_total

                # replace IND with IND1..INDk
                final_row = final_row.drop(columns=['IND'])

                for i in range(k):
                    final_row[f'IND{i+1}'] = ind_means[i]

            else:
                # single IND case → just rename
                final_row = final_row.rename(columns={'IND': 'IND1'})

        
    if (not COAL_double_div) and not (missing_party == 'COAL'):
        COAL_replacement_list = ['LP', 'NP', 'LNP', 'CLP']

        coal_party = next(p for p in Results_row.columns if p in COAL_replacement_list)
        if 'COAL' in final_row.columns:
            final_row = final_row.rename(columns={'COAL': coal_party})

    assert set(final_row.columns) == set(Results_row.columns)
    

    return final_row


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
                alr_error_matrix, new_cols, ref_col = transform_alr_matrix(alr_error_matrix, missing_party)
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

        for s in range(n_simulations):
            alr_error_row = pd.DataFrame([alr_error_matrix[s]], columns=new_cols)

            out = process_division(
                poll_df,
                prior_row,
                Results_row,
                div,
                election_year,
                alr_error_row=alr_error_row,
                missing_party=missing_party,
                ref_col=ref_col,
                prior_alpha=prior_alpha,
                poll_alpha = poll_alpha,
                precomputed_dict=precomputed_dict,
                GLOBAL_CSVs=GLOBAL_CSVs,
                n_simulations=n_simulations
            )

            results.append(out)

        seat_poll_sims[div] = pd.concat(results)

        # return numpy array in order of Ballot appearance
        Ballot_order = Results_dict_active[election_year][div].columns.tolist()
        seat_poll_sims[div] = seat_poll_sims[div][Ballot_order].to_numpy()

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





def validate_prior_alpha(n_simulations, coverage_level, coverage_weight=5):
    """
    Optimized LOO-CV for poll_alpha in the seat model.
    Precomputes all simulations to avoid O(N^2) complexity.
    """

    prior_alphas = np.linspace(2, 50, 25)

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

                mae, coverage = get_election_MAE(
                    nat_sim,
                    Results_dict_active_per_election[election],
                    coverage_level
                )

                penalty = max(0, coverage_level - coverage)
                val_scores.append(mae + coverage_weight * penalty)

            results.append((alpha, np.mean(val_scores)))

        # =====================================================
        # SELECT BEST (Fixed index to [1] to match (alpha, score))
        # =====================================================
        best_alpha, best_val_score = min(results, key=lambda x: x[1])

        # =====================================================
        # HELDOUT EVAL (Lookup, no re-simulation needed)
        # =====================================================
        heldout_sim = all_sims[best_alpha][heldout]

        heldout_mae, heldout_coverage = get_election_MAE(
            heldout_sim,
            Results_dict_active_per_election[heldout],
            coverage_level
        )

        best_params[heldout] = {
            "poll_alpha": best_alpha,
            "val_score": best_val_score,
            "test_mae": heldout_mae,
            "test_coverage": heldout_coverage
        }

        #import pdb; pdb.set_trace()
        
        print(f"  Best Alpha: {best_alpha} | Test MAE: {heldout_mae:.4f} | Test Coverage: {heldout_coverage:.4f}")

    return best_params




def validate_seat_poll_model(n_simulations, coverage_level, coverage_weight=5, prior_alpha = 12):
    """
    Optimized LOO-CV for poll_alpha in the seat model.
    Precomputes all simulations to avoid O(N^2) complexity.
    """

    poll_alphas = np.linspace(12, 50, 40)

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

            asd = time.time()
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
            print(time.time()-asd)
            
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

                mae, coverage = get_election_MAE(
                    seat_sim,
                    Results_dict_active_per_election[election],
                    coverage_level
                )

                penalty = max(0, coverage_level - coverage)
                val_scores.append(mae + coverage_weight * penalty)

            results.append((alpha, np.mean(val_scores)))

        # =====================================================
        # SELECT BEST (Fixed index to [1] to match (alpha, score))
        # =====================================================
        best_alpha, best_val_score = min(results, key=lambda x: x[1])

        # =====================================================
        # HELDOUT EVAL (Lookup, no re-simulation needed)
        # =====================================================
        heldout_sim = all_sims[best_alpha][heldout]

        heldout_mae, heldout_coverage = get_election_MAE(
            heldout_sim,
            Results_dict_active_per_election[heldout],
            coverage_level
        )

        best_params[heldout] = {
            "poll_alpha": best_alpha,
            "val_score": best_val_score,
            "test_mae": heldout_mae,
            "test_coverage": heldout_coverage
        }
        
        print(f"  Best Alpha: {best_alpha} | Test MAE: {heldout_mae:.4f}")

    return best_params



#results_1 = validate_prior_alpha(n_simulations=10000, coverage_level=0.9, coverage_weight=10)

#import pdb; pdb.set_trace()


results = validate_seat_poll_model(n_simulations=10, coverage_level=0.95, coverage_weight=5)

#results = perform_validation_testing(n_simulations=100,coverage_level=0.9,coverage_weight=10)

#print(results['2016'])
#print(results['2019'])
#print(results['2022'])
#print(results['2025'])
#print(results['Byelection'])

import pdb; pdb.set_trace()

# TO DO

# make prominent-seat-poll-IND's share variable via dirichlet! - maybe unsafe, as it could eat into OTH...? FIX LATER
# transformed Linear model for C200 averages - not fixed! plt.scatter(C200_IND_splits['IND'].iloc[:-4], C200_IND_splits['Ratio'].iloc[:-4]) DONE

# By-eleciton specific: 
# ensure by-elections get appropriate priors for IND. (Wentworth only?) DONE 
# check main prior thread logic