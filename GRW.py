import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import os
from pathlib import Path
import matplotlib.pyplot as plt


# Simulated data
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
#az.style.use("seaborn-darkgrid")
az.rcParams['plot.max_subplots'] = 100


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

election_year = '2016'
SAMPLE_ERROR_SCALING_FACTOR = 1.5





num_polling_days = 100

import arviz as az

election_year = '2016'
last_election_year = str(int(election_year) - 3)
election_date_num = {'2013':1113, '2016':1028, '2019':1050, '2022':1099}

national_results = {'2016': np.array([42.04,34.73,10.23,13.00]), }
# national/state results should come from the summed prior, assuming similar enrolment to last time!


# 1. Total votes per seat last time.
# 2. Adjusted via linear transformation
# 3. Weighted sum of prior_weights and # votes

#name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}

last_election_vote_totals = pd.read_csv(f"{last_election_year}HouseVotesCountedByDivision.csv", skiprows=1, index_col=None).rename(columns={'DivisionNm':'old_div'})[['old_div', 'TotalVotes']]
#last_election_vote_totals['old_div'] = last_election_vote_totals['old_div'].replace(name_changes_year_dict[str(int(election_year)-3)])

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




def get_Prior_estimates_df(election_year):

    if election_year == '2016':
        Prior_estimates_df = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv", index_col = None) # ONLY WORKS FOR 2016 - FOR OTHER YEARS WILL REQUIRE Polling_Prior_Votes
    else:
        Prior_estimates_df = pd.read_csv(f"Polling_Prior_Votes_ON_add_For_{election_year}.csv", index_col = None)



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

    true_prior_estimates = pd.read_csv(f"Polling_Prior_Votes_For_{election_year}.csv", index_col = None)

    Prior_estimates_dict = {
        div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
        for div, group in Prior_estimates_df.groupby("div_nm")
    }
    # get true prior estimates wide format
    Prior_estimates_list = []
    for div in Prior_estimates_dict.keys():
        Prior_estimates_list.append(group_into_Categories(Prior_estimates_dict[div], div, election_year))

    Prior_estimates_original_df = pd.concat(Prior_estimates_list)

    ON_transfer_percent = {}

    # transfer prior %s from ON to original_df parties
    for div in Prior_estimates_dict.keys():
        if Prior_estimates_original_df.loc[Prior_estimates_original_df['ON']==0,]:
            curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df['div_nm'] == div,]
            curr_div_True = Prior_estimates_original_df.loc[Prior_estimates_original_df['div_nm'] == div,]

            transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).sum(axis=1)) # This should provide -1 for ON automatically!
            ON_transfer_percent[div] = transfer_proportions

            import pdb;pdb.set_trace()

    # distribute these ON votes to actual 
    for div in ON_transfer_percent.keys():
        Polling_estimates.loc[Polling_estimates.index == div,] =  Polling_estimates.loc[Polling_estimates.index == div,'ON'][0] * ON_transfer_percent[div]
        import pdb;pdb.set_trace()


    import pdb;pdb.set_trace()

    return Polling_estimates


Prior_estimates_df = get_Prior_estimates_df(election_year).rename(columns={'Other':'OTH'}) # adds ON to every seat if no ON (for 2019 and 2022)

# add 0.001 to Gorton Other!
if election_year == '2016':
    Prior_estimates_df.loc[Prior_estimates_df.index=='Gorton',['GRN','OTH']] += (-0.01,+0.01)

day_of_interest = 80
day_80_polling_avg =  pd.DataFrame([[0.0]*len(Prior_estimates_df.columns)], columns=Prior_estimates_df.columns)



merged_totals = Prior_estimates_df.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
weights = merged_totals['new_vote_totals']/merged_totals['new_vote_totals'].sum()
National_prior = (merged_totals.iloc[:,:-1] * weights.values[:,None]).sum().to_frame().T



def plot_GRW(x_posterior):

    # Compute summary statistics
    x_mean = np.mean(x_posterior, axis=(0, 1))  # Mean trajectory (T, K)
    x_lower = np.percentile(x_posterior, 2.5, axis=(0, 1))  # 2.5th percentile (T, K)
    x_upper = np.percentile(x_posterior, 97.5, axis=(0, 1))  # 97.


    # Time axis (100 days)
    time = np.arange(100)


    plt.figure(figsize=(10, 6))

    plt.plot(time, x_mean)
    plt.fill_between(time, x_lower, x_upper, alpha=0.2)

    plt.xlabel("Days")
    plt.ylabel("ALR Vote Share")
    plt.title("ALR Vote Share Trajectory with 95% Credible Intervals")
    plt.legend()
    plt.grid()
    plt.show()

    window_size = 5  # Adjust for smoothing level
    mean_trend_smooth = np.convolve(x_mean, np.ones(window_size)/window_size, mode='same')
    hdi_low_smooth = np.convolve(x_lower, np.ones(window_size)/window_size, mode='same')
    hdi_high_smooth = np.convolve(x_upper, np.ones(window_size)/window_size, mode='same')

    plt.figure(figsize=(10, 6))

    plt.plot(time, mean_trend_smooth)
    plt.fill_between(time, hdi_low_smooth, hdi_high_smooth, alpha=0.2)

    plt.xlabel("Days")
    plt.ylabel("ALR Vote Share")
    plt.title("ALR Vote Share Trajectory with 95% Credible Intervals")
    plt.legend()
    plt.grid()
    plt.show()

    return 1

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



# only do inference for last 100 days of election:
starting_point = election_date_num[election_year] - num_polling_days # start 100 days before last day of polling

National_polls = pd.read_csv(f'NationalPollsforMGRW{election_year}.csv')

for party in National_polls.columns[2:]:

    # Step 1: Load Data (Placeholder, replace with actual data)
    df = National_polls[['Days since last election','Sample size',party]] # Columns: [Days since last election, Sample size, COAL, ALP, GRN, party_4,...,Other]


    prior_poll_avg = df.loc[df['Days since last election'] < starting_point,].iloc[-10:,2:].mean()
    df = df.loc[df['Days since last election'] >= starting_point,]
    df.loc[:,'Days since last election'] -= starting_point
    df = df.rename(columns={'Days since last election':'Day_index'})

    days = df['Day_index'].values
    observed_days = np.array(sorted(set(days))) 
    num_polls = len(days)
    day_indices = np.array([np.where(observed_days == d)[0][0] + 1 for d in days]) 

    observed_votes = df[party].values
    sample_sizes = df['Sample size'].values

    # Compute precisions
    df["precision"] = df["Sample size"] / (df[party] * (1 - df[party]))

    # Aggregate by day (precision-weighted mean & new variance)
    agg_polls = df.groupby("Day_index").agg(
        vote_share_weighted=(party, lambda x: np.sum(x * df.loc[x.index, "precision"]) / np.sum(df.loc[x.index, "precision"])),
        total_precision=("precision", "sum")  # Sum of precisions
    ).reset_index()

    # Convert precision back to standard deviation
    agg_polls["poll_sd"] = np.sqrt(1 / agg_polls["total_precision"])




    with pm.Model() as model:

        init_dist = pm.Normal.dist(mu=prior_poll_avg, sigma=0.02)  # Adjust sigma based on uncertainty
        # Latent vote share following a Gaussian random walk
        vote_trend = pm.GaussianRandomWalk("vote_trend", sigma=0.005, shape=100, init_dist=init_dist)

        # Observed polls (Normal likelihood with poll-dependent variance)
        #poll_sd = pm.Deterministic("poll_sd", agg_polls["poll_sd"])
        observed = pm.Normal("observed", mu=vote_trend[agg_polls["Day_index"].values], sigma=SAMPLE_ERROR_SCALING_FACTOR*agg_polls["poll_sd"], observed=agg_polls["vote_share_weighted"])

        trace = pm.sample(1000, tune=1000, target_accept=0.9)


    x_posterior = trace.posterior["vote_trend"].values
    x_mean = np.mean(x_posterior, axis=(0, 1))  # Mean vote share over time
    day_80_polling_avg[party] = x_mean[day_of_interest]

    #plot_GRW(x_posterior)

day_80_polling_avg = day_80_polling_avg/ day_80_polling_avg.sum(axis=1)[0]

# get alr-swing of polls compared to prior
ref_col = 'COAL'
polling_alr = np.log(day_80_polling_avg.drop(columns=[ref_col]).div(day_80_polling_avg[ref_col], axis=0))
prior_alr = np.log(National_prior.drop(columns=[ref_col]).div(National_prior[ref_col], axis=0))
national_alr_swing = polling_alr - prior_alr

# transform prior votes to ALR, apply swing, back-transform
Prior_estimates_alr = np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))

Polling_estimates_alr = Prior_estimates_alr.add(national_alr_swing.iloc[0], axis=1)



Polling_estimates = alr_to_simplex_vectorized(Polling_estimates_alr,ref_col)[['ALP','COAL','GRN','OTH']]

# check that the aggregated results match the national polling average
merged_totals_polling = Polling_estimates.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
weights_polling = merged_totals_polling['new_vote_totals']/merged_totals_polling['new_vote_totals'].sum()
weighted_national_polling = (merged_totals.iloc[:,:-1] * weights.values[:,None]).sum().to_frame().T
import pdb;pdb.set_trace()

remove_ON_back_to_its_country(Polling_estimates)


plot_GRW(x_posterior)
import pdb;pdb.set_trace()
