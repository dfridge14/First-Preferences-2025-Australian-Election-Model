import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import multivariate_normal, dirichlet

from scipy.stats import multivariate_t
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize


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

election_date_num = {'2013':1113, '2016':1028, '2019':1050, '2022':1099,'2025':1078}



election_years = ['2013','2016','2019','2022','2025']


def group_into_Categories(party_votes_shares_df, div, election_year, is_Other = True):
    # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

    ALP_cat = {'ALP','CLR'}
    COAL_cat = {'COAL','COALNP','COALLP','LP','NP','NAT','CLP','LNP','LNQ'}
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
    if election_year == '2016':
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum6}], index=[div])
    elif election_year in ['2019','2022','2025']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'UAPP':sum5, 'Other':sum6}], index=[div])

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




def get_results_dict(election_year, CAGO = True):

    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    Actual_results.loc[:,'PartyAb'] = Actual_results['PartyAb'].replace({'CYA':'TOP'})

    # Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
    Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
    Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
    Actual_results.loc[Actual_results['PartyAb']=='GVIC','PartyAb'] = 'GRN'
    Actual_results.loc[Actual_results['PartyAb']=='CLR','PartyAb'] = 'ALP'


    Four_party_results_list = []

    Actual_results_dict = {}

    # rename IND to INDX by order
    target = 'IND'

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

        if CAGO:
            Four_party_results_list.append(group_into_Fundamentals_Categories(Actual_results_dict[div], div))
        else:
            Four_party_results_list.append(group_into_Categories(Actual_results_dict[div], div, election_year))

    results_df = pd.concat(Four_party_results_list)
    results_df = results_df.div(results_df.sum(axis=1), axis=0).rename(columns={'Other':'OTH'})

    #import pdb;pdb.set_trace()

    return results_df


# create dicts of seat poll and results dfs for each election

Seat_poll_year_dict = {}
Actual_results_dict_year = {}

for election_year in election_years:

    Seat_polls = pd.read_csv(f"SeatPolls{election_year}Formatted.csv", index_col=None)
    Seat_polls.loc[:,'Days since last election'] = election_date_num[election_year] - Seat_polls.loc[:,'Days since last election']
    Seat_polls.rename(columns={'Days since last election':'Days before election'}, inplace=True)

    Seat_poll_year_dict[election_year] = Seat_polls

    # get actual results 
    Actual_results_df = get_results_dict(election_year)
    Actual_results_dict_year[election_year] = Actual_results_df

# add by_election seat polls/results to dicts
By_election_data = pd.read_csv("Federal_by_election_data.csv")
Seat_polls =  pd.read_csv(f"SeatPollsByelectionFormatted.csv", index_col=None)

# format first col into days before byelection via merge with By_election_data:
By_election_data['byelection_date'] = pd.to_datetime(By_election_data['byelection_date'],format='%d.%m.%y')
By_election_data['byelection_year'] = By_election_data['byelection_date'].dt.year

Seat_polls = Seat_polls.merge(By_election_data[['days_since_election','byelection_year','div_nm']], left_on = ['byelection_year','Electorate'], right_on = ['byelection_year','div_nm']).drop(columns=['div_nm'])
Seat_polls.loc[:,'Days since last election'] = Seat_polls.loc[:,'days_since_election'] - Seat_polls.loc[:,'Days since last election']
Seat_polls = Seat_polls.rename(columns={'Days since last election':'Days before election'}).drop(columns=['days_since_election'])
Seat_poll_year_dict['Byelection'] = Seat_polls

# get results dict for byelections
By_election_results = pd.read_csv('By-election_results.csv')
byelection_results_CAGO = []

By_election_results.loc[:,'div_nm'] = By_election_results['div_nm'] + By_election_results['byelection_year'].astype(str)

for (div, year), subdf in By_election_results.groupby(['div_nm','byelection_year']):

    
    # pivot individual byelection
    wide = subdf.pivot_table(index=None,columns='PartyAb',values='FirstPreferencePercent',aggfunc='sum')
    wide = wide / 100

    grouped = group_into_Fundamentals_Categories(wide, div).rename(columns={'Other':'OTH'})

    byelection_results_CAGO.append(grouped)

Actual_results_dict_year['Byelection'] = pd.concat(byelection_results_CAGO)


def process_curr_seat_polls(election_year, Seat_poll_year_dict, Actual_results_dict_year, filter_multi_polls=False):
    
    df = Seat_poll_year_dict[election_year].copy()
    df = df.loc[df['GRN'] > 0]

    if election_year != '2013':
        info_party_cols = df.columns[:3 + (election_year=='Byelection')].tolist() + ["COAL", "ALP", "GRN"]
        df["OTH"] = df.drop(columns=info_party_cols).sum(axis=1)
        df = df[info_party_cols + ["OTH"]]

        if election_year=='Byelection':
            df.loc[:,'Electorate'] = df['Electorate'] + df['byelection_year'].astype(str)
            df = df.drop(columns = ['byelection_year'])

    df = df[(df != 0).all(axis=1)].set_index('Electorate')

    # Apply filter for electorates with multiple seat polls
    if filter_multi_polls:
        counts = df.index.value_counts()
        df = df.loc[counts[counts > 1].index]



    # --- ALR transform ---
    ref_col = 'COAL'
    poll_alr = np.log(df.iloc[:, -4:].drop(columns=[ref_col]).div(df.iloc[:, -4:][ref_col], axis=0))

    actual = Actual_results_dict_year[election_year].loc[df.index]
    actual_alr = np.log(actual.drop(columns=[ref_col]).div(actual[ref_col], axis=0))

    polling_miss = actual_alr - poll_alr
    polling_miss['Days before election'] = df['Days before election']
    polling_miss['Election_year'] = election_year

    # --- Absolute differences ---
    for party in ["COAL", "ALP", "GRN", "OTH"]:
        df[f"{party}_abs_diff"] = (
            df[party] - actual[party]
        ).abs()

    # --- Long format ---
    parties = ["COAL","ALP","GRN","OTH"]

    poll_val_df = df.reset_index().melt(
        id_vars=["Days before election","Sample size",'Electorate'],
        value_vars=parties,
        var_name="PartyAb",
        value_name="Poll"
    )

    abs_diff_df = df.reset_index().melt(
        id_vars=["Days before election","Sample size",'Electorate'],
        value_vars=[p + "_abs_diff" for p in parties],
        var_name="PartyAb_diff",
        value_name="Abs Difference"
    ).assign(PartyAb=lambda d: d.PartyAb_diff.str.replace("_abs_diff",""))

    store_df = poll_val_df.merge(
        abs_diff_df[["Abs Difference"]],
        left_index=True,
        right_index=True
    )[['Days before election','Sample size','Electorate','PartyAb','Abs Difference']]

    store_df['election_year'] = election_year

    return df, polling_miss, store_df


# get all COAL-ALP-GRN first i.e. all for whom GRN is not 0 and compare!


plt.figure(figsize=(10, 6))
x_axis = 'Sample size'


CAGO_abs_diff_list = []
Polling_misses_ALR_list = []
store_list = []

for election_year in election_years + ['Byelection']:
    df, polling_miss, store_df = process_curr_seat_polls(
        election_year,
        Seat_poll_year_dict,
        Actual_results_dict_year,
        filter_multi_polls=False   # 🔁 toggle here
    )

    CAGO_abs_diff_list.append(df)
    Polling_misses_ALR_list.append(polling_miss)
    store_list.append(store_df)

all_store = pd.concat(store_list, ignore_index=True)
party_colors = {"COAL": "blue", "ALP": "red", "GRN": "green", "OTH": "gray"}
markers = {'2013':'.','2016':'s','2019':'^','2022':'x','2025':'p','Byelection':'o'}

plt.figure(figsize=(10,6))
for (year, party), subset in all_store.groupby(['election_year','PartyAb']):
    plt.scatter(
        subset['Days before election'],   # <- your new x-axis
        subset['Abs Difference'],
        color=party_colors[party],
        marker=markers[year],
        s=10 if year != 'Byelection' else 40,
        alpha=0.7
    )
plt.xlabel("Days before election")
plt.ylabel("Absolute Difference")
plt.title("Absolute Difference Between Polls and Results")
plt.legend()
#plt.show()

poll_level = (
    all_store
    .groupby(['election_year','Electorate','Days before election'])['Abs Difference']
    .mean()
    .reset_index()
)

def compute_slope(df):
    if df['Days before election'].nunique() < 2:
        return np.nan
    return np.polyfit(df['Days before election'], df['Abs Difference'], 1)[0]

slopes = (
    poll_level
    .groupby(['election_year','Electorate'])
    .apply(compute_slope)
    .reset_index(name='slope')
    .dropna()
)


# plot slopes as histograms, by election, and test significance
plt.hist(slopes['slope'], bins=30)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Slope (AbsDiff vs Days before election)")
plt.title("Within-electorate trend in polling error")
#plt.show()

import seaborn as sns

sns.boxplot(data=slopes, x='election_year', y='slope')
plt.axhline(0, color='red', linestyle='--')
plt.title("Within-electorate error trends by year")
#plt.show()

# significance tests -> wide tails skewed towards positive slope (outliers!)
print(stats.wilcoxon(slopes['slope']))
print(stats.ttest_1samp(slopes['slope'], 0))


Polling_misses_ALR_df = pd.concat(Polling_misses_ALR_list)

# plot std vs number of seat polls

counts_df = (
    Polling_misses_ALR_df.reset_index()
      .groupby(['Electorate','Election_year'])
      .size()
      .rename('n_polls')
      .reset_index()
)
Polling_misses_vs_n_polls = Polling_misses_ALR_df.copy()

Polling_misses_vs_n_polls = (
    Polling_misses_vs_n_polls.reset_index()
    .merge(counts_df, on=['Electorate','Election_year'], how='left')
    .set_index('Electorate')
)

# bucket poll counts: 1, 2, 3, 4+
Polling_misses_vs_n_polls['poll_bucket'] = Polling_misses_vs_n_polls['n_polls'].clip(upper=4)

# reshape
alr_long = Polling_misses_vs_n_polls.reset_index().melt(
    id_vars=['Electorate','Election_year','poll_bucket'],
    value_vars=['ALP','GRN','OTH'],
    var_name='Party',
    value_name='ALR_error'
)

avg_df = (
    Polling_misses_vs_n_polls
    .groupby(['Electorate','Election_year'])
    .agg({
        'ALP':'mean',
        'GRN':'mean',
        'OTH':'mean',
        'n_polls':'first'
    })
    .reset_index()
)

# different covms depending on number of polls in seat average
avg_df_general = avg_df.loc[avg_df['Election_year']!='Byelection',]


# bucket
avg_df['poll_bucket'] = avg_df['n_polls'].clip(upper=3)
avg_df_general['poll_bucket'] = avg_df_general['n_polls'].clip(upper=3)

print(avg_df_general.groupby('n_polls').count())
print(np.cov(avg_df_general.loc[avg_df_general['poll_bucket']>=3,['ALP','GRN','OTH']],rowvar=False))

avg_df_general.to_csv("X_gen_seat.csv", index = False)

R = np.corrcoef(avg_df_general[['ALP','GRN','OTH']], rowvar=False)
avg_df_general[['ALP','GRN','OTH']].var()

avg_df_general.loc[avg_df_general['poll_bucket']==1,['ALP','GRN','OTH']].var().mean()
avg_df_general.loc[avg_df_general['poll_bucket']==2,['ALP','GRN','OTH']].var().mean()
avg_df_general.loc[avg_df_general['poll_bucket']==3,['ALP','GRN','OTH']].var().mean()

# 1. The Base: Lock in the robust structural covariance from ALL data
base_covm = avg_df_general[['ALP', 'GRN', 'OTH']].cov()
base_stds = avg_df_general[['ALP', 'GRN', 'OTH']].std()

# 2. Extract standard deviations for each bucket
stds_1 = avg_df_general.loc[avg_df_general['poll_bucket'] == 1, ['ALP', 'GRN', 'OTH']].std()
stds_2 = avg_df_general.loc[avg_df_general['poll_bucket'] == 2, ['ALP', 'GRN', 'OTH']].std()
stds_3 = avg_df_general.loc[avg_df_general['poll_bucket'] >= 3, ['ALP', 'GRN', 'OTH']].std()

# 3. Calculate the global decay scalar for each bucket
# We calculate the ratio of the bucket's std to the base std, and average it across the 3 parties.
# This prevents a noisy OTH variance in bucket 3 from hijacking the multiplier.
scale_1 = (stds_1 / base_stds).mean()
scale_2 = (stds_2 / base_stds).mean()
scale_3 = (stds_3 / base_stds).mean()


import numpy as np
from scipy.optimize import curve_fit

# Use your bucket means and average variances
# Assume n values for buckets: bucket 1 (n=1), bucket 2 (n=2), bucket 3+ (avg n~3.5)
n_vals = np.array([1, 2, 3.5])
var_vals = np.array([
    avg_df_general.loc[avg_df_general['poll_bucket'] == 1, ['ALP','GRN','OTH']].var().mean(),
    avg_df_general.loc[avg_df_general['poll_bucket'] == 2, ['ALP','GRN','OTH']].var().mean(),
    avg_df_general.loc[avg_df_general['poll_bucket'] >= 3, ['ALP','GRN','OTH']].var().mean()
])

# Fit a decay curve: Variance = structural_error + sampling_error/n
def var_model(n, alpha, beta):
    return alpha + beta / n

params, _ = curve_fit(var_model, n_vals, var_vals)
alpha, beta = params # alpha is the 'intrinsic' seat error, beta is the 'polling noise'

# Now define a scalar function relative to your 'base_covm'
v_baseline = var_model(1.54, alpha, beta) # The variance level inherent in Sigma_gen_seat

def get_n_scaling(n):
    # Returns the variance multiplier for a specific number of polls
    return var_model(n, alpha, beta) / v_baseline

def process_partial_weighted(partial_list, Sigma_base, alpha, beta, v_baseline):
    energies = []
    
    for obs in partial_list:
        n = obs.get("n", 2.15) # Default to your by-election average
        scalar = get_n_scaling(n)
        
        # Scale the covariance for THIS specific observation
        Sigma_n = Sigma_base * scalar
        
        p_poll = obs["p_poll"]
        p_res  = obs["p_result"]
        parties = obs["parties"]
        
        ref = parties.index("COAL") if "COAL" in parties else 0
        y = to_alr(p_poll, ref) - to_alr(p_res, ref)
        
        keep = [i for i in range(3) if i != ref]
        A = np.zeros((len(y), 3))
        A[:, keep] = np.eye(len(y))
        
        S = A @ Sigma_n @ A.T
        energies.append(y.T @ np.linalg.solve(S, y))
        
    return sum(energies)



import pdb; pdb.set_trace()

# compare covariance magnitude
for k in [1,2,3,4]:
    df_k = avg_df[avg_df['poll_bucket'] == k]
    cov = np.cov(df_k[['ALP','GRN','OTH']].values, rowvar=False)
    print(k, np.linalg.norm(cov))

import pdb; pdb.set_trace()

# plot std by bucket
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

for party in ['ALP','GRN','OTH']:
    df_p = alr_long[alr_long['Party'] == party]
    grouped = df_p.groupby('poll_bucket')['ALR_error'].std()
    plt.plot(grouped.index, grouped.values, marker='o', label=party)

plt.xticks([1,2,3,4], ['1','2','3','4+'])
plt.xlabel("Number of polls")
plt.ylabel("Std of ALR error")
plt.title("ALR error vs poll count (bucketed)")
plt.legend()
plt.grid(True)
#plt.show()


byelection_seat_poll_avg_err = Polling_misses_ALR_df.loc[Polling_misses_ALR_df['Election_year']=='Byelection',].iloc[:,:3].groupby('Electorate').mean()

# get Covm error estimates per general/byelection regime
byelection_seat_poll_avg_err = Polling_misses_ALR_df.loc[Polling_misses_ALR_df['Election_year']=='Byelection',].iloc[:,:3].groupby('Electorate').mean()

print(np.cov(Polling_misses_ALR_df.loc[~(Polling_misses_ALR_df['Election_year']=='Byelection'),].groupby(['Election_year','Electorate']).mean().iloc[:,:3], rowvar=False))
print(np.cov(byelection_seat_poll_avg_err, rowvar=False))

Polling_misses_ALR_df.iloc[:,:3] = Polling_misses_ALR_df.iloc[:,:3] - Polling_misses_ALR_df.iloc[:,:3].mean()
Polling_misses_df = pd.concat(store_list, ignore_index=True)

import pdb;pdb.set_trace()


# expand polling weighted based on recency and sample size, across to electorates

import statsmodels.api as sm
errors_df = Polling_misses_ALR_df.loc[Polling_misses_ALR_df['Election_year'] == '2013',].copy()


def get_days_coef(y):
    X = sm.add_constant(errors_df[['Days before election']])
    model = sm.OLS(errors_df[y] ** 2, X).fit()
    return model.params['Days before election']

# Get slope (variance/day) for each component
slopes = {party: get_days_coef(party) for party in ['ALP', 'GRN']} # 'OTH' is too volatile

avg_slope = sum(slopes.values()) / len(slopes)

# get intercepts with this forced slope!
intercepts = {}
for party in ['ALP', 'GRN', 'OTH']:
    y = (errors_df[party] ** 2).values  # squared error
    x = errors_df['Days before election'].values
    intercept = (y - avg_slope * x).mean()
    intercepts[party] = intercept

def variance_estimate(party, days):
    return intercepts[party] + avg_slope * days

#import pdb;pdb.set_trace()


# Target: log squared error

for p in ['ALP','GRN','OTH']:
    errors_df.loc[:,f'sq_err_{p}'] = errors_df[p]**2  # add epsilon to avoid log(0)



    X = errors_df[['Days before election']]
    X = sm.add_constant(X)
    y = errors_df[f'sq_err_{p}']

    #0.04; 0.0013
    #0.083; 0.0011


    model = sm.OLS(y, X).fit()

    var, drift = model.params
    print(model.summary())





# Seat poll groupings:

Division_Other_groupings = {'2016': {'XEN': ['Higgins','Calare','Lindsay','Macarthur','Warringah','Groom','Moreton','Adelaide','Barker','Boothby','Grey','Hindmarsh','Kingston','Makin','Mayo','Port Adelaide','Sturt','Wakefield'],'NSW_IND':{'New England','Cowper','Lyne'}},
                            '2019': {'IND': ['Wentworth', 'Warringah']},
                            '2022': {'C200':['Boothby','Bradfield','Calare','Casey','Clark','Cowper','Curtin','Flinders','Goldstein','Grey','Hughes','Indi','Kooyong','Mackellar','Mayo','North Sydney','Page','Wannon','Warringah','Wentworth']},
                            '2025': {'C200':['Curtin','Goldstein','Indi','Kooyong','Mackellar','Mayo','Wentworth','Clark','Moncrieff','Moore','Bradfield','Berowra','Forrest','Sturt','Gilmore','Wannon','Casey','Franklin','Cowper','Groom','Calare','Fremantle','Fisher','Grey','Monash','Lyne','Farrer','Mcpherson','Deakin','Bean','Riverina','Solomon','Flinders','Dickson','Fairfax'],'Muslim':['Blaxland','Calwell','Watson']}
                            }





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

    return Prior_estimates_df, Prior_estimates_dict



def reallocate_oth(poll_df, prior_df):

    # fills in zeros for polls where they missed a candidate

    updated_poll = poll_df.copy().rename(columns={'TOP':'UAPP'}) # rename TOP to UAPP in 2025 case
    party_cols = ['COAL', 'ALP', 'GRN', 'ON', 'UAPP', 'OTH']
    parties = [p for p in party_cols if p != 'OTH']

    for idx in range(len(updated_poll)):
        poll_row = updated_poll.iloc[idx]
        div_nm = poll_row.name  # assumes index is div_nm; use poll_row['div_nm'] if it's a column

        prior_row = prior_df.loc[div_nm, party_cols]


        missing_parties_mask = (poll_row[parties] == 0) & (prior_row[parties] > 0)
        if not missing_parties_mask.any():
            continue

        oth_value = poll_row['OTH']

        if oth_value == 0:
            continue

        parties_to_fill = missing_parties_mask[missing_parties_mask].index.tolist()

        prior_subset = prior_row[parties_to_fill + ['OTH']]
        prior_weights = prior_subset / prior_subset.sum()

        redistribution = oth_value * prior_weights
        
        total_redistributed = 0.0
        for party, value in redistribution.items():
            updated_poll.iat[idx, updated_poll.columns.get_loc(party)] += value
            total_redistributed += value

        # Subtract only the redistributed portion from OTH
        updated_poll.iat[idx, updated_poll.columns.get_loc('OTH')] -= total_redistributed

    return updated_poll


ALR_error_list = []
Prop_error_list = []

ref_col = 'COAL'

for election_year in ['2019','2022','2025']:

    Prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0]
    Results_df = get_results_dict(election_year, CAGO = False)

    Seat_Poll_curr_df = Seat_poll_year_dict[election_year].set_index('Electorate')

    if 'NAT' in Seat_Poll_curr_df.columns:
        Seat_Poll_curr_df.loc[:,'COAL'] = (Seat_Poll_curr_df['COAL'] + Seat_Poll_curr_df['NAT'])
        Seat_Poll_curr_df = Seat_Poll_curr_df.drop(columns='NAT')

    if election_year != '2013': # already formatted in 2013

        main_party_list = ["COAL", "ALP", "GRN",'ON','UAPP','IND'] if election_year != '2025' else ["COAL", "ALP", "GRN",'ON','TOP','IND']

        # Want to reduce to COAL, ALP, GRN, and all the rest into Other
        info_party_cols = Seat_Poll_curr_df.columns[:2].tolist() + main_party_list
        Seat_Poll_curr_df["OTH"] = Seat_Poll_curr_df.drop(columns=info_party_cols).sum(axis=1)     # Create the 'OTH' column by summing all other columns
        Seat_Poll_curr_df = Seat_Poll_curr_df[info_party_cols + ["OTH"]]

    Prior_estimates_df = Prior_estimates_df.rename(columns={'Other':'OTH'})

    Reallocated_polls = reallocate_oth(Seat_Poll_curr_df.iloc[:,2:].drop('IND', axis=1), Prior_estimates_df)
    Reallocated_polls.loc[:,'OTH'] += Seat_Poll_curr_df['IND']

    # replace non-contesting places with small value i.e. 0.001, in both poll and prior values - this means only shift that is determined is between COAL in prior & poll
    SUBSTITUTE_SMALL_VAL = 1
    if SUBSTITUTE_SMALL_VAL:
        SMALL_VALUE_FOR_NON_RUNNNING = 0.001
        #Reallocated_polls = Reallocated_polls.replace(0.0,SMALL_VALUE_FOR_NON_RUNNNING)
        
        #DIVS_WITH_NO_ON = Prior_estimates_df[Prior_estimates_df['ON'] == 0].index
        Prior_estimates_Reallocated = Prior_estimates_df.copy().replace(0.0,SMALL_VALUE_FOR_NON_RUNNNING)
        Results_df_adjusted = Results_df.copy().replace(0.0,SMALL_VALUE_FOR_NON_RUNNNING)

    Reallocated_polls_ALR = np.log(Reallocated_polls.drop(columns=[ref_col]).div(Reallocated_polls[ref_col], axis=0))
    Results_ALR = np.log(Results_df_adjusted.drop(columns=[ref_col]).div(Results_df_adjusted[ref_col], axis=0)) 

    # DON'T NEED to compare with prior_estimates: Prior_estimates_Reallocated_ALR = np.log(Prior_estimates_Reallocated.drop(columns=[ref_col]).div(Prior_estimates_Reallocated[ref_col], axis=0))

    Error_ALR = Reallocated_polls_ALR - Results_ALR.loc[Reallocated_polls_ALR.index]
    Error_Prop = Reallocated_polls - Results_df.loc[Reallocated_polls.index]

    ALR_error_list.append(Error_ALR)
    Prop_error_list.append(Error_Prop)

    #import pdb;pdb.set_trace()

    # remove ON_boosts where no ON finally

Prop_errors = pd.concat(Prop_error_list)
print(Prop_errors.mean())
#import pdb;pdb.set_trace()



# How best to assess/compare performance of seat polls? 

# Must compare their performance when projected onto the correct set of candidates (including dispersing 'OTH')


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

def transform_alr_draw(error, missing_party):
    # linear transformation for lower-dimensional error realisation when major party is missing
    # CHECK: should this be applied when vote of 'OTH' = 0 too?
    if missing_party == 'COAL':
        A = np.array([
            [-1, 1, 0], # GRN/ALP
            [-1, 0, 1], # OTH/ALP
        ])
        new_cols = ['GRN','Other']
        ref_col = 'ALP'
    elif missing_party == 'GRN':
        A = np.array([
            [1, 0, 0], # ALP/COAL
            [0, 0, 1], # OTH/COAL
        ])
        new_cols = ['ALP','Other']
        ref_col = 'COAL'
    elif missing_party == 'ALP':
        # change ref to GRN
        A = np.array([
            [0, 1, 0],  # GRN/COAL
            [0, 0, 1],  # OTH/COAL
        ])
        new_cols = ['GRN','Other']
        ref_col = 'COAL'
    else:
        raise ValueError("Unsupported party")

    new_error = A @ (error.values[0])
    return new_error, new_cols, ref_col

def process_division(
    poll_df,
    prior_df,
    div,
    alr_error_row,
    missing_party = None,
    ref_col='COAL',
    dirichlet_alpha=1.0
):
    
    # ADAPTATION NEEDED: split for LP/NP, INDs (with poll or not)
    # FIX: currently requires alr_error_row instead of having already added it.


    minor_cols = [p for p in prior_df.columns if p not in ['ALP','COAL','LP','NP','LNP','CLP','GRN']]


    # if GRN vote share not reported, adjust GRN/OTH split (Longman, Bennelong, 2019 Flinders, Cowan, 2022 Mackellar, Norht Sydney, 2025 Bullwinkel)
    if (poll_df['GRN']==0).any():
        import pdb; pdb.set_trace()
        poll_df_reset = poll_df.reset_index()
        mask = poll_df_reset['GRN'] == 0  # only apply to GRN = 0 polls
        grn_nonzero = poll_df_reset.loc[~mask, 'GRN'] 

        if len(grn_nonzero) > 0: # try use other poll average
            grn_val = grn_nonzero.mean()
        else:
            denom = prior_df.loc[div, minor_cols].sum() + prior_df.loc[div, 'GRN'] # otherwise use prior proportion of OTH
            grn_val = poll_df.loc[div, 'OTH'] * (prior_df.loc[div, 'GRN'] / denom)

        poll_df_reset.loc[mask, 'GRN'] = grn_val
        poll_df_reset.loc[mask, 'OTH'] -= grn_val

        poll_df = poll_df_reset.set_index('Electorate')
    

    # reduce to CAGO categories, combine for poll average
    CAGO_rows = []
    for i in range(len(poll_df)):
        curr_row = poll_df.iloc[i:i+1,].iloc[:,2+('byelection_year' in poll_df.columns):]
        curr_row_normalised = curr_row/curr_row.sum(axis=1)[div]
        CAGO_rows.append(group_into_Fundamentals_Categories(curr_row_normalised, div)) # extra col for byelecitons
    CAGO_row = pd.concat(CAGO_rows).mean()


    # add alr error and return to prop space, accounting for missing major party
    if missing_party is not None:
        CAGO_row = CAGO_row.drop([missing_party])
    
    alr = np.log(CAGO_row.drop(ref_col) / CAGO_row[ref_col])
    alr_adj = alr + alr_error_row
    adjusted = alr_to_simplex_vectorized(alr_adj, ref_col)

    macro_oth = adjusted['Other'].iloc[0]

    # reallocate minor parties - first according to poll proportions, then for remaining according to prior proportions
    # 1. Polled party 'OTH' reallocation
    poll_minors = poll_df.reindex(columns = minor_cols+ ['OTH']).astype(float).fillna(0)
    poll_minor_means = poll_minors.replace(0, np.nan).mean() # mean shares of any minor parties polled
    poll_minor_split = poll_minor_means.dropna()
    poll_minor_split['OTH'] = CAGO_row['Other'] - poll_minor_split.loc[poll_minor_split.index !='OTH'].sum() # remaining OTH vote after minor polled party averages accounted for (safe when polled minors differ from poll to poll)
    poll_minor_split = poll_minor_split.loc[poll_minor_split.clip(lower = 0) > 0] # remove 'OTH' if OTH == 0
    poll_weights = poll_minor_split / poll_minor_split.sum()

    poll_alloc = pd.Series(0.0, index=minor_cols + ['OTH'])
    alpha_curr = poll_weights.values*dirichlet_alpha  # or scaled version if you want concentration
    draw = np.random.dirichlet(alpha_curr)
    poll_alloc.loc[poll_weights.index] = macro_oth * draw

    # 2. Prior reallocation for remaining minor parties - only if poll is incomplete
    remaining_oth = poll_alloc['OTH']
    unpolled_minor_cols = poll_alloc[poll_alloc == 0].index.tolist()


    if len(unpolled_minor_cols) > 0 and remaining_oth > 0:
        prior_minors = prior_df.loc[div, unpolled_minor_cols]
        prior_weights = prior_minors / prior_minors.sum()
        alpha_prior = prior_weights.values * dirichlet_alpha
        draw_prior = np.random.dirichlet(alpha_prior)

        poll_alloc.loc[prior_weights.index] = remaining_oth * draw_prior

        poll_alloc = poll_alloc.drop('OTH')

    assert np.isclose(poll_alloc.sum(), macro_oth) # oth vote share correctly distirbuted

    final_row = pd.concat([adjusted.drop(columns=['Other']),poll_alloc.to_frame().T], axis=1)

    return final_row



# combine by-election fundamentals
if not os.path.exists('Fundamentals_Votes_For_Byelection.csv'):
    byelection_fund_list = []
    for data_year in ['2013','2016','2019','2022']:
        byelection_prior = pd.read_csv(f'By_election_Fundamentals_after_{data_year}.csv')
        byelection_fund_list.append(byelection_prior)
    byelection_fundamentals = pd.concat(byelection_fund_list)
    byelection_fundamentals.to_csv("Fundamentals_Votes_For_Byelection.csv", index=False)


# currently for each election year
for election_year in ['Byelection','2016','2019','2022','2025']:
    prior_long = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv")

    # perform for each electorate with seat polling
    curr_seat_polls = Seat_poll_year_dict[election_year]
    for div in curr_seat_polls['Electorate'].unique():

        if prior_long.loc[prior_long['div_nm']==div,].empty: # e.g., Canberra 1995
            continue

        prior_row = prior_long.loc[prior_long['div_nm']==div,][['PartyAb','FP_Votes']].set_index('PartyAb').T.rename(index = {'FP_Votes':div})
        poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate']==div].set_index('Electorate')
        ref_col = 'COAL' # standard

        alr_error_row=pd.DataFrame([[0, 0, 0]], columns=['ALP','GRN','Other'])
        prior_CAGO =  group_into_Fundamentals_Categories(prior_row, div, is_Other = True)
        missing_party = next((p for p in ['COAL','ALP','GRN'] if prior_CAGO.loc[div, p] == 0),None)

        if election_year == 'Byelection' and missing_party is not None: # actually missing party

            if poll_df[missing_party].sum() == 0: # missing party not polled - Farrer 2026 will be the only exception
                new_error, new_cols, ref_col = transform_alr_draw(alr_error_row, missing_party)
                alr_error_row = pd.DataFrame([new_error], columns=new_cols)

            print(div)
            print(alr_error_row)
            import pdb; pdb.set_trace()

        final_row = process_division(poll_df,prior_row,div,alr_error_row=alr_error_row,missing_party=missing_party, ref_col=ref_col,dirichlet_alpha=22)
        
        assert np.isclose(final_row.sum(axis=1), 1)

        print(div)
        print(poll_df.iloc[:,2:].mean().to_frame().T)
        print(final_row, final_row.sum(axis=1))
        #import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()