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

election_date_num = {'2013':1113, '2016':1028, '2019':1050, '2022':1099}



election_years = ['2013','2016','2019','2022']



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




def get_results_dict(election_year):

    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})

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

        Four_party_results_list.append(group_into_Fundamentals_Categories(Actual_results_dict[div], div))

    results_df = pd.concat(Four_party_results_list)
    results_df = results_df.div(results_df.sum(axis=1), axis=0).rename(columns={'Other':'OTH'})

    #import pdb;pdb.set_trace()

    return results_df


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


# get all COAL-ALP-GRN first i.e. all for whom GRN is not 0 and compare!

plt.figure(figsize=(10, 6))
x_axis = 'Sample size'
CAGO_abs_diff_list = []

Polling_misses_ALR_list = []
for election_year in election_years:
    CAGO_poll_df = Seat_poll_year_dict[election_year].loc[Seat_poll_year_dict[election_year]['GRN']>0,]

    if not election_year == '2013': # already formatted in 2013

        # Want to reduce to COAL, ALP, GRN, and all the rest into Other
        info_party_cols = CAGO_poll_df.columns[:3].tolist() + ["COAL", "ALP", "GRN"]
        CAGO_poll_df["OTH"] = CAGO_poll_df.drop(columns=info_party_cols).sum(axis=1)     # Create the 'OTH' column by summing all other columns
        CAGO_poll_df = CAGO_poll_df[info_party_cols + ["OTH"]] # Keeps only COAL, ALP, GRN, OTH

    ref_col = 'COAL'
    CAGO_poll_df = CAGO_poll_df[(CAGO_poll_df != 0).all(axis=1)].set_index('Electorate')
    CAGO_poll_ALR = np.log(CAGO_poll_df.iloc[:,-4:].drop(columns=[ref_col]).div(CAGO_poll_df.iloc[:,-4:][ref_col], axis=0))

    Actual_results_polled_electorates = Actual_results_dict_year[election_year].loc[CAGO_poll_df.index]
    Actual_results_polled_electorates_ALR = np.log(Actual_results_polled_electorates.drop(columns=[ref_col]).div(Actual_results_polled_electorates[ref_col], axis=0))

    Polling_misses = Actual_results_polled_electorates_ALR - CAGO_poll_ALR
    Polling_misses.loc[:,'Days before election'] = CAGO_poll_df[['Days before election']]
    Polling_misses.loc[:,'Election_year'] = election_year

    Polling_misses_ALR_list.append(Polling_misses)

    # Compute absolute differences and store them in poll_df
    for party in ["COAL", "ALP", "GRN", "OTH"]:
        CAGO_poll_df[f"{party}_abs_diff"] = (CAGO_poll_df[party] - Actual_results_dict_year[election_year].loc[CAGO_poll_df.index].values[:, Actual_results_dict_year[election_year].columns.get_loc(party)]).abs()

    CAGO_abs_diff_list.append(CAGO_poll_df)

    plot_df = CAGO_poll_df.melt(id_vars=[x_axis], 
                        value_vars=["COAL_abs_diff", "ALP_abs_diff", "GRN_abs_diff", "OTH_abs_diff"], 
                        var_name="PartyAb", 
                        value_name="Abs Difference")

    # Map party names to colors
    party_colors = {"COAL": "blue", "ALP": "red", "GRN": "green", "OTH": "gray"}

    # Extract party names for coloring
    plot_df["Party"] = plot_df["PartyAb"].str.replace("_abs_diff", "")  # Remove suffix
    plot_df["Color"] = plot_df["Party"].map(party_colors)


    # Loop through each party and plot separately
    markers = {'2013':'o','2016':'s','2019':'^','2022':'x'}
    for party, color in party_colors.items():
        subset = plot_df[plot_df["Party"] == party]
        plt.scatter(subset[x_axis], subset["Abs Difference"], label=party, color=color, marker = markers[election_year], s = 10, alpha=0.7)

CAGO_abs_diff = pd.concat(CAGO_abs_diff_list)

plt.xlabel(x_axis)
plt.ylabel("Absolute Difference")
plt.title("Absolute Difference Between Polls and Results")
plt.legend()
plt.show()


Polling_misses_ALR_df = pd.concat(Polling_misses_ALR_list)

Polling_misses_ALR_df.iloc[:,:3] = Polling_misses_ALR_df.iloc[:,:3] - Polling_misses_ALR_df.iloc[:,:3].mean()

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

    return Prior_estimates_df, Prior_estimates_dict


election_year = '2019'
Prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0]

Seat_Poll_curr_df = Seat_poll_year_dict[election_year].set_index('Electorate')

if not election_year == '2013': # already formatted in 2013

    main_party_list = ["COAL", "ALP", "GRN",'ON','UAPP','IND'] if election_year != '2025' else ["COAL", "ALP", "GRN",'ON','TOP','IND']

    # Want to reduce to COAL, ALP, GRN, and all the rest into Other
    info_party_cols = Seat_Poll_curr_df.columns[:2].tolist() + main_party_list
    Seat_Poll_curr_df["OTH"] = Seat_Poll_curr_df.drop(columns=info_party_cols).sum(axis=1)     # Create the 'OTH' column by summing all other columns
    Seat_Poll_curr_df = Seat_Poll_curr_df[info_party_cols + ["OTH"]]




def reallocate_oth(poll_df, prior_df):

    # fills in zeros for polls where they missed a candidate

    updated_poll = poll_df.copy()
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


Prior_estimates_df = Prior_estimates_df.rename(columns={'Other':'OTH'})

Reallocated_polls = reallocate_oth(Seat_Poll_curr_df.iloc[:,2:].drop('IND', axis=1), Prior_estimates_df)
Reallocated_polls.loc[:,'OTH'] += Seat_Poll_curr_df['IND']

# replace non-contesting places with small value i.e. 0.001, in both poll and prior values - this means only shift that is determined is between COAL in prior & poll
SMALL_VALUE_FOR_NON_RUNNNING = 0.001
Reallocated_polls = Reallocated_polls.replace(0.0,SMALL_VALUE_FOR_NON_RUNNNING)
DIVS_WITH_NO_ON = Prior_estimates_df[Prior_estimates_df['ON'] == 0].index
Prior_estimates_Reallocated = Prior_estimates_df.copy().replace(0.0,SMALL_VALUE_FOR_NON_RUNNNING)

Reallocated_polls_ALR = np.log(Reallocated_polls.drop(columns=[ref_col]).div(Reallocated_polls[ref_col], axis=0))
Prior_estimates_Reallocated_ALR = np.log(Prior_estimates_Reallocated.drop(columns=[ref_col]).div(Prior_estimates_Reallocated[ref_col], axis=0))

Swing_ALR = Reallocated_polls_ALR - Prior_estimates_Reallocated_ALR.loc[Reallocated_polls_ALR.index]

# remove ON_boosts where no ON finally
import pdb;pdb.set_trace()




