import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict




import seaborn as sns
import scipy.stats as stats

from pingouin import multivariate_normality

from collections import Counter
from itertools import groupby

import plotly.graph_objects as go
from urllib.parse import quote



# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Necessary CSV Files"
os.chdir(base_dir)

SAMPLE_ERROR_SCALING_FACTOR = 2 # assume polls have double their theoretical error 
ELECTION_DAYS_SINCE_PREV = {'2007': 1141,'2010':1002, '2013':1113, '2016':1028, '2019':1050, '2022':1099,'2025':1078}




_CSV_CACHE = {}

_real_read_csv = pd.read_csv
_real_to_csv = pd.DataFrame.to_csv

def _cached_read_csv(path, *args, **kwargs):
    key = str(path)
    if key in _CSV_CACHE:
        return _CSV_CACHE[key].copy()
    df = _real_read_csv(path, *args, **kwargs)
    _CSV_CACHE[key] = df
    return df.copy()

def _cached_to_csv(self, path, *args, **kwargs):
    key = str(path)
    _CSV_CACHE[key] = self.copy()
    # do NOTHING else (no disk write)

pd.read_csv = _cached_read_csv
pd.DataFrame.to_csv = _cached_to_csv



def plot_GRW(x_posterior, day_of_interest):

    # Compute summary statistics
    x_mean = np.mean(x_posterior, axis=(0, 1))  # Mean trajectory (T, K)
    x_lower = np.percentile(x_posterior, 2.5, axis=(0, 1))  # 2.5th percentile (T, K)
    x_upper = np.percentile(x_posterior, 97.5, axis=(0, 1))  # 97.


    # Time axis (100 days)
    time = np.arange(day_of_interest+1)


    plt.figure(figsize=(10, 6))

    plt.plot(time, x_mean)
    plt.fill_between(time, x_lower, x_upper, alpha=0.2)

    plt.xlabel("Days")
    plt.ylabel("Vote Share")
    plt.title("Vote Share Trajectory with 95% Credible Intervals")
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
    plt.ylabel("Smoothed Vote Share")
    plt.title("Smoothed Vote Share Trajectory with 95% Credible Intervals")
    plt.legend()
    plt.grid()
    plt.show()

    return 1

def simulate_polling_average_GRW(election_year, num_polling_days, days_to_election, produce_plot = False):

    # only do inference for last x days before election
    starting_point = ELECTION_DAYS_SINCE_PREV[election_year] - num_polling_days # start 100 days before last day of polling

    # initialise polling everage df
    if election_year == '2025':
        Polling_avg_parties = ['COAL','ALP','GRN','ON','TOP','OTH']
    elif election_year in ['2019','2022']:
        Polling_avg_parties = ['COAL','ALP','GRN','ON','UAPP','OTH']
    else:
        Polling_avg_parties = ['COAL','ALP','GRN','OTH']

    day_of_interest = num_polling_days - days_to_election
    current_polling_avg =  pd.DataFrame([[0.0]*len(Polling_avg_parties)], columns=Polling_avg_parties)



    National_polls = pd.read_csv(f'NationalPollsforMGRW{election_year}.csv')

    sigma_drift_prior = {'COAL':0.004,'ALP':0.004,'GRN':0.002,'OTH':0.003,'ON':0.003,'UAPP':0.003,'TOP':0.003} # larger relative uncertainty for minor parties ON/UAPP/TOP


    for party in National_polls.columns[2:]:

        if party == 'OTH': # OTH estimate inferred from remaining averages
            continue 


        # format party's polling data into df with columns Day_index, Sample_size, PartyAb, precision
        df = National_polls[['Days since last election','Sample size',party]] 

        # exclude 0 poll values (i.e., UAPP), deal with lack of polling data long before the election
        df = df.loc[df[party]>0,]

        if party == 'UAPP':
            prior_poll_avg = df.loc[df['Days since last election'] < starting_point,].iloc[:,2:].mean() if election_year == '2022' else 0.04 # for 2019
        elif party == 'TOP':
            prior_poll_avg = 0.015 # mix of 1% and 2%
        else:
            prior_poll_avg = df.loc[df['Days since last election'] < starting_point,].iloc[-10:,2:].mean()


        df = df.loc[df['Days since last election'] >= starting_point,]
        df.loc[:,'Days since last election'] -= starting_point
        df = df.rename(columns={'Days since last election':'Day_index'})

        # only model the polling until day_of_interest
        df = df.loc[df['Day_index']<=day_of_interest,]

        days = df['Day_index'].values
        observed_days = np.array(sorted(set(days))) 

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

            init_dist = pm.Normal.dist(mu=prior_poll_avg, sigma=0.02) 
            sigma = pm.TruncatedNormal("sigma", mu=0.005, sigma=sigma_drift_prior[party], lower=1e-5)  # Prior for daily drift of random walk

            # Latent vote share following a Gaussian random walk
            vote_trend = pm.GaussianRandomWalk("vote_trend", sigma=sigma, shape=day_of_interest+1, init_dist=init_dist)

            # Observed polls (Normal likelihood with poll-dependent variance)

            observed = pm.Normal("observed", mu=vote_trend[agg_polls["Day_index"].values], sigma=SAMPLE_ERROR_SCALING_FACTOR*agg_polls["poll_sd"], observed=agg_polls["vote_share_weighted"])

            trace = pm.sample(2000, tune=4000, chains=4, cores=4, target_accept=0.99)


        x_posterior = trace.posterior["vote_trend"].values
        x_mean = np.mean(x_posterior, axis=(0, 1))  # Mean vote share over time
        current_polling_avg[party] = x_mean[day_of_interest]
        print(current_polling_avg)

        print(party, election_year, "estimated_sigma", az.summary(trace, var_names=["sigma"]))

        if produce_plot:
            plot_GRW(x_posterior, day_of_interest)

    # very important to ensure that Others is imputed off correct distribution of minor parties (for 2025 issues where ON/TOP are often blank!)
    current_polling_avg['OTH'] = 1 - current_polling_avg.sum(axis=1)[0]
    
    return current_polling_avg




election_year = '2025'
num_polling_days = 100 # length of interval for Gaussian Random Walk
days_to_election = 2 # final model - 2 days before

# save 2025 final polling average
#simulate_polling_average_GRW(election_year, num_polling_days, days_to_election, produce_plot = False).to_csv(f"2025_National_Day_{num_polling_days-days_to_election}_Poll_Average.csv")


# generate polling averages for 2016-2025
#days_to_election = 10
#poll_avg_list = []

#for election_year in ['2016','2019','2022']:
#    current_polling_avg = simulate_polling_average_GRW(election_year, num_polling_days, days_to_election, produce_plot = False)
#    current_polling_avg['Election'] = election_year
#    poll_avg_list.append(current_polling_avg)

#National_day_x_polls = pd.concat(poll_avg_list)



#import pdb;pdb.set_trace()


########################################################################## ESTIMATE NATIONAL ALR COVARIANCE MATRICES ###################################################################



def extend_corr_matrix_to_5x5(corr_matrix_3x3, var4, var5, cov_matrix_3x3, ref_col):
    # Extends to 5x5 corr and covMs through imputation, checking for positive definiteness and adding column names, returning 5x5 df

    # expands to 5x5
    R5 = np.pad(corr_matrix_3x3, ((0, 2), (0, 2)), mode='constant', constant_values=0)

    ON_UAPP_CORRELATION = 0.5
    ALP_GRN_CORRELATION_LOSS = 0.3
    COAL_CORRELATION_LOSS = 0.1

    if ref_col == 'COAL':



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

    elif ref_col == 'ALP':
        # ON/ALP; UAPP/ALP
        R5[0,3] = R5[0,2] - COAL_CORRELATION_LOSS
        R5[3,0] = R5[0,2] - COAL_CORRELATION_LOSS
        R5[0,4] = R5[0,2] - COAL_CORRELATION_LOSS
        R5[4,0] = R5[0,2] - COAL_CORRELATION_LOSS

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


def estimate_National_ALR_Covariance_Matrices(ref_col = 'COAL', Day = 90, plot_histogram = False):

    # Inputs alr reference column (default Coalition) and Day = {100 - days until election}
    # Uses historical federal and state polling data to estimate ALR-Covariance between main parties after applying reference column
    # Updates relevant only for updates in "NationalElectionPollingAveragesGRW_Day_{Day}"

    for curr_election_year in ['2016','2019','2022','2025']:

        election_year_to_remove = curr_election_year if REMOVE_ELECTION_YEAR else ' '


        for Type in ['Polling','Election_swing']:

            if Type == 'Polling':
                # 1. Correlation estimate of 3x3 - 6 GRW 2007-2022 + 14 State Elections + 15 State Results
                

                FederalStatePolls = pd.read_csv("StatePollingWeightedAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
                StateElectionsPolls = pd.read_csv("StateElectionsWeightedPollingAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
                OldFederalElectionPollingAverage = pd.read_csv("OldFederalElectionPollingAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
                NationalElectionPollingAveragesGRW = pd.read_csv(f"NationalElectionPollingAveragesGRW_Day_{Day}.csv", index_col = None).set_index('Election')


                FederalStateResults =  pd.read_csv("StateFederalResults.csv", index_col = None).set_index('Election')
                StateElectionResults = pd.read_csv("StateElectionResults.csv", index_col = None).set_index('Election')
                OldFederalElectionResults = pd.read_csv("OldFederalElectionResults.csv", index_col = None).set_index('Election')
                NationalElectionResults = pd.read_csv("NationalElectionResults.csv", index_col = None).set_index('Election')

                NationalElectionPollingAveragesGRW.index = NationalElectionPollingAveragesGRW.index.astype(str)
                NationalElectionResults.index = NationalElectionResults.index.astype(str)
                OldFederalElectionPollingAverage.index = OldFederalElectionPollingAverage.index.astype(str)
                OldFederalElectionResults.index = OldFederalElectionResults.index.astype(str)

                #Arbitrarily selected State per election
                Selected_states = ['2022NSW','2019VIC','2016QLD']
                GRN_OldFederalPolls = OldFederalElectionPollingAverage.rename(columns={'DEM<=1996/GRN':'GRN'})
                GRN_OldFederalResults = OldFederalElectionResults.rename(columns={'DEM<=1996/GRN':'GRN'})


                CAGO_Polling_Avg = pd.concat([FederalStatePolls.iloc[FederalStatePolls.index.isin(Selected_states),], StateElectionsPolls, NationalElectionPollingAveragesGRW, GRN_OldFederalPolls.loc[GRN_OldFederalPolls.index.isin(['2001','2004']),]], ignore_index = False)
                CAGO_Results = pd.concat([FederalStateResults.iloc[FederalStateResults.index.isin(Selected_states),], StateElectionResults, NationalElectionResults, GRN_OldFederalResults.loc[GRN_OldFederalResults.index.isin(['2001','2004']),]], ignore_index = False)

                CAGO_Polling_ALR = np.log(CAGO_Polling_Avg.drop(columns=[ref_col]).div(CAGO_Polling_Avg[ref_col], axis=0))
                CAGO_Results_ALR = np.log(CAGO_Results.drop(columns=[ref_col]).div(CAGO_Results[ref_col], axis=0))

                CAGO_ALR_swings = CAGO_Results_ALR - CAGO_Polling_ALR
                # correct for past polling bias!
                CAGO_ALR_swings_centered = CAGO_ALR_swings - CAGO_ALR_swings.mean()

                if REMOVE_ELECTION_YEAR:
                    CAGO_ALR_swings_centered = CAGO_ALR_swings_centered.loc[~(CAGO_ALR_swings_centered.index.str.startswith(election_year_to_remove)),]

                corr_matrix = np.corrcoef(CAGO_ALR_swings_centered.values, rowvar=False)



                # VARIANCE ESTIMATION


                CAGO_variance_estimation_polls = pd.concat([CAGO_Polling_Avg,GRN_OldFederalPolls.iloc[:-2]])
                CAGO_variance_estimation_results = pd.concat([CAGO_Results,GRN_OldFederalResults.iloc[:-2]])

                # extract GRN from before 2004 - very different party vote share today to back then!
                CAGO_variance_estimation_polls.loc[CAGO_variance_estimation_polls.index.isin(['1987','1990','1993','1996','1998','2001']),'GRN'] = np.nan
                CAGO_variance_estimation_results.loc[CAGO_variance_estimation_results.index.isin(['1987','1990','1993','1996','1998','2001']),'GRN'] = np.nan


                CAGO_Variance_Polling_ALR = np.log(CAGO_variance_estimation_polls.drop(columns=[ref_col]).div(CAGO_variance_estimation_polls[ref_col], axis=0))
                CAGO_Variance_Results_ALR = np.log(CAGO_variance_estimation_results.drop(columns=[ref_col]).div(CAGO_variance_estimation_results[ref_col], axis=0))



                CAGO_Variance_estimation_swings = CAGO_Variance_Results_ALR - CAGO_Variance_Polling_ALR
                CAGO_Variance_estimation_swings_centered = CAGO_Variance_estimation_swings - CAGO_Variance_estimation_swings.mean()


                # Suggested weighting scheme of polling data - 3 Federal states = 0.3, State election = 0.5, Recent Federal election = 1, 2001/2004 federal: 0.6, Pre-2000 federal: 0.5
                weights = np.array([0.3,0.3,0.3,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,0.6,0.6,0.5,0.5,0.5,0.5,0.5])

                # remove data from current election
                if election_year_to_remove and election_year_to_remove != '2025':
                    if election_year_to_remove == '2016':
                        indices = [4,5,14,15,20,21,22,23,24,25,26,27,28,29]
                    if election_year_to_remove == '2019':
                        indices = [2,3,4,5,7,9,10,14,15,16,19,20,21,22,23,24,25,26,27,28,29]
                    if election_year_to_remove == '2022':
                        indices = [0,2,3,4,5,7,9,10,12,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29]

                    weights = weights[indices]
                    CAGO_Variance_estimation_swings_centered = CAGO_Variance_estimation_swings_centered.iloc[indices]



            elif Type == 'Election_swing':
                Election_swings_df = pd.read_csv("ElectionSwings.csv", index_col=None).set_index("Election")
                Election_results_curr = Election_swings_df.iloc[:,:4]
                Election_results_prev = Election_swings_df.iloc[:,4:]
                Election_results_curr_ALR = np.log(Election_results_curr.drop(columns=[ref_col]).div(Election_results_curr[ref_col], axis=0))
                Election_results_prev_ALR = np.log(Election_results_prev.drop(columns=[ref_col+'_prev']).div(Election_results_prev[ref_col+'_prev'], axis=0))
                Election_swings_ALR = Election_results_curr_ALR - Election_results_prev_ALR.values

                Election_swings_ALR_centered = Election_swings_ALR - Election_swings_ALR.mean()
                Election_swings_ALR_centered_after_1996 = Election_swings_ALR_centered.loc[~(Election_swings_ALR_centered.index.isin(['1987','1990','1993','1996'])),]


                if REMOVE_ELECTION_YEAR:
                    CAGO_ALR_swings_centered = Election_swings_ALR_centered_after_1996.loc[~(Election_swings_ALR_centered_after_1996.index.str.startswith(election_year_to_remove)),]

                corr_matrix_Elec = np.corrcoef(Election_swings_ALR_centered_after_1996.values, rowvar=False)
                #print(corr_matrix_Elec)

                Election_swings_ALR_for_Var = Election_swings_ALR.copy()
                Election_swings_ALR_for_Var.loc[Election_swings_ALR_for_Var.index.isin(['1987','1990','1993','1996','1998','2001']),'GRN'] = np.nan
                Election_swings_ALR_for_Var_centered = Election_swings_ALR_for_Var - Election_swings_ALR_for_Var.mean()

                # 1 to post-2004 elections, 0.5 to state elections, 0.6 to 2004/2001, 0.5 to 20th century elecs
                weights = np.array([1,1,1,1,1,1,0.6,0.6] + [0.5]*24)

                if election_year_to_remove and election_year_to_remove != '2025':
                    if election_year_to_remove == '2016':
                        indices = [3,4,5,6,7,8,9,10,11,12,15,18,19,22,23,27]
                    if election_year_to_remove == '2019':
                        indices = [2,3,4,5,6,7,8,9,10,11,12,14,15,17,18,19,21,22,23,26,27,29]
                    if election_year_to_remove == '2022':
                        indices = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,18,19,21,22,23,25,26,27,29,31]

                    weights = weights[indices]
                    Election_swings_ALR_for_Var_centered = Election_swings_ALR_for_Var_centered.iloc[indices]



            CAGO_Variance_estimation_swings_centered = CAGO_Variance_estimation_swings_centered if Type == 'Polling' else Election_swings_ALR_for_Var_centered


            def weighted_nanstd(data, weights):
                # estimates variances while respecting np.nan for GRN before 2004
                weighted_std = []

                for i in range(data.shape[1]):
                    col = data[:, i]
                    mask = ~np.isnan(col)
                    w = weights[mask]
                    x = col[mask]
                    w /= w.sum()  # normalize weights
                    mean = np.sum(w * x)
                    var = np.sum(w * (x - mean)**2)
                    weighted_std.append(np.sqrt(var))
                return np.array(weighted_std)

            weighted_std = weighted_nanstd(CAGO_Variance_estimation_swings_centered.values, weights)

            correlation_matrix = corr_matrix if Type == 'Polling' else corr_matrix_Elec

            cov_matrix = np.outer(weighted_std, weighted_std) * correlation_matrix
        


            if REMOVE_ELECTION_YEAR:

                Election_swing_ON_UAPP_ALR_variances = {'2016':0.0751,'2019':0.0904,'2022':0.0901,'2025':0.0944}
                Polling_swing_ON_UAPP_ALR_variances = {'2016':0.0136,'2019':0.0139,'2022':0.0309,'2025':0.0343}

                CAGO_cols = [p for p in ['COAL','ALP','GRN','OTH'] if p != ref_col]


                if Type == 'Polling':
                    ON_UAPP_variances = Polling_swing_ON_UAPP_ALR_variances[curr_election_year]

                    if curr_election_year == '2016':
                        CAGO_Polling_2016_corr, CAGO_Polling_2016_cov = corr_matrix,cov_matrix
                        CAGO_Polling_2016_cov = pd.DataFrame(CAGO_Polling_2016_cov, index = CAGO_cols, columns = CAGO_cols)
                        CAGO_Polling_2016_cov.to_csv(f"PollingErrorALRCovarianceNational2016_Day_{Day}_{ref_col}.csv", index = True)

                    elif curr_election_year == '2019':
                        CAGO_Polling_2019_corr, CAGO_Polling_2019_cov = extend_corr_matrix_to_5x5(corr_matrix, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                        CAGO_Polling_2019_cov.to_csv(f"PollingErrorALRCovarianceNational2019_Day_{Day}_{ref_col}.csv", index = True)


                    elif curr_election_year == '2022':
                        CAGO_Polling_2022_corr, CAGO_Polling_2022_cov = extend_corr_matrix_to_5x5(corr_matrix, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                        CAGO_Polling_2022_cov.to_csv(f"PollingErrorALRCovarianceNational2022_Day_{Day}_{ref_col}.csv", index = True)


                    elif curr_election_year == '2025':
                        CAGO_Polling_2025_corr, CAGO_Polling_2025_cov = extend_corr_matrix_to_5x5(corr_matrix, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                        CAGO_Polling_2025_cov.to_csv(f"PollingErrorALRCovarianceNational2025_Day_{Day}_{ref_col}.csv", index = True)

                elif Type == 'Election_swing':
                    ON_UAPP_variances = Election_swing_ON_UAPP_ALR_variances[curr_election_year]


                    if curr_election_year == '2016':
                        CAGO_swing_2016_corr, CAGO_swing_2016_cov = corr_matrix,cov_matrix
                        CAGO_swing_2016_cov = pd.DataFrame(CAGO_swing_2016_cov, index = CAGO_cols, columns = CAGO_cols)
                        CAGO_swing_2016_cov.to_csv(f"ElectionErrorALRCovarianceNational2016_{ref_col}.csv", index = True)


                    elif curr_election_year == '2019':
                        CAGO_swing_2019_corr, CAGO_swing_2019_cov = extend_corr_matrix_to_5x5(corr_matrix_Elec, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                        CAGO_swing_2019_cov.to_csv(f"ElectionErrorALRCovarianceNational2019_{ref_col}.csv", index = True)


                    elif curr_election_year == '2022':
                        CAGO_swing_2022_corr, CAGO_swing_2022_cov = extend_corr_matrix_to_5x5(corr_matrix_Elec, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                        CAGO_swing_2022_cov.to_csv(f"ElectionErrorALRCovarianceNational2022_{ref_col}.csv", index = True)


                    elif curr_election_year == '2025':
                        CAGO_swing_2025_corr, CAGO_swing_2025_cov = extend_corr_matrix_to_5x5(corr_matrix_Elec, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                        CAGO_swing_2025_cov.to_csv(f"ElectionErrorALRCovarianceNational2025_{ref_col}.csv", index = True)


# manage removal of data for given election_year
REMOVE_ELECTION_YEAR = 1
NO_OF_STATES = 8
NO_OF_ELECTORATES = {'2016':150,'2019':151,'2022':151,'2025':150}
DIM_OF_COV_MATRIX = {'2016':3,'2019':5,'2022':5,'2025':5}

STATES = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']
STATE_TO_INDEX = {state: i for i, state in enumerate(STATES)}

COAL_PARTIES = {"LNP", "LP", "NP", "CLP"}
ALP_NAME = "ALP"
IND_PARTIES = {'IND1','IND2','IND3','IND4','IND5'}

TCP_COMBINATION_INDEX = {('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, ('COAL','Right'):6,  \
                                ('LP','NP'): 7, ('IND','IND'): 8, ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14}

Day = 90 # currently acts as global - in future rewrite to make Day fluid


def estimate_state_polling_deviations_from_national(ref_col = 'COAL',Type = 'Election_swing', Day = Day, remove_election_year = 0):

    # inputs party as ref_col and 'Type' as either 'Election_swing' or 'Polling'
    # remove_election_year = 0 due to only 3 years of data - minimal, but necessary, leakage likely for 2016-2022 to get a minimum viable sample size of polling deviations

    # impute results about ON/UAPP form relative general polling error
    National_Polling_covm_2019 = pd.read_csv(f"PollingErrorALRCovarianceNational2019_Day_{Day}_{ref_col}.csv", index_col=0)
    National_Polling_covm_2022 = pd.read_csv(f"PollingErrorALRCovarianceNational2022_Day_{Day}_{ref_col}.csv", index_col=0)
    National_Polling_covm_2025 = pd.read_csv(f"PollingErrorALRCovarianceNational2025_Day_{Day}_{ref_col}.csv", index_col=0)
    National_Election_covm_2019 = pd.read_csv(f"ElectionErrorALRCovarianceNational2019_{ref_col}.csv", index_col=0)
    National_Election_covm_2022 = pd.read_csv(f"ElectionErrorALRCovarianceNational2022_{ref_col}.csv", index_col=0)
    National_Election_covm_2025 = pd.read_csv(f"ElectionErrorALRCovarianceNational2025_{ref_col}.csv", index_col=0)

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

        
            curr_year_Election_swing_covM.to_csv(f'ElectionErrorALRCovarianceStateDeviation{curr_election_year}.csv', index = True)





    elif Type == 'Polling':

        State_Polling = pd.read_csv("StatePollingWeightedAverage_rel_to_Nat.csv", index_col=None)
        State_Results = pd.read_csv("StateFederalResults_including_ON.csv", index_col=None)

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

        #print(National_poll_ALR.iloc[:,2:] - National_result_ALR.iloc[:,2:])

        #import pdb;pdb.set_trace()





        # get state - national, for each of Poll and results
        State_NAT_poll_merged = pd.merge(State_poll_ALR, National_poll_ALR, on='Year', how='left', suffixes=('','_Nat'))
        State_NAT_result_merged = pd.merge(State_result_ALR, National_result_ALR, on='Year', how='left', suffixes=('','_Nat'))

        # get Nat - State differences
        alr_cols = [col for col in ['COAL','ALP','GRN','OTH'] if col != ref_col]
        for col in alr_cols:
            State_NAT_poll_merged[col + '_rel_to_Nat'] = State_NAT_poll_merged[col] - State_NAT_poll_merged[col + '_Nat']
            State_NAT_result_merged[col + '_rel_to_Nat'] = State_NAT_result_merged[col] - State_NAT_result_merged[col + '_Nat']

        

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
            election_year_to_remove = curr_election_year if remove_election_year else ' ' # disabled at the moment

            # remove data from current election
            State_NAT_result_merged_curr = State_NAT_result_merged.loc[State_NAT_result_merged['Year']!=curr_election_year,].iloc[:,-3:].reset_index(drop=True)
            State_NAT_poll_merged_curr = State_NAT_poll_merged.loc[State_NAT_poll_merged['Year']!=curr_election_year,].iloc[:,-3:].reset_index(drop=True)


            means_df_curr = means_df.loc[means_df['Year']!= curr_election_year,].drop('Year', axis = 1)
            std_weights_df_curr = std_weights_df.loc[std_weights_df['Year']!= curr_election_year,].drop('Year', axis = 1)
            std_weights_array = 1/std_weights_df_curr.values

            # convert precisions to ALR, using formula 1 / (prec_alp * mu_alp**2) + 1 / (prec_coal * mu_coal**2) - Delta Method approximation (for simplicity, use independence)
            precision_weights = vectorized_log_ratio_precision(means_df_curr, std_weights_array, ref_col='COAL', numerator_list=['ALP', 'GRN', 'OTH'], rho=0.0)

            if curr_election_year == '2025':
                Relative_state_polling_precision = State_NAT_poll_merged.iloc[:,:2].reset_index(drop=True)
                Relative_state_polling_precision.loc[:,"Mean_precision"] = precision_weights.mean(axis = 1).values

                # now, 2025 values:
                State_stds_2025 = pd.read_csv(f"2025StatePollingWeightedAverage_rel_to_Nat_Day_{Day}.csv", index_col = None).iloc[:-1,]
                State_stds_2025.loc[:,'Year'] = State_stds_2025['Election'].str.extract(r'(\d{4})')
                State_stds_2025.loc[:,'Scope'] = State_stds_2025['Election'].str.extract(r'(\D+)$')[0]
                #State_stds_2025.loc[:,'OTH'] = State_stds_2025.loc[:,'OTH'] + State_stds_2025.loc[:,'ON']
                #State_stds_2025 = State_stds_2025.drop(['ON','ON_stds'], axis = 1)
                std_weights_array_2025 = 1/State_stds_2025.iloc[:,-7:-2].to_numpy()

                precision_weights_2025 = vectorized_log_ratio_precision(State_stds_2025.iloc[:,:5], std_weights_array_2025, ref_col='COAL', numerator_indices=[1,2,3,4], \
                                                                numerator_list=['ALP', 'GRN', 'ON','OTH'], rho=0.0)
                
                State_stds_2025_states = State_stds_2025.iloc[:,-2:]
                State_stds_2025_states.loc[:,"Mean_precision"] = precision_weights_2025.mean(axis=1).values

                Average_precisions = pd.concat([Relative_state_polling_precision, State_stds_2025_states], ignore_index = True)
                Average_precisions = Average_precisions.loc[~((Average_precisions['Year'] == '2016') & (Average_precisions['Scope'] == 'TAS')),]

                # add other states
                missing_states = ['TAS','NT','ACT']

                for year in ['2016','2019','2022','2025']:
                    grouped_by_year  = Average_precisions.loc[Average_precisions['Year'] == year,]
                    grouped_by_year = grouped_by_year.loc[grouped_by_year['Scope'] != 'TAS',]
                    min_precision = grouped_by_year['Mean_precision'].min()
                    #import pdb;pdb.set_trace()

                    for missing_state in missing_states:
                        Average_precisions = pd.concat([Average_precisions,pd.DataFrame([[year, missing_state, min_precision / 2]], columns = ['Year', 'Scope',"Mean_precision"])])


                Average_precisions = Average_precisions.sort_values(by = ['Year','Scope']).reset_index(drop=True)

                Average_precisions.to_csv("State_Polling_Average_Precisions.csv", index = False)

                Precisions_by_year = Average_precisions.groupby('Year')['Mean_precision'].agg('mean')/Average_precisions.groupby('Year')['Mean_precision'].agg('mean').sum()
                Precisions_by_year_weighting = ((Precisions_by_year/0.25)**1.5).to_frame() # divide by mean, intermediate of linear and quadratic

                Precisions_by_year_weighting.to_csv("State_Polling_Precisions_by_Year.csv", index = True)



                
            #import pdb;pdb.set_trace()

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

            #import pdb;pdb.set_trace()

            curr_year_Polling_swing_covM.to_csv(f'PollingErrorALRCovarianceStateDeviation{curr_election_year}.csv', index = True)




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


def impute_missing_polling(missing_states, State_Correlation_Matrix, State_Polling_CAGO_ALR_swing, National_State_ALR_df_last):
    """
    Impute the missing polling in ALR space based on correlations with other states.
    
    missing_states: list of states for which polling needs to be imputed.
    State_Correlation_Matrix: The state-to-state correlation matrix.
    State_Polling_CAGO_ALR_swing: The DataFrame containing available state polling swings in ALR
    National_State_ALR_df_last: The DataFrame containing the previous election's state vote proportions in ALR
    
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

    imputed_polling_df = pd.concat(imputed_pollings)

    
    return imputed_polling_df



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



def get_imputed_state_deviations_from_national(ref_col = 'COAL'):

    for election_year in ['2016','2019','2022','2025']:

        if election_year != '2025':

            # use global state polling average file

            StatePolling = pd.read_csv("StatePollingWeightedAverage_rel_to_Nat.csv", index_col=None)

            StatePolling.loc[:,'Election_year'] = StatePolling['Election'].str.extract(r'(\d{4})')[0]
            StatePolling.loc[:,'State'] = StatePolling['Election'].str.extract(r'\d{4}([A-Z]+)')[0]
            StatePolling = StatePolling.drop('Election', axis = 1)

            State_Polling_combined = StatePolling.copy()

            State_Polling_combined.loc[State_Polling_combined['ON'].isna(),'ON'] = 0.0
            State_Polling_combined.loc[:,'OTH'] += State_Polling_combined.loc[:,'ON'] # combine ON with Others
            State_Polling_CAGO = State_Polling_combined.drop(['ON','ON_stds'], axis = 1)

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

                


                missing_states = {'2016': ['NT','ACT'],'2019':['NT','ACT','TAS'],'2022':['NT','ACT','TAS'],'2025':['NT','ACT','TAS']}

                imputed_polling_ALR = impute_missing_polling(missing_states[election_year], State_Correlation_Matrix, State_Polling_CAGO_ALR_swing, National_State_ALR_df_last)
                existing_polling_ALR = State_Polling_CAGO_ALR_curr.set_index('State').iloc[:,:3]
                combined_polling_ALR = pd.concat([existing_polling_ALR,imputed_polling_ALR])

                imputed_state_polling = alr_to_simplex_vectorized(imputed_polling_ALR, 'COAL').reset_index(names='State')
                imputed_state_polling.loc[:,'Election_year'] = election_year

                StatePolling = pd.concat((StatePolling, imputed_state_polling), ignore_index=True)

            StatePolling.to_csv("Full_State_Inputed_Polling.csv", index=False)

            State_Polling_Deviations_from_National = get_ALR_deviations_from_National(StatePolling, ref_col, include_ON = 1)[0]
            State_Polling_Deviations_from_National.iloc[:,:4] = State_Polling_Deviations_from_National.iloc[:,-4:].values
            State_Polling_Deviations_from_National = State_Polling_Deviations_from_National.iloc[:,:6].sort_values(by=['Election_year','State'])
            State_Polling_Deviations_from_National.loc[:,'Election_year'] = State_Polling_Deviations_from_National.loc[:,'Election_year'].astype(str)

            State_Polling_Deviations_from_National.to_csv("State_Polling_Deviations_from_National.csv", index=False)




        else:

            StatePolling = pd.read_csv(f"2025StatePollingWeightedAverage_rel_to_Nat_Day_{Day}.csv", index_col=None)

            StatePolling.loc[:,'Election_year'] = StatePolling['Election'].str.extract(r'(\d{4})')[0]
            StatePolling.loc[:,'State'] = StatePolling['Election'].str.extract(r'\d{4}([A-Z]+)')[0]
            StatePolling = StatePolling.drop('Election', axis = 1)


            State_Polling_combined = StatePolling.copy()

            State_Polling_combined.loc[State_Polling_combined['ON'].isna(),'ON'] = 0.0
            State_Polling_combined.loc[:,'OTH'] += State_Polling_combined.loc[:,'ON'] # combine ON with Others
            State_Polling_CAGO = State_Polling_combined.drop(['ON','ON_stds'], axis = 1)

            State_NAT_poll_merged, State_Polling_CAGO_ALR =  get_ALR_deviations_from_National(State_Polling_CAGO, ref_col)

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

            


            missing_states = {'2016': ['NT','ACT'],'2019':['NT','ACT','TAS'],'2022':['NT','ACT','TAS'],'2025':['NT','ACT','TAS']}

            imputed_polling_ALR = impute_missing_polling(missing_states[election_year], State_Correlation_Matrix, State_Polling_CAGO_ALR_swing, National_State_ALR_df_last)
            existing_polling_ALR = State_Polling_CAGO_ALR_curr.set_index('State').iloc[:,:3]
            combined_polling_ALR = pd.concat([existing_polling_ALR,imputed_polling_ALR])


            imputed_state_polling = alr_to_simplex_vectorized(imputed_polling_ALR, 'COAL').reset_index(names='State')
            imputed_state_polling.loc[:,'Election_year'] = election_year

            StatePolling = pd.concat((StatePolling, imputed_state_polling), ignore_index=True)

            StatePolling.to_csv("2025_Full_State_Inputed_Polling.csv", index=False)

            State_Polling_Deviations_from_National = get_ALR_deviations_from_National(StatePolling, ref_col, include_ON = 1)[0]
            State_Polling_Deviations_from_National.iloc[:,:4] = State_Polling_Deviations_from_National.iloc[:,-4:].values
            State_Polling_Deviations_from_National = State_Polling_Deviations_from_National.iloc[:,:6].sort_values(by=['Election_year','State'])
            State_Polling_Deviations_from_National.loc[:,'Election_year'] = State_Polling_Deviations_from_National.loc[:,'Election_year'].astype(str)

            State_Polling_Deviations_from_National.to_csv(f"2025_State_Polling_Deviations_from_National_Day_{Day}.csv", index=False)








def non_uniform_swing_weight_exp(O_series, beta=0.5, start=0.2, k=10):

    O = O_series.values
    weight = np.ones_like(O)

    mask = O > start
    decay = np.exp(-k * (O[mask] - start))
    weight[mask] = (1 - beta) * decay + beta

    return pd.Series(weight, index=O_series.index)

def full_weight_vector(all_divisions, high_others_df, others_column="OTH", division_state_indices=None, beta=0.5, k=10, start=0.2):
    """
    Parameters:
        all_divisions: list or Index of all 150 electorates
        high_others_df: DF with OTH column and index = electorates with high Others
        division_state_map: Series mapping each electorate to its state index
    """

    # Step 1: Exponential weights for high Others
    high_weights = non_uniform_swing_weight_exp(high_others_df[others_column], beta=beta, k=k, start=start)

    # Step 2: Build full weight vector (init with NaNs)
    full_weights = pd.Series(index=all_divisions, dtype=float)
    full_weights.loc[high_weights.index] = high_weights

    # Step 3: Fill in low Others weights to preserve state average
    for state_idx in np.unique(division_state_indices):
        # Get all divisions in this state
        div_mask = division_state_indices == state_idx
        state_divisions = np.array(all_divisions)[div_mask]
        state_weights = full_weights.loc[state_divisions]

        n_known = state_weights.notna().sum()
        sum_known = state_weights.dropna().sum()
        n_total = len(state_weights)
        n_unknown = n_total - n_known

        if n_unknown > 0:
            required_total = n_total * 1.0
            constant = (required_total - sum_known) / n_unknown
            full_weights.loc[state_weights[state_weights.isna()].index] = constant
            #print(constant)
        else:
            # All weights known — nothing to fill
            continue

        #import pdb;pdb.set_trace()

    return full_weights


def get_weights_by_beta(election_years = ['2016','2019','2022','2025'], beta_list = [r for r in np.arange(0,1.01,0.1)]):

    weights_by_beta = {}

    for election_year in election_years:

        if election_year != '2025':
            div_to_state = pd.read_csv(f"{election_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'}).set_index('div_nm')

        else:
            div_to_state = pd.read_csv(f"2022HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
            div_to_state.loc[div_to_state['div_nm'] == 'North Sydney',] = 'Bullwinkel', 'WA'
            div_to_state = div_to_state.loc[~(div_to_state['div_nm'] == 'Higgins'),]
            div_to_state = div_to_state.set_index('div_nm')

            # re-order
            Electorate_order_2025 = pd.read_csv('Electorate_names_2025.csv')['post_title']

            div_to_state = div_to_state.loc[Electorate_order_2025.values]


        division_state_indices = div_to_state['StateAb'].map(STATE_TO_INDEX).values 

        all_divisions = div_to_state.index

        High_Others_df = pd.read_csv(f"High_Prior_OTH_Electorates_{election_year}.csv", index_col = 0)

        #import pdb;pdb.set_trace()


        # Pre-compute for each election_year and beta value: 0 to 1, intervals 0.1
        weights_by_beta[election_year] = {
            np.round(beta_val,2): full_weight_vector(
                all_divisions, High_Others_df,
                others_column="OTH",
                division_state_indices=division_state_indices,
                beta=beta_val,
                k=10,
                start=0.2
            )
            for beta_val in beta_list 
        }

    return weights_by_beta


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
    if election_year in ['2013','2016']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum6}], index=[div])
    elif election_year in ['2019','2022']:
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'UAPP':sum5, 'Other':sum6}], index=[div])
    elif election_year == '2025':
        Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'ON':sum4, 'TOP':sum5, 'Other':sum6}], index=[div])


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


def get_results_df(election_year, to_Fundamentals = True):

    # returns Actual_results_dict with results by party per division, and Fundamentals_results_df


    Actual_results = pd.read_csv(f"{election_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    # CountNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


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




def remove_ON_back_to_its_country(Prior_estimates_df, election_year):

    # determine transfer ratio of ON votes for those divisions where they are not running

    Prior_estimates_ON_add_df = Prior_estimates_df

    true_prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON = True)[0].rename(columns={'Other':'OTH'})

    ON_transfer_percent = {}
    #import pdb;pdb.set_trace()


    # transfer prior %s from ON to original_df parties
    if election_year != '2025':
        for div, proportions in true_prior_estimates_df.iterrows():
            if (proportions['ON']==0):
                curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
                curr_div_True = proportions.to_frame().T

                transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop('ON', axis=1).sum(axis=1).iloc[0]) # This should provide -1 for ON automatically!
                ON_transfer_percent[div] = transfer_proportions

    else:
        # remove both TOP and ON
        for div, proportions in true_prior_estimates_df.iterrows():
            if (proportions['TOP']==0) & (proportions['ON']==0):
                curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
                curr_div_True = proportions.to_frame().T

                transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop(['ON','TOP'], axis=1).sum(axis=1).iloc[0]) # ON and TOP share the -1 between them - careful!
                ON_transfer_percent[div] = transfer_proportions

            elif (proportions['TOP']==0):
                curr_div_ON = Prior_estimates_ON_add_df.loc[Prior_estimates_ON_add_df.index == div,]
                curr_div_True = proportions.to_frame().T

                transfer_proportions = (curr_div_True - curr_div_ON)/((curr_div_True - curr_div_ON).drop('TOP', axis=1).sum(axis=1).iloc[0]) # This should provide -1 for ON automatically!
                ON_transfer_percent[div] = transfer_proportions


    return ON_transfer_percent


def get_National_State_Prior_estimates(election_year, new_vote_totals_states, dont_add_ON = False, adjust_No_OTHs = True):

    Prior_estimates_df = get_Prior_estimates_df(election_year, dont_add_ON)[0].rename(columns={'Other':'OTH'}) # adds ON to every seat if no ON (for 2019 and 2022)

    merged_totals = Prior_estimates_df.merge(new_vote_totals_states.set_index('div_nm')[['new_vote_totals']], left_index=True, right_index=True)
    weights = merged_totals['new_vote_totals']/merged_totals['new_vote_totals'].sum()
    National_prior = (merged_totals.iloc[:,:-1] * weights.values[:,None]).sum().to_frame().T

    merged_totals_states = Prior_estimates_df.merge(new_vote_totals_states.set_index('div_nm'), left_index=True, right_index=True)

    State_prior_df_list = [] 
    for state in sorted(merged_totals_states['StateAb'].unique()):
        merged_totals_curr_state = merged_totals_states.loc[merged_totals_states['StateAb']==state,].drop('StateAb', axis = 1) # no longer need StateAb
        curr_weights = merged_totals_curr_state['new_vote_totals']/merged_totals_curr_state['new_vote_totals'].sum()
        State_prior = (merged_totals_curr_state.iloc[:,:-1] * curr_weights.values[:,None]).sum().to_frame().T
        State_prior.index = [state]
        State_prior_df_list.append(State_prior)
    State_prior_df = pd.concat(State_prior_df_list)



    # add 0.001 to Gorton Other in 2016, or 0 others in 2019/2022 (use ON votes as they are higher --> less distortion)!
    No_OTH_divisions = []

    if election_year == '2016':
        Prior_estimates_df.loc[Prior_estimates_df.index=='Gorton',['GRN','OTH']] += (-0.005,+0.005)
    elif adjust_No_OTHs:
        No_OTH_divisions = Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,].index
        Prior_estimates_df.loc[Prior_estimates_df['OTH']==0.0,['GRN','OTH']] += (-0.005,+0.005)


    return Prior_estimates_df, National_prior, State_prior_df, No_OTH_divisions



def simulate_Polling_Fundamentals_model(n_simulations, election_year, Day, ref_col = 'COAL', df_t = 0, v = 0.1, s = 0.6, beta = 0.5, forced_polling_average = []):

    # forced_polling_average allows input of list of 6 vote proportions summing to 1 for [ALP, COAL, GRN, ON, TOP, OTH]

    dist = "Normal" if df_t == 0 else "t"
    #print(dist)
    Volatility_cat = pd.read_csv(f"Volatility_weights_df_{election_year}.csv", index_col= None)

    weights_idx_dict = defaultdict(list)
    for idx, scale in enumerate(Volatility_cat['Volatility_weights']):
        weights_idx_dict[scale].append(idx)

    weights_idx_dict = dict(weights_idx_dict)
    


    National_Polling_error_ALR_cov = pd.read_csv(f"PollingErrorALRCovarianceNational{election_year}_Day_{Day}_{ref_col}.csv", index_col=0)
    National_Simulated_polling_error = np.random.multivariate_normal(mean = np.zeros(len(National_Polling_error_ALR_cov)), cov = National_Polling_error_ALR_cov.values, size=n_simulations)[:, None, :] 
    # Broadcast national polling results across 10000 simulations and 150 electorates
    National_Simulated_polling_error_expanded = np.repeat(National_Simulated_polling_error, NO_OF_ELECTORATES[election_year], axis=1)


    National_Election_error_ALR_cov = pd.read_csv(f"ElectionErrorALRCovarianceNational{election_year}_{ref_col}.csv", index_col=0)
    National_Simulated_election_error = np.random.multivariate_normal(mean = np.zeros(len(National_Election_error_ALR_cov)), cov = National_Election_error_ALR_cov.values, size=n_simulations)[:, None, :] 
    National_Simulated_election_error_expanded = np.repeat(National_Simulated_election_error, NO_OF_ELECTORATES[election_year], axis=1)


    # state errors
    State_Polling_error_ALR_cov = pd.read_csv(f"PollingErrorALRCovarianceStateDeviation{election_year}.csv", index_col=0) 
    State_Simulated_polling_error = np.random.multivariate_normal(mean = np.zeros(len(State_Polling_error_ALR_cov)), cov = State_Polling_error_ALR_cov.values, size=n_simulations*NO_OF_STATES)
    State_Simulated_polling_error = State_Simulated_polling_error.reshape(n_simulations, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])

    State_Election_error_ALR_cov = pd.read_csv(f"ElectionErrorALRCovarianceStateDeviation{election_year}.csv", index_col=0)  
    State_Simulated_election_error = np.random.multivariate_normal(mean = np.zeros(len(State_Election_error_ALR_cov)), cov = State_Election_error_ALR_cov.values, size=n_simulations*NO_OF_STATES)
    State_Simulated_election_error = State_Simulated_election_error.reshape(n_simulations, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])




    # electorate errors
    if election_year in ['2016','2019','2022','2025']:
        last_election_year = str(int(election_year) - 3)
        last_election_vote_totals = pd.read_csv(f"{last_election_year}HouseVotesCountedByDivision.csv", skiprows=1, index_col=None).rename(columns={'DivisionNm':'old_div'})[['old_div', 'TotalVotes']]
        redistribution_df = pd.read_csv(f'Correspondence_CED_{str(int(election_year)-4)}_{str(int(election_year)-1)}.csv', index_col = None)

        merged_df = redistribution_df.merge(last_election_vote_totals, on="old_div")
        merged_df["new_vote_totals"] = merged_df["TotalVotes"] * merged_df["RATIO_FROM_TO"]
        new_vote_totals = merged_df.groupby("new_div")["new_vote_totals"].sum().reset_index().rename(columns={'new_div':'div_nm'})

        if election_year == '2025':
            div_to_state = pd.read_csv(f"2022HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
            div_to_state.loc[div_to_state['div_nm'] == 'North Sydney',] = 'Bullwinkel', 'WA'
            div_to_state = div_to_state.loc[~(div_to_state['div_nm'] == 'Higgins'),]
        else:
            div_to_state = pd.read_csv(f"{election_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        new_vote_totals_states = new_vote_totals.merge(div_to_state, on = 'div_nm', how='left')


    # get csv of electorates with high OTH vote share - likely to be more volatile
    for election_year in ['2016','2019','2022','2025']:
        Prior_estimates_df, National_prior, State_prior_df, No_OTH_divisions = get_National_State_Prior_estimates(election_year, new_vote_totals_states)
        #import pdb;pdb.set_trace()
        Prior_estimates_df.loc[Prior_estimates_df['OTH']>0.2,['OTH']].to_csv(f"High_Prior_OTH_Electorates_{election_year}.csv", index = True)



    

    # weights of each state and division:
    Div_relative_weights_dict = {}
    State_relative_weights_dict = {}
    for year in ['2016','2019','2022','2025']:
        last_election_year = str(int(year) - 3)

        Enrolment_by_Div_prev = pd.read_csv(f"{last_election_year}GeneralEnrolmentByDivision.csv",index_col=None, skiprows=1).rename(columns={'DivisionNm':'old_div','StateAb':'State'})[['old_div','State','Enrolment']]
        # adjust for redistribution
        Correspondence_old_new = pd.read_csv(f"Correspondence_CED_{str(int(year) - 4)}_{str(int(year) - 1)}.csv")
        merged = Correspondence_old_new.merge(Enrolment_by_Div_prev, on='old_div')
        merged['Enrolment'] = merged['Enrolment'] * merged['RATIO_FROM_TO']
        Enrolment_by_Div = merged.groupby(['new_div','State'])['Enrolment'].sum().reset_index().rename(columns={'new_div':'div_nm'})

        Enrolment_by_State_prev = Enrolment_by_Div.groupby('State')['Enrolment'].sum()
        State_relative_weights = Enrolment_by_State_prev / Enrolment_by_State_prev.sum()
        Div_relative_weights = Enrolment_by_Div.iloc[:,:2]
        Div_relative_weights.loc[:,'Relative weights'] = Enrolment_by_Div['Enrolment'] / Enrolment_by_Div.groupby('State')['Enrolment'].transform('sum')
        Div_relative_weights = Div_relative_weights.set_index('div_nm')

        Div_relative_weights_dict[year] = Div_relative_weights
        State_relative_weights_dict[year] = State_relative_weights


    Div_relative_weights = Div_relative_weights_dict[election_year]
    State_relative_weights = State_relative_weights_dict[election_year]

    

    # Obtain Electorate residual CovMs -  variability of electorate results corrected for state swings

    def obtain_Electorate_Residuals_CovM(new_vote_totals_states, Div_relative_weights, year_to_remove):

        State_Results_2016_2022 = pd.read_csv('StateResults2016_2022.csv', index_col=None)

        CAGO = 1 if year_to_remove == '2016' else 0

        Electorate_residuals_list = []

        for year in [p for p in ['2016','2019','2022'] if p != year_to_remove]: # ['2016','2019','2022']:
            NUM_DIMS = DIM_OF_COV_MATRIX[year_to_remove] if year != '2016' else 3

            Prior_estimates_df1, National_prior1, State_prior_df1, No_OTH_divisions1 = get_National_State_Prior_estimates(year, new_vote_totals_states, dont_add_ON = True, adjust_No_OTHs=False)
            State_Results_curr = State_Results_2016_2022.loc[State_Results_2016_2022['Election']==int(year),]
            Results_df = get_results_df(year)[0].rename(columns={'Other':'OTH'})
            
            if CAGO and (year in ['2019','2022']):
                Prior_estimates_df1.loc[:,'OTH'] = Prior_estimates_df1.iloc[:,-3:].sum(axis=1)
                Prior_estimates_df1 = Prior_estimates_df1.drop(columns=['ON','UAPP'])
                State_prior_df1.loc[:,'OTH'] = State_prior_df1.iloc[:,3:6].sum(axis=1)
                State_prior_df1 = State_prior_df1.drop(columns=['ON','UAPP'])
                State_Results_curr.loc[:,'OTH'] = State_Results_curr.iloc[:,3:6].sum(axis=1)
                State_Results_curr = State_Results_curr.drop(columns=['ON','UAPP'])
                Results_df.loc[:,'OTH'] = Results_df.iloc[:,3:6].sum(axis=1)
                Results_df = Results_df.drop(columns=['ON','UAPP'])

            elif year == '2016':
                State_Results_curr = State_Results_curr.drop(columns=['ON','UAPP'])

            else: # not CAGO and year is 2019/2022 - replace 0s with nan!
                Prior_estimates_df1 = Prior_estimates_df1.replace(0,np.nan)
                State_prior_df1 = State_prior_df1.replace(0,np.nan)
                State_Results_curr = State_Results_curr.replace(0,np.nan)
                Results_df = Results_df.replace(0,np.nan)

            State_Results_curr = State_Results_curr.drop('Election', axis=1).set_index('State')
            
            # convert both prior and results to ALR
            ref_col = 'COAL'
            State_Results_ALR = np.log(State_Results_curr.drop(columns=[ref_col]).div(State_Results_curr[ref_col], axis=0))
            State_Prior_ALR = np.log(State_prior_df1.drop(columns=[ref_col]).div(State_prior_df1[ref_col], axis=0))
            Prior_estimates_ALR_df1 = np.log(Prior_estimates_df1.drop(columns=[ref_col]).div(Prior_estimates_df1[ref_col], axis=0))
            #import pdb;pdb.set_trace()

            Div_relative_weights = Div_relative_weights_dict[year]

            # add to corresponding divisions in states
            True_State_ALR_swings = State_Results_ALR - State_Prior_ALR
            Prior_estimates_ALR_df1.loc[:,'State'] = Div_relative_weights['State'].values
            merged = pd.merge(Prior_estimates_ALR_df1, True_State_ALR_swings, left_on = 'State',right_index = True, suffixes = ('','_state_swing'))
            #import pdb;pdb.set_trace()
            merged.iloc[:,:NUM_DIMS] += merged.iloc[:,-NUM_DIMS:].values
            State_swing_ALR = merged.iloc[:,:NUM_DIMS]
            #

            # get actual results for 4/6 parties
            Results_df_ALR = np.log(Results_df.drop(columns=[ref_col]).div(Results_df[ref_col], axis=0))
            Electorate_residuals = Results_df_ALR - State_swing_ALR

            Electorate_residuals.index = year + Electorate_residuals.index

            if (year == '2016') and (year_to_remove != '2016'):
                Electorate_residuals.loc[:,['ON','UAPP']] = np.nan,np.nan

            Electorate_residuals_list.append(Electorate_residuals)


        Electorate_residuals_ALR_df = pd.concat(Electorate_residuals_list)
        Electorate_residuals_ALR_df = Electorate_residuals_ALR_df.loc[~(Electorate_residuals_ALR_df.index.str.startswith(year_to_remove)),]

        if year_to_remove != '2016':
            Electorate_residuals_ALR_df = Electorate_residuals_ALR_df[['ALP','GRN','ON','UAPP','OTH']]


        return Electorate_residuals_ALR_df.cov(min_periods=1)


    Electorate_residuals_covMs = {}
    for year_to_remove in ['2016','2019','2022','2025']:
        Electorate_residuals_covMs[year_to_remove] = obtain_Electorate_Residuals_CovM(new_vote_totals_states, Div_relative_weights, year_to_remove)

        Electorate_residuals_covMs[year_to_remove].to_csv(f"ElectorateResidualALRCovariance{year_to_remove}.csv", index = True)



    # this should be tuned:
    Electorate_Residuals_cov = pd.read_csv(f"ElectorateResidualALRCovariance{election_year}.csv", index_col=0) 

    Scaled_covs = {}
    category_weights = {0:0.95, 1:1.25,2:1.5,3:4}

    OTH_index = -1
    for cat, weight in category_weights.items():
        scaling = (1 + (weight-1) * v) if (weight > 1) else (1 - (1-weight)*v) # proper variability of Others
        
        # Adjust the OTH variance and covariances
        cov_adj = Electorate_Residuals_cov.values.copy()
        
        # Amount to add to the OTH row and column
        current_var = Electorate_Residuals_cov.values[OTH_index, OTH_index]
        target_var = scaling**2 * current_var
        delta = target_var - current_var
        
        # Add delta to variance
        cov_adj[OTH_index, OTH_index] += delta
        
        # Add delta to covariances with OTH, assuming proportional increase
        for i in range(DIM_OF_COV_MATRIX[election_year]):
            if i != OTH_index:
                cov_adj[OTH_index, i] *= scaling
                cov_adj[i, OTH_index] = cov_adj[OTH_index, i]  # keep symmetry
        
        Scaled_covs[cat] = cov_adj
        #import pdb;pdb.set_trace()


    #Electorate_Residuals_Simulated_error = np.random.multivariate_normal(mean = np.zeros(len(Electorate_Residuals_cov)), cov = Electorate_Residuals_cov.values, size=n_simulations*NO_OF_ELECTORATES[election_year])
    #Electorate_Residuals_Simulated_error = Electorate_Residuals_Simulated_error.reshape(n_simulations, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year])



    Electorate_Residuals_Simulated_error = np.empty((n_simulations, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year]))

    d = len(Electorate_Residuals_cov)
    n = n_simulations * NO_OF_ELECTORATES[election_year]
    cov = Electorate_Residuals_cov.values

    # Batch simulate by category:
    for scale, indices in weights_idx_dict.items():
        cov = Scaled_covs[scale]  # Use the scaled covariance matrix
        n_group = len(indices)

        if n_group == 0:
            continue  # skip empty groups

        # Generate multivariate normal samples for this group - # 2. Standard multivariate normal samples # shape: (n, d)

        group_sims = np.random.multivariate_normal(
            mean=np.zeros(d),
            cov=cov,
            size=(n_simulations * n_group)
        )

        if dist == 't':
        
            g = np.random.gamma(df_t / 2., 2. / df_t, size=n_simulations *n_group)   # 1. Gamma samples (for scaling) ; shape: (n,)
            group_sims = group_sims / np.sqrt(g)[:, None]  # shape: (n, d) # 3. Scale by sqrt(gamma) to simulate from t-distribution


        # Place the group simulations into the correct positions
        Electorate_Residuals_Simulated_error[:, indices, :] = group_sims.reshape(n_simulations, n_group, DIM_OF_COV_MATRIX[election_year]) # 4. Reshape to (n_simulations, electorates, dim)



    # centre state polling/swing deviations to weighted sum of 0!
    w = State_relative_weights.values.reshape(1,NO_OF_STATES,1)
    weighted_means = np.sum(State_Simulated_polling_error * w, axis=1, keepdims=True)  # Sum over states
    State_Simulated_polling_error_centered = State_Simulated_polling_error - weighted_means # Subtract the weighted mean from each state: shape still (10000, 8, 5)

    weighted_means = np.sum(State_Simulated_election_error * w, axis=1, keepdims=True)  # Sum over states
    State_Simulated_election_error_centered = State_Simulated_election_error - weighted_means # Subtract the weighted mean from each state: shape still (10000, 8, 5)

    Scaled_precisions_curr = pd.read_csv("State_Polling_Scaled_Precisions.csv", index_col = 0).loc[election_year] 
    relative_state_precisions = Scaled_precisions_curr.set_index('Scope', drop = True).drop('Mean_precision', axis = 1)

    s_i = s * relative_state_precisions

    #print("s_i", v, s)

    # weight the State poll deviaiton and electorate poll deviation by s_i per state
    s_i_reshaped = s_i.values.reshape(1,NO_OF_STATES,1)

    State_Simulated_polling_error_centered = s_i_reshaped * State_Simulated_polling_error_centered + (1-s_i_reshaped) * State_Simulated_election_error_centered


    #import pdb;pdb.set_trace()


    # reshape State simulations correctly - map to correct div_nms
    division_state_indices = Div_relative_weights['State'].map(STATE_TO_INDEX).values  # shape (150,)

    State_Simulated_polling_error_centered_expanded = State_Simulated_polling_error_centered[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]
    State_Simulated_election_error_centered_expanded = State_Simulated_election_error_centered[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]

    #import pdb;pdb.set_trace()


    # Centre Electorate Residuals to 0!
    w_div = Div_relative_weights.iloc[:,1:].values.reshape(1,NO_OF_ELECTORATES[election_year],1)
    weighted_means = np.sum(Electorate_Residuals_Simulated_error * w_div, axis=1, keepdims=True)

    centered_residuals = np.empty_like(Electorate_Residuals_Simulated_error)

    for state_idx in range(len(STATES)):  # 8 states

        div_indices = np.where(division_state_indices == state_idx)[0] #  Find divisions that belong to this state
        weights = Div_relative_weights.iloc[div_indices]['Relative weights'].values.reshape(1, -1, 1) # Get relative weights for these divisions
        residuals = Electorate_Residuals_Simulated_error[:, div_indices, :] # Get the residuals for these divisions
        weighted_mean = np.sum(residuals * weights, axis=1, keepdims=True)  # Compute weighted state mean
        centered_residuals[:, div_indices, :] = residuals - weighted_mean # Center the residuals

    Electorate_Residuals_Simulated_error_centered = centered_residuals



    def alr_to_simplex_simulation_array(alr_array):
        """
        Perform inverse ALR transformation on a 3D array of shape (S, D, K).
        Returns a 3D array of shape (S, D, K+1) in the probability simplex.
        """
        exp_vals = np.exp(alr_array)  # shape (S, D, K)
        ref_vals = 1 / (1 + np.sum(exp_vals, axis=-1, keepdims=True))  # shape (S, D, 1)
        simplex = np.concatenate([ref_vals, exp_vals * ref_vals], axis=-1)  # shape (S, D, K+1)
        return simplex




    Day = 90

    if election_year in ['2016','2019','2022']:
        day_z_polling_avg_df = pd.read_csv(f"National_Day_{Day}_Polls.csv")

        day_z_polling_avg = day_z_polling_avg_df.loc[day_z_polling_avg_df['Election'] == int(election_year),].drop('Election', axis = 1).reset_index(drop=True)

        if election_year == '2016':
            day_z_polling_avg = day_z_polling_avg[['COAL','ALP','GRN','OTH']]

    elif election_year == '2025':
        day_z_polling_avg_df = pd.read_csv(f"National_Day_{Day}_Polls.csv")

        day_z_polling_avg = day_z_polling_avg_df.loc[day_z_polling_avg_df['Election'] == int(election_year),].drop('Election', axis = 1).reset_index(drop=True)
        day_z_polling_avg = day_z_polling_avg.rename(columns = {'UAPP':'TOP'})

        

        # try arbitrary polling average for experiments
        if forced_polling_average:


            if np.array(forced_polling_average).sum()!=1:
                raise ValueError("Vote shares do not sum to 1")

            day_z_polling_avg.iloc[0,:] = forced_polling_average # 0.39,0.305,0.12,0.07,0.01,0.105





    #day_80_polling_avg_dict = {'2016': pd.DataFrame([[0.412262, 0.351972, 0.105693, 0.130074]], columns = ['COAL','ALP','GRN','OTH']), \
    #                    '2019':pd.DataFrame([[0.384782, 0.36451, 0.095751, 0.035, 0.031, 0.088957]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
    #                    '2022':pd.DataFrame([[0.355905, 0.362643, 0.118432, 0.0383,0.0244,0.10032,]], columns = ['COAL','ALP','GRN','ON','UAPP','OTH']), \
    #                    '2025':pd.DataFrame([[0.346862,  0.319669,  0.128995,  0.071243,  0.016797,  0.116434]], columns = ['COAL','ALP','GRN','ON','TOP','OTH'])}
    
                        # 0.352028, 0.3153, 0.127548, 0.07171, 0.011641, 0.121772

                        # 0.346862  0.319669  0.128995  0.071243  0.016797  0.116434


    
    state_poll_dev_alr = pd.read_csv("State_Polling_Deviations_from_National.csv", index_col=None)
    state_poll_dev_alr_2025 = pd.read_csv(f"2025_State_Polling_Deviations_from_National_Day_{Day}.csv", index_col=None)


    State_Polls_Deviations_from_National_df_dict = {'2016': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2016,].drop(['ON','Election_year'], axis=1), \
                                                    '2019': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2019,].drop(['Election_year'], axis=1).fillna(0), \
                                                    '2022': state_poll_dev_alr.loc[state_poll_dev_alr['Election_year']==2022,].drop(['Election_year'], axis=1).fillna(0), \
                                                    '2025': state_poll_dev_alr_2025.drop(['Election_year'], axis=1).fillna(0),}



    # get alr values of all quantities
    ref_col = 'COAL'
    polling_alr = np.log(day_z_polling_avg.drop(columns=[ref_col]).div(day_z_polling_avg[ref_col], axis=0))
    National_prior_alr = np.log(National_prior.drop(columns=[ref_col]).div(National_prior[ref_col], axis=0))

    State_prior_alr =  np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))
    State_prior_expanded = np.tile(State_prior_alr.to_numpy(), (n_simulations, 1, 1)).reshape(n_simulations, NO_OF_STATES, DIM_OF_COV_MATRIX[election_year])
    State_prior_expanded = State_prior_expanded[np.arange(n_simulations)[:, None], division_state_indices[None, :], :] # Then use advanced indexing to map divisions to their state across all samples

    #import pdb;pdb.set_trace()
    # get initial state deviations and expand
    State_prior_deviation_alr_expanded = State_prior_expanded - National_prior_alr.values.flatten()








    Prior_estimates_alr =  np.log(Prior_estimates_df.drop(columns=[ref_col]).div(Prior_estimates_df[ref_col], axis=0))
    Prior_estimates_alr_expanded = np.tile(Prior_estimates_alr.to_numpy(), (n_simulations, 1, 1))

    # get State deviations into a (10000, 8, 5) array
    State_polling_deviation_alr = State_Polls_Deviations_from_National_df_dict[election_year].set_index('State')
    if election_year in ['2019','2022']:
        State_polling_deviation_alr.loc[:,'UAPP'] = 0.0 # add 0 deviation from National if no state polling!
        State_polling_deviation_alr = State_polling_deviation_alr[['ALP','GRN','ON','UAPP','OTH']]
    elif election_year == '2025':
        State_polling_deviation_alr.loc[:,'TOP'] = 0.0 # add 0 deviation from National if no state polling!
        State_polling_deviation_alr = State_polling_deviation_alr[['ALP','GRN','ON','TOP','OTH']]

    # scale the state polling deviations, based on relative polling precision (sample size etc.)

    #relative_state_precisions = relative_state_precisions['Scaled_Precisions']/relative_state_precisions['Scaled_Precisions'].mean()

    
    State_polling_deviation_alr = State_polling_deviation_alr.mul(s_i.values, axis = 0) # now scaled based on state precision


    #import pdb;pdb.set_trace()


    State_polling_deviation_alr_matrix = State_polling_deviation_alr.values  # Convert to numpy array for easy broadcasting
    State_polling_deviation_alr_matrix = np.expand_dims(State_polling_deviation_alr_matrix, axis=0)  # Add an extra dimension for broadcasting
    State_polling_deviation_alr_matrix_expanded = np.repeat(State_polling_deviation_alr_matrix, n_simulations, axis=0)  
    State_polling_deviation_alr_matrix_expanded = State_polling_deviation_alr_matrix_expanded[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]

    #State_prior_alr = np.log(State_prior_df.drop(columns=[ref_col]).div(State_prior_df[ref_col], axis=0))

    # apply National Polling error
    Simulated_national_result_alr = National_Simulated_polling_error_expanded + polling_alr.values  # shape: [1M, 5]

    # apply State Polling error
    Simulated_state_polling_deviation = State_polling_deviation_alr_matrix_expanded + State_Simulated_polling_error_centered_expanded

    # weight State Polling, relative to National: use linear combination of last year's 
    #import pdb;pdb.set_trace()
    
    #state_weights = s * Year_state_polling_weights.loc[int(election_year)].iloc[0]

    # get 1 - s_i per state

    Scaled_national_prior_deviations = (State_prior_alr - National_prior_alr.values).mul(1-s_i.values, axis = 0)
    Scaled_national_prior_deviations.index = Scaled_national_prior_deviations.index.map(STATE_TO_INDEX)
    seat_state_alr = Scaled_national_prior_deviations.iloc[division_state_indices]
    Scaled_national_prior_deviations_expanded = np.broadcast_to(seat_state_alr, (n_simulations, NO_OF_ELECTORATES[election_year], DIM_OF_COV_MATRIX[election_year])).copy()
    
    #Scaled_national_polling_deviations_array = np.expand_dims(Scaled_national_polling_deviations, axis=0)  # Add an extra dimension for broadcasting
    #Scaled_national_polling_deviations_expanded = np.repeat(Scaled_national_polling_deviations_array, n_simulations, axis=0)  
    #Scaled_national_polling_deviations_expanded = Scaled_national_polling_deviations_expanded[np.arange(n_simulations)[:, None], division_state_indices[None, :], :]


    # get High Others votes and index set of all divisions

    weights_by_beta = get_weights_by_beta() # weights for seat volatility (esp. seats with high-profile/emerging Independent candidates)
   
    Non_Uniformity_weights = weights_by_beta[election_year][np.round(beta,2)].values[None,:,None]


    Combined_state_mean = polling_alr.values + Scaled_national_prior_deviations_expanded + State_polling_deviation_alr_matrix_expanded # already scaled!
    Combined_state_errors = National_Simulated_polling_error_expanded + State_Simulated_polling_error_centered_expanded

    Simulated_State_Weighted_Polling_Results = Combined_state_mean + Combined_state_errors

    #Simulated_State_Polling_Results = Simulated_national_result_alr + (1 - state_weights) * (National_prior_alr.values - State_prior_expanded) + (state_weights) * Simulated_state_polling_deviation

    #import pdb;pdb.set_trace()

    Projected_Electorate_Results = Prior_estimates_alr_expanded + Non_Uniformity_weights*(Simulated_State_Weighted_Polling_Results - State_prior_expanded)

    Simulated_Electorate_Polling_Results_ALR = Projected_Electorate_Results + Electorate_Residuals_Simulated_error_centered

    Simulated_Electorate_Polling_Results = alr_to_simplex_simulation_array(Simulated_Electorate_Polling_Results_ALR)


    # Now, do the same for Fundamentals:

    # 1. GET NATIONAL SWING (centered at 0)
    # 2. START WITH PRIOR STATE DEVIATIONS ( from state_prior) - essentially from last election
    # 3. ADD SIMULATED STATE DEVIATIONS
    # 4. ADD THIS TO NATIONAL SWING RESULT
    # 5. ADD TO ALL ELECTORATES
    # 6. ADD ELECTORATE ERROR

    Simulated_national_swing_alr = National_Simulated_election_error_expanded + National_prior_alr.values # 1
    Simulated_state_election_deviation = State_prior_deviation_alr_expanded + State_Simulated_election_error_centered_expanded # 2
    Simulated_State_election_Results = Simulated_national_swing_alr + Simulated_state_election_deviation

    Projected_Electorate_Swing_Results = Prior_estimates_alr_expanded + Non_Uniformity_weights*(Simulated_State_election_Results - State_prior_expanded)

    Simulated_Electorate_Swing_Results_ALR = Projected_Electorate_Swing_Results + Electorate_Residuals_Simulated_error_centered

    Simulated_Electorate_Swing_Results = alr_to_simplex_simulation_array(Simulated_Electorate_Swing_Results_ALR)


    #import pdb;pdb.set_trace()




    # Adjust how uniform the seats swing based on difference to national polling:
    




    # Post-processing to remove artificial additions (includding ON in 2019/22)!
    if election_year == '2016':
        div_idx = Div_relative_weights.index.get_loc('Gorton')
        OTH = Simulated_Electorate_Polling_Results[:, div_idx, 3]  # shape (10000,)
        # Apply the shift: increase GRN, decrease OTH
        Simulated_Electorate_Polling_Results[:, div_idx, 2] += 1.0 * OTH  # GRN (index 2)
        Simulated_Electorate_Polling_Results[:, div_idx, 3] -= 1.0 * OTH  # OTH (index 3)

        #import pdb;pdb.set_trace()

    def shift_share(sim, div_idx, from_party_idx, to_party_idx, proportion=1.0):
        """Shifts a proportion of vote share from one party to another in one division, across all simulations."""
        shift_amount = proportion * sim[:, div_idx, from_party_idx]
        sim[:, div_idx, to_party_idx] += shift_amount
        sim[:, div_idx, from_party_idx] -= shift_amount

        return 1

    def redistribute_ON_votes(sim, division_names, party_names, ON_transfer_dict, election_year):
        party_index_map = {name: idx for idx, name in enumerate(party_names)}

        if election_year == '2025':
            ON_idx = party_index_map['TOP'] # confusing - quick adaptation to 2025, where TOP is running fewer candidates
        else:
            ON_idx = party_index_map['ON']

        for div, transfer_row in ON_transfer_dict.items():
            div_idx = division_names.index(div)

            if (election_year == '2025') and (div in ['Canberra', 'Fenner', 'Bean']):

                ON_votes = sim[:, div_idx, [3,4]]

                sim[:, div_idx, :] += np.outer(ON_votes.sum(axis = -1), transfer_row.values)

                # Zero out ON votes
                sim[:, div_idx, [3,4]] = 0.0

            else:

                # Get ON vote array for this division
                ON_votes = sim[:, div_idx, ON_idx]  # shape (10000,)

                sim[:, div_idx, :] += np.outer(ON_votes, transfer_row.values) # add proportions scaled by ON_votes to df

                # Zero out ON votes
                sim[:, div_idx, ON_idx] = 0.0

        return 1  # confirmation


    if election_year in ['2019','2022','2025']:
        for div_nm in No_OTH_divisions:
            div_idx = Div_relative_weights.index.get_loc(div_nm)
            shift_share(Simulated_Electorate_Polling_Results, div_idx, 5, 2, proportion=1.0) # always from OTH to UAPP (maybe different in 2025?)
            shift_share(Simulated_Electorate_Swing_Results, div_idx, 5, 2, proportion=1.0)

            # return ON back to its country in divs it did not run in!
            ON_transfer_dict = remove_ON_back_to_its_country(Prior_estimates_df, election_year)

            Polled_parties = ['COAL','ALP','GRN','ON','UAPP','OTH'] if  election_year != '2025' else ['COAL','ALP','GRN','ON','TOP','OTH']
            redistribute_ON_votes(Simulated_Electorate_Polling_Results, Div_relative_weights.index.tolist(), Polled_parties, ON_transfer_dict, election_year)
            redistribute_ON_votes(Simulated_Electorate_Swing_Results, Div_relative_weights.index.tolist(), Polled_parties, ON_transfer_dict, election_year)






    return Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results




def expand_all_divisions_from_prior_df(sim, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=100):

    LP_NP_VOLATILITY_FACTOR = 2

    final_sim = {}
    party_name_dict = {}
    NUM_MAIN_PARTIES = DIM_OF_COV_MATRIX[election_year]

    multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
    C200_IND_splits = pd.read_csv(f"Independent_splits_multiple.csv", index_col=None)
    C200_IND_positions_df = pd.read_csv("C200_IND_positions_df.csv", index_col=None)
    C200_IND_positions_curr = C200_IND_positions_df.loc[C200_IND_positions_df['Election_year'] == int(election_year),]

    Complex_contests = {'Calare': 0.4788,'Monash': 0.5002,'Moore': 0.43987} # proportions of split to C200 independent and Incumbent independent (based on % after adding Historical proportions in (div,div) pair)



    Major_parties = ['COAL','LP','NP','LNP','LNQ','CLP','ALP','CLR','GRN','GVIC']
    Polling_parties = ['COAL','ALP','GRN']
    if election_year != '2016':
        Major_parties += ['ON','UAPP','TOP']
        if election_year != '2025':
            Polling_parties += ['ON','UAPP']
        else:
            Polling_parties += ['ON','TOP']


    # For the split of COAL_double_divs
    if election_year == '2025':
        NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df_2025.csv", index_col=None)
    else:
        NP_ratios_curr = pd.read_csv("NP_ratio_estimated_df.csv", index_col=None)

    NP_ratios_curr = NP_ratios_curr.loc[(NP_ratios_curr['election_year']==int(election_year)) ,] # & (NP_ratios_curr['State'].isin(['VIC','NSW']))
        

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



            alpha = np.array([1-NP_est,NP_est]) * alpha_scalar/LP_NP_VOLATILITY_FACTOR
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
                means_array = np.full(num_INDs_curr, 1/num_INDs_curr) # initialises an even split



                if div in C200_IND_positions_curr['div_nm'].unique():
                    # split according to C200 splits!

                    C200_ratio = C200_IND_splits.loc[C200_IND_splits['Election'] == f'AverageFor{election_year}','Ratio'].iloc[0]

                    C200_IND_position = C200_IND_positions_curr.loc[C200_IND_positions_curr['div_nm'] == div,'Number'].iloc[0]

                    if (election_year == '2025') and div in Complex_contests.keys():

                        C200_ratio = Complex_contests[div]
                    

                    rest_value = (1 - C200_ratio) / (len(means_array) - 1)
                    means_array[:] = rest_value  # fill all
                    means_array[C200_IND_position - 1] = C200_ratio 

                elif (election_year == '2025') and (div == 'Groom'):
                    import pdb;pdb.set_trace()
                    # mix between C200 and last split! - or ignore? TBD


                splits = np.random.dirichlet(means_array* alpha_scalar, size=sim.shape[0])
                All_IND_votes = splits * IND_votes[:, None]

                combined = np.concatenate([combined, All_IND_votes], axis=1)
                combined = np.delete(combined,IND_index, axis = 1) # remove 'IND' col

                all_party_names = all_party_names + ['IND'+str(i) for i in range(1,num_INDs_curr.iloc[0]+1)]
                all_party_names = [p for p in all_party_names if p != 'IND'] # remove IND


            else:
                all_party_names = [p if p != 'IND' else 'IND1' for p in all_party_names] # renames to IND1 if only single IND


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

def First_Preference_Model_Simulation(election_year, Day, ref_col, w, alpha, v, s, beta, n_simulations = 1000, forced_polling_average = []):



    # Simulate votes for ALP, COAL, GRN, ON, TOP/UAPP, OTH (No ON/TOP/UAPP in 2016)
    Simulated_Electorate_Polling_Results, Simulated_Electorate_Swing_Results = simulate_Polling_Fundamentals_model(n_simulations, election_year, Day, ref_col, v = v, s=s, beta=beta, forced_polling_average=forced_polling_average)


    Prior_estimates_dict = get_Prior_estimates_df(election_year, dont_add_ON = True)[1] # single row df for each div_nm


    Candidates_2025 = pd.read_csv("2025Candidates_By_Division.csv", index_col = None)

    # Make a faux-Results_dict
    Results_dict = {}

    # enumerate independent candidates
    target = 'IND'
    for div, sub_df in Candidates_2025.groupby('div_nm', sort=False):

        div_parties = sub_df
        div_parties.loc[:,'Count'] = div_parties.groupby('PartyAb').cumcount() + 1     # Count instances of the target string

        # To distinguish 'IND', Replace duplicates of 'IND' with increasing strings IND1, IND2, IND3, ...
        adjusted_party_names = div_parties.apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
        ).reset_index(drop=True)

        # separate independents
        div_parties.loc[div_parties['div_nm'] == div,'PartyAb'] = adjusted_party_names.values
        div_results_combined = div_parties.drop('Count', axis = 1)

        ordered_parties = div_results_combined['PartyAb'].drop_duplicates()

        # convert to wide format to match existing Results_dict format
        pivoted = div_results_combined.pivot(index='div_nm', columns='PartyAb')
        Results_dict[div] = pivoted.reindex(columns = ordered_parties)

    # randomly mix Polling Simulations and Fundamentals simulations wiht weight w
    indices = np.random.permutation(n_simulations)

    Weighted_Simulations = np.concatenate((Simulated_Electorate_Polling_Results[indices[:int(w*n_simulations)]],  Simulated_Electorate_Swing_Results[indices[int(w*n_simulations):]]), axis=0)

    # Expand to full candidate size, expanding 'COAL', 'OTH' and the 'IND' categories using Dirichlet distribution with parameter alpha
    final_simulated_votes = expand_all_divisions_from_prior_df(Weighted_Simulations, Prior_estimates_dict, Results_dict, election_year, alpha_scalar=alpha)[0]

    return final_simulated_votes, Results_dict



def scale_state_polling_precision():

    def sigmoid(x, midpoint=1.0, steepness=2.0):
        """Smooth saturation from 0 to 1 centered at midpoint."""
        return 1 / (1 + np.exp(-steepness * (x - midpoint)))

    def compute_effective_state_weights(P_s_y, steepness=2.0):
        """
        Compute effective weights for state polling using a sigmoid of relative precision.
        
        Parameters:
        - P_s_y: pd.Series or np.ndarray of state polling precision per state/year.
        - s: Global trust level in state polling (between 0 and 1).
        - steepness: Controls how rapidly weight saturates toward 1.

        Returns:
        - effective_s: np.ndarray of weights between 0 and s (smoothly scaled).
        """
        # Compute average precision (per-year if needed)
        avg_precision = np.mean(P_s_y)
        relative_precision = P_s_y / avg_precision

        # Apply sigmoid to smooth-saturate the weighting
        scaled = sigmoid(relative_precision, midpoint=1.0, steepness=steepness)
        return scaled

    # scale state weights nicely
    Average_Precisions = pd.read_csv("State_Polling_Average_Precisions.csv")
    Scaled_Precisions =  compute_effective_state_weights(Average_Precisions['Mean_precision'], steepness=2.0)
    Average_Precisions.loc[:,"Scaled_Precisions"] = Scaled_Precisions
    Average_Precisions.set_index('Year').to_csv("State_Polling_Scaled_Precisions.csv", index = True)









def make_party_category_dict():

        all_parties = pd.read_csv('Grand_Party_Category_df_2004_2022.csv', index_col=None)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['CLR'],'Ideo_Category':['ALP'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['NGS'],'Ideo_Category':['Right'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties = pd.concat([all_parties,pd.DataFrame({'PartyAb':['ARTS'],'Ideo_Category':['Left'],'Ideo_Category_Data':[np.nan],'HouseYears':[[]],'SenateYears':[[]]})], ignore_index=True)
        all_parties_house = all_parties.loc[all_parties['Ideo_Category'].notna(),].iloc[:,:2].set_index('PartyAb') # excludes only senates, who don't yet have Ideology written
        party_category_dict = all_parties_house.to_dict()['Ideo_Category']
        party_category_dict['IND'] = 'Centre'
        party_category_dict['COALLP'] = 'COAL'
        party_category_dict['COALNP'] = 'COAL'

        return party_category_dict



def make_TCP_pair_category_dict(election_year, election_years = ['2016','2019','2022','2025'], party_category_dict={}):

    from collections import defaultdict


    data_year = str(int(election_year) - 3)
    

    name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
    new_seats_year_dict = {'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner','Gorton']}

    replacement_seats_year_dict = {'2022': {'Hasluck':'Bullwinkel'}, '2019':{'Gorton':'Hawke'}, '2016':{'Canberra':'Bean', 'Maribyrnong':'Fraser'}, '2013':{'Hasluck':'Burt'}}
    abolished_divs_dict = {'2022':set(['Higgins','North Sydney']), '2016': set(['Port Adelaide']),'2019':set(['Stirling']),'2013':set(['Charlton'])}


    data_years = [str(int(year) - 3) for year in election_years]


    next_year = election_year
    # 1. Get names of next election's parties in each div for comparison to senate
    if next_year != '2025':

        DOP_By_Division_next = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'})[["div_nm","PartyAb"]].drop_duplicates()

    else:
        DOP_By_Division_next = pd.read_csv("2025Candidates_By_Division.csv", index_col = None)
    
    DOP_By_Division_next.loc[:,'PartyAb'] = DOP_By_Division_next.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
    Div_parties_next_dict = {div: group['PartyAb'].tolist() for div, group in DOP_By_Division_next.groupby("div_nm")}

    Div_parties_next_dict_COAL = {div: ['COAL' if p in ['LP', 'NP','CLP','LNP'] else p for p in Div_parties_next_dict[div]] for div in Div_parties_next_dict.keys()}


    TCP_Preference_Flows = pd.read_csv(f"{data_year}HouseTCPFlowByDivision.csv", skiprows = 1, index_col = None).rename(columns = {'DivisionNm':'div_nm','FromCandidatePartyAb':'PartyAb', \
                                                'FromCandidateBallotPosition':'Ballot_Position','ToCandidatePartyAb':'TCP_Ab','ToCandidateBallotPosition':'TCP_Ballot_Position'})
    TCP_Preference_Flows = TCP_Preference_Flows[['div_nm','PartyAb','Ballot_Position','TCP_Ab','TCP_Ballot_Position','TransferPercentage']]
    TCP_Preference_Flows = TCP_Preference_Flows.loc[TCP_Preference_Flows['Ballot_Position']>0,]
    TCP_Preference_Flows = TCP_Preference_Flows[['div_nm','PartyAb','TCP_Ab','TransferPercentage']]
    TCP_Preference_Flows['div_nm'] = TCP_Preference_Flows['div_nm'].replace(name_changes_year_dict[data_year])

    TPP = ('ALP', 'COAL')


    TPP_by_State = pd.read_csv(f"{data_year}HouseTPPFlowByStateByParty.csv", skiprows = 1, index_col = None)
    TPP_by_State = TPP_by_State[['StateAb', 'PartyAb', 'Australian Labor Party Transfer Percentage']]
    TPP_by_State = TPP_by_State.loc[(TPP_by_State['PartyAb'].notna()) & (TPP_by_State['PartyAb'] != 'NAFD'), ].rename(columns={'Australian Labor Party Transfer Percentage':'ALP%'})

    TPP_nationally = pd.read_csv(f"{data_year}HouseTPPFlowByParty.csv", skiprows = 1, index_col = None)[['PartyAb', 'Australian Labor Party Transfer Percentage']].rename(columns={'Australian Labor Party Transfer Percentage':'ALP%'})
    
    if next_year != '2025':
        div_to_state = pd.read_csv(f"{next_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})

    else:
        div_to_state = pd.read_csv(f"2022HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        div_to_state.loc[div_to_state['div_nm'] == 'North Sydney',] = 'Bullwinkel', 'WA'
        div_to_state = div_to_state.loc[~(div_to_state['div_nm'] == 'Higgins'),]

    div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}




    # 1. Get PartyAb: 1st alphabetically for each pair:

    tcp_pairs = (TCP_Preference_Flows.groupby("div_nm")["TCP_Ab"].unique().apply(lambda x: tuple(sorted(set(x)))))  # optional: sort so (ALP, LP) and (LP, ALP) match)

    tcp_coalified = tcp_pairs.apply(lambda tup: tuple('COAL' if x in ['LP', 'NP','LNP','CLP'] else x for x in tup))






    # add TCP to dict!

    Preference_flows_dict = {}
    Non_classic_divs = defaultdict(list)


    def normalize_party(p):
        return 'COAL' if p in ['LP', 'NP'] else p

    def sorted_tcp_pair(tcp1, tcp2):
        return tuple(sorted([normalize_party(tcp1), normalize_party(tcp2)]))


    # Step 1: Precompute the sorted and COALified 2CP pair per division
    tcp_pairs_by_div = (
        TCP_Preference_Flows.groupby("div_nm")["TCP_Ab"]
        .unique()
        .apply(lambda x: sorted_tcp_pair(*x))
    )

    # Step 2: Build the desired structure
    for _, row in TCP_Preference_Flows.iterrows():
        div = row['div_nm']
        party = row['PartyAb']
        tcp = normalize_party(row['TCP_Ab'])
        pct = row['TransferPercentage']

        if div in abolished_divs_dict[data_year]:
            continue
        
        tcp_pair = tuple(sorted([p if p not in ['LP','NP','LNP','CLP'] else 'COAL' for p in tcp_pairs_by_div[div]]))

        if tcp_pair != ('ALP', 'COAL'):
            # we have a Non-classic contest!

            tcp_pair = tuple(sorted([party_category_dict[p] if party_category_dict[p] != 'Centre' else 'IND' for p in tcp_pair]))

            if tcp_pair == ('ALP', 'COAL'):
                import pdb;pdb.set_trace() # should not happen

            if div not in Non_classic_divs[tcp_pair]:
                Non_classic_divs[tcp_pair].append(div)



        first, second = tcp_pair  # alphabetical order
        #if div == 'Brisbane':
        #    import pdb;pdb.set_trace() 
        # We want % transferred to the *first* in alphabetical order
        if party_category_dict[tcp] == first:
            transfer_pct = pct
        else:
            transfer_pct = 100 - pct

        Preference_flows_dict.setdefault(div, {})

        party = party if party not in ['LP','NP','LNP','CLP'] else 'COAL'

        if party in Div_parties_next_dict_COAL[div]:
            Preference_flows_dict[div][party] = {tcp_pair: transfer_pct}

        if div in replacement_seats_year_dict[data_year].keys():
            new_div = replacement_seats_year_dict[data_year][div]
            Preference_flows_dict.setdefault(new_div, {})
            if party in Div_parties_next_dict_COAL[new_div]:
                Preference_flows_dict[new_div][party] = {tcp_pair: transfer_pct}



    #import pdb;pdb.set_trace()

    from collections import defaultdict

    # Assume `result` and `party_categories` are already defined

    # Make a new version of result with category rollups
    Preference_flows_dict_with_categories = {}

    for div, party_dict in Preference_flows_dict.items():
        category_values = defaultdict(list)
        div_result = dict(party_dict)  # copy original party entries

        # Get the TCP pair from any of the entries (they all have the same)
        tcp_pair = next(iter(next(iter(party_dict.values())).keys()))

        for party, tcp_data in party_dict.items():
            category = party_category_dict.get(party)
            if category in ['Left', 'Right', 'Centre']:
                pct = tcp_data[tcp_pair]
                category_values[category].append(pct)

        

        # Compute category averages and add them to the division result
        for category, values in category_values.items():
            avg_pct = sum(values) / len(values)
            div_result[category] = {tcp_pair: avg_pct}

        Preference_flows_dict_with_categories[div] = div_result

    #import pdb;pdb.set_trace()


    ######### Now 2PP values if missing - extend to all categories!

    # STEP 1 — Fill in missing PARTY entries by div
    for div, state in div_to_state_dict.items():
        if div not in Preference_flows_dict_with_categories: # should be redundant!
            Preference_flows_dict_with_categories[div] = {}
        
        for _, row in TPP_by_State[TPP_by_State['StateAb'] == state].iterrows():
            party = row['PartyAb']
            percent_to_alp = row['ALP%']

            if party in ['LP','NP']:
                continue
            
            # Add the party only if not already present
            if party in Div_parties_next_dict_COAL[div]:
                if party not in Preference_flows_dict_with_categories[div]:
                    Preference_flows_dict_with_categories[div][party] = {}
                #import pdb;pdb.set_trace()

                if ('ALP', 'COAL') not in Preference_flows_dict_with_categories[div][party]:
                    Preference_flows_dict_with_categories[div][party][('ALP', 'COAL')] = percent_to_alp





    state_category_values = defaultdict(lambda: defaultdict(list))

    # STEP 2 — Fill in missing PARTY entries by div
    for _, row in TPP_by_State.iterrows():
        state = row['StateAb']
        party = row['PartyAb']
        percent_to_alp = row['ALP%']
        
        category = party_category_dict.get(party)
        if category and category not in ['ALP', 'COAL']:  # Skip major parties
            state_category_values[state][category].append(percent_to_alp)

    # STEP 3 — Fill in missing category averages per div
    for div, state in div_to_state_dict.items():
        #if div not in Preference_flows_dict_with_categories:
        #    continue

        subdict = Preference_flows_dict_with_categories[div]

        for category in ['Left', 'Right', 'Centre']:
            if category not in subdict:
                values = state_category_values[state].get(category)
                if values:
                    median_value = np.median(values)
                    subdict.setdefault(category, {})[('ALP', 'COAL')] = median_value
            else:
                if ('ALP','COAL') not in subdict[category]:
                    values = state_category_values[state].get(category)
                    if values:
                        median_value = np.median(values)

                        subdict[category][('ALP','COAL')] = median_value

    for div in Preference_flows_dict_with_categories.keys():
        for _, row in TPP_nationally.iterrows():

            party = row['PartyAb']
            percent_to_alp = row['ALP%']

            if party in ['LP','NP']:
                    continue
                
            # Add the party only if not already present
            if party in Div_parties_next_dict_COAL[div]:
                if party not in Preference_flows_dict_with_categories[div]:
                    Preference_flows_dict_with_categories[div][party] = {}
                #import pdb;pdb.set_trace()

                if ('ALP', 'COAL') not in Preference_flows_dict_with_categories[div][party]:
                    Preference_flows_dict_with_categories[div][party][('ALP', 'COAL')] = percent_to_alp

                


    #import pdb;pdb.set_trace()
    # reformat into dict of dicts of dfs:


    rows = []

    for division, party_dict in Preference_flows_dict_with_categories.items():
        for party, tcp_dict in party_dict.items():
            for tcp_pair, percent in tcp_dict.items():
                rows.append({
                    'division': division,
                    'tcp_pair': tcp_pair,
                    'party': party,
                    'percent': percent
                })



    # Step 2: Convert to DataFrame and pivot
    long_df = pd.DataFrame(rows)
    #import pdb;pdb.set_trace()

    # Step 3: Pivot to wide format: one row per division, one column per party
    # Step 2: Create the dictionary of wide DataFrames
    result_dict = {}

    # Iterate over the unique divisions
    for div, group in long_df.groupby('division'):
        div_result = pd.DataFrame(index=range(len(TCP_COMBINATION_INDEX)), columns=Preference_flows_dict_with_categories[div].keys())
        
        # Fill the DataFrame with NaN (or 0 if preferred) for all TCP pairs initially
        div_result[:] = None  # Or use np.nan if you prefer NaN

        #if division == 'Melbourne':
        # import pdb;pdb.set_trace()


        # Iterate over each unique tcp_pair for this division
        for tcp_pair, tcp_group in group.groupby('tcp_pair'):

            if tcp_pair not in TCP_COMBINATION_INDEX.keys():
                # convert to party category:
                
                tcp_pair = tuple(sorted([party_category_dict[p] if p != 'Centre' else 'IND' for p in tcp_pair]))



            # Map tcp_pair to the corresponding row index
            row_index = TCP_COMBINATION_INDEX.get(tcp_pair, None)
            
            # If the index is not found in mapping, skip (shouldn't happen if mapping is correct)
            if row_index is None:
                #import pdb;pdb.set_trace()
                continue
            
            # Pivot the tcp_group into a wide format (party as columns, percent as values)
            for _, row in tcp_group.iterrows():
                div_result.at[row_index, row['party']] = row['percent']
        
        # Store the result for this division in the final dictionary
        result_dict[div] = div_result
        if 'ALP' not in result_dict[div]:
            result_dict[div].loc[:,'ALP'] = None
        if 'COAL' not in result_dict[div]:
            result_dict[div].loc[:,'COAL'] = None

        #if div == 'Macnamara':
        #    import pdb;pdb.set_trace()



    # Outlier - Canberra 2013 House had no Right parties!
    if data_year == '2016':
        for div in ['Bean','Canberra','Fenner']:
            result_dict[div].loc[:,'Right'] = None
            result_dict[div].loc[0,'Right'] = result_dict['Bean'].loc[0,'LDP']

        #import pdb;pdb.set_trace()


    #import pdb;pdb.set_trace()

    # now, find non-classic divisions and extrapolate

    for tcp_pair in Non_classic_divs.keys():

        if tcp_pair == ('COAL','COAL'):
            continue

        # first get average of all results with said tcp:
        i = TCP_COMBINATION_INDEX[tcp_pair]

        series_list = []

        for div in Non_classic_divs[tcp_pair]:
            
            series_list.append(result_dict[div].iloc[i]) # curr reusults
        
        tcp_overall = pd.concat(series_list, axis=1)
        tcp_average = tcp_overall.mean(axis=1).dropna()




        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                filtered_update = tcp_average[result_dict[div].columns.intersection(tcp_average.index)]
                result_dict[div].loc[i, filtered_update.index] = filtered_update


        #if tcp_pair == ('COAL','Left'):
        #    import pdb;pdb.set_trace()

    for tcp_pair in [('IND','IND'),('Left','Left'),('LP','NP'),('Right','Right'),('COAL','COAL')]:

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                result_dict[div].loc[i, :] = 50.0

    for tcp_pair in [('IND','Right'),('IND','Left')]:

        # use IND-ALP/COAL i.e. 2 or 1

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                # symmetrically apply!

                source_row = result_dict[div].loc[i - 8, :].copy()

                # Invert each value ONLY if it's not None
                new_row = source_row.apply(lambda x: 100 - x if x is not None else None).astype('object')
                new_row = new_row.where(new_row.notna(), None)
                #new_row = 100 - result_dict[div].loc[i-8, :].copy() # correct indexing!
                if tcp_pair == ('IND', 'Right'):

                    if 'Right' not in new_row.index:
                        #print(div)
                        import pdb;pdb.set_trace()

                    new_row['COAL'] = new_row['Right'] 
                    
                elif tcp_pair == ('IND', 'Left'):
                    new_row['ALP'] = new_row['Left'] 

                # new_row.index.map(lambda x: 'ALP' if x == 'Left' else x)

                #result_dict[div].iloc[i,:] = new_row

                for col in result_dict[div].columns:
                    if new_row[col] is not None:  # Only overwrite if the new value isn't None
                        result_dict[div].at[i, col] = new_row[col]

            #import pdb;pdb.set_trace()
            #4


    for tcp_pair in [('Left','Right')]:
        #import pdb;pdb.set_trace()

        # use ALP/COAL

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():
                new_row = result_dict[div].loc[0, :].copy()

                # ALP gets Left's 
                # COAL gets Right's 
                new_row.loc['ALP'] = new_row['Left']
                new_row.loc['COAL'] = new_row['Right']

                result_dict[div].loc[i, :] = new_row

    for tcp_pair in [('ALP','Right'),('COAL','Left')]:
        #import pdb;pdb.set_trace()

        # use ALP/COAL

        i = TCP_COMBINATION_INDEX[tcp_pair]

        for div in result_dict.keys():
            #import pdb;pdb.set_trace()
            if result_dict[div].loc[i].isnull().all():

                # ALP gets Left's 
                # COAL gets Right's 

                if tcp_pair == ('ALP','Right'):

                    new_row = result_dict[div].loc[0, :].copy()

                    new_row.loc['COAL'] = new_row['Right']

                    result_dict[div].loc[i, :] = new_row


                elif tcp_pair == ('COAL','Left'):

                    source_row = result_dict[div].loc[0, :].copy()
                    # switch order
                    new_row = source_row.apply(lambda x: 100 - x if x is not None else None).astype('object')
                    new_row = new_row.where(new_row.notna(), None)

                    new_row.loc['ALP'] = new_row['Left']              

                    for col in result_dict[div].columns:
                        if new_row[col] is not None:  # Only overwrite if the new value isn't None
                            result_dict[div].at[i, col] = new_row[col]      

                    #import pdb;pdb.set_trace()          


    #import pdb;pdb.set_trace()

    # If RIght is None, get corresponding COAL, smae for left and ALP
    Right_COAL_Right_Preferences = 100 - 58.39 # ON in 2016/9 Maranoa
    Left_COAL_Left_Preferences = 30.5 # 2016/19 Kooyong/Higgins/Melbourne
    Left_ALP_Left_Preferences = 34.0 # 2016/19 ALP/GRN Seats
    Right_COAL_IND_Preferences = 39.8 # for XEN: 2013 Indi/NE preferences
    Right_COAL_Left_Preferences = 40.0 # for 2016, from 2010 Grayndler/Batman
    Right_COAL_COAL_Preferences = 50.0


    for div in result_dict.keys():
        #print(div)
        div = div
        for i, row in result_dict[div].iterrows():
            if pd.isna(row['Right']):
                #import pdb;pdb.set_trace()
                result_dict[div].at[i, 'Right'] = row['COAL']
            if pd.isna(row['Left']):
                #import pdb;pdb.set_trace()
                result_dict[div].at[i, 'Left'] = row['ALP']




            # Fix no Right for RIght-COAL contests:
            if (data_year == '2022') and (i == 6) and (pd.isna(result_dict[div].loc[6,'Right'])):
                result_dict[div] = result_dict[div].copy()
                result_dict[div].at[6,'Right'] = Right_COAL_Right_Preferences

            
            if (data_year == '2022') and (i == 5) and (pd.isna(result_dict[div].loc[5,'Left'])):
                result_dict[div].at[5, 'Left'] = Left_COAL_Left_Preferences

            if (data_year == '2022') and (i == 3) and (pd.isna(result_dict[div].loc[3,'Left'])):
                result_dict[div].at[3, 'Left'] = Left_ALP_Left_Preferences

            if (data_year == '2019') and (i == 3) and (pd.isna(result_dict[div].loc[3,'Left'])):
                result_dict[div].at[3, 'Left'] = Left_ALP_Left_Preferences

            if (data_year == '2016') and (i == 1) and (pd.isna(result_dict[div].loc[1,'Right'])):
                result_dict[div].at[1, 'Right'] = Right_COAL_IND_Preferences

            if (data_year == '2016') and (i == 9) and (pd.isna(result_dict[div].loc[9,'Right'])):
                result_dict[div].at[9, 'Right'] = 1 - Right_COAL_IND_Preferences

            if (data_year == '2016') and (i == 5) and (pd.isna(result_dict[div].loc[5,'Right'])):
                result_dict[div].at[5, 'Right'] = Right_COAL_Left_Preferences

            if (data_year == '2016') and (i == 6) and (pd.isna(result_dict[div].loc[6,'Right'])):
                result_dict[div].at[6, 'Right'] = Right_COAL_Right_Preferences # cheating from future - but quick fix!

            if (data_year == '2016') and (i == 14) and (pd.isna(result_dict[div].loc[14,'Right'])):
                result_dict[div].at[14, 'Right'] = Right_COAL_COAL_Preferences


            # fetch all changes so far!
            row = result_dict[div].loc[i]



            if pd.isna(row['Centre']):
                #import pdb;pdb.set_trace()
                #import pdb;pdb.set_trace()
                if (row['Right']) and (row['Left']):
                    

                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['Left'])/2

                    if pd.isna(result_dict[div].at[i, 'Centre']):
                        import pdb;pdb.set_trace()
                        2
                    
                elif (row['Right']) and (row['ALP']):
                    result_dict[div].at[i, 'Centre'] = (row['Right'] + row['ALP'])/2

                elif (row['Left']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['Left'])/2

                elif (row['ALP']) and (row['COAL']):
                    result_dict[div].at[i, 'Centre'] = (row['COAL'] + row['ALP'])/2

                if pd.isna(result_dict[div].at[i, 'Centre']):
                    print(div, i, row)
                    import pdb;pdb.set_trace()
                    3

                    






            

    for division in result_dict.keys():
        #print(division)
        if result_dict[division]['Centre'].isna().sum():
            import pdb;pdb.set_trace()
            1
        if result_dict[division]['Right'].isna().sum():
            import pdb;pdb.set_trace()
            6
        if result_dict[division]['Left'].isna().sum():
            import pdb;pdb.set_trace()
            5


            

    #import pdb;pdb.set_trace()


    # find COALITION_double_divs_last_year
    DOP_By_Division_curr = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'})[["div_nm","PartyAb"]].drop_duplicates()    
    DOP_By_Division_curr.loc[:,'PartyAb'] = DOP_By_Division_curr.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
    DOP_By_Division_curr = {div: group['PartyAb'].tolist() for div, group in DOP_By_Division_curr.groupby("div_nm")}

    COAL_double_divs_curr = []
    for div in DOP_By_Division_curr.keys():
        if ('LP' in DOP_By_Division_curr[div]) and ('NP' in DOP_By_Division_curr[div]):
            COAL_double_divs_curr.append(div)


    COAL_double_div_transfers = TCP_Preference_Flows.loc[((TCP_Preference_Flows['PartyAb']=='NP') & (TCP_Preference_Flows['TCP_Ab']=='LP')) | ((TCP_Preference_Flows['PartyAb']=='LP') & (TCP_Preference_Flows['TCP_Ab']=='NP')),].rename(columns = {'TransferPercentage':'COAL%'})[['div_nm','COAL%']]
    COAL_double_div_transfers = COAL_double_div_transfers.set_index('div_nm')
    COAL_double_div_transfers.loc['Average'] = COAL_double_div_transfers['COAL%'].mean()
    #import pdb;pdb.set_trace()

    COAL_double_divs_next = []
    for div in Div_parties_next_dict.keys():
        if ('LP' in Div_parties_next_dict[div]) and ('NP' in Div_parties_next_dict[div]):
            COAL_double_divs_next.append(div)



    COAL_list = ['LP','NP','CLP','LNP']


    for div in result_dict.keys():

        # convert all None to nan
        result_dict[div] = result_dict[div].where(pd.notna(result_dict[div] ), None)  # no-op if already NaN
        result_dict[div]  = result_dict[div] .astype(float)


        for i, party in enumerate(Div_parties_next_dict_COAL[div]):

            # new parties
            if party not in result_dict[div].columns:
                result_dict[div].loc[:,party] =  result_dict[div][party_category_dict[party]] # replace with the category!

            else: # fill in missing vals for old parties
                result_dict[div][party] = result_dict[div][party].combine_first(result_dict[div][party_category_dict[party]])
                #import pdb;pdb.set_trace()

            if party == 'COAL':
                PartyAb = Div_parties_next_dict[div][i]
                result_dict[div].loc[:,PartyAb] = result_dict[div]['COAL']


        # Finally, add COAL -> COAL support if COAL_double_div

        if div in COAL_double_divs_next:

            if div in COAL_double_div_transfers.index:
                COAL_Pct = COAL_double_div_transfers.loc[div].iloc[0]
            else:
                COAL_Pct = COAL_double_div_transfers.loc['Average'].iloc[0]

            for col in ['LP','NP']:
                row_indexer = [0,1,5,6]
                result_dict[div].loc[row_indexer,col] = [100 - COAL_Pct, COAL_Pct, COAL_Pct, COAL_Pct]

        #import pdb;pdb.set_trace()

        # fill in ALP/COAL rows with 0


            

            

        # Remove Centre, LeftLRight, "COAL"
        result_dict[div] = result_dict[div].drop(['Left','Centre','Right','COAL'], axis = 1)


    #import pdb;pdb.set_trace()
    cols_to_drop = ['ALP', 'LNP', 'LP', 'NP', 'CLP']

    for div in result_dict.keys():

        df_dropped = result_dict[div].drop(columns=[col for col in cols_to_drop if col in result_dict[div].columns])

        # Now check for NaN values in the remaining columns
        if df_dropped.isna().any().any():
            import pdb;pdb.set_trace()

                

    def expand_and_reorder_duplicate(df, new_names):
        return pd.concat([df[col] for col in new_names], axis=1, keys=new_names)


    

    # get them in Ballot order: 

    # Div_parties_next_dict groups all the INDs together - must use Div_Ballot_Order_next_dict instead to get all INDs in Ballot order
    if next_year != '2025':
        DOP_by_div_full = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1).rename(columns={'DivisionNm': 'div_nm'}).rename(columns = {'DivisionNm':'div_nm'}) 
        DOP_by_div_full = DOP_by_div_full.loc[(DOP_by_div_full['CountNumber']==0) & (DOP_by_div_full['CalculationType'] == 'Preference Count'),['div_nm', 'PartyAb']]
        DOP_by_div_full.loc[:,'PartyAb'] = DOP_by_div_full.loc[:,'PartyAb'].fillna('IND').replace('GVIC','GRN')
        Div_Ballot_Order_next_dict = DOP_by_div_full.groupby('div_nm')['PartyAb'].apply(list).to_dict()
    else:
        Div_Ballot_Order_next_dict = Div_parties_next_dict

    for div in result_dict.keys():

        result_dict[div] = expand_and_reorder_duplicate(result_dict[div], Div_Ballot_Order_next_dict[div]) 


    #import pdb;pdb.set_trace()

    # adjust preferences of Defecting Independents in 2022 to be more favourable to their original parties (~ like ON)

    if data_year == '2022':
        Defected_INDs_Ballot_order = {'Moore':1, 'Monash':3,'Calare':7}
        for div in ['Monash','Moore','Calare']:
            idx = Defected_INDs_Ballot_order[div] - 1
            result_dict[div].iloc[:,idx] = result_dict[div]['ON']

        # Macnamara - Josh Burns Open Preference in 2025
        result_dict['Macnamara'].loc[[5],'ALP'] += 6.6

        #import pdb;pdb.set_trace()


    TCP_pair_category_dict = result_dict



    return TCP_pair_category_dict


def add_randomness_to_proportions(proportions_transferred_to_first, sigma_joint, sigma_ind):


    global_noise = np.random.normal(0, sigma_joint, (15,))  # 15 values for the 15 rows

    # Iterate through each division (electorate)
    for div, proportions_df in proportions_transferred_to_first.items():

        proportions_transferred_to_first[div].fillna(0.5, inplace=True)

        # Add global noise (sigma_joint) to the DataFrame across all rows
        for i in range(15):
            proportions_transferred_to_first[div].iloc[i, :] += global_noise[i]  # Add the global noise for each row
        
        # Add independent noise (sigma_ind) - random noise for each element
        independent_noise = np.random.normal(0, sigma_ind, proportions_df.shape)  # Independent noise for each element
        proportions_transferred_to_first[div] += independent_noise  # Add independent noise element-wise


        # Step 4: Replace values below 5 with 5, and values above 95 with 95
        proportions_transferred_to_first[div] = proportions_transferred_to_first[div].clip(lower=5, upper=95)

    return proportions_transferred_to_first


def distribution_to_top_2(final_simulated_votes, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind):

    electorate_names = Results_dict.keys()
    n_simulations = len(final_simulated_votes['Farrer'])
    
    joint_noise = np.random.normal(0, sigma_joint, size=(n_simulations, 15))

    from collections import defaultdict, Counter

    # For goal 1
    per_electorate_winners = defaultdict(list)  # {'Electorate A': ['ALP', 'ALP', 'LP', ...]} # care about IND1,IND2

    # For goal 2
    per_simulation_winners = [ [] for _ in range(n_simulations) ]  # [[ALP, LP, ...], [LP, IND, ...], ...] # don't care about IND1/IND2 - show for COAL, ALP, GRN, ON, CA, XEN, IND, OTH


    for i, electorate in enumerate(electorate_names):
        #print(electorate)
        #electorate = 'Kennedy'
        # Get the simulations for this electorate
        sims = final_simulated_votes[electorate]  # shape: (10000, n_parties)
        proportions_df = proportions_transferred_to_first[electorate]  # shape: (15, n_parties)

        curr_parties = Results_dict[electorate].columns
        
        for sim_id in range(n_simulations):
            sim_votes = sims[sim_id]  # shape: (n_parties,)
            #print(sim_votes)

            # shape the new proportions_df:
            proportions_with_joint_noise = proportions_df.copy()


            proportions_with_joint_noise.iloc[:] += joint_noise[sim_id][:, np.newaxis] + np.random.normal(0, sigma_ind, size=sim_votes.shape)
            #import pdb;pdb.set_trace()




            
            # Step 1: Get top 2 parties by vote share
            top2_indices = np.argsort(sim_votes)[-2:][::-1]
            #top2_votes = sim_votes[top2_indices]
            top2_parties = [Results_dict[electorate].columns[i] for i in top2_indices]  # depends on how parties are ordered
            
            # Step 2: Determine the category (alphabetically)
            cat_index_pairs = [
                (party_to_category_centered_IND.get(party, 'IND'), idx)
                for party, idx in zip(top2_parties, top2_indices)
            ]
            #print("cat pairs: ", cat_index_pairs)
            #import pdb;pdb.set_trace()


            sorted_cats_with_indices = sorted(cat_index_pairs, key=lambda x: x[0])  # [('GRN', 4), ('LNP', 1)]
            top2_category = tuple(cat for cat, _ in sorted_cats_with_indices)  # ('GRN', 'LNP')
            
            #top2_category = tuple(sorted([party_category_dict.get(p, 'IND') for p in top2_parties])) # Centre if party is IND1,IND2,IND3 etc.
            #first_idx = [i for i in top2_indices if party_to_category[i] == first_cat][0] # FIXXXX
            #row_index = TCP_COMBINATION_INDEX[tuple(sorted(top2_category))]


            #first_cat = top2_category[0]
            first_idx, second_idx = sorted_cats_with_indices[0][1], sorted_cats_with_indices[1][1]
            #print("first_idx: ", first_idx)
            
            # Step 3: Fetch transfer proportions
            row_index = TCP_COMBINATION_INDEX[top2_category]
            transfer_proportions = proportions_df.iloc[row_index].values # shape: (n_parties,)

            #transfer_proportions = transfer_proportions.copy()  # don’t overwrite source!
            #transfer_proportions[top2_indices] = 0
            #print("transfer_proportions: ", transfer_proportions)
            #import pdb;pdb.set_trace()


            # Step 6: Compute 2PP Allocation for the alphabetically-first party 
            non_top2_indices = [i for i in range(len(sim_votes)) if i not in top2_indices]

            # Total transfer to the first top-2 party is the dot product
            transferred_to_P1 = np.dot(sim_votes[non_top2_indices], transfer_proportions[non_top2_indices])

            # Add this to the original primary vote of P1
            P1_2PP_vote = sim_votes[first_idx] + transferred_to_P1/100 # as decimal
            #print("2PP: ", P1_2PP_vote)
            #import pdb;pdb.set_trace()



            if P1_2PP_vote > 0.5:
                winner_idx = first_idx
            else:
                winner_idx = second_idx

            winner_party = curr_parties[winner_idx]

            per_electorate_winners[electorate].append(winner_party) # vertical lists
            per_simulation_winners[sim_id].append(winner_party) # horizontal lists
            #print("winner: ", winner_party)
            #import pdb;pdb.set_trace()


    return (per_electorate_winners,  per_simulation_winners)  # dict[str, dict[str, int]] — useful for percentages ; list[dict[str, int]] or pd.DataFrame



def seat_distribution_plot(per_simulation_winners):


    colour_df = pd.read_csv("Party Colours.csv")

    colour_df.loc[:,'Party'] = colour_df['Party'].str.replace("’", "'", regex=False)

    colors = colour_df.set_index("Party")["Colour"].to_dict()


    party_name_dict = {
        'ALP':'Labor',
        'NP': 'National',
        'COAL': 'Coalition',
        'ON' : 'One Nation',
        'GRN' : 'Greens',
        'HMP' : 'Legalise Cannabis',
        'FFP' : 'Family First',
        'ASP' : 'Shooters Fishers & Farmers',
        'LP' : 'Liberal' ,
        'TOP' : 'Trumpet of Patriots' ,
        'IND' : 'Independent' ,
        'AJP' : 'Animal Justice',
        'SOPA' :'FUSION' ,
        'LDP' : 'Libertarian' ,
        'AUD' : 'Australian Democrats',
        'CEC' : 'Citizens Party',
        'VNS' : 'Victorian Socialists',
        'IMO' : 'HEART',
        'PFP' : "People's First",
        'AUC' : 'Christians',
        'SAL' : 'Socialist Alliance',
        'IAP' : 'Indigenous-Aboriginal',
        'GAP' : 'Great Australian Party',
        'KAP' : "Katter's Australian",
        'CLP' : 'Country Liberal',
        'LNP' : 'Liberal National',
        'XEN' : 'Centre Alliance',
        'IND1' : 'Independent 1',
        'IND2' : 'Independent 2',
        'IND3' : 'Independent 3',
        'IND4' : 'Independent 4',
        'IND5' : 'Independent 5',
        'OTH' : 'Other',
        'IND1Goldstein': 'Ind. Zoe Daniel',
        'IND1Calare': 'Ind. Kate Hook',
        'IND2Calare': 'Ind. Andrew Gee',
        'IND1Kooyong': 'Ind. Monique Ryan',
        'IND3Mackellar': 'Ind. Sophie Scamps',
        'IND1Moore': 'Ind. Ian Goodenough',
        'IND1McPherson': 'Ind. Erchana Murray-Barlett',
        'IND1Wannon': 'Ind. Alex Dyson',
        'IND1Curtin' : 'Ind. Kate Chaney',
        'IND1Indi' : 'Ind. Helen Haines',
        'IND1Clark' : 'Ind. Andrew Wilkie',
        'IND1Moncrieff' : 'Ind. Nicole Arrowsmith',
        'IND2Moore' : 'Ind. Nathan Barton',
        'IND2Bradfield': 'Ind. Nicolette Boele',
        'IND2Berowra' : 'Ind. Tina Smith',
        'IND1Forrest' : 'Ind. Sue Chapman',
        'IND1Sturt' : 'Ind. Verity Cooper',
        'IND1Gilmore' : 'Ind. Kate Dezarnaulds',
        'IND1Casey' : 'Ind. Claire Miles',
        'IND1Franklin' : 'Ind. Peter George',
        'IND2Cowper' : 'Ind. Caz Heise',
        'IND1Fremantle' : 'Ind. Kate Hulett',
        'IND1Fisher' : 'Ind. Kerryn Jones',
        'IND1Grey' : 'Ind. Anita Kuss',
        'IND1Monash' : 'Ind. Russell Broadbent',
        'IND2Monash' : 'Ind. Deb Leonard',
        'IND1Lyne' : 'Ind. Jeremy Miller',
        'IND1Farrer' : 'Ind. Michelle Milthorpe',
        'IND1Deakin' : 'Ind. Jess Ness',
        'IND1Bean' : 'Ind. Jessie Price',
        'IND5Riverina' : 'Ind. Jenny Rolfe',
        'IND1Solomon' : 'Ind. Phil Scott',
        'IND1Flinders' : 'Ind. Ben Smith',
        'IND1Dickson' : 'Ind. Ellie Smith',
        'IND1Fairfax' : 'Ind. Francine Wiig',
        'IND1Fowler' : 'Ind. Dai Le',
        'IND2Wentworth' : 'Ind. Allegra Spender',
        'IND1Groom' : 'Ind. Suzie Holt',
        'IND1Warringah' : 'Ind. Zali Steggall',
        'CYA': 'Trumpet of Patriots',
        'GRPF': "People's First",
        'FFPA': 'Family First',
        'LTP': 'Libertarian'
    }

    site_abbreviation_dict = {
        "Labor": "LAB",
        "Liberal": "LIB",
        "National" : 'NAT',
        "Coalition" : 'COAL',
        "Other" : 'OTH',
        "Greens": "GRN",
        "One Nation" : 'ON',
        'Legalise Cannabis': 'LCP',
        'Family First': 'FFP',
        'Shooters Fishers & Farmers': 'SFF',
        'Trumpet of Patriots': 'TOP',
        'Independent': 'IND',
        'Animal Justice': 'AJP',
        'FUSION': 'FUSN' ,
        'Libertarian':'LBT' ,
        'Australian Democrats': 'AUD',
        'Citizens Party': 'CIT',
        'Victorian Socialists':'VSOC',
        'HEART':'HRT',
        "People's First":'PPF',
        'Christians':'AUC',
        'Socialist Alliance':'SA',
        'Indigenous-Aboriginal': 'IAP',
        'Great Australian Party':'GAP',
        "Katter's Australian": 'KAP',
        'Country Liberal': 'CLP',
        'Liberal National': 'LNP',
        'Centre Alliance': 'CA',
        'Independent 1' : 'IND1',
        'Independent 2' : 'IND2',
        'Independent 3' : 'IND3',
        'Independent 4' : 'IND4',
        'Independent 5' : 'IND5',
    }

    #sample_size = 50

    #all_indices = np.arange(num_simulations)
    #pick = np.random.choice(all_indices, size=min(sample_size, len(all_indices)), replace=False)


    # get df of seat counts per simulation
    coal_parties = {'LP', 'NP', 'LNP', 'CLP'}
    ind_prefixes = {'IND'}  
    parties_out = ['ALP', 'COAL', 'GRN', 'IND', 'OTH'] # ['ALP', 'COAL', 'GRN', 'IND', 'ON', 'KAP', 'XEN', 'OTH']
    
    all_rows = []

    for sim_winners in per_simulation_winners:
        counts = Counter()

        for party in sim_winners:
            if party in coal_parties:
                counts['COAL'] += 1
            elif any(party.startswith(prefix) for prefix in ind_prefixes):
                counts['IND'] += 1
            elif party in parties_out:
                counts[party] += 1
            else:
                counts['OTH'] += 1

        row = [counts.get(party, 0) for party in parties_out]
        all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=parties_out)


    result = [94,43,1,10,2]

    df.rename(columns=party_name_dict, inplace=True)

    party_means = df.mean().to_dict()

    fig = go.Figure()

    for party, seat_count in zip(df.columns, result):
        party_data = df[party].to_numpy()

        x0 = site_abbreviation_dict.get(party, party)

        # 1) Add violin trace
        fig.add_trace(go.Violin(
            x=[x0] * len(party_data),
            y=party_data,
            name=party,
            hoveron='violins',
            hoverinfo='none',  # you are overriding it below with hovertemplate
            #hovertemplate="<b>%{x}</b><br>Seats: %{y}<extra></extra>",
            #hovertemplate="<b>%{x}</b><br>%{y} Seats<br>",
            box_visible=False,
            meanline_visible=True,
            meanline=dict(color='#5d3fd3'),
            line_color='rgba(0,0,0,0)',
            fillcolor=colors.get(party, 'gray'),
            opacity=0.7,
            width=1,
            bandwidth=0.6,
            points=False,   # hide default points
            hovertemplate=f"<b>{party}</b><br>Seats: %{{y}}<extra></extra>",  # Format the hovertext with party name and seat count

        ))

        #sample_size = 100 # min(1000, len(party_data))
        #hover_y = np.random.choice(party_data, sample_size, replace=False)

        sample_size = 100  # total size
        sorted_data = np.sort(party_data)  # sort from smallest to largest
        lowest_20 = sorted_data[:20]
        highest_15 = sorted_data[-15:]
        middle_values = sorted_data[15:-15]
        middle_65 = np.random.choice(middle_values, 65, replace=False)
        hover_y = np.concatenate([lowest_20, highest_15, middle_65])

        fig.add_trace(go.Scatter(
            x=[x0] * sample_size,
            y=hover_y,
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0)',  # fully transparent markers
                size=8
            ),
            hovertemplate=f"<span style='color:{colors.get(party, '#999')}'><b>{party}</b><br>Seats: %{{y}}</span><extra></extra>",
            
        hoverlabel=dict(
            bgcolor='white',  # background of the hover box
            font=dict(color=colors.get(party, 'gray'))  # party-colored text
        ),
            name='',  # hides extra legend entry
            showlegend=False
        ))

         # 1) Large invisible hitbox (captures hover / pointer)
        fig.add_trace(go.Scatter(
            x=[x0],
            y=[seat_count],
            mode='markers',
            name=f'Hitbox::{party}',
            marker=dict(
                size=50,                    # LARGE hit area
                color='rgba(0,0,0,0)'       # fully invisible
            ),
            hoverinfo='skip',               # do not interfere with violin/scatter hovers
            showlegend=False
        ))

        # 2) Actual visible black disc
        fig.add_trace(go.Scatter(
            x=[x0],
            y=[seat_count],
            mode='markers',
            name=f'Disc::{party}',
            marker=dict(
                size=25,
                color='rgba(0,0,0,0.7)'      # 0.7 transparent black
            ),
            hovertemplate=(
                        f"<span style='font-size:16px'><b>{party}</b></span><br>"
                        f"<span style='font-size:0.1px'>&nbsp;</span><br>"
                        f"<span style='font-size:16px'>Result: <b>{seat_count:.0f}<b></span><br>"
                        "<extra></extra>"
                    ),
            showlegend=False
        ))

        
    fig.update_layout(

        title=dict(
        text="Number of Seats Won by Party",
        x=0.95,  # far right
        xanchor='right',
        yanchor='top',
        y=0.95,
        font=dict(size=40)
    ),
        legend=dict(
        font=dict(
            size=30,  # Increase legend font size
        ),
        x=0.9,  # Adjust legend position horizontally
        y=1,  # Adjust legend position vertically
        traceorder='normal',
        bgcolor='rgba(255, 255, 255, 0)',  # Transparent background for legend
        bordercolor='rgba(0,0,0,0)',  # no border
        itemwidth=30,  # Set width of legend items (if needed for large labels)
    ),
        #yaxis_title="First Preference Vote in 1,000 Simulations",
        violinmode="group",
        violingap=0.1,
        margin=dict(l=30, r=30, t=80, b=100),
        xaxis_tickangle=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    

    fig.update_yaxes(
        tick0=0,
        dtick=5,
        tickformat='.0f',
        tickfont=dict(size=20),  # make y-axis numbers larger
        range=[0, df.max().max() + 5]
    )

    fig.add_shape(
        type="line",
        x0=0.01, x1=0.9,  # full width in paper coordinates
        y0=76, y1=76,
        line=dict(
            color="purple",  
            width=1,
            dash="dash"  # dashed line
        ),
        xref='paper',  # use paper coordinates for x-axis (0 to 1)
        yref='y'       # use y-axis coordinates
    )

    fig.add_annotation(
        text="<b>Seats Won<b>",      
        x=-0.02,                # slightly left of y-axis
        y=1.05,                  # center of y-axis (in paper coords)
        xref="paper",
        yref="paper",
        showarrow=False,
        #textangle=-90,          # vertical text
        font=dict(size=16),
        align="center"
    )

    fig.add_annotation(
        text="<b>Majority Government<b>: 76 Seats",
        x=0.7,
        y=76.5,
        xref="paper",
        yref="y",
        showarrow=False,
        yshift=10,  # push it a little above the line
        font=dict(
            size=18,
            color="purple"  # match your line color
        ),
        align="center"
    )

    for i, party in enumerate(df.columns):
        fig.add_annotation(
            x=i,  # directly on the center of the violin
            y=party_means[party],  # place at the mean seat number
            text=f"<b>{party_means[party]:.1f}</b>",  # BOLD + correct closing tag
            showarrow=False,
            yanchor="middle",
            xanchor="center",  # center horizontally too
            font=dict(
                size=24,  # slightly larger for visibility
                color="white" # always white
            ),
            xref="x",  # map x to x-axis categories
            yref="y"   # map y to seat numbers
        )

    fig.show()

    #fig_json = fig.to_json()
    #from datetime import datetime
    #current_date = datetime.now().strftime("%Y%m%d")

    #file_name = f"2025SeatDistribution-{current_date}.json"

    #with open(f"{file_name}", "w") as f:
    #    f.write(fig_json)


    return 1


def export_simulations_to_csv(sim_dict, Results_dict, output_dir="data/generated/2025_Election_First_Preferences"):
    """
    sim_dict: dict with keys = electorate names (str), values = (10000 x n_parties) arrays
    party_names: list of party names in column order (length = n_parties)
    output_dir: directory to save each electorate's CSV file
    """

    os.makedirs(output_dir, exist_ok=True)

    for div, sim_array in sim_dict.items():
        n_sims, n_parties = sim_array.shape
        df = pd.DataFrame(sim_array, columns=Results_dict[div].columns)
        df['sim_no'] = df.index
        long_df = df.melt(id_vars='sim_no', var_name='Party', value_name='First Preference Votes')

        csv_path = os.path.join(output_dir, f"{div.replace(' ', '_')}.csv")
        long_df.to_csv(csv_path, index=False)

    print(f"✅ Exported {len(sim_dict)} electorates to {output_dir}/")


def export_winner_proportions_to_csv(winner_dict, output_dir="data/generated/2025_Election_Winner_Proportions"):
    """
    winner_dict: dict with keys = electorate names (str), values = list of winner party names (length = n_sims)
    output_dir: directory to save each electorate's CSV file

    For each electorate, outputs a CSV with columns:
    - Party
    - Win Percentage
    """

    os.makedirs(output_dir, exist_ok=True)

    for div, winners in winner_dict.items():
        counts = Counter(winners)
        total = sum(counts.values())
        proportions = {party: count / total for party, count in counts.items()}

        df = pd.DataFrame({
            'Party': list(proportions.keys()),
            'Win Percentage': list(proportions.values())
        })

        csv_path = os.path.join(output_dir, f"{div.replace(' ', '_')}.csv")
        df.to_csv(csv_path, index=False)

    print(f"✅ Exported winner proportions for {len(winner_dict)} electorates to {output_dir}/")


def export_simulation_seat_counts(per_simulation_winners, output_csv="2025_Simulated_seat_counts.csv"):
    """
    Export seat counts per simulation, reduced to 8 meta-party categories.
    Each row is a simulation; each column is the seat count for a meta-party.
    """
    # Define party groupings
    coal_parties = {'LP', 'NP', 'LNP', 'CLP'}
    ind_prefixes = {'IND'}  
    parties_out = ['ALP', 'COAL', 'GRN', 'IND', 'OTH'] # ['ALP', 'COAL', 'GRN', 'IND', 'ON', 'KAP', 'XEN', 'OTH']
    
    all_rows = []

    for sim_winners in per_simulation_winners:
        counts = Counter()

        for party in sim_winners:
            if party in coal_parties:
                counts['COAL'] += 1
            elif any(party.startswith(prefix) for prefix in ind_prefixes):
                counts['IND'] += 1
            elif party in parties_out:
                counts[party] += 1
            else:
                counts['OTH'] += 1

        row = [counts.get(party, 0) for party in parties_out]
        all_rows.append(row)

    import pdb;pdb.set_trace()


    # Create DataFrame and export
    df = pd.DataFrame(all_rows, columns=parties_out)
    df.to_csv(output_csv, index=False)

    print(f"✅ Exported seat counts for {len(per_simulation_winners)} simulations to {output_csv}/")


def obtain_Winner_table(per_simulation_winners, Results_dict, n_simulations):

        # Initialize counters
    alp_majority_count = 0
    coal_majority_count = 0
    alp_lead_count = 0
    coal_lead_count = 0
    hung_count = 0

    alp_avg = 0
    coal_avg = 0
    grn_avg = 0
    ind_avg = 0

    alp_seat_list = []
    coal_seat_list = []


    # Loop over simulations
    for i, sim_results in enumerate(per_simulation_winners):
        alp_seats = sum(1 for winner in sim_results if winner == ALP_NAME)
        coal_seats = sum(1 for winner in sim_results if winner in COAL_PARTIES)
        ind_seats = sum(1 for winner in sim_results if winner in IND_PARTIES)
        grn_seats = sum(1 for winner in sim_results if winner == 'GRN')

        coal_seat_list.append(coal_seats)
        alp_seat_list.append(alp_seats)

        alp_avg += alp_seats
        coal_avg += coal_seats
        grn_avg += grn_seats
        ind_avg += ind_seats
        
        if alp_seats >= 76:
            alp_majority_count += 1
        elif coal_seats >= 76:
            coal_majority_count += 1
        elif alp_seats > coal_seats:
            alp_lead_count += 1
        elif coal_seats > alp_seats:
            coal_lead_count += 1
        else:
            hung_count += 1

    Winner_table = pd.DataFrame( per_simulation_winners, columns=Results_dict.keys()).T
    Winner_table["favourite"] = Winner_table.mode(axis=1)[0]
    Winner_table.loc["ALP_count"] = (Winner_table == "ALP").sum(axis=0) 
    Winner_table.loc["COAL_count"] = (Winner_table.isin(['LP','NP','LNP',"CLP"])).sum(axis=0)
    Winner_table.loc["GRN_count"] = (Winner_table =='GRN').sum(axis=0)
    Winner_table.loc["IND_count"] = Winner_table.apply(lambda s: s.str.startswith("IND")).sum(axis=0)
    Winner_table.loc["OTH_count"] = 150 - Winner_table.iloc[-4:,].sum()

    # Compute probabilities
    total_simulations = len(per_simulation_winners)
    alp_prob = alp_majority_count / total_simulations
    coal_prob = coal_majority_count / total_simulations
    alp_lead_prob = alp_lead_count / total_simulations
    coal_lead_prob = coal_lead_count / total_simulations
    hung_prob = hung_count / total_simulations


    #print(Winner_table)
    #print("Average seats: ALP", alp_avg/n_simulations, "COAL:", coal_avg/n_simulations, "GRN:", grn_avg/n_simulations, "IND:", ind_avg/n_simulations)
    #print(f"ALP majority probability: {alp_prob:.3f}")
    #print(f"COAL majority probability: {coal_prob:.3f}")

    Average_seats_df = np.round(pd.Series({'ALP': alp_avg, 'COAL': coal_avg, 'GRN': grn_avg, 'IND': ind_avg}) / n_simulations,2)
    Average_seats_df['OTH'] = 150 - Average_seats_df.sum()
    

    return Winner_table, Average_seats_df


def run_model(election_year = '2025', n_simulations=1000, ref_col = 'COAL', forced_polling_average = [], export_simulation_csvs = 0):

    #forced_polling_average = []#[0.3159,0.3443,0.1196,0.0618,0.0299] # TOP shifts -0.136502 -0.233139 -0.235509 -0.22493  1.079983 -0.249903 # [0.26, 0.35, 0.115, 0.14, 0.03]
    #forced_polling_average.append(1-np.array(forced_polling_average).sum()) if forced_polling_average else [] # ensure sum-to-1-constraint

    ########################################################################## ESTIMATE STATE ALR COVARIANCE MATRICES, POLLING PRECISION ###################################################################

    Day = 90

    estimate_National_ALR_Covariance_Matrices(ref_col = 'COAL', Day = Day, plot_histogram = False)

    # save Cov matrices for election-to-election swing and for polling error
    for Type in ['Election_swing','Polling']:
        estimate_state_polling_deviations_from_national(ref_col = 'COAL',Type = Type, remove_election_year = 0)


    ########################################################################## IMPUTE POLLING FROM MISSING STATES ###################################################################

    get_imputed_state_deviations_from_national(ref_col = 'COAL')

    ########################################################################## SIMULATE 2025 FIRST PREFERENCE VOTES ###################################################################

    scale_state_polling_precision() # generate csv of scaled state polling average precisions



    if election_year == '2025':
        w,alpha,v,s,beta = 0.9 if not forced_polling_average else 1,22,0.15,0.75,0.5 # can set w to 1 ensure the forced result is the true result
    if election_year == '2022':
        w, alpha, s, v, beta =0.95,30,0.9,0.5,0.4
    elif election_year == '2019':
        w, alpha, s, v, beta = 0.9,20,0.65,0.2,0.6
    elif election_year == '2016':
        w, alpha, s, v, beta = 0.85,20,0.7,0.05,0.8

    final_simulated_votes, Results_dict =  First_Preference_Model_Simulation(election_year = election_year, Day = Day, ref_col='COAL', w = w, alpha = alpha, v = v, s = s, beta = beta, n_simulations = n_simulations, forced_polling_average = forced_polling_average)

    ########################################################################## SIMULATE 2025 DISTRIBUTION OF PREFERENCES ######################################################################

    party_category_dict = make_party_category_dict()
    party_to_category_centered_IND = {k: ('IND' if v == 'Centre' else v) for k, v in party_category_dict.items()}

    proportions_transferred_to_first = make_TCP_pair_category_dict(election_year = election_year, party_category_dict=party_category_dict)

    sigma_joint, sigma_ind = 4.2, 0.5

    per_electorate_winners, per_simulation_winners = distribution_to_top_2(final_simulated_votes, proportions_transferred_to_first, Results_dict, party_to_category_centered_IND, sigma_joint, sigma_ind)

    ##################### Results and exporting data ##############################

    Winner_table, Average_seats = obtain_Winner_table(per_simulation_winners, Results_dict, n_simulations)

    # open plotly plot of seat distribution
    seat_distribution_plot(per_simulation_winners=per_simulation_winners)



    if export_simulation_csvs:
        export_simulations_to_csv(final_simulated_votes, Results_dict, output_dir="data/generated/2025_Election_First_Preferences")
        export_winner_proportions_to_csv(per_electorate_winners, output_dir="data/generated/2025_Election_Winner_Proportions")
        export_simulation_seat_counts(per_simulation_winners, output_csv="2025_Simulated_seat_counts.csv")


    return {
    "Winner_table": Winner_table,
    "Average_seats": Average_seats,
    "per_electorate_winners": per_electorate_winners,
    "per_simulation_winners": per_simulation_winners,
    "Results_dict": Results_dict,
}


if __name__ == "__main__":
    outputs = run_model(election_year = '2025', n_simulations=1000, ref_col = 'COAL', forced_polling_average = [], export_simulation_csvs = 0)

    print(outputs["Average_seats"])



