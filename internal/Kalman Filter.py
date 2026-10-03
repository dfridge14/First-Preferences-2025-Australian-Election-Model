import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
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


# Simulated data
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

SAMPLE_ERROR_SCALING_FACTOR = 2
KALMAN_START = -150 # days before election
START_OF_AVERAGE = -120
LOW_DENSITY_SS = 500


election_date_num = {'2007': 1141,'2010':1002, '2013':1113, '2016':1028, '2019':1050, '2022':1099,'2025':1078}

Grand_CAGO_polling_df = pd.read_csv('CAGO_Polling_for_Kalman_filter.csv')
party_cols = {'COAL':'blue','ALP':'red','GRN':'green','ON':'orange','OTH':'grey'}
sigma2_drift_prior = {'COAL':1e-5,'ALP':1e-5,'GRN':1e-6,'ON':1e-5}
# sigma_drift_prior = {'COAL':0.004,'ALP':0.004,'GRN':0.002,'OTH':0.003,'ON':0.003,'UAPP':0.003,'TOP':0.003} # larger relative uncertainty for minor parties ON/UAPP/TOP



def kalman_filter(full_days, poll_dict, sigma2, m0, P0):
    m = m0
    P = P0

    m_series = []
    P_series = []

    for day in full_days:

        # --- Predict ---
        P = P + sigma2

        # --- Update only if poll exists ---
        if day in poll_dict:
            y_obs, var_obs = poll_dict[day]

            K = P / (P + var_obs)
            m = m + K * (y_obs - m)
            P = (1 - K) * P

        m_series.append(m)
        P_series.append(P)

    return np.array(m_series), np.array(P_series)


def kalman_loglike(y, t, poll_var, sigma2, m0, P0):
    m = m0
    P = P0
    ll = 0
    current_day = t[0]

    for i in range(1, len(y)):
        dt = t[i] - current_day
        P = P + dt * sigma2
        current_day = t[i]

        S = P + poll_var[i]   # total variance
        ll += -0.5 * (np.log(S) + (y[i] - m)**2 / S)

        K = P / S
        m = m + K * (y[i] - m)
        P = (1 - K) * P

    return ll

def by_election_polling_averages(Grand_CAGO_polling_df,election_date_num):

    By_election_data = pd.read_csv("Federal_by_election_data.csv")

    Byelection_CAGO_polling_list = []

    for by_election in By_election_data.loc[By_election_data['prev_election'].astype(int) >= 2013,'div_nm']:
        # recalibrate Day_index to match days before by_election
        By_election_days, prev_election = By_election_data.loc[By_election_data['div_nm']==by_election,['days_since_election','prev_election']].iloc[0].astype(int)
        next_election = str(prev_election + 3)
        By_election_day_index = By_election_days - election_date_num[next_election]

        polls_before_by_election = Grand_CAGO_polling_df.loc[(Grand_CAGO_polling_df['Election']==next_election),]
        polls_before_by_election = polls_before_by_election.loc[polls_before_by_election['Day_index']<=By_election_day_index,]
        polls_before_by_election.loc[:,'Day_index'] = polls_before_by_election['Day_index']-By_election_day_index


        # too few days since election for 2014 Griffith by-election
        if by_election == 'Griffith' and prev_election==2013:
            Additional_row = pd.DataFrame([[-154,0.456,0.334,0.087,0.123,np.nan,next_election]], columns=polls_before_by_election.columns)
            polls_before_by_election = pd.concat([Additional_row,polls_before_by_election])
        
        polls_before_by_election.loc[:,'Election'] = f'{by_election}{next_election}'
        polls_before_by_election = polls_before_by_election[polls_before_by_election['Day_index']!=0] # exclude polls released on day of by_election
        Byelection_CAGO_polling_list.append(polls_before_by_election)


    Byelection_CAGO_polling_df = pd.concat(Byelection_CAGO_polling_list)

    return Byelection_CAGO_polling_df.reset_index(drop=True)



def obtain_kalman_polling_averages(Grand_CAGO_polling_df, to_plot = False, is_Farrer_2026=False):

    # Obtain polling averages for each election with data in Grand_CAGO_polling_df, 



    polling_averages = []

    for election in Grand_CAGO_polling_df['Election'].unique():

        print(election)
        #if election in ['1987','1990','1993','1996']: # ['2025','1996','1993','1990','1987','TAS2025','TAS2024','TAS2021','TAS2018','SA2026','SA2014','SA2010','NSW2015','QLD2012','QLD2009','WA2021','WA2013']:   
            #continue
        
        Polling_curr = Grand_CAGO_polling_df.loc[Grand_CAGO_polling_df['Election']==election,]
        has_result = (Polling_curr['Day_index'] == 0).any()
        if has_result:
            Result_row = Polling_curr.loc[Polling_curr['Day_index'] == 0].rename(columns = {'Sample size':'Density'}).iloc[-1,]



        prior_row = Polling_curr.loc[Polling_curr['Day_index'] < KALMAN_START].iloc[:,1:5].tail(3).mean() # lightly smoothed average

        party_results = {}

        plt.figure(figsize=(10,5))

        for party in ['COAL','ALP','GRN'] if not is_Farrer_2026 else ['COAL','ALP','GRN','ON'] :

            # only model polling for 150 days prior to election
            df_curr = Polling_curr.loc[(Polling_curr['Day_index'] >= KALMAN_START) & (Polling_curr['Day_index'] < 0)].copy()

            # Compute precisions
            p = df_curr[party]
            df_curr["Sample size"] = df_curr["Sample size"].fillna(1000) # where sample size not found
            df_curr["precision"] = df_curr["Sample size"] / (p * (1 - p))

            # Aggregate by day (precision-weighted mean & new variance)
            agg_polls = df_curr.groupby("Day_index").agg(
                vote_share_weighted=(party, lambda x: np.sum(x * df_curr.loc[x.index, "precision"]) / np.sum(df_curr.loc[x.index, "precision"])),
                total_precision=("precision", "sum")  # Sum of precisions
            ).reset_index()
            # Convert precision back to standard deviation
            agg_polls["poll_sd"] = np.sqrt(1 / agg_polls["total_precision"])

        
            y = agg_polls["vote_share_weighted"].values  # ALR-transformed
            t = agg_polls["Day_index"].values
            poll_var = (SAMPLE_ERROR_SCALING_FACTOR * agg_polls["poll_sd"].values) ** 2

            full_days = np.arange(KALMAN_START, 0)
            poll_dict = {
                day: (y[i], poll_var[i])
                for i, day in enumerate(t)
            }
            m0 = prior_row[party]
            P0 = 0.01**2


            res = minimize(lambda x: -kalman_loglike(y, t, poll_var, np.exp(x[0]), m0, P0),x0=[np.log(0.005)])
            sigma2_est = np.exp(res.x[0])

            sigma2_est = sigma2_drift_prior[party]
            #print(sigma2_est)

            #import pdb; pdb.set_trace()

            m_series, P_series = kalman_filter(full_days, poll_dict, sigma2_est, m0, P0)

            mask = full_days >= START_OF_AVERAGE
            days_kept = full_days[mask]
            means_kept = m_series[mask]

            party_results[party] = np.round(means_kept,3)


            # plot

            plt.plot(full_days, m_series, label='Estimated vote share', color=party_cols[party])

            # 95% confidence interval
            plt.fill_between(
                full_days,
                m_series - 1.96 * np.sqrt(P_series),
                m_series + 1.96 * np.sqrt(P_series),
                color=party_cols[party],
                alpha=0.1
            )
            plt.scatter(df_curr['Day_index'].to_list(),df_curr[party].to_list(),color=party_cols[party], marker='.')
            
            if has_result:
                plt.scatter([0],[Result_row[party]],color=party_cols[party])

        party_df = pd.DataFrame({"COAL": party_results['COAL'],"ALP": party_results['ALP'],"GRN": party_results['GRN']})
        if is_Farrer_2026: 
            party_df['ON'] = party_results['ON']
        party_df['OTH'] = 1 - party_df.sum(axis=1)
        party_df['Day_index'] = days_kept
        party_df['Election'] = election

        window = 30
        # Create a Series with full_days as index, 0 as default
        poll_sizes_series = pd.Series(0, index=full_days)
        # Fill in the actual polls from Polling_curr
        # Polling_curr['Day_index'] has the day, 'Sample size' the weight
        Polling_trimmed = Polling_curr.loc[(Polling_curr['Day_index']>= KALMAN_START) & (Polling_curr['Day_index'] < 0),]
        poll_sizes_per_day = Polling_trimmed.groupby('Day_index')['Sample size'].sum()
        poll_sizes_series.loc[poll_sizes_per_day.index] = poll_sizes_per_day.values
        # Compute rolling sum for the look-back window
        density_series = poll_sizes_series.rolling(window=window, min_periods=1).sum()
        # Keep only the days you are storing means for (days_kept)
        density_kept = density_series.loc[days_kept].values
        density_kept[density_kept == 0] = LOW_DENSITY_SS
        party_df['Density'] = density_kept

        plt.xlabel("Days since starting point")
        plt.ylabel("Vote share")
        plt.title("Vote share trend with uncertainty (Kalman filter)")
        plt.legend()
        if to_plot:
            plt.show()


        # add result to bottom of df
        if has_result:
            party_df.loc[len(party_df)] = Result_row
        polling_averages.append(party_df)

    plt.close('all') 

    Global_daily_polling_averages = pd.concat(polling_averages, ignore_index=True)

    return Global_daily_polling_averages

def df_to_alr(df, ref_col):
    """Convert a DataFrame of proportions to ALR, dropping ref column."""
    df_alr = np.log(df.drop(columns=[ref_col]).div(df[ref_col], axis=0))
    return df_alr


df = obtain_kalman_polling_averages(pd.read_csv('Farrer_Byelection_Polling_for_Kalman_filter.csv'), is_Farrer_2026=True, to_plot=True)
import pdb; pdb.set_trace()

Global_daily_polling_averages = obtain_kalman_polling_averages(Grand_CAGO_polling_df)


alr_df = df_to_alr(Global_daily_polling_averages[['COAL','ALP','GRN','OTH']],ref_col='COAL')
alr_df['Election'] = Global_daily_polling_averages['Election']
alr_df['Day_index'] = Global_daily_polling_averages['Day_index']
alr_df['Density'] = Global_daily_polling_averages['Density']



# get alr_swing to election result
parties = ['ALP', 'GRN', 'OTH']
df = alr_df.copy()

election_day_vals = (df[df['Day_index'] == 0].set_index('Election')[parties]) # Get election-day result per election
df[parties] = df[parties] - df['Election'].map(election_day_vals.to_dict(orient='index')).apply(pd.Series) # Subtract from each row via mapping



def exp_decay(t, sigma_start, sigma_end, k): 
    return sigma_end + (sigma_start - sigma_end) * np.exp(k * t) 

def plot_fit_exponential_decay_curve(alr_df, to_plot=False):


    plt.figure(figsize=(10,5))

    error_per_party_df = {}

    for party in ['ALP','GRN','OTH']:

        pivot = alr_df.pivot(index='Election', columns='Day_index', values=party)
        pivot_den = alr_df.pivot(index='Election', columns='Day_index', values='Density')

        # subtract election day
        diff = pivot.sub(pivot[0], axis=0)

        # keep -120 to -1
        diff = diff.loc[:, range(-120, 0)]

        if party == 'GRN':
            diff = diff.loc[~diff.index.isin(['1987','1990','1993','1996']),]

        error_per_party_df[party] = diff
        density = pivot_den.loc[:, range(-120, 0)]
        weights = np.sqrt(density)

        # weighted mean
        mu_w = (weights * diff).sum(axis=0) / weights.sum(axis=0)

        # weighted variance
        var_w = (weights * (diff - mu_w)**2).sum(axis=0) / weights.sum(axis=0)

        std_w = np.sqrt(var_w)


        # 🔴 scatter ALL points (each election, each day)
        x = np.tile(diff.columns.values, diff.shape[0])   # repeat days
        y = diff.values.flatten()                         # all values

        #plt.scatter(x, y, s=1, alpha=0.2, color=party_cols[party])
        # 🔵 overlay mean line (what you already had)
        std_series = std_w #diff.std(axis=0) # *1.96 std_w #* 1.96 # 

        from scipy.optimize import curve_fit
        
        day_index = np.array(std_series.index) 
        var_values = np.array(std_series) 
        # Initial guesses 
        
        sigma_start0 = var_values[0]
        sigma_end0 = var_values[-1] 
        k0 = 0.02 # small positive 
        popt, _ = curve_fit(exp_decay, day_index, var_values, p0=[sigma_start0, sigma_end0, k0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]) ) 
        

        def plot_fitted_curve(popt_list):

            for popt in popt_list:
                fitted_curve = exp_decay(day_index, *popt)
                
                plt.plot(day_index, fitted_curve, '-', label='Parametric sigmoid fit')
                plt.plot(std_series.index, std_series.values, color=party_cols[party], label=party)
            if to_plot:
                plt.show()
        
        plot_fitted_curve([popt])
        #import pdb; pdb.set_trace()
        #plt.plot(std_series.index, diff.std(axis=0).values, color=party_cols[party], label=party)
        #plt.plot(std_series.index, -std_series.values, color=party_cols[party], label=party)

    plt.xlabel("Days since starting point")
    plt.ylabel("Vote share")
    plt.title("Vote share trend with uncertainty (Kalman filter)")
    plt.legend()
    if to_plot:
        plt.show()

    return error_per_party_df, weights, day_index

error_per_party_df, weights, day_index = plot_fit_exponential_decay_curve(alr_df, to_plot=False)

OPT_exp_decay_parameters = {'OTH':[0.18562381, 0.28067983, 0.0507943],'GRN':[0.15782239, 0.24219055, 0.02555061],'ALP':[0.09403604, 0.20026747, 0.02794699], 'ON': [0.1957, 0.2800, 0.02555]} # sigma_start, sigma_end, k

excluded_grn = ['1987','1990','1993','1996'] 

def estimate_Corr_CovM(error_per_party_df, weights, day_index):

    days = error_per_party_df['ALP'].columns 

    # Initialize storage for covariance matrices for each day
    covm_time = {t: pd.DataFrame(index=parties, columns=parties, dtype=float) for t in days}
    corr_time = {t: pd.DataFrame(index=parties, columns=parties, dtype=float) for t in days}

    for t in days:
        # build a temporary matrix of values for this day
        vals = {}
        wts = {}
        for p in parties:
            vals[p] = error_per_party_df[p].loc[:, t].copy()
            wts = weights.loc[:, t].copy()
            
            # Exclude GRN rows if needed
            if p == 'GRN':
                if '1987' in vals[p].index:
                    vals[p] = vals[p].drop(index=excluded_grn)
                wts = wts.drop(index=excluded_grn)
        
        # Stack into matrix: rows=elections, cols=parties
        mat_vals = pd.DataFrame(vals)
        mat_wts = pd.DataFrame(wts)

        # Weighted mean for each column
        mu_w = (mat_vals.mul(mat_wts.sum(axis=1), axis=0)).sum(axis=0) / mat_wts.sum().iloc[0]
        
        # Centered values
        mat_centered = mat_vals - mu_w
        
        # Weighted covariance
        cov = pd.DataFrame(index=parties, columns=parties, dtype=float)
        for i in parties:
            for j in parties:
                cov.loc[i,j] = ((mat_centered[i] * mat_centered[j]).mul(mat_wts.sum(axis=1), axis=0)).sum() / mat_wts.sum().iloc[0]
        
        # Store covariance
        covm_time[t] = cov
        
        # Compute correlation for monitoring
        std_i = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_i, std_i)
        corr_time[t] = pd.DataFrame(corr, index=parties, columns=parties)

    corr_ALP_GRN = [corr_time[t].loc['ALP', 'GRN'] for t in days]
    corr_ALP_OTH = [corr_time[t].loc['ALP', 'OTH'] for t in days]
    corr_GRN_OTH = [corr_time[t].loc['GRN', 'OTH'] for t in days]

    # Optional: put in a DataFrame for convenience
    corr_df = pd.DataFrame({
        'ALP-GRN': corr_ALP_GRN,
        'ALP-OTH': corr_ALP_OTH,
        'GRN-OTH': corr_GRN_OTH
    }, index=days)

    # Plot
    plt.figure(figsize=(10,5))
    for col in corr_df.columns:
        plt.plot(corr_df.index, corr_df[col], label=col)
    plt.xlabel('Days before election')
    plt.ylabel('Weighted correlation')
    plt.title('Weighted correlation between parties over time')
    plt.legend()
    plt.grid(True)
    plt.show()


    t_vals = np.array(day_index)  # e.g. [-120, ..., -1]

    sigma_t = {}

    for p in parties:
        sigma_start, sigma_end, k = OPT_exp_decay_parameters[p]
        sigma_t[p] = exp_decay(t_vals, sigma_start, sigma_end, k)

    sigma_df = pd.DataFrame(sigma_t, index=t_vals)

    R = corr_time[-1]

    sigma_end_vec = sigma_df.loc[-1].values
    D_end = np.diag(sigma_end_vec)

    Cov_end = D_end @ R.values @ D_end

    Cov_end = pd.DataFrame(Cov_end, index=parties, columns=parties)


    covm_model_time = {}

    sigma_end = sigma_df.loc[-1]

    for t in t_vals:
        scale = sigma_df.loc[t] / sigma_end  # vector of ratios
        
        # outer product gives scaling matrix
        scale_mat = np.outer(scale, scale)
        
        cov_t = Cov_end.values * scale_mat
        
        covm_model_time[t] = pd.DataFrame(cov_t, index=parties, columns=parties)

    return R,Cov_end,covm_model_time

corr_final,covm_final,covm_time_model = estimate_Corr_CovM(error_per_party_df, weights, day_index)
import pdb; pdb.set_trace()

final_polling_error_alr_df = pd.DataFrame({party: df.iloc[:, -1]for party, df in error_per_party_df.items()})
final_polling_error_alr_df['sqrt_density'] = weights[-1]

final_polling_error_alr_df.to_csv('Final_nat_poll_alr_error.csv')

################ update by_election priors


Byelection_CAGO_polling_df = by_election_polling_averages(Grand_CAGO_polling_df,election_date_num)
Byelection_daily_polling_averages = obtain_kalman_polling_averages(Byelection_CAGO_polling_df)

Byelection_daily_polling_averages.to_csv("Byelection_daily_polling_averages.csv", index=False)

import pdb; pdb.set_trace()
