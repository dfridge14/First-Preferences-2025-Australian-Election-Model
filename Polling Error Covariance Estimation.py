import pandas as pd
import numpy as np
from itertools import product
import os,time
from datetime import datetime
from pathlib import Path

from pingouin import multivariate_normality


# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):

    print("\a")  # Rings the system bell
    os.system('echo -e "\\a"')  # Extra bell command for reliability
    os.system('tput bel')  # This forces the terminal to beep

    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler



base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

Type = 'Election_swing' # Election_swing Polling
ref_col = 'COAL'

# manage removal of data point
curr_election_year = '2025'
remove_election_year = 1



def extend_corr_matrix_to_5x5(corr_matrix_3x3, var4, var5, cov_matrix_3x3, ref_col):
    # Extends to 5x5 corr and covMs through rickety imputation, checking for positive definiteness and adding column names, reutrning 5x5 df


    R5 = np.pad(corr_matrix_3x3, ((0, 2), (0, 2)), mode='constant', constant_values=0)

    ON_UAPP_CORRELATION = 0.5
    ALP_GRN_CORRELATION_LOSS = 0.3

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



for curr_election_year in ['2016','2019','2022','2025']:

    election_year_to_remove = curr_election_year if remove_election_year else ' '


    for Type in ['Polling','Election_swing']:

        if Type == 'Polling':
            # 1. Correlation estimate of 3x3 - 6 GRW 2007-2022 + 14 State Elections + 15 State Results

            

            FederalStatePolls = pd.read_csv("StatePollingWeightedAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
            StateElectionsPolls = pd.read_csv("StateElectionsWeightedPollingAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
            OldFederalElectionPollingAverage = pd.read_csv("OldFederalElectionPollingAverage.csv", index_col = None).iloc[:,:5].set_index('Election')
            NationalElectionPollingAveragesGRW = pd.read_csv("NationalElectionPollingAveragesGRW.csv", index_col = None).set_index('Election')


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

            if remove_election_year:
                CAGO_ALR_swings_centered = CAGO_ALR_swings_centered.loc[~(CAGO_ALR_swings_centered.index.str.startswith(election_year_to_remove)),]

            corr_matrix = np.corrcoef(CAGO_ALR_swings_centered.values, rowvar=False)
            print(corr_matrix)
            #import pdb;pdb.set_trace()



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

            weights = np.array([0.3,0.3,0.3,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,0.6,0.6,0.5,0.5,0.5,0.5,0.5])
            #import pdb;pdb.set_trace()

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


            if remove_election_year:
                CAGO_ALR_swings_centered = Election_swings_ALR_centered_after_1996.loc[~(Election_swings_ALR_centered_after_1996.index.str.startswith(election_year_to_remove)),]

            corr_matrix_Elec = np.corrcoef(Election_swings_ALR_centered_after_1996.values, rowvar=False)
            print(corr_matrix_Elec)

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



            #import pdb;pdb.set_trace()


        CAGO_Variance_estimation_swings_centered = CAGO_Variance_estimation_swings_centered if Type == 'Polling' else Election_swings_ALR_for_Var_centered


        def weighted_nanstd(data, weights):
            # estimates variances while respecting np.nan for GRN before 2004
            weighted_std = []
            #import pdb;pdb.set_trace()

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
        print(cov_matrix)

        #import pdb;pdb.set_trace()

        # test for multivariate normality (vs t)
        result = multivariate_normality(CAGO_Variance_estimation_swings_centered.dropna(), alpha=0.05)
        print(result)

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import scipy.stats as stats

        CAGO_Variance_estimation_swings_centered.dropna().hist(bins=20, figsize=(10, 4), density=True)
        plt.suptitle("Histogram of ALR Swings (w/ KDE overlay)", y=1.02)
        plt.tight_layout()
        #plt.show()
        for col in CAGO_Variance_estimation_swings_centered.dropna().columns:
            sns.kdeplot(CAGO_Variance_estimation_swings_centered.dropna()[col], fill=True, label=col)
        plt.title("KDEs of ALR Swings")
        plt.legend()
        #plt.show()
        # Clean dataframe
        clean_df = CAGO_Variance_estimation_swings_centered.dropna()

        # passes test - looks reasonably normal!

        # Simulate 1000 samples from multivariate normal in ALR space
        n_samples = 100000
        alr_samples = np.random.multivariate_normal(mean=np.zeros(3), cov=cov_matrix, size=n_samples)
        alr_samples_df = pd.DataFrame(alr_samples, columns = ['ALP','GRN','OTH'])

        alr_prediction_df = CAGO_Variance_Polling_ALR.loc['2022'] + alr_samples_df if Type == 'Polling' else Election_results_curr_ALR.loc['2019'] + alr_samples_df


        #import pdb;pdb.set_trace()


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


        # Convert each ALR sample back to the 4D probability space
        prob_samples = alr_to_simplex_vectorized(alr_prediction_df, ref_col=ref_col)
        stds = np.sqrt(prob_samples.var())
        print("stds", stds)


        if remove_election_year:

            Election_swing_ON_UAPP_ALR_variances = {'2016':0.0751,'2019':0.0904,'2022':0.0901,'2025':0.0944}
            Polling_swing_ON_UAPP_ALR_variances = {'2016':0.0136,'2019':0.0139,'2022':0.0309,'2025':0.0343}

            CAGO_cols = [p for p in ['COAL','ALP','GRN','OTH'] if p != ref_col]

            if Type == 'Polling':
                ON_UAPP_variances = Polling_swing_ON_UAPP_ALR_variances[curr_election_year]

                if curr_election_year == '2016':
                    CAGO_Polling_2016_corr, CAGO_Polling_2016_cov = corr_matrix,cov_matrix
                    CAGO_Polling_2016_cov = pd.DataFrame(CAGO_Polling_2016_cov, index = CAGO_cols, columns = CAGO_cols)
                    CAGO_Polling_2016_cov.to_csv("PollingErrorALRCovarianceNational2016.csv", index = True)

                elif curr_election_year == '2019':
                    CAGO_Polling_2019_corr, CAGO_Polling_2019_cov = extend_corr_matrix_to_5x5(corr_matrix, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                    CAGO_Polling_2019_cov.to_csv("PollingErrorALRCovarianceNational2019.csv", index = True)


                elif curr_election_year == '2022':
                    CAGO_Polling_2022_corr, CAGO_Polling_2022_cov = extend_corr_matrix_to_5x5(corr_matrix, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                    CAGO_Polling_2022_cov.to_csv("PollingErrorALRCovarianceNational2022.csv", index = True)


                elif curr_election_year == '2025':
                    CAGO_Polling_2025_corr, CAGO_Polling_2025_cov = extend_corr_matrix_to_5x5(corr_matrix, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                    CAGO_Polling_2025_cov.to_csv("PollingErrorALRCovarianceNational2025.csv", index = True)

            elif Type == 'Election_swing':
                ON_UAPP_variances = Election_swing_ON_UAPP_ALR_variances[curr_election_year]


                if curr_election_year == '2016':
                    CAGO_swing_2016_corr, CAGO_swing_2016_cov = corr_matrix,cov_matrix
                    CAGO_swing_2016_cov = pd.DataFrame(CAGO_swing_2016_cov, index = CAGO_cols, columns = CAGO_cols)
                    CAGO_swing_2016_cov.to_csv("ElectionErrorALRCovarianceNational2016.csv", index = True)


                elif curr_election_year == '2019':
                    CAGO_swing_2019_corr, CAGO_swing_2019_cov = extend_corr_matrix_to_5x5(corr_matrix_Elec, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                    CAGO_swing_2019_cov.to_csv("ElectionErrorALRCovarianceNational2019.csv", index = True)


                elif curr_election_year == '2022':
                    CAGO_swing_2022_corr, CAGO_swing_2022_cov = extend_corr_matrix_to_5x5(corr_matrix_Elec, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                    CAGO_swing_2022_cov.to_csv("ElectionErrorALRCovarianceNational2022.csv", index = True)


                elif curr_election_year == '2025':
                    CAGO_swing_2025_corr, CAGO_swing_2025_cov = extend_corr_matrix_to_5x5(corr_matrix_Elec, ON_UAPP_variances, ON_UAPP_variances, cov_matrix, ref_col)
                    CAGO_swing_2025_cov.to_csv("ElectionErrorALRCovarianceNational2025.csv", index = True)



import pdb;pdb.set_trace()



#        ALP REF  COAL REF, 100000 simulations - almost the same! GOOD! Use COAL as REF due to largest party simplicity
#COAL    0.019945 0.019009
#GRN     0.015708 0.014136
#OTH     0.024270 0.024097
#ALP     0.026623 0.026647

# For election_swing:
#ALP     0.053119
#GRN     0.019539
#OTH     0.051396
#COAL    0.055078


# Now, ON and UAPP adjustment - corr_matrix:
# OTH-ON;UAPP - high, like GRN: 0.4;0.4 = GRNxOTH
# GRN-ON;UAPP = GRN-OTH - 0.3
# ALP-ON;UAPP = ALP-OTH - 0.3
# ON-UAPP = 0.5


# Variances (computed in Excel ON_UAPP_Polling_Estimates):
#                   ALL   UNTIL 2016 UNTIL 2019  UNTIL 2022
# Election_swing: 0.0944 | 0.0751   | 0.0904    | 0.0901
# Polling_error:  0.0322 | 0.0136   | 0.0139    | 0.0316

# Covariance Matrices in ALR:
# 2016: CAGO Swing; CAGO Polling
# 2019: CAGOUO Swing; CAGOUO Polling
# 2022: CAGOUO Swing; CAGOUO Polling
# 2025: COGOUO Swing; CAGOUO Polling

CAGO_Polling_2016_corr = np.array([[ 1.        , -0.09068868,  0.11310681],
       [-0.09068868,  1.        ,  0.44217864],
       [ 0.11310681,  0.44217864,  1.        ]])

CAGO_Polling_2016_cov = np.array([[ 0.00979676, -0.00141595,  0.00178047],
       [-0.00141595,  0.02488332,  0.01109316],
       [ 0.00178047,  0.01109316,  0.02529339]])

CAGO_swing_2016_corr = np.array([[1.         ,0.40035235, 0.34365332]
 [0.40035235, 1.         ,0.09447634]
 [0.34365332, 0.09447634, 1.        ]])

CAGO_swing_2016_cov = np.array([[0.08033778, 0.0281828 , 0.04173239],
       [0.0281828 , 0.06168277, 0.01005305],
       [0.04173239, 0.01005305, 0.18356321]])

CAGO_swing_2019_corr = np.array([[ 1.        ,  0.40035235,  0.04365332,  0.04365332,  0.34365332],
       [ 0.40035235,  1.        , -0.20552366, -0.20552366,  0.09447634],
       [ 0.04365332, -0.20552366,  1.        ,  0.5       ,  0.09447634],
       [ 0.04365332, -0.20552366,  0.5       ,  1.        ,  0.09447634],
       [ 0.34365332,  0.09447634,  0.09447634,  0.09447634,  1.        ]])

CAGO_swing_2019_cov = np.array([[ 0.07467742,  0.0247121 ,  0.00357877,  0.00357877,  0.04144836],
       [ 0.0247121 ,  0.05102055, -0.01392694, -0.01392694,  0.00941863],
       [ 0.00357877, -0.01392694,  0.09      ,  0.045     ,  0.0125094 ],
       [ 0.00357877, -0.01392694,  0.045     ,  0.09      ,  0.0125094 ],
       [ 0.04144836,  0.00941863,  0.0125094 ,  0.0125094 ,  0.19479789]])

CAGO_Polling_2019_corr = np.array([[ 1.        , -0.03469821, -0.28525785, -0.28525785,  0.01474215],
       [-0.03469821,  1.        ,  0.14963755,  0.14963755,  0.44963755],
       [-0.28525785,  0.14963755,  1.        ,  0.5       ,  0.44963755],
       [-0.28525785,  0.14963755,  0.5       ,  1.        ,  0.44963755],
       [ 0.01474215,  0.44963755,  0.44963755,  0.44963755,  1.        ]])

CAGO_Polling_2019_cov = np.array([[ 0.01033446, -0.00050932, -0.00341892, -0.00341892,  0.00027032],
       [-0.00050932,  0.02084902,  0.00254736,  0.00254736,  0.01171074],
       [-0.00341892,  0.00254736,  0.0139    ,  0.00695   ,  0.009562  ],
       [-0.00341892,  0.00254736,  0.00695   ,  0.0139    ,  0.009562  ],
       [ 0.00027032,  0.01171074,  0.009562  ,  0.009562  ,  0.03253552]])


CAGO_Polling_2022_corr = np.array([[ 1.        , -0.0842303 , -0.19954114, -0.19954114,  0.10045886],
       [-0.0842303 ,  1.        ,  0.11277465,  0.11277465,  0.41277465],
       [-0.19954114,  0.11277465,  1.        ,  0.5       ,  0.41277465],
       [-0.19954114,  0.11277465,  0.5       ,  1.        ,  0.41277465],
       [ 0.10045886,  0.41277465,  0.41277465,  0.41277465,  1.        ]])

CAGO_Polling_2022_cov = np.array([[ 0.01110245, -0.0012253 , -0.00373754, -0.00373754,  0.00183895],
       [-0.0012253 ,  0.01906027,  0.0027677 ,  0.0027677 ,  0.00990035],
       [-0.00373754,  0.0027677 ,  0.0316    ,  0.0158    ,  0.01274763],
       [-0.00373754,  0.0027677 ,  0.0158    ,  0.0316    ,  0.01274763],
       [ 0.00183895,  0.00990035,  0.01274763,  0.01274763,  0.03018185]])

CAGO_swing_2022_corr = 1

CAGO_swing_2022_cov = 1

# develop systematic way of calculating these!



# eigenvalue check 
# np.all(np.linalg.eigvalsh(CAGO_swing_2019_cov) > 0)

CAGO_Polling_2022_corr, CAGO_Polling_2022_cov = extend_corr_matrix_to_5x5(corr_matrix_Elec, 0.09, 0.09, cov_matrix, ref_col)

import pdb;pdb.set_trace()
