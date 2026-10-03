# 1. National polling covms
# 2. Seat residual covms
# 3. Seat polling covms

# In all, estimate one each election by removing 2016,2019,2022,2025, and add in ON

import numpy as np
import pandas as pd
import os

import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler

os.chdir('/home/dania-freidgeim/Australian Election')


BY_ELECTION_SCALING = 1.3 # 1.18055492771072
BY_ELECTION_ON_VAR = 0.0383
IND_RESIDUAL_VAR = 0.1

PARTIES = ['ALP', 'GRN', 'OTH']
PARTIES_ON = ['ALP', 'GRN', 'ON', 'OTH']


# 1. Naitonal polling COVMs

Final_nat_poll_alr_error = pd.read_csv("Final_nat_poll_alr_error.csv", index_col='Election')

def compute_cov_end(poll_error_df, excluded_grn = ['1987','1990','1993','1996']):

    parties = poll_error_df.columns[:3].to_list()
    mat_vals = poll_error_df.loc[:, parties].copy()
    w = poll_error_df['sqrt_density'].copy()

    # ✔ correct GRN handling (apply to BOTH consistently)
    if excluded_grn is not None:
        mat_vals = mat_vals.drop(index=excluded_grn, errors='ignore')
        w = w.drop(index=excluded_grn, errors='ignore')

    # align after filtering (safe + necessary)
    mat_vals, w = mat_vals.align(w, join="inner", axis=0)

    w = w.astype(float)

    # weighted mean
    mu = (mat_vals.mul(w, axis=0)).sum(axis=0) / w.sum()

    mat_centered = mat_vals - mu

    # weighted covariance
    cov = pd.DataFrame(index=parties, columns=parties, dtype=float)

    for i in parties:
        for j in parties:
            cov.loc[i, j] = (mat_centered[i] * mat_centered[j] * w).sum() / w.sum()

    return cov

Nat_poll_covm_dict = {}
for election_year in ['2016','2019','2022','2025','Byelection']:
    CovM = compute_cov_end(Final_nat_poll_alr_error.drop(election_year, errors='ignore')) # don't drop any for byelection
    Nat_poll_covm_dict[election_year] = CovM * (BY_ELECTION_SCALING)**(election_year=='Byelection')


# Electorate residual covms
# separate for with/without ON

Electorate_residual_alr_error_CAGO = pd.read_csv('Electorate_residual_alr_error_CAGO.csv')
Electorate_residual_alr_error_CAGO.set_index(Electorate_residual_alr_error_CAGO.columns[0], inplace=True)
Electorate_residual_alr_error_CAGO.index.name = "div_nm"
Electorate_residual_alr_error_ON = pd.read_csv('Electorate_residual_alr_error_CAGOO.csv')
Electorate_residual_alr_error_ON.set_index(Electorate_residual_alr_error_ON.columns[0], inplace=True)
Electorate_residual_alr_error_ON.index.name = "div_nm"

Electorate_residual_covm_dict = {}
Electorate_residual_covm_dict_ON = {}

for election_year in ['2016','2019','2022','2025','Byelection']:

    # remove current election year 
    Electorate_residual_CAGO_curr = Electorate_residual_alr_error_CAGO.loc[Electorate_residual_alr_error_CAGO['Election'].astype(str)!=election_year,['ALP','GRN','OTH']]
    CovM = np.cov(Electorate_residual_CAGO_curr, rowvar=False)
    R = np.corrcoef(Electorate_residual_CAGO_curr, rowvar=False) * (BY_ELECTION_SCALING)**(election_year=='Byelection')

    Electorate_residual_ON_curr = Electorate_residual_alr_error_ON.loc[Electorate_residual_alr_error_ON['Election'].astype(str)!=election_year,['ALP','GRN','ON','OTH']]
    CovM_CAGOO = np.cov(Electorate_residual_ON_curr, rowvar=False) * (BY_ELECTION_SCALING)**(election_year=='Byelection')

    Electorate_residual_covm_dict[election_year] =  pd.DataFrame(CovM, index=PARTIES, columns=PARTIES)
    Electorate_residual_covm_dict_ON[election_year] = pd.DataFrame(CovM_CAGOO, index=PARTIES_ON, columns=PARTIES_ON) 



# seat poll CovMs


Seat_poll_alr_error = pd.read_csv('Seat_poll_alr_error.csv')
Seat_poll_covm_dict = {}

for election_year in ['2016','2019','2022','2025','Byelection']:
    Seat_poll_alr_error_curr = Seat_poll_alr_error.loc[Seat_poll_alr_error['Election_year'].astype(str)!=election_year,['ALP','GRN','OTH']]
    CovM = np.cov(Seat_poll_alr_error_curr, rowvar=False) * (BY_ELECTION_SCALING)**(election_year=='Byelection')
    Seat_poll_covm_dict[election_year] = pd.DataFrame(CovM, index=PARTIES, columns=PARTIES)


import pickle

with open("covms_all.pkl", "wb") as f:
    pickle.dump({
        "Seat_poll_covm_dict": Seat_poll_covm_dict,
        "Electorate_residual_covm_dict": Electorate_residual_covm_dict,
        "Nat_poll_covm_dict": Nat_poll_covm_dict
    }, f)






# Residuals: comparison w.r.t ON-included divs
CovM_ON_incl = np.cov(Electorate_residual_alr_error_ON.drop(columns=['Election']), rowvar=False)
CovM_CAGO_ON_subset = np.cov(Electorate_residual_alr_error_CAGO.reset_index().set_index(['Election','div_nm']).loc[Electorate_residual_alr_error_ON.reset_index().set_index(['Election','div_nm']).index], rowvar=False)
CovM_CAGO = np.cov(Electorate_residual_alr_error_CAGO.drop(columns=['Election']), rowvar=False)

R4 =  np.corrcoef(Electorate_residual_alr_error_ON.drop(columns=['Election']), rowvar=False)

ON_inflation_ratio = CovM_ON_incl[-1,-1]/CovM_CAGO_ON_subset[-1,-1]

# Electorate_residuals_stds
Residual_vars = np.array(np.diag(CovM_CAGO)[:-1].tolist() + [0.11831] + [(np.diag(CovM_CAGO)[-1] * ON_inflation_ratio)])

def build_5x5_covm(R4, vars_4, ind_var=0.01):
    # Expand 4x4 correlation to 5x5 (insert IND at index 3 with 0 correlation, 1 on diagonal)
    R5 = np.insert(np.insert(np.array(R4), 3, 0, axis=1), 3, 0, axis=0)
    R5[3, 3] = 1.0
    
    # Reconstruct covariance with the inserted IND standard deviation
    S = np.diag(np.insert(np.array(np.sqrt(vars_4)), 3, np.sqrt(ind_var)))
    C5 = S @ R5 @ S
    
    # Enforce positive-definiteness via vectorized eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(C5)
    C5_clean = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    
    return pd.DataFrame(C5_clean, index=['ALP', 'GRN', 'ON', 'IND', 'OTH'], columns=['ALP', 'GRN', 'ON', 'IND', 'OTH'])

Electorate_residual_covm = build_5x5_covm(R4, Residual_vars, ind_var=IND_RESIDUAL_VAR)



def expand_3x3_to_5x5_with_on_corr(cov_3x3, var_on, var_ind, proxy_corr_4x4):
    """
    Expands a 3x3 covariance matrix (ALP, GRN, OTH) into a 5x5 matrix.
    Injects IND as an orthogonal variable, but maps ON covariances
    using the provided 4x4 proxy correlation matrix.
    """
    parties = ['ALP', 'GRN', 'ON', 'IND', 'OTH']
    
    # 1. Initialize a 5x5 zero matrix
    C5 = np.zeros((5, 5))
    
    # 2. Map the base 3x3 covariances (ALP, GRN, OTH map to indices 0, 1, 4)
    base_idx = [0, 1, 4]
    cov_array = cov_3x3.values if isinstance(cov_3x3, pd.DataFrame) else cov_3x3
    
    for i in range(3):
        for j in range(3):
            C5[base_idx[i], base_idx[j]] = cov_array[i, j]
            
    # 3. Inject the specific variances for ON and IND (indices 2, 3)
    C5[2, 2] = var_on
    C5[3, 3] = var_ind
    
    # 4. Calculate standard deviations needed for the ON covariance math
    std_alp = np.sqrt(C5[0, 0])
    std_grn = np.sqrt(C5[1, 1])
    std_oth = np.sqrt(C5[4, 4])
    std_on = np.sqrt(var_on)
    
    # 5. Extract specific ON correlations from the 4x4 proxy array 
    # In proxy_corr_4x4, indices are: 0=ALP, 1=GRN, 2=ON, 3=OTH
    corr_alp_on = proxy_corr_4x4[0, 2]  # 0.18435345
    corr_grn_on = proxy_corr_4x4[1, 2]  # 0.21531544
    corr_oth_on = proxy_corr_4x4[3, 2]  # 0.10319365
    
    # 6. Convert correlations to covariances and inject them into row/col 2
    C5[0, 2] = C5[2, 0] = corr_alp_on * std_alp * std_on  # ALP & ON
    C5[1, 2] = C5[2, 1] = corr_grn_on * std_grn * std_on  # GRN & ON
    C5[4, 2] = C5[2, 4] = corr_oth_on * std_oth * std_on  # OTH & ON
    
    # Note: IND (row/col 3) remains perfectly 0.0 except for its variance on the diagonal
    
    # 7. Guarantee Positive-Definiteness (clipping any numerical artifacts)
    eigvals, eigvecs = np.linalg.eigh(C5)
    C5_clean = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    
    return pd.DataFrame(C5_clean, index=parties, columns=parties)



# National Polling Matrix (Tamed IND)
Nat_poll_covm = expand_3x3_to_5x5_with_on_corr(
    cov_3x3=compute_cov_end(Final_nat_poll_alr_error), 
    var_on=0.0383, 
    var_ind=0.0001,
    proxy_corr_4x4 = R4
)

# Seat Polling Matrix
Seat_poll_covm = expand_3x3_to_5x5_with_on_corr(
    cov_3x3= np.cov(Seat_poll_alr_error_curr, rowvar=False), 
    var_on=0.230878, 
    var_ind=0.070821,
    proxy_corr_4x4 = R4
)


import pickle
Farrer_byelection_covms = {
    'Nat_poll_covm': Nat_poll_covm*BY_ELECTION_SCALING,
    'Seat_poll_covm': Seat_poll_covm*BY_ELECTION_SCALING,
    'Electorate_residual_covm': Electorate_residual_covm*BY_ELECTION_SCALING
}

with open('Farrer_byelection_covms.pkl', 'wb') as file:
    pickle.dump(Farrer_byelection_covms, file)

print("Covariance matrices successfully saved!")


import pdb; pdb.set_trace()



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

# Seat polls for ON/IND

Seat_poll_year_dict = {}

election_years = ['2013','2016','2019','2022','2025']
election_date_num = {'2013':1113, '2016':1028, '2019':1050, '2022':1099,'2025':1078}


for election_year in election_years:

    Seat_polls = pd.read_csv(f"SeatPolls{election_year}Formatted.csv", index_col=None)
    Seat_polls.loc[:,'Days since last election'] = election_date_num[election_year] - Seat_polls.loc[:,'Days since last election']
    Seat_polls.rename(columns={'Days since last election':'Days before election'}, inplace=True)

    Seat_poll_year_dict[election_year] = Seat_polls



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





# find seat polls containing either ON or IND
Seat_polls_with_IND = []
Seat_polls_with_ON = []
Seat_polls_with_UAPP = []
for election_year in ['2016','2019','2022','2025','Byelection']:

    if election_year != 'Byelection':
        Results_dict = get_results_df(election_year, to_Fundamentals=False)[1]
    else:
        By_elections_Results = pd.read_csv('By-election_results.csv').iloc[:,:4]
        By_elections_Results.loc[By_elections_Results['div_nm']=='Batman','div_nm'] = 'Cooper'
        By_elections_Results = By_elections_Results[By_elections_Results['byelection_year']>=2013]
        Results_dict =  {div: g.assign(FP=g['FirstPreferencePercent']).groupby('PartyAb')['FP'].sum().to_frame().T.reset_index(drop=True) for div, g in By_elections_Results.groupby('div_nm')}
            



    polls = Seat_poll_year_dict[election_year]

    polls_with_IND = polls[polls['IND'] > 0].replace(0, np.nan)
    # Identify existing columns among COAL and NAT to sum for the 'COAL' total
    coal_cols = [c for c in ['COAL', 'NAT'] if c in polls_with_IND.columns]

    # Perform the assign and groupby only on columns that actually exist
    COAL_IND_grouped = polls_with_IND.assign(COAL=polls_with_IND[coal_cols].sum(axis=1)).groupby('Electorate')[['COAL', 'IND']].mean()
    COAL_IND_grouped['election_year']=election_year

    prev_totals = pd.DataFrame({
        elec: {
            'COAL_Result': df[[c for c in ['LP', 'NP', 'LNP', 'CLP'] if c in df.columns]].sum().sum(),
            'IND_Result': df[[c for c in df.columns if str(c).startswith('IND')]].sum().sum()
        } for elec, df in Results_dict.items()
    }).T

    Seat_polls_with_IND.append(COAL_IND_grouped.join(prev_totals))

    if election_year != '2016':

        # 1. Create a fallback Series of 0s that perfectly matches your DataFrame's index
        z = pd.Series(0, index=polls.index)
        # 2. Use .get() with the fallback, and bitwise operators (&, |) for the logic
        polls_with_ON = polls.loc[(polls.get('ON', z) > 0)]
        COAL_ON_grouped = polls_with_ON.assign(COAL=polls_with_ON[coal_cols].sum(axis=1)).groupby('Electorate')[['COAL', 'ON']].mean()
        COAL_ON_grouped['election_year']=election_year

        prev_totals = pd.DataFrame({
            elec: {
                'COAL_Result': df[[c for c in ['LP', 'NP', 'LNP', 'CLP'] if c in df.columns]].sum().sum(),
                'ON_Result': df['ON'].sum().sum()
            } for elec, df in Results_dict.items() if 'ON' in df.columns
        }).T


        Seat_polls_with_ON.append(COAL_ON_grouped.join(prev_totals))

        right_party = 'UAPP' if  'UAPP' in polls.columns else None

        if right_party is not None:
            polls_with_Rights = polls.loc[(polls.get('ON', z) == 0) & ((polls.get('UAPP', z) > 0) | (polls.get('PUP', z) > 0))]
            COAL_Right_grouped = polls_with_Rights.assign(COAL=polls_with_Rights[coal_cols].sum(axis=1)).groupby('Electorate')[['COAL', right_party]].mean()
            COAL_Right_grouped['election_year']=election_year
            prev_totals = pd.DataFrame({
                elec: {
                    'COAL_Result': df[[c for c in ['LP', 'NP', 'LNP', 'CLP'] if c in df.columns]].sum().sum(),
                    'UAPP_Result': df['UAPP'].sum().sum()
                } for elec, df in Results_dict.items() if 'UAPP' in df.columns
            }).T
            Seat_polls_with_UAPP.append(COAL_Right_grouped.join(prev_totals))

Seat_polls_with_UAPP_df = pd.concat(Seat_polls_with_UAPP).reset_index()
Seat_polls_with_ON_df = pd.concat(Seat_polls_with_ON).reset_index()
Seat_polls_with_IND_df = pd.concat(Seat_polls_with_IND).reset_index()


Seat_polls_with_IND_df = Seat_polls_with_IND_df[Seat_polls_with_IND_df['IND']>=0.07]
Seat_polls_with_IND_df = Seat_polls_with_IND_df.loc[~Seat_polls_with_IND_df['Electorate'].isin(['Lyons','Eden-Monaro','Canberra']),]
Seat_polls_with_IND_df = Seat_polls_with_IND_df.assign(
    Poll_ALR=np.log(Seat_polls_with_IND_df['IND'] / Seat_polls_with_IND_df['COAL']),
    Result_ALR=np.log(Seat_polls_with_IND_df['IND_Result'] / Seat_polls_with_IND_df['COAL_Result'])
)
Seat_polls_with_IND_df.loc[:,'ALR_error'] = Seat_polls_with_IND_df['Result_ALR'] - Seat_polls_with_IND_df['Poll_ALR']

Seat_polls_with_ON_df = Seat_polls_with_ON_df[Seat_polls_with_ON_df['ON']>0.05]
Seat_polls_with_ON_df = Seat_polls_with_ON_df.assign(
    Poll_ALR=np.log(Seat_polls_with_ON_df['ON'] / Seat_polls_with_ON_df['COAL']),
    Result_ALR=np.log(Seat_polls_with_ON_df['ON_Result'] / Seat_polls_with_ON_df['COAL_Result'])
)
Seat_polls_with_ON_df.loc[:,'ALR_error'] = Seat_polls_with_ON_df['Result_ALR'] - Seat_polls_with_ON_df['Poll_ALR']

print(Seat_polls_with_IND_df['ALR_error'].var())
print(Seat_polls_with_ON_df['ALR_error'].var())

print(Seat_polls_with_IND_df['ALR_error'].var()*BY_ELECTION_SCALING)
print(Seat_polls_with_ON_df['ALR_error'].var()*BY_ELECTION_SCALING)
import pdb; pdb.set_trace()




    

# extend to 4x4 matrix with ON

def extend_corr_matrix_to_4x4(corr_matrix_3x3, var_on, cov_matrix_3x3, ref_col):

    # expand 3x3 -> 4x4
    R4 = np.pad(corr_matrix_3x3, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    # heuristic constants (keep consistent with your prior calibration)
    ALP_GRN_CORRELATION_LOSS = 0.3
    COAL_CORRELATION_LOSS = 0.1
    ON_REF_CORRELATION = 0.5

    if ref_col == 'COAL':

        # ON correlations vs ALP / GRN / OTH (assumed index structure preserved)
        R4[0, 3] = R4[0, 2] - ALP_GRN_CORRELATION_LOSS
        R4[3, 0] = R4[0, 2] - ALP_GRN_CORRELATION_LOSS

        R4[1, 3] = R4[1, 2] - ALP_GRN_CORRELATION_LOSS
        R4[3, 1] = R4[1, 2] - ALP_GRN_CORRELATION_LOSS

        R4[2, 3] = R4[1, 2]
        R4[3, 2] = R4[1, 2]

        R4[3, 3] = 1

    elif ref_col == 'ALP':

        # ON correlations vs COAL / GRN / OTH
        R4[0, 3] = R4[0, 2] - COAL_CORRELATION_LOSS
        R4[3, 0] = R4[0, 2] - COAL_CORRELATION_LOSS

        R4[1, 3] = R4[1, 2] - ALP_GRN_CORRELATION_LOSS
        R4[3, 1] = R4[1, 2] - ALP_GRN_CORRELATION_LOSS

        R4[2, 3] = R4[1, 2]
        R4[3, 2] = R4[1, 2]

        R4[3, 3] = 1

    else:
        raise ValueError("Unsupported ref_col")

    # ---- covariance reconstruction ----
    std_devs = np.zeros(4)
    std_devs[:3] = np.sqrt(np.diag(cov_matrix_3x3))
    std_devs[3] = np.sqrt(var_on)

    S = np.diag(std_devs)
    C4 = S @ R4 @ S

    # ordering (same idea as before: keep your preferred final layout)
    idx = [0, 1, 3, 2]  # move ON into position 3 or adjust as needed

    R4 = R4[np.ix_(idx, idx)]
    C4 = C4[np.ix_(idx, idx)]

    cols = [p for p in ['COAL', 'ALP', 'GRN', 'ON'] if p != ref_col]

    R4 = pd.DataFrame(R4, index=cols, columns=cols)
    C4 = pd.DataFrame(C4, index=cols, columns=cols)

    # positive definiteness check
    assert np.all(np.linalg.eigvalsh(C4) > 0)

    return R4, C4