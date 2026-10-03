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



def compute_polling_avg(poll_df,
    prior_df,
    div,
    election_year,
    missing_party = None,
    ):
    # compute CAGO_row
    minor_cols = [p for p in prior_df.columns if p not in ['ALP','COAL','LP','NP','LNP','CLP','GRN']]
    # if GRN vote share not reported, adjust GRN/OTH split (Longman, Bennelong, 2019 Flinders, Cowan, 2022 Mackellar, Norht Sydney, 2025 Bullwinkel)
    if (poll_df['GRN']==0).any():
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


     # reallocate minor parties - first according to poll proportions, then for remaining according to prior proportions
    # 1. Polled party 'OTH' reallocation
    poll_minors = poll_df.reindex(columns = minor_cols+ ['OTH']).astype(float).fillna(0)
    poll_minor_means = poll_minors.replace(0, np.nan).mean() # mean shares of any minor parties polled
    poll_minor_split = poll_minor_means.dropna()
    poll_minor_split['OTH'] = CAGO_row['Other'] - poll_minor_split.loc[poll_minor_split.index !='OTH'].sum() # remaining OTH vote after minor polled party averages accounted for (safe when polled minors differ from poll to poll)
    
    if set(minor_cols).issubset(poll_minor_split.index):
        if not np.isclose(poll_minor_split['OTH'],0):
            import pdb; pdb.set_trace() # just check in case fails 
        poll_minor_split = poll_minor_split.loc[poll_minor_split.clip(lower = 0) > 0] # remove 'OTH' if OTH == 0
    
    import pdb; pdb.set_trace()

    if 'OTH' in poll_minor_split and poll_minor_split['OTH'] <= 0:
        poll_minor_split = calculate_rigorous_split(poll_minors, CAGO_row['Other'])

    poll_minor_split = poll_minor_split.loc[poll_minor_split.clip(lower = 0) > 0] # remove 'OTH' if OTH == 0

    seat_polling_avg = pd.concat([CAGO_row.drop('Other'), poll_minor_split]) if not poll_minor_split.empty else CAGO_row.rename({'Other':'OTH'})

    if ('NAT' in poll_df.columns) and (poll_df['NAT'].sum()> 0):
        NAT_polled = poll_df['NAT'].replace(0,np.nan).mean()
        seat_polling_avg['NAT'] = NAT_polled
        seat_polling_avg['COAL']-=NAT_polled



    return seat_polling_avg



election_years = ['2016','2019','2022','2025','Byelection']

Seat_poll_year_dict = {}
ELECTION_DATE_NUM = {'2013':1113, '2016':1028, '2019':1050, '2022':1099,'2025':1078}


for election_year in election_years:

    if election_year != 'Byelection':

        Seat_polls = pd.read_csv(f"SeatPolls{election_year}Formatted.csv", index_col=None)
        Seat_polls.loc[:,'Days since last election'] = ELECTION_DATE_NUM[election_year] - Seat_polls.loc[:,'Days since last election']
        Seat_polls.rename(columns={'Days since last election':'Days before election'}, inplace=True)

        Seat_poll_year_dict[election_year] = Seat_polls
    else:

        # add by_election seat polls/results to dicts
        By_election_data = pd.read_csv("Federal_by_election_data.csv")
        Seat_polls =  pd.read_csv(f"SeatPollsByelectionFormatted.csv", index_col=None)

        # format first col into days before byelection via merge with By_election_data:
        By_election_data['byelection_date'] = pd.to_datetime(By_election_data['byelection_date'],format='%d.%m.%y')
        By_election_data['byelection_year'] = By_election_data['byelection_date'].dt.year
        By_election_data.loc[By_election_data['div_nm']=='Batman','div_nm'] = 'Cooper' # manual adjustment

        Seat_polls = Seat_polls.merge(By_election_data[['days_since_election','byelection_year','div_nm']], left_on = ['byelection_year','Electorate'], right_on = ['byelection_year','div_nm']).drop(columns=['div_nm'])
        Seat_polls.loc[:,'Days since last election'] = Seat_polls.loc[:,'days_since_election'] - Seat_polls.loc[:,'Days since last election']
        Seat_polls = Seat_polls.rename(columns={'Days since last election':'Days before election'}).drop(columns=['days_since_election'])

        Seat_polls = Seat_polls[Seat_polls['Electorate'] != 'Canberra']
        Seat_poll_year_dict['Byelection'] = Seat_polls



FARRER_ONLY = 1

for election_year in election_years:

    if FARRER_ONLY:
        if election_year != 'Byelection':
            continue
        else:
            prior_long = pd.read_csv(f"Farrer_candidates.csv")
    else:
        prior_long = pd.read_csv(f"Fundamentals_Votes_For_{election_year}.csv")

    averages_rows = []

    # perform for each electorate with seat polling
    curr_seat_polls = Seat_poll_year_dict[election_year]
    for div in curr_seat_polls['Electorate'].unique():

        if FARRER_ONLY:
            div = 'Farrer'
            curr_seat_polls = pd.read_csv('SeatPollsByelectionFormatted.csv')


        if prior_long.loc[prior_long['div_nm']==div,].empty: # e.g., Canberra 1995
            continue

        prior_row = prior_long.loc[prior_long['div_nm']==div,][['PartyAb','FP_Votes']].set_index('PartyAb').T.rename(index = {'FP_Votes':div})
        poll_df = curr_seat_polls.loc[curr_seat_polls['Electorate']==div].set_index('Electorate')

        
        prior_CAGO =  group_into_Fundamentals_Categories(prior_row, div, is_Other = True)

        if FARRER_ONLY:
            missing_party = 'ALP'
            prior_row = prior_row.rename(columns={'IND1':'IND'})
        else:
            missing_party = next((p for p in ['COAL','ALP','GRN'] if prior_CAGO.loc[div, p] == 0),None)

        poll_avg = compute_polling_avg(poll_df,prior_row,div,election_year,missing_party = None,)

        curr_row = curr_seat_polls[curr_seat_polls['Electorate']==div].iloc[[0],:].copy()
        curr_row.loc[:, poll_avg.index] = poll_avg.values
        curr_row['Sample size'] = np.min([len(poll_df),3])
        curr_row['Days before election'] = 1000
        curr_row = curr_row.rename(columns = {'Sample size': 'n_polls'})

        import pdb; pdb.set_trace()


    

        curr_seat_polls = curr_seat_polls[~(curr_seat_polls['Electorate']==div)] # get rid of existing rows so that it can be reborn


        averages_rows.append(curr_row)


    

    import pdb; pdb.set_trace()

    if not FARRER_ONLY:
        averaged_polls = pd.concat(averages_rows)

        import pdb; pdb.set_trace()

        averaged_polls.to_csv(f'SeatPollAverage{election_year}Formatted.csv', index=False)
    else:

        curr_row = curr_row.drop(columns='Days before election')
        curr_row['OTH'] -= curr_row['IND']
        curr_row.to_csv(f'SeatPollAverageFarrerByelectionFormatted.csv', index=False)

import pdb; pdb.set_trace()