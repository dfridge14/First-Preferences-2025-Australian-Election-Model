import pandas as pd
import numpy as np
import os, time
from pathlib import Path


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


MIN_OBSERVABLE_RATIO = 0.001 # If SA1 somehow has 1000 voters, 0.01*1000=1 vote - anything else will not be observed due to rounding errors

SA1_year_dict = {'2025':'2021','2022':'2016','2019':'2016','2016':'2011','2013':'2011','2010':'2006'}
Redistribution_SA1_year_dict = {'2022':'2021','2019':'2016','2016':'2011'}

data_year = '2019'
correspondence_years = [SA1_year_dict[data_year],SA1_year_dict[str(int(data_year)+3)]]

# TO DO: 1. Make separate correspondence functions for each year - too few cases to generalise, especially since 2006--> 2001 is whole different structure!


def perform_SA1_Correspondence_to_SA1_By_PP(SA1_Correspondence_old_new, SA1_By_PP_SA1_CODE16):

    ### inputs SA1_Correspondence_old_new: a df of proportions of each 2016 SA1 that were transferred to 2021 SA1s, and SA1_By_PP_SA1_CODE16, the Votes_By_PP_Complete file from the previous election

    SA1_Correspondence_old_new_redistribution = SA1_Correspondence_old_new.loc[SA1_Correspondence_old_new["SA1_CODE16"].str.startswith(("1","2","5","7")),] # NSW,VIC,WA,NT
    SA1_Correspondence_old_new_redistribution_changed = SA1_Correspondence_old_new_redistribution.loc[(SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]<1-MIN_OBSERVABLE_RATIO) & (SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]>MIN_OBSERVABLE_RATIO),]

    SA1_Correspondence_old_new_redistribution_changed.iloc[:,:2] = SA1_Correspondence_old_new_redistribution_changed.iloc[:,:2].astype(int)

    # only 6 SA1s changed, but numbers kept:
    #kept_SA1_nos = SA1_Correspondence_old_new_redistribution_changed.loc[SA1_Correspondence_old_new_redistribution_changed["SA1_CODE21"].isin(SA1_Correspondence_old_new_redistribution.loc[:,"SA1_CODE16"].unique()),]
    # ones that are not between 0 and  MIN_OBSERVABLE_RATIO and vice versa
    #SA1_Correspondence_old_new_redistribution.loc[((SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]>1-MIN_OBSERVABLE_RATIO) & (SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]<1)) | ((SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]<MIN_OBSERVABLE_RATIO)&(SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]>0)),]

    df_to_add = pd.DataFrame(columns=SA1_By_PP_SA1_CODE16.columns.tolist())

    # for every changed SA1 in redistribution state where someone voted last election
    change_set = set(SA1_Correspondence_old_new_redistribution_changed.iloc[:,0]) & set(SA1_By_PP_SA1_CODE16.loc[:,"SA1_CODE16"])
    
    for changed_redistributed_SA1 in change_set: # 1377 of them

        votes = SA1_By_PP_SA1_CODE16.loc[SA1_By_PP_SA1_CODE16["SA1_CODE16"]==changed_redistributed_SA1,]
        weights = SA1_Correspondence_old_new_redistribution_changed.loc[SA1_Correspondence_old_new_redistribution_changed["SA1_CODE16"]==changed_redistributed_SA1,]

        new_SA1_section = pd.concat([votes] * weights.shape[0], ignore_index=True)
        new_SA1_section.loc[:,"votes"] = np.round(new_SA1_section.loc[:,"votes"] * np.repeat(weights['RATIO_FROM_TO'].values, votes.shape[0]))
        new_SA1_section.loc[:,'SA1_CODE16'] = np.repeat(weights['SA1_CODE21'], votes.shape[0]).reset_index(drop=True).astype(int) 

        df_to_add = pd.concat([df_to_add, new_SA1_section], ignore_index=True)

    print(time.time()-start)
    SA1_By_PP_SA1_CODE16 = SA1_By_PP_SA1_CODE16[~SA1_By_PP_SA1_CODE16["SA1_CODE16"].isin(change_set)]

    # SA1_CODE16 and SA1_CODE21 different, but RATIO is 1
    name_changes_or_full_donations = SA1_Correspondence_old_new_redistribution.loc[(SA1_Correspondence_old_new_redistribution["SA1_CODE16"]!=SA1_Correspondence_old_new_redistribution["SA1_CODE21"]) &  (SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]==1),]
    name_changes_or_full_donations_set = set(name_changes_or_full_donations.iloc[:,0].astype(int))  & set(SA1_By_PP_SA1_CODE16.loc[:,"SA1_CODE16"]) # only those with votes at last election
    # almost full donations (up to MIN_OBSERVABLE_RATIO)
    almost_full_donations = SA1_Correspondence_old_new_redistribution.loc[(SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]>1-MIN_OBSERVABLE_RATIO) & (SA1_Correspondence_old_new_redistribution["RATIO_FROM_TO"]<1),]
    almost_full_donations_set = set(almost_full_donations.iloc[:,0].astype(int)) & set(SA1_By_PP_SA1_CODE16.loc[:,"SA1_CODE16"]) # only those with votes at last election
    # combine both
    all_donations = pd.concat([name_changes_or_full_donations,almost_full_donations], ignore_index=True).iloc[:,:2]
    all_donations.loc[:,['SA1_CODE16','SA1_CODE21']] = all_donations.loc[:,['SA1_CODE16','SA1_CODE21']].astype(int)
    
    all_donations_set = almost_full_donations_set | name_changes_or_full_donations_set # mutually exclusive

    # rename the SA1s that have been renamed/donated
    mask = SA1_By_PP_SA1_CODE16["SA1_CODE16"].isin(all_donations_set)

    merger = SA1_By_PP_SA1_CODE16.loc[SA1_By_PP_SA1_CODE16["SA1_CODE16"].isin(all_donations_set),].merge(all_donations.loc[all_donations["SA1_CODE16"].isin(all_donations_set),], on = "SA1_CODE16", how='left')
    merger = merger[['div_nm','SA1_CODE21','pp_id','votes']]
    merger.loc[:,['SA1_CODE21','votes']] = merger.loc[:,['SA1_CODE21','pp_id','votes']].astype(int)

    SA1_By_PP_SA1_CODE16.loc[mask,['div_nm', 'SA1_CODE16','pp_id','votes']] = merger.rename(columns={'SA1_CODE21': 'SA1_CODE16'})[['div_nm', 'SA1_CODE16','pp_id','votes']].reset_index(drop=True).values


    SA1_By_PP_SA1_CODE16 = pd.concat([SA1_By_PP_SA1_CODE16, df_to_add], ignore_index=True).rename(columns = {'SA1_CODE16': 'SA1_CODE21'})

    return SA1_By_PP_SA1_CODE16.groupby(['div_nm', 'SA1_CODE21','pp_id'], as_index=False)['votes'].sum() # aggregate split up SA1s and result correct SA1_By_PP


if data_year == '2022': # different edition of SA1s - need for correspondence

    start = time.time()


    SA1_Correspondence_old_new = pd.read_csv("CG_SA1_2016_SA1_2021.csv", index_col=None)
    SA1_Correspondence_old_new.rename(columns={"SA1_MAINCODE_2016": "SA1_CODE16", "SA1_CODE_2021":"SA1_CODE21"}, inplace=True)
    SA1_Correspondence_old_new = SA1_Correspondence_old_new[["SA1_CODE16","SA1_CODE21","RATIO_FROM_TO"]].drop(SA1_Correspondence_old_new.index[-1]) # removes last misbehaving row
    SA1_Correspondence_old_new['SA1_CODE16'] = SA1_Correspondence_old_new['SA1_CODE16'].apply(lambda x: int(x))

    SA1_Correspondence_old_new['SA1_CODE21'] = SA1_Correspondence_old_new['SA1_CODE21'].astype(str).str[:1] + SA1_Correspondence_old_new['SA1_CODE21'].astype(str).str[5:]
    SA1_Correspondence_old_new['SA1_CODE16'] = SA1_Correspondence_old_new['SA1_CODE16'].astype(str).str[:1] + SA1_Correspondence_old_new['SA1_CODE16'].astype(str).str[5:]

    # fix exception where SA1 donates to nowhere
    SA1_Correspondence_old_new.loc[SA1_Correspondence_old_new["SA1_CODE16"]=='1153109',"RATIO_FROM_TO"] = 1
    SA1_Correspondence_old_new = SA1_Correspondence_old_new.loc[~(SA1_Correspondence_old_new["SA1_CODE21"]=='n'),]

    # expand SA1_By_PP_Complete to SA1_CODE21
    SA1_By_PP_SA1_CODE16 = pd.read_csv(f"{data_year}SA1ByPPComplete.csv", index_col=None)


    # perform SA1 Correspondence!

    SA1_By_PP_Votes_new = perform_SA1_Correspondence_to_SA1_By_PP(SA1_Correspondence_old_new, SA1_By_PP_SA1_CODE16)
    #SA1_By_PP_Votes_new.to_csv(f"{data_year}SA1_By_PP_Votes.csv", index=False)

elif data_year in ['2016','2019']:
    # post-2016 data still in 2011, post-2019 data all in 2016!!
    SA1_By_PP_Votes_new = pd.read_csv(f"{data_year}SA1ByPPComplete.csv", index_col=None)
    #SA1_By_PP_Votes_new.to_csv(f"{data_year}SA1_By_PP_Votes.csv", index=False)
    import pdb;pdb.set_trace()





def format_state_rdst_full(df,SA1_suffix):
    

    df = df.rename(columns={df.columns[0]: f'SA1_CODE{SA1_suffix}',df.columns[1]: 'new_div', df.columns[2]: 'old_div', df.columns[4]: 'curr_enrol',df.columns[5]: 'proj_enrol'})
    df = df[[f'SA1_CODE{SA1_suffix}',"new_div","old_div",'curr_enrol','proj_enrol']].drop(df.index[-1]) # removes last misbehaving row
    #import pdb;pdb.set_trace()

    if df.iloc[:, -2:].dtypes.isin([np.dtype(np.float64), np.dtype(np.int64)]).sum() < 2:
        df.iloc[:,-2:] = df.iloc[:,-2:].replace(',', '', regex=True).apply(lambda col: col.str.strip()).replace('-', '0').astype(int)

    df = df.loc[(df['curr_enrol'] > 10) & (df['proj_enrol'] > 10)].reset_index(drop=True) # ignore small changes

    #import pdb;pdb.set_trace()


    df.loc[:,f'SA1_CODE{SA1_suffix}'] = df[f'SA1_CODE{SA1_suffix}'].astype(str).str[:7].astype(int)  # remove alpha characters at end - split SA1s don't matter as they are mostly taken care of

    return df

def format_state_rdst_full_without_enrol(df,SA1_suffix):

    df = df.rename(columns={df.columns[0]: 'new_div',df.columns[1]: 'old_div', f'SA1 Code\n(20{SA1_suffix} SA1s)': f'SA1_CODE{SA1_suffix}'})
    df = df[[f'SA1_CODE{SA1_suffix}',"new_div","old_div"]].drop(df.index[-1]) # removes last misbehaving row
    #import pdb;pdb.set_trace()

    df.loc[:,f'SA1_CODE{SA1_suffix}'] = df[f'SA1_CODE{SA1_suffix}'].astype(str).str[:7].astype(int) # remove alpha characters at end - split SA1s don't matter as they are mostly taken care of

    return df


SA1_suffix = Redistribution_SA1_year_dict[data_year][-2:]


if data_year == '2022':

    states = ['VIC','NSW','WA','NT']
    state_dfs = [pd.read_csv(f"Redistribution2024{state}-by-SA2-and-SA1.csv", index_col=None) for state in states]

    state_dfs_Redistribution = [format_state_rdst_full(state_df, SA1_suffix) for state_df in state_dfs]
    Redistribution_SA1s = pd.concat(state_dfs_Redistribution, ignore_index=True)

    import pdb;pdb.set_trace()

    #VIC_SA1s_Redistribution_full = pd.read_csv("Redistribution2024VIC-by-SA2-and-SA1.csv", index_col=None)
    #NSW_SA1s_Redistribution_full = pd.read_csv("Redistribution2024NSW-by-SA2-and-SA1.csv", index_col=None)
    #WA_SA1s_Redistribution_full = pd.read_csv("Redistribution2024WA-by-SA2-and-SA1.csv", index_col=None)



    # combine into one Redistributions df
    #VIC_SA1s_Redistribution = format_state_rdst_full(VIC_SA1s_Redistribution_full, SA1_suffix)
    #NSW_SA1s_Redistribution = format_state_rdst_full(NSW_SA1s_Redistribution_full, SA1_suffix)
    #WA_SA1s_Redistribution = format_state_rdst_full(WA_SA1s_Redistribution_full, SA1_suffix)

    #Redistribution_SA1s = pd.concat([VIC_SA1s_Redistribution,NSW_SA1s_Redistribution,WA_SA1s_Redistribution], ignore_index=True)
    #Redistribution_SA1s.iloc[:,-2:] = Redistribution_SA1s.iloc[:,-2:].astype(int)

elif data_year == '2019':
    #VIC_SA1s_Redistribution_full = pd.read_csv("Redistribution2021VIC-by-SA2-and-SA1.csv", index_col=None)
    #WA_SA1s_Redistribution_full = pd.read_csv("Redistribution2021WA-by-SA2-and-SA1.csv", index_col=None)


    #VIC_SA1s_Redistribution = format_state_rdst_full(VIC_SA1s_Redistribution_full, SA1_suffix)
    #WA_SA1s_Redistribution = format_state_rdst_full(WA_SA1s_Redistribution_full, SA1_suffix)

    #Redistribution_SA1s = pd.concat([VIC_SA1s_Redistribution,WA_SA1s_Redistribution], ignore_index=True)


    states = ['VIC','WA']
    state_dfs = [pd.read_csv(f"Redistribution2021{state}-by-SA2-and-SA1.csv", index_col=None) for state in states]

    state_dfs_Redistribution = [format_state_rdst_full(state_df, SA1_suffix) for state_df in state_dfs]
    Redistribution_SA1s = pd.concat(state_dfs_Redistribution, ignore_index=True)

    import pdb;pdb.set_trace()


    #Redistribution_SA1s.iloc[:,-2:] = Redistribution_SA1s.iloc[:,-2:].astype(int)

elif data_year == '2016':

    states = ['VIC','TAS','QLD','NT','ACT']
    state_dfs = [pd.read_csv(f"Redistribution2018{state}-by-SA2-and-SA1.csv", index_col=None) for state in states]
    # format SA SA1s correctly starting with a '4'
    SA_df = pd.read_csv(f"Redistribution2018SA-by-SA2-and-SA1.csv", index_col=None)
    SA_df.iloc[:,0] = '4' + SA_df.iloc[:,0].astype(str)
    state_dfs.append(SA_df)
    
    # combine into one Redistributions df
    state_dfs_Redistribution = [format_state_rdst_full(state_df, SA1_suffix) if len(state_df.columns)==6 
                                                                else format_state_rdst_full_without_enrol(state_df, SA1_suffix) for state_df in state_dfs]

    Redistribution_SA1s = pd.concat(state_dfs_Redistribution, ignore_index=True)




# Rename old divisions prior to analysis, treating as the natural state; ignores the effect of renaming divisions!
name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'}}

import pdb;pdb.set_trace()
# remame old_div to new_div if there was a name change!
if name_changes_year_dict[data_year]:
    Redistribution_SA1s.loc[Redistribution_SA1s['old_div'].isin(name_changes_year_dict[data_year].keys()),'old_div'] = Redistribution_SA1s['old_div'].map(name_changes_year_dict[data_year])

# ensure re-case-ment before selecting only changes across divisions
recase_map = {'Mcmahon':'McMahon', 'Mcewen':'McEwen','Eden-monaro':'Eden-Monaro',"O'connor": "O'Connor",'LINGIARI':'Lingiari','SOLOMON':'Solomon'}
Redistribution_SA1s.loc[:,['new_div','old_div']] = Redistribution_SA1s.loc[:,['new_div','old_div']].replace(recase_map)

Redistribution_SA1s_changes = Redistribution_SA1s.loc[Redistribution_SA1s['new_div']!=Redistribution_SA1s['old_div'],][[f'SA1_CODE{SA1_suffix}','new_div','old_div']]

#import pdb;pdb.set_trace()


#Redistribution_SA1s_changes.to_csv(f"Redistribution_SA1_changes{str(int(data_year)+2)}.csv", index=False)

# get df of just the Redistribution pairs, for information on which electorate margins are necessary to calculate
SA1_By_PP_Votes = pd.read_csv(f"{data_year}SA1_By_PP_Votes.csv", index_col=None)
SA1s_with_votes = set(SA1_By_PP_Votes.iloc[:,1])


Redistribution_SA1s_changes = Redistribution_SA1s_changes.loc[Redistribution_SA1s_changes[f"SA1_CODE{SA1_suffix[-2:]}"].isin(SA1s_with_votes),]
Redistribution_pair_SA1s = Redistribution_SA1s_changes.groupby(['old_div', 'new_div'])[f"SA1_CODE{SA1_suffix[-2:]}"].apply(list).reset_index()
#Redistribution_pairs = list(Redistribution_SA1s_changes_dict.keys())

Redistribution_pairs = Redistribution_pair_SA1s.iloc[:,:2]
#Redistribution_pairs.to_csv(f"RedistributionPairs{str(int(data_year)+2)}.csv", index = False)

#import pdb;pdb.set_trace()


name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'}}
abolished_divs = {'2016': set(['Port Adelaide']),'2019':set(['Stirling'])}


def add_1_to_1_rows_non_abolished(Redistribution_pairs_year, old_year,new_year, abolished_divs):
    # returns complete transfers with 1:1 rows for divisiosn that are not abolished. 
    # redist_year is 2 more than data_year i.e. 
    unique_old_divs = Redistribution_pairs_year[f'div_nm_{old_year}'].unique()
    unique_old_divs_minus_abolished = set(unique_old_divs) - abolished_divs[str(int(old_year) + 1)] # latter is data_year
    new_rows = pd.DataFrame({f'div_nm_{old_year}': list(unique_old_divs_minus_abolished), f'div_nm_{new_year}': list(unique_old_divs_minus_abolished)})

    return pd.concat([Redistribution_pairs_year,new_rows], ignore_index=True)




def format_2018_2021_correspondence():
    
    Correspondence_CED_2018_2021 = pd.read_csv(f'CG_CED_2018_CED_2021.csv', index_col=None)[['CED_NAME_2018','CED_NAME_2021','RATIO_FROM_TO']].dropna()

    Correspondence_CED_2018_2021.rename(columns={'CED_NAME_2018':'div_nm_2018','CED_NAME_2021':'div_nm_2021'}, inplace=True)


    Correspondence_CED_2018_2021 = Correspondence_CED_2018_2021.loc[~Correspondence_CED_2018_2021['div_nm_2021'].str.contains(r'\(', na=False)]

    Redistribution_pairs_2021 = pd.read_csv(f"RedistributionPairs2021.csv", index_col = None).rename(columns={'old_div':'div_nm_2018','new_div':'div_nm_2021'}).iloc[:,:2]

    # add 1:1 transfers of dfs to themselves
    unique_old_divs = Redistribution_pairs_2021['div_nm_2018'].unique()
    new_rows = pd.DataFrame({'div_nm_2018': unique_old_divs, 'div_nm_2021': unique_old_divs})
    Redistribution_pairs_2021_full = pd.concat([Redistribution_pairs_2021,new_rows], ignore_index=True)

    # remove insignificant transfers not appearing in Redistribution_pairs_2021
    redist_map = Redistribution_pairs_2021_full.groupby('div_nm_2018')['div_nm_2021'].apply(set).to_dict()
    mask_valid_old_div = Correspondence_CED_2018_2021['div_nm_2018'].isin(redist_map)
    mask_invalid_new_div = ~Correspondence_CED_2018_2021.loc[mask_valid_old_div].apply(lambda row: row['div_nm_2021'] in redist_map[row['div_nm_2018']], axis=1)
    mask = mask_valid_old_div.copy()
    mask.loc[mask_valid_old_div] = mask_invalid_new_div.values
    Correspondence_CED_2018_2021 = Correspondence_CED_2018_2021.loc[~mask,]

    # if old_div is not in Redistribution_pairs_2021_full's rows, then replace it with div, div, 1 & remove duplicates
    mask_not_in_old_divs = ~Correspondence_CED_2018_2021['div_nm_2018'].isin(unique_old_divs)
    Correspondence_CED_2018_2021.loc[mask_not_in_old_divs, 'div_nm_2021'] = Correspondence_CED_2018_2021.loc[mask_not_in_old_divs, 'div_nm_2018']
    Correspondence_CED_2018_2021.loc[mask_not_in_old_divs, 'RATIO_FROM_TO'] = 1
    Correspondence_CED_2018_2021.drop_duplicates(inplace=True)

    Correspondence_CED_2018_2021['RATIO_FROM_TO'] = Correspondence_CED_2018_2021.groupby('div_nm_2018')['RATIO_FROM_TO'].transform(lambda x: x / x.sum())

    Correspondence_CED_2018_2021.to_csv('Correspondence_CED_2018_2021.csv', index=False)

    return Correspondence_CED_2018_2021



def format_old_new_correspondence(old_year, new_year, abolished_divs):
    data_year = str(int(old_year) + 1)
    
    if old_year == '2015':
        Correspondence_CED_old_new = pd.read_csv(f'CG_CED_{old_year}_CED_{new_year}.csv', index_col=None).rename(columns={f'CED_NAME_{data_year}':f'div_nm_{old_year}',f'CED_NAME_{new_year}':f'div_nm_{new_year}'})
                                            
    else:
        Correspondence_CED_old_new = pd.read_csv(f'CG_CED_{old_year}_CED_{new_year}.csv', index_col=None).rename(columns={f'CED_NAME_{data_year}':f'div_nm_{old_year}',f'CED_NAME_{new_year}':f'div_nm_{new_year}'})

    Correspondence_CED_old_new = Correspondence_CED_old_new[[f'div_nm_{old_year}',f'div_nm_{new_year}','RATIO_FROM_TO']].dropna()


    Correspondence_CED_old_new = Correspondence_CED_old_new.loc[~Correspondence_CED_old_new[f'div_nm_{new_year}'].str.contains(r'\(', na=False)]

    if old_year == '2015':
        Redistribution_pairs_new = pd.read_csv("Redistribution_pairs_2015_2021.csv", index_col = None).rename(columns={'old_div':f'div_nm_{old_year}','new_div':f'div_nm_{new_year}'}).iloc[:,:2]
    else:
        Redistribution_pairs_new = pd.read_csv(f"RedistributionPairs{new_year}.csv", index_col = None).rename(columns={'old_div':f'div_nm_{old_year}','new_div':f'div_nm_{new_year}'}).iloc[:,:2]


    # add 1:1 transfers of dfs to themselves
    Redistribution_pairs_new_full = add_1_to_1_rows_non_abolished(Redistribution_pairs_new, old_year, new_year, abolished_divs).drop_duplicates()
    import pdb;pdb.set_trace()


    # remove insignificant transfers not appearing in Redistribution_pairs_2021
    redist_map = Redistribution_pairs_new_full.groupby(f'div_nm_{old_year}')[f'div_nm_{new_year}'].apply(set).to_dict()
    mask_valid_old_div = Correspondence_CED_old_new[f'div_nm_{old_year}'].isin(redist_map) # old_year in Redist_pairs_full
    mask_invalid_new_div = ~Correspondence_CED_old_new.loc[mask_valid_old_div].apply(lambda row: row[f'div_nm_{new_year}'] in redist_map[row[f'div_nm_{old_year}']], axis=1) 
    mask = mask_valid_old_div.copy()
    mask.loc[mask_valid_old_div] = mask_invalid_new_div.values # old_year in Redist_pairs_full & new_div is not in Redist_pairs_full
    Correspondence_CED_old_new = Correspondence_CED_old_new.loc[~mask,]

    import pdb;pdb.set_trace()


    # if old_div is not in Redistribution_pairs_2021_full's rows, then replace it with div, div, 1 & remove duplicates
    unique_old_divs = Redistribution_pairs_new_full[f'div_nm_{old_year}'].unique()
    mask_not_in_old_divs = ~Correspondence_CED_old_new[f'div_nm_{old_year}'].isin(unique_old_divs)
    Correspondence_CED_old_new.loc[mask_not_in_old_divs, f'div_nm_{new_year}'] = Correspondence_CED_old_new.loc[mask_not_in_old_divs, f'div_nm_{old_year}']
    Correspondence_CED_old_new.loc[mask_not_in_old_divs, 'RATIO_FROM_TO'] = 1
    Correspondence_CED_old_new.drop_duplicates(inplace=True)

    Correspondence_CED_old_new['RATIO_FROM_TO'] = Correspondence_CED_old_new.groupby(f'div_nm_{old_year}')['RATIO_FROM_TO'].transform(lambda x: x / x.sum()) # rescale to 1

    import pdb;pdb.set_trace()


    Correspondence_CED_old_new.to_csv(f'Correspondence_CED_{old_year}_{new_year}.csv', index=False)

    return Correspondence_CED_old_new



def reverse_2021_2018_correspondence(Correspondence_df):
    # matches pretty closely with proportions_moved file

    Correspondence_df = Correspondence_df.copy()
    total_per_new = Correspondence_df.groupby('div_nm_2021')['RATIO_FROM_TO'].transform('sum')
    Correspondence_df['proportion'] = Correspondence_df['RATIO_FROM_TO'] / total_per_new

    Correspondence_df = Correspondence_df[['div_nm_2021','div_nm_2018','proportion']].rename(columns = {'div_nm_2021':'new_div', 'div_nm_2018':'old_div'}).sort_values(by='new_div')

    import pdb;pdb.set_trace()

    Correspondence_df.to_csv('Correspondence_CED_2021_2018_Reversed.csv', index=False)

    return Correspondence_df



def normalise_2015_2018_Correspondence(Correspondence_df):
    import pdb;pdb.set_trace()

    Correspondence_df['ratio'] = Correspondence_df['proportion'] / Correspondence_df.groupby('old_div')['proportion'].transform('sum')

    import pdb;pdb.set_trace()

    Correspondence_df = Correspondence_df[['old_div','new_div','ratio']].rename(columns={'ratio': 'RATIO_FROM_TO'})

    return Correspondence_df



def convert_2016_2021_proportions_to_2015_2018(name_changes_year_dict, abolished_divs):


    # This probably replaces all other attempts - use proportions_moved methodology!
    Correspondence_CED_2018_2015_Reversed = pd.read_csv('Correspondence_CED_2018_2015_Reversed.csv', index_col=None)
    Correspondence_CED_2015_2018 = normalise_2015_2018_Correspondence(Correspondence_CED_2018_2015_Reversed)
    Correspondence_CED_2015_2018['old_div'] = Correspondence_CED_2015_2018['old_div'].replace(name_changes_year_dict['2016'])

    Correspondence_CED_2015_2018.sort_values(by='old_div').to_csv("Correspondence_CED_2015_2018.csv", index=False)

    import pdb;pdb.set_trace()


    Correspondence_CED_2016_2021 = pd.read_csv(f'CG_CED_2015_CED_2021.csv', index_col=None)[['CED_NAME_2016','CED_NAME_2021','RATIO_FROM_TO']].dropna()
    Correspondence_CED_2016_2021.rename(columns={'CED_NAME_2016':'div_nm_2015','CED_NAME_2021':'div_nm_2021'}, inplace=True)
    Correspondence_CED_2016_2021 = Correspondence_CED_2016_2021.loc[~Correspondence_CED_2016_2021['div_nm_2021'].str.contains(r'\(', na=False)]
    Correspondence_CED_2018_2021 = format_2018_2021_correspondence()

    
    Redistribution_pairs_2018 = pd.read_csv(f"RedistributionPairs2018.csv", index_col = None).rename(columns={'old_div':'div_nm_2015','new_div':'div_nm_2018'}).iloc[:,:2]
    Redistribution_pairs_2021 = pd.read_csv(f"RedistributionPairs2021.csv", index_col = None).rename(columns={'old_div':'div_nm_2018','new_div':'div_nm_2021'}).iloc[:,:2]

    # merge on div_nm_2018. First, add all unchanging divs (except ones that are abolished!!!)


    #maintaining_divs_2016_set = set(Correspondence_CED_2016_2021['div_nm_2015']) - abolished_divs['2016'] - set(name_changes_year_dict['2016'].keys())

    #to_extend = pd.DataFrame([(div, div) for div in maintaining_divs_2016_set], columns=['div_nm_2015', 'div_nm_2018'])
    #to_extend_name_changes = pd.DataFrame([(div, name_changes_year_dict['2016'][div]) for div in set(name_changes_year_dict['2016'].keys())], columns=['div_nm_2015', 'div_nm_2018'])

    # full spread of 2015-2018 div changes, including those that stayed the same
    #Redistribution_pairs_2018_extended = pd.concat([Redistribution_pairs_2018,to_extend,to_extend_name_changes])

    # for those divs in div_nm_2018 (that are changing), introduce div-div row in 2015 df ; name changes were already adjusted in Redistribution_pairs_2018

    # 1. introduce 1:1 rows for 2015-2018 divs already in Redistribution_pairs_2018 (unless abolished)
    Redistribution_pairs_2018_part_ext = add_1_to_1_rows_non_abolished(Redistribution_pairs_2018, '2015','2018', abolished_divs)

    import pdb;pdb.set_trace()


    # 2. find div_nm_2018s in 2021 that are not old_divs in 2018 file, add them too!
    unique_old_divs = Redistribution_pairs_2021[f'div_nm_2018'].unique()
    unique_old_divs_minus_abolished = set(unique_old_divs) - set(Redistribution_pairs_2018_part_ext['div_nm_2015']) # not a 2015 old_div
    new_rows = pd.DataFrame({f'div_nm_2015': list(unique_old_divs_minus_abolished), f'div_nm_2018': list(unique_old_divs_minus_abolished)})

    Redistribution_pairs_2018_ext = pd.concat([Redistribution_pairs_2018_part_ext,new_rows], ignore_index=True)

    # 3. merged them on div_nm_2018, removing duplicates!
    rd = Redistribution_pairs_2018_ext.merge(Redistribution_pairs_2021, on='div_nm_2018', how='left')
    rd.loc[rd['div_nm_2021'].isna(),'div_nm_2021'] = rd.loc[rd['div_nm_2021'].isna(), 'div_nm_2018'] # add in missing values from 2021 by copying 2018
    rd = rd[['div_nm_2015','div_nm_2021']]
    rd.drop_duplicates(inplace=True)

    #rd.to_csv("Redistribution_pairs_2015_2021.csv", index=False)

    import pdb;pdb.set_trace()



    # now we have all the changed divisions from 2015-2021! Perform the format_2018_2021_correspondence!
    Correspondence_CED_2015_2021 = format_old_new_correspondence('2015', '2021', abolished_divs)
    Correspondence_CED_2021_2018_Reversed = pd.read_csv(f'Correspondence_CED_2021_2018_Reversed.csv', index_col=None)


    Correspondence_CED_2015_2018 = Correspondence_CED_2015_2021.merge(Correspondence_CED_2021_2018_Reversed, on='div_nm_2021')

    Correspondence_CED_2015_2018['ratio'] = Correspondence_CED_2015_2018['RATIO_FROM_TO'] * Correspondence_CED_2015_2018['REVERSE_RATIO']
    Correspondence_CED_2015_2018 = Correspondence_CED_2015_2018[['div_nm_2015','div_nm_2018','ratio']] # reduce to 3 cols

    # combine duplicated, adding ratios
    Correspondence_CED_2015_2018 = Correspondence_CED_2015_2018.groupby(['div_nm_2015','div_nm_2018'], as_index=False)['ratio'].sum()

    Correspondence_CED_2015_2018[['div_nm_2018','div_nm_2015','ratio']].sort_values(by='div_nm_2018')






    Correspondence_CED_2015_2018.to_csv(f'Correspondence_CED_2015_2018.csv', index=False)





    import pdb;pdb.set_trace()



    return Correspondence_CED_2015_2018


reverse_2021_2018_correspondence(format_2018_2021_correspondence())

convert_2016_2021_proportions_to_2015_2018(name_changes_year_dict, abolished_divs)
import pdb;pdb.set_trace()


# Construct proportions of electorates transferred!
Proportions_transferred = pd.read_csv(f'CG_CED_{str(int(data_year) - 1)}_CED_{str(int(data_year) +2)}.csv', index_col=None) # will fix 2016 --> 2015
Proportions_transferred = Proportions_transferred.rename(columns = {f'CED_NAME_{str(int(data_year) - 1)}':'old_div', f'CED_NAME_{str(int(data_year) +2)}':'new_div'})
Proportions_transferred = Proportions_transferred[['new_div','old_div','RATIO_FROM_TO']]
# ensure all corrrespondence is from 4 years before election to 1 year before election

Redistribution_pairs_ratios = Redistribution_pairs.merge(Proportions_transferred, on = ['new_div','old_div'], how='left')

import pdb;pdb.set_trace()
Redistribution_pairs_ratios.to_csv(f"RedistributionPairs{str(int(data_year)+2)}.csv", index = False)





















def check_SA1_consistency(VIC_SA1s_Redistribution):
    # investigates SA1s that were spread between divisions at 2022 election, how they changed in correspondence, and which are again split in 
    # new electoral boundaries in 2024 Redistribution (examples for VIC)

    # find which 2016 SA1s are spread across divisions (have votes in both divisions)
    spread = (SA1_By_PP_SA1_CODE16.groupby("SA1_CODE16")["div_nm"].nunique() > 1)
    spread=spread[spread].index
    spread_over_divs = SA1_By_PP_SA1_CODE16.iloc[:, :2][SA1_By_PP_SA1_CODE16["SA1_CODE16"].isin(spread)].drop_duplicates().sort_values("SA1_CODE16")
    spread_over_divs = spread_over_divs.loc[spread_over_divs['SA1_CODE16'].astype(str).str.startswith(("1","2","5","7")),] # In redistribution staes
    spread_over_divs_SA1s = set(spread_over_divs["SA1_CODE16"])

    # check if any have been updates in 2021 SA1s
    changed_spread_divs = SA1_Correspondence_old_new.loc[(SA1_Correspondence_old_new['SA1_CODE16'].astype(int).isin(spread_over_divs_SA1s)) & (SA1_Correspondence_old_new['SA1_CODE16'] != SA1_Correspondence_old_new['SA1_CODE21']),]
    changed_spread_divs_set = set(changed_spread_divs.iloc[:,0].unique())
    changed_spread_divs_set = {int(x) for x in changed_spread_divs_set if str(x).startswith(("1","2","5","7"))} # only redistr
    changed_spread_divs_newSA1s_set = {int(x) for x in set(SA1_Correspondence_old_new.loc[SA1_Correspondence_old_new['SA1_CODE16'].astype(int).isin(changed_spread_divs_set),]['SA1_CODE21'])}

    spread_SA1s = (spread_over_divs_SA1s - changed_spread_divs_set)| changed_spread_divs_newSA1s_set


    # solving complications with SA1s, where there is mixing going on, or 0 populations
    VIC_SA1s_Redistribution = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution['curr_enrol'] > 10) & (VIC_SA1s_Redistribution['proj_enrol'] > 10)] # potentially be stricter removing any with <10 -- explore
    Mixed_SA1s = VIC_SA1s_Redistribution.loc[VIC_SA1s_Redistribution["SA1_CODE21"].apply(lambda x: isinstance(x,str) and x[-1].isalpha()),] # finds which ones end in alphabet characters A to H (i.e. are mixed SA1s)
    print(Mixed_SA1s)
    ### improve: can remove any solitary ones remaining as their partner is area of 0 votes!

    # these are completely fine, as if they come from multiple electorates, you will anyway select the section that is from old_div to be added to the new_div
    Multiple_Mixed = (Mixed_SA1s.groupby("SA1_CODE21")["old_div"].nunique() > 1)[(Mixed_SA1s.groupby("SA1_CODE21")["old_div"].nunique() > 1)].index.tolist()


    # divided SA1s that had been name changed in 2021, changing hands in VIC - only 2143724 is relevant (but only E matters, hence all goes to scullin)
    changed_SA1s_redistribution_relevant = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution['SA1_CODE21'].str[:7].astype(int).isin(changed_spread_divs_newSA1s_set)) &  (VIC_SA1s_Redistribution['old_div']!= VIC_SA1s_Redistribution['new_div']),]
    # divided SA1s unchanged in 2021 - all going to single place - fine because those votes that were in division before are already accounted for - only new votes are interesting
    unchanged_SA1s_redistribution_relevant = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution['SA1_CODE21'].str[:7].astype(int).isin(spread_over_divs_SA1s - changed_spread_divs_set)) &  (VIC_SA1s_Redistribution['old_div']!= VIC_SA1s_Redistribution['new_div']),]

    return 1


import pdb;pdb.set_trace()


# initial exploration for Higgins
print("Higgins stuff")

VIC_SA1s_Redistribution_full = pd.read_csv("Redistribution2024VIC-by-SA2-and-SA1.csv", index_col=None)
VIC_SA1s_Redistribution = format_state_rdst_full(VIC_SA1s_Redistribution_full, SA1_suffix)

SA1s_From_Higgins = VIC_SA1s_Redistribution.loc[VIC_SA1s_Redistribution["old_div"]=="Higgins","SA1_CODE21"].tolist()
print(sorted(SA1s_From_Higgins), len(SA1s_From_Higgins))
Higgins_SA1_Correspondence_old_new = SA1_Correspondence_old_new.loc[SA1_Correspondence_old_new["SA1_CODE21"].isin(SA1s_From_Higgins),]
Dodgy_SA1s = Higgins_SA1_Correspondence_old_new.loc[Higgins_SA1_Correspondence_old_new["RATIO_FROM_TO"] < 1,]
print(Dodgy_SA1s, Dodgy_SA1s.shape)

print("Aston stuff")
SA1s_to_Aston = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution["old_div"]=="Deakin") & (VIC_SA1s_Redistribution["new_div"]=="Aston"),"SA1_CODE21"].tolist()
print(sorted(SA1s_to_Aston), len(SA1s_to_Aston))
Aston_SA1_Correspondence_old_new = SA1_Correspondence_old_new.loc[SA1_Correspondence_old_new["SA1_CODE21"].isin(SA1s_to_Aston),]
Dodgy_SA1s = Aston_SA1_Correspondence_old_new.loc[Aston_SA1_Correspondence_old_new["RATIO_FROM_TO"] < 1,]
print(Dodgy_SA1s, Dodgy_SA1s.shape)
print(time.time()-start, "seconds")

import pdb;pdb.set_trace()
