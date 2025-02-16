import pandas as pd
import geopandas as gpd
import numpy as np
import os, time


os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()

MIN_OBSERVABLE_RATIO = 0.001 # If SA1 somehow has 1000 voters, 0.01*1000=1 vote - anything else will not be observed due to rounding errors

SA1_Correspondence_2016_2021 = pd.read_csv("CG_SA1_2016_SA1_2021.csv", index_col=None)
SA1_Correspondence_2016_2021.rename(columns={"SA1_MAINCODE_2016": "SA1_CODE16", "SA1_CODE_2021":"SA1_CODE21"}, inplace=True)
SA1_Correspondence_2016_2021 = SA1_Correspondence_2016_2021[["SA1_CODE16","SA1_CODE21","RATIO_FROM_TO"]].drop(SA1_Correspondence_2016_2021.index[-1]) # removes last misbehaving row
SA1_Correspondence_2016_2021['SA1_CODE16'] = SA1_Correspondence_2016_2021['SA1_CODE16'].apply(lambda x: int(x))

SA1_Correspondence_2016_2021['SA1_CODE21'] = SA1_Correspondence_2016_2021['SA1_CODE21'].astype(str).str[:1] + SA1_Correspondence_2016_2021['SA1_CODE21'].astype(str).str[5:]
SA1_Correspondence_2016_2021['SA1_CODE16'] = SA1_Correspondence_2016_2021['SA1_CODE16'].astype(str).str[:1] + SA1_Correspondence_2016_2021['SA1_CODE16'].astype(str).str[5:]

# fix exception where SA1 donates to nowhere
SA1_Correspondence_2016_2021.loc[SA1_Correspondence_2016_2021["SA1_CODE16"]=='1153109',"RATIO_FROM_TO"] = 1
SA1_Correspondence_2016_2021 = SA1_Correspondence_2016_2021.loc[~(SA1_Correspondence_2016_2021["SA1_CODE21"]=='n'),]

# expand SA1_By_PP_Complete to SA1_CODE21
SA1_By_PP_SA1_CODE16 = pd.read_csv("SA1_By_PP_Complete.csv", index_col=None)

def perform_SA1_Correspondence_to_SA1_By_PP(SA1_Correspondence_2016_2021, SA1_By_PP_SA1_CODE16):

    SA1_Correspondence_2016_2021_redistribution = SA1_Correspondence_2016_2021.loc[SA1_Correspondence_2016_2021["SA1_CODE16"].str.startswith(("1","2","5","7")),] # NSW,VIC,WA,NT
    SA1_Correspondence_2016_2021_redistribution_changed = SA1_Correspondence_2016_2021_redistribution.loc[(SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]<1-MIN_OBSERVABLE_RATIO) & (SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]>MIN_OBSERVABLE_RATIO),]

    SA1_Correspondence_2016_2021_redistribution_changed.iloc[:,:2] = SA1_Correspondence_2016_2021_redistribution_changed.iloc[:,:2].astype(int)

    # only 6 SA1s changed, but numbers kept:
    #kept_SA1_nos = SA1_Correspondence_2016_2021_redistribution_changed.loc[SA1_Correspondence_2016_2021_redistribution_changed["SA1_CODE21"].isin(SA1_Correspondence_2016_2021_redistribution.loc[:,"SA1_CODE16"].unique()),]
    # ones that are not between 0 and  MIN_OBSERVABLE_RATIO and vice versa
    #SA1_Correspondence_2016_2021_redistribution.loc[((SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]>1-MIN_OBSERVABLE_RATIO) & (SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]<1)) | ((SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]<MIN_OBSERVABLE_RATIO)&(SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]>0)),]

    df_to_add = pd.DataFrame(columns=SA1_By_PP_SA1_CODE16.columns.tolist())

    # for every changed SA1 in redistribution state where someone voted last election
    change_set = set(SA1_Correspondence_2016_2021_redistribution_changed.iloc[:,0]) & set(SA1_By_PP_SA1_CODE16.loc[:,"SA1_CODE16"])
    
    for changed_redistributed_SA1 in change_set: # 1377 of them

        votes = SA1_By_PP_SA1_CODE16.loc[SA1_By_PP_SA1_CODE16["SA1_CODE16"]==changed_redistributed_SA1,]
        weights = SA1_Correspondence_2016_2021_redistribution_changed.loc[SA1_Correspondence_2016_2021_redistribution_changed["SA1_CODE16"]==changed_redistributed_SA1,]

        new_SA1_section = pd.concat([votes] * weights.shape[0], ignore_index=True)
        new_SA1_section.loc[:,"votes"] = np.round(new_SA1_section.loc[:,"votes"] * np.repeat(weights['RATIO_FROM_TO'].values, votes.shape[0]))
        new_SA1_section.loc[:,'SA1_CODE16'] = np.repeat(weights['SA1_CODE21'], votes.shape[0]).reset_index(drop=True) 

        df_to_add = pd.concat([df_to_add, new_SA1_section], ignore_index=True)

    print(time.time()-start)
    SA1_By_PP_SA1_CODE16 = SA1_By_PP_SA1_CODE16[~SA1_By_PP_SA1_CODE16["SA1_CODE16"].isin(change_set)]

    # SA1_CODE16 and SA1_CODE21 different, but RATIO is 1
    name_changes_or_full_donations = SA1_Correspondence_2016_2021_redistribution.loc[(SA1_Correspondence_2016_2021_redistribution["SA1_CODE16"]!=SA1_Correspondence_2016_2021_redistribution["SA1_CODE21"]) &  (SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]==1),]
    name_changes_or_full_donations_set = set(name_changes_or_full_donations.iloc[:,0].astype(int))  & set(SA1_By_PP_SA1_CODE16.loc[:,"SA1_CODE16"]) # only those with votes at last election
    # almost full donations (up to MIN_OBSERVABLE_RATIO)
    almost_full_donations = SA1_Correspondence_2016_2021_redistribution.loc[(SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]>1-MIN_OBSERVABLE_RATIO) & (SA1_Correspondence_2016_2021_redistribution["RATIO_FROM_TO"]<1),]
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


# perform SA1 Correspondence!

#SA1_By_PP_Votes_2022 = perform_SA1_Correspondence_to_SA1_By_PP(SA1_Correspondence_2016_2021, SA1_By_PP_SA1_CODE16)
#SA1_By_PP_Votes_2022.to_csv("2022SA1_By_PP_Votes.csv", index=False)







VIC_SA1s_Redistribution_full = pd.read_csv("Redistribution2024VIC-by-SA2-and-SA1.csv", index_col=None)
NSW_SA1s_Redistribution_full = pd.read_csv("Redistribution2024NSW-by-SA2-and-SA1.csv", index_col=None)
WA_SA1s_Redistribution_full = pd.read_csv("Redistribution2024WA-by-SA2-and-SA1.csv", index_col=None)


def format_state_rdst_full(df):
    df = df.rename(columns={df.columns[0]: 'SA1_CODE21',df.columns[1]: 'new_div', df.columns[2]: 'old_div', df.columns[4]: 'curr_enrol',df.columns[5]: 'proj_enrol'})
    df = df[["SA1_CODE21","new_div","old_div",'curr_enrol','proj_enrol']].drop(df.index[-1]) # removes last misbehaving row
    df.iloc[:,-2:] = df.iloc[:,-2:].astype(int)
    df = df.loc[(df['curr_enrol'] > 10) & (df['proj_enrol'] > 10)] # ignore small changes
    df.loc[:,"SA1_CODE21"] = df["SA1_CODE21"].astype(str).str[:7] # remove alpha characters at end - split SA1s don't matter as they are taken care of

    return df


# combine into one Redistributions df
VIC_SA1s_Redistribution = format_state_rdst_full(VIC_SA1s_Redistribution_full)
NSW_SA1s_Redistribution = format_state_rdst_full(NSW_SA1s_Redistribution_full)
WA_SA1s_Redistribution = format_state_rdst_full(WA_SA1s_Redistribution_full)

Redistribution_SA1s_2024 = pd.concat([VIC_SA1s_Redistribution,NSW_SA1s_Redistribution,WA_SA1s_Redistribution], ignore_index=True)
Redistribution_SA1s_2024.iloc[:,-2:] = Redistribution_SA1s_2024.iloc[:,-2:].astype(int)

# redistribution changes
Redistribution_SA1_changes_2024 = Redistribution_SA1s_2024.loc[Redistribution_SA1s_2024['new_div']!=Redistribution_SA1s_2024['old_div'],][['SA1_CODE21','new_div','old_div']]
Redistribution_SA1_changes_2024.to_csv("Redistribution_SA1_changes2024.csv", index=False)

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
    changed_spread_divs = SA1_Correspondence_2016_2021.loc[(SA1_Correspondence_2016_2021['SA1_CODE16'].astype(int).isin(spread_over_divs_SA1s)) & (SA1_Correspondence_2016_2021['SA1_CODE16'] != SA1_Correspondence_2016_2021['SA1_CODE21']),]
    changed_spread_divs_set = set(changed_spread_divs.iloc[:,0].unique())
    changed_spread_divs_set = {int(x) for x in changed_spread_divs_set if str(x).startswith(("1","2","5","7"))} # only redistr
    changed_spread_divs_newSA1s_set = {int(x) for x in set(SA1_Correspondence_2016_2021.loc[SA1_Correspondence_2016_2021['SA1_CODE16'].astype(int).isin(changed_spread_divs_set),]['SA1_CODE21'])}

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


print("Higgins stuff")

SA1s_From_Higgins = VIC_SA1s_Redistribution.loc[VIC_SA1s_Redistribution["old_div"]=="Higgins","SA1_CODE21"].tolist()
print(sorted(SA1s_From_Higgins), len(SA1s_From_Higgins))
Higgins_SA1_Correspondence_2016_2021 = SA1_Correspondence_2016_2021.loc[SA1_Correspondence_2016_2021["SA1_CODE21"].isin(SA1s_From_Higgins),]
Dodgy_SA1s = Higgins_SA1_Correspondence_2016_2021.loc[Higgins_SA1_Correspondence_2016_2021["RATIO_FROM_TO"] < 1,]
print(Dodgy_SA1s, Dodgy_SA1s.shape)

print("Aston stuff")
SA1s_to_Aston = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution["old_div"]=="Deakin") & (VIC_SA1s_Redistribution["new_div"]=="Aston"),"SA1_CODE21"].tolist()
print(sorted(SA1s_to_Aston), len(SA1s_to_Aston))
Aston_SA1_Correspondence_2016_2021 = SA1_Correspondence_2016_2021.loc[SA1_Correspondence_2016_2021["SA1_CODE21"].isin(SA1s_to_Aston),]
Dodgy_SA1s = Aston_SA1_Correspondence_2016_2021.loc[Aston_SA1_Correspondence_2016_2021["RATIO_FROM_TO"] < 1,]
print(Dodgy_SA1s, Dodgy_SA1s.shape)
print(time.time()-start, "seconds")

import pdb;pdb.set_trace()
