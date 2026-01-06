import pandas as pd
import numpy as np
import os, time
from pathlib import Path


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

x = 4

Final_x_HS_df2022 = pd.read_csv(f"2022Final_{x}_HS_df.csv", index_col = None)
Final_x_HS_df2019 = pd.read_csv(f"2019Final_{x}_HS_df.csv", index_col = None)
Final_x_HS_df2016 = pd.read_csv(f"2016Final_{x}_HS_df.csv", index_col = None)

data_years = ['2022','2019','2016']

combined_HS_df = pd.DataFrame(columns = ['div_nm', 'PartyAb', 'is_incumbent', 'is_historic_incumbent', 'elections_won','copied_PartyAb','House_Pct','Senate_Pct','Demographic','StateAb'])

Final_x_HS_df_year_list = []

for data_year in data_years:
    Final_x_HS_df_year = pd.read_csv(f"{data_year}Final_{x}_HS_df.csv", index_col = None)
    Demographic_Classification_State_df = pd.read_csv(f'{data_year}DemographicClassification.csv', index_col=None)

    Final_x_HS_df_year = Final_x_HS_df_year.merge(Demographic_Classification_State_df, on = 'div_nm', how='left')
    Final_x_HS_df_year.loc[:,'PartyAb'] = Final_x_HS_df_year['copied_PartyAb'].values
    Final_x_HS_df_year = Final_x_HS_df_year.drop('copied_PartyAb', axis = 1)

    # add year suffixes to differentiate division observation years
    Final_x_HS_df_year.loc[:,'div_nm'] = Final_x_HS_df_year.loc[:,'div_nm'] + data_year[-2:]

    Final_x_HS_df_year_list.append(Final_x_HS_df_year)

combined_HS_df = pd.concat(Final_x_HS_df_year_list, ignore_index=True)

import pdb;pdb.set_trace()



# model just incumbent effects:
# make CLP into LNP - in total get ALP/LP/LNP/Other (NP + GRN) 
incumbent_df_for_R_model = combined_HS_df.loc[combined_HS_df['is_incumbent'] == 1,].copy()[['div_nm','PartyAb','StateAb','Demographic','elections_won','House_Pct','Senate_Pct']]
incumbent_df_for_R_model.loc[incumbent_df_for_R_model['PartyAb']=='CLP','PartyAb'] = 'LNP'
incumbent_df_for_R_model.loc[~incumbent_df_for_R_model['PartyAb'].isin(['ALP','LP','LNP']),'PartyAb'] = 'Other'
incumbent_df_for_R_model.rename(columns={'PartyAb':'PartyCat'},inplace=True)

incumbent_df_for_R_model.loc[incumbent_df_for_R_model['div_nm']=='Monash16',['StateAb','Demographic']] = 'VIC','Rural'
incumbent_df_for_R_model.loc[incumbent_df_for_R_model['div_nm']=='Spence16',['StateAb','Demographic']] = 'SA','Outer Metropolitan'
incumbent_df_for_R_model.loc[incumbent_df_for_R_model['div_nm']=='Macnamara16',['StateAb','Demographic']] = 'VIC','Inner Metropolitan'
incumbent_df_for_R_model.loc[incumbent_df_for_R_model['div_nm']=='Cooper16',['StateAb','Demographic']] = 'VIC','Inner Metropolitan'

incumbent_df_for_R_model.loc[:,'elections_won'] -= 1

import pdb;pdb.set_trace()

incumbent_df_for_R_model.to_csv(f'Incumbent_House_Senate_Final{x}_for_R.csv', index = False)

# Find average starting incumbency advantage (Inner Metropolitan and elections_won == 0)
incumbent_df_for_R_model.loc[:,'Diff_Pct'] = incumbent_df_for_R_model.loc[:,'House_Pct'].values - incumbent_df_for_R_model.loc[:,'Senate_Pct'].values
IND_average_advantage = incumbent_df_for_R_model.loc[(incumbent_df_for_R_model['elections_won']==0)& (incumbent_df_for_R_model['Demographic']=='Inner Metropolitan'),]['Diff_Pct'].mean()
print('average startign incumbency advantage: x =', x, IND_average_advantage)








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


party_category_dict = make_party_category_dict()

# label each seat with its INC (ALP,LP,LNP,Other)
incumbent_combined_HS_df = combined_HS_df.copy()
incumbent_combined_HS_df.loc[incumbent_combined_HS_df['PartyAb']=='CLP','PartyAb'] = 'LNP'
incumbent_party = incumbent_combined_HS_df.loc[incumbent_combined_HS_df['is_incumbent']==1,].groupby('div_nm')['PartyAb'].agg('sum').rename('incumbent_party')
incumbent_combined_HS_df = incumbent_combined_HS_df.merge(incumbent_party, on = 'div_nm',how='left')

# retain only incumbents (and remove double incumbent ALPLP in Cowan22)
incumbent_combined_HS_df = incumbent_combined_HS_df.loc[~incumbent_combined_HS_df['incumbent_party'].isna(),]
incumbent_combined_HS_df = incumbent_combined_HS_df.loc[~(incumbent_combined_HS_df['incumbent_party']=='ALPLP'),]
incumbent_combined_HS_df = incumbent_combined_HS_df.loc[~(incumbent_combined_HS_df['div_nm']=='Mayo19'),]


incumbent_combined_HS_df.loc[~incumbent_combined_HS_df['incumbent_party'].isin(['ALP','LP','LNP']),'incumbent_party'] = 'Other'

incumbent_combined_HS_df.loc[incumbent_combined_HS_df['div_nm']=='Monash16',['StateAb','Demographic']] = 'VIC','Rural'
incumbent_combined_HS_df.loc[incumbent_combined_HS_df['div_nm']=='Spence16',['StateAb','Demographic']] = 'SA','Outer Metropolitan'
import pdb;pdb.set_trace()

#incumbent_combined_HS_df.loc[incumbent_combined_HS_df['is_incumbent']==1,'elections_won'] -= 1

incumbent_combined_HS_df.loc[:,"Ideology"] = incumbent_combined_HS_df.loc[:,"PartyAb"].replace(party_category_dict)
incumbent_combined_HS_df.loc[:,'Diff_Pct'] = incumbent_combined_HS_df.loc[:,'House_Pct'].values - incumbent_combined_HS_df.loc[:,'Senate_Pct'].values

non_incumbent_df = incumbent_combined_HS_df.loc[incumbent_combined_HS_df['is_incumbent']==0,]

# group by the incumbent party and Ideology of minor party for estimate of weight
non_incumbent_df.groupby(['incumbent_party','Ideology'], as_index=False)['Diff_Pct'].agg('mean')
non_incumbent_df.groupby(['incumbent_party','Ideology'], as_index=False)['is_incumbent'].agg('count')

non_incumbent_df = non_incumbent_df[['incumbent_party','Ideology','Diff_Pct']]

#non_incumbent_df.to_csv('Non-incumbent_HS_for_R.csv', index=False)

import pdb;pdb.set_trace()

# now get averages for IND incumbent:
IND_Left = non_incumbent_df.loc[non_incumbent_df['Ideology']=='Left','Diff_Pct'].mean()
IND_ALP = non_incumbent_df.loc[non_incumbent_df['Ideology']=='ALP','Diff_Pct'].mean()
IND_Centre = non_incumbent_df.loc[non_incumbent_df['Ideology']=='Centre','Diff_Pct'].mean()
IND_COAL = non_incumbent_df.loc[non_incumbent_df['Ideology']=='COAL','Diff_Pct'].mean()
IND_Right = non_incumbent_df.loc[non_incumbent_df['Ideology']=='Right','Diff_Pct'].mean()








import pdb;pdb.set_trace()




# count how many incumbents there are 
incumbent_counts = combined_HS_df.groupby('div_nm')['is_incumbent'].sum().rename('incumbents_in_div')
hist_incumbent_counts = combined_HS_df.groupby('div_nm')['is_historic_incumbent'].sum().rename('hist_incumbents_in_div')

combined_HS_df = combined_HS_df.merge(incumbent_counts, on = 'div_nm',how='left')
combined_HS_df = combined_HS_df.merge(hist_incumbent_counts, on = 'div_nm',how='left')
combined_HS_df.loc[:,"Diff_Pct"] = combined_HS_df.loc[:,"House_Pct"] - combined_HS_df.loc[:,"Senate_Pct"]



# PARTY EFFECTS FOR NON-INCUMBENTS
combined_HS_df.loc[(combined_HS_df["is_historic_incumbent"]!= 1) & (combined_HS_df["incumbents_in_div"]==1) & (combined_HS_df["is_incumbent"] == 0),].groupby("PartyAb")['Diff_Pct'].mean()
# mean and var for non-incumbents
combined_HS_df.loc[(combined_HS_df["is_historic_incumbent"]!= 1)& (combined_HS_df["incumbents_in_div"]==1) & (combined_HS_df["is_incumbent"] == 0),"Diff_Pct"].mean() 
# individual years
combined_HS_df.loc[(combined_HS_df["is_historic_incumbent"]!= 1)& (combined_HS_df["incumbents_in_div"]==1) & (combined_HS_df["is_incumbent"] == 1) & (combined_HS_df["div_nm"].str.endswith('16')),"Diff_Pct"].mean()

import pdb;pdb.set_trace()

# see how much incumbency advantage is transferred uniformly vs not uniformly!

incumbent_combined_HS_df = combined_HS_df.loc[combined_HS_df['is_incumbent']==1,].groupby('div_nm')['PartyAb'].agg('first').rename('incumbent_party')
combined_HS_df = combined_HS_df.merge(incumbent_combined_HS_df, on = 'div_nm',how='left')

combined_HS_df.loc[combined_HS_df['PartyAb']=='GRN',].groupby('incumbent_party')['Diff_Pct'].agg('mean')










import pdb;pdb.set_trace()

Final_x_Incumbency = pd.read_csv("2022Final_x_for_Incumbency.csv", index_col=None)


# INCONSISTENCIES IN ORIGINAL DESCRIPTION, SO WILL USE TEMPORARY MERGED FILE
merged = Final_x_Incumbency.merge(Final_x_HS_df2022.iloc[:,[0,1,-2,-1]], on=["div_nm","PartyAb","House_Pct"], how='left')
merged.loc[:,"Diff_Pct"] = merged.loc[:,"House_Pct"] - merged.loc[:,"Senate_Pct"]




true_counts = merged.groupby('div_nm')['is_incumbent'].sum()

merged = merged.merge(true_counts, on = 'div_nm',how='left')




# Count how many times each true count appears
grouped_counts = true_counts.value_counts().reset_index()
grouped_counts.columns = ['num_of_True', 'count_of_divs']

import pdb;pdb.set_trace()