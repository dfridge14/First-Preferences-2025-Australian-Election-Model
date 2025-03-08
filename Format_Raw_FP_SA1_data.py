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


if os.name == "nt":  # Windows
    base_dir = Path('C:\\Dania\\2024\\Australian Election')
else:  # Ubuntu/Linux/macOS
    base_dir = Path.home() / "Australian Election"

os.chdir(base_dir)

data_year = "2016"
SA1_year_dict = {'2022':'2016','2019':'2016','2016':'2011','2013':'2011','2010':'2006'}


# creates 3 files  
# xxxx_PP_data - a df of all necessary information about PPs at the given election
# xxxxFirstPrefsByPPComplete - long form df or FP votes per candidate per polling booth
# xxxxVotesBySA1 - file of raw Votes By SA1 from the AEC


PP_data = pd.read_csv(f'{data_year}GeneralPollingPlaces.csv',skiprows=1, index_col = None)
PP_data = PP_data.iloc[:,[2,3,4,5,-2,-1]].rename(columns={'DivisionNm': 'div_nm', "PollingPlaceID": "pp_id", "PollingPlaceTypeID": "pp_type", "PollingPlaceNm": "pp_nm", "Latitude": "Lat", "Longitude": "Long"})

PPVC_ids = PP_data.loc[(PP_data["pp_type"]==5) & (~PP_data["pp_nm"].str.startswith("EAV")),"pp_id"].unique() # includes 22 with no votes
PB_ids = PP_data.loc[PP_data["pp_type"]==1,"pp_id"].unique() # includes 6 with no votes
EAV_ids = PP_data.loc[PP_data["pp_nm"].str.startswith("EAV"),"pp_id"].unique().tolist()
RMT_ids = PP_data.loc[PP_data["pp_type"]==3,"pp_id"].unique().tolist()
OMT_ids = PP_data.loc[PP_data["pp_type"]==4,"pp_id"].unique().tolist()
SHT_ids = PP_data.loc[PP_data["pp_type"]==2,"pp_id"].unique().tolist()
Exception_ids = EAV_ids + RMT_ids + OMT_ids + SHT_ids


print("checking size of pp_ids!")
print(len(PPVC_ids),len(PB_ids),len(EAV_ids),len(RMT_ids),len(OMT_ids),len(SHT_ids))
print(sum([len(PPVC_ids),len(PB_ids),len(EAV_ids),len(RMT_ids),len(OMT_ids),len(SHT_ids)]))
print(len(PP_data.loc[:,"pp_id"].unique().tolist()))

assert len(PP_data.loc[:,"pp_id"].unique().tolist()) == sum([len(PPVC_ids),len(PB_ids),len(EAV_ids),len(RMT_ids),len(OMT_ids),len(SHT_ids)])



# Now, assign values of 5 to PPVC, 1 to PB, 2-4 and starting with EAVs to Other and give 0 pp_id
PP_data.loc[:,'Booth_type'] = ''
PP_data.loc[(PP_data["pp_type"]==5) & (~PP_data["pp_nm"].str.startswith("EAV")),"Booth_type"] = 'PPVC'
PP_data.loc[(PP_data["pp_type"]==1),'Booth_type'] = 'PB'
PP_data.loc[PP_data["pp_type"].isin([2,3,4]) | (PP_data["pp_type"]==5) & (PP_data["pp_nm"].str.startswith("EAV")),['Lat','Long','Booth_type']] = np.nan,np.nan,'Other'


PP_data.drop(columns = 'pp_type',inplace=True)



# Now, for FPByPPComplete

def create_FP_APPP_Votes_by_Div():
    ### collects non-polling place votes from Absent,Postal,Prepoll,Provisional

    FP_Vote_Type_by_Div = pd.read_csv(f"{data_year}HouseFirstPrefsByCandidateByVoteType.csv", skiprows=1).rename(columns={'DivisionNm':'div_nm'}) 

    FP_APPP_Votes_by_Div = FP_Vote_Type_by_Div[["div_nm","PartyAb",'PartyNm','BallotPosition',"AbsentVotes",'ProvisionalVotes','PostalVotes','PrePollVotes']]
    FP_APPP_Votes_by_Div.loc[:,'votes'] = FP_APPP_Votes_by_Div.iloc[:,-4:].sum(axis=1)
    FP_APPP_Votes_by_Div = FP_APPP_Votes_by_Div[["div_nm","PartyAb",'PartyNm','BallotPosition','votes']]


    # format similarly to FP_by_PP
    FP_APPP_Votes_by_Div.loc[:,"Booth_type"] = "Other"
    FP_APPP_Votes_by_Div.loc[:,"pp_id"] = 0
    FP_APPP_Votes_by_Div.loc[:,"pp_nm"] = "Other"

    # sort in alphabetical order of div_nm
    FP_APPP_Votes_by_Div = FP_APPP_Votes_by_Div.sort_values(by=["div_nm",'BallotPosition'],kind='mergesort') # alphabetical order by Division Name

    return FP_APPP_Votes_by_Div

FP_APPP_Votes_by_Div = create_FP_APPP_Votes_by_Div()

import ipdb;ipdb.set_trace()

states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
FP_By_PP = pd.DataFrame(columns=['div_nm','pp_id','pp_nm','PartyAb','votes'])

for state in states:
    state_df = pd.read_csv(f'{data_year}HouseStateFirstPrefsByPollingPlace-{state}.csv', skiprows=1, index_col=None)

    state_df.rename(columns = {'DivisionNm': 'div_nm', "PollingPlaceID": "pp_id", "PollingPlace": "pp_nm", "OrdinaryVotes":'votes'}, inplace=True)
    state_df = state_df[['div_nm','pp_id','pp_nm','PartyAb','PartyNm','BallotPosition','votes']]

    FP_By_PP= pd.concat([FP_By_PP,state_df], ignore_index=True)


#import ipdb;ipdb.set_trace()



# check 0 vote PPs
Collated_Votes_by_PP = FP_By_PP.groupby(["pp_id","pp_nm","div_nm"], as_index=False)["votes"].sum()
PPs_0_votes_ids = Collated_Votes_by_PP.loc[Collated_Votes_by_PP["votes"]==0,]["pp_id"].tolist() # pp_ids that have 0 votes
print(PPs_0_votes_ids)



FP_By_PP = FP_By_PP[~FP_By_PP["pp_id"].isin(PPs_0_votes_ids)] 

# concatenate Ordinary and non-Ordinary votes into full df

FP_By_PP_merged = FP_By_PP.merge(PP_data[['pp_id','Booth_type','Lat','Long']],on = 'pp_id',how='left')
#import ipdb;ipdb.set_trace()

FP_By_PP_merged.loc[FP_By_PP_merged['Booth_type']=='Other',['pp_id','pp_nm']] = 0,'Other'
#import ipdb;ipdb.set_trace()


FP_By_PP_merged = pd.concat([FP_By_PP_merged,FP_APPP_Votes_by_Div], ignore_index=True) # Lat and Long for APPP votes set as np.nan


# correct for missing party names like Steve Khouw
no_names = FP_By_PP_merged.loc[FP_By_PP_merged["PartyNm"].isna(),['div_nm','PartyAb']].drop_duplicates()

error = 0
if error and not no_names.empty:
    raise ValueError("No-name candidate - Please convert to Independent")

FP_By_PP_merged.loc[FP_By_PP_merged["PartyNm"].isna(),'PartyAb'] = 'IND'
FP_By_PP_merged['PartyAb'] = FP_By_PP_merged['PartyAb'].fillna('INFORMAL')
FP_By_PP_merged = FP_By_PP_merged.drop('PartyNm', axis=1)

import ipdb;ipdb.set_trace()

# merge together


FP_By_PP_grouped = FP_By_PP_merged.groupby(['div_nm','pp_id','PartyAb','BallotPosition','Booth_type'], as_index=False)["votes"].sum() #don't group by pp_nm due to 'Other', and Lat and Long grouping leads to nans making Others rows disappear!
FP_By_PP_grouped = FP_By_PP_grouped.sort_values(by=["div_nm",'pp_id','BallotPosition'],kind='mergesort')

FP_By_PP_Complete = FP_By_PP_grouped.merge(FP_By_PP_merged.drop(columns=['votes']), on=FP_By_PP_grouped.columns[:-1].tolist(), how='left').drop_duplicates()

FP_By_PP_Complete.drop('BallotPosition', axis=1, inplace=True)
import ipdb;ipdb.set_trace()




FP_By_PP_Complete.to_csv(f'{data_year}FirstPrefsByPPComplete.csv', index = False)




# Now, onto SA1s

SA1_By_PP_full = pd.read_csv(f"{data_year}VotesBySA1.csv")

SA1_col_name = 'ccd_id' if data_year=='2022' else 'SA1_id'

SA1_By_PP = SA1_By_PP_full[["div_nm",SA1_col_name,"pp_id","votes"]]
SA1_By_PP = SA1_By_PP.rename(columns={SA1_col_name: f'SA1_CODE{SA1_year_dict[data_year][-2:]}'})

# convert Other pp ids to 0
SA1_By_PP_merged = SA1_By_PP.merge(PP_data[['pp_id','Booth_type']],on = 'pp_id',how='left')
SA1_By_PP_merged.loc[SA1_By_PP_merged['Booth_type']=='Other','pp_id'] = 0
SA1_By_PP_grouped = SA1_By_PP_merged.groupby(["div_nm",f"SA1_CODE{SA1_year_dict[data_year][-2:]}","pp_id"], as_index=False)["votes"].sum()
import ipdb;ipdb.set_trace()

SA1_By_PP_grouped.to_csv(f'{data_year}SA1ByPPComplete.csv', index = False)

import ipdb;ipdb.set_trace()


# Finally, submit PP_data

PP_data.loc[PP_data['Booth_type']=='Other',['pp_id','pp_nm']] = 0,'Other'
PP_data.drop_duplicates(inplace=True)
PP_data.to_csv(f'{data_year}_PP_data.csv', index=False)