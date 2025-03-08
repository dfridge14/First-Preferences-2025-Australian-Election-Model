import pandas as pd
import numpy as np
import os, time

# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


os.chdir('C:\\Dania\\2024\\Australian Election')

data_year = "2022"

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
PP_data.loc[PP_data["pp_type"].isin([2,3,4]) | (PP_data["pp_type"]==5) & (PP_data["pp_nm"].str.startswith("EAV")),['pp_id','pp_nm','Lat','Long','Booth_type']] = 0, 'Other',np.nan,np.nan,'Other'


PP_data.drop(columns = 'pp_type',inplace=True)
PP_data.drop_duplicates(inplace=True)

PP_data.to_csv(f'{data_year}_PP_data.csv', index=False)

import pdb;pdb.set_trace()

