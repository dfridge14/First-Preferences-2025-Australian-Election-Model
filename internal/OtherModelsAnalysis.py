import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


os.chdir("/home/dania-freidgeim/Australian Election")

Yougov_results = pd.read_excel("Publication-YouGov-MRP-Results.xlsx")

import pdb; pdb.set_trace()


# Obtain averages per state (assume electorates in each state have same number of voters)

div_to_state = pd.read_csv(f"2025HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'}).set_index('div_nm')
Yougov_FPs = Yougov_results.merge(div_to_state, left_on = 'CED', right_index = True)[['CED','StateAb','CoalitionShare','LaborShare','GreensShare','OneNationShare','IndependentShare','OtherShare']]
Yougov_FPs.columns = ['div_nm','StateAb','COAL','ALP','GRN','ON','IND','OTH']

# state polls only capture 'OTH' vote, does not isolate 'IND'. Will try at first ignoring IND (probably good idea due to poor performance of IND electorates)
Yougov_FPs['OTH'] = Yougov_FPs['OTH'] + Yougov_FPs['IND']
Yougov_FPs = Yougov_FPs.drop(columns=['IND'])
Yougov_FPs.iloc[:,2:] = Yougov_FPs.iloc[:,2:].div(Yougov_FPs.iloc[:,2:].sum(axis=1), axis=0) # normalise

State_MRP_averages = Yougov_FPs.groupby('StateAb')[['COAL','ALP','GRN','ON','OTH']].mean()
State_MRP_averages.loc[:,'Election'] = '2025' + State_MRP_averages.index
State_MRP_averages.loc['NAT',:] = [0.311,0.314,0.126,0.091,0.158,'2025NAT']


parties = ['ALP', 'GRN', 'ON', 'OTH']
ref_col = 'COAL'

alr = np.log(State_MRP_averages[parties].replace(0, np.nan).div(State_MRP_averages[ref_col], axis=0))
    
# Subtract the NAT row from all states, then drop the NAT row
deviations = alr.sub(alr.loc['NAT']).drop('NAT')

# Format metadata columns to match your target image
deviations['Election_year'] = State_MRP_averages['Election'].str[:4].drop('NAT')
deviations['State'] = deviations.index

State_deviations = deviations.reset_index(drop=True)

State_deviations.to_csv("YouGov_State_deviations_ALR.csv", index = False)
