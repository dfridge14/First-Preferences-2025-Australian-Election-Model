import pandas as pd
import numpy as np
import os,time

os.chdir('C:\\Dania\\2024\\Australian Election')

election_years = ['1990','1993','1996','1998','2001','2004','2007','2010','2013','2016','2019','2022']
StateAbs = ['NSW','VIC','QLD','WA','SA','TAS','NT','ACT']

state_correlation_data = pd.DataFrame(columns =StateAbs)

for data_year in reversed(election_years[5:]):

    State_FP_swings = pd.read_csv(f'{data_year}HouseFirstPrefsByStateByParty.csv', skiprows=1, index_col = None)[['StateAb','PartyAb','TotalSwing']]

    Parties_list = ['LP','NP','LNP','LNQ','CLP','ALP','GRN','GVIC','UAPP','PUP','ON','HAN']

    State_FP_swings = State_FP_swings.loc[State_FP_swings['PartyAb'].isin(Parties_list),]

    # combine LP and NP
    Coalition = ['LP','NP','LNP','LNQ','CLP']

    State_FP_swings['PartyAb'] = State_FP_swings['PartyAb'].apply(lambda x: 'COAL'+data_year[-2:] if x in Coalition else x+data_year[-2:])

    State_FP_swings = State_FP_swings.groupby(['StateAb', 'PartyAb'], as_index=False)['TotalSwing'].sum()

    # remove any nan rows in the df!

    State_FP_swings_wide = State_FP_swings.pivot(index='PartyAb', columns='StateAb', values='TotalSwing').dropna()
    #import pdb;pdb.set_trace()

    state_correlation_data = pd.concat([state_correlation_data,State_FP_swings_wide])

# IDEAS: Is it better to use 2 majot parties or minors to reduce dependencies???
# 2001:
swings_2001 = {'ALP01': [-3.67,-2.72,-1.40,-0.29,-0.74,-1.73,0.6,-3.66],
               'COAL01': [4.41,2.35,4.74,2.95,3.33,-1.07,0.94,1.87],
               'GRN01': [2.09,3.81,1.11,0.93,3.15,2.25,0.99,2.89],
               'AUD01': [0.08,0.23,0.29,0.7,0.4,1.22,0.14,0.63],
               'ON01': [-4.19,-2.44,-7.28,-3.01,-5.05,0.42,-4.31,-2.33]}

swings_1998 = {'ALP98': [0.56,1.45,2.91,1.48,-0.35,4.6,-1.2,2.73],
               'COAL98': [-7.14,-4.72,-14.35,-5.9,-7.42,-6.29,-5.44,-10.4],
               'GRN98': [0.24,0.19,-2.42,-0.26,-2.46,-0.78,-3.23,-4.52],
               'ON98': [8.96,3.72,14.35,9.27,9.80,2.46,8.14,5.08]}
            # 'AUD98': [-2.38,-1.33,-2.69,-1.61,-0.06,-0.83,np.nan,np.nan] ---> perhaps use senate to make this up

swings_1996 = {'ALP96': [-8.76,-3.53,-6.77,-4.62,-4.01,-2.45,-11.81,-5.43],
               'COAL96': [4.01,-0.7,8.37,-3.9,4.33,2.81,0.35,6.73],}


swings_1993 = {'ALP93': [7.20,9.38,-1.10,4.01,0.89,7.02,5.35,8.46],
               'COAL93': [-1.73,-0.46,0.72,2.93,3.31,-6.58,4.15,-0.64],}

swings_1990 = {'ALP90': [4.01,-9.8,-3.37,-12.14,-6.6,-3.25,3.1,-9.04],
               'COAL90': [-4.72,+1.4,-4.6,-2.36,0.14,-2.44,4.5,4.55],}

swings_1987 = {'ALP87': [-3.09,-1.99,0.9,-0.71,-3.83,-0.43,2.0,0.58],
               'COAL87': [1.74,1.13,-0.99,1.64,4.16,-0.08,4.3,-1.91],}

for swings in [swings_2001,swings_1998,swings_1996,swings_1993,swings_1990,swings_1987]:

    state_correlation_data = pd.concat([state_correlation_data,pd.DataFrame(swings, index = StateAbs).T])


import pdb;pdb.set_trace()
correlation_matrix = state_correlation_data.corr()
eigenvalues = np.linalg.eigvals(correlation_matrix)

if np.any(eigenvalues<0):
    raise ValueError("Not positive definite!")

state_correlation_data.to_csv('State_Sample_Correlation_matrix.csv', index=True)


# AUD can perhaps get more data from senate elections



