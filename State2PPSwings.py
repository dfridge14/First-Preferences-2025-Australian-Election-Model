import pandas as pd

import numpy as np
from itertools import product
import os,time


# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler

os.chdir("C:\\Dania\\2024\\Australian Election\\ABS downloads for Similarity\\States") 

NO_OF_STATES = 8
MEDIAN_AGE_THRESHOLD = 1
RENT_THRESHOLD = 2
FT_THRESHOLD = 2 # Full Time
NR_THRESHOLD = 2 # no religion
AUS_THRESHOLD = 2
HSIZE_THRESHOLD = 0.2
PROFESSIONALS_THRESHOLD = 2
POPULATION_RATIO = 2/3

TPP_ROLLING_AVERAGE_NO = 4

MIN_SIMILARITY = 0

election_years = ['2004','2007','2010','2013','2016','2019','2022']
election_years_extended = ['1993','1996','1998','2001'] # for TPP rolling average
election_years_extended.extend(election_years) # add older info

census_years = ['2006','2011','2016','2021']

election_year_to_census_year_dict = {'2004':'2006','2007':'2006','2010':'2011','2013':'2011','2016':'2016','2019':'2016','2022':'2021'}


binary = 0


def state_by_state_matrix(characteristic_df, threshold):
    ### characteristic_df must be an n * 2 df with 1st row corresponding to states or divisions, second row to characteristics
    n = len(characteristic_df)
    comparison_matrix = np.zeros((n, n), dtype=int)

    # Compare all row pairs
    for i, j in product(range(n), repeat=2):  # Compare all (i, j) pairs
        if threshold:
            if abs(characteristic_df.iloc[i, 1] - characteristic_df.iloc[j, 1]) <= threshold:# Apply condition
                comparison_matrix[i, j] = 1 
        else:
            if characteristic_df.iloc[i, 1] == characteristic_df.iloc[j, 1]:
                comparison_matrix[i, j] = 1 

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_matrix, index=characteristic_df.iloc[:,0], columns=characteristic_df.iloc[:,0])   
    #print(comparison_df)

    return comparison_df


def state_by_state_matrix_ratio(characteristic_df, threshold):
    
    n = len(characteristic_df)
    comparison_matrix = np.zeros((n, n), dtype=int)

    # Compare all row pairs
    for i, j in product(range(n), repeat=2):  # Compare all (i, j) pairs
        i_val, j_val = characteristic_df.iloc[i, 1], characteristic_df.iloc[j, 1]
        if ((i_val<=j_val) & (i_val/j_val >= threshold)) | ((j_val<=i_val) & (j_val/i_val >= threshold)):# Apply if smaller/larger is above threshold (e.g. above 0.7)
            comparison_matrix[i, j] = 1 

    # Convert to DataFrame 
    comparison_df = pd.DataFrame(comparison_matrix, index=characteristic_df.iloc[:,0], columns=characteristic_df.iloc[:,0])   
    #print(comparison_df)

    return comparison_df





def weighted_median(characteristics, freqs):
    cumsum = freqs[:-1].cumsum()
    median_idx = cumsum.searchsorted(freqs.iloc[-1] / 2)  # Find where 50% is reached
    return characteristics[median_idx]

def weighted_median_without_total(characteristics, freqs):
    cumsum = freqs[:-1].cumsum()
    median_idx = cumsum.searchsorted(freqs.sum() / 2)  # Find where 50% is reached
    return characteristics[median_idx]

def create_StateAb_dict():

    StateAb_df = pd.read_csv("2022HouseTppByState.csv", skiprows=1).iloc[:,:2]

    StateAb_dict = {}
    for i in range(len(StateAb_df)):
        StateAb_dict[StateAb_df.iloc[i,1]] = StateAb_df.iloc[i,0]
    
    return StateAb_dict

StateAb_dict = create_StateAb_dict()


def calculate_range_mean(range_str):

    # Remove the dollar sign and split the string on the dash
    lower, upper = range_str.replace('$', '').split('-')
    # Convert to integers and calculate the mean
    return (int(lower) + int(upper)) / 2


def get_characteristic_matrices(binary):


    # 1. median age
    Median_Age_matrix_dict = {} # keys are census_years
    for data_year in census_years:

        print(data_year)

        Age_df = pd.read_csv(f"{data_year}StateAge.csv", skiprows=9, skipfooter=8, engine='python')
        Age_df = Age_df.iloc[1:,:-1].rename(columns=lambda x: x.replace(" years", "") if " years" in x else x).rename(columns={'1 year':'1','AGEP Age': 'StateAb','AGEP Age in Single Years':'StateAb'})
        Age_df.iloc[:, 0] = Age_df.iloc[:, 0].replace(StateAb_dict)
        # Apply median to each state
        ages = Age_df.columns[1:-1].astype(int)
        Age_df.loc[:,"Median_age"] = Age_df.iloc[:,1:].apply(lambda row: weighted_median(ages, row), axis=1)
        Age_df = Age_df.iloc[:NO_OF_STATES,[0,-1]]
        #Age_df.iloc[:,-1] = Age_df.iloc[:,-1].astype(int)

        #scaled_characteristic = 0.2 + (Age_df - Age_df.min()) * (0.8 - 0.2) / (Age_df.max() - Age_df.min())
        ser = Age_df.set_index('StateAb')
        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY)  / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            Median_Age_matrix_dict[data_year] = state_by_state_matrix(Age_df, MEDIAN_AGE_THRESHOLD)
        else:
            Median_Age_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)


    # 2. HIED Median (income)
    Median_HIED_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        HIED_df = pd.read_csv(f"{data_year}StateHIED.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:-5] # filter out not applicable columns
        HIED_df.iloc[:, 0] = HIED_df.iloc[:, 0].replace(StateAb_dict)
    
        HIED_df.loc[:,'Median_HIED'] =  HIED_df.iloc[:,1:].apply(lambda row: weighted_median_without_total(HIED_df.columns[1:], row), axis = 1)
        HIED_df = HIED_df.iloc[:,[0,-1]].rename(columns={HIED_df.columns[0]: "StateAb"})

        ser = HIED_df.set_index('StateAb')
        #import pdb;pdb.set_trace()

        if data_year == '2006':
            ser["Median_HIED"] = round(ser["Median_HIED"].str.replace(',','').apply(calculate_range_mean))
        else:
            ser.loc[:,"Median_HIED"] = ser.loc[:,"Median_HIED"].str.replace(',','').str.split(' ').str[0]
            ser["Median_HIED"] = round(ser["Median_HIED"].apply(calculate_range_mean))

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])


        if binary:
            Median_HIED_matrix_dict[data_year] = state_by_state_matrix(HIED_df, 0)
        else:
            Median_HIED_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)
        #import pdb;pdb.set_trace()

    # 3. Median Morgage Repayment
    Median_Mortgage_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        Mortgage_df = pd.read_csv(f"{data_year}StateMortgageRep.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:-4] # filter out not applicable columns
        Mortgage_df.iloc[:, 0] = Mortgage_df.iloc[:, 0].replace(StateAb_dict)
    
        Mortgage_df.loc[:,'Median_Mortgage'] =  Mortgage_df.iloc[:,1:].apply(lambda row: weighted_median_without_total(Mortgage_df.columns[1:], row), axis = 1)
        Mortgage_df = Mortgage_df.iloc[:,[0,-1]].rename(columns={Mortgage_df.columns[0]: "StateAb"})

        ser = Mortgage_df.set_index('StateAb')

        if data_year == '2006':
            ser["Median_Mortgage"] = round(ser["Median_Mortgage"].str.replace(',','').apply(calculate_range_mean))
        else:
            ser.loc[:,"Median_Mortgage"] = ser.loc[:,"Median_Mortgage"].str.replace(',','').str.split(' ').str[0]
            ser["Median_Mortgage"] = round(ser["Median_Mortgage"].apply(calculate_range_mean))

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            Median_Mortgage_matrix_dict[data_year] = state_by_state_matrix(Mortgage_df, 0)
        else:
            Median_Mortgage_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)

        
        #import pdb;pdb.set_trace()

    # 4. Rent Percent
    Rent_Percent_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        Rent_df = pd.read_csv(f"{data_year}StateRent.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:-4] # filter out not applicable columns - always 1 more than you think for some reason
        Rent_df.iloc[:, 0] = Rent_df.iloc[:, 0].replace(StateAb_dict)

        Rent_df.loc[:,'Rent%'] =  Rent_df.loc[:,'Rented']/Rent_df.iloc[:,1:].sum(axis=1)*100  # percentage of renteds out of valid responses
        Rent_df = Rent_df.iloc[:,[0,-1]].rename(columns={Rent_df.columns[0]: "StateAb"})

        ser = Rent_df.set_index('StateAb')

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            Rent_Percent_matrix_dict[data_year] = state_by_state_matrix(Rent_df, RENT_THRESHOLD)
        else:
            Rent_Percent_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)

        
       
        #import pdb;pdb.set_trace()
        
    # 5. Full time Percent
    FT_Percent_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        FT_df = pd.read_csv(f"{data_year}StateEmployment.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:-6] # filter out not applicable columns
        FT_df.iloc[:, 0] = FT_df.iloc[:, 0].replace(StateAb_dict)

        FT_df.loc[:,'FT%'] =  FT_df.loc[:,'Employed, worked full-time']/FT_df.iloc[:,1:].sum(axis=1)*100  # percentage of FTeds out of valid responses
        FT_df = FT_df.iloc[:,[0,-1]].rename(columns={FT_df.columns[0]: "StateAb"})


        ser = FT_df.set_index('StateAb')

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            FT_Percent_matrix_dict[data_year] = state_by_state_matrix(FT_df, FT_THRESHOLD)
        else:
            FT_Percent_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)


        
        #import pdb;pdb.set_trace()

    # 6. Professionals Percent
    Prof_Percent_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        Prof_df = pd.read_csv(f"{data_year}StateOccupation.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:-6] # filter out not applicable columns
        Prof_df.iloc[:, 0] = Prof_df.iloc[:, 0].replace(StateAb_dict)

        # correct for change in later censuses
        Prof_df.loc[:,'Prof%'] =  Prof_df.loc[:,'Professionals']/Prof_df.iloc[:,1:].sum(axis=1)*100  # percentage of Profeds out of valid responses
        Prof_df = Prof_df.iloc[:,[0,-1]].rename(columns={Prof_df.columns[0]: "StateAb"})


        ser = Prof_df.set_index('StateAb')

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            Prof_Percent_matrix_dict[data_year] = state_by_state_matrix(Prof_df, PROFESSIONALS_THRESHOLD)
        else:
            Prof_Percent_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)


        
        #import pdb;pdb.set_trace()

    # 7. No Religion Percent
    NR_Percent_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        NR_df = pd.read_csv(f"{data_year}StateReligion.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:-4] # filter out not applicable columns
        NR_df.iloc[:, 0] = NR_df.iloc[:, 0].replace(StateAb_dict)

        # correct for change in later censuses
        if data_year in ['2006','2011']:
            NR_df.loc[:,'NR%'] =  NR_df.loc[:,'No Religion']/NR_df.iloc[:,1:].sum(axis=1)*100  # percentage of NReds out of valid responses
        else:
            NR_df = NR_df.iloc[:,:-1]
            NR_df.loc[:,'NR%'] =  NR_df.iloc[:,-1]/NR_df.iloc[:,1:].sum(axis=1)*100  # percentage of NReds out of valid responses
        
        NR_df = NR_df.iloc[:,[0,-1]].rename(columns={NR_df.columns[0]: "StateAb"})


        ser = NR_df.set_index('StateAb')

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            NR_Percent_matrix_dict[data_year] = state_by_state_matrix(NR_df, NR_THRESHOLD)
        else:
            NR_Percent_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)

        
        #import pdb;pdb.set_trace()


    # 8. Birthplace Percent
    Australian_Percent_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        Australian_df = pd.read_csv(f"{data_year}StateBirthAustralia.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:2] # only 2 cols - Australian Born freqs
        Birthplace_df = pd.read_csv(f"{data_year}StateBirthCountry.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,]
        Australian_df.loc[:,'Australia'] = Australian_df.iloc[:,1]/ (Birthplace_df.loc[:,'Total'])*100 # include not_stated in here, as that is what 
        Australian_df.iloc[:, 0] = Australian_df.iloc[:, 0].replace(StateAb_dict)

        Australian_df = Australian_df.iloc[:,[0,2]]
        Australian_df = Australian_df.iloc[:,[0,-1]].rename(columns={Australian_df.columns[0]: "StateAb"})


        ser = Australian_df.set_index('StateAb')

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            Australian_Percent_matrix_dict[data_year] = state_by_state_matrix(Australian_df, AUS_THRESHOLD)
        else:
            Australian_Percent_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)



        
        #import pdb;pdb.set_trace()

    # 9. Mean Household Size
    Mean_Hsize_matrix_dict = {} # keys are census_years
    for data_year in census_years:
        Hsize_df = pd.read_csv(f"{data_year}StateHouseholdSize.csv", skiprows=9, skipfooter=8, engine='python').iloc[1:NO_OF_STATES+1,:-2]

        if data_year == '2006':
            Hsize_df = Hsize_df.iloc[:,:-1]

        #import pdb;pdb.set_trace()
        sizes = Hsize_df.columns[1:].str.extract(r'(\d+)').astype(int).squeeze() # get only numeric parts of columns - remove 'person/s'
        mean_size = (Hsize_df.iloc[:,1:] * sizes.values).sum(axis=1) / Hsize_df.iloc[:,1:].sum(axis=1)
        Hsize_df.loc[:,'Mean_HSize'] = mean_size
        Hsize_df.iloc[:, 0] = Hsize_df.iloc[:, 0].replace(StateAb_dict)

        Hsize_df = Hsize_df.iloc[:,[0,-1]].rename(columns={Hsize_df.columns[0]: "StateAb"})

        ser = Hsize_df.set_index('StateAb')

        similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

        #import pdb;pdb.set_trace()

        if binary:
            Mean_Hsize_matrix_dict[data_year] = state_by_state_matrix(Hsize_df, HSIZE_THRESHOLD)

        else:
            Mean_Hsize_matrix_dict[data_year] = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)




        #import pdb;pdb.set_trace()

    

    # 10. Population
    Population_matrix_dict = {}
    State_populations = pd.read_csv("CensusPopulations.csv", skiprows=1) # columns are census years

    for data_year in census_years:

        ser = State_populations.loc[:,["StateAb",data_year]].set_index('StateAb')

        pairwise_ratios = np.minimum(ser.values[:, None], ser.values).squeeze(axis=-1) / np.maximum(ser.values[:, None], ser.values).squeeze(axis=-1)

        #import pdb;pdb.set_trace()

        if binary:
            Population_matrix_dict[data_year] = state_by_state_matrix_ratio(State_populations.loc[:,["StateAb",data_year]], POPULATION_RATIO)
        else:
            Population_matrix_dict[data_year] = pd.DataFrame(pairwise_ratios, index=ser.index, columns=ser.index)



    list_of_similarity_matrix_dicts = [Median_Age_matrix_dict, Median_HIED_matrix_dict, Median_Mortgage_matrix_dict, Rent_Percent_matrix_dict, FT_Percent_matrix_dict, Prof_Percent_matrix_dict, NR_Percent_matrix_dict, Australian_Percent_matrix_dict, Mean_Hsize_matrix_dict, Population_matrix_dict]

    return list_of_similarity_matrix_dicts

# 1. download and check all remaining 2011,2016,2021 censuses                                                       DONE
# 2. For each election_year since 2004 (maybe 2001?) combine all characteristic matrices into similarity matrix     DONE
# 3. Make a df to give to R for regression!

list_of_similarity_matrix_dicts = get_characteristic_matrices(binary)

#import pdb;pdb.set_trace()


def make_similarity_matrix(list_of_similarity_matrix_dicts, election_years, election_year_to_census_year_dict):
    ### average all characteristic similarity matrices and allocate to census years
    similarity_matrix_dict = {}

    for year in election_years:
        similarity_matrix_dict[year] = np.zeros((NO_OF_STATES,NO_OF_STATES)) # correct size, then add 

        census_year = election_year_to_census_year_dict[year]
        for i in range(len(list_of_similarity_matrix_dicts)):

            similarity_matrix_dict[year] += list_of_similarity_matrix_dicts[i][census_year] # add matrix for each of 10 characteristics

        similarity_matrix_dict[year] /= len(list_of_similarity_matrix_dicts) # 10 characteristics - average

    return similarity_matrix_dict

similarity_matrix_dict = make_similarity_matrix(list_of_similarity_matrix_dicts, election_years, election_year_to_census_year_dict)


def create_TPP_swing_matrix(election_years_extended,binary):
    TPP_swing_matrix_dict = {}



    for data_year in election_years_extended:
        TPP = pd.read_csv(f"{data_year}HouseTppByState.csv", skiprows=1)

        if binary:
            if int(data_year) > 2001:
                TPP = TPP.iloc[:,[0,-2,-1]]
                national_swing = TPP['Swing'].dot(TPP['TotalVotes'])/sum(TPP['TotalVotes'])
                TPP.loc[:,"Relative Swing"] = np.sign(TPP.loc[:,"Swing"] - national_swing)
                Relative_swing = TPP.loc[:,["StateAb","Relative Swing"]]
            else:
                national_swing = TPP.iloc[-1,-1]
                TPP = TPP.iloc[:-1,]
                TPP["Relative Swing"] = np.sign(TPP.iloc[:,-1] - national_swing)
                Relative_swing = TPP.iloc[:,[0,-1]]

        else:
            if int(data_year) > 2001:
                TPP = TPP.iloc[:,[0,-2,-1]]
                national_swing = TPP['Swing'].dot(TPP['TotalVotes'])/sum(TPP['TotalVotes'])
                TPP.loc[:,"Relative Swing"] = TPP.loc[:,"Swing"] - national_swing
                Relative_swing = TPP.loc[:,["StateAb","Relative Swing"]]
            else:
                national_swing = TPP.iloc[-1,-1]
                TPP = TPP.iloc[:-1,]
                TPP["Relative Swing"] = TPP.loc[:,"Swing"] - national_swing
                Relative_swing = TPP.iloc[:,[0,-1]]

        # Switch around ACT and NT to coincide with ABS order
        Relative_swing_copy = Relative_swing.copy()
        Relative_swing_copy.iloc[[-2, -1]] = Relative_swing.iloc[[-1, -2]].values

        if binary:
            TPP_swing_matrix = state_by_state_matrix(Relative_swing_copy, threshold=0)   

        else:
            ser = Relative_swing_copy.set_index('StateAb')
            similarity_matrix = 1 - (np.abs(ser.values[:, None] - ser.values).squeeze(axis=-1) * (1-MIN_SIMILARITY) / (ser.max() - ser.min()).iloc[0])

            TPP_swing_matrix = pd.DataFrame(similarity_matrix, index=ser.index, columns=ser.index)

        
            
        # Convert to DataFrame for readability
        TPP_swing_matrix_dict[data_year] = TPP_swing_matrix
        
    return TPP_swing_matrix_dict


# TPP swing matrices
TPP_swing_matrix_dict = create_TPP_swing_matrix(election_years_extended,binary)


def make_Avg_TPP_Swing_matrix(TPP_swing_matrix_dict, election_years_extended):
    Avg_TPP_matrix_dict = {}
    int_election_years_extended = [int(i) for i in election_years_extended] 

    for i in range(TPP_ROLLING_AVERAGE_NO, len(int_election_years_extended)):  # Start from index 3 to have 3 preceding elections
        avg_df = np.zeros((NO_OF_STATES,NO_OF_STATES))
        year = int_election_years_extended[i]
        prev_years = int_election_years_extended[i-TPP_ROLLING_AVERAGE_NO:i]
        for j in range(len(prev_years)):
            avg_df += TPP_swing_matrix_dict[str(prev_years[j])]

        avg_df /= TPP_ROLLING_AVERAGE_NO

        Avg_TPP_matrix_dict[str(year)] = avg_df

    return Avg_TPP_matrix_dict

TPP_Swing_Avg_matrix_dict = make_Avg_TPP_Swing_matrix(TPP_swing_matrix_dict, election_years_extended)



# Eternal Proximity matrix
Proximity_matrix_dict = {}

Proximity_matrix = pd.read_csv("StateProximity.csv", index_col='StateAb').fillna(0).astype(int)
for year in election_years:
    to_add = Proximity_matrix.copy()
    to_add.loc[:,'Election_Year'] = year
    Proximity_matrix_dict[year] = to_add




def dict_to_long_df(matrix_dict, dict_name):
    df_list = []
    for year, df in matrix_dict.items():
        df["Election_Year"] = year  # Add election year
        df["Dict"] = dict_name  # Add dictionary name (dict1, dict2, etc.)
        df["Row_Index"] = df.index  # Track original row position
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


#dict_to_long_df(similarity_matrix_dict, "Similarity")

# Combine all dictionaries into one DataFrame
Similarity_df = pd.concat([
    dict_to_long_df(similarity_matrix_dict, "Similarity"),
    dict_to_long_df(TPP_Swing_Avg_matrix_dict, "TPPSwing"),
    dict_to_long_df(Proximity_matrix_dict,'Adjacency')
], ignore_index=True)



def wide_to_long(df):
    """
    Convert a wide-form correlation DataFrame into long-form.

    Parameters:
        df (pd.DataFrame): The wide-form DataFrame where columns represent states.
        predictors (list): List of predictor names (e.g., ['Similarity', 'TPPSwing', 'Adjacency']).
        years (list): List of election years corresponding to each predictor.

    Returns:
        pd.DataFrame: Long-form DataFrame with 'row_index', 'column_index', 'Dict', and 'Election_Year'.
    """


    # Melt the dataframe to get state pairs in long format
    df = df.rename(columns={'Row_Index':'State1'})
    df_long = df.melt(id_vars=["Dict", "Election_Year", "State1"], var_name="State2",  value_name="Value")
    df_long = df_long[df_long['State1'] != df_long['State2']]

    df_pivot = df_long.pivot_table(index=["State1", "State2", "Election_Year"],  columns="Dict", values="Value").reset_index()

    return df_pivot


# Assuming 'df' has states as column names and a 'row_index' column
sample_correlation_matrix = pd.read_csv('State_Sample_Correlation_matrix.csv')
sample_correlation_matrix.rename(columns={'Unnamed: 0': 'Row_Index'}, inplace=True)
corr_df = pd.concat([sample_correlation_matrix.assign(Election_Year=year) for year in election_years], ignore_index=True)
corr_df['Dict'] = 'Correlation'

Regression_Format_df_wide = pd.concat([corr_df,Similarity_df], ignore_index=True)

Regression_Format_df = wide_to_long(Regression_Format_df_wide)

# remove duplicate state pairs!
Regression_Format_df[['State1', 'State2']] = Regression_Format_df.apply(lambda row: pd.Series(sorted([row['State1'], row['State2']])), axis=1)
Regression_Format_df = Regression_Format_df.drop_duplicates()
#print(long_df.head())


if binary:
    Regression_Format_df.to_csv("State_similarity_df.csv", index=False)
else:
    Regression_Format_df.to_csv("State_similarity_df_continuous.csv", index=False)

import pdb;pdb.set_trace()

