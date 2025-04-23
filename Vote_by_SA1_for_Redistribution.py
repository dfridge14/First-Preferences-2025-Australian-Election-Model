import pandas as pd
import numpy as np
import os, time
import glob
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

census_years_for_data_years = {'2022':'2021','2019':'2016','2016':'2011','2013':'2011'}


data_year = "2022"
census_year = census_years_for_data_years[data_year]
redistribution_type = 'redistribution' # 'New Candidates' #'redistribution'
ON_add = False

start = time.time()

name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}
new_seats_year_dict = {'2022': ['Bullwinkel'],'2019': ['Hawke'],'2016':['Bean','Fraser'],'2013':['Burt'],'2010':[],'2007':['Wright'],'2004':['Flynn'],'2001':['Bonner']}
abolished_divs_dict = {'2022':set(['Higgins','North Sydney']), '2016': set(['Port Adelaide']),'2019':set(['Stirling']),'2013':set(['Charlton'])}






def load_FPBPPRed_dict(data_year, redistribution_type, ON_add = False):
    df_dict_reloaded = {}
    folder_name = f'Redistribution pairs {str(int(data_year)+2)}' if redistribution_type == 'redistribution' else f'New Candidates {'ON_add ' if ON_add else ''}for {str(int(data_year)+3)}'
    output_folder = f"feather {folder_name}"
    for filepath in glob.glob(f"{output_folder}/*.feather"):
        # Extract keys from filename
        parts = filepath.split("_")
        key = (parts[-2], parts[-1].split(".")[0])  # Convert second part to int
        
    
        # Read the file back
        df_dict_reloaded[key] = pd.read_feather(filepath)
    
    return df_dict_reloaded



def unique_SA1s_dict(Div_SA1_By_PP_dict, census_year):
    # generates dict of list of unique SA1s for each division
    unique_SA1s_dict = {}

    for div in Div_SA1_By_PP_dict.keys():
        unique_SA1s_dict[div] = Div_SA1_By_PP_dict[div][f"SA1_CODE{census_year[-2:]}"].unique()

    return unique_SA1s_dict

def convert_to_wide_format(df, df_type, census_year):
    # converts to wide format indexed by pp_id for either First Preferences or SA1 dfs. If First Prefs, it first adjusts the Party names in case of multiple independents

    if df_type == "First Preferences":
        # adjust to order independents!!!
        cand_set = df.sort_values(by=['pp_id', 'PartyAb'], key=lambda x: x != 0, ascending=[True, False]) #df.loc[df['pp_id']==0,'PartyAb']


        target = 'IND'
        cand_set['Count'] = cand_set.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
        # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
        adjusted_party_names = cand_set.loc[cand_set["pp_id"] == 0,].apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
        )
        num_pp_ids = len(set(cand_set['pp_id'])) # num of final count + original FP count

        # now, add to df!
        df.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * num_pp_ids, ignore_index=True).values
        df.loc[df["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!





    if df_type == "First Preferences":
        pivot_df = df.pivot_table(index=['pp_id'], 
                                columns=['PartyAb'], 
                                values='votes', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()


    if df_type == "SA1s":
        pivot_df = df.pivot(index='pp_id', columns=f'SA1_CODE{census_year[-2:]}', values='votes')
        pivot_df = pivot_df.fillna(0)
        pivot_df = pivot_df.astype(int)
        pivot_df = pivot_df.reset_index()
    
    if df_type == "DOP":
        pivot_df = df.pivot_table(index=['CountNumber'], 
                                columns=['PartyAb'], 
                                values='CalculationValue', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()

    return pivot_df

def convert_dict_to_wide_format(dict, df_type):

    converted_dict = {key: convert_to_wide_format(dict[key], df_type, census_year) for key in dict.keys()}

    return converted_dict



def rectify_div_SA1_votes(div, Div_SA1_By_PP_dict_wide, Div_First_Prefs_By_PP_dict_wide):
    ### updates Div_SA1_By_PP_dict_wide by matching # votes in SA1sByPP to the true numbers of House votes in each PP, using sampling without 
    ### replacement to make up the difference. This will be the same irrespective of whether Pure or Redistributed First Prefs are used

    Div_SA1_By_PP_dict_wide[div]["n"] = Div_SA1_By_PP_dict_wide[div].iloc[:, 1:].sum(axis=1)
    Div_First_Prefs_By_PP_dict_wide[div]["K"] = Div_First_Prefs_By_PP_dict_wide[div].iloc[:, 1:].sum(axis=1)

    all = pd.merge(Div_SA1_By_PP_dict_wide[div],Div_First_Prefs_By_PP_dict_wide[div][["pp_id","K"]], on = 'pp_id',how='left')
    all.loc[:,"vote_difference"] = all["n"] - all["K"]

    Div_First_Prefs_By_PP_dict_wide[div] = Div_First_Prefs_By_PP_dict_wide[div].drop(columns=["K"]) # reverse alteration

    def sample_with_replacement(row):
        ### samples a vote from the set of SA1s for given pp_id proportional to their frequency, vote_difference times

        values = row[:-1].values
        if values.sum()>0:
            probs = values / values.sum()  # Normalize row to probabilities
        else:
            probs = np.full(len(values), 1/len(values)) # if no vote - Dunif!

        labels = row[:-1].index  
        
        # Sample with replacement
        samples = np.random.choice(labels, size=np.abs(int(row["vote_difference"])), p=probs)
        return samples.tolist()

    # Apply sampling to all rows, unless house has 0 votes!
    all.dropna(inplace = True) # when 0 votes
    all["Sampled"] = all.drop(columns=['pp_id','n','K']).apply(lambda row: sample_with_replacement(row), axis=1)



    def decrement_counts(row):
        """
        Decrement the counts in the sampled columns for a row, retrying if the count is zero.
        Args:
        row (pd.Series): A row of the DataFrame.
        Returns:
        pd.Series: Updated row with decremented counts.
        """
        counts = row[:-2].copy()  # Exclude 'N' and 'Sampled' columns
        vote_diff = row["vote_difference"]
        sampled = row["Sampled"].copy()
        
        if vote_diff > 0: # remove from SA1 - possible multiple removal errors

            for i, SA1 in enumerate(sampled):
                while True:
                    # Check if the count for the sampled column is > 0
                    if counts[SA1] > 0:
                        counts[SA1] -= 1
                        break
                    else:
                        # If count is 0, resample a column with nonzero value
                        nonzero_probs = counts / counts.sum()  # Normalize probabilities
                        SA1 = np.random.choice(counts.index, p=nonzero_probs)
                        sampled[i] = SA1

        if vote_diff < 0:
            for SA1 in sampled:
                counts[SA1] += 1

        counts["Sampled"] = sampled # replace sampled column if changed to keep track of removals

        return counts


    # Apply decrementing logic
    updated_counts = all.drop(columns=['pp_id','n','K']).apply(decrement_counts, axis=1)

    updated_counts_32 = updated_counts.iloc[:, :-1].astype('int32')
    updated_counts = pd.concat([updated_counts_32, updated_counts.iloc[:,-1]], axis=1)

    # Update the DataFrame with decremented values
    all.loc[:, all.columns[1:-4].tolist()+all.columns[-1:].tolist()] = updated_counts
    all = all.copy() # to evade PerformanceWarning

    # check if now totals reach K - Yes they do!!!
    all["New_K"] = all.iloc[:, 1:-4].sum(axis=1)
    #print(all)

    Div_SA1_By_PP_dict_wide[div] = all.drop(columns=['n','K','vote_difference','Sampled','New_K'])
    #Sampled_Deakin = all[["pp_id","Sampled"]]

    return Div_SA1_By_PP_dict_wide[div]




def prepare_wide_dicts_for_MMHg(data_year, census_year, name_changes_year_dict, redistribution_type = 'redistribution', ON_add = False):

    # create dictionary of SA1s for each giver_div
    Redistribution_SA1_changes = pd.read_csv(f"Redistribution_SA1_changes{str(int(data_year)+2)}.csv", index_col = None)

    redistribution_SA1s_dict = {div: group.iloc[:,0] for div, group in Redistribution_SA1_changes.groupby(["old_div",'new_div'])}
    SA1s_giver_div_dict = {div: group.iloc[:,0] for div, group in Redistribution_SA1_changes.groupby("old_div")}

    ### NOTE: Redistribution_SA1_changes MISSES SOME OF THE SA1S THAT HAVE FEWER THAN 10 VOTES!!!

    old_divs_set = set(Redistribution_SA1_changes.loc[:,'old_div'])
    new_divs_set = set(Redistribution_SA1_changes.loc[:,'new_div'])

    redistribution_divs_set = old_divs_set | new_divs_set


    # obtain wide dfs for all relevant redistribution pairs
    if redistribution_type in ['redistribution','New Candidates']:
        Redist_Div_First_Prefs_By_PP_dict_wide = load_FPBPPRed_dict(data_year, redistribution_type, ON_add)

    elif redistribution_type == 'omnipresent':
        # transform 3PP df into dictionary structure with tuples as keys
        Redistribution_pairs = pd.read_csv(f"RedistributionPairs{str(int(data_year)+2)}.csv", index_col = None)
        df_3PP = pd.read_csv(f'{data_year}OmnipresentPartiesByPP.csv', index_col=None).drop('pp_nm', axis=1)
        # need to transform so that only old-new pairs get a key
        Redist_Div_First_Prefs_By_PP_dict_wide = {tuple(row): df_3PP[df_3PP['div_nm'] == row[0]].drop('div_nm', axis=1) for row in Redistribution_pairs.itertuples(index=False)}


    #import pdb;pdb.set_trace()

    # unaltered First Preferences
    First_Prefs_by_PP_Complete = pd.read_csv(f'{data_year}FirstPrefsByPPComplete.csv', index_col=None)
    First_Prefs_by_PP_Complete['div_nm'] = First_Prefs_by_PP_Complete['div_nm'].replace(name_changes_year_dict[data_year])


    # load some data for PPs and SA1s (types, centroids)
    PP_coords = First_Prefs_by_PP_Complete[['div_nm','pp_id','Booth_type','Lat','Long']].drop_duplicates()
    SA1_By_PP_Votes = pd.read_csv(f"{data_year}SA1_By_PP_Votes.csv", index_col=None) # SA1_By_PP_Complete /
    SA1_By_PP_Votes['div_nm'] = SA1_By_PP_Votes['div_nm'].replace(name_changes_year_dict[data_year])


    SA1_centroids = pd.read_csv(f"SA1_centroids_{census_year}.csv")
    if census_year != '2021':
        SA1_centroids = SA1_centroids.rename(columns={f"SA1_7DIG{census_year[-2:]}": f"SA1_CODE{census_year[-2:]}"})

    ExistingSA1s = SA1_By_PP_Votes.iloc[:,1].drop_duplicates()
    Ghost_SA1s = ExistingSA1s.loc[~(ExistingSA1s.isin(SA1_centroids[f'SA1_CODE{census_year[-2:]}'].tolist())) & (ExistingSA1s.astype(str).str.startswith(('1', '2', '5', '7'))),].sort_values().tolist()

    # remove Ghost Votes rows from SA1_By_PP_Votes
    SA1_By_PP_Votes = SA1_By_PP_Votes.loc[~SA1_By_PP_Votes[f'SA1_CODE{census_year[-2:]}'].isin(Ghost_SA1s),]


    # check to make sure all redistribution sa1s are in SA1_centroids
    # SA1_centroids = SA1_centroids.loc[SA1_centroids[f'SA1_CODE{census_year[-2:]}'].isin(Redistribution_SA1_changes[f'SA1_CODE{census_year[-2:]}'].tolist())]
    # Redistribution_SA1_changes.loc[Redistribution_SA1_changes[f'SA1_CODE{census_year[-2:]}'].isin(SA1_centroids[f'SA1_CODE{census_year[-2:]}'].tolist()),]  



    # Dictionaries of all division first_prefs and SA1s
    Div_First_Prefs_By_PP_dict = {div: group for div, group in First_Prefs_by_PP_Complete.groupby("div_nm")}
    Div_SA1_By_PP_dict = {div: group for div, group in SA1_By_PP_Votes.groupby("div_nm")}

    # get wide formats for First Prefs and SA1s
    Div_First_Prefs_By_PP_dict_wide = convert_dict_to_wide_format(Div_First_Prefs_By_PP_dict, "First Preferences")
    Div_SA1_By_PP_dict_wide = convert_dict_to_wide_format(Div_SA1_By_PP_dict, "SA1s")


    # ensure match between K and n for each PP in Div_First_Prefs_By_PP_dict Div_SA1_By_PP_dict_wide
    Div_SA1_By_PP_dict_wide = {div: rectify_div_SA1_votes(div, Div_SA1_By_PP_dict_wide, Div_First_Prefs_By_PP_dict_wide) for div in old_divs_set}

    if redistribution_type == 'omnipresent':
        Div_First_Prefs_By_PP_dict_wide = {div: group.drop('div_nm', axis=1) for div, group in df_3PP.groupby('div_nm')}

    #if redistribution_type == 'redistribution':
    #    Redist_Div_First_Prefs_By_PP_dict_wide = 0


    return Redist_Div_First_Prefs_By_PP_dict_wide, Div_First_Prefs_By_PP_dict_wide, Div_SA1_By_PP_dict_wide, PP_coords, SA1_centroids, SA1s_giver_div_dict, redistribution_SA1s_dict, redistribution_divs_set, old_divs_set




Redist_Div_First_Prefs_By_PP_dict_wide, Div_First_Prefs_By_PP_dict_wide, Div_SA1_By_PP_dict_wide, PP_coords, SA1_centroids, SA1s_giver_div_dict, redistribution_SA1s_dict \
        , redistribution_divs_set, old_divs_set = prepare_wide_dicts_for_MMHg(data_year, census_year, name_changes_year_dict, redistribution_type = redistribution_type, ON_add=ON_add)


print("Got all dfs: time = ", time.time() - start)



#import pdb;pdb.set_trace()




def MMHg_mean(N,K,n):
    K_array = np.array(K)
    n_array = np.array(n)

    mean_X = n_array[:,None]*K_array/N

    return mean_X

def multivariate_hypergeometric(N, K, n):
    """
    Simulate a sample from the multivariate hypergeometric distribution.
    
    Parameters:
    N (int): Total number of items.
    K (list of int): Number of items in each category (votes for each candidate).
    n (int): Number of items to sample.
    
    Returns:
    list of int: Sampled items from each category.
    """
    K = np.array(K)
    sampled = np.zeros_like(K)
    remaining_population = N
    remaining_sample = n

    if remaining_sample == 0: # no votes from this SA1
        return sampled


    for i in range(len(K) - 1): # last variable is degenerate
        if remaining_sample > remaining_population:  # Check if remaining sample exceeds remaining population
            return sampled

        sampled[i] = np.random.hypergeometric(K[i], remaining_population - K[i], remaining_sample)
        remaining_population -= K[i]  # equivalent to N1-n1
        remaining_sample -= sampled[i] # equivalent to K1-x1

        if remaining_sample == 0: # run out of votes to assign
            for j in range(i+1,len(K)-1):
                sampled[j] = 0
            break

    sampled[-1] = remaining_sample
    return sampled

def doubly_multivariate_hypergeometric_simulate(N,K,n):
    """
    Simulate a sample from the doubly multivariate hypergeometric distribution.
    
    Parameters:
    N (int): Total number of items (votes).
    K (list of int): Number of items in each category (votes for each candidate).
    n (list of int): Number of items to sample in each sampling round (number of SA1s).
    
    Returns:
    array of int: Sampled items from each category for each sampling round.
    """

    X = np.empty((len(n), len(K)))

    # X_1 vector
    s = len(n)
    N_remaining = N
    K_remaining = K

    for i in range(s - 1): # 0,1,...,s-2 (altogether s-1):
        X_i = multivariate_hypergeometric(N_remaining, K_remaining, n[i])
        X[i] = X_i # add to X array

        N_remaining -= n[i]
        K_remaining = [a - b for a, b in zip(K_remaining, X_i)]


    X[s-1] = K_remaining

    return X



def candidate_prior_simulation(mean, SA1_wide, FP_wide):
    ### generates array of simulated votes per candidate per SA1 using the doubly-multivariate hypergeometric distribution with given K and n vectors

    num_candidates = FP_wide.shape[1] - 1 # includes informal
    num_SA1s = SA1_wide.shape[1]-1

    #print(num_candidates,num_pps,num_SA1s)


    result_array = np.zeros((num_SA1s,num_candidates))
    #print(result_array) # memory issue

    # iterate for each pp_id in division
    for curr_pp_id in FP_wide["pp_id"].unique():

        if curr_pp_id not in SA1_wide['pp_id'].tolist():
            print('Zero votes pp_id', curr_pp_id)
            continue # don't need to worry about these!!!

        K = FP_wide[FP_wide['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
        n = SA1_wide[SA1_wide['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
        #print("pp_id = ", curr_pp_id, "len_K = ", len(K), "len_n = ", len(n))
        N = sum(K)
        #import pdb;pdb.set_trace()
        #print("Cand Values for the given pp_id:", K)
        #print("SA1 Values for the given pp_id:", n)
        if mean:
            curr_array = MMHg_mean(N,K,n) # expected value
        else:
            curr_array = doubly_multivariate_hypergeometric_simulate(N,K,n)


        result_array += curr_array


    #print(time.time()-start, "seconds")

    return np.round(result_array).astype(int)






def haversine(PP_Lat, PP_Long, SA1_Lat, SA1_Long): # written by chatgpt
    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    PP_Lat, PP_Long, SA1_Lat, SA1_Long = map(np.radians, [PP_Lat, PP_Long, SA1_Lat, SA1_Long])

    # Calculate differences in latitude and longitude
    dlat = PP_Lat - SA1_Lat.T  # PP_Lat is column vector, SA1_Lat is row vector -> transpose SA1_Lat
    dlon = PP_Long - SA1_Long.T  # Same for longitude

    # Apply Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(PP_Lat) * np.cos(SA1_Lat.T) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Compute the distances
    distance = R * c  # Distance in kilometers
    return distance

def distance_formula(PP_Lat, PP_Long, SA1_Lat, SA1_Long):
    return haversine(PP_Lat, PP_Long, SA1_Lat, SA1_Long)

def distances_SA1_PP_array(div, Booth_type, PP_coords, SA1_centroids, SA1_wide, census_year):
    ### computes distances array of pp_id * SA1 for the given division. Sorts pp_id numerically, but SA1s already sorted in SA1_by_PP_Complete!
    ### uses haversine formula for computations

    #PP_centroids_curr_div_2022 = Div_First_Prefs_PP_dict[div_nm][["pp_id","Lat","Long","Booth_type"]].drop_duplicates().sort_values(by='pp_id').reset_index(drop=True)

    PP_coords_div = PP_coords.loc[(PP_coords['div_nm']==div) & (PP_coords["Booth_type"] == Booth_type),["pp_id","Lat","Long"]].sort_values(by='pp_id')
    pp_ids_used = set(PP_coords_div["pp_id"])

    # get SA1 coords of giver division
    div_SA1s = pd.DataFrame(SA1_wide.columns[1:], columns=[f'SA1_CODE{census_year[-2:]}']) 
    SA1_coords = pd.merge(div_SA1s, SA1_centroids, on=f'SA1_CODE{census_year[-2:]}', how="left") # adds lat/long to SA1s in div

    # Extract latitude and longitude arrays
    PP_Lat = PP_coords_div.iloc[:, 1].to_numpy()[:, np.newaxis] 
    PP_Long = PP_coords_div.iloc[:, 2].to_numpy()[:, np.newaxis]
    SA1_Lat = SA1_coords.iloc[:, 1].to_numpy()  
    SA1_Long = SA1_coords.iloc[:, 2].to_numpy()  

    distances = distance_formula(PP_Lat, PP_Long, SA1_Lat, SA1_Long)   

    return distances, pp_ids_used


def weights_SA1_PP_array(distances_SA1_PP_array, pp_ids_set, dist_func, SA1_wide):
    ### dist_func is a string of either inverse or ... TBD
    ### renormalises weights according to size from each SA1
    rec = np.reciprocal(distances_SA1_PP_array)


    # truncates Div_SA1_By_PP_dict_wide to relevant rows
    DSBPDW_array = SA1_wide.loc[SA1_wide["pp_id"].isin(pp_ids_set),:].drop(columns=['pp_id']).to_numpy() # removes pp_id col for array conversion
    n_i_per_SA1 = np.sum(DSBPDW_array, axis=0)
    
    # performs sum(l^n_i)/dot products of each column n_i * d_i, 
    denominator = np.sum(rec * DSBPDW_array, axis=0)
    denominator_safe = np.where(denominator == 0.0, 1e-10, denominator)
    
    n_i_d_i_recalibration =  n_i_per_SA1 / denominator_safe
    weights_SA1_PP_array = rec*n_i_d_i_recalibration # sum_ni_SA1 of col in PP_By_SA1_Dict and col %*% row (rec[i,:])
    return weights_SA1_PP_array



def candidate_totals_rebalancing(SA1_vote_array, Booth_type_actual_vote_totals):
    ### inputs an array of SA1*candidates, and a 1d array of actual vote_totals for the division, and aims to shift votes around proportionally
    ### to each SA1's size to match the overall division candidate totals

    curr_cand_totals = np.sum(SA1_vote_array, axis=0)
    SA1_vote_totals = np.sum(SA1_vote_array, axis=1)
    overall_vote_total = sum(SA1_vote_totals)

    if np.round(overall_vote_total,2) != np.sum(Booth_type_actual_vote_totals):
        print(np.round(overall_vote_total,2), np.sum(Booth_type_actual_vote_totals))
        import pdb;pdb.set_trace()

        #raise ValueError("Mine: SA1_vote_array's global vote totals don't match")

    adj_weights = SA1_vote_totals / overall_vote_total # proportional to size of SA1
    difference_vector = Booth_type_actual_vote_totals - curr_cand_totals

    weighted_difference_array = np.tile(difference_vector, (len(adj_weights),1)) * adj_weights[:,None] # of height equal to #SA1s
    SA1_vote_array += weighted_difference_array

    return SA1_vote_array



def candidate_prior_simulation_weighted(div, mean, weight_Others, PP_coords, SA1_centroids, SA1_wide, FP_wide, census_year):
    ### generates array of simulated votes per candidate per SA1 using the doubly-multivariate hypergeometric distribution with given K and n vectors

    num_candidates = FP_wide.shape[1] - 1 # includes informal
    num_SA1s = SA1_wide.shape[1] - 1

    result_array = np.zeros((num_SA1s,num_candidates))

    #Div_PP_Booth_type = Div_First_Prefs_By_PP_dict[div][["pp_id","Booth_type"]].drop_duplicates().sort_values(by='pp_id').reset_index(drop=True)
    Div_PP_Booth_type = PP_coords.loc[PP_coords['div_nm'] == div,["pp_id","Booth_type"]].sort_values(by='pp_id').reset_index(drop=True)

    for Booth_type in ["PB","PPVC"]:# first PBs, then PPVCs, then Other
        Booth_type_result_array = np.zeros((num_SA1s,num_candidates))
    
        # get distances bw division PPs and SA1s
        distances_PB_array, distances_PB_pp_ids = distances_SA1_PP_array(div, Booth_type, PP_coords, SA1_centroids, SA1_wide, census_year)
        # make them into weights
        weights_PB = weights_SA1_PP_array(distances_PB_array, distances_PB_pp_ids, "inverse", SA1_wide)

        booth_type_pp_id_set = set()

        for i, curr_pp_id in enumerate(Div_PP_Booth_type.loc[Div_PP_Booth_type["Booth_type"] == Booth_type,"pp_id"]): #only pp_ids of "PB" or "PPVC"
            
            booth_type_pp_id_set.add(curr_pp_id) # to calculate candidate totals later

            K = FP_wide[FP_wide['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
            n = SA1_wide[SA1_wide['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
            N = sum(K)

            if sum(K)!=sum(n):
                import pdb;pdb.set_trace()

            if mean:
                curr_array = MMHg_mean(N,K,n) # expected value
            else:
                curr_array = doubly_multivariate_hypergeometric_simulate(N,K,n)




            weighted_array = curr_array * weights_PB[i,:][:, np.newaxis]

            if np.isnan(weighted_array).any():
                weighted_array = np.nan_to_num(weighted_array, nan=0)
                #import pdb;pdb.set_trace()
                print('replaced')


            Booth_type_result_array += weighted_array

        candidate_totals_for_booth_type = FP_wide.loc[FP_wide['pp_id'].isin(booth_type_pp_id_set),].sum().values[1:]
        Booth_type_result_array = candidate_totals_rebalancing(Booth_type_result_array, candidate_totals_for_booth_type)

        result_array += Booth_type_result_array



    # lastly, Other: pp_id is always 0!
    if weight_Others:
        # obtain totals and percentages of Other_PPs vs PB+PPVCs
        Other_PP_cand_totals = FP_wide.loc[FP_wide['pp_id']==0,].sum().values[1:]
        Other_PP_cand_pcts = Other_PP_cand_totals / np.sum(Other_PP_cand_totals)
        Div_cand_totals = FP_wide.sum().values[1:]
        PB_PPVC_cand_totals = Div_cand_totals - Other_PP_cand_totals
        PB_PPVC_cand_pcts = PB_PPVC_cand_totals / np.sum(PB_PPVC_cand_totals)
        
        # convert result_array to percentages to give estimate for Other vote (good estimate if percentages were to be held constant)
        row_sums = result_array.sum(axis=1,keepdims=True)
        result_array_pcts = np.where(row_sums == 0, 1/result_array.shape[1], result_array / row_sums) # replace with even prior over all candidates

        expected_Other_array = result_array_pcts * SA1_wide.iloc[0,1:].to_numpy()[:,np.newaxis] # multiply by the n_i from 'Other' for each SA1

        #print(expected_Other_array)
        #print("Comparison")
        #print(Other_PP_cand_totals)
        #print(np.round(np.sum(expected_Other_array, axis=0)))

        Other_adjusted_array = candidate_totals_rebalancing(expected_Other_array, Other_PP_cand_totals)    
        #print("Here we go - Other_adjusted_array")
        #print(Other_adjusted_array)   
        print((Other_adjusted_array<0).any())
        neg_indices = np.where(Other_adjusted_array < 0)
        #print(neg_indices)
        #for i in neg_indices:
        #    print(np.round(Other_adjusted_array[neg_indices],2)) 

        result_array += Other_adjusted_array
        #print("overall negative votes")
        print((result_array<0).any())
        neg_indices = np.where(result_array < 0)
        #print(neg_indices)
        #for i in neg_indices:
        #    print(np.round(Other_adjusted_array[neg_indices],2)) 

        return np.round(result_array,6)
    
    else:
        K = FP_wide[div][FP_wide[div]['pp_id'] == 0].iloc[0, 1:].tolist()
        n = SA1_wide[SA1_wide['pp_id'] == 0].iloc[0, 1:].tolist()
        N = sum(K)

        if mean:
            curr_array = MMHg_mean(N,K,n) # expected value
        else:
            curr_array = doubly_multivariate_hypergeometric_simulate(N,K,n)
        result_array += curr_array

    return np.round(result_array,6)


def SA1_candidate_prior_df_output(div, mean, to_weight, PP_coords, SA1_centroids, SA1_wide, FP_wide, census_year):
    ### gets array of SA1 vote using weighted algorithm, working with the mean if mean == 1 else using MMHg


    if to_weight:
        array = candidate_prior_simulation_weighted(div, mean, weight_Others=1, PP_coords=PP_coords, SA1_centroids=SA1_centroids, SA1_wide=SA1_wide,FP_wide=FP_wide, census_year=census_year)
    else:
        array = candidate_prior_simulation(mean, SA1_wide, FP_wide)

    SA1s = SA1_wide.columns[1:].tolist()
    candidates = FP_wide.columns[1:].tolist()
    #print("printing SA1s and candidates numbers excluding informal")
    #print(len(SA1s))
    #print(len(candidates) - 1)

    SA1_candidate_prior_df = pd.DataFrame(array, index=SA1s, columns=candidates)

    return SA1_candidate_prior_df

#print("Getting df of output!!!")
#div = 'Melbourne'
#Vote_by_SA1_df = SA1_candidate_prior_df_output(div,1,0,PP_coords, SA1_centroids, Div_SA1_By_PP_dict_wide[div],Div_First_Prefs_By_PP_dict_wide[div], census_year)

# print overall vote counts to check
#print(round(Vote_by_SA1_df.sum()).astype(int))
print(time.time()-start)
#import pdb;pdb.set_trace()

#print(Deakin_Aston_df.loc[Deakin_Aston_df.index.isin(Deakin_Aston_SA1s),])


def perform_redistribution_effects(Redist_Div_First_Prefs_By_PP_dict_wide, Div_First_Prefs_By_PP_dict_wide, Div_SA1_By_PP_dict_wide, old_divs_set, PP_coords, SA1_centroids, SA1s_giver_div_dict, redistribution_SA1s_dict, census_year, new_div_list, abolished_div_list, redistribution_type = 'redistribution', ON_add = False):

    # 1. Get total electorate votes from last election
    #if redistribution_type == 'redistribution':
    if redistribution_type == 'New Candidates':
        Electorate_total_FP_dict = {remaining_div[0] : Redist_Div_First_Prefs_By_PP_dict_wide[remaining_div].set_index('pp_id').sum() for remaining_div in [key for key in Redist_Div_First_Prefs_By_PP_dict_wide if key[0] == key[1]]}
    else:
        Electorate_total_FP_dict = {div: Div_First_Prefs_By_PP_dict_wide[div].set_index('pp_id').sum() for div in Div_First_Prefs_By_PP_dict_wide.keys()}
    #elif redistribution_type == 'omnipresent':
    #    Electorate_total_FP_dict = {div: Redist_Div_First_Prefs_By_PP_dict_wide[div].set_index('pp_id').sum() for div in Redist_Div_First_Prefs_By_PP_dict_wide.keys()}

    for new_div in new_div_list:
        if redistribution_type == 'redistribution':
            standard_parties = ['LP','ALP','GRN','ON','UAPP', 'INFORMAL']
        elif redistribution_type == 'omnipresent':
            standard_parties = ['COAL','ALP','GRN', 'INFORMAL']
        elif redistribution_type == 'New Candidates':
            # get New candidates list from Redist_Div_First_Prefs_By_PP_dict_wide
            standard_parties = [df.set_index('pp_id').columns.tolist() for (div1, div2), df in Redist_Div_First_Prefs_By_PP_dict_wide.items() if div2 == new_div][0]

        Electorate_total_FP_dict[new_div] = pd.Series(0, index=standard_parties, dtype='int32')

    Electorate_total_FP_dict = {k: v for k, v in Electorate_total_FP_dict.items() if k not in abolished_divs_dict[data_year]} # remove abolished divs!


    print(len(Electorate_total_FP_dict.keys()))

    #import pdb;pdb.set_trace()


    # 2. 
    mean = 1
    to_weight = 1

    for div in (old_divs_set - abolished_divs_dict[data_year]):
        print(div)
        if redistribution_type == 'redistribution':
            Vote_by_SA1_df = SA1_candidate_prior_df_output(div, mean, to_weight, PP_coords, SA1_centroids, Div_SA1_By_PP_dict_wide[div], Div_First_Prefs_By_PP_dict_wide[div], census_year)
        elif redistribution_type == 'omnipresent':
            Vote_by_SA1_df = SA1_candidate_prior_df_output(div, mean, to_weight, PP_coords, SA1_centroids, Div_SA1_By_PP_dict_wide[div], Div_First_Prefs_By_PP_dict_wide[div], census_year)
        elif redistribution_type == 'New Candidates':
            # get the moving SA1s (from each old_div)
            curr_div_new_candidate_votes = Redist_Div_First_Prefs_By_PP_dict_wide[(div,div)]
            Vote_by_SA1_df = SA1_candidate_prior_df_output(div, mean, to_weight, PP_coords, SA1_centroids, Div_SA1_By_PP_dict_wide[div], curr_div_new_candidate_votes, census_year)

        # subtract votes being given away from 2022 election results
        curr_div_SA1_list = SA1s_giver_div_dict[div].tolist()

        # check to make sure abolished divisions go to close to 0
        #low_vote_difference = Vote_by_SA1_df.sum() - Vote_by_SA1_df.loc[Vote_by_SA1_df.index.isin(curr_div_SA1_list),].sum()
        #print("low vote difference", round(low_vote_difference,2).tolist())

        Electorate_total_FP_dict[div] -= Vote_by_SA1_df.loc[Vote_by_SA1_df.index.isin(curr_div_SA1_list),].sum() # assumed SA1s are in the index!
    #import pdb;pdb.set_trace()

    print("Got all dfs: time = ", time.time() - start)

    # 3. give to new redistribution
    for div_pair in Redist_Div_First_Prefs_By_PP_dict_wide.keys():

        print(div_pair)
        giver_div, receiver_div = div_pair

        if redistribution_type == 'New Candidates':
            if giver_div == receiver_div:
                continue # want to process only redistirbutions!

        Redist_Vote_by_SA1_df = SA1_candidate_prior_df_output(giver_div, mean, to_weight, PP_coords, SA1_centroids, Div_SA1_By_PP_dict_wide[giver_div], Redist_Div_First_Prefs_By_PP_dict_wide[div_pair], census_year)

        curr_SA1_list = redistribution_SA1s_dict[div_pair].tolist()
        SA1_totals_to_transfer = Redist_Vote_by_SA1_df.loc[Redist_Vote_by_SA1_df.index.isin(curr_SA1_list),].sum() # assumed SA1s are in the index!

        # rename COAL parties if we are in NSW/VIC
        COAL_parties = [p for p in Electorate_total_FP_dict[receiver_div].index if p in ['LP','NP']]

        if redistribution_type != 'New Candidates':

            if len(COAL_parties) == 1:
                SA1_totals_to_transfer = SA1_totals_to_transfer.rename({'COAL':COAL_parties[0]})
            elif len(COAL_parties) == 2:
                SA1_totals_to_transfer = SA1_totals_to_transfer.rename({'COALLP':'LP','COALNP':'NP'})



        Electorate_total_FP_dict[receiver_div] += SA1_totals_to_transfer

    print("Got all dfs: time = ", time.time() - start)



    import pdb;pdb.set_trace()

    if redistribution_type == 'omnipresent':
        Electorate_3PPs_redistributed = pd.DataFrame.from_dict(Electorate_total_FP_dict, orient='index').drop('INFORMAL', axis = 1)
        Electorate_3PPs_redistributed = Electorate_3PPs_redistributed.loc[~Electorate_3PPs_redistributed.index.isin(abolished_div_list),].astype(int)
        Electorate_3PPs_redistributed.reset_index(names='div_nm',inplace=True)
        Electorate_3PPs_redistributed.to_csv(f'{data_year}Electorate_3PPs_redistributed.csv', index=False)

        import pdb;pdb.set_trace()

    elif redistribution_type == 'New Candidates':
        # make long df with following cols:
        # data_year
        # div_nm
        # PartyAb ( drop INFORMAL)
        # Votes

        for div in Electorate_total_FP_dict:
            Electorate_total_FP_dict[div] = Electorate_total_FP_dict[div].loc[~(Electorate_total_FP_dict[div].index=='INFORMAL')].div(Electorate_total_FP_dict[div].iloc[:-1].sum())

        long_df = pd.concat(Electorate_total_FP_dict).reset_index().rename(columns={'level_0':'div_nm','level_1':'PartyAb',0:'FP_Votes'})
        long_df.loc[:,'Preceding_election'] = data_year
        long_df = long_df[['Preceding_election','div_nm','PartyAb','FP_Votes']].sort_values(by='div_nm')

        import pdb;pdb.set_trace()

        long_df.to_csv(f"Fundamentals_Votes_{"ON_add_" if ON_add else ''}For_{str(int(data_year)+3)}.csv", index = False)





    elif redistribution_type == 'redistribution':








        # 4. TCP Preference Flows!!!
        TCP_dict = {}

        TCP_Preference_Flows = pd.read_csv(f"{data_year}TCPPreferenceFlows.csv", skiprows = 1, index_col = None).rename(columns = {'DivisionNm':'div_nm','FromCandidatePartyAb':'PartyAb', \
                                                'FromCandidateBallotPosition':'Ballot_Position','ToCandidatePartyAb':'TCP_Ab','ToCandidateBallotPosition':'TCP_Ballot_Position'})
        TCP_Preference_Flows = TCP_Preference_Flows[['div_nm','PartyAb','Ballot_Position','TCP_Ab','TCP_Ballot_Position','TransferPercentage']]
        TCP_Preference_Flows = TCP_Preference_Flows.loc[TCP_Preference_Flows['Ballot_Position']>0,]


        for div in redistribution_divs_set:

            print(div)
            if div in abolished_div_list:
                continue

            First_Preferences = Electorate_total_FP_dict[div]

            if div in new_div_list:
                Top_2 = First_Preferences.loc[First_Preferences.index.isin(['LP','ALP'])].index
                Remaining_votes = First_Preferences.loc[~(First_Preferences.index.isin(Top_2))].rename("Non-TCP_votes") 

                TCP_votes = First_Preferences[Top_2]

                transfers = TCP_Preference_Flows.loc[TCP_Preference_Flows['div_nm']=='Hasluck',][['PartyAb','TransferPercentage']] # happens to be correct order of ALP/LP
                transfers = transfers.loc[transfers['PartyAb'].isin(Remaining_votes.index)].merge(Remaining_votes.reset_index().rename(columns={'index': 'PartyAb'}) , on='PartyAb',how='left')
                transfers.loc[:,'TransferVotes'] = transfers.loc[:,'TransferPercentage'] * transfers.loc[:,'Non-TCP_votes'] / 100

                parties = ['GRN','UAPP','ON']
                for j in range(len(parties)):
                    TCP_votes += transfers.loc[transfers['PartyAb'] == parties[j],'TransferVotes'].values 

                TCP_dict[div] = TCP_votes/TCP_votes.sum()

            else:

                TCP_Preference_Flows_div = TCP_Preference_Flows.loc[TCP_Preference_Flows['div_nm']==div,]


                TCP_Ballot_Positions = TCP_Preference_Flows_div['TCP_Ballot_Position'].unique().tolist()

                Top_2 = First_Preferences.index[np.sort(TCP_Preference_Flows_div['TCP_Ballot_Position'].unique())-1] 

                #assert Top_2.tolist() == Max_2_votes.tolist() # can just use Ballot position to align parties - Nonsense! Can be that 3rd/4th party jumps ahead!!!



                TCP_votes = First_Preferences[Top_2]
                Remaining_votes = First_Preferences.loc[~(First_Preferences.index.isin(Top_2))].rename("Non-TCP_votes") 

                # Indices of INDs who didn't make top 2 --> use these to rename INDs in TCP_Preference_Flows_div!
                IND_indices = np.where(First_Preferences.index.str.startswith('IND') & ~First_Preferences.index.isin(Top_2))[0] 

                for ballot_no in (IND_indices+1).tolist():

                    TCP_Preference_Flows_div.loc[TCP_Preference_Flows_div['Ballot_Position'] == ballot_no,'PartyAb'] = First_Preferences.index[ballot_no-1] 
                    

                # merge remaining parties: Note that Informal votes vanish naturally!
                TCP_Transfer_Percents = TCP_Preference_Flows_div[['PartyAb','TransferPercentage','TCP_Ballot_Position']].merge(Remaining_votes, on = 'PartyAb', how='left')

                # for party in partyab, multiply by corresponding transfer percentage, add to top 2 sum
                TCP_Transfer_Percents.loc[:,'TransferVotes'] = TCP_Transfer_Percents.loc[:,'TransferPercentage'] * TCP_Transfer_Percents.loc[:,'Non-TCP_votes'] / 100
                TCP_Transfer_Votes = TCP_Transfer_Percents[['TCP_Ballot_Position','TransferVotes']].groupby('TCP_Ballot_Position')['TransferVotes'].agg('sum')

                TCP_votes += TCP_Transfer_Votes.values
                TCP_votes.index = TCP_votes.index.where(~TCP_votes.index.str.startswith('IND'), 'IND')



                TCP_dict[div] = TCP_votes/TCP_votes.sum()



        import pdb;pdb.set_trace()

        TCP_Redistributed = pd.DataFrame.from_dict(TCP_dict, orient='index').fillna(0)[['ALP','LP','NP','CLP','GRN','IND']]
        TCP_Redistributed = TCP_Redistributed.sort_index().rename(columns={'LP':'LIB','NP':'NAT'})

        # format as 2-Column table!
        
        def extract_contest_info(row):
            # Get the name of the electorate from the index

            # Identify non-zero parties
            non_zero_parties = [party for party in row.index if row[party] != 0]
            non_zero_parties.sort()

            # Skip rows that don't have exactly 2 non-zero entries
            if len(non_zero_parties) != 2:
                return None

            p1, p2 = non_zero_parties

            val1 = round(row[p1]*100, 1)
            val2 = round(row[p2]*100, 1)

            # Determine if it's an ALP vs COAL (LP or NP) contest
            is_alp_vs_coal = (
                ('ALP' in non_zero_parties) and
                ('LIB' in non_zero_parties)
            )

            # Add comment only if not ALP vs COAL
            comment = '' if is_alp_vs_coal else f'{p1} vs {p2}'

            return pd.Series({
                "ALP": val1,
                "COAL": val2,
                "Contest Comments": comment
            })

        # Apply the reorder_columns function to each row and update the DataFrame
        TCP_table = TCP_Redistributed.apply(extract_contest_info, axis=1)

        #import pdb;pdb.set_trace()

        #TCP_Redistributed.to_csv('PostRedistributionTPPMargins2024_InverseWeighted.csv')


        #import pdb;pdb.set_trace()

    return TCP_table


new_div_list = new_seats_year_dict[data_year]
abolished_div_list = abolished_divs_dict[data_year]
TCP_table = perform_redistribution_effects(Redist_Div_First_Prefs_By_PP_dict_wide, Div_First_Prefs_By_PP_dict_wide, Div_SA1_By_PP_dict_wide, old_divs_set, PP_coords, SA1_centroids, SA1s_giver_div_dict, redistribution_SA1s_dict, census_year, new_div_list, abolished_div_list, redistribution_type = redistribution_type, ON_add=ON_add)




make_table = 1

def style_contest_table_dynamic(df):
    # Define base column colors

    colour_map = {
        'ALP': '#ffa6a6',
        'COAL': '#c3d7ea',
        'LIB': '#c3d7ea',
        'GRN': '#84edab', 
        'IND': '#918e8e', 
        'NAT':  '#afd095', 
        'CLP': '#ffb66c'
    }

    # Define row-wise color for "Contest Comments" based on party pairs
    def highlight_by_parties(row):
        styles = [''] * len(row)

        # Default background for ALP and COAL columns
        styles[1] = f'background-color: {colour_map.get("ALP", "#fdd")}'
        styles[2] = f'background-color: {colour_map.get("COAL", "#ddf")}'

        if row["Contest Comments"]:
            try:
                parties = row["Contest Comments"].split(" vs ")
                parties.sort()  # Alphabetical

                if len(parties) == 2:
                    styles[1] = f'background-color: {colour_map.get(parties[0], "#eee")}'
                    styles[2] = f'background-color: {colour_map.get(parties[1], "#eee")}'
            except Exception:
                pass
        return styles

    styled = (
        df.style
        .apply(highlight_by_parties, axis=1)
        .format({'ALP': '{:.1f}', 'COAL': '{:.1f}'})
        .hide(axis="index")  # This hides the index column (removes row numbering)
        .set_table_attributes('class="sortable" style="border-collapse: collapse; table-layout: auto;"')
        .set_properties(**{
        'border': '1px solid #ccc',
        'padding': '20px',
        'font-family': 'Arial, sans-serif',  # Set a nicer font
        'font-size': '14px'  # Increase the font size slightly
        })
        .set_table_styles([  # Notice the list here
        {
            'selector': 'th',
            'props': [
                ('font-family', 'Arial, sans-serif'),
                ('font-size', '16px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        }
    ])
    )

    container_style = """
    <html>
    <head>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                overflow: visible;
                font-family: Arial, sans-serif;
            }
            .table-container {
                width: 100%;
                display: flex;
                justify-content: center;
                padding: 20px;
            }
        </style>
        <script>
            window.addEventListener('load', function () {
                window.parent.postMessage({ frameHeight: document.body.scrollHeight }, '*');
            });
            window.addEventListener('resize', function () {
                window.parent.postMessage({ frameHeight: document.body.scrollHeight }, '*');
            });
        </script>
    </head>
    <body>
        <div class="table-container">
    """
    html = styled.to_html()
    sortable_script = """
    <script>
document.querySelectorAll('th').forEach(headerCell => {
    headerCell.addEventListener('click', () => {
        const table = headerCell.closest('table');
        const tbody = table.querySelector('tbody'); // Get tbody (body rows only)
        const rows = Array.from(tbody.rows); // Get all rows in tbody

        const index = Array.prototype.indexOf.call(headerCell.parentElement.children, headerCell);
        const ascending = !headerCell.classList.contains('asc');

        // Clear arrows in all headers
        table.querySelectorAll('th').forEach(th => {
            th.innerHTML = th.innerHTML.replace(/ ↑| ↓/, '');
        });

        // Toggle sort direction
        if (ascending) {
            headerCell.innerHTML += ' ↑';
        } else {
            headerCell.innerHTML += ' ↓';
        }

        // Sort the rows based on the clicked column
        rows.sort((a, b) => {
            const aText = a.children[index].innerText;
            const bText = b.children[index].innerText;
            return ascending
                ? aText.localeCompare(bText, undefined, { numeric: true })
                : bText.localeCompare(aText, undefined, { numeric: true });
        });

        // Append sorted rows back to tbody
        rows.forEach(row => tbody.appendChild(row));

        // Manage the sorting direction classes
        table.querySelectorAll('th').forEach(th => th.classList.remove('asc', 'desc'));
        headerCell.classList.toggle('asc', ascending);
        headerCell.classList.toggle('desc', !ascending);
    });
});
</script>
    """

    return container_style + html + sortable_script


if make_table:
    TCP_table = TCP_table.reset_index().rename(columns={'index': 'Electorate'})
    html = style_contest_table_dynamic(TCP_table)
    with open("Redistribution_TCP.html", "w") as f:
        f.write(html)