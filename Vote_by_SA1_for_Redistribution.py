import pandas as pd
import numpy as np
import os, time
import glob


os.chdir('C:\\Dania\\2024\\Australian Election')

data_year = "2022"

div_nm = "Melbourne"


start = time.time()


# will need to create dictionary of SA1s
Redistribution_SA1_changes2024 = pd.read_csv("Redistribution_SA1_changes2024.csv", index_col = None)
redistribution_SA1s_dict = {div: group for div, group in Redistribution_SA1_changes2024.groupby("old_div")}


# preliminary dictionary imports and massaging
First_Prefs_by_PP_Complete = pd.read_csv(f'{data_year}FirstPrefsByPPComplete.csv', index_col=None)
SA1_By_PP_Complete = pd.read_csv("SA1_By_PP_Complete.csv", index_col=None)


# Dictionaries of all division first_prefs and SA1s
#Div_First_Prefs_PP_dict = {div: group for div, group in First_Prefs_by_PP_Complete.groupby("div_nm")}
Div_SA1_By_PP_dict = {div: group for div, group in SA1_By_PP_Complete.groupby("div_nm")}


def load_FPBPPRed_dict():
    df_dict_reloaded = {}
    output_folder = "feather Redistribution pairs 2024"
    for filepath in glob.glob(f"{output_folder}/*.feather"):
        # Extract keys from filename
        parts = filepath.split("_")
        key = (parts[1], int(parts[2].split(".")[0]))  # Convert second part to int

        # Read the file back
        df_dict_reloaded[key] = pd.read_feather(filepath)
    
    return df_dict_reloaded

# obtain all relevant redistribution pairs
Div_First_Prefs_Red_PP_dict = load_FPBPPRed_dict()


def unique_SA1s_dict(Div_SA1_By_PP_dict):
    # generates dict of list of unique SA1s for each division
    unique_SA1s_dict = {}

    for division in Div_SA1_By_PP_dict:
        unique_SA1s_dict[division] = Div_SA1_By_PP_dict[division]["SA1_CODE16"].unique()

    return unique_SA1s_dict


def convert_to_wide_format(df, df_type):
    # converts to wide format indexed by pp_id for either First Preferences or SA1 dfs
    if df_type == "First Preferences":
        pivot_df = df.pivot_table(index=['pp_id'], 
                                columns=['PartyAb'], 
                                values='votes', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()
    if df_type == "SA1s":
        pivot_df = df.pivot(index='pp_id', columns='SA1_CODE16', values='votes')
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
    converted_dict = {}

    for key in dict:
        converted_dict[key] = convert_to_wide_format(dict[key], df_type)

    return converted_dict

# get wide formats for First Prefs and SA1s
#Div_First_Prefs_PP_dict_wide = convert_dict_to_wide_format(Div_First_Prefs_PP_dict, "First Preferences")
Div_SA1_By_PP_dict_wide = convert_dict_to_wide_format(Div_SA1_By_PP_dict, "SA1s")




def rectify_div_SA1_votes(div_nm):
    ### updates Div_SA1_By_PP_dict_wide by matching # votes in SA1sByPP to the true numbers of House votes in each PP, using sampling without 
    ### replacement to make up the difference

    Div_SA1_By_PP_dict_wide[div_nm]["n"] = Div_SA1_By_PP_dict_wide[div_nm].iloc[:, 1:].sum(axis=1)
    Div_First_Prefs_PP_dict_wide[div_nm]["K"] = Div_First_Prefs_PP_dict_wide[div_nm].iloc[:, 1:].sum(axis=1)

    all = pd.merge(Div_SA1_By_PP_dict_wide[div_nm],Div_First_Prefs_PP_dict_wide[div_nm][["pp_id","K"]], on = 'pp_id',how='left')
    all.loc[:,"vote_difference"] = all["n"] - all["K"]

    Div_First_Prefs_PP_dict_wide[div_nm] = Div_First_Prefs_PP_dict_wide[div_nm].drop(columns=["K"]) # reverse alteration

    def sample_with_replacement(row):
        ### samples a vote from the set of SA1s for given pp_id proportional to their frequency, vote_difference times

        values = row[:-1].values
        probs = values / values.sum()  # Normalize row to probabilities
        labels = row[:-1].index  
        
        # Sample with replacement
        samples = np.random.choice(labels, size=np.abs(int(row["vote_difference"])), p=probs)
        return samples.tolist()

    # Apply sampling to all rows
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

    # Update the DataFrame with decremented values
    all.loc[:, all.columns[1:-4].tolist()+all.columns[-1:].tolist()] = updated_counts
    all = all.copy() # to evade PerformanceWarning

    # check if now totals reach K - Yes they do!!!
    all["New_K"] = all.iloc[:, 1:-4].sum(axis=1)
    #print(all)

    Div_SA1_By_PP_dict_wide[div_nm] = all.drop(columns=['n','K','vote_difference','Sampled','New_K'])
    #Sampled_Deakin = all[["pp_id","Sampled"]]

    return 1

def rectify_redistribution_SA1_votes(Div_SA1_By_PP_dict_wide, redistribution_divs):
    
    rectify_div_SA1_votes = {}

    for div in redistribution_divs:
        rectify_div_SA1_votes[div] = rectify_div_SA1_votes(div_nm)

    return rectify_div_SA1_votes


rectify_div_SA1_votes(div_nm)















start = time.time()

# load some data for PPs and SA1s (types, centroids)
PP_Booth_type = pd.read_csv(f"{data_year}PPBoothtype.csv")

SA1_centroids_2016 = pd.read_csv("SA1_centroids_2016.csv")
SA1_centroids_2016 = SA1_centroids_2016.rename(columns={"SA1_7DIG16": "SA1_CODE16"})

print("Got all dfs: time = ", time.time() - start)



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

def distances_SA1_PP_array(div_nm, Booth_type):
    ### computes distances array of pp_id * SA1 for the given division. Sorts pp_id numerically, but SA1s already sorted in SA1_by_PP_Complete!
    ### uses haversine formula for computations


    PP_centroids_curr_div_2022 = Div_First_Prefs_PP_dict[div_nm][["pp_id","Lat","Long","Booth_type"]].drop_duplicates().sort_values(by='pp_id').reset_index(drop=True)

    PP_coords = PP_centroids_curr_div_2022.loc[PP_centroids_curr_div_2022["Booth_type"] == Booth_type,["pp_id","Lat","Long"]].sort_values(by='pp_id')
    pp_ids_used = set(PP_coords["pp_id"])

    SA1_coords = pd.merge(Div_SA1_By_PP_dict[div_nm][["SA1_CODE16"]].drop_duplicates().reset_index(drop=True).sort_values(by='SA1_CODE16').reset_index(drop=True), SA1_centroids_2016, on="SA1_CODE16", how="left") # adds lat/long to SA1s in div

    # Extract latitude and longitude arrays
    PP_Lat = PP_coords.iloc[:, 1].to_numpy()[:, np.newaxis] 
    PP_Long = PP_coords.iloc[:, 2].to_numpy()[:, np.newaxis]
    SA1_Lat = SA1_coords.iloc[:, 1].to_numpy()  
    SA1_Long = SA1_coords.iloc[:, 2].to_numpy()  

    distances = distance_formula(PP_Lat, PP_Long, SA1_Lat, SA1_Long)   

    return distances, pp_ids_used


def weights_SA1_PP_array(distances_SA1_PP_array, pp_ids_set, div_nm, dist_func):
    ### dist_func is a string of either inverse or ... TBD
    ### renormalises weights according to size from each SA1
    rec = np.reciprocal(distances_SA1_PP_array)

    # truncates Div_SA1_By_PP_dict_wide to relevant rows
    DSBPDW_array = Div_SA1_By_PP_dict_wide[div_nm].loc[Div_SA1_By_PP_dict_wide[div_nm]["pp_id"].isin(pp_ids_set),:].drop(columns=['pp_id']).to_numpy() # removes pp_id col for array conversion
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
        raise ValueError("Mine: SA1_vote_array's global vote totals don't match")

    adj_weights = SA1_vote_totals / overall_vote_total # proportional to size of SA1
    difference_vector = Booth_type_actual_vote_totals - curr_cand_totals

    weighted_difference_array = np.tile(difference_vector, (len(adj_weights),1)) * adj_weights[:,None] # of height equal to #SA1s
    SA1_vote_array += weighted_difference_array

    return SA1_vote_array



def candidate_prior_simulation_weighted(div_nm, mean, weight_Others):
    ### generates array of simulated votes per candidate per SA1 using the doubly-multivariate hypergeometric distribution with given K and n vectors

    num_candidates = Div_First_Prefs_PP_dict_wide[div_nm].shape[1] - 1 # includes informal
    num_SA1s = Div_SA1_By_PP_dict_wide[div_nm].shape[1]-1

    result_array = np.zeros((num_SA1s,num_candidates))

    Div_PP_Booth_type = Div_First_Prefs_PP_dict[div_nm][["pp_id","Booth_type"]].drop_duplicates().sort_values(by='pp_id').reset_index(drop=True)

    for Booth_type in ["PB","PPVC"]:# first PBs, then PPVCs, then Other
        Booth_type_result_array = np.zeros((num_SA1s,num_candidates))
    
        # get distances bw division PPs and SA1s
        distances_PB_array, distances_PB_pp_ids = distances_SA1_PP_array(div_nm, Booth_type)
        # make them into weights
        weights_PB = weights_SA1_PP_array(distances_PB_array, distances_PB_pp_ids, div_nm, "inverse")

        booth_type_pp_id_set = set()

        for i, curr_pp_id in enumerate(Div_PP_Booth_type.loc[Div_PP_Booth_type["Booth_type"] == Booth_type,"pp_id"]): #only pp_ids of "PB" or "PPVC"
            
            booth_type_pp_id_set.add(curr_pp_id) # to calculate candidate totals later

            K = Div_First_Prefs_PP_dict_wide[div_nm][Div_First_Prefs_PP_dict_wide[div_nm]['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
            n = Div_SA1_By_PP_dict_wide[div_nm][Div_SA1_By_PP_dict_wide[div_nm]['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
            N = sum(K)

            if mean:
                curr_array = MMHg_mean(N,K,n) # expected value
            else:
                curr_array = doubly_multivariate_hypergeometric_simulate(N,K,n)

            weighted_array = curr_array * weights_PB[i,:][:, np.newaxis]

            Booth_type_result_array += weighted_array

        candidate_totals_for_booth_type = Div_First_Prefs_PP_dict_wide[div_nm].loc[Div_First_Prefs_PP_dict_wide[div_nm]['pp_id'].isin(booth_type_pp_id_set),].sum().values[1:]
        Booth_type_result_array = candidate_totals_rebalancing(Booth_type_result_array, candidate_totals_for_booth_type)

        result_array += Booth_type_result_array



    # lastly, Other: pp_id is always 0!
    if weight_Others:
        # obtain totals and percentages of Other_PPs vs PB+PPVCs
        Other_PP_cand_totals = Div_First_Prefs_PP_dict_wide[div_nm].loc[Div_First_Prefs_PP_dict_wide[div_nm]['pp_id']==0,].sum().values[1:]
        Other_PP_cand_pcts = Other_PP_cand_totals / np.sum(Other_PP_cand_totals)
        Div_cand_totals = Div_First_Prefs_PP_dict_wide[div_nm].sum().values[1:]
        PB_PPVC_cand_totals = Div_cand_totals - Other_PP_cand_totals
        PB_PPVC_cand_pcts = PB_PPVC_cand_totals / np.sum(PB_PPVC_cand_totals)
        
        # convert result_array to percentages to give estimate for Other vote (good estimate if percentages were to be held constant)
        row_sums = result_array.sum(axis=1,keepdims=True)
        result_array_pcts = np.where(row_sums == 0, 1/result_array.shape[1], result_array / row_sums) # replace with even prior over all candidates

        expected_Other_array = result_array_pcts * Div_SA1_By_PP_dict_wide[div_nm].iloc[0,1:].to_numpy()[:,np.newaxis] # multiply by the n_i from 'Other' for each SA1

        #print(expected_Other_array)
        print("Comparison")
        print(Other_PP_cand_totals)
        print(np.round(np.sum(expected_Other_array, axis=0)))

        Other_adjusted_array = candidate_totals_rebalancing(expected_Other_array, Other_PP_cand_totals)    
        print("Here we go - Other_adjusted_array")
        #print(Other_adjusted_array)   
        print((Other_adjusted_array<0).any())
        neg_indices = np.where(Other_adjusted_array < 0)
        #print(neg_indices)
        for i in neg_indices:
            print(np.round(Other_adjusted_array[neg_indices],2)) 

        result_array += Other_adjusted_array
        print("overall negative votes")
        print((result_array<0).any())
        neg_indices = np.where(result_array < 0)
        #print(neg_indices)
        for i in neg_indices:
            print(np.round(Other_adjusted_array[neg_indices],2)) 

        return np.round(result_array,6)
    
    else:
        K = Div_First_Prefs_PP_dict_wide[div_nm][Div_First_Prefs_PP_dict_wide[div_nm]['pp_id'] == 0].iloc[0, 1:].tolist()
        n = Div_SA1_By_PP_dict_wide[div_nm][Div_SA1_By_PP_dict_wide[div_nm]['pp_id'] == 0].iloc[0, 1:].tolist()
        N = sum(K)

        if mean:
            curr_array = MMHg_mean(N,K,n) # expected value
        else:
            curr_array = doubly_multivariate_hypergeometric_simulate(N,K,n)
        result_array += curr_array

    return np.round(result_array,6)


def SA1_candidate_prior_df_output(div_nm,mean):
    ### gets array of SA1 vote using weighted algorithm, working with the mean if mean == 1 else using MMHg

    array = candidate_prior_simulation_weighted(div_nm, mean,weight_Others=1)

    SA1s = Div_SA1_By_PP_dict_wide[div_nm].columns[1:].tolist()
    candidates = Div_First_Prefs_PP_dict_wide[div_nm].columns[1:].tolist()
    print("printing SA1s and candidates numbers")
    print(len(SA1s))
    print(len(candidates))

    SA1_candidate_prior_df = pd.DataFrame(array, index=SA1s, columns=candidates)

    return SA1_candidate_prior_df

print("Getting df of output!!!")
Vote_by_SA1_df = SA1_candidate_prior_df_output(div_nm,1)

# print overall vote counts to check
print(round(Vote_by_SA1_df.sum()).astype(int))
print(time.time()-start)
import pdb;pdb.set_trace()

#print(Deakin_Aston_df.loc[Deakin_Aston_df.index.isin(Deakin_Aston_SA1s),])


# Goal: do one per redistribution pair - as candidate change occurs beforehand
# :. need dict of redistribution pairs with candidate/vote sets per PB!