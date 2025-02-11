import pandas as pd
import geopandas as gpd
import numpy as np
import os, time
import matplotlib
from matplotlib import pyplot as plt


os.chdir('C:\\Dania\\2024\\Australian Election')

SA1_By_PP_Complete = pd.read_csv("SA1_By_PP_Complete.csv", index_col=None)


VIC_SA1s_Redistribution_full = pd.read_csv("Vic-2024-electoral-divisions-SA1-and-SA2.csv", index_col=None)
#print(VIC_SA1s_Redistribution_full)
VIC_SA1s_Redistribution_full = VIC_SA1s_Redistribution_full.rename(columns={'SA1_Code_2021': 'SA1_CODE16',"New Electoral Division": 'new_div', "Old Electoral Division": 'old_div', "Actual Enrolment": 'curr_enrol',"Projected Enrolment": 'proj_enrol'})
VIC_SA1s_Redistribution = VIC_SA1s_Redistribution_full[["SA1_CODE16","new_div","old_div",'curr_enrol','proj_enrol']]

#print(VIC_SA1s_Redistribution)
# artificially remove As/Bs - will need systematic approach! - using rstrip # manually checked to be well behaved!
Aston_redistribution = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution["new_div"] == "Aston") & (VIC_SA1s_Redistribution["old_div"] != "Aston") & (VIC_SA1s_Redistribution["curr_enrol"] + VIC_SA1s_Redistribution["proj_enrol"]>0),]
Aston_redistribution.loc[:, "SA1_CODE16"] = Aston_redistribution["SA1_CODE16"].str.rstrip('A')

# Deakin-> Aston redistirbution SA1 list:
Deakin_Aston_SA1s = list(Aston_redistribution["SA1_CODE16"].unique())
Deakin_Aston_SA1s = [int(x) for x in Deakin_Aston_SA1s]



#print(Aston_redistribution)








# preliminary dictionary imports and massaging
First_Prefs_by_PP_Complete = pd.read_csv('FirstPrefsByPP2022Complete.csv', index_col=None)

# Dictionaries of all division first_prefs and SA1s

def create_Div_PP_dict(First_Prefs_by_PP_Complete):
    ### creates dictionary of first preferences by division, with division names as keys. Can be done in one line! Return to this!
    Div_First_Prefs_PP_dict = {}
    
    div_names = list(First_Prefs_by_PP_Complete["div_nm"].unique())

    for div in div_names:
        Div_First_Prefs_PP_dict[div] = First_Prefs_by_PP_Complete.loc[First_Prefs_by_PP_Complete["div_nm"] == div,]

    return Div_First_Prefs_PP_dict


Div_First_Prefs_PP_dict = create_Div_PP_dict(First_Prefs_by_PP_Complete)
Div_SA1_By_PP_dict = create_Div_PP_dict(SA1_By_PP_Complete)

#print("Deakin")
#print(Div_SA1_By_PP_dict["Deakin"])



def unique_SA1s_dict(Div_SA1_By_PP_dict):
    # generates dict of list of unique SA1s for each division
    unique_SA1s_dict = {}

    for division in Div_SA1_By_PP_dict:
        unique_SA1s_dict[division] = Div_SA1_By_PP_dict[division]["SA1_CODE16"].unique()

    return unique_SA1s_dict

#print(unique_SA1s_dict(Div_SA1_By_PP_dict))


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
    
    #unique_SA1s = unique_SA1s_dict(Div_SA1_By_PP_dict)

    for key in dict:
        converted_dict[key] = convert_to_wide_format(dict[key], df_type)

        # check that all relevant SA1s are included - All are!!!!!
        #if df_type == "SA1s":
        #    for SA1 in unique_SA1s[key]:
        #        if SA1 not in converted_dict[key].columns.tolist():
        #            print(key,SA1)

    return converted_dict

print("Here!")
#print(convert_to_wide_format(Div_First_Prefs_PP_dict["Kooyong"], "First Preferences"))
#print(convert_to_wide_format(Div_SA1_By_PP_dict["Aston"], "SA1s"))

Div_First_Prefs_PP_dict_wide = convert_dict_to_wide_format(Div_First_Prefs_PP_dict, "First Preferences")
Div_SA1_By_PP_dict_wide = convert_dict_to_wide_format(Div_SA1_By_PP_dict, "SA1s")
#print(Div_First_Prefs_PP_dict_wide["Deakin"])
#print(Div_SA1_By_PP_dict_wide["Deakin"])

#Heathmont_West_trial = (Div_First_Prefs_PP_dict_wide["Deakin"].loc[Div_First_Prefs_PP_dict_wide["Deakin"]["pp_id"] == 11514,], Div_SA1_By_PP_dict_wide["Deakin"].loc[Div_SA1_By_PP_dict_wide["Deakin"]["pp_id"] == 11514,])
#print(Heathmont_West_trial)

#row = Heathmont_West_trial[1][Heathmont_West_trial[1]['pp_id'] == 11514].iloc[0]

# Get a list of column names for which the value is non-zero
#non_zero_columns = row[row != 0].index.tolist()
#nonzero_votes = row[row != 0][1:].tolist()
#print(nonzero_votes)

#n = nonzero_votes
#n = Heathmont_West_trial[1][Heathmont_West_trial[1]['pp_id'] == 11514].iloc[0][1:].tolist()
#K = Heathmont_West_trial[0][Heathmont_West_trial[0]['pp_id'] == 11514].iloc[0][1:].tolist()
#print(K)
#print(n)
#print(sum(n),sum(K))




def rectify_Deakin_SA1_votes():

    Div_SA1_By_PP_dict_wide["Deakin"]["n"] = Div_SA1_By_PP_dict_wide["Deakin"].iloc[:, 1:].sum(axis=1)
    Div_First_Prefs_PP_dict_wide["Deakin"]["K"] = Div_First_Prefs_PP_dict_wide["Deakin"].iloc[:, 1:].sum(axis=1)

    all = pd.merge(Div_SA1_By_PP_dict_wide["Deakin"],Div_First_Prefs_PP_dict_wide["Deakin"][["pp_id","K"]], on = 'pp_id',how='left')
    all.loc[:,"vote_difference"] = all["n"] - all["K"]

    Div_First_Prefs_PP_dict_wide["Deakin"] = Div_First_Prefs_PP_dict_wide["Deakin"].drop(columns=["K"]) # reverse alteration

    def sample_with_replacement(row):
        
        values = row[:-1].values
        probs = values / values.sum()  # Normalize row to probabilities
        labels = row[:-1].index  # Get the column names (labels)
        
        # Sample with replacement
        samples = np.random.choice(labels, size=np.abs(int(row["vote_difference"])), p=probs)
        return samples.tolist()

    # Apply sampling to all rows
    num_samples = 5  # Number of samples to draw from each row
    all["Sampled"] = all.drop(columns=['pp_id','n','K']).apply(lambda row: sample_with_replacement(row), axis=1)

    #print(all)


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

    # check if now totals reach K - Yes they do!!!
    all["New_K"] = all.iloc[:, 1:-4].sum(axis=1)
    #print(all)

    Div_SA1_By_PP_dict_wide["Deakin"] = all.drop(columns=['n','K','vote_difference','Sampled','New_K'])
    Sampled_Deakin = all[["pp_id","Sampled"]]
    #print(Div_SA1_By_PP_dict_wide["Deakin"])

    return 1

rectify_Deakin_SA1_votes()



def MMHg_mean(N,K,n):
    K_array = np.array(K)
    n_array = np.array(n)

    mean_X = n_array[:,None]*K_array/N

    return mean_X

def produce_MMHg_MVN_approx(N,K,n):

    import numpy as np

    mean_X = np.empty((len(n)-1, len(K)-1))

    for i in range(len(n)-1):
        for l in range(len(K)-1):
            mean_X[i][l] = n[i]*K[l]/N


    # Flatten the array
    mean_X_flat = mean_X.flatten()
    print("Flattened array:", mean_X_flat, sum(mean_X_flat))



    def generate_covariance_matrix(N, K, n):
        """
        Generates a covariance matrix with specified structures.
        
        Parameters:
        - N
        - K
        - n
        
        Returns:
        - cov_matrix (ndarray): Covariance matrix of size (n_rows * n_cols) x (n_rows * n_cols).
        """

        n_rows = len(n) - 1
        n_cols = len(K) - 1
        size = n_rows * n_cols
        cov_matrix = np.zeros((size, size))
        
        for i in range(n_rows):
            for l in range(n_cols):
                for j in range(n_rows):
                    for m in range(n_cols):
                        idx_1 = i * n_cols + l
                        idx_2 = j * n_cols + m
                        
                        if i == j and l == m:
                            cov_matrix[idx_1, idx_2] = n[i]*(N - n[i])*K[l]*(N - K[l])/(N^2*(N-1)) # var
                        elif i == j and l != m:
                            cov_matrix[idx_1, idx_2] = -n[i]*(N - n[i])*K[l]*K[m]/(N^2*(N-1))
                        elif i != j and l == m:
                            cov_matrix[idx_1, idx_2] = -n[i]*n[j]*K[l]*(N - K[l])/(N^2*(N-1))
                        elif i != j and l != m:
                            cov_matrix[idx_1, idx_2] = n[i]*n[j]*K[l]*K[m]/(N^2*(N-1)) # all different
                            
        return cov_matrix



    # Generate covariance matrix
    cov_matrix = generate_covariance_matrix(N,K,n)
    print("Covariance matrix:\n", cov_matrix)
    print(cov_matrix.shape)
    print(len(mean_X_flat))
    sample = np.random.multivariate_normal(mean_X_flat, cov_matrix, 1)

    # Print the first 1 samples
    print(sample)

    return 1


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


start = time.time()

#trial_sim = doubly_multivariate_hypergeometric_simulate(N,K,n)
#for row in trial_sim:
#    print(row)
#print(trial_sim.shape)


def candidate_prior_simulation(div_nm):
    ### generates array of simulated votes per candidate per SA1 using the doubly-multivariate hypergeometric distribution with given K and n vectors

    num_candidates = Div_First_Prefs_PP_dict_wide[div_nm].shape[1] - 1 # includes informal
    num_SA1s = Div_SA1_By_PP_dict_wide[div_nm].shape[1]-1

    #print(num_candidates,num_pps,num_SA1s)


    result_array = np.zeros((num_SA1s,num_candidates))
    #print(result_array) # memory issue

    # iterate for each pp_id in division
    for curr_pp_id in Div_First_Prefs_PP_dict_wide[div_nm]["pp_id"].unique():
        K = Div_First_Prefs_PP_dict_wide[div_nm][Div_First_Prefs_PP_dict_wide[div_nm]['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
        n = Div_SA1_By_PP_dict_wide[div_nm][Div_SA1_By_PP_dict_wide[div_nm]['pp_id'] == curr_pp_id].iloc[0, 1:].tolist()
        #print("pp_id = ", curr_pp_id, "len_K = ", len(K), "len_n = ", len(n))
        N = sum(K)
        #import pdb;pdb.set_trace()
        #print("Cand Values for the given pp_id:", K)
        #print("SA1 Values for the given pp_id:", n)
        curr_array = doubly_multivariate_hypergeometric_simulate(N,K,n)
        result_array += curr_array


    #print(time.time()-start, "seconds")

    return np.round(result_array).astype(int)











PP_Booth_type = pd.read_csv("PPBoothtype2022.csv")

SA1_centroids_2016 = pd.read_csv("SA1_centroids_2016.csv")
SA1_centroids_2016 = SA1_centroids_2016.rename(columns={"SA1_7DIG16": "SA1_CODE16"})
PP_centroids_2022 = First_Prefs_by_PP_Complete[["pp_id","Lat","Long", "Booth_type"]]




def create_centroids_by_div_dict(location_type):
    ### creates dictionary of relevant centroids for each division ############################################## Largely redundant - have Div_First_Prefs_PP_dict/Div_SA1_PP_dict
    centroids_by_div_dict = {}

    if location_type == "PP":
        centroids_by_div_dict = {div: group.drop(columns=['div_nm']).drop_duplicates().reset_index(drop=True) for div, group in First_Prefs_by_PP_Complete[["div_nm","pp_id","Lat", "Long","Booth_type"]].groupby("div_nm")}
        print(centroids_by_div_dict["Deakin"])
            #centroids_by_div_dict[div] = First_Prefs_by_PP_Complete.loc[First_Prefs_by_PP_Complete["div_nm"] == div,["pp_id","Lat", "Long"]]
    if location_type == "SA1":
        #SA1_df = pd.merge([SA1_By_PP_Complete,PP_Booth_type], on='pp_id', how='left')
        SA1_df = pd.merge(SA1_By_PP_Complete[["div_nm","SA1_CODE16"]], SA1_centroids_2016, on="SA1_CODE16", how="left")
        centroids_by_div_dict = {div: group.drop(columns=['div_nm']).drop_duplicates().reset_index(drop=True) for div, group in SA1_df.groupby("div_nm")}
        print(centroids_by_div_dict["Deakin"])

    return centroids_by_div_dict

#PP_centroids_by_div_2022 = create_centroids_by_div_dict("PP")
#SA1_centroids_by_div_2016 = create_centroids_by_div_dict("SA1")

#print("comparison")
#print(PP_centroids_by_div_2022["Deakin"].sort_values(by='pp_id').reset_index(drop=True))
#print(Div_First_Prefs_PP_dict["Deakin"][["pp_id","Lat","Long","Booth_type"]].drop_duplicates().sort_values(by='pp_id').reset_index(drop=True))
#print("next comparison")
#print(SA1_centroids_by_div_2016["Deakin"])
#print(Div_SA1_By_PP_dict["Deakin"][["SA1_CODE16"]].drop_duplicates().reset_index(drop=True).sort_values(by='SA1_CODE16').reset_index(drop=True))
#print(pd.merge(Div_SA1_By_PP_dict["Deakin"][["SA1_CODE16"]].drop_duplicates().reset_index(drop=True).sort_values(by='SA1_CODE16').reset_index(drop=True), SA1_centroids_2016, on="SA1_CODE16", how="left"))















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

    # return DEGREE_TO_KM*np.sqrt((PP_Lat - SA1_Lat)**2 + (PP_Long - SA1_Long)**2)
    return haversine(PP_Lat, PP_Long, SA1_Lat, SA1_Long)

def distances_SA1_PP_array(div_nm, Booth_type):
    ### computes distances array of pp_id * SA1 for the given division. Sorts pp_id numerically, but SA1s already sorted in SA1_by_PP_Complete!
    ### uses haversine formula for computations

    #PP_coords = PP_centroids_2022.loc[PP_centroids_2022["pp_id"] == pp_id, ["Lat", "Long"]].values.flatten().tolist()
    #SA1_coords = SA1_centroids_2016.loc[SA1_centroids_2016["SA1_CODE16"] == SA1, ["Lat", "Long"]]
    PP_centroids_curr_div_2022 = Div_First_Prefs_PP_dict[div_nm][["pp_id","Lat","Long","Booth_type"]].drop_duplicates().sort_values(by='pp_id').reset_index(drop=True)

    PP_coords = PP_centroids_curr_div_2022.loc[PP_centroids_curr_div_2022["Booth_type"] == Booth_type,["pp_id","Lat","Long"]].sort_values(by='pp_id')
    #print(PP_coords)
    pp_ids_used = set(PP_coords["pp_id"])

    SA1_coords = pd.merge(Div_SA1_By_PP_dict[div_nm][["SA1_CODE16"]].drop_duplicates().reset_index(drop=True).sort_values(by='SA1_CODE16').reset_index(drop=True), SA1_centroids_2016, on="SA1_CODE16", how="left") # adds lat/long to SA1s in div


    start_time = time.time()

    # Extract latitude and longitude arrays
    PP_Lat = PP_coords.iloc[:, 1].to_numpy()[:, np.newaxis]  # Suburb latitudes as a column vector
    PP_Long = PP_coords.iloc[:, 2].to_numpy()[:, np.newaxis]  # Suburb longitudes as a column vector
    SA1_Lat = SA1_coords.iloc[:, 1].to_numpy()  # Station latitudes as a row vector
    SA1_Long = SA1_coords.iloc[:, 2].to_numpy()  # Station longitudes as a row vector

    # Compute the distances using broadcasting

    distances = distance_formula(PP_Lat, PP_Long, SA1_Lat, SA1_Long)

    end_time = time.time()
    #print("Vectorized Computation Time:", end_time - start_time)
    #print("Distances Shape:", distances.shape)  # Should be (50, 400)

    return distances, pp_ids_used

#print("Starting distances function")
#Deakin_distances_PPVC = np.round(distances_SA1_PP_array("Deakin", "PPVC"),1)
#print(Deakin_distances_PPVC[0,:])
#print(Deakin_distances_PPVC[:,0])
#print(Deakin_distances_PPVC)
#Deakin_distances_PB = np.round(distances_SA1_PP_array("Deakin", "PB")[0],1)
#print(Deakin_distances_PB[0,:])
#print(Deakin_distances_PB[:,0])
#print(Deakin_distances_PB)
Deakin_distances_PB_array, Deakin_distances_PB_pp_ids = distances_SA1_PP_array("Deakin", "PB")

def weights_SA1_PP_array(distances_SA1_PP_array, pp_ids_set, div_nm, dist_func):
    ### dist_func is a string of either inverse or ...
    ### renormalises weights according to size from each SA1
    rec = np.reciprocal(distances_SA1_PP_array)


    # truncates Div_SA1_By_PP_dict_wide to relevant rows
    DSBPDW_array = Div_SA1_By_PP_dict_wide[div_nm].loc[Div_SA1_By_PP_dict_wide[div_nm]["pp_id"].isin(pp_ids_set),:].drop(columns=['pp_id']).to_numpy() # removes pp_id col for array conversion
    #print(Div_SA1_By_PP_dict_wide[div_nm].loc[Div_SA1_By_PP_dict_wide[div_nm]["pp_id"].isin(pp_ids_set),:])
    n_i_per_SA1 = np.sum(DSBPDW_array, axis=0)   

    
    
    # performs sum(l^n_i)/dot products of each column n_i * d_i, 
    denominator = np.sum(rec * DSBPDW_array, axis=0)
    denominator_safe = np.where(denominator == 0.0, 1e-10, denominator)

    #print([round(num, 1) for num in denominator.tolist()], len(denominator.tolist()))
    #print(np.delete(n_i_per_SA1,241) / np.delete(denominator,241))
    
    n_i_d_i_recalibration =  n_i_per_SA1 / denominator_safe
    weights_SA1_PP_array = rec*n_i_d_i_recalibration # sum_ni_SA1 of col in PP_By_SA1_Dict and col %*% row (rec[i,:])
    return weights_SA1_PP_array

#print("weights")
#print(weights_SA1_PP_array(Deakin_distances_PB_array, Deakin_distances_PB_pp_ids, "Deakin", "inverse"))
#print("Done")


def candidate_totals_rebalancing(SA1_vote_array, Booth_type_actual_vote_totals):
    ### inputs an array of SA1*candidates, and a 1d array of actual vote_totals for the division, and aims to shift votes around proportionally
    ### to each SA1's size to match the overall division candidate totals

    curr_cand_totals = np.sum(SA1_vote_array, axis=0)
    SA1_vote_totals = np.sum(SA1_vote_array, axis=1)
    overall_vote_total = sum(SA1_vote_totals)

    if np.round(overall_vote_total,2) != np.sum(Booth_type_actual_vote_totals):
        raise ValueError("Mine: SA1_vote_array's global vote totals don't match")

    adj_weights = SA1_vote_totals / overall_vote_total # proportional to size of SA1
    difference_vector = curr_cand_totals - Booth_type_actual_vote_totals
    # print(difference_vector)

    weighted_difference_array = np.tile(difference_vector, (len(adj_weights),1)) * adj_weights[:,None] # of height equal to #SA1s

    SA1_vote_array += weighted_difference_array

    return SA1_vote_array



def candidate_prior_simulation_weighted(div_nm, mean):
    ### generates array of simulated votes per candidate per SA1 using the doubly-multivariate hypergeometric distribution with given K and n vectors


    num_candidates = Div_First_Prefs_PP_dict_wide[div_nm].shape[1] - 1 # includes informal
    num_SA1s = Div_SA1_By_PP_dict_wide[div_nm].shape[1]-1

    #print(num_candidates,num_pps,num_SA1s)

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

        candidate_totals_for_booth_type = Div_First_Prefs_PP_dict_wide["Deakin"].loc[Div_First_Prefs_PP_dict_wide["Deakin"]['pp_id'].isin(booth_type_pp_id_set),].sum().values[1:]
        Booth_type_result_array = candidate_totals_rebalancing(Booth_type_result_array, candidate_totals_for_booth_type)

        result_array += Booth_type_result_array



    # lastly, Other: pp_id is always 0!
    Other_adjust = 1

    if Other_adjust:
        # obtain totals and percentages of Other_PPs vs PB+PPVCs
        Other_PP_cand_totals = Div_First_Prefs_PP_dict_wide["Deakin"].loc[Div_First_Prefs_PP_dict_wide["Deakin"]['pp_id']==0,].sum().values[1:]
        Other_PP_cand_pcts = Other_PP_cand_totals / np.sum(Other_PP_cand_totals)
        Div_cand_totals = Div_First_Prefs_PP_dict_wide["Deakin"].sum().values[1:]
        PB_PPVC_cand_totals = Div_cand_totals - Other_PP_cand_totals
        PB_PPVC_cand_pcts = PB_PPVC_cand_totals / np.sum(PB_PPVC_cand_totals)
        
        Other_minus_PB_PPVC_pcts = Other_PP_cand_pcts - PB_PPVC_cand_pcts

        # convert result_array to percentages to give estimate for Other vote (good estimate if percentages were to be held constant)
        row_sums = result_array.sum(axis=1,keepdims=True)
        result_array_pcts = np.where(row_sums == 0, 1/result_array.shape[1], result_array / row_sums) # replace with even prior over all candidates

        expected_Other_array = result_array_pcts * Div_SA1_By_PP_dict_wide["Deakin"].iloc[0,1:].to_numpy()[:,np.newaxis] # multiply by the n_i from 'Other' for each SA1

        print(expected_Other_array)
        print("Comparison")
        print(Other_PP_cand_totals)
        print(np.sum(expected_Other_array, axis=0))

        Other_adjusted_array = candidate_totals_rebalancing(expected_Other_array, Other_PP_cand_totals)    
        print("Here we go")
        print(Other_adjusted_array)   
        print((Other_adjusted_array<0).any())
        neg_indices = np.where(Other_adjusted_array < 0)
        print(neg_indices)
        for i in neg_indices:
            print(np.round(Other_adjusted_array[neg_indices],2)) 

        result_array += Other_adjusted_array
        return np.round(result_array,6)

        #import pdb;pdb.set_trace()




    K = Div_First_Prefs_PP_dict_wide[div_nm][Div_First_Prefs_PP_dict_wide[div_nm]['pp_id'] == 0].iloc[0, 1:].tolist()
    n = Div_SA1_By_PP_dict_wide[div_nm][Div_SA1_By_PP_dict_wide[div_nm]['pp_id'] == 0].iloc[0, 1:].tolist()
    N = sum(K)

    if mean:
        curr_array = MMHg_mean(N,K,n) # expected value
    else:
        curr_array = doubly_multivariate_hypergeometric_simulate(N,K,n)
    result_array += curr_array


    #print(time.time()-start, "seconds")
    return np.round(result_array,6)


def SA1_candidate_prior_df_output(div_nm,mean):

    array = candidate_prior_simulation_weighted("Deakin", mean)

    SA1s = Div_SA1_By_PP_dict_wide[div_nm].columns[1:].tolist()
    candidates = Div_First_Prefs_PP_dict_wide[div_nm].columns[1:].tolist()
    print(SA1s)
    print(candidates)

    SA1_candidate_prior_df = pd.DataFrame(array, index=SA1s, columns=candidates)

    return SA1_candidate_prior_df

print("Getting df of output!!!")
Deakin_Aston_df = SA1_candidate_prior_df_output("Deakin",1)
print(Deakin_Aston_df.loc[Deakin_Aston_df.index.isin(Deakin_Aston_SA1s),])
import pdb;pdb.set_trace()

print("starting sim")
Deakin_prior = candidate_prior_simulation_weighted("Deakin", 1)

# check votes of 1st SA1 - out of interest
#sum1 = np.sum(Deakin_prior[0,:-1])
#print(Deakin_prior[0,:]/sum1)
#print(Deakin_prior[0,:])

# check SA1 vote totals are almost integer
np.set_printoptions(precision=2,suppress=True)
print("rounded array", np.round(np.sum(Deakin_prior, axis = 1),2))
print(np.sum(Deakin_prior))
#import pdb;pdb.set_trace()

print(Deakin_prior)

print(np.sum(Deakin_prior,axis=0)) # candidate vote totals - we should adjust!

print("MMHg simulation")
MMHG_Deakin = candidate_prior_simulation("Deakin")
print(np.sum(MMHG_Deakin,axis=0))
print(np.round(np.sum(MMHG_Deakin, axis = 1),1))
print(np.sum(MMHG_Deakin))
print(MMHG_Deakin)





#Deakin_actual_vote_totals_array = Div_First_Prefs_PP_dict_wide["Deakin"].sum().values[1:]
Deakin_adjusted = Deakin_prior #candidate_totals_rebalancing(Deakin_prior, Deakin_actual_vote_totals_array)
print(Deakin_adjusted)
print((Deakin_adjusted<0).any())
neg_indices = np.where(Deakin_adjusted < 0)
print(neg_indices)
for i in neg_indices:
    print(Deakin_adjusted[neg_indices])


print(time.time() - start, "seconds")


SA1 = 1000000 # to add
#SA1_Lat = SA1_centroids_2016.loc[SA1_centroids_2016["SA1_CODE16"] == SA1,"Lat"].iloc[0]
#SA1_Long = SA1_centroids_2016.loc[SA1_centroids_2016["SA1_CODE16"] == SA1,"Long"].iloc[0]
#Golstein_Distances_Locations = pd.DataFrame({
#    "Distance": DEGREE_TO_KM*np.sqrt((Goldstein_PB[["Lat"]].iloc[:, 0]- SA1_Lat)**2 + (Goldstein_PB[["Long"]].iloc[:, 0]- SA1_Long)**2),
#    "Booth_type": Goldstein_PB["Booth_type"]})

### TO DO:
### 1. Get centroids of SA1s, and of polling places
### 2. Find distance between SA1 and polling places
### 3. Weight function for each distance
### 4. Weighted combination for each pp_type (PB, PPVC, Other)
### 5. Adjustment towards actual result for each pp_type (PB, PPVC, Other)





#Deakin_array = candidate_prior_simulation("Deakin")
#print(Deakin_array)

#print(Deakin_Aston_SA1s)


#Deakin_SA1s = unique_SA1s_dict(Div_SA1_By_PP_dict)["Deakin"]
#print(Deakin_SA1s)

#indices = [Deakin_SA1s.tolist().index(sa1) for sa1 in Deakin_Aston_SA1s]

# Extract the corresponding rows from the data array and sum them
#result = np.sum(Deakin_array[indices, :], axis=0)

#print("Summed rows for the subset of SA1s:", result)
