import pandas as pd
import geopandas as gpd
import numpy as np
import os, time
import matplotlib
from matplotlib import pyplot as plt

os.chdir('C:\\Dania\\2024\\Australian Election')

SA1_By_PP_Complete = pd.read_csv("SA1_By_PP_Complete.csv", index_col=None)



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
Div_SA1_By_PP_dict_wide["Deakin"]["n"] = Div_SA1_By_PP_dict_wide["Deakin"].iloc[:, 1:].sum(axis=1)
Div_First_Prefs_PP_dict_wide["Deakin"]["K"] = Div_First_Prefs_PP_dict_wide["Deakin"].iloc[:, 1:].sum(axis=1)

all = pd.merge(Div_SA1_By_PP_dict_wide["Deakin"],Div_First_Prefs_PP_dict_wide["Deakin"][["pp_id","K"]], on = 'pp_id',how='left')
all.loc[:,"vote_difference"] = all["n"] - all["K"]

#print(Div_SA1_By_PP_dict_wide["Deakin"])
#print(sum(Div_SA1_By_PP_dict_wide["Deakin"]["n"]))
#print(Div_First_Prefs_PP_dict_wide["Deakin"])
#print(sum(Div_First_Prefs_PP_dict_wide["Deakin"]["K"]))

#print(all)


def rectify_Deakin_SA1_votes():

    Div_SA1_By_PP_dict_wide["Deakin"]["n"] = Div_SA1_By_PP_dict_wide["Deakin"].iloc[:, 1:].sum(axis=1)
    Div_First_Prefs_PP_dict_wide["Deakin"]["K"] = Div_First_Prefs_PP_dict_wide["Deakin"].iloc[:, 1:].sum(axis=1)

    all = pd.merge(Div_SA1_By_PP_dict_wide["Deakin"],Div_First_Prefs_PP_dict_wide["Deakin"][["pp_id","K"]], on = 'pp_id',how='left')
    all.loc[:,"vote_difference"] = all["n"] - all["K"]
    Div_First_Prefs_PP_dict_wide["Deakin"].drop(columns=['K']) # reverse alteration

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



arr = [2.7200e+02, 3.8200e+02, 4.1600e+02].to_numpy()
print(round(arr,2))