import pandas as pd
import numpy as np
import os, time
import gc  # Garbage Collection
from collections import Counter


os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()

SenateCandidates_2016 = pd.read_csv("2016SenateCandidates.csv", index_col = None)
SenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016["nom_ty"] == 'S',["state_ab","ticket","party_ballot_nm"]]
SenateCandidates_2016.rename(columns={"state_ab": "StateAb", "party_ballot_nm": "party_nm"}, inplace=True)

Formal_Prefs_colnames = {}





def abbreviate_party_names(party_names_list, general_party_df):
    # handle exceptions to party names
    party_abvs_list = []

    for party in party_names_list:
        #print(party)
        if party:
            if party.lower() == "Liberal/The Nationals".lower() or party.lower() == "Liberal & Nationals".lower() or party.lower() == "Liberal/National".lower(): # handle LIB/NAT Exception - I think best to treat them as one party in the Senate as they always contest together, and then reverse engineer House split if needed
                party_abvs_list.append('COAL')
            elif party == " Science, Pirate, Secular, Climate Emergency": # SOPA exception
                party_abvs_list.append('SOPA')
            elif party == "Labor/Country Labor":
                party_abvs_list.append('ALP')
            elif party == "Science Party/Australian Cyclists Party":
                party_abvs_list.append('FTCY')
            elif party == 'Australian Sex Party/Marijuana (HEMP) Party':
                party_abvs_list.append('SXHM')
            else:
                party_abvs_list.append(general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].iloc[0])
        else:
            party_abvs_list.append('')


        #import pdb;pdb.set_trace()

    return party_abvs_list



def get_Senate_party_abvs_dict():
    # quickly extracts abvs from the senate without needing to read all of Formal Prefs

    # get state-to-div dict
    div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
    div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}



    Senate_party_abvs_dict = {}
    for div in div_to_state_dict.keys():
        #import pdb;pdb.set_trace()
        state = div_to_state_dict[div] # gets StateAb
        formal_prefs_full = Formal_prefs_dict[state]
        formal_prefs = formal_prefs_full.iloc[:, 6:]
        formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
        start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated

        # store group party names (from ATL) in Senate_party_names_dict
        group_party_names = formal_prefs_full.iloc[:, 6:].columns[:start_of_BTL_index] # includes both group and party names
        party_names_list = group_party_names.str.split(':').str[-1].tolist() # records only party names
        print(div)
        party_abvs_list = abbreviate_party_names(party_names_list, general_party_df)
        Senate_party_abvs_dict[div] = party_abvs_list
    
    # write to csv
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) 

    return 1

def make_unique(names):
    """ Append suffixes to duplicate column names to make them unique. """
    counts = Counter()
    unique_names = []
    
    for name in names:
        if counts[name] > 0:
            new_name = f"{name}_dup{counts[name]}"  # Add suffix
        else:
            new_name = name
        counts[name] += 1
        unique_names.append(new_name)

    return unique_names




general_party_df = pd.read_csv("2016GeneralPartyDetails.csv", skiprows = 1)

state_group_party_names_dict = {}

states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
for state in states:
    StateSenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016['StateAb'] == state,:]
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].fillna('')

    # COALITION(S) inspect if there are coalitions and give them PartyAb
    nonUG = StateSenateCandidates_2016.loc[~(StateSenateCandidates_2016['ticket']=='UG'),]
    coalition_df = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1].iloc[:,1:].drop_duplicates(ignore_index=True).groupby("ticket", as_index=False)['party_nm'].agg("/".join)


    if not coalition_df.empty:
        coalition_group_dict = coalition_df.set_index("ticket")["party_nm"].to_dict()
        #coalition_party_names = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1]['party_nm'].unique()

        # map dictionary
        StateSenateCandidates_2016.loc[:,"party_nm"] = StateSenateCandidates_2016["ticket"].map(coalition_group_dict).where(StateSenateCandidates_2016["ticket"].isin(coalition_group_dict), StateSenateCandidates_2016["party_nm"])


    party_names_list = StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket']!='UG','party_nm'].drop_duplicates(ignore_index=True).tolist()

    # convert to PartyAb and format for Formal Preferences
    party_abvs = abbreviate_party_names(party_names_list, general_party_df)

    party_names_abvs_dict = dict(zip(party_names_list,party_abvs))
    StateSenateCandidates_2016 = StateSenateCandidates_2016.copy() # avoid warning ..?
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].replace(party_names_abvs_dict)
    StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket'] == 'UG','party_nm'] = ''

    group_party_names = StateSenateCandidates_2016['ticket'].astype(str) + ':' + StateSenateCandidates_2016['party_nm'].astype(str)
    unique_groups = group_party_names[~group_party_names.str.startswith('UG')].drop_duplicates(ignore_index=True)

    group_party_names = unique_groups.tolist() + group_party_names.tolist()

    state_group_party_names_dict[state] = group_party_names

import pdb;pdb.set_trace()









def split_and_convert(series, cols, chunk_size=500000, output_file="processed_data.parquet", return_df=True):
    """Processes large CSV-like columns in chunks, writes to disk, and optionally returns a final DataFrame."""
    
    max_columns = len(cols)

    def parse_row(s):
        """Splits a row and converts elements to integers, handling empty values."""
        items = s.rstrip(',').split(',')
        return [int(i) if i.isdigit() else np.nan for i in items[:max_columns]] + [np.nan] * (max_columns - len(items))

    first_chunk = True
    for chunk_start in range(0, len(series), chunk_size):
        chunk = series.iloc[chunk_start : chunk_start + chunk_size]  # Select batch
        chunk_parsed = np.array([parse_row(row) for row in chunk], dtype=np.float32)  # Process batch
        
        # ✅ Convert to Pandas DataFrame (but do not store it in memory)
        chunk_df = pd.DataFrame(chunk_parsed)

        # ✅ Convert to Pandas nullable Int16 (fixes NaN issues)
        chunk_df = chunk_df.convert_dtypes().astype("Int16")

        chunk_df.columns = make_unique(cols) 

        # ✅ Write chunk directly to file (No concatenation)
        chunk_df.to_parquet(output_file, engine="fastparquet", index=False, compression="snappy", append=not first_chunk)
        first_chunk = False  # Append mode for next chunks

        # ✅ Explicitly delete unused objects to free memory
        del chunk, chunk_parsed, chunk_df
        gc.collect()  # Force garbage collection

        print(f"Processed and saved chunk {chunk_start // chunk_size + 1}")

    print("Processing complete. Data saved to", output_file)

    # ✅ Optionally return DataFrame (but loads everything into memory)
    if return_df:
        print("Loading final DataFrame into memory...")
        return pd.read_parquet(output_file, engine="fastparquet")
    





    





# read in FPs and format correctly: eliminate first row and convert preferences string to csv format
for state in states:
    curr_formal_prefs_2016 = pd.read_csv(f"2016FormalPrefs{state}.csv", index_col = None, usecols=['ElectorateNm', 'VoteCollectionPointNm','Preferences'], dtype={-1: str}).rename(columns = {'ElectorateNm': 'div_nm', 'VoteCollectionPointNm': 'pp_nm'})
    curr_formal_prefs_2016 = curr_formal_prefs_2016.iloc[1:,] # eliminate ---- row

    last_col = curr_formal_prefs_2016.columns[-1]
    # replace any non-int or empty values with 1 (* or /)
    curr_formal_prefs_2016.iloc[:,-1] = curr_formal_prefs_2016.iloc[:,-1].apply(lambda x: ','.join(['1' if not val.isdigit() and val != '' else val for val in x.split(',')]))

    max_col_no = len(state_group_party_names_dict[state]) # number of candidate/party boxes
    expanded_cols = split_and_convert(curr_formal_prefs_2016.iloc[:,-1],cols = state_group_party_names_dict[state])
    #expanded_cols = expanded_cols.astype('float32')

    new_column_names = state_group_party_names_dict[state]

    if len(new_column_names) == expanded_cols.shape[1]:
        expanded_cols.columns = new_column_names

    curr_formal_prefs_2016 = pd.concat([curr_formal_prefs_2016.drop(columns=[last_col]), expanded_cols], axis=1)


    curr_formal_prefs_2016.to_csv(f"2016FormalPrefs{state}Formatted.csv", index = False)

    print("done", time.time() - start)

    import pdb;pdb.set_trace()


