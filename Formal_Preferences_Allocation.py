import pandas as pd
import numpy as np
import os,time
import ast
from collections import Counter
import io


import gc

os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()


FP_ID_COLUMNS = [3,4,5] # remove id columns
START_OF_PREFS = 2 # Prefs begin on the 3th column (after div_nm,pp_nm) - deleted stateab to accomodate 2016 file


incumbent_advantage_dict = {5:4.68,4:4.5,3:6}
final_cand_no_dict = {"2022":5, "2019": 4, "2016": 4,"2013": 5, "2010": 3, "2007": 4, "2004": 4,"2001":4}
data_year = '2022'
FINAL_CANDIDATE_NO = final_cand_no_dict[data_year]
INCUMBENT_ADVANTAGE = incumbent_advantage_dict[FINAL_CANDIDATE_NO]

NONINCUMBENT_DISADVANTAGE =  INCUMBENT_ADVANTAGE/(FINAL_CANDIDATE_NO-1)

data_year = '2022'



# Game plan:
# Four different places where FormalPrefs will be necessary:
# 1/2. Redistribution changes & candidate changes - (1. Simultaneously: ~140+110 or 2. nowcast ~110 for 2025) (IF nowcast model)
# 3. Incumbent advantage - (150) (Possibly simultaneously across all elections)
# 4. Polling_Top_x (150)
# - First Preferences required for each
# - 1/2(possibly 3) need polling-booth specifics
# Should be done simultaenously, saved to candidate-wide dfs of length 150 or 150*50 (without or with polling booth specifics)
# Write 4 csv files: 
# 1/2. Candidate changes | div_nm, pp_id vs parties     (len(parties) might differ)
# 3. Incumbent advantage | div_nm (, pp_id) vs parties  (len(parties) the same!)
# 4. Polling_Top_x | div_nm vs parties (len(parties) the same!)

# First, read and generate dict of FormalPrefs by div, from the state csv's
# Produce FPrefs once - modify existing dict
# For each of 1,2,3,4:
# If div is in 1list,2list,3list & all for 4: allocate, aggregate and add to corresponding df to eventually write to csv.

######### Candidate Pairs stuff

DOP_By_PP_Expand = pd.read_csv("2022DOP_By_PP_Expand.csv", index_col=None)
DOP_By_PP_Pref_Percent = pd.read_csv("2022DOP_By_PP_Pref_Percent.csv", index_col=None)

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

    if df_type == "DOP_By_PP":

        pivot_df = df.pivot_table(index=['pp_id','CountNumber'], # double index for info across pp_ids
                                columns=['PartyAb'], 
                                values='CalculationValue', 
                                aggfunc='first',
                                sort = False)  # No duplicates, so we can use 'first'
        
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df.reset_index()

    return pivot_df

def convert_long_to_wide_format(DOP_table_long, div_to_state_dict):
    ### creates dict for div_nm as key and wide DOP table for each pp_id as value

    DOP_By_PP_dict = {div: group for div, group in DOP_table_long.groupby('div_nm')}

    for div, group in DOP_By_PP_dict.items():

        target = 'IND' # relabel independents in order of ballot appearance if there are multiple

        group_sample_zero = group.loc[group['pp_id']==0,] # always will have one
        group_sample_zero = group_sample_zero.copy()
        group_sample_zero.loc[:,'Count'] = (group_sample_zero.groupby('PartyAb').cumcount() + 1)     # Count instances of the target string

        adjusted_party_names = group_sample_zero.loc[group_sample_zero["CountNumber"] == 0,].apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1)
        
        num_pref_counts = (group_sample_zero.iloc[-1,3] + 1) # num of final count + original FP count

        num_rows = len(group['pp_id'].unique()) * num_pref_counts

        group.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * (num_rows), ignore_index=True).values
        group.loc[group["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!

        DOP_table_wide = convert_to_wide_format(group, "DOP_By_PP")

        # give INDs distinct names based on division and convert LP and NP into COAL in Victoria/NSW
        for party in DOP_table_wide.columns[2:]:
            if party.startswith('IND'):
                DOP_table_wide.rename(columns = {party: party + div}, inplace = True) # e.g. IND1Goldstein

            # convert LP and NP in VIC/NSW to COAL
            if div_to_state_dict[div] in ['VIC','NSW']:
                if (party=='NP') | (party =='LP'):
                    DOP_table_wide.rename(columns = {party: 'COAL'}, inplace = True)

        DOP_By_PP_dict[div] = DOP_table_wide

    return DOP_By_PP_dict


def create_wide_DOP_dict(Div_DOP_dict, DOP_type):
    
    DOP_table_wide_dict = {}

    if DOP_type == "EliminationOrder":
        # get state-to-div dict
        div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
        div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}

        for div in Div_DOP_dict.keys():
            #print(div)
            FP_pcts = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] == 0) & (Div_DOP_dict[div]["CalculationType"] == "Preference Percent"),]
            Transfer_pcts = Div_DOP_dict[div].loc[(Div_DOP_dict[div]["CountNumber"] > 0) & (Div_DOP_dict[div]["CalculationType"] == "Transfer Percent"),]
            DOP_table_long = pd.concat([FP_pcts, Transfer_pcts], ignore_index=True)

            # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
            DOP_table_long.loc[:,'PartyAb'] = DOP_table_long['PartyAb'].fillna('IND') 

            # relabel independents in order of ballot appearance if there are multiple
            target = 'IND'
            DOP_table_long['Count'] = DOP_table_long.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
            # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
            adjusted_party_names = DOP_table_long.loc[DOP_table_long["CountNumber"] == 0,].apply(
                lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
            )
            num_pref_counts = (DOP_table_long.iloc[-1,0] + 1) # num of final count + original FP count

            DOP_table_long.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * num_pref_counts, ignore_index=True)
            DOP_table_long.loc[DOP_table_long["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!


            DOP_table_long = DOP_table_long.drop(columns=['Count'])
            DOP_table_wide = convert_to_wide_format(DOP_table_long, "DOP")
            
            # record elimination order
            Elim_order_list_part = DOP_table_wide.iloc[1:,].apply(lambda row: row[row == -100.00].index[0], axis=1).tolist()# Apply the function row-wise to get the column names
            Final_2_Parties = DOP_table_wide.iloc[-1,1:][DOP_table_wide.iloc[-1,] > 0].index.tolist()
            Elim_order_list = Elim_order_list_part + Final_2_Parties

            # give INDs distinct names based on division and convert LP and NP into COAL in Victoria
            for i, party in enumerate(Elim_order_list):
                if party.startswith('IND'):
                    Elim_order_list[i] = party + div # e.g. IND1Goldstein

                # convert LP and NP in VIC/NSW to COAL
                if div_to_state_dict[div] in ['VIC','NSW']:
                    if (party=='NP') | (party =='LP'):
                        Elim_order_list[i] = 'COAL'

            


            DOP_table_wide_dict[div] = Elim_order_list[::-1] # need to still reverse


    

    return DOP_table_wide_dict


# get state-to-div dict
div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
div_to_state_dict = {div: div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}

# create wide format eliminaation_order_dict
DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)
DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
Div_DOP_dict = {div: group.drop(columns=['div_nm']) for div, group in DOP_By_Division[["div_nm","CountNumber","cand_id", "PartyAb","CalculationType", "CalculationValue"]].groupby("div_nm")}

Elimination_order_dict = create_wide_DOP_dict(Div_DOP_dict, DOP_type = "EliminationOrder")

DOP_By_PP_Pref_Percent_wide_dict = convert_long_to_wide_format(DOP_By_PP_Pref_Percent, div_to_state_dict)
DOP_By_PP_Expand_wide_dict = convert_long_to_wide_format(DOP_By_PP_Expand, div_to_state_dict)












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
                if general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].empty:
                    import pdb;pdb.set_trace()
                party_abvs_list.append(general_party_df.loc[(general_party_df["PartyNm"] == party) | (general_party_df["RegisteredPartyAb"] == party),"PartyAb"].iloc[0])
        else:
            party_abvs_list.append('')


        #import pdb;pdb.set_trace()

    return party_abvs_list


general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',"PartyAb"] = 'GRN' # handle exceptions, but think GVIC is the only one



def get_2016_Senate_party_names(state, return_PartyAbs = False):
    ### reads in the unusual 2016 Formal Prefs csv file, filling in the header column with senate groupings

    #1. LOAD DATA & CLEAN
    SenateCandidates_2016 = pd.read_csv("2016SenateCandidates.csv", index_col = None)
    SenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016["nom_ty"] == 'S',["state_ab","ticket","party_ballot_nm"]]
    SenateCandidates_2016.rename(columns={"state_ab": "StateAb", "party_ballot_nm": "party_nm"}, inplace=True)

    StateSenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016['StateAb'] == state,:]
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].fillna('') # for ungrouped

    # 2. COALITION(S) inspect if there are coalitions and give them PartyAb
    nonUG = StateSenateCandidates_2016.loc[~(StateSenateCandidates_2016['ticket']=='UG'),]
    coalition_df = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1].iloc[:,1:].drop_duplicates(ignore_index=True).groupby("ticket", as_index=False)['party_nm'].agg("/".join)


    if not coalition_df.empty:
        coalition_group_dict = coalition_df.set_index("ticket")["party_nm"].to_dict() #coalition_party_names = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1]['party_nm'].unique()
        # map dictionary
        StateSenateCandidates_2016.loc[:,"party_nm"] = StateSenateCandidates_2016["ticket"].map(coalition_group_dict).where(StateSenateCandidates_2016["ticket"].isin(coalition_group_dict), StateSenateCandidates_2016["party_nm"])

    party_names_list = StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket']!='UG','party_nm'].drop_duplicates(ignore_index=True).tolist()

    # 3. convert to PartyAb and format for Formal Preferences
    party_abvs = abbreviate_party_names(party_names_list, general_party_df)

    if return_PartyAbs:
        return party_abvs

    party_names_abvs_dict = dict(zip(party_names_list,party_abvs))
    StateSenateCandidates_2016 = StateSenateCandidates_2016.copy() # avoid warning ..?
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].replace(party_names_abvs_dict)
    StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket'] == 'UG','party_nm'] = ''

    # format string column names
    group_party_names = StateSenateCandidates_2016['ticket'].astype(str) + ':' + StateSenateCandidates_2016['party_nm'].astype(str)
    unique_groups = group_party_names[~group_party_names.str.startswith('UG')].drop_duplicates(ignore_index=True)
    group_party_names = unique_groups.tolist() + group_party_names.tolist()

    return group_party_names


def get_Senate_party_abvs_dict(data_year, div_to_state_dict, to_csv = False):
    # quickly extracts abvs from the senate without needing to read all of Formal Prefs


    Formal_prefs_dict = {}
    states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']

    #### basic version to get the party names lists for cheap - read only 2 rows each!
    if data_year == '2016':
        state_party_abvs_list = {}
        for state in states:
            state_party_abvs_list[state] = get_2016_Senate_party_names(state, return_PartyAbs = True)
        
        Senate_party_abvs_dict = {}
        for div in div_to_state_dict.keys():
            state = div_to_state_dict[div]
            Senate_party_abvs_dict[div] = state_party_abvs_list[state]

    else:
        for state in states: # currently only 2016 onwards
            filename = f"{data_year}FormalPrefs{state}.csv"

            state_Formal_prefs = pd.read_csv(filename, nrows=1)
            state_Formal_prefs.drop(columns=state_Formal_prefs.columns[FP_ID_COLUMNS], inplace=True)
            

            state_Formal_prefs_dict = {state: group.reset_index(drop=True).apply(
                lambda col: pd.to_numeric(col, downcast='float') if pd.api.types.is_numeric_dtype(col) else col
            ) for state, group in state_Formal_prefs.groupby("State")} 

            for key, group in state_Formal_prefs_dict.items():
                group.pop('State') # remove State for concordance with later dfs
                Formal_prefs_dict[key] = group # assumes no keys (divs) overlap for different states :)

        Senate_party_abvs_dict = {}
        for div in div_to_state_dict.keys():
            #import pdb;pdb.set_trace()
            state = div_to_state_dict[div] # gets StateAb
            formal_prefs_full = Formal_prefs_dict[state]
            formal_prefs = formal_prefs_full.iloc[:, START_OF_PREFS:]
            formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
            start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated

            # store group party names (from ATL) in Senate_party_names_dict
            group_party_names = formal_prefs_full.iloc[:, START_OF_PREFS:].columns[:start_of_BTL_index] # includes both group and party names
            party_names_list = group_party_names.str.split(':').str[-1].tolist() # records only party names
            party_abvs_list = abbreviate_party_names(party_names_list, general_party_df)
            Senate_party_abvs_dict[div] = party_abvs_list
    
    # write to csv
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) 

    return Senate_party_abvs_dict


Senate_party_abvs_dict = get_Senate_party_abvs_dict(data_year, div_to_state_dict, to_csv = False)


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




def split_and_convert_2016(series, cols, chunk_size=500000, output_file="processed_data.parquet", return_df=True):
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
    
def get_2016_Senate_party_names(state):
    ### reads in the unusual 2016 Formal Prefs csv file, filling in the header column with senate groupings

    #1. LOAD DATA & CLEAN
    SenateCandidates_2016 = pd.read_csv("2016SenateCandidates.csv", index_col = None)
    SenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016["nom_ty"] == 'S',["state_ab","ticket","party_ballot_nm"]]
    SenateCandidates_2016.rename(columns={"state_ab": "StateAb", "party_ballot_nm": "party_nm"}, inplace=True)

    StateSenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016['StateAb'] == state,:]
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].fillna('') # for ungrouped

    # 2. COALITION(S) inspect if there are coalitions and give them PartyAb
    nonUG = StateSenateCandidates_2016.loc[~(StateSenateCandidates_2016['ticket']=='UG'),]
    coalition_df = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1].iloc[:,1:].drop_duplicates(ignore_index=True).groupby("ticket", as_index=False)['party_nm'].agg("/".join)


    if not coalition_df.empty:
        coalition_group_dict = coalition_df.set_index("ticket")["party_nm"].to_dict() #coalition_party_names = nonUG[nonUG.groupby('ticket')['party_nm'].transform('nunique')>1]['party_nm'].unique()
        # map dictionary
        StateSenateCandidates_2016.loc[:,"party_nm"] = StateSenateCandidates_2016["ticket"].map(coalition_group_dict).where(StateSenateCandidates_2016["ticket"].isin(coalition_group_dict), StateSenateCandidates_2016["party_nm"])

    party_names_list = StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket']!='UG','party_nm'].drop_duplicates(ignore_index=True).tolist()

    # 3. convert to PartyAb and format for Formal Preferences
    party_abvs = abbreviate_party_names(party_names_list, general_party_df)

    party_names_abvs_dict = dict(zip(party_names_list,party_abvs))
    StateSenateCandidates_2016 = StateSenateCandidates_2016.copy() # avoid warning ..?
    StateSenateCandidates_2016.loc[:,'party_nm'] = StateSenateCandidates_2016.loc[:,'party_nm'].replace(party_names_abvs_dict)
    StateSenateCandidates_2016.loc[StateSenateCandidates_2016['ticket'] == 'UG','party_nm'] = ''

    # format string column names
    group_party_names = StateSenateCandidates_2016['ticket'].astype(str) + ':' + StateSenateCandidates_2016['party_nm'].astype(str)
    unique_groups = group_party_names[~group_party_names.str.startswith('UG')].drop_duplicates(ignore_index=True)
    group_party_names = unique_groups.tolist() + group_party_names.tolist()

    return group_party_names

def get_2016_Formal_Prefs(state):
    ### reads in the unusual 2016 Formal Prefs csv file, filling in the header column with senate groupings, and expanding the csv string into a df

    # 1. get group party names
    group_party_names = get_2016_Senate_party_names(state)


    # 2. PROCESS Formal_Prefs_file and add columns
    curr_formal_prefs_2016 = pd.read_csv(f"2016FormalPrefs{state}.csv", index_col = None, usecols=['ElectorateNm', 'VoteCollectionPointNm','Preferences'], dtype={-1: str}).rename(columns = {'ElectorateNm': 'div_nm', 'VoteCollectionPointNm': 'pp_nm'})
    curr_formal_prefs_2016 = curr_formal_prefs_2016.iloc[1:,].reset_index(drop=True) # eliminate ---- row

    last_col = curr_formal_prefs_2016.columns[-1]
    # replace any non-int or empty values with 1 (* or /)
    curr_formal_prefs_2016.iloc[:,-1] = curr_formal_prefs_2016.iloc[:,-1].apply(lambda x: ','.join(['1' if not val.isdigit() and val != '' else val for val in x.split(',')]))

    max_col_no = len(group_party_names) # number of candidate/party boxes - group_party_names is, as above, the intended header for the df
    expanded_cols = split_and_convert_2016(curr_formal_prefs_2016.iloc[:,-1],cols = group_party_names)     #expanded_cols = expanded_cols.astype('float32')

    new_column_names = group_party_names

    if len(new_column_names) == expanded_cols.shape[1]:
        expanded_cols.columns = new_column_names

    curr_formal_prefs_2016 = pd.concat([curr_formal_prefs_2016.drop(columns=[last_col]), expanded_cols], axis=1)

    return curr_formal_prefs_2016

def get_2019_NSW_Formal_Prefs(filename, state):
    df_columns = pd.read_csv(filename, index_col=None, nrows=1).columns
    # fix malformed file using readlines:
    expected_columns = len(df_columns)  # You can adjust this number to match your actual expected columns
    batch_size = 1_000_000  # Process 1 million rows at a time
    dataframes = []  # Store DataFrames in a lis

    # Read the CSV file line by line
    with open(filename, 'r') as file:
        
        buffer = io.StringIO()  # Create an in-memory file buffer

        batch = []
        row_count = 0  # Keep track of rows read

        for i, line in enumerate(file, start = -1):
            if i == -1:
                continue  # Skip the first row

            row = line.strip().split(',') # Split the line by commas

            # Fix malformed rows by padding or truncating
            if len(row) < expected_columns:
                row.extend([np.nan] * (expected_columns - len(row)))
            elif len(row) > expected_columns:
                import pdb;pdb.set_trace()

                print("AHHHHHHHHHHHH")
                row = row[:expected_columns]
            
            buffer.write(",".join(map(str, row)) + "\n")
            row_count += 1

            # Process and store each batch
            if row_count % batch_size == 0:
                print(f"Processing batch {len(dataframes) + 1}, rows read: {row_count}")  # Debugging
                buffer.seek(0)  # Move to start of buffer
                df_batch = pd.read_csv(buffer, names=df_columns)
                df_batch.drop(columns=df_batch.columns[FP_ID_COLUMNS], inplace=True)
                dataframes.append(df_batch)

                buffer = io.StringIO()  # Reset buffer
                del df_batch
                gc.collect() # free memory
                batch = []  # free memory
                print("done", time.time() - start)

        # Process any remaining rows
        if buffer.tell() > 0:
            buffer.seek(0)
            df_batch = pd.read_csv(buffer, names=df_columns)
            print(f"Processing final batch, rows read: {row_count}")  # Debugging


            df_batch.drop(columns=df_batch.columns[FP_ID_COLUMNS], inplace=True) # remove id columns
            #df_batch.iloc[:, START_OF_PREFS:] = df_batch.iloc[:, START_OF_PREFS:]#.apply(pd.to_numeric, errors='coerce') #.astype('Int16')
            dataframes.append(df_batch)

            del df_batch
            gc.collect() # free memory # free memory
            print("done", time.time() - start)

    # Combine all batches into a final DataFrame
    if dataframes:
        curr_Formal_prefs = pd.concat(dataframes, ignore_index=True)
        for col in curr_Formal_prefs.columns[3:]:
            curr_Formal_prefs[col] = curr_Formal_prefs[col].astype(np.float32) # convert to float32 when finished
            #print(col)
        dataframes = [] # free memory
        print("Final DataFrame shape:", curr_Formal_prefs.shape)
    else:
        print("No data was read from the file.")

    return curr_Formal_prefs


def optimize_dataframe(df):
    # Convert all remaining numeric columns to smallest possible int type
    df.iloc[:, 3:] = df.iloc[:, 3:].apply(pd.to_numeric, downcast='integer')

    return df


Formal_prefs_dict = {}
states = ['VIC']
#states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
for state in states: # currently only 2016 onwards

    gc.collect()
    print(state)
    filename = f"{data_year}FormalPrefs{state}.csv"

    if data_year == '2016':
        curr_Formal_prefs = get_2016_Formal_Prefs(state)

        # convert to float 32
        for col in curr_Formal_prefs.columns[2:]:
            curr_Formal_prefs[col] = curr_Formal_prefs[col].fillna(np.nan).astype('float32', copy=False)
        #import pdb;pdb.set_trace()


    elif (state == 'NSW') & (data_year == '2019'):
        curr_Formal_prefs = get_2019_NSW_Formal_Prefs(filename, state) # deal with malformed csv file
        curr_Formal_prefs.drop(columns=["State"], inplace=True)


    else:
        # low memory try - convert to float32 later when using!
        columns = pd.read_csv(filename, nrows=1).columns
        dtype_dict = {col: 'str' for col in columns[:3]}
        for col in columns[3:]:
            dtype_dict[col] = 'float32'

        curr_Formal_prefs = pd.read_csv(filename, index_col=None, na_values=["NaN", "nan"], dtype=dtype_dict)
        curr_Formal_prefs.drop(columns=curr_Formal_prefs.columns[FP_ID_COLUMNS].tolist() + ['State'], inplace=True) # remove id  and State columns

    
    curr_Formal_prefs.rename(columns={"Division": "div_nm", "Vote Collection Point Name": "pp_nm"}, inplace=True)

    for div, group in curr_Formal_prefs.groupby("div_nm"):
        Formal_prefs_dict[div] = group.reset_index(drop=True)  # optimize_dataframe(group.reset_index(drop=True)

    del curr_Formal_prefs

    print("done", time.time() - start)

#import pdb;pdb.set_trace()

# TO DO:
# 1. Fix up indexing so each element of dict starts with 0                          DONE
# 2. First preferences allocate - useful for all!!!                                 DONE
# 3. get ordered list of 5 parties from Incumbent Advantage csv                     DONE
# 4. For each div allocate to specific one                                          DONE
# 5. aggregate into whole or by pp_id                                               DONE
# 6. Use PollingPlacesRepository for correspondence between names and pp_id
# 7. write to csv of senate prefs                                                   DONE
# 8. For 3. Make list of all top 5 that match senate: Senate_Party_Abs per division DONE
        





general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',"PartyAb"] = 'GRN' # handle exceptions, but think GVIC is the only one


def find_earliest_preference_id(preferences):
    # get indices where there may be duplication, store in multiple_min_mask
    # input: df with unique alphabetical column names and integer or nan values

    #votes = preferences.idxmin(axis=1, skipna=True)
    #votes[preferences.isna().all(axis=1)] = np.nan 

    #votes = preferences.astype("float32") # return to float32!
    # votes = votes.fillna(float("inf")).idxmin(axis=1) # min in row, avoids warning
    votes = preferences.fillna(float("inf")).idxmin(axis=1) # min in row, avoids warning
   
    votes = votes.where(preferences.notna().any(axis=1), other=pd.NA) # no prefs in row

    min_values = preferences.min(axis=1)
    mask = preferences.eq(min_values, axis=0)
    multiple_min_mask = mask.sum(axis=1) > 1 # series with True when there are multiple minimum preferences - set as nan and deal with later

    votes[multiple_min_mask] = np.nan

    return votes, multiple_min_mask   # Returns NaN in votes if no preference for the candidate set, series if row min is not unique



def allocate_votes(df, allocation_set): 
    ### allocates vote following algorithm: First, ATL decides. If ATL not decisive, record any duplicates, try BTL. 

    start_of_BTL_index = next(i for i, col in enumerate(df.columns) if df.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated
    ATL = df.iloc[:,:start_of_BTL_index]
    BTL = df.iloc[:,start_of_BTL_index:]
    allocated_votes = pd.DataFrame(index=df.index, columns=['Vote'])

    # ATL preferences
    allocation_set_UG = [col for col in allocation_set if col != 'UG'] # remove UG from ATL preferences if exists - only useful for First_Preferences
    ATL_preferences = ATL[allocation_set_UG]     # Filter the row to only include the candidates of interest
    allocated_votes.loc[:,'Vote'], ATL_non_unique_min = find_earliest_preference_id(ATL_preferences)

    # If no vote is allocated using ATL ('Vote' is nan), use BTL
    BTL_allocations, BTL_non_unique_min = find_earliest_preference_id(BTL[allocation_set]) # selects lowest preference of combined BTL groups
    allocated_votes.loc[:,'Vote'] = allocated_votes.loc[:,'Vote'].fillna(BTL_allocations) 

    ## BTL_non_unique_min should be unnecessary as vote MUST be Formal
    
    if allocation_set != df.columns.unique().tolist(): # if allocation_set not just first prefs, handle duplicates by adding index to list
        ATL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & ATL_non_unique_min
        BTL_duplicates_series = allocated_votes.loc[:,'Vote'].isna() & ~ATL_duplicates_series & BTL_non_unique_min # duplicate but not in ATL

        # collect indices 
        duplicates = ATL_duplicates_series+BTL_duplicates_series
        duplicate_indices = duplicates.loc[duplicates].index # return indices where duplicates == True
    else:
        duplicate_indices = [] # no duplicated 1st prefs

    #print("done", time.time() - start)

    return allocated_votes, duplicate_indices # duplicate_indices are index object!

def allocate_votes_duplicates(df, allocation_set):
        
        start_of_BTL_index = next(i for i, col in enumerate(df.columns) if df.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated
        ATL_dup = df.iloc[:,:start_of_BTL_index]
        BTL_dup = df.iloc[:,start_of_BTL_index:]

        # ATL preferences
        ATL_dup_preferences = ATL_dup[allocation_set]     # Filter the row to only include the candidates of interest
        duplicate_votes_series = ATL_dup_preferences.apply(lambda row: list(row[row == row.min()].index), axis = 1)

        # If no vote is allocated using ATL ('Vote' is nan), use BTL
        BTL_dup_preferences = BTL_dup[allocation_set].apply(lambda row: list(row[row == row.min()].index), axis = 1) # selects lowest preferences of combined BTL groups
        
        #empty_mask = duplicate_votes_series.apply(lambda x: len(x) == 0)  #Find empty lists, replace them with BTL_dup_preferences (effectively using a mask)
        #duplicate_votes_series.loc[empty_mask] = BTL_dup_preferences[empty_mask].values
        duplicate_votes_series.loc[duplicate_votes_series.apply(lambda x: len(x) == 0)] = BTL_dup_preferences

        return duplicate_votes_series

def allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict, general_party_df, Senate_party_abvs_dict):

    # produce df or dictionary of dfs with concatenated ATL&uniqueBTL and the First Preference vote in the last column

    Senate_party_abvs_dict = {}

    for div in Formal_prefs_dict.keys():
        #import pdb;pdb.set_trace()
        formal_prefs_full = Formal_prefs_dict[div]
        formal_prefs = formal_prefs_full.iloc[:, START_OF_PREFS:]
        formal_prefs.columns = formal_prefs.columns.str.split(':').str[0] # keep only party grouping as key
        start_of_BTL_index = next(i for i, col in enumerate(formal_prefs.columns) if formal_prefs.columns[:i].tolist().count(col) == 1) # locates first instance of column name count repeated



        first_prefs_set = formal_prefs.columns.unique().tolist() # all cols including 'UG'
        # fix BTL into single group and concatenate
        ATL = formal_prefs.iloc[:,:start_of_BTL_index]
        BTL = formal_prefs.iloc[:,start_of_BTL_index:]
        BTL = BTL.apply(pd.to_numeric)
        BTL = BTL.T.groupby(BTL.columns).min().T
        formal_prefs_by_group = pd.concat([ATL, BTL], axis=1)

        # allocate first preferences using formal_prefs
        first_pref_allocated_votes = allocate_votes(formal_prefs_by_group, first_prefs_set)[0]
        formal_prefs_first_prefs = pd.concat([formal_prefs_by_group, first_pref_allocated_votes], axis=1) # add allocated first_pref vote to formal_prefs_by_group df
        Formal_prefs_dict[div] = pd.concat([Formal_prefs_dict[div].iloc[:,:START_OF_PREFS], formal_prefs_first_prefs], axis=1)
        #print(Formal_prefs_dict[div])

    return Formal_prefs_dict



list1 = []
list2 = []
incumbency_advantage_dict3 = [] # div: [list of PartyAbs]
list4 = []



# get correpondence between Booth name and pp_id for given year
Polling_Places_df = pd.read_csv(f"{data_year}GeneralPollingPlaces.csv", index_col = None, skiprows = 1)
Polling_Places_df = Polling_Places_df.iloc[:,2:6].rename(columns={'DivisionNm': 'div_nm','PollingPlaceID': 'pp_id','PollingPlaceNm':'pp_nm'})
Polling_Places_df = Polling_Places_df.loc[Polling_Places_df['PollingPlaceTypeID'].isin([1,5]),].drop('PollingPlaceTypeID', axis=1)

for div in Polling_Places_df['div_nm'].unique().tolist():
    Other_row = pd.DataFrame({"div_nm": [div],"pp_id":[0],"pp_nm":["Other"]})
    Polling_Places_df = pd.concat([Polling_Places_df,Other_row], ignore_index=True)

Booth_name_pp_id = Polling_Places_df


# First_Prefs_by_PP_Complete = pd.read_csv(f"{data_year}FirstPrefsByPPComplete.csv", index_col = None)
# Booth_name_pp_id = First_Prefs_by_PP_Complete.iloc[:,:3].drop_duplicates()



def allocate_formal_preferences_to_allocation_set(Formal_prefs_div, allocation_set, by_pp_id = False, as_percent = True):

    Final_allocated_votes = pd.DataFrame(index=Formal_prefs_div.index, columns=allocation_set, data = 0.0) # df of allocated votes for each candidate, should preserve order of df
    Final_allocated_votes["First_Preferences"] = Formal_prefs_div["Vote"]

    # building groups based on first preference vote, either allocate vote directly to allocation_set if one of them, else allocate by group among later preferences

    for party, formal_subsection in Formal_prefs_div.iloc[:,START_OF_PREFS:].groupby("Vote"): # ignore first 2 rows in calculation (div/pp)
        
        if party in allocation_set:
            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party, party] = 1.0 # put in a 1 into the party column while preserving index

        else:
            allocated_votes_subsection, duplicate_indices_subsection = allocate_votes(formal_subsection, allocation_set)
            Subsection_final_votes = Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party] # just working with this subsection

            # Add allocation preferences where clear
            mask = pd.get_dummies(allocated_votes_subsection.loc[allocated_votes_subsection["Vote"].notna(), "Vote"])
            mask = mask.reindex(Subsection_final_votes.index, fill_value=0)     # Align the mask with the indices of Subsection_final_votes
            Subsection_final_votes.loc[:, mask.columns] = mask.astype(float)         # Update Subsection_final_votes with the mask

            # Add duplicate preferences 
            duplicate_for_party_df = formal_subsection[formal_subsection.index.isin(duplicate_indices_subsection)].copy()

            # iteratively add duplicate votes proportionate to # of cnadidates duplicated
            duplicate_for_party_df["Vote"] = allocate_votes_duplicates(duplicate_for_party_df, allocation_set) # get series of candidates for each duplicate votes
            
            Subsection_final_votes = Subsection_final_votes.astype({col: "float64" for col in Subsection_final_votes.columns[:-1]})

            for row in duplicate_for_party_df.index:
                duplicate_vote_list = duplicate_for_party_df.loc[duplicate_for_party_df.index == row,"Vote"].iloc[0] # iloc makes it a list
                for vote in duplicate_vote_list:
                    Subsection_final_votes.loc[Subsection_final_votes.index==row, vote] = 1/len(duplicate_vote_list)
            

            # handle remainingg nan values - assign votes proportional to how rest of their subsection voted
            Subsection_final_votes = Subsection_final_votes.drop(columns=['First_Preferences'])
            Party_preferences_proportions = Subsection_final_votes.sum() / np.sum(Subsection_final_votes.sum()) # row of proportions
            mask = allocated_votes_subsection["Vote"].isna() & ~allocated_votes_subsection.index.isin(duplicate_indices_subsection)
            Subsection_final_votes.loc[mask] = pd.DataFrame([Party_preferences_proportions.values] * sum(mask), index=Subsection_final_votes.index[mask], columns=Subsection_final_votes.columns) # changed from mask.sum()

            #Final_allocated_votes.iloc[:, :-1] = Final_allocated_votes.iloc[:, :-1].astype(float)
            if Final_allocated_votes.columns.duplicated().any():
                import pdb;pdb.set_trace()
                print("Warning: Duplicate column names found!")
                Final_allocated_votes = Final_allocated_votes.loc[:, ~Final_allocated_votes.columns.duplicated()]  # Drop duplicates

            # Select numeric columns correctly
            numeric_columns = Final_allocated_votes.select_dtypes(include=['number']).columns  

            # Convert the selected numeric columns to float (ensuring proper data types)
            Final_allocated_votes[numeric_columns] = Final_allocated_votes[numeric_columns].astype(float)

            Final_allocated_votes.loc[Final_allocated_votes["First_Preferences"] == party,Final_allocated_votes.columns[:-1]] = Subsection_final_votes.values # fill out full table


    Final_allocated_votes_df = pd.concat([Formal_prefs_div.iloc[:,:START_OF_PREFS], Final_allocated_votes], axis=1).drop(columns = "First_Preferences") # return 1st 3 cols & remove last


    # This is where to return in the pp_id column 
    if not by_pp_id:
        Final_allocated_votes_aggregated_df = Final_allocated_votes_df.drop(columns = ["pp_nm"]).groupby(["div_nm"], as_index=False).sum() # group all together
        #import pdb;pdb.set_trace()
    
    else:
        Final_allocated_votes_aggregated_df = Final_allocated_votes_df.groupby(["div_nm", "pp_nm"], as_index=False).sum()

        # GROUP THE STARTSWITH ABSENT,PREPOLL,POSTAL,PROVISIONAL,EAV,REMOTEMT,SPECIALMT,OTHERMT TOGETHER WITH PP_ID 0, THE REST MERGE WITH PP_IDS
        Other_booth_type_prefixes = ['Remote Mobile', 'Other Mobile','Special Hospital','EAV','ABSENT','PROVISIONAL','PRE_POLL','POSTAL']

        Final_allocated_votes_aggregated_df.loc[:,"pp_nm"] = Final_allocated_votes_aggregated_df.loc[:,"pp_nm"].apply(lambda x: 'Other' if any(x.startswith(prefix) for prefix in Other_booth_type_prefixes) else x)
        Final_allocated_votes_aggregated_df = Final_allocated_votes_aggregated_df.groupby(["div_nm", "pp_nm"], as_index=False).sum() # group again

        # switch pp_nm to pp_id
        Final_allocated_votes_aggregated_df = pd.merge(Final_allocated_votes_aggregated_df, Booth_name_pp_id, on = ['div_nm','pp_nm'], how='left')
        Final_allocated_votes_aggregated_df.loc[:,'pp_nm'] = Final_allocated_votes_aggregated_df.loc[:,'pp_id']

        Final_allocated_votes_aggregated_df.drop(columns=['pp_id'], inplace=True)
        Final_allocated_votes_aggregated_df.rename(columns={"pp_nm":"pp_id"}, inplace=True)

    if as_percent:
        Final_allocated_votes_aggregated_df.iloc[:, 1+by_pp_id:] = Final_allocated_votes_aggregated_df.iloc[:, 1+by_pp_id:].div(Final_allocated_votes_aggregated_df.drop(columns=['div_nm','pp_id'], errors='ignore').sum(axis=1), axis=0)


    return Final_allocated_votes_aggregated_df



def allocate_Formal_prefs_by_1234(Formal_prefs_dict, Senate_party_abvs_dict, application_dict, by_pp_id = False, as_percent = True):
    #### application_dict is one of 1,2,3 (incumbency_advantage),4

    Final_allocated_pcts_aggregated_dict = {}

    for div in application_dict.keys(): # only apply to relevant divisions

        # convert allocation_abvs into Senate Group letters
        allocation_abvs_list = application_dict[div] # list of PartyAb to allocate to

        allocation_set = []
        # Goal is to preserve order of allocation_abvs_list in allocation_set
        for party in allocation_abvs_list:  # Iterate through allocation_abvs_list directly

            if party in Senate_party_abvs_dict[div]: 
                i = Senate_party_abvs_dict[div].index(party) # Find the index of the party in this div and use it to get the corresponding Senate Group name
                allocation_set.append(Formal_prefs_dict[div].columns[START_OF_PREFS:START_OF_PREFS+len(Senate_party_abvs_dict[div])][i])  # Append the corresponding group 'letter'
       
        # allocate to allocation_set (already converted to percentages!)
        Final_allocated_pcts_aggregated_dict[div] = allocate_formal_preferences_to_allocation_set(Formal_prefs_dict[div], allocation_set, by_pp_id = False, as_percent = True)

        #Final_allocated_pcts_aggregated_dict[div].iloc[:, 1+by_pp_id:] = Final_allocated_pcts_aggregated_dict[div].iloc[:, 1+by_pp_id:].div(Final_allocated_pcts_aggregated_dict[div].drop(columns=['div_nm','pp_id'], errors='ignore').sum(axis=1), axis=0)

    return Final_allocated_pcts_aggregated_dict


















DOP_By_PP_2022 = pd.read_csv("2022DOP_By_PP_full.csv", index_col=None)

######## TO DO: Bring in Candidate_Pairs into here, adjusted for DOP_By_PP
### Change order within DOP_By_PP to match rather arbitrary m-c1-c2 ordering (consistency)
### Apply the function to existing counts, returning final counts!



def convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, div):
    ### convert allocation_abvs into Senate Group letters
    import pdb;pdb.set_trace()

    allocation_set = []
    # Goal is to preserve order of allocation_abvs_list in allocation_set
    for party in allocation_abvs_list:  # Iterate through allocation_abvs_list directly
        if party in Senate_party_abvs_dict[div]: 
            i = Senate_party_abvs_dict[div].index(party) # Find the index of the party in this div and use it to get the corresponding Senate Group name
            allocation_set.append(Formal_prefs_dict[div].columns[START_OF_PREFS:START_OF_PREFS+len(Senate_party_abvs_dict[div])][i])  # Append the corresponding group 'letter'
    return allocation_set
    
def allocate_Formal_prefs_Redistribution_change(Formal_prefs_dict, Senate_party_abvs_dict, Redistribution_pair_c1_c2_lists, DOP_By_PP_Pref_Percent_wide_dict, DOP_By_PP_Expand_wide_dict):
    #### TO BE MADE REDUNDANT - WILL RUN ALL REDISTRIBUTION CHANGES IS FPA FILE, ONE BY ONE INSTEAD OF AS A HEAP

    Final_allocated_pcts_aggregated_dict = {}

    div_pair_keys = list(Redistribution_pair_c1_c2_lists.iloc[:, :2].itertuples(index=False, name=None))
    import pdb;pdb.set_trace()


    for index, row in Redistribution_pair_c1_c2_lists.iterrows(): # only apply to relevant division pairs

        mlist = ast.literal_eval(row['m_list'])
        c1list = len(ast.literal_eval(row['c1_list']))
        c2list = len(ast.literal_eval(row['c2_list']))
        m,c1,c2 = len(mlist),len(c1list),len(c2list)

        c1_m_c2_dict = {}

        import pdb;pdb.set_trace()

        giver_div = row[0]

        # determine full c1 set to start with - redistribution_votes
        wide_df = DOP_By_PP_Pref_Percent_wide_dict[giver_div]
        Final_Count_Number = wide_df.iloc[-1,1]
        redistribution_votes = wide_df.loc[wide_df['CountNumber'] ==  (Final_Count_Number+2) - c1,1:].drop('CountNumber', axis = 1)



        for idx, (col_name, value) in enumerate(row.iloc[2:].items()): # iterate over the 3 lists of PartyAb
            allocation_abvs_list = ast.literal_eval(value) #(= row[col_name]) list of PartyAb to allocate to
            
            import pdb;pdb.set_trace()
            if idx>=1 and value == row.iloc[2:].iloc[idx-1]: # c1 or c2 is same as m --> don't need to repeat
                c1_m_c2_dict[col_name] = c1_m_c2_dict[row.columns[2+idx-1]] # copies previous column
            else:
                allocation_set = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, giver_div)

                # allocate to allocation_set and convert to percentages - BE CAREFUL TO DO IT PER ROW AND NOT TOTALLY
                Final_allocated_pcts_aggregated_dict[giver_div] = allocate_formal_preferences_to_allocation_set(Formal_prefs_dict[giver_div], allocation_set)
                df = Final_allocated_pcts_aggregated_dict[giver_div]
                import pdb;pdb.set_trace()
                df.iloc[:, 2:] = df.iloc[:, 2:].div(df.drop(columns=['div_nm','pp_id']).sum(axis=1), axis=0) # percentages
                c1_m_c2_dict[col_name] = df.iloc[:,1:].set_index("pp_id")

        # do all the fancy calculations now
        import pdb;pdb.set_trace()
        # 1-> 2. Percentage transfer

        # first m candidates the same, remaining c1 - 

        if c1 > m:
            sum_c1_extras = c1_m_c2_dict['c1_list'].iloc[:,m:].sum(axis=1) # sum values in row for extra c1 candidates (first 2 rows are info) - PPID USED AS INDEX NOW SO ALL NUMERIC
            c1_m_c2_dict['transfer_percent'] = (c1_m_c2_dict['m_list'] - c1_m_c2_dict['c1_list'].iloc[:,:m])/sum_c1_extras # must be positive
        else:
            #zero_df = c1_m_c2_dict['m_list'].iloc[:,2:].loc[:, c1_m_c2_dict['m_list'].iloc[:,2:].columns] = 0
            #c1_m_c2_dict['transfer_percent'] =zero_df
            c1_m_c2_dict['transfer_percent'] = np.nan # 
        import pdb;pdb.set_trace()

        if c2 > m:
            # separately, save proportion donated by m parties, and proportions of donation total recieved by extra c2 parties
            c1_m_c2_dict['donation_proportion'] = 1 - (c1_m_c2_dict['c2_list'].iloc[:,:m] / c1_m_c2_dict['m_list'].replace(0, float('nan'))).fillna(0) # avoid division by 0
            sum_c2_etras = c1_m_c2_dict['c2_list'].iloc[:,m:].sum(axis=1)
            c1_m_c2_dict['receiving_proportion'] = c1_m_c2_dict['c2_list'].iloc[:,m:].div(sum_c2_etras, axis=0) # of c2 extra candidates, get proportion donated to each
        else: 
            c1_m_c2_dict['donation_proportion'] = 0
            c1_m_c2_dict['receiving_proportion'] = 0

        # align together redistribution_votes, c1_m_c2_dict['transfer_percent'], c1_m_c2_dict['donation_proportion'],  c1_m_c2_dict['receiving_proportion']

        # sort rows by pp_id, apply to columns
        extra_c1_parties = list(set(c1) - set(m))
        total_percentages_to_transfer = redistribution_votes[extra_c1_parties].sum(axis=1)
        transfers_by_PP = c1_m_c2_dict['transfer_percent'].mul(total_percentages_to_transfer, axis=0) 
        
        redistribution_votes = redistribution_votes[mlist] + transfers_by_PP # add new transferred values to original redistribution votes

        total_donation_percentages = redistribution_votes.multiply(c1_m_c2_dict['donation_proportion']).sum(axis=1) # WILL COLUMNS PARTYABS ALIGN???
        receiving_percentages = c1_m_c2_dict['receiving_proportion'].mul(total_donation_percentages, axis=0)
        redistribution_votes = redistribution_votes.multiply(1 - c1_m_c2_dict['donation_proportion']) # proportions remaining for the m parties
        redistribution_votes = pd.concat([redistribution_votes, receiving_percentages], axis = 1)
            

        # 1. Combine all state division DOPByPP together for each redistribution state
        # 2. Select wide format percentages corresponding to c1 candidates
        # 3. c1-> m transition

    return df

  
def allocate_Formal_prefs_complex(Formal_prefs_dict, Senate_party_abvs_dict, reduced_votes_by_PP, complex_pair_row):
    #### want to produce a df that 

    c1_m_c2_dict = {}

    import pdb;pdb.set_trace()


    row = pd.DataFrame([complex_pair_row])

    mlist = row['m_list'][0]
    c1list = row['c1_list'][0]
    c2list = row['c2_list'][0]
    m,c1,c2 = len(mlist),len(c1list),len(c2list)
    giver_div = row['old_div'][0]

    redistribution_votes = reduced_votes_by_PP     # full c1 set to start with - redistribution_votes

    import pdb;pdb.set_trace()


    # iterate over the 3 lists of PartyAb, apply Formal Preferences, store results in c1_m_c2_dict
    for idx, (col_name, value) in enumerate(row.iloc[:,2:].items()): 
        allocation_abvs_list = value[0] #(= row[col_name]) list of PartyAb to allocate to
        
        import pdb;pdb.set_trace()
        if idx>=1 and value[0] == row.iloc[:,2:].iloc[0,idx-1]: # c1 or c2 is same as m --> don't need to repeat
            c1_m_c2_dict[col_name] = c1_m_c2_dict[row.columns[2+idx-1]] # copies previous column
        else:
            allocation_set = convert_partyab_to_senate_group_names(allocation_abvs_list, Formal_prefs_dict, Senate_party_abvs_dict, giver_div)

            # allocate to allocation_set and convert to percentages - BE CAREFUL TO DO IT PER ROW AND NOT TOTALLY
            Final_allocated_pcts_aggregated = allocate_formal_preferences_to_allocation_set(Formal_prefs_dict[giver_div], allocation_set, by_pp_id = True, as_percent = True)
            #df = Final_allocated_pcts_aggregated
            import pdb;pdb.set_trace()
            #df.iloc[:, 2:] = df.iloc[:, 2:].div(df.drop(columns=['div_nm','pp_id']).sum(axis=1), axis=0) # percentages
            c1_m_c2_dict[col_name] = Final_allocated_pcts_aggregated.iloc[:,1:].set_index("pp_id").sort_index()


    # do all the fancy calculations now
    # 1-> 2. Percentage transfer     # first m candidates the same, remaining c1 - 
    import pdb;pdb.set_trace()

    # align together redistribution_votes, c1_m_c2_dict['transfer_percent'], c1_m_c2_dict['donation_proportion'],  c1_m_c2_dict['receiving_proportion']

    if c1 > m:
        sum_c1_extras = c1_m_c2_dict['c1_list'].iloc[:,m:].sum(axis=1) # sum values in row for extra c1 candidates (first 2 rows are info) - PPID USED AS INDEX NOW SO ALL NUMERIC
        c1_m_c2_dict['transfer_percent'] = (c1_m_c2_dict['m_list'] - c1_m_c2_dict['c1_list'].iloc[:,:m])/sum_c1_extras # must be positive
        c1_m_c2_dict['transfer_percent'].columns = mlist

        extra_c1_parties = list(set(c1list) - set(mlist))
        total_percentages_to_transfer = redistribution_votes[extra_c1_parties].sum(axis=1)
        transfers_by_PP = c1_m_c2_dict['transfer_percent'].mul(total_percentages_to_transfer, axis=0) 
        
        redistribution_votes = redistribution_votes[mlist] + transfers_by_PP # add new transferred values to original redistribution votes
 
    import pdb;pdb.set_trace()
    if c2 > m:
        # separately, save proportion donated by m parties, and proportions of donation total recieved by extra c2 parties
        c1_m_c2_dict['donation_proportion'] = 1 - (c1_m_c2_dict['c2_list'].iloc[:,:m] / c1_m_c2_dict['m_list'].replace(0, float('nan'))).fillna(0) # avoid division by 0
        sum_c2_etras = c1_m_c2_dict['c2_list'].iloc[:,m:].sum(axis=1)
        c1_m_c2_dict['receiving_proportion'] = c1_m_c2_dict['c2_list'].iloc[:,m:].div(sum_c2_etras, axis=0) # of c2 extra candidates, get proportion donated to each
        c1_m_c2_dict['donation_proportion'].columns = mlist
        c1_m_c2_dict['receiving_proportion'].columns = c2list[m:]

        total_donation_percentages = redistribution_votes.multiply(c1_m_c2_dict['donation_proportion']).sum(axis=1) # WILL COLUMNS PARTYABS ALIGN???
        receiving_percentages = c1_m_c2_dict['receiving_proportion'].mul(total_donation_percentages, axis=0)
        redistribution_votes = redistribution_votes.multiply(1 - c1_m_c2_dict['donation_proportion']) # proportions remaining for the m parties
        redistribution_votes = pd.concat([redistribution_votes, receiving_percentages], axis = 1)
    
    import pdb;pdb.set_trace()

    return redistribution_votes


def reduce_candidates_to_set_size(div, DOP_By_PP_Pref_Percent_wide_dict, reduced_c_size):
    wide_df1 = DOP_By_PP_Pref_Percent_wide_dict[div]

    import pdb;pdb.set_trace()

    Final_Count_Number = wide_df1.iloc[-1,1] # last index of CountNumber (2nd column)
    reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == (Final_Count_Number+2)-reduced_c_size,].set_index('pp_id').iloc[:,1:]  # the correct count number!
    import pdb;pdb.set_trace()

    zero_columns = reduced_votes_by_PP.iloc[0] == 0
    zero_columns = zero_columns[zero_columns].index.tolist()

    reduced_votes_by_PP.loc[:,zero_columns] = np.nan # convert other cols to nan

    return reduced_votes_by_PP

def expand_candidates_to_set_size(div, reduced_votes_by_PP, DOP_By_PP_Expand_wide_dict, c_size, expanded_c_size):
    
    wide_df_expand = DOP_By_PP_Expand_wide_dict[div]

    expanded_votes = reduced_votes_by_PP

    Final_Count_Number = wide_df_expand.iloc[-1,1]
    if expanded_c_size == 'full': # specify the full size if previously unknown what the full size is
        expanded_c_size = Final_Count_Number + 2

    start_range = 1 + (Final_Count_Number + 2) - expanded_c_size # 1 + total num of candidates - c1
    end_range = (Final_Count_Number+2) - c_size

    import pdb;pdb.set_trace()

    for i in reversed(range(start_range, end_range)): #(i.e. from count 4 to count 1, where the difference 4-1=c1-m)
        expand_div = wide_df_expand.loc[wide_df_expand['CountNumber'] == i,].drop('CountNumber', axis = 1)
    
        to_expand_party = expand_div.columns[(expand_div.iloc[0] == -1)].tolist()[0] # party to expand to will have -1 as value
        import pdb;pdb.set_trace()

        lost_votes = expanded_votes.multiply(expand_div.set_index("pp_id"))

        expanded_votes = expanded_votes.subtract(lost_votes)
        expanded_votes[to_expand_party] = lost_votes.sum(axis=1).values
        #import pdb;pdb.set_trace()

    return expanded_votes

def simple_redistribution(div1,div2,DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict,m,c1,c2):

    import pdb;pdb.set_trace()
    
    reduced_votes_by_PP = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, m)
    import pdb;pdb.set_trace()
    if m < c2:
        expanded_votes_by_PP = expand_candidates_to_set_size(div2, reduced_votes_by_PP, DOP_By_PP_Expand_wide_dict, m, c2)
    else:
        expanded_votes_by_PP = reduced_votes_by_PP
    
    return expanded_votes_by_PP

def complex_redistribution(div1,div2,DOP_By_PP_Pref_Percent_wide_dict, DOP_By_PP_Expand_wide_dict,complex_pair_row, c1_votes = None):
    ### returns votes reallocated to c2, using FPA
    import pdb;pdb.set_trace()

    if isinstance(c1_votes, pd.DataFrame): # already have reduced votes due to independent
        reduced_votes_by_PP = c1_votes 

    else: # need to construct reduced votes to c1 directly (no independent)
        m = len(complex_pair_row['m_list'])
        c1 = len(complex_pair_row['c1_list'])
        c2 = len(complex_pair_row['c2_list'])

        reduced_votes_by_PP = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1) # want to reduce div1 to c1 candidates

    import pdb;pdb.set_trace()

    # apply First Preferences allocation
    allocated_votes_to_c2 = allocate_Formal_prefs_complex(Formal_prefs_dict, Senate_party_abvs_dict, reduced_votes_by_PP, complex_pair_row)
    # Use FP to get senate versions of A) reduced_votes = c1 B) common votes = m C) expanded votes = c2
    # calculate the % shifts in both A>B,B>C
    # apply to reduced_votes --> c2


    import pdb;pdb.set_trace()
    #expanded_c_size = DOP_By_PP_Pref_Percent_wide_dict[div2].shape[1] - 2 #MAKE SURE TO CHECK THIS!!!!!!!!!!!!!!!!!!!!
    #redistributed_votes = expand_candidates_to_set_size(div2, reduced_votes_by_PP, DOP_By_PP_Expand_wide_dict, c2, expanded_c_size)

    return allocated_votes_to_c2

def independent_to_c1(div1, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO):
    ### adjusts votes of c1 candidate for incumbency advantage
    # 1. get top 5 cands, 2. reverse incumb.advantage, 3. expand to full vote

    wide_df1 = DOP_By_PP_Pref_Percent_wide_dict[div1]
    wide_df_expand = DOP_By_PP_Expand_wide_dict[div1] # same div here!

    # Reduce to Final 5
    Final_Count_Number = wide_df1.iloc[-1,1] # last index of CountNumber (2nd column)
    reduced_votes_by_PP = wide_df1.loc[wide_df1['CountNumber'] == Final_Count_Number - (FINAL_CANDIDATE_NO-2),] # when 5 candidates remaining
    reduced_votes_by_PP = reduced_votes_by_PP.drop(columns=[reduced_votes_by_PP.columns[1]]) # remove CountNumber col!
    reduced_votes_by_PP.iloc[:, 1:] = reduced_votes_by_PP.iloc[:, 1:].where(reduced_votes_by_PP.iloc[:, 1:] > 0)
    reduced_votes_by_PP = reduced_votes_by_PP.sort_index()

    top_5_columns = reduced_votes_by_PP.columns[~reduced_votes_by_PP.iloc[0].isna()].tolist()[1:]

    # Reverse incumbency advantage: e.g. -4 to inc, +1 to each non-inc 
    Incumbent_in_div = Incumbency_by_div.loc[Incumbency_by_div['div_nm']==div1,'PartyAb'].tolist()

    for party in Incumbent_in_div:
        reduced_votes_by_PP[top_5_columns] += NONINCUMBENT_DISADVANTAGE
        reduced_votes_by_PP[party] -= (INCUMBENT_ADVANTAGE + NONINCUMBENT_DISADVANTAGE)


    # expand to c1
    expanded_votes = reduced_votes_by_PP

    start_range = 1 + (Final_Count_Number + 2) - c1 # 1 + total num of candidates - c1
    end_range = 1 + Final_Count_Number - (FINAL_CANDIDATE_NO-2)

    for i in reversed(range(start_range, end_range)): #(i.e. from count 4 to count 1, where the difference 4-1=c1-m)
        expand_div = wide_df_expand.loc[wide_df_expand['CountNumber'] == i,].drop('CountNumber', axis = 1)
    
        to_expand_party = expand_div.columns[(expand_div.iloc[0] == -1)].tolist()[0] # party to expand to will have -1 as value

        lost_votes = expanded_votes.set_index("pp_id").multiply(expand_div.set_index("pp_id")).reset_index()

        expanded_votes = expanded_votes.set_index("pp_id").subtract(lost_votes.set_index("pp_id")).reset_index()
        expanded_votes[to_expand_party] = lost_votes.iloc[:,1:].sum(axis=1).values
        #import pdb;pdb.set_trace()

    return expanded_votes


def independent_redistribution_reduce(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO, list_div1_FP):

    house_votes = independent_to_c1(div1, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO).set_index('pp_id')
    
    senate_allocation_list = list_div1_FP # list of parties excluding non-senate parties
    allocation_set = convert_partyab_to_senate_group_names(senate_allocation_list, Formal_prefs_dict, Senate_party_abvs_dict, div1)

    import pdb;pdb.set_trace()
    senate_votes = allocate_formal_preferences_to_allocation_set(Formal_prefs_dict[div1], allocation_set, by_pp_id = True, as_percent = True).set_index('pp_id').sort_index().iloc[:,1:] # only data columns
    senate_votes = senate_votes.div(senate_votes.sum(axis=1), axis=0)*100
    
    senate_votes.columns = senate_allocation_list

    sum_c1_extras = 100 - house_votes.loc[:,list_div1_FP].sum(axis=1) # sum values in row for extra c1 candidates
    Senate_minus_IND_house = senate_votes - house_votes.loc[:,list_div1_FP]
    import pdb;pdb.set_trace()

    negative_difference_totals = Senate_minus_IND_house.apply(lambda row: row[row < 0].sum(), axis=1)
    import pdb;pdb.set_trace()

    Proportion_df = Senate_minus_IND_house.apply(lambda row: row.where(row < 0, row + (row / row[row > 0].sum())*(negative_difference_totals.loc[row.name])), axis=1) # temporarily houses proportions of positive % differences, multiplies by negative difference totals
    #Senate_minus_IND_house['negative_sum'] = Senate_minus_IND_house.apply(lambda row: row[row < 0].sum(), axis=1)
    Proportion_df[Proportion_df<0] = 0

    Transfer_percent = Proportion_df.div(sum_c1_extras, axis = 0) # must be positive
    import pdb;pdb.set_trace()

    # get actual c1 votes (for len(list_div1_FP) parties) and allocate transfers
    c1_votes = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1)
    non_senate_cands = [cand for cand in (set(c1_votes.columns) - set(list_div1_FP))]
    sum_non_senates = c1_votes.loc[:,c1_votes.columns.isin(non_senate_cands)].sum(axis=1)
    transferred_votes = Transfer_percent.mul(sum_non_senates, axis=0)
    reduced_votes = c1_votes.loc[:,~c1_votes.columns.isin(non_senate_cands)] + transferred_votes
    import pdb;pdb.set_trace()

    return reduced_votes



def independent_redistribution_reduce_expand(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO, list_div1_FP, votes_to_expand = 0):

    house_votes = independent_to_c1(div1, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO).set_index('pp_id')
    
    senate_allocation_list = list_div1_FP # list of parties excluding non-senate parties
    allocation_set = convert_partyab_to_senate_group_names(senate_allocation_list, Formal_prefs_dict, Senate_party_abvs_dict, div1)

    import pdb;pdb.set_trace()
    senate_votes = allocate_formal_preferences_to_allocation_set(Formal_prefs_dict[div1], allocation_set, by_pp_id = True, as_percent = True).set_index('pp_id').sort_index().iloc[:,1:] # only data columns
    senate_votes = senate_votes.div(senate_votes.sum(axis=1), axis=0)*100
    
    senate_votes.columns = senate_allocation_list

    # investigate where house vote > senate vote
    Senate_minus_IND_house = senate_votes - house_votes.loc[:,list_div1_FP]
    negative_difference_totals = Senate_minus_IND_house.apply(lambda row: row[row < 0].sum(), axis=1)

    if not votes_to_expand: # i.e. direction == 'Reduce'
        Proportion_df = Senate_minus_IND_house.apply(lambda row: row.where(row < 0, row + (row / row[row > 0].sum())*(negative_difference_totals.loc[row.name])), axis=1) # temporarily houses proportions of positive % differences, multiplies by negative difference totals
        Proportion_df[Proportion_df<0] = 0

        sum_c1_extras = 100 - house_votes.loc[:,list_div1_FP].sum(axis=1) # sum values in row for extra c1 candidates
        Transfer_percent = Proportion_df.div(sum_c1_extras, axis = 0) # must be positive
        import pdb;pdb.set_trace()

        # get actual c1 votes (for len(list_div1_FP) parties) and allocate transfers
        c1_votes = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1)
        non_senate_cands = [cand for cand in (set(c1_votes.columns) - set(list_div1_FP))]
        sum_non_senates = c1_votes.loc[:,c1_votes.columns.isin(non_senate_cands)].sum(axis=1)
        transferred_votes = Transfer_percent.mul(sum_non_senates, axis=0)
        redistribution_votes = c1_votes.loc[:,~c1_votes.columns.isin(non_senate_cands)] + transferred_votes

    else: # direction == 'Expand'!
        # This should ensure that house>senate cases are nullfied
        add_to_house_rebalancing = Senate_minus_IND_house.apply(lambda row: row.where(row < 0, row + (row / row[row > 0].sum())*(negative_difference_totals.loc[row.name]*-1)), axis=1)
        #add_to_house_rebalancing[add_to_house_rebalancing<0] = 0
        import pdb;pdb.set_trace()

        house_votes.loc[:,list_div1_FP] += add_to_house_rebalancing # now fully rebalanced

        # perform classic expand trick
        donation_proportion = 1 - house_votes / senate_votes
        #c1_m_c2_dict['donation_proportion'] = 1 - (c1_m_c2_dict['c2_list'].iloc[:,:m] / c1_m_c2_dict['m_list'].replace(0, float('nan'))).fillna(0) # avoid division by 0

        non_senate_cands = [cand for cand in (set(house_votes.columns) - set(list_div1_FP))]
        sum_non_senates = house_votes.loc[:,house_votes.columns.isin(non_senate_cands)].sum(axis=1)
        import pdb;pdb.set_trace()

        receiving_proportion = house_votes.loc[:,house_votes.columns.isin(non_senate_cands)].div(sum_non_senates, axis=0) # if multiple non-senates

        total_donation_percentages = votes_to_expand.multiply(donation_proportion).sum(axis=1) # WILL COLUMNS PARTYABS ALIGN???
        import pdb;pdb.set_trace()
        receiving_percentages = receiving_proportion.mul(total_donation_percentages, axis=0)
        redistribution_votes = redistribution_votes.multiply(1 - donation_proportion) # proportions remaining for the m parties
        redistribution_votes = pd.concat([redistribution_votes, receiving_percentages], axis = 1)

    import pdb;pdb.set_trace()

    return redistribution_votes



def find_c1_c2(elimination_list, common_set):
    ### finds length of subsection of list upon which all elements of common_set are seen
    
    seen = set()
    for c in range(len(elimination_list)):  # Iterate from the start
        if elimination_list[c] in common_set:
            seen.add(elimination_list[c])  # Track seen elements from the set
        if seen == common_set:  # Stop once all elements have appeared
            return c+1 # inedx+1  
        
    return 0/0 # should never happen!

def remove_non_senate_cands(list_div1_c1,list_div2_c2,Senate_parties):
    non_senate_parties_div1 = []

    for party in list_div1_c1:
        if party not in Senate_parties:
            non_senate_parties_div1.append(party)
    list_div1_FP = [x for x in list_div1_c1 if x not in non_senate_parties_div1]

    non_senate_parties_div2 = []
    for party in list_div2_c2:
        if party not in Senate_parties:
            non_senate_parties_div2.append(party)
    list_div2_FP = [x for x in list_div2_c2 if x not in non_senate_parties_div2]

    return list_div1_FP,list_div2_FP, non_senate_parties_div1, non_senate_parties_div2 # returns parties in elimination lists excluding those that are not in the senate, and then the remaining ones


def transform_to_raw_votes(redistributed_votes, giver_div):

    First_Prefs_By_PP_Complete = pd.read_csv(f"{data_year}FirstPrefsByPPComplete.csv", index_col = None)[['pp_id','div_nm','PartyAb','votes']]
    First_Prefs_By_PP_div = First_Prefs_By_PP_Complete.loc[First_Prefs_By_PP_Complete['div_nm'] == giver_div,].drop(columns = 'div_nm', axis = 1)

    INFORMAL_df = First_Prefs_By_PP_div[First_Prefs_By_PP_div['PartyAb'] == 'INFORMAL'].rename(columns = {'votes':'INFORMAL'})
    FORMAL_df = First_Prefs_By_PP_div[First_Prefs_By_PP_div['PartyAb'] != 'INFORMAL']

    import pdb;pdb.set_trace()

    grouped_FORMAL = FORMAL_df.groupby('pp_id', as_index=False).agg({'votes': 'sum'}).drop(columns = ['PartyAb'], axis = 1)
    #grouped_FORMAL.loc[:,'PartyAb'] = 'FORMAL'

    grouped_FORMAL = grouped_FORMAL.set_index('pp_id').sort_index()

    # scale redistributed_votes by grouped_FORMAL
    import pdb;pdb.set_trace()
    redistributed_votes_raw = (redistributed_votes_raw / 100).mul(grouped_FORMAL, axis=0) 
    import pdb;pdb.set_trace()

    # adjust to get integer values for votes
    redistributed_votes_rounded = redistributed_votes_raw.round().astype(int)
    redistributed_votes_sum = redistributed_votes_rounded.sum(axis=1)
    adjustment = grouped_FORMAL - redistributed_votes_sum

    for i in range(len(redistributed_votes_raw)):
        if adjustment[i] != 0:
            fractional_part = redistributed_votes_raw.iloc[i] - np.floor(redistributed_votes_raw.iloc[i])
            order = np.argsort(-fractional_part)  # Sort descending to find largest fractions
            for idx in order[:abs(adjustment[i])]:  # Distribute adjustments
                redistributed_votes_rounded.iloc[i, idx] += np.sign(adjustment[i])
            # Ensure sum is correct after adjustment
            assert redistributed_votes_rounded.iloc[i].sum() == redistributed_votes_rounded.round().astype(int)[i]

    redistributed_votes_raw_plus_informal = pd.concat([redistributed_votes_rounded, INFORMAL_df], axis=1)

    return redistributed_votes_raw_plus_informal



def full_redistribution_candidate_change(Formal_prefs_dict, Elimination_order_dict, Senate_parties_by_div, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, Incumbency_by_div, new_seats_list):
    # input is df with columns corresponding to giver division and receiver division, respectively. Returns the candidate sets of size c1 and c2 for those requiring complex redistr.
    # If independent is involved, this should be dealt with prior to this function.

    # initialise df to input into Formal Preferences Allocation
    Redistribution_pairs_df = pd.read_csv("RedistributionPairs2024.csv", index_col = None)

    import pdb;pdb.set_trace()
    columns_list = Redistribution_pairs_df.columns.tolist()
    columns_list.extend(['c1_list','c2_list'])
    Redistribution_pair_complex_c1_c2_lists = pd.DataFrame(columns=Redistribution_pairs_df.columns.tolist())

    redistribution_divs_set = set(Redistribution_pairs_df.iloc[1:,0])|set(Redistribution_pairs_df.iloc[1:,:1]) - {'old_div'} # all relevant divisions
    simplerd, simpleindrd, complexrd = 0,0, 0

    # check if double LP/NP contest
    LPNP_double_divs = []
    for div in redistribution_divs_set:
        if Elimination_order_dict[div].count('COAL') == 2:
            LPNP_double_divs.append(div)



    for i in range(Redistribution_pairs_df.shape[0]):
        div1, div2 = Redistribution_pairs_df.iloc[i].tolist() # get giver and receiver for pair i

        # perform different calculations if it double divs are involved
        if (div1 in LPNP_double_divs) or (div2 in LPNP_double_divs):
            print("doubledivs", div1, div2)


        # get elimination orders from last election, adjusting if this is new seat
        if div2 in new_seats_list:
            print("new seat", div1,div2)
            list_div2 = ['LP','ALP','GRN','UAPP','ON'] ################################### TO DO: generalise to all common parties accounting for COAL or NP or LP for specific election
            list_div1 = Elimination_order_dict[div1]
        else:
            list_div1, list_div2 = Elimination_order_dict[div1], Elimination_order_dict[div2] # get elimination orders

        set1,set2 = set(list_div1), set(list_div2)

        common = set1 & set2
        m = len(common)

        # find common candidates, see if simple/complex
        if set(list_div1[:m]) == set(list_div2[:m]):

            # SIMPLE REDISTRIBUTION
            import pdb;pdb.set_trace()
            c1 = list_div1
            c2 = list_div2

            redistributed_votes = simple_redistribution(div1,div2,DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict,m,c1,c2)

            redistribution_party_type = "simple"
            print("simple", div1,div2)
            simplerd += 1

            
        else:
            import pdb;pdb.set_trace()

            c1 = find_c1_c2(list_div1, common) # length of relevant subset of list_div1
            c2 = find_c1_c2(list_div2, common) # length of relevant subset of list_div2

            Senate_parties = Senate_parties_by_div.loc[Senate_parties_by_div['div_nm'] == div1,"PartyAbList"].iloc[0] # both div1 and div2 are in the same state so are identical
            list_div1_FP,list_div2_FP, non_senate_parties_div1, non_senate_parties_div2 = remove_non_senate_cands(list_div1[:c1],list_div2[:c2],Senate_parties)

            newset1,newset2 = set(list_div1_FP), set(list_div2_FP)
            newcommon = newset1 & newset2
            m = len(newcommon)

            if set(list_div1_FP[:m]) == set(list_div2_FP[:m]): 

                # SIMPLE REDISTRIBUTION WITH INDEPENDENTS/NON-SENATE

                if non_senate_parties_div1: # Must reduce c1 further to m
                    reduced_votes = independent_redistribution_reduce_expand(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO, list_div1_FP)
                else:
                    reduced_votes = reduce_candidates_to_set_size(div1, DOP_By_PP_Pref_Percent_wide_dict, c1) # reduce initial to c1 == m

                if non_senate_parties_div2: # Now, expand from m to c2, or directly to full
                    reduced_votes = independent_redistribution_reduce_expand(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c2,Incumbency_by_div, FINAL_CANDIDATE_NO, list_div2_FP, reduced_votes)

                redistributed_votes = expand_candidates_to_set_size(div2, reduced_votes, DOP_By_PP_Expand_wide_dict, c2, expanded_c_size = 'full') # expand from c2 to full (irrelevant whether there has been an independent update)

                redistribution_party_type = "simple non-senate"
                print("simple non-senate", div1,div2)
                simpleindrd += 1
            else:
                
                # COMPLEX REDISTRIBUTION (PERHAPS WITH NON-SENATE/IND)

                mlist = [x for x in newcommon]
                list1 = mlist + [x for x in list_div1_FP if x not in newcommon] # order with first m parties, then remaining ci-m parties
                list2 = mlist + [x for x in list_div2_FP if x not in newcommon]

                complex_pair_row = {'old_div': div1, 'new_div': div2, 'c1_list': list1, 'm_list': mlist,'c2_list': list2}
                import pdb;pdb.set_trace()

                # before complex redistirbution, we need to have the votes_by_PP of the c1- candidates.
                # If independent involved, start with c1- (i.e. perform independent redistribution reduce)
                # Otherwise, reduce to c1 inside complex_redistribution
                # Complex redistribution transforms from c1 to c2

                if non_senate_parties_div1: # Must reduce c1 further
                    c1_votes = independent_redistribution_reduce_expand(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c1,Incumbency_by_div, FINAL_CANDIDATE_NO, list_div1_FP)
                    c2_redistributed_votes = complex_redistribution(div1,div2,DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict,complex_pair_row, c1_votes = c1_votes)
                else:
                    c2_redistributed_votes = complex_redistribution(div1,div2,DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict,complex_pair_row, c1_votes = None)

                # expand from c2 onwards
                if non_senate_parties_div2:
                    c2_redistributed_votes = independent_redistribution_reduce_expand(div1, Formal_prefs_dict, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, c2,Incumbency_by_div, FINAL_CANDIDATE_NO, list_div2_FP, c2_redistributed_votes)

                redistributed_votes = expand_candidates_to_set_size(div2, c2_redistributed_votes, DOP_By_PP_Expand_wide_dict, c2, expanded_c_size = 'full') # expand from c2 to full (irrelevant whether there has been an independent update)

                print("complex or independent complex",div1,div2) 
                complexrd += 1   

        FirstPrefsByPPCompleteRedistributed = transform_to_raw_votes(redistributed_votes, div1)
        print(redistributed_votes)
        import pdb;pdb.set_trace()

         

    print(simplerd, simpleindrd, complexrd)
    import pdb;pdb.set_trace()

    

    return redistributed_votes






def whole_procedure(Formal_prefs_dict,general_party_df, Senate_party_abvs_dict, x = 5):
    Formal_prefs_dict = allocate_Formal_preferences_to_First_Preferences(Formal_prefs_dict, general_party_df, Senate_party_abvs_dict)

    print("done", time.time() - start)
    # make list of senate parties for check if they match house ones
    Senate_parties_by_div =  pd.DataFrame(list(Senate_party_abvs_dict.items()), columns=["div_nm", "PartyAbList"])
    Senate_parties_by_div.to_csv(f"{data_year}Senate_parties_by_div.csv", index=False) # currently off
    #import pdb;pdb.set_trace()

    Incumbent_advantage = 0
    candidate_change_redistribution = 1

    if Incumbent_advantage:

        Final_x_House_df = pd.read_csv(f"{data_year}Final_{x}_for_Incumbency.csv")
        #Final_x_House_df = Final_x_House_df.loc[Final_x_House_df['div_nm'].isin(list(Formal_prefs_dict.keys())),] # extra for testing!
        Final_x_House_dict = {name: list(Final_x_House_df.loc[Final_x_House_df['div_nm'] == name, 'PartyAb']) for name in Final_x_House_df['div_nm'].unique()}


        application_dict = Final_x_House_dict

        Final_allocated_pcts_aggregated_dict = allocate_Formal_prefs_by_1234(Formal_prefs_dict, Senate_party_abvs_dict, application_dict)

        Senate_votes = pd.concat([df.melt(id_vars=["div_nm"], value_vars=df.columns[1:], var_name="PartyAb", value_name="Senate_Pct") for df in Final_allocated_pcts_aggregated_dict.values()], ignore_index=True).reset_index(drop=True)

        import pdb;pdb.set_trace()
        Final_x_HS_df = pd.concat([Final_x_House_df, Senate_votes.drop(columns=['div_nm','PartyAb'])], axis=1)[["div_nm","PartyAb","is_incumbent","is_historic_incumbent","House_Pct","Senate_Pct"]]
        Final_x_HS_df.loc[:,"Senate_Pct"] = (pd.to_numeric(Final_x_HS_df.loc[:, "Senate_Pct"].astype(str), errors="coerce") * 100).round(2) # fixes issues with string values somehow????


        Final_x_HS_df.to_csv(f"{data_year}Final_{x}_HS_df.csv", index=False)

    if candidate_change_redistribution:
        #Redistribution_pair_c1_c2_lists = pd.read_csv("Redistribution_pair_c1_c2_lists2024.csv", index_col = None)
        # borrows from Candidate_Pairs
        Incumbency_by_div = pd.read_csv(f"{data_year}Incumbents.csv", index_col = None)
        SA1_By_PP_Votes_2022 = pd.read_csv("2022SA1_By_PP_Votes.csv", index_col=None)
        SA1s_with_votes = set(SA1_By_PP_Votes_2022.iloc[:,1])

        new_seats_list = ['Bullwinkel']

        Redistribution_SA1_changes_2024 = pd.read_csv("Redistribution_SA1_changes2024.csv", index_col=None)
        Redistribution_SA1_changes_2024 = Redistribution_SA1_changes_2024.loc[Redistribution_SA1_changes_2024["SA1_CODE21"].isin(SA1s_with_votes),]
        Redistribution_SA1_changes_2024_dict = Redistribution_SA1_changes_2024.groupby(['old_div', 'new_div'])['SA1_CODE21'].apply(list).to_dict()
        Redistribution_pairs = list(Redistribution_SA1_changes_2024_dict.keys())

        Redistribution_pair = {key: [] for key in Redistribution_pairs}

        Redistribution_pair_SA1s = Redistribution_SA1_changes_2024.groupby(['old_div', 'new_div'])['SA1_CODE21'].apply(list).reset_index()

        recase_map = {'Mcmahon':'McMahon', 'Mcewen':'McEwen','Eden-monaro':'Eden-Monaro',"O'connor": "O'Connor"}

        Redistribution_pair_SA1s.iloc[:,:2] = Redistribution_pair_SA1s.iloc[:,:2].replace(recase_map)
        import pdb;pdb.set_trace()

        transformed_votes = full_redistribution_candidate_change(Formal_prefs_dict, Elimination_order_dict, Senate_parties_by_div, DOP_By_PP_Expand_wide_dict, DOP_By_PP_Pref_Percent_wide_dict, Incumbency_by_div, new_seats_list)

    return Final_allocated_pcts_aggregated_dict, Final_x_HS_df








Final_allocated_pcts_aggregated_dict, Final_x_HS_df = whole_procedure(Formal_prefs_dict,general_party_df, Senate_party_abvs_dict, x=5)



import pdb;pdb.set_trace()
