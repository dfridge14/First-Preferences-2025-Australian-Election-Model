import pandas as pd
import numpy as np
import os,time
import ast


os.chdir('C:\\Dania\\2024\\Australian Election')


incumbent_df = pd.read_csv("incumbent_df.csv")

election_years = ['1993','1996','1998','2001','2004','2007','2010','2013','2016','2019','2022']
FINAL_CAND_NO = 5


# Next task - add 2 columns to Top 5 DOP in each electorate: was elected last year (or byelection???) 

DOP_By_Division = pd.read_csv("HouseDOPByDivisionDownload-27966.csv", skiprows=1)

DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)


between_election_year_range = [str(i) for i in range(int(election_years[-2]),int(election_years[-1]))] # accounts for last election and byelections
print(between_election_year_range)

DOP_By_Division_Incumbency = DOP_By_Division.merge(incumbent_df, on=['Surname', 'GivenNm'], how='left')
DOP_By_Division_Incumbency["is_incumbent"] = DOP_By_Division_Incumbency['Year'].apply(lambda years: any(year in between_election_year_range for year in years) if isinstance(years, list) and years else 0) # true if recent incumbent



Div_DOP_dict = {div: group for div, group in DOP_By_Division[["div_nm","CountNumber","Surname", "GivenNm","PartyAb","HistoricElected","CalculationType", "CalculationValue"]].groupby("div_nm")}

Candidates_By_Division_df = pd.DataFrame(columns = ["div_nm","PartyAb","Surname", "GivenNm","HistoricElected"])

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

Final_x_div_dict = {}
for div in Div_DOP_dict.keys():
    #print(div)
    DOP_table_vote_pcts_long = Div_DOP_dict[div].loc[Div_DOP_dict[div]["CalculationType"] == "Preference Percent",].copy()

    # fill in empty PartyAb column with IND - in 2022, only Steve Khouw
    DOP_table_vote_pcts_long.loc[:,'PartyAb'] = DOP_table_vote_pcts_long['PartyAb'].fillna('IND') 

    # relabel independents in order of ballot appearance if there are multiple
    target = 'IND'
    DOP_table_vote_pcts_long.loc[:,'Count'] = (DOP_table_vote_pcts_long.groupby('PartyAb').cumcount() + 1)     # Count instances of the target string
    # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
    adjusted_party_names = DOP_table_vote_pcts_long.loc[DOP_table_vote_pcts_long["CountNumber"] == 0,].apply(
        lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
    )

    num_pref_counts = (DOP_table_vote_pcts_long.iloc[-1,1] + 1) # num of final count + original FP count

    DOP_table_vote_pcts_long.loc[:,'PartyAb'] = pd.concat([adjusted_party_names] * num_pref_counts, ignore_index=True).values
    DOP_table_vote_pcts_long.loc[DOP_table_vote_pcts_long["PartyAb"] == "GVIC","PartyAb"] = 'GRN' # change any GVIC into GRN ------ manual fix!



    # add to candidates_df - only relevant columns
    Candidates_By_Division_df = pd.concat([Candidates_By_Division_df, DOP_table_vote_pcts_long.loc[DOP_table_vote_pcts_long["CountNumber"]==0,["div_nm","PartyAb","Surname", "GivenNm","HistoricElected"]]], ignore_index=True) 
    Candidates_By_Division_df.loc[(Candidates_By_Division_df["div_nm"] == "La Trobe") & (Candidates_By_Division_df["PartyAb"] == "ALP"),["Surname", "GivenNm"]] = 'KUMAR', 'Abhimanyu'  # manually add Abi Kumar's name correctly


    DOP_table_vote_pcts_long = DOP_table_vote_pcts_long[["CountNumber","PartyAb","CalculationValue"]] # only values needed
    DOP_table_vote_percents_wide = convert_to_wide_format(DOP_table_vote_pcts_long, "DOP")


    final_count_no =  num_pref_counts - 1 - (FINAL_CAND_NO - 2) # final CountNumber - (desired_candidate_no - # of remaining parties in final count)
    Final_x = DOP_table_vote_percents_wide.iloc[final_count_no,1:].reset_index() #.rename(columns={1: 'House_Pct'})
    Final_x.columns = ['PartyAb','House_Pct']

    Final_x = Final_x.loc[Final_x['House_Pct'] > 0,] # only 5 remaining parties
    Final_x.loc[:,"div_nm"] = div
    
    Final_x_div_dict[div] = Final_x[["div_nm","PartyAb",'House_Pct']]

    if "IND" in Final_x['PartyAb'].values:
        print(div)


no_indeps_Final_x_divs = []
for div in Final_x_div_dict.keys():
    for i in range(FINAL_CAND_NO):
        if "IND" in Final_x_div_dict[div]["PartyAb"].values[i]:
            no_indeps_Final_x_divs.append(div)
print(no_indeps_Final_x_divs)


# df with div_nm,PartyAb,House_Pct,Surname,GivenNm,HistoricElected
Final_x_df = pd.concat(Final_x_div_dict.values(), ignore_index=False)
Final_x_df = Final_x_df.merge(Candidates_By_Division_df, on = ['div_nm','PartyAb'], how = 'left')


# remove middle name!
Final_x_df.loc[Final_x_df["GivenNm"].apply(lambda x: len(x.split(' ')) > 1),"GivenNm"] = Final_x_df.loc[Final_x_df["GivenNm"].apply(lambda x: len(x.split(' ')) > 1), "GivenNm"].apply(lambda x: x.split(' ')[0]) # only first name



#Final_x_df.apply(lambda row: 1 if row["HistoricElected"]=="Y" else 0)


between_election_year_range = [str(i) for i in range(int(election_years[-2]),int(election_years[-1]))] # accounts for last election and byelections
before_last_election_year_range = [str(i) for i in range(int(election_years[0]),int(election_years[-2]))]
print(between_election_year_range)

Candidate_Incumbency = Final_x_df.loc[Final_x_df["HistoricElected"]=="Y",].merge(incumbent_df[['Surname', 'GivenNm','Year']], on=['Surname', 'GivenNm'], how='left')
Candidate_Incumbency['Year'] = Candidate_Incumbency['Year'].apply(ast.literal_eval)  # make it back into list - somehow a mistake has been made!
Candidate_Incumbency.loc[:,"is_incumbent"] = Candidate_Incumbency['Year'].apply(lambda years: any(year in between_election_year_range for year in years)) # true if recent incumbent
Candidate_Incumbency.loc[:,"is_historic_incumbent"] = Candidate_Incumbency['Year'].apply(lambda years: not any(year in between_election_year_range for year in years) and any(year in between_election_year_range for year in years))

Final_x_df = Final_x_df.merge(Candidate_Incumbency, on=Final_x_df.columns.tolist(), how='left')
Final_x_df.loc[:,["is_incumbent","is_historic_incumbent"]].fillna(0) # replace non-historic-elected with 0s
Final_x_df = Final_x_df.drop(columns = ['Surname', 'GivenNm','HistoricElected','Year'])




states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
for state in states: # currently only 2016 onwards
    filename = f"2022FormalPrefs{state}.csv"

Formal_prefs_VIC = pd.read_csv("2022FormalPrefsVIC", index_col=None)
Formal_prefs_NSW = 1





Final_x_df.loc[:,"Senate_Pct"] = "allocate using Formal Preferences"


import pdb;pdb.set_trace()


def fill_multiple_independents_order(DOP_div_df):

    # relabel independents in order of ballot appearance if there are multiple - DOP_df must have col 'PartyAb'
    target = 'IND'
    DOP_div_df.loc[:,'Count'] = (DOP_div_df.groupby('PartyAb').cumcount() + 1)     # Count instances of the target string
    # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
    adjusted_party_names = DOP_div_df.loc[DOP_div_df["CountNumber"] == 0,].apply(
        lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
    )
    return adjusted_party_names

