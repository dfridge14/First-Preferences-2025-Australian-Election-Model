import pandas as pd
import numpy as np
import os,time
import ast
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

data_year = "2016"

incumbent_df = pd.read_csv("incumbent_df.csv")

election_years = ['1993','1996','1998','2001','2004','2007','2010','2013','2016','2019','2022']

final_cand_no_dict = {"2022":5, "2019": 4, "2016": 3,"2013": 5, "2010": 3, "2007": 4, "2004": 4,"2001":4}
name_changes_year_dict = {'2022': {},'2019':{},'2016':{'Denison':'Clark','Batman':'Cooper','McMillan':'Monash','Melbourne Ports':'Macnamara','Murray':'Nicholls','Wakefield':'Spence'},'2013':{'Fraser':'Fenner','Throsby':'Whitlam'},'2010':{},'2007':{'Prospect':'McMahon','Kalgoorlie':'Durack'},'2004':{}}

FINAL_CAND_NO = final_cand_no_dict[data_year]

INTERESTED_NO_CANDS = 5


# Next task - add 2 columns to Top 5 DOP in each electorate: was elected last year (or byelection???) 

DOP_By_Division = pd.read_csv(f"{data_year}HouseDOPByDivision.csv", skiprows=1)
DOP_By_Division.rename(columns={'DivisionNm': 'div_nm', 'CandidateID': 'cand_id'}, inplace=True)
DOP_By_Division.loc[:,'div_nm'] = DOP_By_Division.loc[:,'div_nm'].replace(name_changes_year_dict[data_year]) # update to new names

for i, year in enumerate(election_years):
    if year == data_year:
        between_election_year_range = [str(j) for j in range(int(election_years[i-1]),int(election_years[i]))] # accounts for last election and byelections
        before_last_election_year_range = [str(k) for k in range(int(election_years[0]),int(election_years[i-1]))]

print(between_election_year_range)
print(before_last_election_year_range)




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

    # ignore divs that have fewer than INTERESTED_NO_CANDS candidates

    if DOP_table_vote_percents_wide.iloc[-1,0] >= INTERESTED_NO_CANDS - 2: # 0 is column index of CountNumber

        final_count_no =  num_pref_counts - 1 - (INTERESTED_NO_CANDS - 2) # final CountNumber - (desired_candidate_no - # of remaining parties in final count)
        Final_x = DOP_table_vote_percents_wide.iloc[final_count_no,1:].reset_index() #.rename(columns={1: 'House_Pct'})
        Final_x.columns = ['PartyAb','House_Pct']

        Final_x = Final_x.loc[Final_x['House_Pct'] > 0,] # only 5 remaining parties
        Final_x.loc[:,"div_nm"] = div
        
        Final_x_div_dict[div] = Final_x[["div_nm","PartyAb",'House_Pct']]

        if "IND" in Final_x['PartyAb'].values:
            print(div)





# df with div_nm,PartyAb,House_Pct,Surname,GivenNm,HistoricElected
Final_x_df = pd.concat(Final_x_div_dict.values(), ignore_index=False)
Final_x_df = Final_x_df.merge(Candidates_By_Division_df, on = ['div_nm','PartyAb'], how = 'left').rename(columns={"HistoricElected": "is_incumbent"}) # use HistoricElected column for incumbency
Final_x_df.loc[:,"is_incumbent"] = Final_x_df.loc[:,"is_incumbent"].map({'Y': 1, 'N': 0})

# remove middle name!
Final_x_df.loc[Final_x_df["GivenNm"].apply(lambda x: len(x.split(' ')) > 1),"GivenNm"] = Final_x_df.loc[Final_x_df["GivenNm"].apply(lambda x: len(x.split(' ')) > 1), "GivenNm"].apply(lambda x: x.split(' ')[0]) # only first name

# handle exceptions where multiple versions of name etc: Prosser, Horne, Smith, Sidebottom, StClair, Barresi
Final_x_df.loc[(Final_x_df["Surname"]=="BARRESI")&(Final_x_df["GivenNm"]=="Phillip"),"GivenNm"] = "Phil"
Final_x_df.loc[(Final_x_df["Surname"]=="PROSSER")&(Final_x_df["GivenNm"]=="Geoffrey"),"GivenNm"] = "Geoff"
Final_x_df.loc[(Final_x_df["Surname"]=="SIDEBOTTOM")&(Final_x_df["GivenNm"]=="Peter"),"GivenNm"] = "Sid"
Final_x_df.loc[(Final_x_df["Surname"]=="Horne")&(Final_x_df["GivenNm"]=="Robert"),"GivenNm"] = "Bob"
Final_x_df.loc[(Final_x_df["Surname"]=="ST CLAIR")&(Final_x_df["GivenNm"]=="Stuart"),"Surname"] = "STCLAIR" # only relevant for 2001...?
# Tony1 SMITH irrelevant as before 2001!!!


def count_earlier_years(years, curr_year):
    if isinstance(years, list):  # Ensure it's a list
        return sum(int(year) < int(curr_year) for year in years)
    return 0  # Keep NaN values as NaN

# is_incumbent is now equivalent to HistoricElected, but is_historic_incumbent captures all previous
Candidate_Incumbency = Final_x_df.merge(incumbent_df[['Surname', 'GivenNm','Year']], on=['Surname', 'GivenNm'], how='left')
Candidate_Incumbency['Year'] = Candidate_Incumbency['Year'].apply(lambda entry: ast.literal_eval(entry) if pd.notna(entry) else np.nan)  # make it back into list - somehow a mistake has been made!
#Candidate_Incumbency.loc[:,"is_incumbent"] = Candidate_Incumbency.loc[:,"is_incumbent"].map({'Y': 1, 'N': 0}) # convert to 1 or 0
#Candidate_Incumbency.loc[:,"is_incumbent"] = Candidate_Incumbency['Year'].apply(lambda years: any(year in between_election_year_range for year in years) if years and isinstance(years, list) else np.nan) # true if recent incumbent
Candidate_Incumbency.loc[:,"is_historic_incumbent"] = Candidate_Incumbency['Year'].apply(lambda years: not any(year in between_election_year_range for year in years) and any(year in before_last_election_year_range for year in years) if years and isinstance(years, list) else 0)
Candidate_Incumbency.loc[:,"is_historic_incumbent"] = Candidate_Incumbency.loc[:,"is_historic_incumbent"].astype(int)
import pdb;pdb.set_trace()
Candidate_Incumbency.loc[:,"elections_won"] = Candidate_Incumbency['Year'].apply(lambda x: count_earlier_years(x, data_year))


# TO DO!!!
# add in number of years incumbent!!!
# convert back to LP or NP in house!




def create_Incumbents_by_div(Candidate_Incumbency, data_year):
    ### creates df of incumbent party by division
    Incumbents_by_div = Candidate_Incumbency.loc[Candidate_Incumbency['is_incumbent'] == 1,['div_nm','PartyAb']]
    import pdb;pdb.set_trace()

    Incumbents_by_div.to_csv(f"{data_year}Incumbents.csv", index=False)

    return 1

create_Incumbents_by_div(Candidate_Incumbency, data_year)



Final_x_df = Final_x_df.merge(Candidate_Incumbency, on=Final_x_df.columns.tolist(), how='left')
Final_x_df.loc[:,["is_incumbent","is_historic_incumbent"]] = (Final_x_df.loc[:,["is_incumbent","is_historic_incumbent"]].fillna(0).infer_objects(copy=False)) # replace non-historic-elected with 0s
Final_x_df = Final_x_df.drop(columns = ['Surname', 'GivenNm','Year'])



# remove those with final 5 that are not in senate - extend to any party whose not in the senate


general_party_df = pd.read_csv(f"{data_year}GeneralPartyDetails.csv", skiprows = 1)
general_party_df.loc[general_party_df["PartyAb"] == 'GVIC',] = 'GRN' # handle exceptions, but think GVIC is the only one



Senate_parties_by_div = pd.read_csv(f"{data_year}Senate_parties_by_div.csv")
Senate_parties_by_div["PartyAbList"] = Senate_parties_by_div["PartyAbList"].apply(ast.literal_eval)


# get state-to-div dict
div_to_state = pd.read_csv(f"{data_year}HouseMembersElected.csv", skiprows=1)[['DivisionNm','StateAb']].rename(columns = {'DivisionNm': 'div_nm'})
div_to_state_dict = {name_changes_year_dict[data_year].get(div, div): div_to_state.loc[div_to_state['div_nm'] == div, 'StateAb'].iloc[0] for div in div_to_state['div_nm'].unique()}


# CURRENTLY IGNORES SEATS WITH BOTH LIBS,NATS BUT ONLY LIBS IN SENATE LIKE IN WESTERN AUSTRALIA
# change parties in NSW/VIC due to Coalition on senate ticket
Final_x_df.loc[:,'copied_PartyAb'] = Final_x_df['PartyAb'].values
Final_x_df.loc[((Final_x_df["div_nm"].map(div_to_state_dict) == 'VIC') | (Final_x_df["div_nm"].map(div_to_state_dict) == 'NSW')) & (Final_x_df["PartyAb"].isin(['LP','NP'])),'PartyAb'] = 'COAL'



Final_x_party_not_in_senate = []

for div in Final_x_df["div_nm"].unique(): #Final_x_div_dict.keys():
    #print(div)
    for i in range(INTERESTED_NO_CANDS):

        if Final_x_df.loc[Final_x_df['div_nm'] == div, "PartyAb"].values[i] not in Senate_parties_by_div.loc[Senate_parties_by_div["div_nm"] == div,"PartyAbList"].iloc[0]:
            print(Final_x_df.loc[Final_x_df['div_nm'] == div, "PartyAb"].values[i])
            Final_x_party_not_in_senate.append(div)

print(Final_x_party_not_in_senate)
#import pdb;pdb.set_trace()

Final_x_df_for_Incumbency = Final_x_df.loc[~Final_x_df["div_nm"].isin(Final_x_party_not_in_senate),]

# combine any VIC/NSW coalition candidates in top x
Final_x_party_has_2_COALs = []
for div, group in Final_x_df_for_Incumbency[['div_nm','PartyAb']].groupby('div_nm'):
    if group.iloc[:,1].tolist().count("COAL") == 2:
        Final_x_party_has_2_COALs.append(div)
print(Final_x_party_has_2_COALs)


Final_x_df_for_Incumbency = Final_x_df_for_Incumbency.loc[~Final_x_df["div_nm"].isin(Final_x_party_has_2_COALs),]

import pdb;pdb.set_trace()

Final_x_df_for_Incumbency.to_csv(f"{data_year}Final_{INTERESTED_NO_CANDS}_for_Incumbency.csv", index=False)



def fill_multiple_independents_order(DOP_div_df):

    # relabel independents in order of ballot appearance if there are multiple - DOP_df must have col 'PartyAb'
    target = 'IND'
    DOP_div_df.loc[:,'Count'] = (DOP_div_df.groupby('PartyAb').cumcount() + 1)     # Count instances of the target string
    # Replace duplicates of the target string with increasing strings A1, A2, A3, ...
    adjusted_party_names = DOP_div_df.loc[DOP_div_df["CountNumber"] == 0,].apply(
        lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
    )
    return adjusted_party_names

