import pandas as pd
import numpy as np
import os
file_path = os.getcwd()
print(file_path)
os.chdir('C:\\Dania\\2024\\Australian Election')
file_path = os.getcwd()
print(file_path)

DOP_data = pd.read_csv("HouseDOPByDivision_Format1.csv")

First_Preferences_Full_Percents = DOP_data.loc[(DOP_data["CountNumber"] == 0) & (DOP_data["CalculationType"] =="Preference Percent")]
First_Preferences_Full_Counts = DOP_data.loc[(DOP_data["CountNumber"] == 0) & (DOP_data["CalculationType"] =="Preference Count")]


First_Preferences_Full_Percents = First_Preferences_Full_Percents.reset_index(drop=True)
First_Preferences_Full_Percents.loc[:,"ID"] = First_Preferences_Full_Percents.index
First_Preferences_Full_Counts = First_Preferences_Full_Counts.reset_index(drop=True)
First_Preferences_Full_Counts.loc[:,"ID"] = First_Preferences_Full_Counts.index

#print(First_Preferences_Full_Percents)
#print(First_Preferences_Full_Counts)

First_Preferences_Full = pd.merge(First_Preferences_Full_Percents, First_Preferences_Full_Counts[["CalculationValue","ID"]], on = "ID")



First_Preferences_Full = First_Preferences_Full.rename(columns={"CalculationValue_x": "PrimaryVote%", "CalculationValue_y": "PrimaryVotes"})


First_Preferences_Full.loc[First_Preferences_Full["PartyNm"] == "Queensland Greens", "PartyNm"] = "The Greens"
First_Preferences_Full.loc[First_Preferences_Full["PartyNm"] == "The Greens (WA)", "PartyNm"] = "The Greens"
First_Preferences_Full.loc[First_Preferences_Full["PartyNm"] == "National Party", "PartyNm"] = "The Nationals"
First_Preferences_Full.loc[First_Preferences_Full["PartyNm"] == "Democratic Alliance", "PartyNm"] = "Drew Pavlou Democratic Alliance"
First_Preferences_Full.loc[First_Preferences_Full["PartyNm"] == "A.L.P.", "PartyNm"] = "Labor"
First_Preferences_Full.loc[First_Preferences_Full["PartyNm"] == "Australian Labor Party", "PartyNm"] = "Labor"
First_Preferences_Full.loc[First_Preferences_Full["PartyNm"] == "NT CLP", "PartyNm"] = "Liberal"
First_Preferences_Full.loc[First_Preferences_Full["PartyNm"].isna(), "PartyNm"] = "Independent"



First_Preferences_Pruned = First_Preferences_Full[["StateAb","DivisionNm","PartyAb","Elected","PrimaryVote%","PrimaryVotes"]]
print(First_Preferences_Pruned.head())



First_Preferences_Pruned.loc[First_Preferences_Pruned["PartyAb"] == "GVIC", "PartyAb"] = "GRN"
First_Preferences_Pruned.loc[First_Preferences_Pruned["PartyAb"] == "CLP", "PartyAb"] = "LP"
First_Preferences_Pruned.loc[First_Preferences_Pruned["PartyAb"].isna(), "PartyAb"] = "IND"


Division_Names = First_Preferences_Pruned["DivisionNm"].unique()
#print(len(Division_Names), Division_Names)

Parties = First_Preferences_Pruned["PartyAb"].unique()
#print(len(Parties), Parties)

Party_Names = First_Preferences_Full["PartyNm"].unique()
#print(len(Party_Names), Party_Names)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.expand_frame_repr', False)

Refined_FP_Table = First_Preferences_Pruned
Coalition_List = ["LP","NP","LNP"]
First_Preferences_Coalition = First_Preferences_Pruned
First_Preferences_Coalition.loc[First_Preferences_Coalition["PartyAb"].isin(Coalition_List), "PartyAb"] = "COAL"
print(First_Preferences_Coalition)

def Vote_By_Parties_Partitioned(First_Preferences_Partitioned):

    Party_Totals = First_Preferences_Partitioned.groupby("PartyAb")["PrimaryVotes"].sum()
    Party_Totals_Dict = Party_Totals.to_dict()

    Party_Percentage_Dict = {}

    for party in Party_Totals_Dict:
        Party_Percentage_Dict[party] = round(Party_Totals_Dict[party]/sum(Party_Totals),4)
    print(Party_Percentage_Dict)
    return Party_Percentage_Dict

First_Preferences_Combined_Dict = Vote_By_Parties_Partitioned(First_Preferences_Coalition)

First_Preferences_Combined_Dict["Other"] = 0
parties_to_remove = []
for party in First_Preferences_Combined_Dict:
    if (First_Preferences_Combined_Dict[party] < 0.005) and (party != "XEN") and (party != "KAP"):

        parties_to_remove.append(party)
        First_Preferences_Combined_Dict["Other"] += First_Preferences_Combined_Dict[party]
for party in parties_to_remove:
    del First_Preferences_Combined_Dict[party]

Combined_Significant_Party_List = list(First_Preferences_Combined_Dict.keys())
print(Combined_Significant_Party_List)


Table = pd.DataFrame(columns = ["Division"] + Combined_Significant_Party_List)
print(Table)

# Ignore insignificant parties - their number ensures everything sums to 1
First_Preferences_Combined_Significant_Parties = First_Preferences_Coalition.loc[
                                                    First_Preferences_Coalition["PartyAb"].isin(Combined_Significant_Party_List), ]

Table1 = First_Preferences_Coalition.pivot_table(index='DivisionNm', columns='PartyAb', values='PrimaryVotes', aggfunc='sum')
Table1 = Table1.fillna(0)
numeric_cols_full = Table1.select_dtypes(include='number')
Votes_per_seat = numeric_cols_full.sum(axis=1)

Table2 = First_Preferences_Combined_Significant_Parties.pivot_table(index='DivisionNm', columns='PartyAb', values='PrimaryVotes', aggfunc='sum')
Table2 = Table2.fillna(0)
numeric_cols = Table2.select_dtypes(include='number')
Table2['Other'] = Votes_per_seat - numeric_cols.sum(axis=1)

sum_row = Table2.sum(axis=0)
sum_df = pd.DataFrame(sum_row).T
sum_df.index = ['Total']

Table2 = pd.concat([Table2, sum_df])

# sort parties by overall vote share, with "Other" column last
sorted_columns = sum_row.drop("Other").sort_values(ascending = False).index.tolist() + ["Other"]

Table2_sorted = Table2[sorted_columns]
print(Table2_sorted)

Votes_per_seat = pd.concat([Votes_per_seat, pd.Series({"Total": sum(Votes_per_seat)})])
print(Votes_per_seat)
Table2_sorted_percent = Table2_sorted.div(Votes_per_seat/100, axis=0).round(2)
print(Table2_sorted_percent)

Table2_sorted_percent.to_csv('PartyVotePercentByDivision2022.csv', index=True)

#print(First_Preferences_Pruned.loc[First_Preferences_Pruned["PartyAb"] == "GRN",].sort_values(by='CalculationValue', ascending = False))