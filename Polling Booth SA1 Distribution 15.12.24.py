import pandas as pd
# import geopandas as gpd
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt


os.chdir('C:\\Dania\\2024\\Australian Election')

DEGREE_TO_KM = 111.319


#print(PB_combined)

def create_PB_combined():

    # create PB_data
    PB_data_full = pd.read_csv("PollingBoothData2022.csv", index_col=None)
    PB_data_full['DivName'] = PB_data_full['DivName'].str.strip()
    PB_data = PB_data_full[["DivName","PPId","Lat","Long","OrdVoteEst","DecVoteEst"]]
    PB_data.set_index('PPId', inplace=True)
    PB_data.index.rename('pp_id', inplace=True)
    PB_data.insert(0,'Booth_type',"PB")
    #print(PB_data)


    # create PPVC_data (includes RMTs)
    PPVC_data_full = pd.read_csv("PPVCData2022.csv")
    PPVC_data = PPVC_data_full[["PPId","DivName","Lat","Long","OrdVoteEst","DecVoteEst"]]
    # Issue resolved: encountering issue - I want to combine all the rows with identical entries, but am wary that there might be PPVC that are valid for the same division. A simple solution would be to introduce the index column as an ordinary column again. Why not?
    PPVC_data = PPVC_data.groupby(["PPId","DivName","Lat","Long","OrdVoteEst","DecVoteEst"]).first().reset_index()
    PPVC_data.set_index('PPId', inplace=True)
    PPVC_data.index.rename('pp_id', inplace=True)
    PPVC_data.insert(0,'Booth_type',"PPVC")

    # check for booths with multiple location coordinates i.e. RMTs
    duplicated_indices = PPVC_data.index[PPVC_data.index.duplicated(keep=False)]
    duplicated_subset = PPVC_data.loc[duplicated_indices].drop_duplicates()

    def aggregate_group(group):
        return group.apply(lambda x: x if x.nunique() == 1 else np.nan)

    RMT_data = duplicated_subset.groupby(duplicated_subset.index).apply(aggregate_group).droplevel(0).drop_duplicates()
    RMT_data["Booth_type"] = "Other" # easier to treat them as 'Other' since they have varying coordinates - sorry about the disadvantage:(
    
    # list of RMT booths
    remote_mobile_team_pp_ids = list(duplicated_indices.unique()) 
    print("Remote mobile teams pp_ids:", remote_mobile_team_pp_ids, len(remote_mobile_team_pp_ids))
    #print(RMT_data.to_string())

    # combine PPVC and RMT together
    PPVC_NonRMT_data = PPVC_data.loc[~PPVC_data.index.isin(duplicated_indices.unique())]
    PPVC_RMT_data = pd.concat([PPVC_NonRMT_data,RMT_data])
    PPVC_RMT_data = PPVC_RMT_data.sort_index()
    
    #print(PPVC_RMT_data)
    #print(PPVC_data)

    PB_combined = pd.concat([PB_data,PPVC_RMT_data], axis=0)
    return PB_combined


#PB_data_full = pd.read_csv("PollingBoothData2022.csv", index_col=None)
#print(PB_data_full.loc[PB_data_full["Status"] == "Appointment",["Status", "DivName","PPId","Lat","Long","OrdVoteEst","DecVoteEst"]].to_string())
### To DO: remove Abolition Status and handle Appointment Status if not present on SA1s list - maybe we will ignore PB_combined


#PB_combined = create_PB_combined()

#print(PB_combined)
#print(PB_combined.index[PB_combined["Booth_type"] == "PB"].tolist())
#print(PB_combined.index[PB_combined["Booth_type"] == "Other"].tolist())

#PB_ids = PB_combined.index[PB_combined["Booth_type"] == "PB"].tolist()
#PPVC_ids = PB_combined.index[PB_combined["Booth_type"] == "PPVC"].tolist()
RMT_ids = [32469, 32470, 32471, 33211, 33212, 33213, 34147, 34148, 34149, 34150, 34151, 34152, 34153, 34154, 34155, 34156, 34157, 34158, 34159, 34160, 34161, 34162, 
46743, 47090, 47091, 47092, 47093, 47363, 47801, 57924, 65689, 65737, 65750, 65891, 66050, 82529, 97681, 108403]
EAV_ids = []
Other_ids = [0,100000]# 0 and 100000 artificially invented for temporary EAVs




SA1_By_PP_full = pd.read_csv("2022VotesBySA1.csv")
SA1_By_PP = SA1_By_PP_full[["div_nm","ccd_id","pp_id","votes"]]
SA1_By_PP = SA1_By_PP.rename(columns={'ccd_id': 'SA1_CODE16'})
#print(SA1_By_PP)
#print(SA1_By_PP.groupby(["div_nm","pp_id"], as_index=False)["votes"].sum())






# checked that all EAV COVID19 Booths that are not on the polling booths data are between 108484 and 108676. Only booths of concern are 108689 in Durack and 108711 in Page (Lismore). My guess is they were added due to natural disaster emergencies, hence similar to COVID Mobile Voting. As they also neither appear on the PPVC nor Polling Booths csvs, they will be given an index of 0 alongside all the absent, postal etc votes.
#list1 = sorted(SA1_By_PP_full.loc[SA1_By_PP_full['pp_nm'].astype(str).str.startswith('EAV COVID19'),"pp_id"].unique())
#list2 = sorted(SA1_By_PP_full["pp_id"].unique())
#not_in_list2 = [x for x in list2 if x not in list1]


# check to see that all remote mobile teams contain duplicated coordinates, which have been dealt with in the PPVC DF :)))
#Remote_PPs = SA1_By_PP_full[SA1_By_PP_full["pp_nm"].str.startswith("Remote Mobile Team")]
#result = Remote_PPs["pp_id"].isin(remote_mobile_team_pp_ids)
#Remote_PPs['pp_id_is_remote'] = result
#print(Remote_PPs.loc[Remote_PPs['pp_id_is_remote'] == False,]) # empty




First_Prefs_by_PP = pd.read_csv("FirstPrefsByPollingPlace2022.csv", skiprows=1) # includes PB, PPVC, EAV COVID PPVC (to do: separate PB,PPVC ; remove EAV/EAV COVID | also, find way to add cumulatively Absent, EAV/EAV COVID, Pre-poll, Provisional, Postal)
First_Prefs_by_PP["PartyAb"] = First_Prefs_by_PP["PartyAb"].fillna("INFORMAL")
First_Prefs_by_PP = First_Prefs_by_PP.sort_values(by="div_nm",kind='mergesort') # alphabetical order by Division Name

Collated_Votes_by_PP = First_Prefs_by_PP.groupby(["pp_id","pp_nm","div_nm"], as_index=False)["votes"].sum()
PPs_0_votes_ids = Collated_Votes_by_PP.loc[Collated_Votes_by_PP["votes"]==0,]["pp_id"].tolist() # pp_ids that have 0 votes
First_Prefs_by_PP = First_Prefs_by_PP[~First_Prefs_by_PP["pp_id"].isin(PPs_0_votes_ids)]



#print(First_Prefs_by_PP[First_Prefs_by_PP["pp_id"].isin(RMT_ids)])
#print(sorted(list(First_Prefs_by_PP[First_Prefs_by_PP["pp_nm"].str.startswith("Remote Mobile")]["pp_id"].unique()))) # check that equivalent
#print(RMT_ids)





# sort out small vote numbers
def small_vote_numbers_pp_id_discrepancies(First_Prefs_by_PP,SA1_By_PP,PB_combined):

    SA1_By_PP_unique_id_count = SA1_By_PP["pp_id"].unique().tolist()
    print(len(SA1_By_PP_unique_id_count))

    # list of unique pp_ids using PBs or PPVCs
    PB_combined_unique_id_count =  PB_combined.index.unique().tolist()
    print(len(PB_combined_unique_id_count))

    # list of unique pp_ids using PBs or PPVCs
    First_Prefs_by_PP = pd.read_csv("FirstPrefsByPollingPlace2022.csv", skiprows=1) 
    First_Prefs_by_PP_unique_id_count =  First_Prefs_by_PP["pp_id"].unique().tolist()
    print(len(First_Prefs_by_PP_unique_id_count))


    Collated_Votes_by_PP = First_Prefs_by_PP.groupby(["pp_id","pp_nm","div_nm"], as_index=False)["votes"].sum()
    PPs_0_votes = Collated_Votes_by_PP.loc[Collated_Votes_by_PP["votes"]==0,]
    PPs_0_votes_indices = PPs_0_votes["pp_id"].tolist()
    # print(Collated_Votes_by_PP.loc[Collated_Votes_by_PP["votes"] <= 5,].to_string())

    j = 0
    No_votes_suspect_list = []
    for i in SA1_By_PP_unique_id_count:
        if i not in First_Prefs_by_PP_unique_id_count:
            No_votes_suspect_list.append(i)
            print(i)
            j += 1
    print("Count = ", j)

    print(sorted(No_votes_suspect_list))
    print(sorted(PPs_0_votes_indices))
    # only 2 booths have votes but no data in SA1 splits - 65509, 65590 - both EAV with 4 votes each - insignificant!

    return 1



def compare_2_pp_id_lists(list1,list2):

    j = 0
    setminuslist = []
    for id in list1:
        if id not in list2:
            setminuslist.append(id)
            j+= 1
    print("Count in first, but not in second: ", j, "List is:")
    print(sorted(setminuslist))

    j = 0
    setminuslist = []
    for id in list2:
        if id not in list1:
            setminuslist.append(id)
            j+= 1
    print("Count in second, but not in first: ", j, "List is:")
    print(sorted(setminuslist))

    return 1






# Plan: TO DO
# 1. Remove 0-vote PPs from FirstPrefsByPollingPlace2022 (ignore 65509, 65590)                      DONE
# 2. Add RMTs/OMTs/SHTs to Other                                                                    DONE
# 3. Combine with Others into Master candidate-vote doc - First must remove EAVs,RMTs from PP df    DONE
# 4. Label everything with PB, PPVC, Other [PPVC - try both via PB_combined AND via ends_with]      DONE
#                   To finish: add Booth_type column and fill for non-PPVC
# 5. Equip PBs, PPVCs with Coordinates using PB_combined? # Perhaps use new download for PBs Lat    DONE
# 6. Absorb EAVs, APPPs, XXTs into 1 categoery in SA1s df                                           DONE
# 7. Compare total votes with SA1s vs Candidates to see gaps                                        DONE

# 8. Make Division dict of candidate votes: FirstPrefsByPollingPlace2022, cands horizontally (K^(l))DONE
# 9. Make SA1s by polling place horizontal too! (n_i)









def create_First_Prefs_APPP_Votes_by_Div():

    First_Prefs_Vote_Type_by_Div = pd.read_csv("FirstPrefsByCandidateByVoteTypeCombined2022.csv", skiprows=1) 

    First_Prefs_APPP_Votes_by_Div = First_Prefs_Vote_Type_by_Div[["div_nm","cand_id","PartyAb","votes_APPP"]]
    First_Prefs_APPP_Votes_by_Div.rename(columns={"votes_APPP": "votes"}, inplace=True)
    First_Prefs_APPP_Votes_by_Div["PartyAb"] = First_Prefs_APPP_Votes_by_Div["PartyAb"].fillna("INFORMAL")



    # format similarly to First_Prefs_by_PP
    First_Prefs_APPP_Votes_by_Div.loc[:,"Booth_type"] = "Other"
    First_Prefs_APPP_Votes_by_Div.loc[:,"pp_id"] = 0
    First_Prefs_APPP_Votes_by_Div.loc[:,"pp_nm"] = "APPP"
    First_Prefs_APPP_Votes_by_Div = First_Prefs_APPP_Votes_by_Div[["pp_id","pp_nm"] + First_Prefs_APPP_Votes_by_Div.columns[:-2].tolist()]

    # sort in alphabetical order of div_nm
    First_Prefs_APPP_Votes_by_Div = First_Prefs_APPP_Votes_by_Div.sort_values(by="div_nm",kind='mergesort') # alphabetical order by Division Name

    First_Prefs_APPP_Votes_by_Div.to_csv('FirstPrefsOtherVotesByCandidate2022.csv', index=False)
    return 1

#create_First_Prefs_APPP_Votes_by_Div()

First_Prefs_APPP_Votes_by_Div = pd.read_csv("FirstPrefsOtherVotesByCandidate2022.csv")
#print(First_Prefs_APPP_Votes_by_Div)
#print(First_Prefs_by_PP)


# add EAV First_Prefs
def combine_APPPs_EAVs(First_Prefs_by_PP,First_Prefs_APPP_Votes_by_Div):
    EAV_First_Prefs = First_Prefs_by_PP[First_Prefs_by_PP["pp_nm"].str.startswith("EAV")]
    EAV_Combined_First_Prefs = EAV_First_Prefs.groupby(["div_nm","cand_id","PartyAb"], as_index=False, sort=False).agg({
        "votes": 'sum',
        "pp_nm": lambda x: "EAV",
        "pp_id": lambda x: 100000
        })
    EAV_Combined_First_Prefs = EAV_Combined_First_Prefs[["pp_id","pp_nm"] + EAV_Combined_First_Prefs.columns[:-2].tolist()]
    #print(EAV_Combined_First_Prefs)
    #print(First_Prefs_APPP_Votes_by_Div)

    # check that the APPP and EAV DFs match up perfectly
    cols = ["div_nm","cand_id","PartyAb"]
    #cols_match = First_Prefs_APPP_Votes_by_Div[cols].equals(EAV_Combined_First_Prefs[cols])
    #print(cols_match)

    # Combines APPP and EAV votes together into Other_Votes DF.
    First_Prefs_Other_Votes_by_Div = First_Prefs_APPP_Votes_by_Div.copy()
    First_Prefs_Other_Votes_by_Div.loc[:,"votes"] = First_Prefs_APPP_Votes_by_Div.loc[:,"votes"] + EAV_Combined_First_Prefs.loc[:,"votes"]
    First_Prefs_Other_Votes_by_Div.loc[:,"pp_nm"] = "Other"
    #print(First_Prefs_Other_Votes_by_Div)

    First_Prefs_by_PP = First_Prefs_by_PP[~First_Prefs_by_PP["pp_nm"].str.startswith("EAV")] # remove EAV from First_Prefs_by_PP

    return First_Prefs_by_PP, First_Prefs_Other_Votes_by_Div


First_Prefs_by_PP, First_Prefs_Other_Votes_by_Div = combine_APPPs_EAVs(First_Prefs_by_PP,First_Prefs_APPP_Votes_by_Div)

#print(First_Prefs_Other_Votes_by_Div)



def combine_APPPs_RMTs(First_Prefs_by_PP,First_Prefs_Other_Votes_by_Div,XXT):
    ### combines APPPs (with EAVs) with the XXTs from PP dfs, XXT is a string

    if XXT == "RMT":
        X_X_T = "Remote Mobile"
    if XXT == "OMT":
        X_X_T = "Other Mobile"
    if XXT == "SHT":
        X_X_T = "Special Hospital"

    XXT_ids = sorted(list(First_Prefs_by_PP[First_Prefs_by_PP["pp_nm"].str.startswith(X_X_T)]["pp_id"].unique()))
    # print(XXT_ids, "Count of", XXT, "is: ", len(XXT_ids))

    # combine all in same division
    XXT_First_Prefs = First_Prefs_by_PP[First_Prefs_by_PP["pp_id"].isin(XXT_ids)]
    XXT_Combined_First_Prefs = XXT_First_Prefs.groupby(["div_nm","cand_id","PartyAb"], as_index=False, sort=False).agg({ 
    "votes": 'sum',
    "pp_nm": lambda x: XXT,
    "pp_id": lambda x: 100001
    })
    XXT_Combined_First_Prefs = XXT_Combined_First_Prefs[["pp_id","pp_nm"] + XXT_Combined_First_Prefs.columns[:-2].tolist()]
    #print(RMT_Combined_First_Prefs)

    # concatenate them with the Other Votes df
    Combined_df = pd.concat([First_Prefs_Other_Votes_by_Div, XXT_Combined_First_Prefs], ignore_index=True)
    # print(Combined_df)
    First_Prefs_Other_Votes_by_Div = Combined_df.groupby(["div_nm","cand_id","PartyAb"], as_index=False, sort=False).agg({
        "votes": 'sum',
        "Booth_type": lambda x: "Other",
        "pp_nm": lambda x: "Other",
        "pp_id": lambda x: 0
        })
    First_Prefs_Other_Votes_by_Div = First_Prefs_Other_Votes_by_Div[["pp_id","pp_nm"] + First_Prefs_Other_Votes_by_Div.columns[:-2].tolist()]
    
    # remove XXTs from First_Prefs_by_PP
    First_Prefs_by_PP = First_Prefs_by_PP[~First_Prefs_by_PP["pp_id"].isin(XXT_ids)]
    return First_Prefs_by_PP, First_Prefs_Other_Votes_by_Div


def combine_First_Prefs_by_PP_First_Prefs_Other(First_Prefs_by_PP,First_Prefs_Other_Votes_by_Div):

    # transfer each of RMT,OMT,SHT from First_Prefs_by_PP to 'Other'
    for XXT in ["RMT","OMT","SHT"]:
        First_Prefs_by_PP, First_Prefs_Other_Votes_by_Div = combine_APPPs_RMTs(First_Prefs_by_PP,First_Prefs_Other_Votes_by_Div,XXT)

    PPVC_pp_ids = First_Prefs_by_PP.loc[First_Prefs_by_PP["pp_nm"].str.endswith("PPVC"),"pp_id"].unique()
    #print(list(PPVC_pp_ids), "number of unique pp_ids that are PPVCs: ",len(PPVC_pp_ids))
    First_Prefs_by_PP.loc[:,"Booth_type"] = np.where(First_Prefs_by_PP["pp_nm"].str.endswith("PPVC"),"PPVC","PB") 
    # print(First_Prefs_Other_Votes_by_Div.to_string())

    First_Prefs_by_PP_Complete = pd.concat([First_Prefs_by_PP, First_Prefs_Other_Votes_by_Div], ignore_index=True)
    #print(First_Prefs_by_PP_Complete)
    return First_Prefs_by_PP_Complete


First_Prefs_by_PP_Incomplete = combine_First_Prefs_by_PP_First_Prefs_Other(First_Prefs_by_PP,First_Prefs_Other_Votes_by_Div)
#print(First_Prefs_by_PP_Incomplete)







def label_PB_PPVC(First_Prefs_by_PP_Incomplete):

    #print(First_Prefs_by_PP_Incomplete)
    PPVC_pp_ids = First_Prefs_by_PP_Incomplete.loc[First_Prefs_by_PP_Incomplete["pp_nm"].str.endswith("PPVC"),"pp_id"].unique()
    PB_pp_ids = First_Prefs_by_PP_Incomplete.loc[~First_Prefs_by_PP_Incomplete["pp_nm"].str.endswith("PPVC"),"pp_id"].unique()
    #print(list(PPVC_pp_ids), "number of unique pp_ids that are PPVCs: ",len(PPVC_pp_ids))
    #print(list(PB_pp_ids), "number of unique pp_ids that are PBs: ",len(PB_pp_ids))

    #First_Prefs_by_PP_Incomplete.loc[First_Prefs_by_PP_Incomplete["pp_nm"].str.endswith("PPVC"),"Booth_type"] = "PPVC"


    return PB_pp_ids, PPVC_pp_ids # First_Prefs_by_PP_Complete

PB_pp_ids, PPVC_pp_ids = label_PB_PPVC(First_Prefs_by_PP_Incomplete)

#PB_combined_PPVC_pp_ids = PB_combined.loc[PB_combined["Booth_type"] == "PPVC",].index.unique().tolist()
#PB_combined_PB_pp_ids = PB_combined.loc[PB_combined["Booth_type"] == "PB",].index.unique().tolist()

#print(PB_combined_PPVC_pp_ids, "number of unique pp_ids in PB_combined that are PPVCs: ", len(PB_combined_PPVC_pp_ids))
#print(PB_combined_PPVC_pp_ids, "number of unique pp_ids in PB_combined that are PBs: ", len(PB_combined_PB_pp_ids))

#compare_2_pp_id_lists(PB_pp_ids,PB_combined_PPVC_pp_ids)


#print(PPs_0_votes_ids) # Most of the missing PPVCS are from the 28 that have been removed due to recieving no votes - extra 8 removed. All fromo PPVCs are in PB_combined - we will go with endswith(PPVC)

#PBs: 170 in PB_pp_ids, but not PB_combined. 124 in PB_combined, but not PB_pp_ids - OK, as some Abolished! 


Polling_places_repository_full = pd.read_csv("PollingPlaces2022Repository.csv", index_col=None, skiprows=1)

Polling_places_repository_full = Polling_places_repository_full.rename(columns={'DivisionNm': 'div_nm', "PollingPlaceID": "pp_id", "PollingPlaceTypeID": "pp_type", "PollingPlaceNm": "pp_nm", "Latitude": "Lat", "Longitude": "Long"})
Polling_places_repository = Polling_places_repository_full[["div_nm","pp_id","pp_type","pp_nm","Lat","Long"]]


#print(Polling_places_repository)

pp_rep_PPVC_ids = Polling_places_repository.loc[(Polling_places_repository["pp_type"]==5) & (~Polling_places_repository["pp_nm"].str.startswith("EAV")),"pp_id"].unique() # includes 22 with no votes
pp_rep_PB_ids = Polling_places_repository.loc[Polling_places_repository["pp_type"]==1,"pp_id"].unique() # includes 6 with no votes
EAV_ids = Polling_places_repository.loc[Polling_places_repository["pp_nm"].str.startswith("EAV"),"pp_id"].unique().tolist()
RMT_ids = Polling_places_repository.loc[Polling_places_repository["pp_type"]==3,"pp_id"].unique().tolist()
OMT_ids = Polling_places_repository.loc[Polling_places_repository["pp_type"]==4,"pp_id"].unique().tolist()
SHT_ids = Polling_places_repository.loc[Polling_places_repository["pp_type"]==2,"pp_id"].unique().tolist()
Exception_ids = EAV_ids + RMT_ids + OMT_ids + SHT_ids

print("checking size of pp_ids!")
print(len(pp_rep_PPVC_ids),len(pp_rep_PB_ids),len(EAV_ids),len(RMT_ids),len(OMT_ids),len(SHT_ids))
print(sum([len(pp_rep_PPVC_ids),len(pp_rep_PB_ids),len(EAV_ids),len(RMT_ids),len(OMT_ids),len(SHT_ids)]))
print(len(Polling_places_repository.loc[:,"pp_id"].unique().tolist()))



#compare_2_pp_id_lists(PB_pp_ids,pp_rep_PB_ids)
#compare_2_pp_id_lists(PPVC_pp_ids,pp_rep_PPVC_ids)
#print(PPs_0_votes_ids)

First_Prefs_by_PP_Complete = pd.merge(First_Prefs_by_PP_Incomplete, Polling_places_repository[['pp_id', 'Lat','Long']], on='pp_id', how='left')

# records the Booth type for each pp_id
PP_Booth_type = First_Prefs_by_PP_Complete[["pp_id","Booth_type"]].drop_duplicates().reset_index(drop=True)
print(PP_Booth_type)
PP_Booth_type.to_csv('PPBoothtype2022.csv', index=False)
#First_Prefs_by_PP_Complete.to_csv('FirstPrefsByPP2022Complete.csv', index=False)


#print(First_Prefs_by_PP_Complete.to_string())



# convert EAV,RMT,OMT,SHT to pp_id of 0!
SA1_By_PP.loc[SA1_By_PP["pp_id"].isin(Exception_ids),"pp_id"] = 0
# group all with pp_id of 0 together!
SA1_By_PP_Complete = SA1_By_PP.groupby(["div_nm","SA1_CODE16","pp_id"], as_index=False)["votes"].sum()




# check if correct for specific SA1s - YES!
#print(SA1_By_PP.loc[SA1_By_PP["SA1_CODE16"]==7103112,].to_string())
#print(SA1_By_PP_Complete.loc[SA1_By_PP_Complete["SA1_CODE16"]==7103112,].to_string())

# create SA1_By_PP_Complete!
#print(SA1_By_PP_Complete)
SA1_By_PP_Complete.to_csv('SA1_By_PP_Complete.csv', index=False)


Vote_Sums_SA1s = SA1_By_PP_Complete.groupby(["div_nm","pp_id"], as_index=False)["votes"].sum()[["div_nm","votes"]].groupby("div_nm", as_index = False)["votes"].sum()
Vote_Sums_First_Prefs = First_Prefs_by_PP_Complete.groupby(["div_nm","pp_id"], as_index=False)["votes"].sum()[["div_nm","votes"]].groupby("div_nm", as_index = False)["votes"].sum()
Vote_Sums_Differences = pd.merge(Vote_Sums_SA1s, Vote_Sums_First_Prefs, on='div_nm', how='left')
Vote_Sums_Differences.loc[:,"vote_difference"] = Vote_Sums_Differences["votes_x"] - Vote_Sums_Differences["votes_y"]
print(Vote_Sums_Differences.to_string(), Vote_Sums_Differences["vote_difference"].sum()) # Average of 60.6 vote discrepancy per electorate - maximum is 186 - not too bad!


# The divisions where there were multiple independents
df = First_Prefs_by_PP_Complete.groupby(["div_nm","cand_id","PartyAb"], as_index=False)["votes"].sum()
duplicate_divs = df[df.duplicated(subset=['div_nm', 'PartyAb'], keep=False)]["div_nm"].unique()
#print(duplicate_divs)











# Dictionaries of all division first_prefs and SA1s

def create_Div_PP_dict(First_Prefs_by_PP_Complete):
    ### creates dictionary of first preferences by division, with division names as keys, 
    Div_First_Prefs_PP_dict = {}
    
    div_names = list(First_Prefs_by_PP_Complete["div_nm"].unique())

    for div in div_names:
        Div_First_Prefs_PP_dict[div] = First_Prefs_by_PP_Complete.loc[First_Prefs_by_PP_Complete["div_nm"] == div,]

    return Div_First_Prefs_PP_dict


Div_First_Prefs_PP_dict = create_Div_PP_dict(First_Prefs_by_PP_Complete)
Div_SA1_By_PP_dict = create_Div_PP_dict(SA1_By_PP_Complete)
#print(Div_SA1_By_PP_dict)


def unique_SA1s_dict(Div_SA1_By_PP_dict):
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
    unique_SA1s = unique_SA1s_dict(Div_SA1_By_PP_dict)

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
#print(Div_First_Prefs_PP_dict_wide)

















def get_SA1_centroids_2016():

    # Currently using 2016 version. Voting data required 2016->2021 Correspondence/Concordance : shapefile_path = 'SA1_2021_AUST_GDA2020.shp'
    shapefile_path = 'SA1_2016_AUST.shp'
    gdf_full = 0 # gpd.read_file(shapefile_path) # old architecture
    
    print(gdf_full.columns)

    gdf = gdf_full.drop(gdf_full.index[-1])
    gdf['centroid'] = gdf.geometry.centroid
    gdf['Lat'] = gdf.centroid.y
    gdf['Long'] = gdf.centroid.x

    print(gdf)
    
    SA1_centroids_7dig = pd.concat([gdf['SA1_7DIG16'],gdf['Lat'],gdf['Long']], axis=1)
    SA1_centroids_7dig['SA1_7DIG16'] = SA1_centroids_7dig['SA1_7DIG16'].astype(int)

    return SA1_centroids_7dig

#SA1_centroids_2016 = get_SA1_centroids_2016()
#SA1_centroids_2016.to_csv('SA1_centroids_2016.csv', index=False)
#print(SA1_centroids_2016)


def Electorate_SA1_PB(SA1_By_PP, DivName, SA1_CODE16):
    
    ### I want access to the latitudes and longitudes of each SA1 centroid (separate dataframe), using get_SA1_centroids_2016()
    
    SA1_Lat = -37.9138684
    SA1_Long = 145.0263502
    Electorate_PB = 1 # PB_combined.loc[PB_combined["DivName"]==DivName,] # old architecture
    Electorate_SA1_PB = SA1_By_PP.loc[SA1_By_PP["SA1_CODE16"]==SA1_CODE16,]
    return Electorate_PB, Electorate_SA1_PB


def PB_SA1_distribution(SA1_By_PP,pp_id):
    # for each PB, I want to have the votes vector for each candidate, and the vector of n_i corresponding to each attached SA1 (what to do about 0s?)
    
    PB_SA1_By_PP = SA1_By_PP.loc[SA1_By_PP["pp_id"]==pp_id,]
    
    
    return PB_SA1_By_PP

#print(PB_SA1_distribution(SA1_By_PP,3853).to_string())
#print(PB_SA1_distribution(SA1_By_PP,3853)["votes"].sum())

# Zoom in on goldstein and 2117417 for practice run ################################################################################################
def Goldstein_specific():

    #print(PB_data.loc[PB_data["DivName"]=="Goldstein",])

    #print(SA1_By_PP.loc[SA1_By_PP["SA1_CODE16"]==2117417,])
    SA1_Lat = -37.9138684
    SA1_Long = 145.0263502
    Goldstein_PB = 1 #PB_combined.loc[PB_combined["DivName"]=="Goldstein",] # old architecture
    Goldstein_PB_2117417 = SA1_By_PP.loc[SA1_By_PP["SA1_CODE16"]==2117417,]

    #print(Goldstein_PB)

    # Euclidean Distances for faster computation time - approximate haversine distance which incorporates curvature 
    Golstein_Distances_Locations = pd.DataFrame({
        "Distance": DEGREE_TO_KM*np.sqrt((Goldstein_PB[["Lat"]].iloc[:, 0]- SA1_Lat)**2 + (Goldstein_PB[["Long"]].iloc[:, 0]- SA1_Long)**2),
        "Booth_type": Goldstein_PB["Booth_type"]
    })
    Other_votes = pd.DataFrame({'Distance': None, "Booth_type": "Other"}, index = [0])
    Golstein_Distances = pd.concat([Golstein_Distances_Locations,Other_votes])
    Golstein_Distances.index.name = Golstein_Distances_Locations.index.name
    #print(Golstein_Distances)

    #print(Goldstein_PB_2117417)

    # Currently doesn't include COVID votes - 2 in Goldstein's case! 
    Vote_Distances_Goldstein_2117417 = pd.merge(Golstein_Distances, Goldstein_PB_2117417[["pp_id","votes"]], on='pp_id', how='left')
    Vote_Distances_Goldstein_2117417.set_index('pp_id', inplace=True)
    Vote_Distances_Goldstein_2117417.loc[Vote_Distances_Goldstein_2117417["votes"].isna(),"votes"] = 0 # fill un-used booths with 0
    #print(Vote_Distances_Goldstein_2117417)
    ### Dataframe indexed by pp_id with columns Distance, Booth_type (PB,PPVC,Other) and vote #

    condition = Vote_Distances_Goldstein_2117417['Booth_type'] == "PB"

    plt.plot(Vote_Distances_Goldstein_2117417['Distance'][condition], Vote_Distances_Goldstein_2117417['votes'][condition], marker='o', linestyle = '', color = 'blue',label="PB")
    plt.plot(Vote_Distances_Goldstein_2117417['Distance'][~condition], Vote_Distances_Goldstein_2117417['votes'][~condition], marker='o', linestyle = '', color = 'green',label="PPVC")

    plt.xlabel('Distance to polling booth (km)')
    plt.ylabel('Number of Votes')
    plt.title('SA1 2117417 Goldstein Distance to Polling Place')
    plt.grid(True)
    plt.show()

    return 1