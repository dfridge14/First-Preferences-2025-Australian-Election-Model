import pandas as pd
import numpy as np
import os,time


os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()

election_years = ['1993','1996','1998','2001','2004','2007','2010','2013','2016','2019','2022']
old_years = election_years[:4]
elected = {}


combined_df = pd.DataFrame(columns=["Surname","GivenNm","div_nm","PartyAb"])


for year in election_years: # currently only 2016 onwards
    filename = f"{year}HouseMembersElected.csv"

    if year in old_years:
        curr_year_elected = pd.read_csv(filename)

        curr_year_elected = curr_year_elected.applymap(lambda x: x.strip() if isinstance(x, str) else x) # strip extra spaces in formatting


        # manual tweaks 1998 - sid sidebottom, Stuart St Clair, phil barresi || 1993 - bob horne, geoff prosser || 1996 Tony1 Smith and triple names
        if year == '1993':
            curr_year_elected.loc[curr_year_elected["Name"] == "Geoffrey Prosser","Name"] = "Geoff Prosser"
            curr_year_elected.loc[curr_year_elected["Name"] == "Robert Horne","Name"] = "Bob Horne"

        if year == '1996':
            curr_year_elected.loc[curr_year_elected["Name"] == "Tony Smith","Name"] = "Tony1 Smith"

        if year == '1998':
            curr_year_elected.loc[curr_year_elected["Name"] == "Peter Sid Sidebottom","Name"] = "Sid Sidebottom"
            curr_year_elected.loc[curr_year_elected["Name"] == "Stuart St Clair","Name"] = "Stuart StClair"
            curr_year_elected.loc[curr_year_elected["Name"] == "Phillip Barresi","Name"] = "Phil Barresi"

        # also manually tweaked apostrophe symbol in O'Byrne, Connor, Keefe from 1996-2001 for some reason


        
        # remove middle names
        curr_year_elected.loc[curr_year_elected["Name"].apply(lambda x: len(x.split(' ')) > 2),"Name"] = curr_year_elected.loc[curr_year_elected["Name"].apply(lambda x: len(x.split(' ')) > 2), "Name"].apply(
        lambda x: ' '.join([x.split(' ')[0], x.split(' ')[-1]])) # replace string of 3 names with 1st and last name

        split_names = curr_year_elected.loc[:,"Name"].str.split(" ")

        # Stuart St Clair & Peter Sid Sidebottom

        # if 3 names, reduce middle name to 2


        if sum(split_names.apply(lambda x: len(x) != 2)) > 0 :
            print("triple name!")
            
        curr_year_elected.iloc[split_names.apply(lambda x: len(x) != 2).loc[split_names.apply(lambda x: len(x) != 2)].index,] 

        curr_year_elected["GivenNm"] = split_names.str[0]
        curr_year_elected["Surname"] = split_names.str[1].str.upper()
        curr_year_elected = curr_year_elected[["Surname","GivenNm","DivisionNm","PartyAb"]].rename(columns = {"DivisionNm": "div_nm"})
        curr_year_elected["Year"] = year

    else:
        curr_year_elected = pd.read_csv(filename, skiprows=1)[["Surname","GivenNm","DivisionNm","PartyAb"]].rename(columns = {"DivisionNm": "div_nm"}) # only relevant columns
        

        
        # make surname all capital - exception for Bert van Manen: VANMANEN
        curr_year_elected.loc[curr_year_elected["Surname"] == 'van MANEN',"Surname"] = "VAN MANEN"

        # remove middle name!
        curr_year_elected.loc[curr_year_elected["GivenNm"].apply(lambda x: len(x.split(' ')) > 1),"GivenNm"] = curr_year_elected.loc[curr_year_elected["GivenNm"].apply(lambda x: len(x.split(' ')) > 1), "GivenNm"].apply(lambda x: x.split(' ')[0]) # only first name
        
        
        curr_year_elected["Year"] = year # add year column

    #print(curr_year_elected)
    elected[f"{year}"] =  curr_year_elected

    combined_df = pd.concat([combined_df, elected[f"{year}"]], ignore_index=True)


byelections_df = pd.read_csv("ByElectionMembersElected.csv")
byelections_df.loc[:,"Year"] = byelections_df.loc[:,"Year"].astype(str)
combined_df = pd.concat([combined_df, byelections_df], ignore_index=True) # add in byelections



# incumbent_df should have elected candidates as keys, with election years and divisions as columns, which should each be lists
incumbent_df = combined_df.groupby(['Surname','GivenNm']).agg(list).reset_index()



# produce csv file of incumbent df
#incumbent_df.to_csv("incumbent_df.csv", index=False)

pd.set_option('display.max_rows', None)
print(incumbent_df)





