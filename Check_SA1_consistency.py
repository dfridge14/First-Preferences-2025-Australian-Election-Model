import pandas as pd
import geopandas as gpd
import numpy as np
import os, time


os.chdir('C:\\Dania\\2024\\Australian Election')

start = time.time()

SA1_Correspondence_2016_2021 = pd.read_csv("CG_SA1_2016_SA1_2021.csv", index_col=None)
print(SA1_Correspondence_2016_2021)
SA1_Correspondence_2016_2021.rename(columns={"SA1_MAINCODE_2016": "SA1_CODE16", "SA1_CODE_2021":"SA1_CODE21"}, inplace=True)
SA1_Correspondence_2016_2021 = SA1_Correspondence_2016_2021[["SA1_CODE16","SA1_CODE21","RATIO_FROM_TO"]].drop(SA1_Correspondence_2016_2021.index[-1]) # removes last misbehaving row
SA1_Correspondence_2016_2021['SA1_CODE16'] = SA1_Correspondence_2016_2021['SA1_CODE16'].apply(lambda x: int(x))
print(SA1_Correspondence_2016_2021)

SA1_Correspondence_2016_2021['SA1_CODE21'] = SA1_Correspondence_2016_2021['SA1_CODE21'].astype(str).str[:1] + SA1_Correspondence_2016_2021['SA1_CODE21'].astype(str).str[5:]
SA1_Correspondence_2016_2021['SA1_CODE16'] = SA1_Correspondence_2016_2021['SA1_CODE16'].astype(str).str[:1] + SA1_Correspondence_2016_2021['SA1_CODE16'].astype(str).str[5:]

#SA1_Correspondence_2016_2021['SA1_CODE21'] = SA1_Correspondence_2016_2021['SA1_CODE21'].astype(int)
#SA1_Correspondence_2016_2021['SA1_CODE16'] = SA1_Correspondence_2016_2021['SA1_CODE16'].astype(int)

print(SA1_Correspondence_2016_2021)

VIC_SA1s_Redistribution_full = pd.read_csv("Vic-2024-electoral-divisions-SA1-and-SA2.csv", index_col=None)
#print(VIC_SA1s_Redistribution_full)
VIC_SA1s_Redistribution_full = VIC_SA1s_Redistribution_full.rename(columns={'SA1_Code_2021': 'SA1_CODE21',"New Electoral Division": 'new_div', "Old Electoral Division": 'old_div', "Actual Enrolment": 'curr_enrol',"Projected Enrolment": 'proj_enrol'})
VIC_SA1s_Redistribution = VIC_SA1s_Redistribution_full[["SA1_CODE21","new_div","old_div",'curr_enrol','proj_enrol']].drop(VIC_SA1s_Redistribution_full.index[-1]) # removes last misbehaving row

# solving complications with SA1s, where there is mixing going on, or 0 populations
VIC_SA1s_Redistribution = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution['curr_enrol'] != 0) & (VIC_SA1s_Redistribution['proj_enrol'] != 0)] # potentially be stricter removing any with <10 -- explore
Mixed_SA1s = VIC_SA1s_Redistribution.loc[VIC_SA1s_Redistribution["SA1_CODE21"].apply(lambda x: isinstance(x,str) and x[-1].isalpha()),] # finds which ones end in alphabet characters A to H (i.e. are mixed SA1s)
print(Mixed_SA1s)
### improve: can remove any solitary ones remaining as their partner is area of 0 votes!


print("Higgins stuff")

SA1s_From_Higgins = VIC_SA1s_Redistribution.loc[VIC_SA1s_Redistribution["old_div"]=="Higgins","SA1_CODE21"].tolist()
print(sorted(SA1s_From_Higgins), len(SA1s_From_Higgins))
Higgins_SA1_Correspondence_2016_2021 = SA1_Correspondence_2016_2021.loc[SA1_Correspondence_2016_2021["SA1_CODE21"].isin(SA1s_From_Higgins),]
Dodgy_SA1s = Higgins_SA1_Correspondence_2016_2021.loc[Higgins_SA1_Correspondence_2016_2021["RATIO_FROM_TO"] < 1,]
print(Dodgy_SA1s, Dodgy_SA1s.shape)

print("Aston stuff")
SA1s_to_Aston = VIC_SA1s_Redistribution.loc[(VIC_SA1s_Redistribution["old_div"]=="Deakin") & (VIC_SA1s_Redistribution["new_div"]=="Aston"),"SA1_CODE21"].tolist()
print(sorted(SA1s_to_Aston), len(SA1s_to_Aston))
Aston_SA1_Correspondence_2016_2021 = SA1_Correspondence_2016_2021.loc[SA1_Correspondence_2016_2021["SA1_CODE21"].isin(SA1s_to_Aston),]
Dodgy_SA1s = Aston_SA1_Correspondence_2016_2021.loc[Aston_SA1_Correspondence_2016_2021["RATIO_FROM_TO"] < 1,]
print(Dodgy_SA1s, Dodgy_SA1s.shape)
print(time.time()-start, "seconds")

import pdb;pdb.set_trace()
