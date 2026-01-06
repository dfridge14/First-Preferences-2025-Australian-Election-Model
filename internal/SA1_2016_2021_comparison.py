import pandas as pd
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt

base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else base_dir = Path.home() / "Australian Election"
os.chdir(base_dir)


SA1s = pd.read_csv("CG_SA1_2016_SA1_2021.csv", index_col=None)
SA1s = SA1s[["SA1_MAINCODE_2016","SA1_CODE_2021","RATIO_FROM_TO"]]
print(SA1s)

SA1s_7dig = SA1s[:-1]
print(SA1s_7dig)
SA1s_7dig.loc[:,'SA1_CODE_2021'] = SA1s_7dig['SA1_CODE_2021'].astype(str).str[:1] + SA1s_7dig['SA1_CODE_2021'].astype(str).str[5:]

#print(SA1s_7dig.loc[~SA1s_7dig['SA1_CODE_2021'].str.isnumeric(),])
SA1s_7dig.loc[:,'SA1_CODE_2021'] = SA1s_7dig['SA1_CODE_2021'].astype(int)
SA1s_7dig.loc[:,'SA1_MAINCODE_2016'] = SA1s_7dig['SA1_MAINCODE_2016'].astype(str).str[:1] + SA1s_7dig['SA1_MAINCODE_2016'].astype(str).str[5:]
SA1s_7dig.loc[:,'SA1_MAINCODE_2016'] = SA1s_7dig['SA1_MAINCODE_2016'].astype(float).astype(int)


#SA1s_7dig['SA1_CODE_2021'] = SA1s_7dig['SA1_CODE_2021'].astype(int)
print(SA1s_7dig)
print(SA1s_7dig.loc[SA1s_7dig["SA1_CODE_2021"] == 2117417,])


SA1s_2016 = SA1s_7dig['SA1_MAINCODE_2016'].unique()
SA1s_2021 = SA1s_7dig['SA1_CODE_2021'].unique()
print(len(SA1s_7dig['SA1_CODE_2021'].unique()))
print(len(SA1s_7dig['SA1_MAINCODE_2016'].unique()))


j = 0
for id in SA1s_2016:
    if id not in SA1s_2021:
        j+= 1
print("Count: ", j)
