import pandas as pd
import numpy as np
import os, time

os.chdir('C:\\Dania\\2024\\Australian Election')

SenateCandidates_2016 = pd.read_csv("2016SenateCandidates.csv", index_col = None)
SenateCandidates_2016 = SenateCandidates_2016.loc[SenateCandidates_2016["nom_ty"] == 'S',:]

import pdb;pdb.set_trace()

states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']
for state in states:
    Formal_Prefs_2016 = pd.read_csv(f"2016FormalPrefs{state}.csv", index_col = None)

