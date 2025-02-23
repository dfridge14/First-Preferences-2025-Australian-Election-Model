import pandas as pd
import glob
import os

os.chdir('C:\\Dania\\2024\\Australian Election') 

csv_files = glob.glob("C:/Dania/2024/Australian Election/DOP_By_PP_2022/*.csv")
DOP_By_PP_2022 = pd.concat((pd.read_csv(f, index_col = None, skiprows=1) for f in csv_files), ignore_index=True)

import pdb;pdb.set_trace()

DOP_By_PP_2022.to_csv("2022DOP_By_PP_full.csv", index=False)
