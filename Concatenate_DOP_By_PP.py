import pandas as pd
import glob
import os

os.chdir('C:\\Dania\\2024\\Australian Election') 

data_year = '2016'

csv_files = glob.glob(f"C:/Dania/2024/Australian Election/DOP_By_PP_{data_year}/*.csv")
DOP_By_PP = pd.concat((pd.read_csv(f, index_col = None, skiprows=1) for f in csv_files), ignore_index=True)

import pdb;pdb.set_trace()

DOP_By_PP.to_csv(f"{data_year}DOP_By_PP_full.csv", index=False)
