import pandas as pd
import glob
import os
from pathlib import Path


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)


data_year = '2013'

directory = Path(f"C:/Dania/2024/Australian Election/DOP_By_PP_{data_year}") if os.name == "nt" else Path.home() / f"Australian Election/DOP_By_PP_{data_year}"

csv_files = glob.glob(str(f"{directory}/*.csv"))
DOP_By_PP = pd.concat((pd.read_csv(f, index_col = None, skiprows=1) for f in csv_files), ignore_index=True)

import pdb;pdb.set_trace()

DOP_By_PP.to_csv(f"{data_year}DOP_By_PP_full.csv", index=False)
