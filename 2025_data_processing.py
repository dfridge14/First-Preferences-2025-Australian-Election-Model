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

election_year = "2025"

Candidates_2025 = pd.read_csv("2025Candidates_By_Division.csv", index_col=None)

Candidates_2025.groupby('div_nm')['PartyAb'].count()
Candidates_2025.groupby('PartyAb')['div_nm'].count()


dup_counts = Candidates_2025.groupby('div_nm')['PartyAb'].value_counts()

# Filter to where any count > 1
duplicates = dup_counts[dup_counts > 1]


parties = {'LP','NP'}
party_sets = Candidates_2025.groupby('div_nm')['PartyAb'].apply(set)

# Filter to those containing both target parties
divs_with_both = party_sets[party_sets.apply(lambda x: parties.issubset(x))].index.tolist()

print(divs_with_both)

import pdb;pdb.set_trace()
