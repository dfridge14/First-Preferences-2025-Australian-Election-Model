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

Multiple_INDs_2025 = pd.DataFrame(duplicates.reset_index()).drop('PartyAb', axis = 1).rename(columns={'count':'No_of_INDs'})
#Multiple_INDs_2025.to_csv("2025_Multiple_INDs_divs.csv", index=False)

parties = {'LP','NP'}
party_sets = Candidates_2025.groupby('div_nm')['PartyAb'].apply(set)

# Filter to those containing both target parties
divs_with_both = party_sets[party_sets.apply(lambda x: parties.issubset(x))].index.tolist()

print(divs_with_both)

election_year = '2025'

for election_year in ['2016','2019','2022','2025']:
    Elec_order = pd.read_csv(f"Electorate_Correlation_Matrix_{election_year}.csv").iloc[:,:1]
    Elec_order.columns = ["Electorate"]
    #num_electorates = len(Elec_order)

    Volatility_curr = pd.read_csv("Volatility_Category_df.csv", index_col = None)
    Volatility_curr = Volatility_curr.loc[Volatility_curr['Election'] == int(election_year),['Electorate','Volatility_Cat']]
    Volatility_curr = Volatility_curr.loc[Volatility_curr['Volatility_Cat'] != 1000,] # remove already-allocated volatilities
    Volatility_curr['Volatility_Cat'] = Volatility_curr['Volatility_Cat'].replace({0:1})

    #sum_volatilities = Volatility_curr['Volatility_Cat'].sum()

    #remaining_volatilities = (num_electorates - sum_volatilities)/(num_electorates - len(Volatility_curr))

    Volatility_Cats = Elec_order.merge(Volatility_curr, on = 'Electorate', how = 'left').fillna(0).rename(columns={'Volatility_Cat':'Volatility_weights'})
    Volatility_Cats['Volatility_weights'] = Volatility_Cats['Volatility_weights'].astype(int)
    import pdb;pdb.set_trace()
    Volatility_Cats.to_csv(f"Volatility_weights_df_{election_year}.csv", index = False)



