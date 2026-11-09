import pandas as pd
import numpy as np
import os
from pathlib import Path



data_year = '2010'
state = 'VIC'

tally_room_dir = Path.home() / f"Australian Election/Victorian Election/Tally Room/{state}-{data_year}"
main_dir =  Path.home() / f"Australian Election/Victorian Election/{state}-{data_year}"


if state == 'VIC':

    # CHECK: any adjustment for data_year != '2022'???



    # rename district_name -> div_nm, party_code -> PartyAb
    party_df = pd.read_csv(main_dir / f'VIC-{data_year}-Parties.csv')
    party_mapping = party_df.set_index('party_code')['PartyAb']

    file_map = {        
        'LA_Candidates': tally_room_dir / f'VIC-{data_year}-LA-Candidates.xlsx',
        'LC_Candidates': tally_room_dir / f'VIC-{data_year}-LC-Candidates.xlsx',
        'LA_results': tally_room_dir / f'VIC-{data_year}-LA-Primary-Electorate.xlsx',
        'LC_results': tally_room_dir / f'VIC-{data_year}-LC-Votes-Electorate.xlsx'
    }

    for name, file_path in file_map.items():
        new_file_path =  main_dir / f"{file_path.stem}-Formatted.csv"

        if not new_file_path.exists():

            df = pd.read_excel(file_path)
            
            # Modify in place and map the standardized values
            df.rename(columns={'district_name': 'div_nm', 'party_code': 'PartyAb'}, inplace=True)
            df['PartyAb'] = df['PartyAb'].map(party_mapping)

            # adjust PartyAb of IND to INDX based on ballot order
            if 'LA' in name:
                is_ind = df['PartyAb'] == 'IND'
                df.loc[is_ind, 'PartyAb'] = ('IND' + (df[is_ind].groupby('div_nm').cumcount() + 1).astype(str))
            import pdb; pdb.set_trace()
            df.to_csv(new_file_path, index=False)
            print(f"Created: {new_file_path.name}")

        else:
            print(f"Already exists, skipping: {new_file_path.name}")
        
        # Save to csv
        import pdb; pdb.set_trace()
        

    # TODO for 2006, combine polling place votes into Electorates-Votes...? 

            

# for pre-2022 add 2CP results

for data_year in [2006,2010,2014,2018]:
    filepath = Path.home() / f"Australian Election/Victorian Election/Tally Room/{state}-{data_year}/{state}-{data_year}-LA-2CP-Electorate.xlsx"
    main_dir =  Path.home() / f"Australian Election/Victorian Election/{state}-{data_year}"
    new_file_path =  main_dir / f"{filepath.stem}-Formatted.csv"
    df = pd.read_excel(filepath)

    df.rename(columns={'district_name': 'div_nm', 'party_code': 'PartyAb'}, inplace=True)
    df['PartyAb'] = df['PartyAb'].map(party_mapping)

    df.to_csv(new_file_path, index=False)
    print(f"Created: {new_file_path.name}")
        
    import pdb; pdb.set_trace()