import pandas as pd
import numpy as np
import os, time
from pathlib import Path
from collections import Counter
import re
import pickle

# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler

data_year = '2022'

base_dir = Path.home() / f"Australian Election/Victorian Election/VIC-{data_year}"
os.chdir(base_dir)



LC_results = pd.read_csv('VIC-2022-LC-Votes-Electorate-Formatted.csv')
LA_results = pd.read_csv('VIC-2022-LA-Primary-Electorate-Formatted.csv')

LA_Candidates = pd.read_csv('VIC-2022-LA-Candidates-Formatted.csv')

# create df matching party code to party name
party_df = pd.read_csv('VIC-2022-Parties.csv')
Ideology_map = party_df.set_index('PartyAb')['Ideo_category']
IDEO_CATEGORIES = ['ALP','COAL','Left','Right','Centre']


if 0:
    div = 'Yan Yan'

    div_LC_results = LC_results.loc[LC_results['div_nm'] == 'Yan Yean',]

    # get total group votes using group (instead of party) to account for different COALition party names under same group
    div_LC_results['group_votes'] = div_LC_results['group_code'].map(div_LC_results.groupby('group_code')['votes'].sum())

    assert div_LC_results['votes'].sum() == div_LC_results.loc[div_LC_results['first_name'].isin(['Above-the-line','Informal']),'group_votes'].sum() # no vote totals lost

    # series of UH votes per party
    LC_party_votes = div_LC_results.loc[div_LC_results['first_name']=='Above-the-line',['PartyAb','group_votes']].sort_values(by='group_votes', ascending = False).set_index('PartyAb')

    LH_UH_comp = LC_party_votes.div(LC_party_votes.sum(), axis=1)

if 0: # obsolete as have access to full DOP table
    
    div_LA_results =  LA_results.loc[LA_results['div_nm'] == 'Yan Yean',]

    # rename LP/NP to LNP
    div_LA_results['PartyAb'] = div_LA_results['PartyAb'].replace({'LP': 'LNP','NP':'LNP'})

    LA_party_votes = div_LA_results[['PartyAb','votes']].set_index('PartyAb')

    # get comparisons
    LC_party_votes['LA_votes'] = LA_party_votes.reindex(LC_party_votes.index).to_numpy()
    #LH_UH_comp['difference'] = LH_UH_comp['LA_votes'] - LH_UH_comp['group_votes']



def extract_DOP_table(LA_Candidates, div_list):

    import requests
    import io


    DOP_table_dict = {}

    # Provide a standard browser User-Agent so the VEC doesn't block the scraper
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
    url_template = "https://www.vec.vic.gov.au/results/state-election-results/2022-state-election-results/results-by-district/{slug}-district-results/{slug}-results-distribution"

    # Hardcode fixes for any VEC URL typos
    url_overrides = {'Pakenham': "https://www.vec.vic.gov.au/results/state-election-results/2022-state-election-results/results-by-district/pakenham-district-results/pakenham-results--distribution"}


    for div in div_list:

        
        # read existing downloads if available
        if Path('/home/dania-freidgeim/Australian Election/Victorian Election/VIC-2022/DOP', f'2022_DOP_{div}.csv').exists():

            DOP_table_dict[div]  = pd.read_csv(f'DOP/2022_DOP_{div}.csv')

        elif Path('/home/dania-freidgeim/Australian Election/Victorian Election/VIC-2022/vec_downloads', f'2022_Indicative_DOP_{div}.csv').exists():
            continue

        else:
            # 1. Use the override URL if one exists for this district, otherwise use the template
            if div in url_overrides:
                url = url_overrides[div]
            else:
                slug = div.lower().replace(" ", "-")
                url = url_template.format(slug=slug)
            
            try:
                # 2. Fetch the page using requests with a disguised User-Agent
                response = requests.get(url, headers=headers, timeout=10)
                # This will force the script to jump to the 'except' block if it hits a 404 or 403
                response.raise_for_status() 
                # 3. Pass the raw HTML text to pandas instead of the URL
                tables = pd.read_html(io.StringIO(response.text))
            
                if not tables:
                    raise ValueError(f"No <table> elements found at {url}")
                
                # Assuming the preference distribution is the first table
                DOP_table = tables[0]                
                
                print(f"Successfully extracted: {url}")
                
            except Exception as e:
                print(f"Failed to extract {url}: {e}")
                if div in ['Niddrie','Pakenham']:
                    import pdb; pdb.set_trace()
                continue


            

            # Instead of CountNumber, have n_remaining (helps with comparison to 22)
            # map candidate names to PartyAb (IND1+2) using Ben's Candidates repository

            DOP_table = DOP_table.rename(columns={'Unnamed: 0': 'CandRemaining'})
            DOP_table['CandRemaining'] = len(DOP_table.columns) - 2 - (DOP_table.index + 1) // 2 # increment every 2
            DOP_table['CountType'] = np.where(DOP_table.index % 2 == 0, 'Votes', 'TransferredVotes') # 'Votes' or 'TransferredVotes'
            DOP_table.insert(0, 'div_nm', div)

            # 4. Replace candidate names in columns with their PartyAb (PartyAb)
            # Concatenate surname and first_name in LA_Candidates to match the "SURNAME, Firstname" format in DOP_table
            LA_Candidates_div = LA_Candidates.loc[LA_Candidates['div_nm'] == div,]

            # 6. Create the match_name and mapping dictionary using the updated dataframe

            # dictionary mapping full candidate names to their party codes
            party_mapping = dict(zip(LA_Candidates_div['candidate_name'], LA_Candidates_div['PartyAb']))

            # Apply the mapping to rename the matching columns in DOP_table
            DOP_table = DOP_table.rename(columns=party_mapping)

            

            # create PrefPercent table & separate transfer % table?
            DOP_pref_percent = DOP_table.copy()
            #DOP_pref_percent.iloc[:,2:-2] = DOP_table.iloc[:,2:-2].div(DOP_table['TOTAL'], axis=0) 


            DOP_table_dict[div] = DOP_table

            out_filename = f"{data_year}_DOP_{div}.csv"
            out_path = os.path.join('DOP', out_filename)
            DOP_table.to_csv(out_path, index=False)

    return DOP_table_dict
    # 


def download_indicative_preferences(data_year):
    import os
    import requests
    import pandas as pd
    import numpy as np
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    from concurrent.futures import ThreadPoolExecutor

    """ Downloads indicative preference files form the VEC website, formatting them into usable DOP tables"""

    if Path('/home/dania-freidgeim/Australian Election/Victorian Election/VIC-2022/vec_downloads', '2022_Indicative_DOP_Bendigo East.csv').exists():

        DOP_indicative_table_dict = {}
        csv_files = list(Path("vec_downloads").glob("*.csv"))
        dfs = {file.stem: pd.read_csv(file) for file in csv_files}

        for df in dfs.keys():
            div = df.split("DOP_")[1].replace(".csv", "")
            DOP_indicative_table_dict[div] = dfs[df]

        return DOP_indicative_table_dict


    URL = "https://www.vec.vic.gov.au/results/electoral-statistics/state-election-statistics/full-preference-distributions"
    OUTPUT_DIR = "vec_downloads"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def process_file(link):
        file_url = urljoin(URL, link['href'])
        filename = file_url.split('/')[-1]
        raw_path = os.path.join(OUTPUT_DIR, filename)
        
        # 1. Download the file
        response = requests.get(file_url)
        with open(raw_path, 'wb') as f:
            f.write(response.content)

        # 2. Convert .xls to .xlsx if necessary
        if raw_path.endswith('.xls'):
            xlsx_path = raw_path + 'x'
            # Read raw without headers to perfectly preserve the layout for your skiprows=12 rule
            temp_df = pd.read_excel(raw_path, engine='xlrd', header=None) 
            temp_df.to_excel(xlsx_path, index=False, header=False)
            os.remove(raw_path) # Clean up the old .xls
        else:
            xlsx_path = raw_path

        # 3. Extract the district name and read the main table
        try:
            header_df = pd.read_excel(xlsx_path, nrows=5, header=None)
            
            row_4_data = header_df.iloc[3].dropna()
            if not row_4_data.empty:
                raw_district = str(row_4_data.iloc[0]).strip()
                div = raw_district.replace(' District', '').strip()
            else:
                div = filename.split('.')[0]

            # Use 'DOP_table' to match your naming convention
            DOP_table = pd.read_excel(xlsx_path, skiprows=12)
            
            # delete any accidental full-nan cols
            # delete \n in candidate names
            # get rid of all final rows before FINAL TOTAL
            
            # A. Delete accidental full-nan cols
            DOP_table = DOP_table.dropna(axis=1, how='all')
            
            # B. Clean strings: remove \n 
            DOP_table.columns = DOP_table.columns.str.replace(r'\n', ' ', regex=True)
            DOP_table = DOP_table.replace(r'\n', ' ', regex=True)
            
            # C. Fix comma spacing: replace a comma followed by ANY number of spaces with just ", "
            DOP_table.columns = DOP_table.columns.str.replace(r',\s+', ', ', regex=True)
            # Strip trailing whitespace just in case
            DOP_table.columns = DOP_table.columns.str.strip() 
            
            # D. Get rid of all final rows AFTER 'FINAL TOTAL'
            mask = DOP_table.astype(str).apply(lambda x: x.str.contains('FINAL TOTAL', case=False, na=False)).any(axis=1)
            if mask.any():
                final_total_idx = mask.idxmax()
                DOP_table = DOP_table.loc[:final_total_idx].copy() # .copy() prevents warnings later
                
            # --- CLEANING PHASE 2: Your Custom Standard Fixes ---
            
            # 1. Rename unnamed column and calculate n_remaining (CandRemaining)
            DOP_table = DOP_table.rename(columns={'Candidates Names (in ballot paper order)': 'CandRemaining'})
            DOP_table['CandRemaining'] = len(DOP_table.columns) - 2 - (DOP_table.index + 1) // 2 
            
            # 2. Assign CountType based on row index parity
            DOP_table['CountType'] = np.where(DOP_table.index % 2 == 0, 'Votes', 'TransferredVotes')
            
            # 3. Insert District Name at the front
            DOP_table.insert(0, 'div_nm', div)
            
            # --- CLEANING PHASE 3: Candidate Mapping ---
            
            # 4. Filter Ben's Candidates repository for this division
            # Using .copy() here is crucial to avoid a SettingWithCopyWarning when updating INDs
            LA_Candidates_div = LA_Candidates.loc[LA_Candidates['div_nm'] == div].copy()

          
            # 6. Create the match_name and mapping dictionary
            party_mapping = dict(zip(LA_Candidates_div['candidate_name'], LA_Candidates_div['PartyAb']))

            # Apply the mapping to rename the matching columns in DOP_table
            DOP_table = DOP_table.rename(columns=party_mapping)
            
            # --- PHASE 4: Save to Disk ---
            
            # Assuming data_year is defined globally (e.g., data_year = 2022)
            out_filename = f"{data_year}_Indicative_DOP_{div}.csv"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            DOP_table.to_csv(out_path, index=False)
                
            return div, DOP_table

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return filename, None

    # Scrape the page for all Excel links
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, 'html.parser')
    excel_links = soup.find_all('a', href=lambda href: href and (href.endswith('.xls') or href.endswith('.xlsx')))

    # Use ThreadPoolExecutor to download and process files in parallel (fastest method)
    DOP_indicative_table_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_file, excel_links)
        
        for filename, df in results:
            if df is not None:
                DOP_indicative_table_dict[filename] = df
                print(f"Successfully processed {filename}")

    return DOP_indicative_table_dict



# obtain DOP tables

div_list = LA_Candidates['div_nm'].unique().tolist()
DOP_table_dict = extract_DOP_table(LA_Candidates, div_list) # div_list
DOP_indicative_table_dict = download_indicative_preferences(data_year)




# Generate full distirbution table of votes (Yan Yean specific data)

if 0:
    DOP_table = DOP_table_dict['Yan Yean']
    votes_df = DOP_table[DOP_table['CountType'] == 'Votes'].copy()

    meta_cols = {'div_nm', 'CandRemaining', 'TOTAL', 'CountType'}
    party_cols = [col for col in votes_df.columns if col not in meta_cols]

    # 3. Optional: Map coalition names if LH_UH_comp uses 'LNP' instead of 'LP/NP'
    coalition_map = {'LP': 'LNP', 'NP': 'LNP'}
    votes_df = votes_df.rename(columns=coalition_map)
    party_cols = [coalition_map.get(col, col) for col in party_cols]

    # 4. Scale candidate vote columns by TOTAL to get proportions (matching LA_votes format)
    votes_df[party_cols] = votes_df[party_cols].div(votes_df['TOTAL'], axis=0)

    # 5. Reshape into (PartyAb x CandRemaining) structure
    # Set CandRemaining as index, slice party columns, transpose
    counts_by_cand = (votes_df.set_index('CandRemaining')[party_cols].T.rename_axis('PartyAb'))
    # Rename columns to clearly describe the count stages (e.g., 'votes_9_cand', 'votes_8_cand', ...)
    counts_by_cand.columns = [f"votes_{c}_cand" for c in counts_by_cand.columns]
    LH_UH_comp = LH_UH_comp.join(counts_by_cand, how='left')

    #print(LH_UH_comp)



# save to file
file = Path(f'{data_year}_DOP_table_dict.pkl')

if not file.exists():
    with open(file, 'wb') as f:
        pickle.dump(DOP_table_dict, f)





# compare the primary votes formally with the indicative DOP table - 100 threshold

for div in DOP_indicative_table_dict.keys():

    indic_FPs = DOP_indicative_table_dict[div].iloc[0,2:-1].T
    indic_FPs = indic_FPs.rename({'LP': 'LNP', 'NP': 'LNP'})

    # get primary votes in this div
    div_LA_results =  LA_results.loc[LA_results['div_nm'] == div,]

    div_LA_results['PartyAb'] = div_LA_results['PartyAb'].replace({'LP': 'LNP','NP':'LNP'})
    LA_party_votes = div_LA_results[['PartyAb','votes']].set_index('PartyAb')

    formal_FPs = LA_party_votes.loc[LA_party_votes.index != 'INFORMAL',]
    formal_FPs.loc['TOTAL'] = formal_FPs['votes'].sum()

    max_deviation = int((formal_FPs['votes'] - indic_FPs).max())

    if max_deviation > 10:
        #print('nontrivial vote deviation:', div, max_deviation)
        1


    # assume indicative DOP as the true DOP
    DOP_table_dict[div] = DOP_indicative_table_dict[div]




# check for missing divs
if len([div for div in div_list if div not in DOP_table_dict.keys()]) != 0:
    print([div for div in div_list if div not in DOP_table_dict.keys()])


# save complete indicative DOP table dict
file = Path(f'{data_year}_complete_DOP_table_dict.pkl')

if not file.exists():
    with open(file, 'wb') as f:
        pickle.dump(DOP_table_dict, f)



# top 4 in each electorate
if data_year == '2022':
    LA_Candidates.loc[LA_Candidates['candidate_name']=='TAYLOR, Nina','incumbent'] = 0 # remove Nina Taylor
seats_with_inc = LA_Candidates.loc[LA_Candidates['incumbent']==1,'div_nm'].unique().tolist()
incumbent_parties = LA_Candidates.loc[LA_Candidates['incumbent'] == 1].set_index('div_nm')['PartyAb']
open_seats = [div for div in div_list if div not in seats_with_inc]

electorate_tallies = []
top_x = 3
COAL_double_divs = []

for div in div_list: # open_seats div_list
    if (data_year == '2022') and (div == 'Narracan'):
        continue
    else:
        DOP_curr = DOP_table_dict[div]
        
        x = top_x

        top_x_parties =  DOP_curr[(DOP_curr['CandRemaining'] <= top_x)].iloc[:, 2:-2].dropna(axis=1, how='all').columns.tolist()
        if 'LP' in top_x_parties and 'NP' in top_x_parties:
            x = top_x + 1
            COAL_double_divs.append(div)

        topx_rows = DOP_curr[(DOP_curr['CandRemaining'] <= x) & (DOP_curr['CountType'] == 'Votes')]
        party_cols = topx_rows.iloc[:, 2:-2] # isolate party cols
        # 3. Count how many times each party appears (is non-NaN) in these 3 rows
        presence = party_cols.count().sort_values(ascending=False)

        # 4. Filter out any parties eliminated before Top 4 (count == 0) and get the final list
        ordered_top_x = presence[presence > 0].index.tolist()

        #print(ordered_top_x, div)

        ideo_list = [Ideology_map.get(re.sub(r'\d+$', '', party), 'Unknown') for party in ordered_top_x]
        counts = Counter(ideo_list)
        if x == top_x + 1:
            counts['COAL'] -= 1
        counts['div_nm'] = div  # Keep track of the electorate name
        electorate_tallies.append(counts)

        
        # get ideological combination

tally_df = pd.DataFrame(electorate_tallies).set_index('div_nm').fillna(0).astype(int)
for cat in IDEO_CATEGORIES:
    if cat not in tally_df.columns:
        tally_df[cat] = 0
tally_df = tally_df[IDEO_CATEGORIES]

combination_counts = tally_df.value_counts().reset_index(name='Electorate_Count')

print(combination_counts)

import pdb; pdb.set_trace()





# Right: ON, UAPP, SFF, LDP, FFP, SDA, FPV, CAP, AVP
# Centre: IND, SPP, TMP, ND, HAP, DHJP, DLP       DLP, TMP, ND, HAP, DHJP
# Left: LGC(HMP), VS, AJP, RP(REAS)




# create lh and uh dfs for estimating UH to 3CP flows and IA

def merge_coalition_preferences_old(df):
    """
    Merges 'LP' and 'NP' into a single 'COAL' entity in a DOP DataFrame,
    adjusts the candidate counts, and drops the redundant exclusion row.
    """
    # Create a copy to prevent SettingWithCopy warnings
    out_df = df.copy()
    
    # 1. Verify both parties are in the dataframe
    has_lp = 'LP' in out_df.columns
    has_np = 'NP' in out_df.columns
    
    if has_lp and has_np:

        # check the 3CP condition to trigger pdb
        row_3cp = out_df[out_df['CandRemaining'] == 3]
        
        lp_alive_at_3cp = pd.notna(row_3cp.iloc[0]['LP'])
        np_alive_at_3cp = pd.notna(row_3cp.iloc[0]['NP'])
            
        if not lp_alive_at_3cp + np_alive_at_3cp:
            print(f"DEBUG: a COAL party missed the 3CP in {out_df.iloc[0]['div_nm']}.")
            import pdb; pdb.set_trace()
                
        # combined COAL column, recalculate CandRemaining
        out_df['LNP'] = out_df[['LP', 'NP']].sum(axis=1, min_count=1)
        party_cols = [col for col in out_df.columns if col not in  ['div_nm', 'CandRemaining', 'TOTAL', 'CountType'] + ['LP', 'NP']]
        
        # Count all active (non-NaN), non-COAL parties left in the race
        out_df['CandRemaining'] = out_df[party_cols].notna().sum(axis=1)
        
        # 5. Drop the redundant internal exclusion row
        # When LP or NP is excluded, the new unified CandRemaining stalls (e.g., stays at 2 for two rows).
        # Keeping 'last' drops the row BEFORE the internal preferences flow, keeping the fully distributed totals.
        out_df = out_df.drop_duplicates(subset=['CandRemaining'], keep='last')
        
        # 6. Clean up and reorder columns
        out_df = out_df.drop(columns=['LP', 'NP'])
        
        party_cols.remove('LNP')
        final_cols = ['div_nm', 'CandRemaining', 'LNP'] + party_cols + ['TOTAL', 'CountType']
        out_df = out_df[final_cols]
        
    return out_df


def merge_coalition_preferences(df):
    """
    Swaps the elimination order of the 4CP and 3CP parties when both LP and NP 
    survive to 3CP. The trailing COAL party is artificially eliminated at 4CP, 
    and its empirical leakage is spread proportionally to non-COAL survivors, 
    creating a synthetic 3CP row while keeping LP and NP columns intact.
    """
    out_df = df.copy()

    # Cast vote columns to float to prevent LossySetitemError during fraction assignment
    vote_cols = [c for c in out_df.columns if c not in ['div_nm', 'CandRemaining', 'CountType']]
    for c in vote_cols:
        out_df[c] = out_df[c].astype(float)
    
    has_lp = 'LP' in out_df.columns
    has_np = 'NP' in out_df.columns
    
    if has_lp and has_np:
        
        row_3cp_df = out_df[out_df['CandRemaining'] == 3]
        
        if not row_3cp_df.empty:
            row_3cp = row_3cp_df.iloc[0]
            lp_alive_at_3cp = pd.notna(row_3cp.get('LP'))
            np_alive_at_3cp = pd.notna(row_3cp.get('NP'))
        else:
            lp_alive_at_3cp = False
            np_alive_at_3cp = False
            
        if lp_alive_at_3cp and np_alive_at_3cp:
            row_2cp_df = out_df[out_df['CandRemaining'] == 2]
            row_4cp_df = out_df[out_df['CandRemaining'] == 4]
            
            if not row_2cp_df.empty and not row_4cp_df.empty:
                row_2cp = row_2cp_df.iloc[0]
                row_3cp = row_3cp_df.iloc[0]
                row_4cp = row_4cp_df.iloc[0]
                
                # identify trailing coalition party
                lp_3cp = float(row_3cp['LP'])
                np_3cp = float(row_3cp['NP'])
                c_trail, c_lead = ('LP', 'NP') if lp_3cp < np_3cp else ('NP', 'LP')
                
                party_cols = [c for c in out_df.columns if c not in ['div_nm', 'CandRemaining', 'TOTAL', 'CountType']]
                non_coal_2cp = next((c for c in party_cols if c not in ['LP', 'NP'] and pd.notna(row_2cp.get(c))), None)
                
                if non_coal_2cp is not None and float(row_3cp[c_trail]) > 0:
                    leaked_votes = float(row_2cp[non_coal_2cp]) - float(row_3cp[non_coal_2cp])
                    leak_prop = max(0.0, min(1.0, leaked_votes / float(row_3cp[c_trail])))
                else:
                    leak_prop = 0.12
                    
                idx_3cp = row_3cp_df.index[0]
                new_row_3cp = row_4cp.copy()
                new_row_3cp['CandRemaining'] = 3
                new_row_3cp[c_trail] = np.nan
                
                v_trail_4cp = float(row_4cp[c_trail])
                v_leak = v_trail_4cp * leak_prop
                v_stick = v_trail_4cp - v_leak
                
                new_row_3cp[c_lead] = float(row_4cp[c_lead]) + v_stick
                
                active_non_coal_4cp = [c for c in party_cols if c not in ['LP', 'NP'] and pd.notna(row_4cp.get(c)) and float(row_4cp.get(c)) > 0]
                sum_non_coal_4cp = sum(float(row_4cp[c]) for c in active_non_coal_4cp)

                allocated = 0
                if sum_non_coal_4cp > 0:
                    for c in active_non_coal_4cp:
                        val = int(round(float(row_4cp[c]) + (v_leak * (float(row_4cp[c]) / sum_non_coal_4cp))))
                        new_row_3cp[c] = val
                        allocated += val
                        
                # Exact remainder goes to Lead COAL to preserve TOTAL
                new_row_3cp[c_lead] = int(round(float(new_row_3cp['TOTAL']))) - allocated

                # Overwrite the historical 3CP row with the synthetic 3CP row
                out_df.loc[idx_3cp] = new_row_3cp

                import pdb; pdb.set_trace()

LH_DOP_dict = {}

for div in DOP_table_dict.keys():
    df = DOP_table_dict[div]
    df = df.loc[df['CountType']=='Votes',]

    LH_DOP_dict[div] = merge_coalition_preferences_old(df)



def get_LC_FPs(LC_results, div_list):

    LC_FPs_dict = {}

    for div in div_list:

        div_LC_results = LC_results.loc[LC_results['div_nm'] == div].copy()
        if div_LC_results.empty:
            continue
        # Get total group votes using group to account for different COALition party names
        div_LC_results['group_votes'] = div_LC_results['group_code'].map(div_LC_results.groupby('group_code')['votes'].sum())
        assert div_LC_results['votes'].sum() == div_LC_results.loc[(div_LC_results['first_name'].isin(['Above-the-line', 'Informal'])) | (div_LC_results['group_code']=='UG'), 'group_votes'].sum() # ensure no vote totals are lost; either ATL or UG candidates
        # Series of UH FP votes per party
        LC_party_votes = div_LC_results.loc[div_LC_results['first_name'] == 'Above-the-line', ['PartyAb', 'group_votes']].sort_values(by='group_votes', ascending=False).set_index('PartyAb')

        LC_party_votes = LC_party_votes.groupby(level=0).sum() # group IND/UG together

        LC_FPs_dict[div] = (LC_party_votes/LC_party_votes.sum())['group_votes'] # .rename(columns = {'group_votes': div})
        
    

    import pdb; pdb.set_trace()

    return pd.concat(LC_FPs_dict, axis=1)


LC_FPs_df =  get_LC_FPs(LC_results, div_list)


import pdb; pdb.set_trace()

def build_3CP_model_dataset(data_year, LH_DOP_dict, LC_results, LA_candidates, Ideology_map, open_seats, incumbent_parties, incumbency_df, target_cp=3):
    """
    Builds the complete DataFrame for the ALS model, merging LH preferences 
    and dynamically extracting UH first preferences.
    """
    # Filter LH data for the target count stage (e.g., 3CP)
    
    meta_cols = {'div_nm', 'CandRemaining', 'TOTAL', 'CountType'}
    rows = []
    
    for div, df in LH_DOP_dict.items():

        if div == 'Narracan' and data_year == '2022':
            continue

        # Filter this specific electorate's dataframe for the 3CP stage
        lh_cp = df[df['CandRemaining'] == target_cp]
        
        if lh_cp.empty:
            import pdb; pdb.set_trace()
            continue
            
        # Extract the single row for the 3CP count
        row = lh_cp.iloc[0] 
        total_lh = row['TOTAL']
        
        # Extract LH 3CP vote shares; Identify the 3CP minor party (Left, Right, Centre)
        lh_shares = {'ALP': 0.0, 'COAL': 0.0, 'Minor': 0.0}
        minor_party = None
        minor_ideo = 'Centre' # Fallback
        
        for col in df.columns:
            if col not in meta_cols and pd.notna(row[col]) and row[col] > 0:
                ideo = Ideology_map.get(re.sub(r'\d+$', '', col), 'Other')
                share = row[col] / total_lh
                
                if ideo in ['ALP', 'COAL']:
                    lh_shares[ideo] += share
                else:
                    lh_shares['Minor'] += share
                    minor_party = col
                    minor_ideo = ideo
                    
        # Extract Upper House (LC) FP vote
        div_LC_results = LC_results.loc[LC_results['div_nm'] == div].copy()
        if div_LC_results.empty:
            continue
            
        # Get total group votes using group to account for different COALition party names
        div_LC_results['group_votes'] = div_LC_results['group_code'].map(div_LC_results.groupby('group_code')['votes'].sum())
        assert div_LC_results['votes'].sum() == div_LC_results.loc[(div_LC_results['first_name'].isin(['Above-the-line', 'Informal'])) | (div_LC_results['group_code']=='UG'), 'group_votes'].sum() # ensure no vote totals are lost; either ATL or UG candidates
        # Series of UH FP votes per party
        LC_party_votes = div_LC_results.loc[div_LC_results['first_name'] == 'Above-the-line', ['PartyAb', 'group_votes']].sort_values(by='group_votes', ascending=False).set_index('PartyAb')
        
        total_uh = LC_party_votes['group_votes'].sum()
        if total_uh == 0:
            continue
            
        # Aggregate UH shares by Ideology / 3CP Minor
        uh_shares = {'ALP': 0.0, 'COAL': 0.0, 'Left': 0.0, 'Right': 0.0, 'Centre': 0.0}
        uh_3cp_minor_share = 0.0
        
        for p, p_row in LC_party_votes.iterrows():
            share = p_row['group_votes'] / total_uh
            ideo = Ideology_map.get(re.sub(r'\d+$', '', p), 'Other')
            
            if p == minor_party:
                uh_3cp_minor_share += share
            else:
                uh_shares[ideo] += share

        # WEAKNESS: assumes incumbent is in 3cp and, if IND, if the only IND
        if div in open_seats:
            incumbent_party = None
        else:
            if isinstance(incumbent_parties[div], pd.Series):
                # multiple incumbents
                continue
            incumbent_party = Ideology_map.get(re.sub(r'\d+$', '', incumbent_parties[div]), 'Other')


        # year served: get name of candidate for concordance with incumbency_df
        inc_names = LA_Candidates.loc[(LA_Candidates['div_nm']==div) & (LA_Candidates['incumbent']),['first_name','surname']]
        if not inc_names.empty:
            inc_name = (inc_names['first_name'] + " " + inc_names['surname'].str.title().str.replace(r'\b(Mc|Mac)([a-z])', lambda m: m.group(1) + m.group(2).upper(), regex=True)).iloc[0]
            years_served = incumbency_df.loc[(incumbency_df[f'{data_year}_tenure'].notna()) & (incumbency_df['inc_name'] == inc_name), f'{data_year}_tenure']
            years_served = years_served.iloc[0] if not years_served.empty else None
        else:
            years_served = None
                
        rows.append({
            'electorate': div,
            'topology': f"ALP_COAL_{minor_ideo}",
            'is_open': div in open_seats,
            'incumbent_party': incumbent_party,   
            'years_served': years_served,  
            'demographic_region': None, # To be populated externally later
            
            # UH results
            'uh_alp': uh_shares['ALP'],
            'uh_coal': uh_shares['COAL'],
            'uh_3cp_minor': uh_3cp_minor_share,
            'uh_left_minor': uh_shares['Left'],
            'uh_right_minor': uh_shares['Right'],
            'uh_other_minor': uh_shares['Centre'],
            
            # LH results
            'lh_alp': lh_shares['ALP'],
            'lh_coal': lh_shares['COAL'],
            'lh_minor': lh_shares['Minor']
        })
        
    return pd.DataFrame(rows).set_index('electorate')

incumbency_df = pd.read_csv(Path.home() / 'Australian Election/Victorian Election/VIC-Incumbents-df-2006-2026.csv')
df_3cp_model = build_3CP_model_dataset(data_year, LH_DOP_dict, LC_results, LA_Candidates, Ideology_map, open_seats, incumbent_parties, incumbency_df, target_cp=3)


import pdb; pdb.set_trace()


# run optimisation

from scipy.optimize import minimize
import statsmodels.api as sm

def fit_transition_matrix(df_subset, dirichlet_alpha=1.1, penalty_weight=0.003):
    """
    Fits a 3x3 row-stochastic transition matrix W.
    Rows = Absent UH Groups: [Left, Right, Other]
    Cols = LH Destinations: [ALP, COAL, Minor]
    """
    if len(df_subset) == 0:
        return np.full((3, 3), 1/3) # Fallback if empty
        
    # Baseline UH votes for the parties actually on the 3CP ballot
    U_pres = df_subset[['uh_alp', 'uh_coal', 'uh_3cp_minor']].values
    
    # Missing UH votes that must be reallocated
    U_abs = df_subset[['uh_left_minor', 'uh_right_minor', 'uh_other_minor']].values
    
    # Target LH outcomes
    Y_actual = df_subset[['lh_alp_clean', 'lh_coal_clean', 'lh_minor_clean']].values

    gamma = penalty_weight * len(df_subset)

    # Objective: Minimize Sum of Squared Errors, with push away from 0
    def loss(w_flat):
        W = w_flat.reshape((3, 3))
        Y_pred = U_pres + U_abs @ W
        mse = np.sum((Y_actual - Y_pred) ** 2)
        
        # Dirichlet Log-Barrier (clip to prevent log(0) domain errors in SLSQP)
        W_safe = np.clip(W, 1e-6, 1.0)
        log_prior = -np.sum((dirichlet_alpha - 1.0) * np.log(W_safe))
        
        return mse + (gamma * log_prior)

    # Constraints: Every row of W must sum to 1.0 (100% of missing votes are distributed)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w[0:3]) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.sum(w[3:6]) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.sum(w[6:9]) - 1.0}
    ]
    bounds = [(0,1)] * 9

    w_init = np.full((3, 3), 1.0 / 3.0).flatten()
    res = minimize(loss, w_init, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x.reshape((3, 3))


def run_2stage_inference(df, max_iter=30, tol=1e-4, shrinkage_lambda=0.4):
    """
    Runs the Iterative Alternating Least Squares model on df_3cp_model.
    """
    # 0. Prep tracking variables
    df = df.copy()

    df = df[df['topology'] != 'ALP_COAL_Centre']
    
    # Initialize "clean" LH columns (starts identical to actual LH)
    df['lh_alp_clean'] = df['lh_alp']
    df['lh_coal_clean'] = df['lh_coal']
    df['lh_minor_clean'] = df['lh_minor']
    
    # Create tracking column for our calculated IA (alpha)
    df['alpha_ia'] = 0.0
    
    incumbent_mask = ~df['is_open'] & df['incumbent_party'].isin(['ALP', 'COAL','Left','Right']) # not IND
    
    # Topologies to evaluate 
    mask_left = df['topology'] == 'ALP_COAL_Left'
    mask_right = df['topology'] == 'ALP_COAL_Right'
    
    for iteration in range(max_iter):
        # ---------------------------------------------------------
        # STAGE 1: Fit Flow Matrices (W)
        # ---------------------------------------------------------
        # In iteration 0, fit strictly on open seats. In later iterations, 
        # fit on all seats (incumbent seats have been 'cleansed').
        if iteration == 0:
            fit_mask_left = mask_left & df['is_open']
            fit_mask_right = mask_right & df['is_open']
        else:
            fit_mask_left = mask_left
            fit_mask_right = mask_right
            
        W_left = fit_transition_matrix(df[fit_mask_left])
        W_right = fit_transition_matrix(df[fit_mask_right])

        print(W_left)
        print(W_right)
        
        # Predict 3CP for ALL seats based on new matrices
        Y_pred = np.zeros((len(df), 3)) # Cols: [ALP, COAL, Minor]
        
        # Apply Left Matrix
        U_pres_L = df.loc[mask_left, ['uh_alp', 'uh_coal', 'uh_3cp_minor']].values
        U_abs_L  = df.loc[mask_left, ['uh_left_minor', 'uh_right_minor', 'uh_other_minor']].values
        Y_pred[mask_left] = U_pres_L + U_abs_L @ W_left
        
        # Apply Right Matrix
        U_pres_R = df.loc[mask_right, ['uh_alp', 'uh_coal', 'uh_3cp_minor']].values
        U_abs_R  = df.loc[mask_right, ['uh_left_minor', 'uh_right_minor', 'uh_other_minor']].values
        Y_pred[mask_right] = U_pres_R + U_abs_R @ W_right
        
        # Create a temporary dataframe mapping predictions
        df_pred = pd.DataFrame(Y_pred, index=df.index, columns=['pred_alp', 'pred_coal', 'pred_minor'])
        
        # ---------------------------------------------------------
        # STAGE 2: Extract & Model Incumbency Advantage (IA)
        # ---------------------------------------------------------
        raw_residuals = []
        for idx, row in df[incumbent_mask].iterrows():
            inc_party = row['incumbent_party'] # 'ALP' or 'COAL' or 'Left' 
            inc_party = inc_party.lower() if inc_party != 'Left' else 'minor'
            
            actual_vote = row[f'lh_{inc_party}']
            pred_vote = df_pred.loc[idx, f'pred_{inc_party}']
            raw_residuals.append(actual_vote - pred_vote)


        # Build DataFrame for Regression
        reg_df = df[incumbent_mask].copy()
        reg_df['raw_residual'] = raw_residuals
        reg_df['is_coalition'] = (reg_df['incumbent_party'] == 'COAL').astype(float)
        reg_df['log_tenure'] = np.log1p(reg_df['years_served'].fillna(0).clip(lower=0))
        
        # Create dummy variables for demographic region
        if 'demographic_region' in reg_df.columns and reg_df['demographic_region'].notna().any():
            region_dummies = pd.get_dummies(reg_df['demographic_region'], drop_first=True, dtype=float)
            X = pd.concat([reg_df[['log_tenure', 'is_coalition']], region_dummies], axis=1)
        else:
            X = reg_df[['log_tenure', 'is_coalition']]
            
        X = sm.add_constant(X)
        y = reg_df['raw_residual']
        
        # Fit OLS structural model
        ia_model = sm.OLS(y, X).fit()
        
        # Structural prediction + Shrunk local shock
        structural_alpha = ia_model.predict(X)
        local_shock = y - structural_alpha
        shrunk_alpha = structural_alpha + (local_shock / (1.0 + shrinkage_lambda))
        
        # ---------------------------------------------------------
        # Check Convergence & Cleanse
        # ---------------------------------------------------------
        alpha_new = pd.Series(0.0, index=df.index)
        alpha_new.loc[incumbent_mask] = shrunk_alpha
        
        max_delta = np.max(np.abs(alpha_new - df['alpha_ia']))
        print(f"Iteration {iteration}: Max IA Change = {max_delta:.6f}")
        df['alpha_ia'] = alpha_new
        
        if max_delta < tol and iteration > 0:
            print("Converged!")
            break
            
        # Cleanse incumbent LH votes for the next loop
        for idx in df[incumbent_mask].index:
            ia = df.loc[idx, 'alpha_ia']
            inc_party = df.loc[idx, 'incumbent_party']
            
            # Reset to true LH
            df.loc[idx, 'lh_alp_clean'] = df.loc[idx, 'lh_alp']
            df.loc[idx, 'lh_coal_clean'] = df.loc[idx, 'lh_coal']
            df.loc[idx, 'lh_minor_clean'] = df.loc[idx, 'lh_minor']
            
            # Apply disgorgement: subtract from incumbent, distribute equally to challengers
            if inc_party == 'ALP':
                df.loc[idx, 'lh_alp_clean'] -= ia
                df.loc[idx, 'lh_coal_clean'] += ia / 2.0
                df.loc[idx, 'lh_minor_clean'] += ia / 2.0
            elif inc_party == 'COAL':
                df.loc[idx, 'lh_coal_clean'] -= ia
                df.loc[idx, 'lh_alp_clean'] += ia / 2.0
                df.loc[idx, 'lh_minor_clean'] += ia / 2.0
            elif inc_party == 'Left':
                df.loc[idx, 'lh_minor_clean'] -= ia
                df.loc[idx, 'lh_alp_clean'] += ia / 2.0
                df.loc[idx, 'lh_coal_clean'] += ia / 2.0

    import pdb; pdb.set_trace()
    return df, W_left, W_right, ia_model

# Run it:
final_df, W_left, W_right, ia_model = run_2stage_inference(df_3cp_model)
print(ia_model.summary())








def generate_full_ia_adjustments(
    master_df: pd.DataFrame,
    final_df: pd.DataFrame,
    ia_model,
    dop_table_dict: dict,
    output_path: str = "ia_adjustments_dict.pkl",
) -> dict:
    """Generates 3CP party adjustments for all incumbent seats (participating + non-participating).

    Splits the IA disgorgement equally (50/50) between the two 3CP challengers.
    """
    master_df = master_df.copy()

    # Ensure electorate is accessible as index
    if "electorate" in master_df.columns:
        master_df = master_df.set_index("electorate")
    if "electorate" in final_df.columns:
        final_df = final_df.set_index("electorate")

    # 1. Identify all incumbent seats across the entire state
    valid_parties = ["ALP", "COAL", "Left", "Right","Centre"]
    incumbent_all = master_df[(~master_df["is_open"]) & (master_df["incumbent_party"].isin(valid_parties))].copy()

    # 2. Extract or Predict alpha for every incumbent seat
    alphas = {}

    # Design matrix columns expected by ia_model
    model_exog_names = ia_model.model.exog_names

    for electorate, row in incumbent_all.iterrows():
        # Case A: Electorate was in final_df and has a shrunk_alpha
        if (electorate in final_df.index and abs(final_df.loc[electorate, "alpha_ia"]) > 1e-7):
            alphas[electorate] = float(final_df.loc[electorate, "alpha_ia"])

        # Case B: Electorate was excluded (e.g. IND in 3CP) -> Predict structural alpha
        else:
            log_tenure = np.log1p(max(0.0, float(row.get("years_served", 0.0))))
            is_coalition = 1.0 if row["incumbent_party"] == "COAL" else 0.0

            # Construct row aligned with OLS specification
            x_dict = {
                "const": 1.0,
                "log_tenure": log_tenure,
                "is_coalition": is_coalition,
            }

            # Populate demographic dummy variables if used in model
            region = row.get("demographic_region", None)
            for col in model_exog_names:
                if col not in x_dict:
                    # Matches dummy column name created by pd.get_dummies
                    x_dict[col] = (
                        1.0 if region is not None and col.endswith(str(region)) else 0.0
                    )

            x_vec = np.array([x_dict[col] for col in model_exog_names]).reshape(1, -1)
            pred_alpha = float(ia_model.predict(x_vec)[0])
            alphas[electorate] = pred_alpha

    # 3. Construct 50/50 3CP adjustments mapped to DOP table columns
    ia_adjustments = {}
    meta_cols = {"div_nm", "CandRemaining", "TOTAL", "CountType"}

    for electorate, alpha in alphas.items():
        if electorate not in dop_table_dict:
            print(f"[WARN] {electorate} missing from DOP_table_dict. Skipping.")
            continue

        table_df = dop_table_dict[electorate]

        # Locate 3CP 'Votes' row
        row_3cp = table_df[(table_df["CandRemaining"] == 3) & (table_df["CountType"] == "Votes")
                           ]
        if row_3cp.empty:
            print(f"[WARN] No 3CP Votes row found for {electorate}. Skipping.")
            continue

        party_cols = [c for c in table_df.columns if c not in meta_cols]
        active_3cp = [c for c in party_cols if pd.notna(row_3cp[c].values[0]) and float(row_3cp[c].values[0]) > 0]

        if len(active_3cp) != 3:
            print(f"[WARN] {electorate} has {len(active_3cp)} candidates at 3CP (expected 3). Skipping.")
            continue

        # Map incumbent party to ballot column
        inc_party = incumbent_all.loc[electorate, "incumbent_party"]
        inc_col = None

        if inc_party in active_3cp:
            inc_col = inc_party
        elif inc_party == "COAL":
            for c in ["LP", "NP", "LNP"]:
                if c in active_3cp:
                    inc_col = c
                    break
        elif inc_party == "Left":
            for c in ["GRN"]:
                if c in active_3cp:
                    inc_col = c
                    break
        elif inc_party == "Centre":
            inc_col = next((f'IND{i}' for i in range(1, 11) if f'IND{i}' in active_3cp),None)

        if inc_col is None:
            print(f"[WARN] Could not identify incumbent '{inc_party}' among 3CP columns {active_3cp} in {electorate}.")
            continue

        # The two challengers (including INDs if present)
        challengers = [c for c in active_3cp if c != inc_col]
        c1, c2 = challengers[0], challengers[1]

        ia_adjustments[electorate] = {
            inc_col: -alpha,
            c1: alpha / 2.0,
            c2: alpha / 2.0,
        }

    with open(output_path, "wb") as f:
        pickle.dump(ia_adjustments, f)

    print(
        f"Generated 50/50 IA adjustments for {len(ia_adjustments)} electorates."
    )
    return ia_adjustments

# build adjustments for ALL incumbent seats (including those with IND in 3CP)
ia_adjustments_dict = generate_full_ia_adjustments(
    master_df=df_3cp_model,  # Full dataset before topology filtering
    final_df=final_df,  # Converged output with shrunk alphas
    ia_model=ia_model,  # Fitted statsmodels OLS object
    dop_table_dict=DOP_table_dict,
    output_path="ia_adjustments_dict.pkl",
)

import pdb; pdb.set_trace()
