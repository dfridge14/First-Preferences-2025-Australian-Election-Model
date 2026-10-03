import numpy as np
import pandas as pd
import os
from pathlib import Path


import re
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler
state = 'VIC'

# VIC

if state == 'VIC':

    VIC_ELECTION_YEARS = [1976,1979,1982,1985,1988,1992,1996,1999, 2002, 2006, 2010, 2014, 2018, 2022, 2026] # ['1976','1979','1982','1985','1988','1992','1996','1999','2002','2006','2010','2014','2018','2022','2026']
    TENURE_YEARS = [2006, 2010, 2014, 2018, 2022, 2026]

    # Wikipedia uses en-dashes (%E2%80%93) for year ranges in URLs
    terms = ["2002%E2%80%932006", "2006%E2%80%932010", "2010%E2%80%932014", "2014%E2%80%932018", "2018%E2%80%932022", "2022%E2%80%932026"]

    all_terms_df = []

    for term in terms:
        # Notice the comma before the year string in the URL
        url = f"https://en.wikipedia.org/wiki/Members_of_the_Victorian_Legislative_Assembly,_{term}"
        
        try:
            # We add storage_options to mimic a standard web browser
            dfs = pd.read_html(
                url, 
                match="Term in office" if term[:4] != '2022' else 'Years in office',
                storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            )
            df = dfs[0].copy()

            # special treatment for 2022-2026 page (maybe since it is live, not historical)
            if term[:4] == '2022':
                df = df.drop(columns=['Party']).rename(columns={'Party.1': 'Party', 'Years in office': 'Term in office'})

            df['Name'] = df['Name'].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.replace(r'\b(The |Hon |Dr |Sir )\b\.?', '', regex=True).str.strip()
            df['Electorate'] = df['Electorate'].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip()
            df['Party'] = df['Party'].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip()
            
            # 3. INJECT THE SOURCE TERM (This fixes your KeyError)
            # Slices the first 4 characters of the term (e.g., '2018' from '2018%E2%80%932022')
            df['source_term'] = int(term[:4])


            all_terms_df.append(df)
            print(f"Successfully scraped: {term}")
        except Exception as e:
            print(f"Failed to scrape {term}. Error: {e}")

    # 1. Combine all scraped terms into one giant dataframe
    master_df = pd.concat(all_terms_df, ignore_index=True)



    def parse_term_spans(term_string):
        """Converts a string like '1999-2014, 2018-present' into a list of (start, end) integers."""
        if pd.isna(term_string): return []
        
        # 1. Replace present with 2026
        s = str(term_string).lower().replace('present', '2026').replace('–', '-').replace(' ', '')
        spans = []
        
        for part in s.split(','):
            if not part: continue
            try:
                start_str, end_str = part.split('-')
                start = int(start_str)
                
                # Handle abbreviated end years like '1999-02'
                if len(end_str) == 2:
                    end = int(str(start)[:2] + end_str)
                else:
                    end = int(end_str)
                    
                spans.append((start, end))
            except ValueError:
                continue
        return spans

    def get_election_wins(spans):
        """Returns a sorted list of years the MP won a seat (general or by-election)."""
        wins = set()
        for start, end in spans:
            # They always win their start year (general or by-election)
            wins.add(start)
            # They also 'win' any general election that falls strictly within their span
            for y in VIC_ELECTION_YEARS:
                if start < y < end:
                    wins.add(y)
        return sorted(list(wins))

    def get_tenure_at_year(spans, target_year):
        """
        Returns tenure prior to the target year IF they are an incumbent going into it.
        Otherwise returns np.nan.
        """
        is_incumbent = False
        tenure = 0
        
        for start, end in spans:
            # Check if they are an incumbent going into this election
            if start < target_year and end >= target_year:
                is_incumbent = True
                
            # Add time served strictly before the target year
            if start < target_year:
                effective_end = min(end, target_year)
                tenure += (effective_end - start)
                
        return tenure if is_incumbent else np.nan

    # --- Main Processing ---

    processed_records = []

    # Group by Name to reconstruct individual histories
    for name, group in master_df.groupby('Name'):
        
        # Get all unique term strings and flatten them into definitive spans
        raw_terms = group['Term in office'].unique()
        all_spans = set()
        for term in raw_terms:
            all_spans.update(parse_term_spans(term))
        
        # Sort and merge overlapping/adjacent spans just in case of Wikipedia weirdness
        sorted_spans = sorted(list(all_spans))
        
        # Calculate all the years this MP won a seat
        election_years = get_election_wins(sorted_spans)
        
        # Create the electorate mapping
        # Build a dictionary of {election_year: electorate} based on the scraped source terms
        known_electorates = dict(zip(group['source_term'], group['Electorate']))
        
        if not known_electorates:
            continue
            
        # Find the earliest known electorate to back-fill pre-2002
        earliest_known_year = min(known_electorates.keys())
        earliest_electorate = known_electorates[earliest_known_year]
        
        div_list = []
        for y in election_years:
            if y < earliest_known_year:
                div_list.append(earliest_electorate) # Carry pre-2002 backwards
            else:
                # Find the closest known electorate for that year
                # (If a by-election happens in 2015, we use the 2014 term electorate)
                closest_term = max([k for k in known_electorates.keys() if k <= y], default=earliest_known_year)
                div_list.append(known_electorates[closest_term])
                
        # Compile the final record
        record = {
            'inc_name': name,
            'list_of_partyAb': list(group['Party'].unique()),
            'raw_term_strings': list(raw_terms),
            'election_years': election_years,
            'div_list': div_list
        }
        
        # Calculate the 6 tenure columns
        for ty in TENURE_YEARS:
            record[f'{ty}_tenure'] = get_tenure_at_year(sorted_spans, ty)
            
        processed_records.append(record)

    final_df = pd.DataFrame(processed_records)

    # adjustements for by-elections occurring in same year as state election
    final_df.loc[final_df['inc_name'] == 'Jill Hennessy', 'election_years'].iat[0].insert(0, 2010)
    final_df.loc[final_df['inc_name'] == 'Jill Hennessy', 'div_list'].iat[0].insert(0, final_df.loc[final_df['inc_name'] == 'Jill Hennessy', 'div_list'].iat[0][0])
    final_df.loc[final_df['inc_name'] == 'Bob Stensholt', 'election_years'].iat[0].insert(0, 1999)
    final_df.loc[final_df['inc_name'] == 'Bob Stensholt', 'div_list'].iat[0].insert(0, final_df.loc[final_df['inc_name'] == 'Bob Stensholt', 'div_list'].iat[0][0])

    # Nepean 2026 by-election fix
    final_df.loc[final_df['inc_name'] == 'Anthony Marsh','2026_tenure'] = 0.0

    # Tally Room data on incumbents contesting
    for year in TENURE_YEARS:
        try:
            df = pd.read_csv(f'~/Australian Election/Victorian Election/VIC-{year}/VIC-{year}-LA-Candidates-Formatted.csv')
        except:
            df = pd.read_csv(f'~/Australian Election/Victorian Election/VIC-{year}/VIC-{year}-Incumbents.csv') # for upcoming election - VIC-2026
            

        contesting_incs = df.loc[df['incumbent']==1,['div_nm','surname','first_name','PartyAb','incumbent']]

        

        # convert name to ordinary name form in final_df, ensure doubly-capitalised surnames, e.g. McM, are preserved
        contesting_incs['inc_name'] = (contesting_incs['first_name'] + " " + contesting_incs['surname'].str.title().str.replace(r'\b(Mc|Mac)([a-z])', lambda m: m.group(1) + m.group(2).upper(), regex=True))
        # remove LC incumbents: Nina Taylor in 2022
        if year == 2022:
            contesting_incs = contesting_incs.loc[~(contesting_incs['div_nm']=='Albert Park'),]
        if year == 2026:
            contesting_incs.loc[contesting_incs['inc_name'] == 'Gabrielle De Vietri','inc_name'] = 'Gabrielle de Vietri'

        missing_names = set(contesting_incs['inc_name']) - set(final_df['inc_name'])
        if len(missing_names) > 0:
            raise ValueError("There are incumbents unaccounted for")
        
        # allocate year_tenure value only to incumbents that re-contest using contesting_incs data
        tenure_col = f"{year}_tenure"
        active_this_year = contesting_incs['inc_name'].unique()
        is_active = final_df['inc_name'].isin(active_this_year)
        final_df[tenure_col] = np.where(is_active, final_df[tenure_col], np.nan)

    
    os.chdir(Path.home() / 'Australian Election/Victorian Election')
    final_df.to_csv('VIC-Incumbents-df-2006-2026.csv', index=False)
    import pdb; pdb.set_trace()