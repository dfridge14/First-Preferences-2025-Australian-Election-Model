
import pandas as pd
import re
from pathlib import Path

# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler



main_dir =  Path.home() / "Australian Election/Victorian Election/VIC-2026"

def clean_ind_name(raw_name: str) -> str:
    """
    Forces any name into the strict IND_First_Last convention.
    Drops middle words. Preserves non-space punctuation (e.g., J.D. or Tosh-Jake).
    """
    if pd.isna(raw_name) or str(raw_name).strip() == 'nan':
        return "UNKNOWN_PARTY"
        
    # Strip 'IND_' if it somehow already exists
    name = re.sub(r'^IND_', '', str(raw_name).strip())
    
    # Split by spaces or underscores
    tokens = re.split(r'[\s_]+', name)
    
    # Take strictly the first and last tokens
    if len(tokens) > 1:
        return f"IND_{tokens[0].title()}_{tokens[-1].title()}"
    return f"IND_{tokens[0].title()}"

def extract_2026_candidates(raw_report_filepath, parties_csv_filepath, output_csv_filepath):
    # 1. Load the Parties Concordance
    df_parties = pd.read_csv(main_dir / parties_csv_filepath)
    party_map = dict(zip(df_parties['party_name'], df_parties['PartyAb']))
    
    # 2. Load the Raw Report
    # (Assuming it's saved as an Excel or CSV file from the VEC report)
    df_raw = pd.read_excel(main_dir / raw_report_filepath, header=None) 
    
    records = []
    current_div = None
    
    # 3. Parse the Report Line by Line
    for idx, row in df_raw.iterrows():
        # Assuming Candidate Name is in column 0 and RPP Name is in column 4 (index 4)
        # Adjust indices if your raw file layout differs slightly from the image
        col1 = str(row[0]).strip()
        
        # Guard clause: Stop processing if we hit the Upper House Regions
        if "Region" in col1:
            break
            
        # Detect new district (e.g., "Albert Park District")
        if "District" in col1:
            current_div = col1.replace(" District", "").strip()
            continue
            
        # Skip headers, metadata, and empty rows
        if not col1 or col1.startswith("(") or col1 == "Candidates" or col1.startswith("Total") or col1.startswith("State Election"):
            continue
            
        # If we have an active division and a valid candidate string, process the row
        if current_div and col1:
            candidate_raw = col1
            rpp_name = str(row[4]).strip() if pd.notna(row[4]) else ""
            
            # Format IND Name: IND_First_Last
            # VEC Raw format: SURNAME, Firstname Middle
            if "," in candidate_raw:
                parts = candidate_raw.split(",", 1)
                surname = parts[0].strip().title()
                # Flip "SURNAME, Firstname Middle" into "Firstname Middle SURNAME"
                raw_full_name = f"{parts[1].strip()} {parts[0].strip()}"
                ind_name = clean_ind_name(raw_full_name)
            else:
                ind_name = clean_ind_name(candidate_raw)
                
            # Assign the correct PartyAb or fallback to the formatted IND name
            if rpp_name:
                party_ab = party_map.get(rpp_name.strip(' '), "UNKNOWN_PARTY")
                if party_ab == 'UNKNOWN_PARTY':
                    import pdb; pdb.set_trace()
            elif candidate_raw=='nan':
                continue
            else:  
                party_ab = ind_name
                
            records.append({
                "div_nm": current_div,
                "PartyAb": party_ab
            })

    # 4. Generate the DataFrame and Export to CSV
    df_clean = pd.DataFrame(records)
    df_clean.to_csv(main_dir / output_csv_filepath, index=False)
    
    # 5. Generate the next_contesting_parties Dictionary Form
    div_dict = {}
    for div, group in df_clean.groupby("div_nm"):
        # dict.fromkeys() removes duplicates while preserving the exact order they appeared on the ballot
        div_dict[div] = list(dict.fromkeys(group["PartyAb"].tolist()))
        
    return df_clean, div_dict

# Example Execution
df_clean, div_dict = extract_2026_candidates("Nominations-CandidateContactPersonalDetails.xls", "VIC-2026-Parties.csv", "VIC-2026-Candidates.csv")

import pdb; pdb.set_trace()
# To print the dictionary exactly as your script expects it:
# print("next_contesting_parties = {")
# for k, v in div_dict.items():
#     print(f"    '{k}': {v},")
# print("}")