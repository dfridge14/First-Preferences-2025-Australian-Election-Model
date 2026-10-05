import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os
from pathlib import Path

import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler

 

base_dir = Path.home() / f"Australian Election/Victorian Election/"
os.chdir(base_dir)

def format_name(name_str):
        if pd.isna(name_str):
            return name_str
        parts = name_str.split(',')
        if len(parts) == 2:
            return f"{parts[1].strip().title()} {parts[0].strip().title()}"
        return name_str.strip().title()

# Councillor extraction for 2012
def scrape_2012_councillors_direct():
    lga_prefixes = [
        'alpine', 'ararat', 'ballarat', 'banyule', 'basscoast', 'bawbaw', 'bayside',
        'benalla', 'boroondara', 'brimbank', 'buloke', 'campaspe', 'cardinia', 'casey',
        'centralgoldfields', 'colacotway', 'corangamite', 'darebin', 'eastgippsland',
        'frankston', 'gannawarra', 'gleneira', 'glenelg', 'goldenplains', 'greaterbendigo',
        'greaterdandenong', 'greatergeelong', 'greatershepparton', 'hepburn', 'hindmarsh',
        'hobsonsbay', 'horsham', 'hume', 'indigo', 'kingston', 'knox', 'latrobe', 'loddon',
        'macedonranges', 'manningham', 'mansfield', 'maribyrnong', 'maroondah', 'melbourne',
        'melton', 'mildura', 'mitchell', 'moira', 'monash', 'mooneevalley', 'moorabool',
        'moreland', 'morningtonpeninsula', 'mountalexander', 'moyne', 'murrindindi',
        'nillumbik', 'northerngrampians', 'portphillip', 'pyrenees', 'queenscliffe',
        'southgippsland', 'southerngrampians', 'stonnington', 'strathbogie', 'surfcoast',
        'swanhill', 'towong', 'wangaratta', 'warrnambool', 'wellington', 'westwimmera',
        'whitehorse', 'whittlesea', 'wodonga', 'wyndham', 'yarra', 'yarraranges'
    ]

    # THE FIX: Using the exact Azure Blob Storage URL from your screenshot
    base_url = "https://itsitecoreblobvecprd01.blob.core.windows.net/public-files/historical-results/council2012"
    headers = {"User-Agent": "VicElectionModel/1.0"}
    results = []

    for prefix in lga_prefixes:
        # 1. Strip 'greater' for the URL request to match VEC's backend (e.g., 'bendigoresult2012.html')
        url_prefix = prefix.replace('greater', '') if prefix.startswith('greater') else prefix
        file_name = f"{url_prefix}result2012.html"
        url = f"{base_url}/{file_name}"
        
        # 2. Format the display name cleanly for the CSV (e.g., "Greater Bendigo")
        display_name = prefix.replace('greater', 'Greater ').title() if prefix.startswith('greater') else prefix.title()
        
        try:
            res = requests.get(url, headers=headers)
            
            if res.status_code != 200:
                # Note: Brimbank will still legitimately 404 here due to state administration
                print(f"[{display_name:<18}] Failed: HTTP {res.status_code}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            text_content = soup.get_text(separator=" | ")
            
            # REGEX: Finds "Elected:", captures the name, stops at a parenthesis or pipe
            regex = r"Elected:\s*(?:\|\s*)?([A-Za-z\s,\-\'\.]+?)(?:\s*\(|\s*\||$)"
            matches = re.findall(regex, text_content, re.IGNORECASE)
            
            if matches:
                clean_names = [" ".join(m.strip().split()) for m in matches if m.strip()]
                print(f"[{display_name:<18}] Found {len(clean_names):02d} | E.g., {', '.join(clean_names[:3])}")
                
                for name in clean_names:
                    if 0 < len(name) < 50:
                        results.append({
                            "Year": 2012,
                            "Council_Name": display_name,
                            "Candidate": name
                        })
            else:
                print(f"[{display_name:<18}] HTML loaded but no 'Elected:' candidates found.")
                
        except Exception as e:
            print(f"[{display_name:<18}] Error: {e}")

        # Polite delay
        time.sleep(0.5)
    
    if not results:
        print("\nExtraction failed. Ensure the base URL matches the network tab.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    

    df['Candidate_Formatted'] = df['Candidate'].apply(format_name)
    
    print("\nExtraction complete!")
    print(f"Total Elected Councillors in 2012: {len(df)}")
    return df


# get 2016 and 2020 councillors

councillors_2020 = pd.read_excel('VIC-2020-Council-Candidates.xlsx').rename(columns={'council_name': 'Council_Name', 'candidate_name': 'Candidate'})
councillors_2016 = pd.read_excel('VIC-2016-Council-Candidates.xlsx').rename(columns={'council_name': 'Council_Name', 'candidate_name': 'Candidate'})

elected_2020 = councillors_2020.loc[councillors_2020['elected']==1,['Council_Name','Candidate']]
elected_2016 = councillors_2016.loc[councillors_2016['elected']==1,['Council_Name','Candidate']]

elected_2020['Candidate_Formatted'] = elected_2020['Candidate'].apply(format_name)
elected_2016['Candidate_Formatted'] = elected_2016['Candidate'].apply(format_name)

elected_2020.insert(0, 'Year', 2020)
elected_2016.insert(0, 'Year', 2016)

df_2012 = scrape_2012_councillors_direct()

councillors_df = pd.concat([df_2012,elected_2016,elected_2020], axis = 0)
import pdb; pdb.set_trace()
councillors_df.to_csv("VEC_Councillors.csv", index=False)






import pdb; pdb.set_trace()



# ==========================================
# 1. EXTRACT INDS FROM THE TABLE
# ==========================================

def get_la_ind_candidates(page_title):
    """
    Parses the rendered HTML of the Wikipedia candidate page to find 
    the Legislative Assembly table and extract all Independent candidates.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "format": "json"
    }
    
    headers = {"User-Agent": "VicElectionModel/1.0 (contact@research.org)"}
    response = requests.get(url, params=params, headers=headers).json()
    html_content = response["parse"]["text"]["*"]
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Target wikitables on the page
    tables = soup.find_all("table", class_="wikitable")
    
    # Regex to match: "Candidate Name (Ind)" or "Candidate Name (Ind. Something)"
    ind_pattern = re.compile(r"([A-Za-z\s\.\'\-]+?)\s*\((?:Ind|Ind\.[^)]*)\)", re.IGNORECASE)
    
    extracted_candidates = []
    
    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        
        # Verify this is the electorate table (must have 'electorate' and 'other')
        if not any("electorate" in h for h in headers) or not any("other" in h for h in headers):
            continue
            
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:
                continue
                
            # Column 0 is Electorate, last column is 'Other candidates'
            electorate = cells[0].get_text(strip=True)
            other_cell = cells[-1]
            
            # Replace <br> with newlines to keep multi-candidate cells distinct
            for br in other_cell.find_all("br"):
                br.replace_with("\n")
                
            cell_text = other_cell.get_text()
            
            # Search for any IND matches within the cell
            matches = ind_pattern.findall(cell_text)
            for cand_name in matches:
                cand_clean = cand_name.strip()
                if cand_clean:
                    extracted_candidates.append({
                        "name": cand_clean,
                        "electorate": electorate
                    })
                    
    return extracted_candidates


# ==========================================
# 2. MEDIAWIKI API REVISION SEARCH
# ==========================================

def get_all_revisions(page_title):
    """Pulls all revision IDs and timestamps chronologically."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "ids|timestamp",
        "rvlimit": "max",
        "rvdir": "newer",
        "format": "json"
    }
    headers = {"User-Agent": "VicElectionModel/1.0 (contact@research.org)"}
    revisions = []
    
    while True:
        res = requests.get(url, params=params, headers=headers).json()
        pages = res["query"]["pages"]
        page_id = list(pages.keys())[0]
        
        if page_id == "-1":
            return []
            
        revisions.extend(pages[page_id].get("revisions", []))
        
        if "continue" in res:
            params["rvcontinue"] = res["continue"]["rvcontinue"]
        else:
            break
            
    return revisions

# In-memory cache across candidates: {revid: page_wikitext}
REV_CACHE = {}

def check_name_in_revid(revid, name):
    """Checks whether the candidate string is present in a past revision, with caching."""
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "VicElectionModel/1.0 (contact@research.org)"}
    
    # 1. Check local cache first to avoid redundant API hits
    if revid in REV_CACHE:
        return name.lower() in REV_CACHE[revid]
        
    params = {
        "action": "query",
        "prop": "revisions",
        "revids": revid,
        "rvprop": "content",
        "rvslots": "main",
        "format": "json"
    }
    
    try:
        res = requests.get(url, params=params, headers=headers).json()
        pages = res["query"]["pages"]
        page_id = list(pages.keys())[0]
        content = pages[page_id]["revisions"][0]["slots"]["main"]["*"].lower()
        REV_CACHE[revid] = content  # Store in cache
        return name.lower() in content
    except Exception:
        return False

def find_first_appearance(name, revisions):
    """Binary search across revisions with step-by-step intermediate logging."""
    left = 0
    right = len(revisions) - 1
    first_seen = None
    step = 0
    
    while left <= right:
        step += 1
        mid = (left + right) // 2
        rev = revisions[mid]
        revid = rev["revid"]
        ts = rev["timestamp"]
        
        found = check_name_in_revid(revid, name)
        
        # Intermediate debug print for each binary search halving step
        # (Prints on one line with \r or indentation)
        # print(f"    Step {step}: checking rev {revid} ({ts[:10]}) -> {'FOUND' if found else 'MISS'}", flush=True)
        
        if found:
            first_seen = ts
            right = mid - 1
        else:
            left = mid + 1
            
    return first_seen

# ==========================================
# 3. PIPELINE RUNNER
# ==========================================

def build_election_ind_df(year, page_title, election_date_str):
    print(f"\n==========================================", flush=True)
    print(f"[{year}] Scraping IND candidates from table...", flush=True)
    candidates = get_la_ind_candidates(page_title)
    total_cands = len(candidates)
    print(f"[{year}] Found {total_cands} Independent candidates.", flush=True)
    
    print(f"[{year}] Fetching revision history...", flush=True)
    revisions = get_all_revisions(page_title)
    print(f"[{year}] Loaded {len(revisions)} revisions.", flush=True)
    print(f"[{year}] Commencing binary search across candidates...\n", flush=True)
    
    election_date = datetime.strptime(election_date_str, "%Y-%m-%d")
    results = []
    start_time = time.time()
    
    for idx, cand in enumerate(candidates, 1):
        name = cand["name"]
        electorate = cand["electorate"]
        
        # Immediate print before querying
        print(f"[{idx:02d}/{total_cands:02d}] {name.ljust(25)} ({electorate})... ", end="", flush=True)
        
        cand_start = time.time()
        first_date_str = find_first_appearance(name, revisions)
        elapsed = time.time() - cand_start
        
        if first_date_str:
            first_date = datetime.strptime(first_date_str, "%Y-%m-%dT%H:%M:%SZ")
            days_before = (election_date - first_date).days
            date_display = first_date.strftime("%Y-%m-%d")
            print(f"Added: {date_display} ({days_before:>3} days prior) [{elapsed:.2f}s]", flush=True)
        else:
            days_before = None
            print(f"NOT FOUND in historical wikitext [{elapsed:.2f}s]", flush=True)
            
        results.append({
            "Year": year,
            "Electorate": electorate,
            "Candidate": name,
            "First_Wiki_Date": first_date_str,
            "Campaign_Length_Days": days_before
        })
        
    total_elapsed = round(time.time() - start_time, 1)
    print(f"\n[{year}] Completed all {total_cands} candidates in {total_elapsed}s (Cached revisions: {len(REV_CACHE)})", flush=True)
    return pd.DataFrame(results)
# Run for 2014, 2018, and 2022
elections = [
    {"year": 2022, "page": "Candidates_of_the_2022_Victorian_state_election", "date": "2022-11-26"},
    {"year": 2018, "page": "Candidates_of_the_2018_Victorian_state_election", "date": "2018-11-24"},
    {"year": 2014, "page": "Candidates_of_the_2014_Victorian_state_election", "date": "2014-11-29"}
   
    
]

all_ind_dfs = [build_election_ind_df(e["year"], e["page"], e["date"]) for e in elections]
ind_wiki_df = pd.concat(all_ind_dfs, ignore_index=True)
print(ind_wiki_df.head(20))

# cleaning for 2014-2022 VIC
ind_wiki_df.loc[(ind_wiki_df['Candidate'] == 'Chris Byrne') & (ind_wiki_df['Year'] == 2014),'Campaign_Length_Days'] = 13
ind_wiki_df.loc[(ind_wiki_df['Candidate'] == 'John Casley') & (ind_wiki_df['Year'] == 2022),'Campaign_Length_Days'] = 18
ind_wiki_df.loc[(ind_wiki_df['Candidate'] == 'Kammy Cordner Hunt') & (ind_wiki_df['Year'] == 2022),'Campaign_Length_Days'] = 15
ind_wiki_df = ind_wiki_df[ind_wiki_df['Electorate'] != 'Narracan[†]']

ind_wiki_df.to_csv('IND_campaign_lengths_VIC.csv', index = False)
import pdb; pdb.set_trace()