import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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

ind_wiki_df = pd.read_csv('IND_campaign_lengths_VIC.csv')


# correct obviously wrong campaign lengths
ind_wiki_df.loc[(ind_wiki_df['Candidate']=='Michael Neoh') & (ind_wiki_df['Year']==2018),'Campaign_Length_Days'] = 65
ind_wiki_df.loc[(ind_wiki_df['Candidate']=='Peter Hemphill')  & (ind_wiki_df['Year']==2018),'Campaign_Length_Days'] = 45
ind_wiki_df.loc[(ind_wiki_df['Candidate']=='Clare Le Serve')  & (ind_wiki_df['Year']==2014),'Campaign_Length_Days'] = 115
ind_wiki_df.loc[(ind_wiki_df['Candidate']=='Nicole Seymour')  & (ind_wiki_df['Year']==2022),'Campaign_Length_Days'] = 88
ind_wiki_df.loc[(ind_wiki_df['Candidate']=='Barry Shea')  & (ind_wiki_df['Year']==2018),'Campaign_Length_Days'] = 40
ind_wiki_df.loc[(ind_wiki_df['Candidate']=='Ray Burgess')  & (ind_wiki_df['Year']==2018),'Campaign_Length_Days'] = 143



# LOAD AND STANDARDISE RESULTS DATA

years = [2010, 2014, 2018, 2022]
all_results_list = []

for year in years:
    filename = f"VIC-{year}/VIC-{year}-LA-Primary-Electorate-Formatted.csv"
    try:
        df = pd.read_csv(filename)
        df['Year'] = year
        all_results_list.append(df)
    except FileNotFoundError:
        print(f"File {filename} not found.")

results_df = pd.concat(all_results_list, ignore_index=True)

# Ignore Narracan 2022 (delayed supplementary election)
results_df = results_df[~((results_df['Year'] == 2022) & (results_df['div_nm'].str.strip().str.title() == 'Narracan'))]

# Filter for Independents
inds_results = results_df[results_df['PartyAb'].str.contains('IND', na=False, case=False)].copy()

def format_name(name_str):
    if pd.isna(name_str):
        return name_str
    parts = name_str.split(',')
    if len(parts) == 2:
        return f"{parts[1].strip().title()} {parts[0].strip().title()}"
    return name_str.strip().title()

def simplify_name(name_str):
    """Strips middle names and initials: 'Norman F. Baker' -> 'Norman Baker'"""
    if pd.isna(name_str):
        return name_str
    tokens = [t for t in name_str.split() if not (len(t) <= 2 and t.endswith('.'))]
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[-1]}"
    return name_str

inds_results['Candidate'] = inds_results['candidate_name'].apply(format_name)
inds_results['Candidate_Simple'] = inds_results['Candidate'].apply(simplify_name)
inds_results['Electorate'] = inds_results['div_nm'].str.strip().str.title()

# Fix known Wikipedia spelling/typo discrepancies
wiki_typo_fixes = {
    ('Melton', 'Mohamad Alfojan'): 'Mohamad Aljofan',
    ('Broadmeadows', 'Mohamed Elmustapha'): 'Mohamad Elmustapha',
    ('Geelong', 'Doug Mann'): 'Douglas James Mann',
    ('Mildura', 'Steve Timmis'): 'Steven John Timmis',
    ('Bendigo West', 'Matthew Bansemer'): 'Matt Bansemer',
}

for (elec, bad_name), good_name in wiki_typo_fixes.items():
    mask = (ind_wiki_df['Electorate'].str.title() == elec) & (ind_wiki_df['Candidate'].str.title() == bad_name)
    ind_wiki_df.loc[mask, 'Candidate'] = good_name

ind_wiki_df['Candidate'] = ind_wiki_df['Candidate'].str.strip().str.title()
ind_wiki_df['Candidate_Simple'] = ind_wiki_df['Candidate'].apply(simplify_name)
ind_wiki_df['Electorate'] = ind_wiki_df['Electorate'].str.strip().str.title()


# CALCULATE ABSOLUTE VOTE SHARE

electorate_totals = results_df[results_df['PartyAb'] != 'INFORMAL'].groupby(['Year', 'div_nm'])['votes'].sum().reset_index()
electorate_totals.rename(columns={'div_nm': 'Electorate', 'votes': 'total_electorate_votes'}, inplace=True)
electorate_totals['Electorate'] = electorate_totals['Electorate'].str.strip().str.title()

inds_results = inds_results.merge(electorate_totals, on=['Year', 'Electorate'], how='left')
inds_results['Primary_Vote_Share'] = (inds_results['votes'] / inds_results['total_electorate_votes']) * 100


# TWO-ROUND MERGE (Exact -> Middle-Name Stripped)

# Round 1: Exact candidate name match
match_r1 = pd.merge(
    ind_wiki_df,
    inds_results[['Year', 'Electorate', 'Candidate', 'votes', 'Primary_Vote_Share', 'PartyAb']],
    on=['Year', 'Electorate', 'Candidate'],
    how='inner'
)

# Identify remaining unmatched entries
matched_wiki_keys = set(zip(match_r1['Year'], match_r1['Electorate'], match_r1['Candidate']))
matched_res_keys = set(zip(match_r1['Year'], match_r1['Electorate'], match_r1['Candidate']))

unmatched_wiki = ind_wiki_df[~ind_wiki_df.apply(lambda r: (r['Year'], r['Electorate'], r['Candidate']) in matched_wiki_keys, axis=1)]
unmatched_res = inds_results[~inds_results.apply(lambda r: (r['Year'], r['Electorate'], r['Candidate']) in matched_res_keys, axis=1)]

# Round 2: Match remaining on simplified First + Last name (preserves full results name)
match_r2 = pd.merge(
    unmatched_wiki.drop(columns=['Candidate']),
    unmatched_res[['Year', 'Electorate', 'Candidate_Simple', 'Candidate', 'votes', 'Primary_Vote_Share', 'PartyAb']],
    on=['Year', 'Electorate', 'Candidate_Simple'],
    how='inner'
)

# Combine both rounds (keeping Candidate_Simple out of final output)
successful_matches = pd.concat([match_r1, match_r2], ignore_index=True)
successful_matches.drop(columns=['Candidate_Simple'], errors='ignore', inplace=True)



# ==========================================
# DIAGNOSTIC: FIND MISSING INDS FROM RESULTS
# ==========================================

# Create identification keys to compare
matched_keys = set(zip(
    successful_matches['Year'], 
    successful_matches['Electorate'], 
    successful_matches['Candidate']
))

# Find candidates in inds_results whose (Year, Electorate, Candidate) are not in successful_matches
missing_from_wiki = inds_results[
    ~inds_results.apply(lambda r: (r['Year'], r['Electorate'], r['Candidate']) in matched_keys, axis=1)
].copy()

print("\n" + "="*60)
print(f"INDEPENDENTS IN VEC RESULTS MISSING FROM WIKI MATCH ({len(missing_from_wiki)} total)")
print("="*60)

# Display breakdown by year
print(missing_from_wiki['Year'].value_counts().sort_index())
print("\nDetailed list:")
print(missing_from_wiki[['Year', 'Electorate', 'Candidate', 'votes', 'Primary_Vote_Share']].to_string(index=False))


# Exclude Top 4 as they contested previously
missing_from_wiki = missing_from_wiki.iloc[4:,:]
imputed_inds = missing_from_wiki.copy()

# Assign 13 days to the remaining ballot-draw candidates
imputed_inds['Campaign_Length_Days'] = 15 
imputed_inds['First_Wiki_Date'] = None

cols_to_keep = [c for c in successful_matches.columns if c in imputed_inds.columns]
imputed_inds = imputed_inds[cols_to_keep]

successful_matches = pd.concat([successful_matches, imputed_inds], ignore_index=True)

import pdb; pdb.set_trace()

# EXCLUDE REPEAT CONTENDERS

# Map candidates who ran in the immediately prior cycle (2014 -> 2018, 2018 -> 2022)
prev_year_map = {2018: 2014, 2022: 2018}

prior_runs_by_elec = set()
for curr_y, prev_y in prev_year_map.items():
    # Grab unique (Electorate, Candidate_Simple) pairs from the prior election
    prior_pairs = inds_results.loc[
        inds_results['Year'] == prev_y, 
        ['Electorate', 'Candidate_Simple']
    ].drop_duplicates().values
    
    for elec, cand in prior_pairs:
        prior_runs_by_elec.add((curr_y, elec, cand))

# Check match in same electorate
successful_matches['Simple_Temp'] = successful_matches['Candidate'].apply(simplify_name)
is_repeat = successful_matches.apply(
    lambda r: (r['Year'], r['Electorate'], r['Simple_Temp']) in prior_runs_by_elec, 
    axis=1
)

print(f"Removed {is_repeat.sum()} repeat contenders running in the same electorate.")
final_IND_model_df = successful_matches.copy()
final_IND_model_df['is_repeat'] = is_repeat
#final_IND_model_df = successful_matches[~is_repeat].drop(columns=['Simple_Temp']).copy()
# Print Diagnostic Summary
print("\n" + "="*50)
print("MERGE DIAGNOSTICS")
print("="*50)
print(f"Total successful matches: {len(successful_matches)}")
print(f"Candidates in Wikipedia data missing from VEC results: {len(unmatched_wiki)}")
print(f"Candidates in VEC results missing from Wikipedia data: {len(unmatched_res)}\n")

if len(unmatched_wiki) > 0:
    print("--- UNMATCHED WIKIPEDIA CANDIDATES ---")
    print("(Likely formatting differences, dropped out before election, or Wiki vandalism)")
    print(unmatched_wiki[['Year', 'Electorate', 'Candidate', 'Campaign_Length_Days']].to_string(index=False))
    print("\n")

if len(unmatched_res) > 0:
    print("--- UNMATCHED VEC RESULTS CANDIDATES ---")
    print("(Likely spelling discrepancies or completely missed by Wikipedia editors)")
    print(unmatched_res[['Year', 'Electorate', 'Candidate', 'votes']].to_string(index=False))
    print("\n")



councillors_df = pd.read_csv("VEC_Councillors.csv")

final_IND_model_df['Is_Former_Councillor'] = final_IND_model_df.apply(
    lambda r: not councillors_df[
        (councillors_df['Candidate_Formatted'] == r['Candidate']) & 
        (councillors_df['Year'] < r['Year'])
    ].empty, 
    axis=1
)

# affiliated with Teals: Tracie Lund, Tammy Atkins, Jacqui Hawkins, Don Firth, Michelle Dunscombe,  Melissa Lowe, Sophie Torney, Kate Lardner, Felicity Frederico, Nomi Kaltmann
# otherwise Public Figuress: Ali Cupper, Carol Altmann, Jenny O'Connor, Darryn Lyons

teals = [
    'Tracie Lund', 'Tammy Atkins', 'Jacqui Hawkins', 'Don Firth', 'Michelle Dunscombe', 
    'Melissa Lowe', 'Sophie Torney', 'Kate Lardner', 'Felicity Frederico', 'Nomi Kaltmann', 'Sarah Fenton', 'Clarke Martin'
]
public_figures = ['Ali Cupper', 'Carol Altmann', "Jenny O'Connor", 'Darryn Lyons']

# 2. Map them to a new column
def get_candidate_archetype(name):
    if name in teals:
        return 'Teal'
    elif name in public_figures:
        return 'Public Figure'
    return 'Standard'

final_IND_model_df['Candidate_Archetype'] = final_IND_model_df['Candidate'].apply(get_candidate_archetype)
final_IND_model_df.loc[(final_IND_model_df['Candidate']=='Darryn Lyons') & (final_IND_model_df['Year']==2018),'Is_Former_Councillor'] = True # Mayor of Geelong


to_plot = 0

if to_plot:

    # 1. Create a categorical grouping column for the facets
    final_IND_model_df['Campaign_Phase'] = np.where(
        final_IND_model_df['Campaign_Length_Days'] <= 15, 
        '<= 15 Days (Ballot Draw / Late)', 
        '> 15 Days (Early Campaign)'
    )

    # 2. Explicitly create a standard matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    phases = ['<= 15 Days (Ballot Draw / Late)', '> 15 Days (Early Campaign)']

    for i, phase in enumerate(phases):
        subset = final_IND_model_df[final_IND_model_df['Campaign_Phase'] == phase]
        ax = axes[i]
        
        # A. Draw the regression line ONLY (scatter=False)
        sns.regplot(
            data=subset,
            x='Campaign_Length_Days',
            y='Primary_Vote_Share',
            ax=ax,
            scatter=False,
            line_kws={'color': 'red', 'linewidth': 2}
        )
        
        # B. Draw the scatter points with HUE for the councillors
        sns.scatterplot(
            data=subset,
            x='Campaign_Length_Days',
            y='Primary_Vote_Share',
            hue='Is_Former_Councillor',
            style='Candidate_Archetype',
            markers={'Teal': 's', 'Public Figure': '^', 'Standard': 'o'}, # s=square, ^=triangle, o=circle
            palette={True: 'purple', False: 'steelblue'},
            alpha=0.8,
            s=80, # Slightly larger to make shapes visible
            ax=ax,
            legend=(i == 1)
        )

        sns.scatterplot(
            data=subset[subset['is_repeat']],
            x='Campaign_Length_Days',
            y='Primary_Vote_Share',
            marker='o',
            s=100,                 # bigger outer circle
            facecolor='none',
            edgecolor='gold',
            linewidth=2,
            alpha = 0.5,
            ax=ax,
            legend=False
        )
        
        ax.set_title(phase, fontweight='bold')
        ax.set_xlabel("Campaign Length (Days Before Election)")
        ax.set_ylabel("Primary Vote Share (%)" if i == 0 else "")
        ax.grid(True, linestyle='--', alpha=0.5)

    # Fix legend title and placement
    axes[1].legend(title="Former Councillor", loc='upper left')

    # 3. Final layout adjustments
    fig.suptitle("Independent Candidate Vote Share vs. Wikipedia Insertion Date", y=1.03, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

import pdb; pdb.set_trace()

# Save final clean dataset for modelling
successful_matches.to_csv("Final_IND_Campaign_Model_Data.csv", index=False)




# TODO

# Add Suzanna Sheed 2014 to rows 

# final_IND_model_df.loc[final_IND_model_df['Candidate'].isin(final_IND_model_df.loc[final_IND_model_df['is_repeat']==False,].groupby('Candidate')['Candidate'].count().sort_values().tail(6).index),].sort_values(by='Candidate')


#1a. Save page electorate and primary vote performance for each is_repeat IND (CLEAN: double-doing the is_repeat section)

# 1. Clean candidate name helper
def clean_cand_name(name):
    return " ".join(str(name).strip().lower().split())

final_IND_model_df['Cand_Clean'] = final_IND_model_df['Candidate'].apply(clean_cand_name)

# 2. Extract prior run history across ALL electorates
is_repeat_list = []
past_vote_list = []
past_elec_list = []

for _, row in final_IND_model_df.iterrows():
    c_clean = row['Cand_Clean']
    curr_yr = row['Year']
    curr_elec = row['Electorate']
    
    # Match candidate anywhere in Victoria strictly before current election year
    priors = final_IND_model_df[
        (final_IND_model_df['Cand_Clean'] == c_clean) & 
        (final_IND_model_df['Year'] == curr_yr - 4) &
        (final_IND_model_df['Electorate'] == curr_elec)
    ]
    
    if not priors.empty:
        last_contest = priors.sort_values(by='Year').iloc[-1]
        is_repeat_list.append(True)
        past_vote_list.append(last_contest['Primary_Vote_Share'])
        past_elec_list.append(last_contest['Electorate'])
    else:
        is_repeat_list.append(False)
        past_vote_list.append(0.0)
        past_elec_list.append(None)

final_IND_model_df['is_repeat'] = is_repeat_list
final_IND_model_df['Past_Vote_Share'] = past_vote_list
final_IND_model_df['Past_Electorate'] = past_elec_list

# 1b. Get is_incumbent (manually apply to really strong repeat contenders - ) # Ali Cupper 2018-22, Jacqui Hawinks 22, Suzanna Sheed 18-22, Russel Northe 18? Plus the well-funded 2022 Teals

excluded_contests = [
    (2018, 'Mildura'),     # Ali Cupper
    (2022, 'Mildura'),     # Ali Cupper
    (2022, 'Benambra'),    # Jacqui Hawkins
    (2018, 'Shepparton'),  # Suzanna Sheed
    (2022, 'Shepparton'),  # Suzanna Sheed
    (2018, 'Morwell'),      # Russell Northe
    (2022, 'Brighton'),      # Felicity Frederico
    (2022, 'Caulfield'),      # Nomi Kaltmann
    (2022, 'Mornington'),      # Kate Lardner
    (2022, 'Kew'),      # Sophie Torney
    (2022, 'Hawthorn'),      # Melissa Lowe
    (2022, 'Sandringham'),  # Clarke Martin
    (2022, 'Bellarine'),  # Sarah Fenton
]

train_model_df = final_IND_model_df[~final_IND_model_df.set_index(['Year', 'Electorate']).index.isin(excluded_contests)].copy()

# 2. Get elec_df: get total IND vote in last electorate; fix for redistributions. 

def generate_complete_elec_df(results_df, party_category_dict, rediv_2014, rediv_2022, excluded_contests):
    # 1. Base Electorates from your snippet
    elect_df_list = []
    for year in [2010, 2014, 2018, 2022]:
        e_df = pd.read_excel(f'Tally Room/VIC-{year}/VIC-{year}-Electorates.xlsx')[['district_name', 'region_name']]
        e_df.insert(0, 'Year', year)
        elect_df_list.append(e_df)
        
    base_elec_df = pd.concat(elect_df_list, ignore_index=True)
    base_elec_df.rename(columns={'district_name': 'Electorate', 'region_name': 'Region'}, inplace=True)

    # 2. Filter Informals & Assign Party Categories
    res = results_df[results_df['PartyAb'] != 'INFORMAL'].copy()
    res['Category'] = res['PartyAb'].map(party_category_dict).fillna('Centre')

    # 3. Calculate Party Group Baselines (P_ALP, P_LNP, etc.)
    electorate_totals = res.groupby(['Year', 'div_nm'])['votes'].sum().reset_index(name='Total_Formal_Votes')
    party_totals = res.groupby(['Year', 'div_nm', 'Category'])['votes'].sum().reset_index()
    
    party_wide = party_totals.pivot(index=['Year', 'div_nm'], columns='Category', values='votes').fillna(0)
    party_wide = party_wide.div(electorate_totals.set_index(['Year', 'div_nm'])['Total_Formal_Votes'], axis=0) * 100
    party_wide.columns = [f'P_{c}' for c in party_wide.columns]
    party_wide.reset_index(inplace=True)
    party_wide.rename(columns={'div_nm': 'Electorate'}, inplace=True)

    # 4. Calculate Lagged Challenger IND Vote
    # Exclude structural giant contests (Cupper, Sheed, etc.)
    res['Is_Excluded_Contest'] = res.apply(lambda r: (r['Year'], r['div_nm']) in excluded_contests, axis=1)
    challenger_inds = res[(res['Category'] == 'IND') & (~res['Is_Excluded_Contest'])]
    
    ind_totals = challenger_inds.groupby(['Year', 'div_nm'])['votes'].sum().reset_index(name='IND_Votes')
    ind_totals = ind_totals.merge(electorate_totals, on=['Year', 'div_nm'], how='right').fillna({'IND_Votes': 0})
    ind_totals['IND_Pct'] = (ind_totals['IND_Votes'] / ind_totals['Total_Formal_Votes']) * 100

    # 5. Map Prior Cycle Results with Redistributions
    lag_records = []
    for year in [2014, 2018, 2022]:
        prev_year = year - 4
        prev_map = ind_totals[ind_totals['Year'] == prev_year].set_index('div_nm')['IND_Pct'].to_dict()
        
        current_districts = base_elec_df[base_elec_df['Year'] == year]['Electorate'].unique()
        rediv_dict = rediv_2022 if year == 2022 else (rediv_2014 if year == 2014 else {})
        
        for elec in current_districts:
            if elec in prev_map:
                lag_vote = prev_map[elec]
            elif elec in rediv_dict:
                # Donor electorate mapping
                donor = rediv_dict[elec]
                lag_vote = prev_map.get(donor, 0.0)
            else:
                lag_vote = 0.0
                
            lag_records.append({'Year': year, 'Electorate': elec, 'Lag_Challenger_IND_Vote': lag_vote})

    lag_df = pd.DataFrame(lag_records)

    # 6. Merge all components onto base_elec_df
    final_elec_df = base_elec_df.merge(party_wide, on=['Year', 'Electorate'], how='left')
    final_elec_df = final_elec_df.merge(lag_df, on=['Year', 'Electorate'], how='left')
    
    # Fill missing target party columns if not contested
    for col in ['P_ALP', 'P_COAL', 'P_Left', 'P_Right', 'P_Centre']:
        if col not in final_elec_df.columns:
            final_elec_df[col] = 0.0

    return final_elec_df


rediv_2022 = {'Ashwood':'Burwood','Berwick':'Gembrook','Eureka':'Buninyong','Glen Waverly':'Forrest Hill','Greenvale':'Yuroke','Kalkallo':'Yuroke','Laverton':'Footscray','Pakenham':'Gembrook','Point Cook': 'Altona'}
rediv_2014 = {'Buninyong':'Ballarat East','Clarinda':'Clayton','Croydon': 'Kilsyth','Eildon':'Seymour','Euroa':'Benalla','Keysborough':'Lyndhurst','Murray Plains':'Rodney','Ovens Valley':'Murray Valley','Ringwood':'Mitcham','Rowille':'Ferntree Gully','St Albans':'Derrimut','Sunbury':'Macedon','Sydenham':'Keilor','Wendouree':'Ballarat West','Werribee':'Tarneit'}

# get global party_df and reformat so that IND mapped to IND (not confused with Centre)
global_party_dict = {}

for data_year in [2006,2010, 2014, 2018, 2022]:
    temp_df = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-Parties.csv').dropna(subset=['PartyAb'])
    year_dict = dict(zip(temp_df['PartyAb'], temp_df['Ideo_category'])) # Add this year's parties to the master dictionary
    global_party_dict.update(year_dict)

for i in range(1, 11):
    global_party_dict[f'IND{i}'] = 'IND' 
global_party_dict['IND'] = 'IND'




elec_df = generate_complete_elec_df(results_df, global_party_dict, rediv_2014, rediv_2022, excluded_contests)




# 3. DOP table flows from minor IND to other parties, grouped by bloc (NOTE: be careful with deeper eliminations and empty blocs!)



def build_dop_flows_from_IND(year, DOP_table_dict, model_df, party_category_dict, 
                             counterfactual_df=None, MIN_CAT_NUM=4, results_df=None):
    """
    Extracts IND preference flows directly from wide DOP tables, replacing IND->IND flows with 0. Where not available, supplements this with IND defection flows (counterfactual_df). 
    Computes electorate-wide average, imputing for missing category. Target categories: ALP, COAL, Left, Right, Centre.
    """
    target_cats = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    flow_cols = [f'Flow_{c}' for c in target_cats]
    candidate_flows = []

    def clean_name(name):
        return " ".join(str(name).strip().lower().split())

    # Pre-build repeat lookup from model_df for the target year
    year_cands = model_df[model_df['Year'] == year].copy()
    if 'Cand_Clean' not in year_cands.columns:
        year_cands['Cand_Clean'] = year_cands['Candidate'].apply(clean_name)
    repeat_lookup = year_cands.set_index(['Electorate', 'Cand_Clean'])['is_repeat'].to_dict()

    # Pre-build PartyAb -> Candidate mapping if results_df is provided
    party_to_cand = {}
    if results_df is not None:
        r_year = results_df[results_df['Year'] == year]
        for _, r in r_year.iterrows():
            party_to_cand[(r['div_nm'], r['PartyAb'])] = r['candidate_name']

    # -------------------------------------------------------------
    # 1. PROCESS WIDE DOP TABLES
    # -------------------------------------------------------------
    for elec, dop in DOP_table_dict.items():
        # Party columns are strictly between index 2 and -2
        party_cols = list(dop.columns[2:-2])
        
        # Identify all IND columns in this electorate
        ind_cols = [c for c in party_cols if c.startswith('IND') or party_category_dict.get(c) == 'IND']
        if not ind_cols:
            continue

        for ind_col in ind_cols:
            # An IND is eliminated in the TransferredVotes row where its column becomes NaN
            # while having been non-NaN in the immediately preceding Votes row
            elim_idx = None
            for idx in range(1, len(dop)):
                if (dop.loc[idx, 'CountType'] == 'TransferredVotes' and 
                    pd.isna(dop.loc[idx, ind_col]) and 
                    pd.notna(dop.loc[idx - 1, ind_col])):
                    elim_idx = idx
                    break

            if elim_idx is None:
                continue

            elim_row = dop.loc[elim_idx]

            # Determine which receiving parties are active (non-NaN in this transfer row)
            # Exclude the eliminated IND and any other INDs on the ballot
            active_party_cols = [
                c for c in party_cols 
                if c != ind_col 
                and not c.startswith('IND') 
                and party_category_dict.get(c) != 'IND'
                and pd.notna(elim_row[c])
            ]

            # Map active receiving parties to target categories
            active_target_cats = set(
                party_category_dict[c] for c in active_party_cols 
                if c in party_category_dict and party_category_dict[c] in target_cats
            )

            # Check threshold: require at least MIN_CAT_NUM target categories
            if len(active_target_cats) >= MIN_CAT_NUM:
                # Sum transferred votes strictly across the target categories
                cat_transfers = {cat: 0.0 for cat in target_cats}
                for c in active_party_cols:
                    cat = party_category_dict.get(c)
                    if cat in target_cats:
                        val = elim_row[c]
                        if pd.notna(val) and val > 0:
                            cat_transfers[cat] += float(val)

                valid_total = sum(cat_transfers.values())
                if valid_total > 0:
                    # Resolve candidate name and is_repeat status
                    cand_name = party_to_cand.get((elec, ind_col), ind_col)
                    cand_key = (elec, clean_name(cand_name))
                    is_rep = repeat_lookup.get(cand_key, False)

                    # If cand_name was just 'IND1' and not found in lookup, fallback to electorate's known repeat status
                    if cand_key not in repeat_lookup:
                        elec_matches = year_cands[year_cands['Electorate'] == elec]
                        if not elec_matches.empty:
                            is_rep = bool(elec_matches['is_repeat'].iloc[0])

                    row_record = {
                        'Year': year,
                        'Electorate': elec,
                        'Candidate': cand_name,
                        'is_repeat': is_rep
                    }

                    # Assign normalized flow where category was active, NaN where absent
                    for cat in target_cats:
                        if cat in active_target_cats:
                            row_record[f'Flow_{cat}'] = cat_transfers[cat] / valid_total
                        else:
                            row_record[f'Flow_{cat}'] = np.nan

                    candidate_flows.append(row_record)

    flow_df = pd.DataFrame(candidate_flows)

    # -------------------------------------------------------------
    # 2. APPEND COUNTERFACTUAL DEFECTION FLOWS
    # -------------------------------------------------------------
    if counterfactual_df is not None and not counterfactual_df.empty:
        cf = counterfactual_df.copy()
        if 'Year' in cf.columns:
            cf = cf[cf['Year'] == year]
        else:
            cf['Year'] = year

        if 'is_repeat' not in cf.columns:
            cf['is_repeat'] = cf.apply(
                lambda r: repeat_lookup.get((r['Electorate'], clean_name(r.get('Candidate', ''))), False),
                axis=1
            )
        flow_df = pd.concat([flow_df, cf], ignore_index=True)

    if flow_df.empty:
        return pd.DataFrame(columns=['Year', 'Electorate'] + flow_cols)

    # -------------------------------------------------------------
    # 3. AGGREGATE PER ELECTORATE (UNWEIGHTED MEAN & RENORMALIZE)
    # -------------------------------------------------------------
    electorate_records = []
    for elec, group in flow_df.groupby('Electorate'):
        # Filter: Prioritize fresh insurgencies over repeat candidates
        fresh_cands = group[~group['is_repeat']]
        pool = fresh_cands if not fresh_cands.empty else group

        # Column-wise unweighted average; skipna=True leaves missing category as empty, is imputed from regional average later in the process
        mean_flows = pool[flow_cols].mean(axis=0, skipna=True)

        

        # renormalisation step
        total_sum = mean_flows.sum()
        if total_sum > 0:
            norm_flows = (mean_flows / total_sum).to_dict()
            norm_flows['Year'] = year
            norm_flows['Electorate'] = elec
            electorate_records.append(norm_flows)

    return pd.DataFrame(electorate_records)


# 3a. get counterfactual_IND_flows_df for each electorate possibel

# 3b. 

for year in [2022]: # ADD: previous years only work once complete_DOP_table_dict.pkl file produced

    with open(f'VIC-{data_year}/{data_year}_complete_DOP_table_dict.pkl', 'rb') as f:
        DOP_table_dict = pickle.load(f)
        if data_year == '2022':
            del DOP_table_dict['Narracan']

    IND_DOP_flows_2022 = build_dop_flows_from_IND(
        year=2022,
        DOP_table_dict=DOP_table_dict,       # Dict of DataFrames: {'Bentleigh': df, ...}
        model_df=train_model_df,       # Candidate roster with 'is_repeat'
        party_category_dict=global_party_dict,
        counterfactual_df=None,              
        MIN_CAT_NUM=4,                       # Strict threshold
        results_df=results_df                
    )

    # Inspect the extracted electorate preference flows
    print(IND_DOP_flows_2022.head(10))











import pdb; pdb.set_trace()


# ==========================================
# PHASE 1: DATA PREP & FILTERING
# ==========================================
def prepare_training_data(cand_df, elec_df, excluded_contests):
    train_cand = cand_df.copy()

    # 1. Flag Excluded Contests (The Unified List: Teals + Incumbents)
    excluded_idx = pd.MultiIndex.from_tuples(excluded_contests, names=['Year', 'Electorate'])
    train_cand['Is_Excluded_Seat'] = train_cand.set_index(['Year', 'Electorate']).index.isin(excluded_idx)

    # 2. Effective Days Logic 
    baseline_days = {2010: 13, 2014: 13, 2018: 14, 2022: 15}
    train_cand['Min_Days'] = train_cand['Year'].map(baseline_days)
    train_cand['Effective_Days'] = np.where(
        train_cand['Campaign_Length_Days'] <= train_cand['Min_Days'], 
        0, train_cand['Campaign_Length_Days']
    )
    train_cand['Log_Days'] = np.log(train_cand['Effective_Days'] + 1)
    
    # ---------------------------------------------------------
    # 3. CALCULATE "HIDDEN THREATS" (For Track 1)
    # ---------------------------------------------------------
    # We must measure New IND pressure even in excluded seats, so Track 1 knows 
    # if the Repeat Teals/Incumbents were actively attacked.
    all_new_inds = train_cand[train_cand['is_repeat'] == False]
    
    threat_pool = all_new_inds.groupby(['Year', 'Electorate']).agg(
        Sum_New_Days=('Effective_Days', 'sum'),
        Has_New_Councillor=('Is_Former_Councillor', 'max')
    ).reset_index()
    
    threat_pool['Log_Sum_New_Days'] = np.log(threat_pool['Sum_New_Days'] + 1)

    train_cand = train_cand.drop(columns=['Min_Days', 'Effective_Days'])

    # ---------------------------------------------------------
    # 4. TRACK 1: THE REPEAT/WALL TRACK
    # ---------------------------------------------------------
    # This automatically grabs Repeaters, Repeat Teals, and Incumbents.
    # First-time Teals safely fall away here because is_repeat == False.
    repeats = train_cand[train_cand['is_repeat'] == True].copy()
    
    repeat_pool = repeats.groupby(['Year', 'Electorate']).agg(
        Repeat_IND_Share=('Primary_Vote_Share', 'sum'),
        Repeat_Past_Vote=('Past_Vote_Share', 'sum')
    ).reset_index()
    
    # Track 1 Target Variable
    repeats['Retention_Rate'] = (repeats['Primary_Vote_Share'] / repeats['Past_Vote_Share']).clip(lower=0.01)
    
    # Merge the threat metrics (so Track 1 can calculate damage)
    repeats = repeats.merge(
        threat_pool, on=['Year', 'Electorate'], how='left'
    ).fillna({'Log_Sum_New_Days': 0, 'Has_New_Councillor': False})
    
    # ---------------------------------------------------------
    # 5. TRACK 2 & 3: THE INSURGENCY POOL
    # ---------------------------------------------------------
    # ENFORCE THE RULE: Excluded contests must NEVER touch Track 2 or 3.
    new_inds = train_cand[(train_cand['is_repeat'] == False) & (~train_cand['Is_Excluded_Seat'])].copy()
    
    new_pool = new_inds.groupby(['Year', 'Electorate']).agg(
        Total_New_IND_Share=('Primary_Vote_Share', 'sum')
    ).reset_index()
    
    # ---------------------------------------------------------
    # 6. MERGE TO ELECTORATE BASELINE
    # ---------------------------------------------------------
    pool_df = elec_df[['Year', 'Electorate', 'Lag_Challenger_IND_Vote']].copy()
    pool_df['Is_Excluded_Seat'] = pool_df.set_index(['Year', 'Electorate']).index.isin(excluded_idx)
    
    # Merge New Pool & Threat Pool
    pool_df = pool_df.merge(new_pool, on=['Year', 'Electorate'], how='left').fillna({'Total_New_IND_Share': 0})
    pool_df = pool_df.merge(threat_pool, on=['Year', 'Electorate'], how='left').fillna({'Log_Sum_New_Days': 0, 'Has_New_Councillor': False})
    
    # Merge Repeat Pool
    pool_df = pool_df.merge(repeat_pool, on=['Year', 'Electorate'], how='left').fillna({'Repeat_IND_Share': 0, 'Repeat_Past_Vote': 0})
    
    # Define generic lags
    pool_df['Generic_Lag_Vote'] = (pool_df['Lag_Challenger_IND_Vote'] - pool_df['Repeat_Past_Vote']).clip(lower=0)
    pool_df['Log_Generic_Lag'] = np.log(1 + pool_df['Generic_Lag_Vote'])
    pool_df['Log_Repeat_Past'] = np.log(1 + pool_df['Repeat_Past_Vote']) 
    
    # ENFORCE THE RULE: Drop excluded seats from pool_df so Track 2 never trains on them
    pool_df = pool_df[~pool_df['Is_Excluded_Seat']].drop(columns=['Is_Excluded_Seat'])
    
    # Split prep for Track 3
    new_inds = new_inds.merge(pool_df[['Year', 'Electorate', 'Total_New_IND_Share']], on=['Year', 'Electorate'])
    new_inds['Share_of_New_Pool'] = (new_inds['Primary_Vote_Share'] / new_inds['Total_New_IND_Share']).clip(lower=0.01)
    
    return new_inds, repeats, pool_df


# ==========================================
# PHASE 2: FIT STATISTICAL MODELS
# ==========================================
def fit_generative_models(new_inds, repeats, pool_df):
    
    # --- TRACK 1: Repeat Candidate Retention (NO INTERCEPT) ---
    y_rep = repeats['Retention_Rate'].astype(float)
    X_rep = repeats[['Log_Sum_New_Days', 'Has_New_Councillor']].astype(float)
    repeat_model = sm.GLM(y_rep, X_rep, family=sm.families.Gamma(link=sm.families.links.Log())).fit()

    # --- TRACK 2: The New IND Pool Model ---
    active_pool_df = pool_df[pool_df['Total_New_IND_Share'] > 0].copy() # ensure only data from elecotrates where contesting
    y_pool = (active_pool_df['Total_New_IND_Share'] / 100.0).astype(float)
    
    # We use threat_pool variables from new_pool since we want the filtered baseline
    X_pool_cols = active_pool_df[['Log_Generic_Lag','Has_New_Councillor', 'Log_Sum_New_Days', 'Log_Repeat_Past']].copy().astype(float)
    
    # (Optional: Re-add 'Has_New_Councillor' to Track 2 if you want it there, 
    # but currently it's acting nicely in Track 1 and Track 3)
    X_pool = sm.add_constant(X_pool_cols.astype(float))
    pool_model = sm.GLM(y_pool, X_pool, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    
    # --- TRACK 3: The Allocation Split Model ---
    contest_counts = new_inds.groupby(['Year', 'Electorate'])['Candidate'].transform('count')
    multi_new_inds = new_inds[contest_counts >= 2].copy()
    
    y_split = np.log(multi_new_inds['Share_of_New_Pool'].astype(float))
    X_split = sm.add_constant(multi_new_inds[['Is_Former_Councillor', 'Log_Days']].astype(float))
    split_model = sm.OLS(y_split, X_split).fit()
    
    return repeat_model, pool_model, split_model

# ==========================================
# EXECUTION CALL
# ==========================================
excluded_contests = [
    (2018, 'Mildura'), (2022, 'Mildura'), (2022, 'Benambra'),
    (2018, 'Shepparton'), (2022, 'Shepparton'), (2018, 'Morwell'), 
    (2022, 'Brighton'), (2022, 'Caulfield'), (2022, 'Mornington'), 
    (2022, 'Kew'), (2022, 'Hawthorn'), (2022, 'Sandringham'), (2022, 'Bellarine')
]

# Pass the raw, unfiltered final_IND_model_df directly in.
# The function will act as the traffic cop.
new_inds, repeats, pool_df = prepare_training_data(final_IND_model_df, elec_df, excluded_contests)



filtered_pool_df = pool_df[pool_df['Year'] != 2010].copy()

repeat_model, pool_model, split_model = fit_generative_models(new_inds, repeats, filtered_pool_df)

print("=== DIAGNOSTIC 1: THE TRACK 2 ZERO-CRASH ===")
zero_seats = len(pool_df[pool_df['Total_New_IND_Share'] == 0])
print(f"Seats with exactly 0.0% New IND vote: {zero_seats} out of {len(pool_df)}")
print("-> If this is > 0, it caused the Gamma log-link crash. We must filter these out of Track 2.\n")


print("=== DIAGNOSTIC 2: THE GIANTS SPOT-CHECK ===")
# Are Sheed and Cupper actually being mapped properly to exert the 'oxygen squeeze'?
giants_check = pool_df[pool_df['Electorate'].isin(['Mildura', 'Shepparton'])].sort_values(['Electorate', 'Year'])
print(giants_check[['Year', 'Electorate', 'Repeat_Past_Vote', 'Log_Repeat_Past', 'Total_New_IND_Share', 'Log_Sum_New_Days']])
print("\n-> Look at 2022 Mildura/Shepparton. If Repeat_Past_Vote is 0, the mapping failed.\n")


print("=== DIAGNOSTIC 3: TRACK 1 RETENTION EXTREMES ===")
# Did a 0.5% IND jump to 5%, creating a 1000% retention rate that destroys Track 1?
bad_retention = repeats.sort_values('Retention_Rate', ascending=False)
print("Highest Retention Rates (Max should ideally not exceed ~2.5):")
print(bad_retention[['Year', 'Electorate', 'Candidate', 'Past_Vote_Share', 'Primary_Vote_Share', 'Retention_Rate']].head(5))
print("\nLowest Retention Rates:")
print(bad_retention[['Year', 'Electorate', 'Candidate', 'Past_Vote_Share', 'Primary_Vote_Share', 'Retention_Rate']].tail(3))
print("\n-> If fringe INDs are jumping 500%, we may need to cap Retention_Rate or exclude INDs < 2% from Track 1.\n")


print("=== DIAGNOSTIC 4: TRACK 2 RESIDUALS (THE OUTLIER HUNT) ===")

print("=== TRACK 1: BIGGEST MISSES (Retention Model) ===")
repeats['Actual_Retention'] = repeats['Retention_Rate']
# Force float here
repeats['Predicted_Retention'] = repeat_model.predict(repeats[['Log_Sum_New_Days', 'Has_New_Councillor']].astype(float))
repeats['Retention_Error'] = repeats['Actual_Retention'] - repeats['Predicted_Retention']
repeats['Predicted_Vote_Share'] = repeats['Past_Vote_Share'] * repeats['Predicted_Retention']
repeats['Vote_Share_Error'] = repeats['Primary_Vote_Share'] - repeats['Predicted_Vote_Share']

track1_diagnostics = repeats[[
    'Year', 'Electorate', 'Candidate', 
    'Past_Vote_Share', 'Primary_Vote_Share', 
    'Actual_Retention', 'Predicted_Retention', 'Vote_Share_Error'
]].sort_values('Vote_Share_Error')

print("Under-predicted their vote (Model thought they'd get less):")
print(track1_diagnostics.tail(5))
print("\nOver-predicted their vote (Model thought they'd get more):")
print(track1_diagnostics.head(5))


print("\n=== TRACK 2: BIGGEST MISSES (Insurgency Pool) ===")
# Strictly filter active pools from the filtered pool (post-2010)
active_pools = filtered_pool_df[filtered_pool_df['Total_New_IND_Share'] > 0].copy().reset_index(drop=True)

# Force float conversion BEFORE add_constant
X_pool_cols = active_pools[['Log_Generic_Lag', 'Has_New_Councillor', 'Log_Sum_New_Days', 'Log_Repeat_Past']].copy().astype(float)
X_pool = sm.add_constant(X_pool_cols)

active_pools['Predicted_Pool_Share'] = pool_model.predict(X_pool) * 100.0
active_pools['Actual_Pool_Share'] = active_pools['Total_New_IND_Share']
active_pools['Pool_Error'] = active_pools['Actual_Pool_Share'] - active_pools['Predicted_Pool_Share']

track2_diagnostics = active_pools[[
    'Year', 'Electorate', 
    'Log_Generic_Lag', 'Log_Repeat_Past', 'Log_Sum_New_Days', 'Has_New_Councillor',
    'Actual_Pool_Share', 'Predicted_Pool_Share', 'Pool_Error'
]].sort_values('Pool_Error')

print("Under-predicted the Insurgency (New INDs got way more than expected):")
print(track2_diagnostics.tail(5))
print("\nOver-predicted the Insurgency (New INDs got way less than expected):")
print(track2_diagnostics.head(5))
import pdb; pdb.set_trace()

# 








import pdb; pdb.set_trace()

# ==============================================================================
# 1. STATEWIDE IND -> IND PREFERENCE POOLING (Isolated by Year)
# ==============================================================================
def extract_statewide_ind_to_ind_flow(dop_table_dict, party_category_dict, data_year=None):
    """
    Scans historical wide DOP tables to find the empirical IND -> IND preference flow.
    If data_year is provided and a 'Year' column exists in the DOP sheets, it strictly filters.
    """
    ind_flows = []

    for elec, dop in dop_table_dict.items():
        if dop.empty or len(dop) < 2:
            continue
            
        # Optional year filter if DOP sheets contain a 'Year' column
        if data_year is not None and 'Year' in dop.columns:
            if dop['Year'].iloc[0] != data_year:
                continue

        party_cols = list(dop.columns[2:-2])
        ind_cols = [
            c for c in party_cols
            if str(c).startswith('IND') or party_category_dict.get(c) == 'IND'
        ]

        if len(ind_cols) < 2:
            continue

        for ind_col in ind_cols:
            elim_idx = None
            for idx in range(1, len(dop)):
                if (
                    dop.loc[idx, 'CountType'] == 'TransferredVotes'
                    and pd.isna(dop.loc[idx, ind_col])
                    and pd.notna(dop.loc[idx - 1, ind_col])
                ):
                    elim_idx = idx
                    break

            if elim_idx is None:
                continue

            elim_row = dop.loc[elim_idx]

            other_active_inds = [
                c for c in ind_cols
                if c != ind_col
                and pd.notna(elim_row[c])
                and pd.notna(dop.loc[elim_idx - 1, c])
            ]
            if not other_active_inds:
                continue

            all_active_receivers = [
                c for c in party_cols
                if c != ind_col
                and pd.notna(elim_row[c])
                and float(elim_row[c]) > 0
            ]

            total_transferred = sum(float(elim_row[c]) for c in all_active_receivers)
            if total_transferred <= 0:
                continue

            transferred_to_inds = sum(float(elim_row[c]) for c in other_active_inds)
            ind_flows.append(transferred_to_inds / total_transferred)

    if not ind_flows:
        return 0.20  # Empirical fallback

    return float(np.mean(ind_flows))


# ==============================================================================
# 2. REGIONAL IMPUTATION, RENORMALIZATION & 3-TIER PRIORS (Isolated by Year)
# ==============================================================================
def impute_and_build_dop_priors(ind_dop_flows_df, elec_df, data_year, S_local=75, S_borrowed=35):
    """
    Strictly filters flows and electorate baselines to data_year.
    Imputes missing categories, renormalizes, and establishes the 3-Tier S lookup.
    """
    target_cats = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    flow_cols = [f'Flow_{c}' for c in target_cats]

    # Isolate strictly to the target year
    df = ind_dop_flows_df[ind_dop_flows_df['Year'] == data_year].copy()
    e_df = elec_df[elec_df['Year'] == data_year].copy()

    # Map Region from the isolated elec_df if missing
    if 'Region' not in df.columns:
        region_map = e_df.drop_duplicates(subset=['Electorate']).set_index('Electorate')['Region'].to_dict()
        df['Region'] = df['Electorate'].map(region_map)

    # Calculate Regional Mean Baselines for the specific year
    reg_means = df.groupby('Region')[flow_cols].mean()
    state_means = df[flow_cols].mean()

    cleaned_rows = []
    for _, row in df.iterrows():
        reg = row['Region']
        clean_row = row.copy()

        for col in flow_cols:
            if pd.isna(clean_row[col]):
                if pd.notna(reg) and reg in reg_means.index and not pd.isna(reg_means.loc[reg, col]):
                    clean_row[col] = reg_means.loc[reg, col]
                else:
                    clean_row[col] = state_means[col]

        row_sum = sum(clean_row[flow_cols])
        if row_sum > 0:
            clean_row[flow_cols] = clean_row[flow_cols] / row_sum

        cleaned_rows.append(clean_row)

    cleaned_df = pd.DataFrame(cleaned_rows)

    final_reg_priors = cleaned_df.groupby('Region')[flow_cols].mean()
    final_state_prior = cleaned_df[flow_cols].mean().values

    dop_lookup = {
        'data_year': data_year,
        'electorates': {},
        'regions': {},
        'state': final_state_prior,
        'target_cats': target_cats,
    }

    for reg, grp in final_reg_priors.iterrows():
        dop_lookup['regions'][reg] = grp.values

    for _, row in cleaned_df.iterrows():
        dop_lookup['electorates'][row['Electorate']] = row[flow_cols].values

    dop_lookup['S_local'] = S_local
    dop_lookup['S_borrowed'] = S_borrowed

    return dop_lookup


# ==============================================================================
# 3. WATERFALL CAPACITY EXTRACTION (Pure Math - No Year Required)
# ==============================================================================
def apply_waterfall_extraction(T_pool, p_propensity, capacities, max_drain=0.90):
    """
    Extracts votes proportionally, enforcing a 10% capacity floor and reallocating
    excess demands to uncapped parties to rigidly conserve total extracted mass.
    """
    K = len(capacities)
    V = np.zeros(K)
    caps = capacities * max_drain

    total_available_capacity = np.sum(caps)
    T_target = min(T_pool, total_available_capacity)

    active = np.ones(K, dtype=bool)
    remaining_T = T_target

    while remaining_T > 1e-9 and np.any(active):
        weights = p_propensity[active] * capacities[active]
        w_sum = np.sum(weights)

        if w_sum <= 1e-12:
            weights = capacities[active]
            w_sum = np.sum(weights)
            if w_sum <= 1e-12:
                break

        w_norm = weights / w_sum
        demands = remaining_T * w_norm

        active_indices = np.where(active)[0]
        remaining_caps = caps[active] - V[active]
        exceeded = demands > remaining_caps

        if not np.any(exceeded):
            V[active] += demands
            remaining_T = 0.0
            break
        else:
            for i, is_exc in enumerate(exceeded):
                orig_idx = active_indices[i]
                if is_exc:
                    v_added = remaining_caps[i]
                    V[orig_idx] += v_added
                    remaining_T -= v_added
                    active[orig_idx] = False

    return V


# ==============================================================================
# 4. FULL MONTE CARLO COUNTERFACTUAL SIMULATOR (Includes data_year tag)
# ==============================================================================
def simulate_electorate_counterfactual(
    data_year,
    elec_name,
    region_name,
    elec_baseline_data,
    counterfactual_counts,
    new_candidates,
    pool_model,
    split_model,
    dop_lookup,
    mu_ind_to_ind,
    M=1000,
):
    target_cats = dop_lookup['target_cats']
    num_new_inds = len(new_candidates)

    if num_new_inds == 0:
        return {'simulations': [], 'mean_results': counterfactual_counts}

    sum_days = sum(c.get('Campaign_Length_Days', 0) for c in new_candidates)
    log_sum_days = np.log(sum_days + 1)
    has_councillor = float(max(c.get('Is_Former_Councillor', False) for c in new_candidates))

    repeat_past = counterfactual_counts.get('Repeat_IND', elec_baseline_data.get('Repeat_Past_Vote', 0.0))
    generic_lag = max(0.0, elec_baseline_data.get('Lag_Challenger_IND_Vote', 0.0) - repeat_past)

    log_generic_lag = np.log(1 + generic_lag)
    log_repeat_past = np.log(1 + repeat_past)

    x_pool = np.array([1.0, log_generic_lag, has_councillor, log_sum_days, log_repeat_past])
    linear_pred = np.dot(x_pool, pool_model.params)
    mu_pool = np.exp(linear_pred) * 100.0
    dispersion = pool_model.scale

    shape_k = 1.0 / dispersion
    scale_theta = mu_pool * dispersion

    split_params = split_model.params
    split_mse = split_model.mse_resid

    has_repeat_ind = 'Repeat_IND' in counterfactual_counts and counterfactual_counts['Repeat_IND'] > 0

    if elec_name in dop_lookup['electorates']:
        mu_base = dop_lookup['electorates'][elec_name].copy()
        S = dop_lookup['S_local']
    elif region_name in dop_lookup['regions']:
        mu_base = dop_lookup['regions'][region_name].copy()
        S = dop_lookup['S_borrowed']
    else:
        mu_base = dop_lookup['state'].copy()
        S = dop_lookup['S_borrowed']

    if has_repeat_ind:
        mu_sim = np.append(mu_base * (1.0 - mu_ind_to_ind), mu_ind_to_ind)
        categories_sim = target_cats + ['Repeat_IND']
        S = S * 0.70  # Tier 3
    else:
        mu_sim = mu_base.copy()
        categories_sim = list(target_cats)

    alpha_dirichlet = S * mu_sim
    initial_caps = np.array([counterfactual_counts[cat] for cat in categories_sim])

    sim_outputs = []

    for m in range(M):
        T_pool_sim = np.random.gamma(shape_k, scale_theta)

        if num_new_inds == 1:
            ind_votes = {new_candidates[0]['Candidate']: T_pool_sim}
        else:
            log_scores = []
            for c in new_candidates:
                eff_days = c.get('Campaign_Length_Days', 0)
                is_counc = float(c.get('Is_Former_Councillor', False))
                x_split = np.array([1.0, is_counc, np.log(eff_days + 1)])
                pred_log = np.dot(x_split, split_params) + np.random.normal(0, np.sqrt(split_mse))
                log_scores.append(pred_log)

            max_s = max(log_scores)
            exp_s = np.exp(np.array(log_scores) - max_s)
            split_shares = exp_s / np.sum(exp_s)
            ind_votes = {c['Candidate']: T_pool_sim * split_shares[i] for i, c in enumerate(new_candidates)}

        p_draw = np.random.dirichlet(alpha_dirichlet)
        extracted_v = apply_waterfall_extraction(T_pool_sim, p_draw, initial_caps, max_drain=0.90)
        post_caps = initial_caps - extracted_v

        final_macro_counts = {cat: post_caps[i] for i, cat in enumerate(categories_sim)}
        
        subparty_breakdown = {}
        if 'subparties' in counterfactual_counts:
            for sub_party, info in counterfactual_counts['subparties'].items():
                parent_bloc = info['bloc']
                parent_baseline = counterfactual_counts[parent_bloc]
                sub_baseline = info['vote_share']

                if parent_baseline > 0:
                    bloc_drain = extracted_v[categories_sim.index(parent_bloc)]
                    sub_drain = bloc_drain * (sub_baseline / parent_baseline)
                    subparty_breakdown[sub_party] = max(0.0, sub_baseline - sub_drain)
                else:
                    subparty_breakdown[sub_party] = 0.0

        sim_outputs.append({
            'Sim_ID': m,
            'Year': data_year,
            'Electorate': elec_name,
            'Total_New_IND_Pool': T_pool_sim,
            'New_IND_Votes': ind_votes,
            'Final_Macro_Blocs': final_macro_counts,
            'Final_Subparties': subparty_breakdown,
        })

    return sim_outputs


# ==============================================================================
# PIPELINE CALL SCRIPT (Year Isolated)
# ==============================================================================
DATA_YEAR = 2022

mu_ind_to_ind = extract_statewide_ind_to_ind_flow(
    dop_table_dict, 
    global_party_dict, 
    data_year=DATA_YEAR
)

dop_lookup = impute_and_build_dop_priors(
    IND_DOP_flows_2022, 
    elec_df, 
    data_year=DATA_YEAR, 
    S_local=75, 
    S_borrowed=35
)

simulation_results = simulate_electorate_counterfactual(
    data_year=DATA_YEAR,
    elec_name=target_electorate,
    region_name=target_region,
    elec_baseline_data=elec_baseline_data,
    counterfactual_counts=counterfactual_counts,
    new_candidates=new_candidates,
    pool_model=pool_model,
    split_model=split_model,
    dop_lookup=dop_lookup,
    mu_ind_to_ind=mu_ind_to_ind,
    M=1000,
)

import pdb; pdb.set_trace()