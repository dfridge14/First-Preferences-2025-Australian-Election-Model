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
final_model_df = successful_matches.copy()
final_model_df['is_repeat'] = is_repeat
#final_model_df = successful_matches[~is_repeat].drop(columns=['Simple_Temp']).copy()
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

final_model_df['Is_Former_Councillor'] = final_model_df.apply(
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

final_model_df['Candidate_Archetype'] = final_model_df['Candidate'].apply(get_candidate_archetype)


to_plot = 0

if to_plot:

    # 1. Create a categorical grouping column for the facets
    final_model_df['Campaign_Phase'] = np.where(
        final_model_df['Campaign_Length_Days'] <= 15, 
        '<= 15 Days (Ballot Draw / Late)', 
        '> 15 Days (Early Campaign)'
    )

    # 2. Explicitly create a standard matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    phases = ['<= 15 Days (Ballot Draw / Late)', '> 15 Days (Early Campaign)']

    for i, phase in enumerate(phases):
        subset = final_model_df[final_model_df['Campaign_Phase'] == phase]
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

# final_model_df.loc[final_model_df['Candidate'].isin(final_model_df.loc[final_model_df['is_repeat']==False,].groupby('Candidate')['Candidate'].count().sort_values().tail(6).index),].sort_values(by='Candidate')


#1a. Save page electorate and primary vote performance for each is_repeat IND (CLEAN: double-doing the is_repeat section)

# 1. Clean candidate name helper
def clean_cand_name(name):
    return " ".join(str(name).strip().lower().split())

final_model_df['Cand_Clean'] = final_model_df['Candidate'].apply(clean_cand_name)

# 2. Extract prior run history across ALL electorates
is_repeat_list = []
past_vote_list = []
past_elec_list = []

for _, row in final_model_df.iterrows():
    c_clean = row['Cand_Clean']
    curr_yr = row['Year']
    
    # Match candidate anywhere in Victoria strictly before current election year
    priors = final_model_df[
        (final_model_df['Cand_Clean'] == c_clean) & 
        (final_model_df['Year'] < curr_yr)
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

final_model_df['is_repeat'] = is_repeat_list
final_model_df['Past_Vote_Share'] = past_vote_list
final_model_df['Past_Electorate'] = past_elec_list

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

train_model_df = final_model_df[~final_model_df.set_index(['Year', 'Electorate']).index.isin(excluded_contests)].copy()

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

# 4. Perform model

# ==========================================
# PHASE 1: DATA PREP & FILTERING
# ==========================================
def prepare_training_data(cand_df, elec_df):
    """Filters incumbents and builds the electorate-level aggregations."""
    
    # 1. Drop Incumbent MPs (they skew the insurgency model)
    train_cand = cand_df

    # 1. Define the baseline nomination cut-off per year
  
    # Calculate Effective Days: 0 if at/below minimum, otherwise keep raw days
    baseline_days = {2014: 13, 2018: 14, 2022: 15}
    train_cand['Min_Days'] = train_cand['Year'].map(baseline_days)
    train_cand['Effective_Days'] = np.where(train_cand['Campaign_Length_Days'] <= train_cand['Min_Days'], 0, train_cand['Campaign_Length_Days'])
    
    # Log transform (add 1 to safely handle the zeros)
    train_cand['Log_Days'] = np.log(train_cand['Effective_Days'] + 1)
    
    # 2. Aggregate to Electorate Level (The Pool)
    pool_df = train_cand.groupby(['Year', 'Electorate']).agg(
        Total_IND_Share=('Primary_Vote_Share', 'sum'),
        Has_Councillor=('Is_Former_Councillor', 'max'),
        Sum_Log_Days=('Log_Days', 'sum')
    ).reset_index()
    
    # Merge with Electorate structural baselines (Margin, Lag Vote)
    pool_df = pool_df.merge(elec_df, on=['Year', 'Electorate'], how='left')
    pool_df['Log_Lag_Vote'] = np.log(1 + pool_df['Lag_Challenger_IND_Vote'])
    
    # 3. Calculate candidate share of the pool (for Stage 2)
    train_cand = train_cand.merge(pool_df[['Year', 'Electorate', 'Total_IND_Share']], on=['Year', 'Electorate'])
    train_cand['Share_of_Pool'] = (train_cand['Primary_Vote_Share'] / train_cand['Total_IND_Share']).clip(lower=0.01)
    
    train_cand = train_cand.drop(columns = ['Min_Days', 'Effective_Days'])

    return train_cand, pool_df


# ==========================================
# PHASE 2: FIT STATISTICAL MODELS
# ==========================================
def fit_generative_models(train_cand, pool_df):
    """Fits the GLM for the Total Pool and OLS for the Split."""
    
    # --- STAGE 1: The Pool Model (Gamma/Log GLM) ---
    # Convert share (0-100) to proportion (0-1) for Gamma regression
    y_pool = pool_df['Total_IND_Share'] / 100.0 
    X_pool_cols = pool_df[['Log_Lag_Vote', 'Has_Councillor', 'Sum_Log_Days']].copy()
    X_pool_cols = X_pool_cols.astype(float)
    X_pool = sm.add_constant(X_pool_cols)
    
    pool_model = sm.GLM(y_pool, X_pool, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    
    # --- STAGE 2: The Allocation Split Model ---
    # We only fit the split model on NEW candidates. 
    # Repeat contenders will use an anchor value based on their past vote.
    new_cands = train_cand[train_cand['is_repeat'] == 0].copy()
    y_split = np.log(new_cands['Share_of_Pool'])
    X_split = sm.add_constant(new_cands[['Is_Former_Councillor', 'Log_Days']].astype(float))
    
    split_model = sm.OLS(y_split, X_split).fit()
    
    return pool_model, split_model



train_cand, pool_df = prepare_training_data(train_model_df, elec_df)
pool_model, split_model = fit_generative_models(train_cand, pool_df.loc[pool_df['Year']!=2010])

import pdb; pdb.set_trace()

# ==========================================
# PHASE 3: COMPUTE BAYESIAN DOP PRIORS
# ==========================================
def build_dop_priors(dop_flows_df, kappa_reg=25, kappa_obs=75):
    """Calculates Regional average flows and prepares the Dirichlet alphas."""
    
    flow_cols = ['Flow_ALP', 'Flow_LNP', 'Flow_GRN', 'Flow_R_MIN', 'Flow_L_MIN']
    
    # 1. Regional Priors (mu_reg)
    reg_priors = dop_flows_df.groupby('Region')[flow_cols].mean()
    
    # 2. Build the lookup dictionary
    dop_lookup = {}
    for region in reg_priors.index:
        mu_reg = reg_priors.loc[region].values
        alpha_reg = kappa_reg * mu_reg
        dop_lookup[region] = {'regional_alpha': alpha_reg, 'electorates': {}}
        
        # Local updates
        local_flows = dop_flows_df[dop_flows_df['Region'] == region]
        for _, row in local_flows.iterrows():
            mu_obs = row[flow_cols].values
            # Bayesian update: Posterior = Prior + (Kappa_obs * Obs_mean)
            alpha_post = alpha_reg + (kappa_obs * mu_obs)
            dop_lookup[region]['electorates'][row['Electorate']] = alpha_post
            
    return dop_lookup, flow_cols

# ==========================================
# PHASE 4: MONTE CARLO SIMULATION ENGINE
# ==========================================
def simulate_electorate(elec_data, candidates, pool_model, split_model, dop_lookup, flow_cols, M=1000):
    """
    Runs M simulations of an election in a single electorate.
    elec_data: dict containing 'Region', 'Electorate', 'Lag_Challenger_IND_Vote', 'Margin_2PP', plus P_ALP, P_LNP, etc.
    candidates: list of dicts for each IND running.
    """
    # 1. Electorate Features
    has_councillor = max([c['Is_Sitting_Councillor'] for c in candidates])
    sum_log_days = sum([np.log(max(1, c['Campaign_Length_Days'])) for c in candidates])
    log_lag = np.log(1 + elec_data['Lag_Challenger_IND_Vote'])
    
    x_pool = np.array([1, log_lag, elec_data['Margin_2PP'], has_councillor, sum_log_days])
    
    # 2. Extract Covariance Matrices for Multivariate Normal Draws
    pool_mean, pool_cov = pool_model.params, pool_model.cov_params()
    split_mean, split_cov = split_model.params, split_model.cov_params()
    
    # 3. Retrieve Dirichlet Alpha
    region = elec_data['Region']
    elec = elec_data['Electorate']
    if elec in dop_lookup.get(region, {}).get('electorates', {}):
        alpha = dop_lookup[region]['electorates'][elec] # Use local posterior
    else:
        alpha = dop_lookup.get(region, {}).get('regional_alpha', np.array([5, 5, 5, 5, 5])) # Fallback
    
    # 4. Simulation Loop
    sim_results = []
    baseline_P = np.array([elec_data[f'P_{k}'] for k in ['ALP', 'LNP', 'GRN', 'R_MIN', 'L_MIN']])
    
    for m in range(M):
        # A. Draw Pool Parameters & Calculate Total Pool (in %)
        b_pool = np.random.multivariate_normal(pool_mean, pool_cov)
        T_e = np.exp(np.dot(x_pool, b_pool)) * 100 
        
        # B. Draw Split Parameters & Allocate
        b_split = np.random.multivariate_normal(split_mean, split_cov)
        
        scores = []
        for c in candidates:
            if c['Is_Repeat_Contender']:
                # Anchoring: Base score off their past performance log
                score = np.exp(np.log(max(1, c['Past_Vote_Share']))) 
            else:
                x_split = np.array([1, c['Is_Sitting_Councillor'], np.log(max(1, c['Campaign_Length_Days']))])
                score = np.exp(np.dot(x_split, b_split))
            scores.append(score)
            
        sum_scores = sum(scores)
        pi_i = np.array(scores) / sum_scores
        V_i = T_e * pi_i # Final predicted primary for each IND in this sim
        
        # C. Draw Cannibalisation Matrix (Dirichlet)
        W = np.random.dirichlet(alpha, size=len(candidates)) # Shape: (Num INDs, 5 Parties)
        
        # D. Base Rate Weighting & Deduction
        P_new = baseline_P.copy()
        
        for idx, v in enumerate(V_i):
            w_raw = W[idx]
            w_adj = (w_raw * P_new) / np.sum(w_raw * P_new) # Modulate by available party votes
            
            deductions = w_adj * v
            P_new = P_new - deductions
            
        # E. Simplex Projection (ALR Equivalent Clipping)
        P_final = np.maximum(0, P_new)
        total_majors_expected = 100 - T_e
        if np.sum(P_final) > 0:
            P_final = (P_final / np.sum(P_final)) * total_majors_expected
        
        sim_results.append({
            'Sim_ID': m,
            'Total_IND_Pool': T_e,
            'V_i': V_i,
            'Final_Majors': P_final
        })
        
    return sim_results