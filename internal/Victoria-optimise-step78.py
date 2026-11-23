import time # NEW
import pickle # NEW
import multiprocessing # NEW
from collections import defaultdict # NEW
import numpy as np # NEW
import pandas as pd # NEW
from tqdm import tqdm # NEW

import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import re
import time
import copy
import json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from pathlib import Path

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from scipy.optimize import minimize
from collections import defaultdict, Counter



from DOPTable import DOPTable


rng = np.random.default_rng(seed=2026)

import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler



# =============================================================================
# 1. POLL BUCKET HELPER & VECTORIZED STEP 7-8 SIMULATOR
# =============================================================================

COAL_PARTIES = {'LP', 'NP', 'LNP', 'COAL'} 
LP_NP_DIRICHLET_ALPHA = 4.1783

def get_poll_bucket_hist(cand: str, data_year: int) -> str: # NEW
    """Maps candidate code to macro polling bucket ('ALP', 'COAL', 'GRN', 'OTH')."""
    p = str(cand).split('_')[0] if '_' in str(cand) else str(cand) # NEW
    if p == 'ALP': # NEW
        return 'ALP' # NEW
    if p in COAL_PARTIES: # NEW
        return 'COAL' # NEW
    if p == 'GRN': # NEW
        return 'GRN' # NEW
    return 'OTH' # NEW



def predraw_polling_residual_simulation_error(
    days_to_election: float,
    raw_poll_covm: np.ndarray,
    raw_res_covm: np.ndarray,
    num_simulations: int,
    NUM_DISTRICTS: int,
    rng: np.random.Generator,
    data_year: int = 2022,
    alr_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    if alr_cols is None:
        import pdb; pdb.set_trace()

    n_alr = len(alr_cols)

    OPT_EXP_DECAY_PARAMETERS = {
        'OTH': [0.18562381, 0.28067983, 0.05079430],
        'GRN': [0.15782239, 0.24219055, 0.02555061],
        'ALP': [0.09403604, 0.20026747, 0.02794699],
        'ON':  [0.19570000, 0.28000000, 0.02555000]
    }

    # 1. Load pickle


    # Map party names to raw 5x5 indices in case pickle stores numpy arrays instead of DataFrames
    # (Adjust RAW_5X5_INDEX if your numpy array uses a different order than [ALP, GRN, ON, IND, OTH])
    RAW_5X5_INDEX = {'ALP': 0, 'GRN': 1, 'ON': 2, 'IND': 3, 'OTH': 4}
    keep_idx = [RAW_5X5_INDEX[p] for p in alr_cols]

    base_poll = (
        raw_poll_covm.loc[alr_cols, alr_cols].values
        if isinstance(raw_poll_covm, pd.DataFrame)
        else raw_poll_covm[np.ix_(keep_idx, keep_idx)]
    )
    C_elec = raw_res_covm

    # 2. Time-decay the statewide polling matrix (4x4 in 2022, 3x3 in <=2018)
    stds_base = np.sqrt(np.diag(base_poll))
    R_poll = base_poll / np.outer(stds_base, stds_base)

    s_diag = [OPT_EXP_DECAY_PARAMETERS[p][1] + (OPT_EXP_DECAY_PARAMETERS[p][0] - OPT_EXP_DECAY_PARAMETERS[p][1]) * np.exp(-OPT_EXP_DECAY_PARAMETERS[p][2] * days_to_election) for p in alr_cols]

    S = np.diag(s_diag)
    C_poll_unclean = S @ R_poll @ S
    eigvals, eigvecs = np.linalg.eigh(C_poll_unclean)
    C_poll = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T

    # 3. Pre-draw simulation error tensors of shape (num_simulations, n_alr) and (num_simulations, NUM_DISTRICTS, n_alr)
    sim_poll_errors = rng.multivariate_normal(np.zeros(n_alr), C_poll, size=num_simulations)
    sim_elec_errors = rng.multivariate_normal(np.zeros(n_alr), C_elec, size=(num_simulations, NUM_DISTRICTS))

    return sim_poll_errors, sim_elec_errors



def compute_statewide_alr_shifts(
    cf_c0_step5_dict: dict,
    kalman_poll: dict,
    dop_table_dict: dict,
    data_year: int,
    ref_col: str = 'COAL'
) -> dict:
    """
    Computes the weighted statewide Step 5 aggregate prior across all districts and calculates the ΔALR shift vector against the normalized Kalman targets.
    """
    bucket_raw_votes = {b: 0.0 for b in kalman_poll.keys()}
    total_statewide_votes = 0.0

    for div, step5 in cf_c0_step5_dict.items():
        # Turnout lookup matching existing logic
        if data_year == 2022 and div == 'Narracan':
            district_total = 45132.0
        elif div in dop_table_dict and 'TOTAL' in dop_table_dict[div].columns:
            district_total = float(dop_table_dict[div]['TOTAL'].iloc[0])
        else:
            district_total = 40000.0

        for cand, vote_prop in step5.items():
            if isinstance(vote_prop, (int, float)) and pd.notna(vote_prop):
                bucket = get_poll_bucket_hist(cand, data_year)
                if bucket in bucket_raw_votes:
                    raw_vote = vote_prop * district_total
                    bucket_raw_votes[bucket] += raw_vote
                    total_statewide_votes += raw_vote

    # Normalize prior and target polls
    prior_shares = {k: v / total_statewide_votes for k, v in bucket_raw_votes.items()}
    poll_norm = sum(kalman_poll.values())
    poll_shares = {k: v / poll_norm for k, v in kalman_poll.items()}

    # Calculate ΔALR shifts against reference column
    alr_shifts = {}
    for b in kalman_poll.keys():
        prior_alr = np.log((prior_shares[b] + 1e-5) / (prior_shares[ref_col] + 1e-5))
        poll_alr = np.log((poll_shares[b] + 1e-5) / (poll_shares[ref_col] + 1e-5))
        alr_shifts[b] = poll_alr - prior_alr

    return alr_shifts


def clean_ind_name(raw_name: str) -> str:
    """
    Forces any name into the strict IND_First_Last convention.
    Drops middle words. Preserves non-space punctuation (e.g., J.D. or Tosh-Jake).
    """
    if pd.isna(raw_name) or str(raw_name).strip() == 'nan':
        return "UNKNOWN_PARTY"

    name = re.sub(r'^IND_', '', str(raw_name).strip())
    tokens = re.split(r'[\s_]+', name)

    if len(tokens) > 1:
        return f"IND_{tokens[0].title()}_{tokens[-1].title()}"
    return f"IND_{tokens[0].title()}"


def sync_dop_table_inds(dop_table_dict: dict, la_candidates_df: pd.DataFrame) -> dict:
    """
    Translates 'IND', 'IND1', 'IND2', etc. in DOP_table_dict DataFrames to
    canonical 'IND_First_Last' headers using clean_ind_name.
    """
    synced_dop_dict = {}
    cand_df = la_candidates_df.copy()
    cand_df['div_nm_clean'] = cand_df['div_nm'].astype(str).str.strip()
    cand_df['ballot_order_num'] = pd.to_numeric(cand_df['ballot_order'], errors='coerce')

    for div, df in dop_table_dict.items():
        div_clean = str(div).strip()
        div_cands = cand_df[cand_df['div_nm_clean'] == div_clean].copy()

        inds = (
            div_cands[div_cands['PartyAb'].astype(str).str.strip().str.startswith('IND')]
            .sort_values('ballot_order_num', kind='mergesort')
        )

        rename_map = {}
        for idx, (_, row) in enumerate(inds.iterrows()):
            raw_full = f"{str(row['first_name']).strip()} {str(row['surname']).strip()}"
            full_ind_name = clean_ind_name(raw_full)

            # Map positional IND1, IND2, IND3... tags
            rename_map[f"IND{idx + 1}"] = full_ind_name
            if idx == 0:
                rename_map["IND"] = full_ind_name

            # Also map explicit PartyAb tag (e.g. 'IND2') if present
            party_ab = str(row['PartyAb']).strip()
            if party_ab.startswith('IND') and party_ab != 'IND':
                rename_map[party_ab] = full_ind_name

        # Canonicalize any pre-formatted IND_* columns already on the DataFrame
        for col in df.columns:
            col_str = str(col).strip()
            if col_str.startswith('IND_') and col_str not in rename_map:
                rename_map[col] = clean_ind_name(col_str)

        synced_dop_dict[div] = df.rename(columns=rename_map)

    return synced_dop_dict

# =============================================================================
# STAGE A: BUILD STEP6_BY_YEAR & RESULTS_FP_BY_YEAR IN MEMORY (RUNS ONCE)
# =============================================================================

os.chdir(Path.home() / 'Australian Election/Victorian Election')

days_to_election = 0 # NEW
with open(Path.home() / "Australian Election/Farrer_byelection_covms.pkl", "rb") as f: # NEW
    covm_data = pickle.load(f) # NEW
raw_poll_covm = covm_data["Nat_poll_covm"] # NEW

with open("add_new_IND_models_and_priors.pkl", "rb") as f: # NEW
    add_new_IND_data = pickle.load(f) # NEW
global_party_dict = add_new_IND_data["global_party_dict"] # NEW

CovM = np.array([ # NEW
    [0.0307, 0.0146, 0.0136], # NEW
    [0.0146, 0.0397, 0.0211], # NEW
    [0.0136, 0.0211, 0.1230] # NEW
]) # NEW

CovM = np.array([
    [0.0388, 0.0209, 0.0207],
    [0.0209, 0.0584, 0.0250],
    [0.0207,  0.0250, 0.1627]
])

parties = ['ALP', 'GRN', 'OTH'] # NEW
raw_res_covm = pd.DataFrame(CovM, index=parties, columns=parties) # NEW

def load_kalman_poll(csv_path: str, election_year: int, ref_col: str = 'COAL') -> Tuple[dict, List[str]]: # NEW
    df = pd.read_csv(csv_path, index_col='election_year') # NEW
    row = df.loc[election_year].dropna() # NEW
    kalman_poll = {k: float(v) for k, v in row.items() if float(v) > 0.0} # NEW
    alr_cols = [k for k in kalman_poll.keys() if k != ref_col] # NEW
    return kalman_poll, alr_cols # NEW

STEP6_BY_YEAR = {} # NEW
RESULTS_FP_BY_YEAR = {} # NEW

for election_year in [2010, 2014, 2018, 2022]:  # <-- FIXED: 2010..2022 (not 2006..2018) # NEW
    yr_str = str(election_year) # NEW
    data_year = election_year - 4               # <-- 2006, 2010, 2014, 2018 # NEW
    ALR_COLS = ['ALP', 'GRN', 'OTH'] # NEW

    # 1. Load saved Step 6 simulations (list of 1,000 dicts: [{div: {cand: share}}, ...])
    with open(f'pre_poll_sims_step6_for_{election_year}.pkl', 'rb') as f: # NEW
        Step6_results = pickle.load(f) # NEW

        Step6_results = [ # NEW
            { # NEW
                div: { # NEW
                    (clean_ind_name(k) if str(k).startswith('IND') else k): v # NEW
                    for k, v in cand_dict.items() # NEW
                } # NEW
                for div, cand_dict in sim_dict.items() # NEW
            } # NEW
            for sim_dict in Step6_results # NEW
        ] # NEW

    with open(f"VIC-{data_year}/VIC-{data_year}-IA-model-params.json", 'r') as f: # NEW
        ia_params_dict = json.load(f) # NEW
    with open(f'inc_next_lookup_for_{election_year}.pkl', 'rb') as f: # NEW
        inc_next_lookup = pickle.load(f) # NEW
    with open(f'ia_tracker_dict_for_{election_year}.pkl', 'rb') as f: # NEW
        ia_tracker_dict = pickle.load(f) # NEW

    kalman_poll, _ = load_kalman_poll('kalman_polls.csv', election_year, ref_col='COAL') # NEW

    demo_df = pd.read_csv(f'VIC-{election_year}/Demographic-classification-{election_year}.csv') # NEW
    demo_lookup = demo_df.set_index('district_name')['classification'].to_dict() # NEW

    with open(f'VIC-{data_year}/{data_year}_complete_DOP_table_dict.pkl', 'rb') as f:
        DOP_table_dict = pickle.load(f)
        if data_year == 2022:
            del DOP_table_dict['Narracan'] # FIX: Narracan dealt with separately!
    with open(f'VIC-{election_year}/{election_year}_complete_DOP_table_dict.pkl', 'rb') as f:
        DOP_table_dict_next = pickle.load(f)
        if election_year == 2022:
            del DOP_table_dict_next['Narracan'] # FIX: Narracan dealt with separately!

    LA_Candidates = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Candidates-Formatted.csv')
    DOP_table_dict = sync_dop_table_inds(DOP_table_dict, LA_Candidates)
    LA_Candidates_next =  pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
    DOP_table_dict_next = sync_dop_table_inds(DOP_table_dict_next, LA_Candidates_next)

    # 3. Extract actual Count 0 FP percentages (0..100) from DOP_table_dict for evaluation
    
    Results_fp_dict = {} # NEW
    for div, raw_df in DOP_table_dict_next.items(): # NEW
        if election_year == 2022 and div == 'Narracan': # NEW
            continue # NEW
        table = DOPTable(raw_df)
        c0 = table.count_0        
        tot = sum(c0.values())
        if tot > 0:
            Results_fp_dict[div] = pd.DataFrame([{c: (v / tot) * 100.0 for c, v in c0.items()}])



    target_districts = [d for d in Step6_results[0].keys() if not (election_year == 2022 and d == 'Narracan')]

    # 5. Pre-draw Gaussian poll errors & base Gaussian seat errors ONCE for this election year
    rng_predraw = np.random.default_rng(2026 + int(election_year)) # NEW
    sim_poll_errors_base, raw_elec_mvn_base = predraw_polling_residual_simulation_error( # NEW
        days_to_election=days_to_election, # NEW
        raw_poll_covm=raw_poll_covm, # NEW
        raw_res_covm=raw_res_covm.loc[ALR_COLS, ALR_COLS].to_numpy(), # NEW
        num_simulations=len(Step6_results), # NEW
        NUM_DISTRICTS=len(target_districts), # NEW
        rng=rng_predraw, # NEW
        data_year=data_year, # NEW
        alr_cols=ALR_COLS # NEW
    ) # NEW

    # 6. Precompute statewide alr_shifts ONCE for each of the 1,000 simulations
    alr_shifts_list = [ # NEW
        compute_statewide_alr_shifts( # NEW
            cf_c0_step5_dict=sim_c0_dict, # NEW
            kalman_poll=kalman_poll, # NEW
            dop_table_dict=DOP_table_dict, # NEW
            data_year=data_year, # NEW
            ref_col='COAL' # NEW
        ) # NEW
        for sim_c0_dict in Step6_results # NEW
    ] # NEW

    if yr_str == '2022':
        import pdb; pdb.set_trace()

    STEP6_BY_YEAR[yr_str] = { # NEW
        "step6_c0": Step6_results, # NEW
        "alr_shifts_list": alr_shifts_list, # NEW
        "sim_poll_errors": sim_poll_errors_base, # NEW
        "raw_elec_mvn": raw_elec_mvn_base, # NEW
        "target_districts": target_districts, # NEW
        "alr_cols": ALR_COLS, # NEW
        "data_year": data_year, # NEW
        "global_party_dict": global_party_dict, # NEW
        "ia_tracker_dict": ia_tracker_dict, # NEW
        "inc_next_lookup": inc_next_lookup, # NEW
        "ia_params_dict": ia_params_dict, # NEW
        "demo_lookup": demo_lookup # NEW
    } # NEW
    RESULTS_FP_BY_YEAR[yr_str] = Results_fp_dict # NEW
    print(f"Loaded & prepared {election_year} ({len(target_districts)} districts, {len(Step6_results)} sims)") # NEW


# =============================================================================
# STAGE A: BUILD STEP6_BY_YEAR & RESULTS_FP_BY_YEAR IN MEMORY (RUNS ONCE)
# =============================================================================
if 0: 
    days_to_election = 0
    with open(Path.home() / "Australian Election/Farrer_byelection_covms.pkl", "rb") as f:
        covm_data = pickle.load(f)

    raw_poll_covm = covm_data["Nat_poll_covm"]

    for election_year in [2010,2014,2018,2022]:

        data_year = election_year - 4
        ALR_COLS = ['ALP','GRN','OTH'] 


        # Load your already-saved Step 6 simulations (or use Step6_results directly in memory)
        with open(f'pre_poll_sims_step6_for_{election_year}.pkl', 'rb') as f: # NEW
            Step6_results = pickle.load(f) # NEW

        with open(f"VIC-{data_year}/VIC-{data_year}-IA-model-params.json", 'r') as f:
            ia_params_dict = json.load(f)

        with open(f'inc_next_lookup_{election_year}.pkl', 'rb') as f: # NEW
            inc_next_lookup = pickle.load(f) # NEW
        with open(f'ia_tracker_dict_{election_year}.pkl', 'rb') as f: # NEW
            ia_tracker_dict = pickle.load(f) # NEW

            
        def load_kalman_poll(csv_path: str, election_year: int, ref_col: str = 'COAL') -> Tuple[dict, List[str]]:
            df = pd.read_csv(csv_path, index_col='election_year')
            row = df.loc[election_year].dropna()
            kalman_poll = {k: float(v) for k, v in row.items() if float(v) > 0.0}
            
            # Dynamic ALR columns (4 in 2022, 3 in <=2018), excluding reference 'COAL'
            alr_cols = [k for k in kalman_poll.keys() if k != ref_col]
            return kalman_poll, alr_cols

        kalman_poll, alr_cols = load_kalman_poll('kalman_polls.csv', election_year, ref_col = 'COAL')

        demo_df = pd.read_csv(f'VIC-{election_year}/Demographic-classification-{election_year}.csv')
        demo_lookup = demo_df.set_index('district_name')['classification'].to_dict()

        with open("add_new_IND_models_and_priors.pkl", "rb") as f:
            add_new_IND_data = pickle.load(f)
        global_party_dict = add_new_IND_data['global_party_dict']

        CovM = np.array([
            [0.0307, 0.0146, 0.0136],
            [0.0146, 0.0397, 0.0211],
            [0.0136, 0.0211, 0.1230]
        ])

        parties = ['ALP', 'GRN', 'OTH']
        raw_res_covm = pd.DataFrame(CovM, index=parties, columns=parties)

        with open(f'VIC-{data_year}/{data_year}_complete_DOP_table_dict.pkl', 'rb') as f:
            DOP_table_dict = pickle.load(f)
            if data_year == 2022:
                del DOP_table_dict['Narracan'] # FIX: Narracan dealt with separately!
        with open(f'VIC-{election_year}/{election_year}_complete_DOP_table_dict.pkl', 'rb') as f:
            DOP_table_dict_next = pickle.load(f)
            if election_year == 2022:
                del DOP_table_dict_next['Narracan'] # FIX: Narracan dealt with separately!

        LA_Candidates = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Candidates-Formatted.csv')
        DOP_table_dict = sync_dop_table_inds(DOP_table_dict, LA_Candidates)
        LA_Candidates_next =  pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
        DOP_table_dict_next = sync_dop_table_inds(DOP_table_dict_next, LA_Candidates)
        
        target_districts = [d for d in Step6_results[0].keys() if not (election_year == 2022 and d == 'Narracan')]

        # 1. Pre-draw Gaussian poll errors and base Gaussian seat errors ONCE for this election
        #    using your exact predraw_polling_residual_simulation_error function:
        rng_predraw = np.random.default_rng(2026 + int(election_year)) # NEW
        sim_poll_errors_base, raw_elec_mvn_base = predraw_polling_residual_simulation_error( # NEW
            days_to_election=days_to_election, # NEW
            raw_poll_covm=raw_poll_covm, # NEW
            raw_res_covm=raw_res_covm, # NEW
            num_simulations=len(Step6_results), # NEW
            NUM_DISTRICTS=88 - (data_year == 2018), # NEW
            rng=rng_predraw, # NEW
            data_year=data_year, # NEW
            alr_cols=ALR_COLS # NEW
        ) # NEW

        # 2. Precompute statewide alr_shifts ONCE for each of the 1,000 simulations
        alr_shifts_list = [ # NEW
            compute_statewide_alr_shifts( # NEW
                cf_c0_step5_dict=sim_c0_dict, # NEW
                kalman_poll=kalman_poll, # NEW
                dop_table_dict=DOP_table_dict, # NEW
                data_year=data_year, # NEW
                ref_col='COAL' # NEW
            ) # NEW
            for sim_c0_dict in Step6_results # NEW
        ] # NEW

        # 3. Extract actual Count 0 FP percentages (0..100) from DOP_table_dict for evaluation
        Results_fp_dict = {} # NEW
        for div, dop_table in DOP_table_dict_next.items(): # NEW
            if election_year == 2022 and div == 'Narracan': # NEW
                continue # NEW
            c0 = dop_table.count_0 # NEW
            tot = sum(c0.values()) # NEW
            if tot > 0: # NEW
                Results_fp_dict[div] = pd.DataFrame([{c: (v / tot) * 100.0 for c, v in c0.items()}]) # NEW

        # 4. Save all loop metadata needed for Steps 7-8 into a companion pickle
        step6_meta = { # NEW
            "alr_shifts_list": alr_shifts_list,           # List of 1,000 dicts # NEW
            "sim_poll_errors": sim_poll_errors_base,      # Shape: (1000, n_alr) # NEW
            "raw_elec_mvn": raw_elec_mvn_base,            # Shape: (1000, n_divs, n_alr) # NEW
            "target_districts": list(target_districts),   # Exact district order matching d_idx # NEW
            "alr_cols": list(ALR_COLS),                   # ['ALP', 'GRN', 'OTH'] # NEW
            "data_year": data_year, # NEW
            "global_party_dict": global_party_dict, # NEW
            "ia_tracker_dict": ia_tracker_dict, # NEW
            "inc_next_lookup": inc_next_lookup, # NEW
            "ia_params_dict": ia_params_dict, # NEW
            "demo_lookup": demo_lookup, # NEW
            "Results_fp_dict": Results_fp_dict # NEW
        } # NEW

        with open(f'step6_meta_for_{election_year}.pkl', 'wb') as f: # NEW
            pickle.dump(step6_meta, f, protocol=pickle.HIGHEST_PROTOCOL) # NEW





def apply_district_alr_shift(
    district_c0: dict,
    alr_shifts: dict,
    data_year: int,
    global_party_dict: dict = None,
    w_poll: float = 0.50,
    DIRICHLET_OTH_SPLIT_POLL: float = 25.0,
    DIRICHLET_ALPHA: float = 25.0,
    simulate: bool = False,
    rng: np.random.Generator = None
) -> dict:
    """
    Applies the exponentiated ALR shifts to a single district dictionary.
    """
    # 1. Floor protection to eliminate log-singularities
    cleaned_c0 = {c: max(v, 0.005) for c, v in district_c0.items()}
    tot_cleaned = sum(cleaned_c0.values())
    cleaned_c0 = {c: v / tot_cleaned for c, v in cleaned_c0.items()}
    shifted_c0 = {}

    # --- SECTION 1: 2022 Sub-OTH Relative Tilt (delta_g) with Shrinkage & Uncertainty ---
    
    oth_sub_deltas = {}

    # --- SECTION 2: Group by Category and Apply ALR Shift ---
    sub_category_priors = {}
    cand_to_sub = {}
    
    for cand, vote in cleaned_c0.items():
        bucket = get_poll_bucket_hist(cand, data_year)
        if bucket == 'OTH' and data_year == 2022:
            if str(cand).startswith('IND') or str(cand).startswith('TWP'):
                sub_g = 'OTH_IND'
            else:
                party = cand.split('_')[0] if '_' in cand else cand
                b_ideo = global_party_dict.get(cand, global_party_dict.get(party, 'Centre')) if global_party_dict else 'Centre'
                sub_g = f"OTH_{b_ideo}" if b_ideo in ('Left', 'Right', 'Centre') else 'OTH_Centre'
        else:
            sub_g = bucket
            
        cand_to_sub[cand] = sub_g
        sub_category_priors[sub_g] = sub_category_priors.get(sub_g, 0.0) + vote

    unnorm_sub = {}
    for sub_g, prior_mass in sub_category_priors.items():
        if sub_g.startswith('OTH_'):
            shift = alr_shifts.get('OTH', 0.0) + oth_sub_deltas.get(sub_g.replace('OTH_', ''), 0.0)
        else:
            shift = alr_shifts.get(sub_g, 0.0)
        unnorm_sub[sub_g] = prior_mass * np.exp(shift)
        
    tot_shifted = sum(unnorm_sub.values())
    post_sub_mass = {g: v / tot_shifted for g, v in unnorm_sub.items()} if tot_shifted > 1e-9 else sub_category_priors

    # --- SECTION 3: Disaggregate Down to Candidates via Prior Proportions (C) ---
    shifted_c0 = {}
    for sub_g, B_mass in post_sub_mass.items():
        cands_in_g = [c for c, g in cand_to_sub.items() if g == sub_g]
        
        if len(cands_in_g) == 1:
            shifted_c0[cands_in_g[0]] = B_mass
            continue
            
        # Extract strict intra-category prior proportions
        w_prior = np.array([cleaned_c0[c] for c in cands_in_g], dtype=float)
        w_prior /= w_prior.sum()
        
        # Apply Dirichlet Jitter within multi-candidate sub-categories (skip OTH_IND, already split in Step 5)
        S_jitter = DIRICHLET_ALPHA
        if sub_g == 'COAL' and 'S_coal_jitter' in locals():
            S_jitter = LP_NP_DIRICHLET_ALPHA
            
        if simulate and rng is not None and S_jitter is not None and S_jitter > 0 and sub_g != 'OTH_IND':
            alphas = np.maximum(w_prior * S_jitter, 1e-3)
            props = rng.dirichlet(alphas)
        else:
            props = w_prior
            
        for idx, c in enumerate(cands_in_g):
            shifted_c0[c] = B_mass * float(props[idx])
            
    return shifted_c0


def execute_reapply_incumbency(
    div: str,
    cf_c0_step5: Dict[str, float],
    ia_tracker_list: Optional[List[Dict[str, Any]]],
    inc_next_info_list: Optional[List[Dict[str, Any]]],
    params_dict: dict,          # Loaded from 'VIC-IA-Model-Params.json'
    demo_class: str,            # E.g., 'Inner Metropolitan'
    simulate: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    """Step 6: Re-applies 2026 Incumbency Advantage dynamically at Count 0."""
    if rng is None: rng = np.random.default_rng()
        
    if not inc_next_info_list or inc_next_info_list[0].get('is_open', True):
        return cf_c0_step5.copy()

    cf_c0_step6 = cf_c0_step5.copy()
    sigma_resid = params_dict['sigma_resid']
    coefs = params_dict['coefficients']
    
    # Sequentially apply IA for each incumbent listed in the seat
    for inc_next_info in inc_next_info_list:
        next_party = inc_next_info['party']
        next_name_clean = str(inc_next_info.get('inc_clean_name', '')).strip().lower()
        next_tenure = inc_next_info.get('years_tenure', np.nan)

        # Look for this specific candidate's tracker in the list from Step 1
        matched_tracker = None
        if ia_tracker_list:
            for tracker in ia_tracker_list:
                t_name = str(tracker.get('inc_name', '')).lower() 
                t_party = str(tracker.get('inc_party', ''))
                if t_name and next_name_clean and t_name == next_name_clean and t_party == next_party:
                    matched_tracker = tracker
                    break

        is_returning = bool(matched_tracker is not None)


        # 1. Calculate Magnitude using JSON coefficients
        if is_returning:
            #print('returning', div)
            base_ia = float(matched_tracker['stripped_ia'])
            beta_tenure = float(coefs.get('log_tenure', 0.0))
            raw_old_t = matched_tracker.get('tenure', 1.0)
            old_tenure = max(1.0, float(raw_old_t)) if pd.notna(raw_old_t) else 1.0
            tenure_next = next_tenure if (pd.notna(next_tenure) and next_tenure > 0) else (old_tenure + 4.0)

            tenure_delta = beta_tenure * (np.log(max(1.0, tenure_next)) - np.log(old_tenure))
            
            campaign_shock = rng.normal(0, sigma_resid) if simulate else 0.0
            target_ia = max(0.0, float(base_ia + tenure_delta + campaign_shock))
            inc_party = matched_tracker['inc_party']
        else:
            tenure_val = max(4.0, float(next_tenure)) if (pd.notna(next_tenure) and next_tenure > 0) else 4.0 # REFINE - augmented to increase change of positive outcome
            x_dict = {"const": 1.0, "log_tenure": np.log(max(1.0, tenure_val)), demo_class: 1.0}
            
            # Build array matching the exact coefficient keys
            x_new = np.array([x_dict.get(k, 0.0) for k in coefs.keys()])
            eps = rng.normal(0, sigma_resid) if simulate else 0.0
            
            pred_ia = float(np.dot(x_new, np.array(list(coefs.values()))) + eps)
            target_ia = max(0.0, pred_ia)
            inc_party = next_party

        if target_ia <= 1e-9:
            #import pdb; pdb.set_trace()
            continue

        # 2. Identify Incumbent and Challengers at Count 0
        coal_aliases = ['LP', 'NP', 'LNP', 'COAL'] if inc_party in ['LP', 'NP', 'COAL'] else [inc_party]
        inc_cand_key = next((c for c in cf_c0_step6.keys() if any(str(c).upper().startswith(p) for p in coal_aliases)), None)
        
        if not inc_cand_key:
            continue

        active_challengers = [c for c in cf_c0_step6.keys() if c != inc_cand_key and cf_c0_step6[c] > 1e-9]
        if not active_challengers:
            continue

        # 3. Dynamic Square-Root Gravity Split Native to Count 0
        sum_sqrt = sum(np.sqrt(cf_c0_step6[c]) for c in active_challengers)
        # (Deleted `cf_c0_step6 = cf_c0_step5.copy()` here so dual incumbents accumulate on cf_c0_step6)
        
        cf_c0_step6[inc_cand_key] += target_ia
        for c in active_challengers:
            weight = np.sqrt(cf_c0_step6[c]) / sum_sqrt if sum_sqrt > 1e-9 else (1.0 / len(active_challengers))
            cf_c0_step6[c] = max(0.0, cf_c0_step6[c] - (target_ia * weight))

    #if div == 'Prahran':
    #    import pdb; pdb.set_trace()

    total_v = sum(cf_c0_step6.values())
    return {k: v / total_v for k, v in cf_c0_step6.items()}



# =============================================================================
# 1. STEP 7-8 SIMULATOR (MATCHED TO NEW STEP6_BY_YEAR STRUCTURE)
# =============================================================================

def simulate_step78_for_election(
    election_year: str,
    step6_bundle: dict,
    results_dict: dict,
    df_t: float,
    dirichlet_alpha_split: float,
    seed: int = 2026
) -> dict:
    """
    Consumes the precomputed STEP6_BY_YEAR[election_year] bundle:
      - Scales pre-drawn raw_elec_mvn by variance-preserving Student-t (df_t).
      - Adds precomputed alr_shifts_list[sim_idx] + sim_poll_errors + sim_elec_errors.
      - Runs apply_district_alr_shift (with DIRICHLET_ALPHA=dirichlet_alpha_split)
        and execute_reapply_incumbency for every seat and simulation.
    """
    base_rng = np.random.default_rng(seed + int(election_year))
    rng_t, rng_step8, rng_ia = base_rng.spawn(3)

    # Unpack precomputed Stage A objects from step6_bundle
    step6_sims = step6_bundle["step6_c0"]              # List of 1,000 dicts: [{div: {cand: share}}, ...]
    alr_shifts_list = step6_bundle["alr_shifts_list"]  # List of 1,000 dicts: [{'ALP': .., 'GRN': .., 'OTH': ..}, ...]
    sim_poll_errors = step6_bundle["sim_poll_errors"]  # Shape: (1000, 3)
    raw_elec_mvn = step6_bundle["raw_elec_mvn"]        # Shape: (1000, n_divs, 3)
    target_districts = step6_bundle["target_districts"]
    alr_cols = step6_bundle["alr_cols"]
    data_year = step6_bundle["data_year"]

    global_party_dict = step6_bundle["global_party_dict"]
    ia_tracker_dict = step6_bundle["ia_tracker_dict"]
    inc_next_lookup = step6_bundle["inc_next_lookup"]
    ia_params_dict = step6_bundle["ia_params_dict"]
    demo_lookup = step6_bundle["demo_lookup"]

    n_sims = len(step6_sims)
    n_divs = raw_elec_mvn.shape[1]

    # --- STEP 7: Variance-Preserving Student-t Scaling on Pre-Drawn Seat Errors ---
    if np.isfinite(df_t) and df_t > 2.0:
        chi2_draws = rng_t.chisquare(df_t, size=(n_sims, n_divs, 1))
        t_scale = np.sqrt((df_t - 2.0) / chi2_draws)
        sim_elec_errors = raw_elec_mvn * t_scale
    else:
        sim_elec_errors = raw_elec_mvn

    # Pre-allocate output matrices aligned to results_dict[div].columns
    target_cols_by_div = {div: list(results_dict[div].columns) for div in results_dict}
    shifted_samples_by_div = {
        div: np.zeros((n_sims, len(cols)), dtype=float)
        for div, cols in target_cols_by_div.items()
    }

    # --- STEP 8: Per-Simulation Loop (ALR Shift + Incumbency Reapplication) ---
    for sim_idx in range(n_sims):
        cf_c0_step5_dict = step6_sims[sim_idx]
        alr_shifts = alr_shifts_list[sim_idx]

        for d_idx, div in enumerate(target_districts):
            if div not in target_cols_by_div:
                continue

            # 1. Total ALR Shift = Precomputed Base + Statewide Poll Error + Seat Shock
            div_shifts = {
                b: alr_shifts[b] + sim_poll_errors[sim_idx, k] + sim_elec_errors[sim_idx, d_idx, k]
                for k, b in enumerate(alr_cols)
            }
            div_shifts['COAL'] = 0.0

            # 2. Apply District ALR Shift with candidate DIRICHLET_ALPHA
            cf_c0_step6 = apply_district_alr_shift(
                district_c0=cf_c0_step5_dict[div],
                alr_shifts=div_shifts,
                data_year=data_year,
                global_party_dict=global_party_dict,
                DIRICHLET_ALPHA=dirichlet_alpha_split,
                simulate=True,
                rng=rng_step8
            )

            # 3. Reapply Incumbency Advantage on Post-ALR Baseline
            cf_c0_step7 = execute_reapply_incumbency(
                div=div,
                cf_c0_step5=cf_c0_step6,
                ia_tracker_list=ia_tracker_dict.get(div, None),
                inc_next_info_list=inc_next_lookup.get(div, None),
                params_dict=ia_params_dict,
                demo_class=demo_lookup.get(div, 'Inner Metropolitan'),
                simulate=True,
                rng=rng_ia
            )

            cols = target_cols_by_div[div]
            shifted_samples_by_div[div][sim_idx, :] = [cf_c0_step7.get(c, 0.0) for c in cols]

    return shifted_samples_by_div

# =============================================================================
# 2. PROPER SCORING METRICS: SIMPLEX CRPS_TVD + POINT TVD + COVERAGE
# =============================================================================

# =============================================================================
# 2. PROPER SCORING METRICS: SIMPLEX CRPS_TVD + POINT TVD + COVERAGE
# =============================================================================

def get_election_metrics_tvd(combined_samples: dict, Results_dict: dict, coverage_levels: list) -> tuple:
    seat_crps_tvds = []
    seat_point_tvds = []
    coverage_hits = {c: 0 for c in coverage_levels}
    total_predictions = 0

    lo_pcts = [(1.0 - c) / 2.0 * 100.0 for c in coverage_levels]
    hi_pcts = [(1.0 + c) / 2.0 * 100.0 for c in coverage_levels]

    for div, pred_raw in combined_samples.items():
        pred = pred_raw * 100.0 if np.max(pred_raw) <= 1.05 else pred_raw
        actual = Results_dict[div].iloc[0].to_numpy(dtype=float)
        if actual.sum() <= 1.05:
            actual = actual * 100.0

        n_sims, n_cands = pred.shape

        # Term 1 of CRPS: E[|X - y|]
        mae_sim_per_cand = np.mean(np.abs(pred - actual[None, :]), axis=0)

        # Term 2 of CRPS: 0.5 * E[|X - X'|] via sorted order statistics
        sorted_pred = np.sort(pred, axis=0)
        weights = (2.0 * np.arange(1, n_sims + 1) - n_sims - 1.0)[:, None]
        half_spread_per_cand = np.sum(weights * sorted_pred, axis=0) / (n_sims ** 2)

        crps_per_cand = mae_sim_per_cand - half_spread_per_cand
        seat_crps_tvds.append(float(0.5 * np.sum(crps_per_cand)))

        mean_pred = np.mean(pred, axis=0)
        seat_point_tvds.append(float(0.5 * np.sum(np.abs(mean_pred - actual))))

        total_predictions += n_cands

        try:
            lo_bounds = np.percentile(pred, lo_pcts, axis=0, method="weibull")
            hi_bounds = np.percentile(pred, hi_pcts, axis=0, method="weibull")

            
        except TypeError:
            lo_bounds = np.percentile(pred, lo_pcts, axis=0)
            hi_bounds = np.percentile(pred, hi_pcts, axis=0)
        except ValueError as e:
            # --- SURGICAL DEBUGGING TRAP ---
            print("\n" + "="*60)
            print("🚨 CAUGHT THE CRASH IN MULTIPROCESSING WORKER 🚨")
            print(f"Electorate (div): {div}")
            print(f"coverage_levels passed in: {coverage_levels}")
            print(f"Calculated lo_pcts: {lo_pcts}")
            print(f"Calculated hi_pcts: {hi_pcts}")
            print(f"pred array shape: {pred.shape}")
            print(f"Contains NaNs in pred? {np.isnan(pred).any()}")
            if not np.isnan(pred).all():
                print(f"Max pred value: {np.nanmax(pred)}")
            print("="*60 + "\n")
            raise e  # Halt the program so you can read the output

        for idx, c in enumerate(coverage_levels):
            within_bounds = (actual >= lo_bounds[idx]) & (actual <= hi_bounds[idx])
            coverage_hits[c] += int(np.sum(within_bounds))

    avg_crps_tvd = float(np.mean(seat_crps_tvds))
    avg_point_tvd = float(np.mean(seat_point_tvds))
    empirical_coverages = {c: coverage_hits[c] / total_predictions for c in coverage_levels}

    return avg_crps_tvd, avg_point_tvd, empirical_coverages, coverage_hits, total_predictions


def compute_part1_objective(
    crps_tvd: float,
    empirical_coverages: dict,
    target_coverage_schedule: list,
    coverage_weight: float = 0.50
) -> float:
    cov_errors_pct = [abs(empirical_coverages[target] - target) * 100.0 for target in target_coverage_schedule]
    return crps_tvd + coverage_weight * float(np.mean(cov_errors_pct))


# =============================================================================
# 3. MULTIPROCESSING WORKER & LOO-CV OPTIMIZER
# =============================================================================

GLOBAL_STEP6_BY_YEAR = {}
GLOBAL_RESULTS_FP_BY_YEAR = {}
GLOBAL_COV_SCHEDULE = [0.80, 0.85, 0.90, 0.95]

def _init_part1_worker(step6_by_year, results_fp_by_year, cov_schedule):
    global GLOBAL_STEP6_BY_YEAR, GLOBAL_RESULTS_FP_BY_YEAR, GLOBAL_COV_SCHEDULE
    GLOBAL_STEP6_BY_YEAR = step6_by_year
    GLOBAL_RESULTS_FP_BY_YEAR = results_fp_by_year
    GLOBAL_COV_SCHEDULE = cov_schedule


def evaluate_part1_grid_point(args):
    df_t, dirichlet_alpha_split, election_year = args

    sim_samples = simulate_step78_for_election(
        election_year=election_year,
        step6_bundle=GLOBAL_STEP6_BY_YEAR[election_year],
        results_dict=GLOBAL_RESULTS_FP_BY_YEAR[election_year],
        df_t=df_t,
        dirichlet_alpha_split=dirichlet_alpha_split
    )

    crps_tvd, point_tvd, emp_cov, cov_hits, n_preds = get_election_metrics_tvd(
        sim_samples,
        GLOBAL_RESULTS_FP_BY_YEAR[election_year],
        GLOBAL_COV_SCHEDULE
    )

    return {
        "df_t": df_t,
        "DIRICHLET_ALPHA": dirichlet_alpha_split,
        "year": election_year,
        "crps_tvd": crps_tvd,
        "point_tvd": point_tvd,
        "emp_cov": emp_cov,
        "cov_hits": cov_hits,
        "n_preds": n_preds
    }


def validate_step78_polling_params(
    STEP6_BY_YEAR: dict,
    RESULTS_FP_BY_YEAR: dict,
    target_coverage_schedule: list = [0.80, 0.85, 0.90, 0.95],
    coverage_weight: float = 0.50,
    n_processes: int = 24,
    df_t_grid  = [4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0, np.inf],
    dirichlet_alpha_grid = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 65.0, 80.0, 100.0]
):
    elections = ['2010', '2014', '2018', '2022']


    all_tasks = [
        (df_t, d_alpha, yr)
        for df_t in df_t_grid
        for d_alpha in dirichlet_alpha_grid
        for yr in elections
    ]

    print(f"Starting Part 1 precomputation for {len(all_tasks)} tasks across {n_processes} cores...")
    t0 = time.time()

    with multiprocessing.Pool(
        processes=n_processes,
        initializer=_init_part1_worker,
        initargs=(STEP6_BY_YEAR, RESULTS_FP_BY_YEAR, target_coverage_schedule)
    ) as pool:
        raw_results = list(tqdm(
            pool.imap_unordered(evaluate_part1_grid_point, all_tasks, chunksize=2),
            total=len(all_tasks)
        ))

    print(f"Precomputation complete in {time.time() - t0:.2f} seconds.")

    metrics_lookup = defaultdict(dict)
    for res in raw_results:
        metrics_lookup[(res["df_t"], res["DIRICHLET_ALPHA"])][res["year"]] = res

    best_params_by_heldout = {}
    for heldout in elections:
        train_elections = [e for e in elections if e != heldout]
        candidates_scores = []

        for (df_t, d_alpha), yr_dict in metrics_lookup.items():
            train_crps = float(np.mean([yr_dict[e]["crps_tvd"] for e in train_elections]))
            tot_preds = sum(yr_dict[e]["n_preds"] for e in train_elections)
            pooled_emp_cov = {
                c: sum(yr_dict[e]["cov_hits"][c] for e in train_elections) / tot_preds
                for c in target_coverage_schedule
            }
            val_score = compute_part1_objective(
                train_crps, pooled_emp_cov, target_coverage_schedule, coverage_weight=coverage_weight
            )
            candidates_scores.append((df_t, d_alpha, val_score))

        best_df_t, best_d_alpha, best_val_score = min(candidates_scores, key=lambda x: x[2])
        heldout_res = metrics_lookup[(best_df_t, best_d_alpha)][heldout]
        heldout_score = compute_part1_objective(
            heldout_res["crps_tvd"], heldout_res["emp_cov"], target_coverage_schedule, coverage_weight=coverage_weight
        )

        best_params_by_heldout[heldout] = {
            "best_df_t": best_df_t,
            "best_DIRICHLET_ALPHA": best_d_alpha,
            "val_score": best_val_score,
            "test_score": heldout_score,
            "test_crps_tvd": heldout_res["crps_tvd"],
            "test_point_tvd": heldout_res["point_tvd"],
            "test_coverage": {k: round(v, 4) for k, v in heldout_res["emp_cov"].items()}
        }

        print(
            f"[Held-out {heldout}] Best df_t={best_df_t}, DIRICHLET_ALPHA={best_d_alpha} | "
            f"Test CRPS_TVD={heldout_res['crps_tvd']:.3f}% (Point TVD={heldout_res['point_tvd']:.3f}%) | "
            f"Test Cov={best_params_by_heldout[heldout]['test_coverage']}"
        )

    full_ranking = []
    for (df_t, d_alpha), yr_dict in metrics_lookup.items():
        full_crps = float(np.mean([yr_dict[e]["crps_tvd"] for e in elections]))
        full_point_tvd = float(np.mean([yr_dict[e]["point_tvd"] for e in elections]))
        tot_preds = sum(yr_dict[e]["n_preds"] for e in elections)
        full_emp_cov = {
            c: sum(yr_dict[e]["cov_hits"][c] for e in elections) / tot_preds
            for c in target_coverage_schedule
        }
        full_obj = compute_part1_objective(
            full_crps, full_emp_cov, target_coverage_schedule, coverage_weight=coverage_weight
        )
        full_ranking.append({
            "df_t": df_t,
            "DIRICHLET_ALPHA": d_alpha,
            "obj_score": full_obj,
            "crps_tvd": full_crps,
            "point_tvd": full_point_tvd,
            "cov_80": full_emp_cov[0.80],
            "cov_85": full_emp_cov[0.85],
            "cov_90": full_emp_cov[0.90],
            "cov_95": full_emp_cov[0.95]
        })

    ranking_df = pd.DataFrame(full_ranking).sort_values("obj_score", ascending=True).reset_index(drop=True)
    print("\n=== TOP 10 PARAMETER COMBINATIONS (POOLED 2010-2022) ===")
    print(ranking_df.head(10).to_string(index=False))

    return best_params_by_heldout, ranking_df



def diagnose_structural_coverage_leak(STEP6_BY_YEAR, RESULTS_FP_BY_YEAR, test_df_t, test_alpha):
    print("\n" + "="*50)
    print("🚨 RUNNING STRUCTURAL COVERAGE DIAGNOSTIC 🚨")
    print("="*50)
    
    hits = {'ALP': 0, 'COAL': 0, 'GRN': 0, 'OTH': 0}
    counts = {'ALP': 0, 'COAL': 0, 'GRN': 0, 'OTH': 0}
    alignment_failures = 0
    
    # We will test the Gaussian baseline with Dirichlet Alpha = 25
    test_df_t = np.inf
    test_alpha = 25.0
    
    for yr in ['2010', '2014', '2018', '2022']:
        samples = simulate_step78_for_election(
            yr, STEP6_BY_YEAR[yr], RESULTS_FP_BY_YEAR[yr], test_df_t, test_alpha
        )
        data_year = int(yr) - 4
        
        for div, pred_raw in samples.items():
            actual = RESULTS_FP_BY_YEAR[yr][div].iloc[0].to_numpy(dtype=float)
            if actual.sum() <= 1.05:
                actual *= 100.0
                
            # Check for candidate dropout (Alignment Failure)
            # If the extracted columns sum to less than 95%, we lost a major party to a key mismatch!
            avg_pred_sum = np.mean(np.sum(pred_raw, axis=1))
            if avg_pred_sum < 0.95:
                alignment_failures += 1
                missing_cands = [c for c in RESULTS_FP_BY_YEAR[yr][div].columns if c not in STEP6_BY_YEAR[yr]['step6_c0'][0][div]]
                if missing_cands:
                    print(f"[{yr} {div}] WARNING: Actual results require {missing_cands} but simulation did not output them!")
                
            # Scale to 0-100 percentages
            pred = pred_raw * 100.0 if np.nanmax(pred_raw) <= 1.05 else pred_raw
            lo, hi = np.percentile(pred, [2.5, 97.5], axis=0, method="weibull") # 95% Interval
            
            for i, cand in enumerate(RESULTS_FP_BY_YEAR[yr][div].columns):
                b = get_poll_bucket_hist(cand, data_year)
                counts[b] += 1
                if lo[i] <= actual[i] <= hi[i]:
                    hits[b] += 1
                    
    print("\n--- 95% COVERAGE BY MACRO BUCKET ---")
    for b in ['ALP', 'COAL', 'GRN', 'OTH']:
        cov_pct = (hits[b] / max(1, counts[b])) * 100.0
        print(f"{b:>4}: {cov_pct:.1f}%  ({hits[b]} / {counts[b]} candidates)")
        
    print(f"\nAlignment Failures: {alignment_failures} seats lost >5% of the vote to mismatched keys.")
    print("="*50 + "\n")

def diagnose_bucket_coverage(
    STEP6_BY_YEAR: dict, 
    RESULTS_FP_BY_YEAR: dict, 
    df_t: float, 
    dirichlet_alpha: float, 
    target_coverages: list = [0.80, 0.90, 0.95]
):
    """
    Runs the simulation at a fixed parameter point and prints empirical coverage 
    broken down by specific party (ALP, COAL, GRN, IND, and all specific minors) 
    and by year.
    """
    print(f"\n" + "="*85)
    print(f"📊 COVERAGE BREAKDOWN: df_t={df_t}, ALPHA={dirichlet_alpha}")
    print("="*85)

    years = ['2010', '2014', '2018', '2022']
    
    # Run the simulation once for each year to get all predictions
    samples_by_year = {}
    for yr in years:
        samples_by_year[yr] = simulate_step78_for_election(
            yr, STEP6_BY_YEAR[yr], RESULTS_FP_BY_YEAR[yr], df_t, dirichlet_alpha
        )

    for target_cov in target_coverages:
        lo_q = (1.0 - target_cov) / 2.0 * 100.0
        hi_q = (1.0 + target_cov) / 2.0 * 100.0

        # Dynamic data structures to track hits and counts
        hits = {yr: defaultdict(int) for yr in years}
        counts = {yr: defaultdict(int) for yr in years}

        for yr in years:
            samples = samples_by_year[yr]
            for div, pred_raw in samples.items():
                actual = RESULTS_FP_BY_YEAR[yr][div].iloc[0].to_numpy(dtype=float)
                if actual.sum() <= 1.05:
                    actual *= 100.0

                pred = pred_raw * 100.0 if np.nanmax(pred_raw) <= 1.05 else pred_raw
                lo, hi = np.percentile(pred, [lo_q, hi_q], axis=0, method="weibull")

                for i, cand in enumerate(RESULTS_FP_BY_YEAR[yr][div].columns):
                    c_str = str(cand).strip().upper()
                    
                    # Map to major buckets or extract the specific minor party abbreviation
                    if c_str.startswith('ALP'): b = 'ALP'
                    elif any(c_str.startswith(p) for p in ['LP', 'NP', 'LNP', 'COAL', 'CLP']): b = 'COAL'
                    elif c_str.startswith('GRN'): b = 'GRN'
                    elif c_str.startswith('IND'): b = 'IND'
                    else: 
                        b = c_str.split('_')[0]  # Extracts 'AJP', 'SPP', 'ON', etc.

                    counts[yr][b] += 1
                    if lo[i] <= actual[i] <= hi[i]:
                        hits[yr][b] += 1

        print(f"\n--- {int(target_cov*100)}% TARGET COVERAGE ---")
        print(f"{'Party':<9} | {'Overall':<7} | {'2010':<7} | {'2014':<7} | {'2018':<7} | {'2022':<7} | {'N (Total)'}")
        print("-" * 75)
        
        # Sort buckets: Main 4 first, then alphabetical minors
        main_buckets = ['ALP', 'COAL', 'GRN', 'IND']
        all_buckets = set()
        for yr in years:
            all_buckets.update(counts[yr].keys())
        
        minor_buckets = sorted([b for b in all_buckets if b not in main_buckets])
        display_buckets = [b for b in main_buckets if b in all_buckets] + minor_buckets
        
        for b in display_buckets:
            b_hits = sum(hits[yr].get(b, 0) for yr in years)
            b_counts = sum(counts[yr].get(b, 0) for yr in years)
            if b_counts == 0: continue
            
            overall_pct = (b_hits / b_counts) * 100
            
            # Format row
            row_str = f"{b[:8]:<9} | {overall_pct:5.1f}% | "
            for yr in years:
                y_hits = hits[yr].get(b, 0)
                y_counts = counts[yr].get(b, 0)
                if y_counts > 0:
                    y_pct = (y_hits / y_counts) * 100
                    row_str += f"{y_pct:5.1f}% | "
                else:
                    row_str += f"{'-':>6} | "
            row_str += f"{b_counts}"
            print(row_str)
            
        print("-" * 75)
        # Overall summary row
        tot_hits = sum(sum(hits[yr].values()) for yr in years)
        tot_counts = sum(sum(counts[yr].values()) for yr in years)
        row_str = f"{'ALL':<9} | {tot_hits/tot_counts*100:5.1f}% | "
        for yr in years:
            y_tot_hits = sum(hits[yr].values())
            y_tot_counts = sum(counts[yr].values())
            row_str += f"{y_tot_hits/y_tot_counts*100:5.1f}% | "
        row_str += f"{tot_counts}"
        print(row_str)


def deep_dive_minor_failures(STEP6_BY_YEAR, RESULTS_FP_BY_YEAR, df_t, dirichlet_alpha):
    targets = [
        ('2010', 'CYA'), ('2018', 'CYA'), ('2014', 'AJP'), 
        ('2010', 'DLP'), ('2022', 'FPV'), ('2014', 'RUA'), 
        ('2018', 'SPP'), ('2022', 'VNS'), ('2018', 'DHJP')
    ]
    
    print("\n" + "="*90)
    print(f"🔍 DEEP DIVE: MINOR PARTY FAILURES (df_t={df_t}, ALPHA={dirichlet_alpha})")
    print("="*90)
    
    for yr, target_party in targets:
        samples = simulate_step78_for_election(
            yr, STEP6_BY_YEAR[yr], RESULTS_FP_BY_YEAR[yr], df_t, dirichlet_alpha
        )
        
        records = []
        
        for div, pred_raw in samples.items():
            cols = list(RESULTS_FP_BY_YEAR[yr][div].columns)
            actual = RESULTS_FP_BY_YEAR[yr][div].iloc[0].to_numpy(dtype=float)
            if actual.sum() <= 1.05:
                actual *= 100.0
                
            pred = pred_raw * 100.0 if np.nanmax(pred_raw) <= 1.05 else pred_raw
            
            for i, cand in enumerate(cols):
                c_str = str(cand).strip().upper()
                # Isolate the specific minor party
                if not c_str.startswith('IND') and c_str.split('_')[0] == target_party:
                    sims = pred[:, i]
                    pcts = np.percentile(sims, [5, 10, 25, 50, 75, 90, 95])
                    sim_mean = np.mean(sims)
                    act_val = actual[i]
                    
                    records.append({
                        'Division': div,
                        'Cand': cand,
                        'Mean': sim_mean,
                        'p05': pcts[0],
                        'p10': pcts[1],
                        'p25': pcts[2],
                        'p50': pcts[3],
                        'p75': pcts[4],
                        'p90': pcts[5],
                        'p95': pcts[6],
                        'ACTUAL': act_val,
                        'Error': act_val - sim_mean
                    })
        
        if records:
            df = pd.DataFrame(records)
            # Sort by the worst misses (Absolute Error)
            df = df.sort_values(by='Error', key=abs, ascending=False).reset_index(drop=True)
            
            # Formatting for clean terminal output
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.float_format', '{:.2f}'.format)
            
            print(f"\n--- {target_party} in {yr} (N={len(df)} seats) ---")
            print(df.to_string(index=False)) # Top 15 worst misses
            print(f">>> AVG ACTUAL: {df['ACTUAL'].mean():.2f}% | AVG SIM MEAN: {df['Mean'].mean():.2f}%")
        else:
            print(f"\n--- {target_party} in {yr} ---")
            print("No candidates found in alignment.")
        import pdb; pdb.set_trace()

#diagnose_structural_coverage_leak(STEP6_BY_YEAR, RESULTS_FP_BY_YEAR, test_df_t=np.inf, test_alpha=15)
#diagnose_bucket_coverage(STEP6_BY_YEAR, RESULTS_FP_BY_YEAR, df_t=np.inf, dirichlet_alpha=15, target_coverages=[0.95,0.9,0.8,0.7,0.6,0.5,0.2])
#deep_dive_minor_failures(STEP6_BY_YEAR, RESULTS_FP_BY_YEAR,  df_t=np.inf, dirichlet_alpha=15)
#import pdb; pdb.set_trace()


def compute_historical_mae( # NEW
    STEP6_BY_YEAR: dict, # NEW
    RESULTS_FP_BY_YEAR: dict, # NEW
    df_t: float = np.inf, # NEW
    dirichlet_alpha: float = 14.0, # NEW
    years: list = ['2010', '2014', '2018', '2022'], # NEW
    use_median: bool = False # NEW
) -> Tuple[pd.DataFrame, pd.DataFrame]: # NEW
    """Computes historical FP MAE (in percentage points) across elections and party buckets.""" # NEW
    records = [] # NEW
    seat_tvds_by_year = defaultdict(list) # NEW

    for yr in years: # NEW
        samples = simulate_step78_for_election( # NEW
            election_year=yr, # NEW
            step6_bundle=STEP6_BY_YEAR[yr], # NEW
            results_dict=RESULTS_FP_BY_YEAR[yr], # NEW
            df_t=df_t, # NEW
            dirichlet_alpha_split=dirichlet_alpha # NEW
        ) # NEW

        for div, pred_raw in samples.items(): # NEW
            cols = list(RESULTS_FP_BY_YEAR[yr][div].columns) # NEW
            actual = RESULTS_FP_BY_YEAR[yr][div].iloc[0].to_numpy(dtype=float) # NEW
            if actual.sum() <= 1.05: # NEW
                actual = actual * 100.0 # NEW

            pred = pred_raw * 100.0 if np.nanmax(pred_raw) <= 1.05 else pred_raw # NEW
            point_pred = np.median(pred, axis=0) if use_median else np.mean(pred, axis=0) # NEW
            abs_err = np.abs(point_pred - actual) # NEW

            seat_tvds_by_year[yr].append(float(0.5 * np.sum(abs_err))) # NEW

            for cand, p_hat, y_true, err in zip(cols, point_pred, actual, abs_err): # NEW
                c_str = str(cand).strip().upper() # NEW
                if c_str.startswith('ALP'): # NEW
                    bucket = 'ALP' # NEW
                elif any(c_str.startswith(p) for p in ['LP', 'NP', 'LNP', 'COAL', 'CLP']): # NEW
                    bucket = 'COAL' # NEW
                elif c_str.startswith('GRN'): # NEW
                    bucket = 'GRN' # NEW
                elif c_str.startswith('IND'): # NEW
                    bucket = 'IND' # NEW
                else: # NEW
                    bucket = 'Minor' # NEW

                records.append({ # NEW
                    'year': yr, # NEW
                    'div': div, # NEW
                    'cand': cand, # NEW
                    'bucket': bucket, # NEW
                    'pred': float(p_hat), # NEW
                    'actual': float(y_true), # NEW
                    'abs_err': float(err) # NEW
                }) # NEW

    err_df = pd.DataFrame(records) # NEW

    # 1. Per-year and pooled summary # NEW
    year_rows = [] # NEW
    for yr in years: # NEW
        sub = err_df[err_df['year'] == yr] # NEW
        maj = sub[sub['bucket'].isin(['ALP', 'COAL', 'GRN'])] # NEW
        seat_mae = sub.groupby('div')['abs_err'].mean().mean() # NEW
        year_rows.append({ # NEW
            'Year': yr, # NEW
            'All_Cand_MAE': sub['abs_err'].mean(), # NEW
            'Seat_Avg_MAE': seat_mae, # NEW
            'Major_MAE': maj['abs_err'].mean(), # NEW
            'Point_TVD': float(np.mean(seat_tvds_by_year[yr])), # NEW
            'N_Cands': len(sub) # NEW
        }) # NEW

    all_maj = err_df[err_df['bucket'].isin(['ALP', 'COAL', 'GRN'])] # NEW
    all_seat_mae = err_df.groupby(['year', 'div'])['abs_err'].mean().mean() # NEW
    all_tvd = float(np.mean([v for yr in years for v in seat_tvds_by_year[yr]])) # NEW
    year_rows.append({ # NEW
        'Year': 'POOLED', # NEW
        'All_Cand_MAE': err_df['abs_err'].mean(), # NEW
        'Seat_Avg_MAE': all_seat_mae, # NEW
        'Major_MAE': all_maj['abs_err'].mean(), # NEW
        'Point_TVD': all_tvd, # NEW
        'N_Cands': len(err_df) # NEW
    }) # NEW
    summary_df = pd.DataFrame(year_rows) # NEW

    # 2. Breakdown by party bucket x year # NEW
    bucket_order = ['ALP', 'COAL', 'GRN', 'IND', 'Minor'] # NEW
    pivot = err_df.pivot_table(index='bucket', columns='year', values='abs_err', aggfunc='mean') # NEW
    pivot['POOLED'] = err_df.groupby('bucket')['abs_err'].mean() # NEW
    pivot['N'] = err_df.groupby('bucket')['abs_err'].count() # NEW
    pivot = pivot.reindex([b for b in bucket_order if b in pivot.index]) # NEW

    print(f"\n=== HISTORICAL FP MAE (df_t={df_t}, DIRICHLET_ALPHA={dirichlet_alpha}) ===") # NEW
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}")) # NEW
    print("\n=== MAE BY PARTY BUCKET (% pts) ===") # NEW
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}")) # NEW

    return err_df, summary_df # NEW


err_df, mae_summary = compute_historical_mae( # NEW
    STEP6_BY_YEAR=STEP6_BY_YEAR, # NEW
    RESULTS_FP_BY_YEAR=RESULTS_FP_BY_YEAR, # NEW
    df_t=np.inf, # NEW
    dirichlet_alpha=14.0 # NEW
) # NEW

import pdb; pdb.set_trace()


df_t_grid  = [4.0, 8.0, 15.0, 30.0, np.inf]
dirichlet_alpha_grid = [4,6, 8,10.0, 12,14,16,18,20,22] # [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 65.0, 80.0, 100.0]


best_loo, ranking_df = validate_step78_polling_params(
    STEP6_BY_YEAR=STEP6_BY_YEAR,
    RESULTS_FP_BY_YEAR=RESULTS_FP_BY_YEAR,
    target_coverage_schedule = [0.80, 0.85, 0.90, 0.95],
    coverage_weight = 10,
    n_processes = 24,
    df_t_grid  = df_t_grid,
    dirichlet_alpha_grid =dirichlet_alpha_grid
)


import pdb; pdb.set_trace()