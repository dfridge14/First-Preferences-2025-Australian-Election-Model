import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import re
import time
import copy
import json

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
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

data_year = 2022
election_year = data_year + 4

LP_NP_DIRICHLET_ALPHA = 4.1783


#base_dir = os.path.dirname(os.path.abspath(__file__))
#os.chdir(base_dir)
os.chdir(Path.home() / 'Australian Election/Victorian Election')

is_redistribution = (data_year in [2010,2018])
if is_redistribution:
    with open(f'rediv_seg_votes_dict_for_{election_year}.pkl', 'rb') as f:
        rediv_seg_votes_dict = pickle.load(f)
    with open(f'rediv_transfer_props_dict_for_{election_year}.pkl', 'rb') as f:
        rediv_transfer_props_dict = pickle.load(f)

LC_results = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LC-Votes-Electorate-Formatted.csv')
LA_results = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Primary-Electorate-Formatted.csv')

LA_Candidates = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Candidates-Formatted.csv')

# electorate to region mapping
electorate_to_region = LA_Candidates[['div_nm','region_name']].drop_duplicates().set_index('div_nm')['region_name'].to_dict() if data_year != 2006 else LC_results[['div_nm','region_name']].drop_duplicates().set_index('div_nm')['region_name'].to_dict()

# create df matching party code to party name
party_df = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-Parties.csv')
Ideology_map = party_df.set_index('PartyAb')['Ideo_category']
BLOCS = ['ALP','COAL','Left','Right','Centre']
ORPHAN_POOLS = ['Left','Right','Centre']

DEEPEST_CAND_NO = 4

with open(f'VIC-{data_year}/{data_year}_complete_DOP_table_dict.pkl', 'rb') as f:
    DOP_table_dict = pickle.load(f)
    if data_year == 2022:
        del DOP_table_dict['Narracan'] # FIX: Narracan dealt with separately!

#with open(f'VIC-{data_year}/ia_adjustments_dict.pkl', 'rb') as f:
#    ia_adjustments_dict = pickle.load(f)

LC_FPs_df = pd.read_csv(f'VIC-{data_year}/{data_year}-LC-First-Prefs-df.csv', index_col = 'PartyAb')


# FIX: temporarily using only 2022 ia model - should obtain a general one
with open(f"VIC-{data_year}/VIC-{data_year}-IA-model-params.json", 'r') as f:
    ia_params_dict = json.load(f)

ia_csv_path = f"VIC-{data_year}/VIC-{data_year}-IA-adjustments.csv"
# Load the Demographic mappings for the next election (all info for data_year already baked into ia_params)
demo_df = pd.read_csv(f'VIC-{election_year}/Demographic-classification-{election_year}.csv')
demo_lookup = demo_df.set_index('district_name')['classification'].to_dict()

new_inds_df = pd.read_csv("new_inds_features.csv")
pool_df = pd.read_csv("pool_df_features.csv")

# load the new IND models, priors, and party dictionary
with open("add_new_IND_models_and_priors.pkl", "rb") as f:
    add_new_IND_data = pickle.load(f)

pool_model = add_new_IND_data['pool_model']
split_model = add_new_IND_data['split_model']
dop_lookups_by_year = add_new_IND_data['dop_lookups_by_year']
mu_ind_to_ind_dict = add_new_IND_data['mu_ind_to_ind_dict']
global_party_dict = add_new_IND_data['global_party_dict']

mu_ind_to_ind_array = mu_ind_to_ind_dict[data_year]
dop_lookup = dop_lookups_by_year[data_year]
class AggregateLatentLogit:
    """Aggregate Latent Class Logit Model with Cumulative State Evaluation.
    Parameter vector theta has length 16.
    """
    def __init__(self, theta: np.ndarray = None):
        self.n_pools = len(ORPHAN_POOLS)  # 3
        self.n_blocs = len(BLOCS)  # 5
        self.n_params = 16

        if theta is not None:
            self.set_params(theta)
        else:
            self.theta = self.get_default_priors()

    def get_default_priors(self) -> np.ndarray:
        # [alpha_COAL, alpha_Left, alpha_Right, alpha_Centre]
        prior_left = np.array([-1.5, 2.0, -2.0, -0.5])
        prior_right = np.array([1.5, -1.5, 2.0, 1.0])
        prior_centre = np.array([0.0, 0.5, 0.5, 1.5])
        # betas: [beta_mass, beta_cap, beta_menu]
        prior_betas = np.array([1.0, 0.5, 0.1]) 
        # gamma
        prior_gamma = np.array([0.8])

        return np.concatenate([prior_left, prior_right, prior_centre, prior_betas, prior_gamma])

    def unpack_params(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        alpha = np.zeros((self.n_pools, self.n_blocs), dtype=np.float64)
        alpha[0, 1:] = theta[0:4]   # Left pool
        alpha[1, 1:] = theta[4:8]   # Right pool
        alpha[2, 1:] = theta[8:12]  # Centre pool
        
        beta = theta[12:15]  
        gamma = float(theta[15])
        
        return alpha, beta, gamma

    def set_params(self, theta: np.ndarray):
        if len(theta) != self.n_params:
            raise ValueError(f"Expected theta of length {self.n_params}, got {len(theta)}")
        self.theta = np.asarray(theta, dtype=np.float64)

    def compute_choice_probabilities(
        self, lc_mass: np.ndarray, c_cap: np.ndarray, menu_size: np.ndarray, 
        alpha: np.ndarray, beta_shared: np.ndarray
    ) -> np.ndarray:
        b_mass, b_cap, b_menu = beta_shared
        
        log_mass = np.log(np.maximum(lc_mass, 1e-6))
        log_cap  = np.log(np.maximum(c_cap, 1e-6))
        log_menu = np.log(np.maximum(menu_size, 1e-6))
        
        u_structural = (b_mass * log_mass) + (b_cap * log_cap) + (b_menu * log_menu)
        v_matrix = alpha + u_structural[np.newaxis, :]

        unavailable_mask = menu_size <= 0
        v_matrix[:, unavailable_mask] = -1e9

        v_max = np.max(v_matrix, axis=1, keepdims=True)
        exp_v = np.exp(v_matrix - v_max) * (~unavailable_mask)[np.newaxis, :]
        sum_exp_v = np.sum(exp_v, axis=1, keepdims=True)
        sum_exp_v = np.where(sum_exp_v <= 1e-12, 1.0, sum_exp_v)

        return exp_v / sum_exp_v

    def evaluate_state_allocation(
        self, lc_mass: np.ndarray, active_cands: List[str], c_cap: np.ndarray, 
        menu_size: np.ndarray, cand_to_bloc: Dict[str, int], cand_demand: Dict[str, float], 
        alpha: np.ndarray, beta: np.ndarray, gamma: float
    ) -> Dict[str, float]:
        
        p_bloc_given_pool = self.compute_choice_probabilities(lc_mass, c_cap, menu_size, alpha, beta)

        cand_intra_share = {}
        for b_idx in range(self.n_blocs):
            bloc_cands = [c for c in active_cands if cand_to_bloc[c] == b_idx]
            if not bloc_cands: continue

            eff_gamma = 1.0 if b_idx in [0, 1] else gamma 
            sum_demand = sum((cand_demand[c] ** eff_gamma) for c in bloc_cands)
            
            if sum_demand > 1e-9:
                for c in bloc_cands:
                    cand_intra_share[c] = (cand_demand[c] ** eff_gamma) / sum_demand
            else:
                for c in bloc_cands:
                    cand_intra_share[c] = 1.0 / len(bloc_cands)

        s_hat = {c: 0.0 for c in active_cands}

        # Base Demand Allocation
        for b_idx in [0, 1]:  
            for c in [c for c in active_cands if cand_to_bloc[c] == b_idx]:
                s_hat[c] += lc_mass[b_idx] * cand_intra_share[c]

        for b_idx in [2, 3, 4]:  
            for c in [c for c in active_cands if cand_to_bloc[c] == b_idx]:
                s_hat[c] += cand_demand[c]

        # Orphan Routing
        orphan_mass = np.zeros(self.n_pools, dtype=np.float64)
        for p_idx, b_idx in enumerate([2, 3, 4]):
            orphan_mass[p_idx] = lc_mass[b_idx] * max(0.0, 1.0 - c_cap[b_idx])

        # Route eliminated ALP via Left pool (0) & COAL via Right pool (1)
        if menu_size[0] == 0 or menu_size[1] == 0:
            if menu_size[0] == 0:
                #print('No ALP', active_cands)
                #import pdb; pdb.set_trace()
                orphan_mass[0] += lc_mass[0]  # Eliminated ALP routes as Left orphan mass
            if menu_size[1] == 0:
                #print('No COAL', active_cands)
                #import pdb; pdb.set_trace()
                orphan_mass[1] += lc_mass[1]  # Eliminated LP/NP routes as Right orphan mass

            # Zero out empty blocs (e.g. missing ALP/COAL) and renormalize row probabilities
            active_bloc_mask = (menu_size > 0).astype(np.float64)
            p_bloc_given_pool = p_bloc_given_pool * active_bloc_mask[None, :]
            row_sums = p_bloc_given_pool.sum(axis=1, keepdims=True)
            p_bloc_given_pool = np.where(
                row_sums > 1e-12, p_bloc_given_pool / np.maximum(row_sums, 1e-12), 0.0
            )

        for p_idx in range(self.n_pools):
            o_m = orphan_mass[p_idx]
            if o_m <= 1e-12: continue
            for b_idx in range(self.n_blocs):
                prob_bloc = p_bloc_given_pool[p_idx, b_idx]
                if prob_bloc <= 1e-12: continue
                for c in [c for c in active_cands if cand_to_bloc[c] == b_idx]:
                    s_hat[c] += o_m * prob_bloc * cand_intra_share[c]

        return s_hat

# Load minor flow model and bootstrapped realisations
final_model = AggregateLatentLogit()

MINOR_MODEL_FILE = f'VIC-{data_year}/{data_year}_LC_flow_model_theta.npy'

if os.path.exists(MINOR_MODEL_FILE):
    print(f"Loading exact fully-estimated baseline parameters from {MINOR_MODEL_FILE}...")
    baseline_theta = np.load(MINOR_MODEL_FILE) if data_year != 2006 else np.array([-0.94748286, 0.05, -1, 0.32609389, 3, 2, 3, 1.20926125, -0.1977211 , -1.55463882,0.01473377,  0.55844421,  0.42967291,  0.75830687,  1.91614176, 0.2]) # HIJACKED SOME PARAMETERS FROM 2010-2022
    final_model.set_params(baseline_theta)
else:
    raise FileNotFoundError(f"Missing {MINOR_MODEL_FILE}! Run training script to generate the definitive point estimate.")



#import pdb; pdb.set_trace()
file_path = Path.home() / f"Australian Election/Victorian Election/VIC-{election_year}/TCP_state_dict_for_{election_year}.pkl"
with open(file_path, "rb") as f:
    TCP_state_dict = pickle.load(f)


def get_party_bloc(party: str, party_df: pd.DataFrame) -> str:
    """Helper to extract ideological bloc for any party in party_df."""
    if str(party).startswith('IND'):
        return 'Centre'
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    return ideo_map.get(party, 'Centre')


ALR_COLS = ['ALP', 'GRN', 'ON', 'OTH'] if data_year == 2022 else ['ALP','GRN','OTH'] # for errors


# build the electorate_pairs and next_contesting_parties dictionaries; rename independents from INDX to IND_First_Last for easier tracking 

def format_specific_party(row):
    """
    Helper function to convert 'IND' into 'IND_First_Last'.
    Leaves standard parties (e.g., 'ALP', 'ON') untouched.
    """
    ab = str(row['PartyAb']).strip()
    if ab.startswith('IND'):
        # Title case and replace spaces with underscores (e.g. "Ali Cupper" -> "Ali_Cupper")
        first = str(row['first_name']).strip().title().replace(' ', '_')
        last = str(row['surname']).strip().title().replace(' ', '_')
        return f"IND_{first}_{last}"
    return ab

# old contesting parties for calibration for kappas
LA_Candidates_prev = LA_Candidates.copy()
LA_Candidates_prev['PartyAb_Specific'] = LA_Candidates_prev.apply(format_specific_party, axis=1)
old_contesting_parties = LA_Candidates_prev.groupby('div_nm')['PartyAb_Specific'].agg(list).to_dict()
old_contesting_parties = {(div, div): old_contesting_parties[div] for div in old_contesting_parties.keys()} # for the calibration data

COAL_set = {'LP', 'NP'}

if election_year != 2026:
    LA_Candidates_next = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
    if election_year == 2022:
        LA_Candidates_next =  LA_Candidates_next.loc[~(LA_Candidates_next['div_nm']=='Narracan'),] # remove Narracan due to no election
    LA_Candidates_next['PartyAb_Specific'] = LA_Candidates_next.apply(format_specific_party, axis=1)
    next_contesting_parties = LA_Candidates_next.groupby('div_nm')['PartyAb_Specific'].agg(list).to_dict()

    if not is_redistribution:
        electorate_pairs = [(div, div) for div in LA_Candidates_next['div_nm'].unique().tolist()]
        next_contesting_parties = {(div, div): next_contesting_parties[div] for div in next_contesting_parties.keys()}
    else:
        electorate_pairs = [pair for pair in rediv_transfer_props_dict.keys() if pair[1] in LA_Candidates_next['div_nm'].unique()] # Narracan 2022 adjustment
        next_contesting_parties = {pair: next_contesting_parties[pair[1]] for pair in electorate_pairs}

    
    #import pdb; pdb.set_trace()

else:
    #LA_Candidates_temp = LA_Candidates.copy()
    #LA_Candidates_temp['PartyAb_Specific'] = LA_Candidates_temp.apply(format_specific_party, axis=1)
    
    # CHANGE for actual parties in 2026
    # Currently, just add ON and VNS in every district, remove deregistered parties
    #deregistered_list = ['REAS','DHJP','AVP','HAP','SDA','TMP','FPV', 'HMP', 'DLP','LDP','SFF'] # FPV included as announced not contesting
    #base_parties = LA_Candidates_temp.groupby('div_nm')['PartyAb_Specific'].agg(list).to_dict()

    #inds_contesting['ind_code'] = 'IND_' + inds_contesting['Candidate'].astype(str).str.strip().str.replace(r'\s+', '_', regex=True)
    #inds_2026_by_div = inds_contesting.groupby('Electorate')['ind_code'].agg(list).to_dict()
    #for div, parties in base_parties.items():
        # Keep only registered parties, stripping out any 2022 INDs
    #    retained_parties = [p for p in parties if p not in deregistered_list and not str(p).startswith('IND')]
        
    #    div_inds_2026 = inds_2026_by_div.get(div, [])
        
    #    next_contesting_parties[div] = list(dict.fromkeys(retained_parties + ['ON', 'VNS'] + div_inds_2026))
    
    # OLD next_contesting_parties = {div: list(dict.fromkeys([p for p in parties if p not in deregistered_list] + ['ON','VNS'])) for div, parties in base_parties.items()}

   # fix IND additions/removals
    #next_contesting_parties['Hawthorn'] = [p for p in next_contesting_parties['Hawthorn'] if p != 'IND_Melissa_Lowe']
    #next_contesting_parties['Mornington'] = [p for p in next_contesting_parties['Mornington'] if p != 'IND_Kate_Lardner']
    #next_contesting_parties['Macedon'] = ['LP','GRN','ALP','ON','VNS','AJP']
    #next_contesting_parties['Narracan'] = ['LP','GRN','ALP','ON','VNS','FFP']
    #next_contesting_parties['Narracan'] = ['LP','GRN','ALP','ON','VNS','FFP']
    #next_contesting_parties['Clarinda'].append('IND_Caroline_White')
    #next_contesting_parties['Macedon'].append('IND_Caroline_White')
    
    candidates_2026 = pd.read_csv('VIC-2026/VIC-2026-Candidates.csv')
    next_contesting_parties = {}
    for div, group in candidates_2026.groupby("div_nm"):
        # dict.fromkeys() removes duplicates while preserving the exact order they appeared on the ballot
        next_contesting_parties[div] = list(dict.fromkeys(group["PartyAb"].tolist()))



        #if div in ['Bendigo East','Bendigo West','Euroa','Mildura','Morwell','Shepparton']:
        #    next_contesting_parties[div] += ['LP']
        #if div in ['Bass']:
        #    next_contesting_parties[div] += ['NP']
    next_contesting_parties['Werribee'] = [p if p != 'TWP' else 'IND_Paul_Hopper' for p in next_contesting_parties['Werribee']]
    next_contesting_parties['Brighton'] = [p if p != 'IND_John_Casley' else 'IND_John_Tiger_Casley' for p in next_contesting_parties['Brighton'] ] # Tiger Casley clause

    electorate_pairs = [(div, div) for div in next_contesting_parties.keys()]
    next_contesting_parties = {(div, div): next_contesting_parties[div] for div in next_contesting_parties.keys()}

    
    
    # INDs only: 
    # LA_Candidates_next = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Candidates-Formatted.csv')
    # LA_Candidates_next['PartyAb_Specific'] = LA_Candidates_next.apply(format_specific_party, axis=1)
    # old_contesting_parties = LA_Candidates_next.groupby('div_nm')['PartyAb_Specific'].agg(list).to_dict()
    # new_INDs = {div:[p for p in next_contesting_parties[div] if p.startswith('IND')] for div in next_contesting_parties.keys()}
    # old_INDs = {div:[p for p in old_contesting_parties[div] if p.startswith('IND')] for div in old_contesting_parties.keys()}
    # same_IND_divs = [new_INDs[div] for div in old_INDs if div in new_INDs and set(old_INDs[div]) & set(new_INDs[div])]

COAL_party_change_dict = {}
for pair, new_parties in next_contesting_parties.items():
    old_coal = set(old_contesting_parties[(pair[0], pair[0])]).intersection(COAL_set)
    new_coal = set(new_parties).intersection(COAL_set)

    if len(old_coal) > 0: # NEW: Only defer when at least 1 COAL party existed in baseline!
        if len(old_coal) != len(new_coal) or old_coal != new_coal:
            COAL_party_change_dict[pair] = {
                'removed': sorted(list(old_coal - new_coal)),
                'added': sorted(list(new_coal - old_coal))
            }

            next_contesting_parties[pair] = [p for p in next_contesting_parties[pair] if p not in COAL_party_change_dict[pair]['added']] + [p for p in COAL_party_change_dict[pair]['removed']]



if election_year == 2026:
    TARGET_TEALS = {'Footscray': 'IND_Jess_Marsh', 'Mornington': 'IND_Melissa_Goffin', 'Hawthorn': 'IND_Shima_Ibuki'} # Sophie Torney already there
    TARGET_WEST = ['Footscray','Kororoit','Laverton','Melton','Niddrie','St Albans','Sunbury','Sydneham', 'Tarneit','Point Cook','Williamstown'] # doesn't include Paul Hopper

elif election_year == 2022:
    TARGET_TEALS = {'Brighton': 'IND_Felicity_Frederico', 'Caulfield': 'IND_Nomi_Kaltmann', 'Hawthorn': 'IND_Melissa_Lowe','Kew':'IND_Sophie_Torney','Bellarine':'IND_Sarah_Fenton'} # Clarke Martin, Jacqui Hawkins already there
    TARGET_WEST = []
else:
    TARGET_TEALS = {}
    TARGET_WEST = []

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

def sync_dop_table_inds(dop_table_dict: dict, la_candidates_df: pd.DataFrame) -> dict:
    """
    Scans the LA_Candidates dataframe, isolates Independents per district, 
    sorts them by ballot order, and translates 'IND1', 'IND2', etc. in the 
    DOP_table_dict DataFrames to explicit 'IND_First_Last' column headers.
    """
    synced_dop_dict = {}
    
    for div, df in dop_table_dict.items():
        # Isolate candidates for this specific district
        div_cands = la_candidates_df[la_candidates_df['div_nm'] == div].copy()
        
        # Filter for Independents and sort chronologically by ballot order
        inds = div_cands[div_cands['PartyAb'].astype(str).str.startswith('IND')].sort_values('ballot_order')
        
        # Build the renaming mapping for this district
        rename_map = {}
        for idx, (_, row) in enumerate(inds.iterrows()):
            # Format the specific name
            first = str(row['first_name']).strip().title().replace(' ', '_')
            last = str(row['surname']).strip().title().replace(' ', '_')
            full_ind_name = f"IND_{first}_{last}"
            
            # Map the chronological tag (IND1, IND2, IND3...) to the new name
            rename_map[f"IND{idx + 1}"] = full_ind_name
            
            # Fallback: In case a single Independent is just labelled "IND" in the DataFrame
            if idx == 0:
                rename_map["IND"] = full_ind_name
                
            # Fallback: If LA_Candidates explicitly stores the tag (like "IND1") in PartyAb, map that too
            party_ab = str(row['PartyAb']).strip()
            if party_ab.startswith('IND'):
                rename_map[party_ab] = full_ind_name

        # 4. Rename the DataFrame columns and store it
        synced_dop_dict[div] = df.rename(columns=rename_map)
        
    return synced_dop_dict

DOP_table_dict = sync_dop_table_inds(DOP_table_dict, LA_Candidates)

#################################################### Step 1: Remove Incumbency advantage (Simulated) ##################################################


def prepare_incumbents_df(
    inc_df: pd.DataFrame, master_inc_df: pd.DataFrame, data_year: int
) -> pd.DataFrame:
  """Enriches inc_df in-place using the exact Mc/Mac capitalization and concordance logic from the original 3CP dataset pipeline.
  """
  tenure_col = f'{data_year}_tenure'

  # 1. Flag open seats
  inc_df['is_open'] = np.where(inc_df['incumbent'] == 1, False, True)
  inc_df['years_tenure'] = np.nan

  # 2. Extract first_name and surname from candidate_name ("SURNAME, First Name")
  # and reconstruct inc_name using your exact Mc/Mac logic
  def extract_inc_name(raw_name: str) -> str:
    if pd.isna(raw_name):
      return ''
    s = str(raw_name).strip()
    if ',' in s:
      surname, first_name = [part.strip() for part in s.split(',', 1)]
    else:
      # Fallback if already "First Last"
      parts = s.rsplit(' ', 1)
      first_name = parts[0].strip() if len(parts) > 1 else ''
      surname = parts[-1].strip()

    # Exact logic from original snippet:
    # (first_name + " " + surname.title().replace(r'\b(Mc|Mac)([a-z])', ...))
    surname_clean = re.sub(
        r'\b(Mc|Mac)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        surname.title(),
    )
    return f'{first_name} {surname_clean}'.strip()

  # 3. Build lookup table from master_inc_df matching the original snippet's iloc[0] logic
  valid_master = master_inc_df.dropna(subset=[tenure_col]).drop_duplicates(subset=['inc_name'], keep='first')
  tenure_map = dict(zip(valid_master['inc_name'], valid_master[tenure_col]))

  # 4. Map tenure for incumbents
  inc_mask = inc_df['incumbent'] == 1
  inc_names = inc_df.loc[inc_mask, 'candidate_name'].apply(extract_inc_name)
  inc_df['inc_clean_name'] = ''
  inc_df.loc[inc_mask, 'inc_clean_name'] = inc_names.str.lower().str.strip()
  inc_df.loc[inc_mask, 'years_tenure'] = inc_names.map(tenure_map)
  # Compatibility alias for the OLS IA model
  inc_df['years_served'] = inc_df['years_tenure']

  return inc_df

def generate_simulated_ia_adjustments(
    inc_df: pd.DataFrame, 
    ia_model: Any, 
    dop_table_dict: dict, 
    simulate: bool = False, 
    rng: Optional[np.random.Generator] = None
) -> dict:
    """
    Simulates IA for data_year incumbents and distributes it zero-sum across the 3CP candidates. Incumbent gets -ia_adj, the other two challengers get +0.5 * ia_adj.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    beta_hat = np.asarray(ia_model.params)
    cov_beta = np.asarray(ia_model.cov_params())
    sigma_resid = np.sqrt(ia_model.mse_resid)
    
    # Draw parameters once for the cycle
    beta_sim = rng.multivariate_normal(mean=beta_hat, cov=cov_beta) if simulate else beta_hat
    ia_adjustments_dict = {}
    ia_tracker_dict = {}
    
    meta_cols = {'div_nm', 'CandRemaining', 'TOTAL', 'CountType'}
    
    for div, df in dop_table_dict.items():
        # 1. Grab the 3CP contenders directly from the raw dataframe
        lh_cp = df[df['CandRemaining'] == 3]
        if lh_cp.empty:
            continue
            
        row = lh_cp.iloc[0]
        top_3 = [c for c in df.columns if c not in meta_cols and pd.notna(row[c]) and row[c] > 0]
        
        if len(top_3) != 3:
            import pdb; pdb.set_trace()
            continue
            
        div_adj = {c: 0.0 for c in top_3}
        
        # 2. Find all incumbents for this electorate
        div_incs = inc_df[(inc_df['div_nm'] == div) & (inc_df['incumbent'] == 1)]
        
        for _, inc_row in div_incs.iterrows():
            tenure = float(inc_row.get('years_tenure', np.nan))
            if pd.isna(tenure) or tenure <= 0:
                continue
                
            party = str(inc_row.get('PartyAb', '')).upper()
            is_coal = 1.0 if party in ['LP', 'NP', 'LNP', 'COAL'] else 0.0
            log_tenure = np.log(max(tenure, 1.0))

            # Prioritize deterministic shrunk IA (alpha_ia) if present
            if 'alpha_ia' in inc_row and pd.notna(inc_row['alpha_ia']):
                ia_adj = float(inc_row['alpha_ia'])
            else:
                # Simulate IA (Systematic + Idiosyncratic Noise)
                eps = rng.normal(0, sigma_resid) if simulate else 0.0
                ia_adj = float(np.dot([1.0, log_tenure, is_coal], beta_sim) + eps)
            
            
            # Match incumbent within the top 3 by party prefix
            coal_aliases = ['LP', 'NP', 'LNP'] if is_coal else [party]
            inc_cand = next((c for c in top_3 if any(str(c).upper().startswith(p) for p in coal_aliases)), None)
            
            # Apply symmetric zero-sum logic (+1.0 to incumbent, -0.5 to challengers)
            if inc_cand:
                div_adj[inc_cand] -= ia_adj
                challengers = [o for o in top_3 if o != inc_cand]
                for o in challengers:
                    div_adj[o] += ia_adj * 0.5

                # Record IA for Step 6 re-application
                ia_tracker_dict[div] = {
                    'inc_name': inc_row.get(
                        'inc_clean_name', inc_row.get('candidate_name', '')
                    ),
                    'inc_cand': inc_cand,
                    'inc_party': party,
                    'is_coal': bool(is_coal),
                    'stripped_ia': ia_adj,
                    'tenure_2022': tenure,
                    'challengers': challengers,
                }
                        
        # Filter out zeroes and store
        final_adj = {k: v for k, v in div_adj.items() if abs(v) > 1e-9}
        if final_adj:
            ia_adjustments_dict[div] = final_adj
            
    return ia_adjustments_dict, ia_tracker_dict

# FIX: currently, takes in existing average modelled ia, but should best be simulated!
def execute_strip_incumbency(dop_table_dict: dict, inc_df: pd.DataFrame, csv_path: str = f"VIC-{data_year}/VIC-{data_year}-IA-adjustments.csv"):
    """
    Step 1: Reads deterministic cleansed baselines and applies them to DOP Tables.
    """
    # Load the pre-calculated adjustments
    ia_df = pd.read_csv(csv_path)
    if data_year == 2022: # REFINE - manually handle missing ia_df rows by running the model
        # x_dict = {"const": 1.0, "log_tenure": np.log(max(1.0, 19)), demo_lookup['Bendigo East']: 1.0}
        # x_new = np.array([x_dict.get(k, 0.0) for k in ia_params_dict['coefficients'].keys()])
        # ia =  float(np.dot(x_new, np.array(list(ia_params_dict['coefficients'].values()))) +0)
        missing_ia_rows = [
            ['Bayswater', 'ALP', 0.0041, np.nan, np.nan, np.nan, np.nan],
            ['Bayswater', 'LP', 0.0361, np.nan, np.nan, np.nan, np.nan],
            ['Bass', 'ALP', 0.0399, np.nan, np.nan, np.nan, np.nan],
        ]
        ia_df = pd.concat([ia_df, pd.DataFrame(missing_ia_rows, columns=ia_df.columns)], ignore_index=True)
    elif data_year == 2018:
        missing_ia_rows = [
            ['Bendigo East', 'ALP', 0.0400, np.nan, np.nan, np.nan, np.nan]
        ]
        ia_df = pd.concat([ia_df, pd.DataFrame(missing_ia_rows, columns=ia_df.columns)], ignore_index=True)
    elif data_year == 2006:
        missing_ia_rows = [ # FIX - bad model
            ['Polwarth', 'LP', 0.0334, np.nan, np.nan, np.nan, np.nan],
            ['Murray Valley', 'NP', 0.0314, np.nan, np.nan, np.nan, np.nan]
        ]
        ia_df = pd.concat([ia_df, pd.DataFrame(missing_ia_rows, columns=ia_df.columns)], ignore_index=True)
    assert not [div for div in inc_df['div_nm'].unique() if div not in ia_df['electorate'].unique()]

    ia_grouped = ia_df.groupby('electorate')

    inc_name_map = inc_df[inc_df['incumbent'] == 1].set_index(['div_nm', 'PartyAb'])['inc_clean_name'].to_dict()
    inc_tenure_map = inc_df[inc_df['incumbent'] == 1].set_index(['div_nm', 'PartyAb'])['years_tenure'].to_dict()
    
    stripped_tables = {}
    static_high_ind_districts = []
    ia_tracker_dict = {}

    for div, raw_df in dop_table_dict.items():
        table = DOPTable(raw_df)
        
        # Normalize to fractions
        tot_votes = sum(table.count_0.values())
        if tot_votes > 1.5:
            table.count_0 = {k: v / tot_votes for k, v in table.count_0.items()}
            for rnd in table.rounds:
                rnd['V_elim'] /= tot_votes
                rnd['transfers'] = {k: v / tot_votes for k, v in rnd['transfers'].items()}

        ind_cands = [c for c in table.candidates if str(c).startswith('IND')]
        if sum(table.count_0.get(c, 0.0) for c in ind_cands) > 0.15:
            static_high_ind_districts.append(div)

        # Apply the strip with sqrt proportionality
        trackers_for_div = []
        if div in ia_grouped.groups:
            # Loop over all incumbents identified in ia_df for this electorate (e.g. ALP and LP in Bayswater)
            for _, row in ia_grouped.get_group(div).iterrows():
                inc_cand = row['inc_cand']
                target_ia = max(0.0, float(row['ia_alpha']))

                if target_ia > 0:
                    table = table.strip_incumbency_forward(target_ia=target_ia, incumbent_party=inc_cand)
                
                inc_party = ''.join(filter(str.isalpha, str(inc_cand)))
                trackers_for_div.append({
                    'inc_name': inc_name_map.get((div, inc_party), ''),
                    'inc_party': inc_party,
                    'stripped_ia': target_ia,
                    'tenure': inc_tenure_map.get((div, inc_party), 0)
                })
        
        if trackers_for_div:
            ia_tracker_dict[div] = trackers_for_div

        stripped_tables[div] = table

    return stripped_tables, static_high_ind_districts, ia_tracker_dict


inc_df = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-Incumbents.csv')
inc_next_df = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-Incumbents.csv')
master_inc_df = pd.read_csv('VIC-Incumbents-df-2006-2026.csv')

prepare_incumbents_df(inc_df, master_inc_df, data_year=data_year)
prepare_incumbents_df(inc_next_df, master_inc_df, data_year=election_year)

stripped_tables, static_high_ind_districts, ia_tracker_dict = execute_strip_incumbency(
    dop_table_dict=DOP_table_dict,
    inc_df = inc_df,
    csv_path=f"VIC-{data_year}/VIC-{data_year}-IA-adjustments.csv"
)

# introduce paired keys for this stage
stripped_tables = {(div, div): stripped_tables[div] for div in stripped_tables.keys()}

#import pdb; pdb.set_trace()
#################################################### Step 2: Remove non-contesting parties and INDs (Simulated) #########################################################################################

def remove_uneliminated_minors_via_model(
    table: Any,
    unelim_minors: List[str],
    div: str,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    simulate: bool = False,
    rng: Optional[np.random.Generator] = None
) -> Any:
    """
    Model-driven minor party removal for when DOP table is not available. Uses State A vs State B Logit delta to extract weights, directly updating Count 0.
    """
    if not unelim_minors: return table
    if rng is None: rng = np.random.default_rng()

    menu_A = [c for c, v in table.count_0.items() if v > 1e-9 and not str(c).startswith('IND')]
    menu_B = [c for c in menu_A if c not in unelim_minors]
    if not menu_B: return table

    lc_series = lc_df[div].dropna().copy() if div in lc_df.columns else pd.Series(dtype=float)
    if lc_series.sum() > 1.5: lc_series = lc_series / 100.0

    zero_lc_minors = [m for m in menu_A if lc_series.get(m, 0.0) <= 1e-9]
    if zero_lc_minors and not lc_series.empty:
        mod_lc = apply_LC_orphan_capture(zero_lc_minors, table.count_0, lc_series, party_df)
    else:
        mod_lc = lc_series.copy()

    alpha, beta, gamma = model.unpack_params(model.theta)

    # State A vs B Delta Calculation
    ms_A, cap_A, ctb_A, cd_A, lc_mass_A = _build_state_vars(menu_A, mod_lc, table.count_0, party_df)
    s_hat_A = model.evaluate_state_allocation(lc_mass_A, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)

    ms_B, cap_B, ctb_B, cd_B, lc_mass_B = _build_state_vars(menu_B, mod_lc, table.count_0, party_df)
    s_hat_B = model.evaluate_state_allocation(lc_mass_B, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)

    pos_deltas = {c: max(0.0, s_hat_B.get(c, 0.0) - s_hat_A.get(c, 0.0)) for c in menu_B}
    sum_deltas = sum(pos_deltas.values())

    if sum_deltas > 1e-9:
        cand_weights = {c: pos_deltas[c] / sum_deltas for c in menu_B}
    else:
        tot_b = sum(table.count_0.get(c, 0.0) for c in menu_B)
        cand_weights = {c: (table.count_0[c] / tot_b if tot_b > 1e-9 else 1.0 / len(menu_B)) for c in menu_B}

    if simulate and len(menu_B) > 1:
        alpha_dir = np.array([max(1e-3, cand_weights[c] * 50.0) for c in menu_B])
        draw = rng.dirichlet(alpha_dir)
        cand_weights = {c: float(draw[i]) for i, c in enumerate(menu_B)}

    # SIMPLIFIED: Directly shift Count 0 mass to Menu B survivors
    for u_min in unelim_minors:
        u_min_vol = table.count_0.pop(u_min, 0.0)
        for c in menu_B:
            table.count_0[c] = table.count_0.get(c, 0.0) + (u_min_vol * cand_weights[c])

    table.candidates = [c for c in table.candidates if c not in unelim_minors]
    return table

def synthesize_dop_composition(div: str, region: str, dop_lookup: dict) -> dict:
    """
    Synthesizes the geographic hierarchy to return a normalized 5-bloc composition array
    for un-defecting minor Independents.
    """
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    
    # 1. Geographic Hierarchy Lookup
    if div in dop_lookup.get('electorates', {}):
        mu_raw = dop_lookup['electorates'][div].copy()
    elif region in dop_lookup.get('regions', {}):
        mu_raw = dop_lookup['regions'][region].copy()
    else:
        mu_raw = dop_lookup['state'].copy()
        
    # mu_raw might contain 5 items (Standard Blocs) or 6 items (including Repeat_IND).
    # We only want the first 5 standard blocs for the pure anchor.
    mu_standard = np.array(mu_raw[:5], dtype=np.float64)
    
    # 2. Strict Normalization
    # We strip out any Repeat_IND probability and ensure 100% flows to the 5 standard blocs
    mu_sum = np.sum(mu_standard)
    if mu_sum > 1e-9:
        mu_normalized = mu_standard / mu_sum
    else:
        # Absolute fallback if data is totally corrupted: dump to Centre
        mu_normalized = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        
    # 3. Map to Dictionary Format for safe JSON-like storage
    comp_dict = {b: float(mu_normalized[i]) for i, b in enumerate(BLOCS)}
    
    return comp_dict



def remove_uneliminated_inds(table, unelim_inds, div, region, dop_lookup, party_df):
    """
    Synthetic IND Liquidation for candidates that survived DOP truncation.
    Distributes primary vote (and any accumulated transfers) using the bloc priors.
    """
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']

    mu_dict = synthesize_dop_composition(div, region, dop_lookup)
    
    for u_ind in unelim_inds:
        # Get active non-IND survivors
        survivors = [c for c, v in table.count_0.items() if c != u_ind and v > 1e-9]
        if not survivors:
            continue 
            
        survivor_blocs = {b: [] for b in BLOCS}
        bloc_totals = {b: 0.0 for b in BLOCS}
        
        for c in survivors:
            b = get_party_bloc(c, party_df)
            survivor_blocs[b].append(c)
            bloc_totals[b] += table.count_0[c]
            
        populated_mu_sum = sum(mu_dict[b] for b in BLOCS if survivor_blocs[b])
        
        # Calculate reallocation weights
        cand_weights = {}
        if populated_mu_sum < 1e-9:
            cand_weights = {c: 1.0 / len(survivors) for c in survivors}
        else:
            for b in BLOCS:
                if not survivor_blocs[b]:
                    continue
                adjusted_mu_b = mu_dict[b] / populated_mu_sum
                for c in survivor_blocs[b]:
                    cand_share_in_bloc = table.count_0[c] / bloc_totals[b] if bloc_totals[b] > 1e-9 else (1.0 / len(survivor_blocs[b]))
                    cand_weights[c] = adjusted_mu_b * cand_share_in_bloc

        # Update Count 0
        u_ind_vol = table.count_0.pop(u_ind, 0.0)
        for c in survivors:
            table.count_0[c] += (u_ind_vol * cand_weights[c])
            
        # Update any surviving intermediate rounds
        if hasattr(table, 'rounds') and isinstance(table.rounds, list):
            for rnd in table.rounds:
                if 'transfers' in rnd and u_ind in rnd['transfers']:
                    trans_vol = rnd['transfers'].pop(u_ind)
                    for c in survivors:
                        rnd['transfers'][c] = rnd['transfers'].get(c, 0.0) + (trans_vol * cand_weights[c])
                        
    table.candidates = [c for c in table.candidates if c not in unelim_inds]

    #import pdb; pdb.set_trace()
    return table



def execute_party_removal(
    stripped_tables: Dict[tuple, Any],
    static_high_ind_districts: Set[str],
    next_contesting_parties: Dict[tuple, List[str]],
    LC_FPs_df: pd.DataFrame,
    party_df: pd.DataFrame,             
    dop_lookup: dict,                   
    electorate_to_region: Dict[str, str], 
    model: Any,
    simulate: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, Any]:
    """
    Executes Step 2 of the architecture:
    Removes INDs and non-recontesting parties stochastically. Bypasses electorates marked as High-IND.
    """

    if rng is None:
        rng = np.random.default_rng()

    latent_tables = {}
    
    for key, orig_table in stripped_tables.items():
        div = key[0]
        table = copy.deepcopy(orig_table)
        contesting_next = next_contesting_parties.get(key, [])
        is_high_ind = div in static_high_ind_districts

        cands_to_remove = []
        inds_to_remove = []
        minors_to_remove = []
        
        # Identify candidates to remove
        COAL_SET = {'LP', 'NP', 'LNP', 'COAL'}
        next_has_coal = bool(COAL_SET & set(contesting_next))

        for c in table.candidates:
            if str(c).startswith('IND'):
                if not is_high_ind:
                    inds_to_remove.append(c)
                    cands_to_remove.append(c)
            elif c not in contesting_next:
                if c in COAL_SET:
                    if next_has_coal: # COAL_double_divs - don't remove at this stage
                        continue
                    else:
                        print("No COAL parties in next")
                        import pdb; pdb.set_trace()
                elif c in ['ALP', 'GRN']:
                    print(f"[ALERT] Major party {c} missing in {div}!") # 2006 Kalkallo should be only case - just let it go through
                    import pdb; pdb.set_trace()
                    minors_to_remove.append(c)
                    cands_to_remove.append(c)
                else:
                    minors_to_remove.append(c)
                    cands_to_remove.append(c)

        if not cands_to_remove:
            latent_tables[key] = table
            continue

        # Partition by DOP-assisted removal vs empirical removal
        elim_in_rounds = {rnd.get('eliminated') for rnd in table.rounds} if hasattr(table, 'rounds') else set()
        elim_cands = [c for c in cands_to_remove if c in elim_in_rounds] # All candidates (IND or Minor) who actually have DOP table transfer data
        
        # Candidates who survived DOP truncation and need empirical removal
        unelim_inds = [c for c in inds_to_remove if c not in elim_in_rounds]
        unelim_minors = [c for c in minors_to_remove if c not in elim_in_rounds]

        #print(elim_cands, unelim_inds, unelim_minors)

        # Execute removals in sequence
                
        # 1. Unified DOP table removal
        if elim_cands:
            if simulate:
                table = table.remove_candidates_stochastic(elim_cands, rng=rng)
            else:
                table = table.remove_candidates(elim_cands)
                
        # 2. Remove INDs using historical flows in dop_lookup
        if unelim_inds:
            region = electorate_to_region.get(div, 'Unknown')
            table = remove_uneliminated_inds(table, unelim_inds, div, region, dop_lookup, party_df)

        # 3. Remove minors via Logit flow model
        if unelim_minors:
            table = remove_uneliminated_minors_via_model(table, unelim_minors, div, LC_FPs_df, party_df, model, simulate=simulate, rng=rng)
        latent_tables[key] = table
        
    return latent_tables




#################################################### Step 3: Add new parties to Count 0 (Simulated) #########################################################################################

# calibrate kappas



def diagnose_and_recalibrate_party(
    party_abbr: str,
    contested_divs: List[str],
    latent_tables: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit'
) -> Tuple[pd.DataFrame, float]:
    """
    Computes kappa (the realization scalar) for a party across contested districts.
    Compares observed LA Count 0 against structural model prediction S_hat.
    """
    alpha, beta, gamma = model.unpack_params(model.theta)
    results = []

    for div in contested_divs:
        if not any(k[0] == div for k in latent_tables) or div not in lc_df.columns:
            import pdb; pdb.set_trace()
            continue

        table = latent_tables[(div, div)] # always (div, div) key for recalibration
        actual_c0 = table.count_0.get(party_abbr, 0.0)
        if actual_c0 <= 1e-9:
            continue

        lc_series = lc_df[div].dropna().copy()
        if lc_series.sum() > 1.5:
            lc_series = lc_series / 100.0

        lc_mass = np.zeros(5)
        for party, share in lc_series.items():
            b_idx = BLOCS.index(get_party_bloc(party, party_df))
            lc_mass[b_idx] += share
        if lc_mass.sum() > 0:
            lc_mass = lc_mass / lc_mass.sum()

        menu = list(table.count_0.keys())
        menu_size = np.zeros(5)
        c_cap = np.zeros(5)
        cand_to_bloc = {}
        cand_demand = {}

        for cand in menu:
            b = BLOCS.index(get_party_bloc(cand, party_df))
            cand_to_bloc[cand] = b
            menu_size[b] += 1.0
            
            # Coalition alias fix
            if b in [0, 1]:
                u = lc_series.get(cand, 0.0)
                if u <= 1e-9:
                    u = table.count_0.get(cand, 0.0)
                    if u <= 1e-9 and cand in ['LP', 'NP', 'LNP']:
                        coal_lc_mass = sum(lc_series.get(a, 0.0) for a in ['LP', 'NP', 'LNP'])
                        coal_la_cands = [c for c in menu if c in ['LP', 'NP', 'LNP']]
                        
                        if len(coal_la_cands) == 1:
                            u = coal_lc_mass
                        elif len(coal_la_cands) > 1:
                            emp_c = table.count_0.get(cand, 0.0)
                            emp_tot = sum(table.count_0.get(c, 0.0) for c in coal_la_cands)
                            u = coal_lc_mass * (emp_c / emp_tot) if emp_tot > 0 else coal_lc_mass / len(coal_la_cands)
            else:
                u = lc_series.get(cand, 0.0)
                
            cand_demand[cand] = u
            if b in [2, 3, 4]:
                c_cap[b] += u

        c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
        c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
        for b in [2, 3, 4]:
            c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0

        s_hat = model.evaluate_state_allocation(
            lc_mass, menu, c_cap, menu_size, cand_to_bloc, cand_demand,
            alpha, beta, gamma
        )

        pred_c0 = s_hat.get(party_abbr, 0.0)
        ratio = actual_c0 / pred_c0 if pred_c0 > 0 else 1.0

        results.append({
            'District': div,
            'Actual_LA_C0': actual_c0,
            'Model_Pred_C0': pred_c0,
            'Residual': pred_c0 - actual_c0,
            'Realization_Ratio': ratio
        })

    diag_df = pd.DataFrame(results)

    if diag_df.empty:
        return diag_df, 1.0
    
    raw_kappa = float(diag_df['Realization_Ratio'].median())

    return diag_df, raw_kappa


def calibrate_new_party_kappa(
    party_abbr: str,
    prior_vote_share: float,
    latent_tables: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit'
) -> float:
    """
    Unilaterally injects new party into LC across all districts, measures mean uncalibrated vote share in LC, and returns the Kappa needed to hit the specified prior_vote_share.
    """
    s_hat_1_list = []
    alpha, beta, gamma = model.unpack_params(model.theta)
    
    for key, table in latent_tables.items():
        div = key[0]
        if div not in lc_df.columns: 
            continue
        
        emp_c0 = table.count_0
        lc_series = lc_df[div].dropna().copy()
        if lc_series.sum() > 1.5: 
            lc_series = lc_series / 100.0
        
        # 1. Unilateral Injection (k=1)
        mod_lc = apply_LC_orphan_capture([party_abbr], emp_c0, lc_series, party_df)
        
        # 2. Add Entrant to Menu
        menu_B = [c for c, v in emp_c0.items() if v > 1e-9]
        if party_abbr not in menu_B:
            menu_B.append(party_abbr)
            
        # 3. Model Evaluation (kappa = 1.0)
        ms_B, cap_B, ctb_B, cd_B, lc_mass = _build_state_vars(menu_B, mod_lc, emp_c0, party_df)
        s_hat_B = model.evaluate_state_allocation(lc_mass, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)
        
        s_hat_1_list.append(s_hat_B.get(party_abbr, 0.0))
        
    mean_s_1 = np.mean(s_hat_1_list) if s_hat_1_list else 0.0
    
    # Linear Ratio Calculation
    if mean_s_1 > 1e-9:
        return prior_vote_share / mean_s_1
    else:
        print(f"[Warning] {party_abbr} has 0.0 baseline everywhere. Defaulting kappa to 1.0")
        return 1.0



def pre_calibrate_all_kappas(
    parties_to_add: List[str],
    latent_tables: Dict[tuple, Any],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    synthetic_priors: Dict[str, dict]  # Now accepts a dict of dicts for pseudo_N
) -> Dict[str, dict]:
    """
    Dynamically categorizes entrants and calculates baseline kappas.
    Extracts pseudo_N for synthetics to inform Bayesian shrinkage.
    """
    calibration_data = {}
    
    for party in parties_to_add:
        if party in synthetic_priors:
            # Route A: Synthetic Party
            prior = synthetic_priors[party]['prior']
            pseudo_n = synthetic_priors[party]['pseudo_N']
            
            k = calibrate_new_party_kappa(
                party_abbr=party, 
                prior_vote_share=prior,
                latent_tables=latent_tables, 
                lc_df=lc_df,
                party_df=party_df, 
                model=model
            )
            calibration_data[party] = {
                'type': 'synthetic',
                'raw_kappa': k,
                'N': pseudo_n  # Pass the pseudo-sample size directly to the shrinkage engine
            }
        else:
            # Route B: Historical Party
            contested_divs = [d[0] for d, t in latent_tables.items() if t.count_0.get(party, 0.0) > 1e-9]
            
            if not contested_divs:
                print(f"[Warning] {party} is historical but contested 0 LA seats. Defaulting kappa=1.0")
                k = 1.0
                n_seats = 0
            else:
                _, k = diagnose_and_recalibrate_party(
                    party_abbr=party, 
                    contested_divs=contested_divs,
                    latent_tables=latent_tables, 
                    lc_df=lc_df,
                    party_df=party_df, 
                    model=model
                )
                n_seats = len(contested_divs)
                
            calibration_data[party] = {
                'type': 'historical',
                'raw_kappa': k,
                'N': n_seats
            }
            
    return calibration_data


def draw_stochastic_kappas(
    calibration_data: Dict[str, dict], 
    simulate: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    """
    Draws probabilistic kappas using unified Beta distribution shrinkage.
    Both historical and synthetic parties shrink towards 1.0 based on their N.
    """

    if simulate and rng is None:
        rng = np.random.default_rng()

    sim_kappas = {}
    
    for party, data in calibration_data.items():
        raw_k = data['raw_kappa']
        N = data['N']
        
        # 1. Unified Bayesian Shrinkage (W)
        W_det = (N**2) / (N**2 + 81.0)  # Midpoint K=9
        
        if simulate:
            alpha_w = max(W_det * 15.0, 1.0)
            beta_w  = max((1.0 - W_det) * 15.0, 1.0)
            W_final = rng.beta(alpha_w, beta_w)
        else:
            # Deterministic expected value
            W_final = W_det
        
        sim_k = (W_final * raw_k) + ((1.0 - W_final) * 1.0)
        
        # 2. Additional aleatoric structural noise exclusively for synthetics
        if simulate and data['type'] == 'synthetic':
            sim_k = rng.normal(loc=sim_k, scale=0.1 * sim_k)
            
        sim_kappas[party] = float(np.clip(sim_k, 0.1, 10.0))
            
    return sim_kappas





def is_minor_bloc_party(party: str, party_df: pd.DataFrame) -> bool:
    """Identifies minor parties. Strictly protects ALP, Coalition, and GRN from orphan capture."""
    return party not in ['ALP', 'LP', 'NP', 'LNP', 'GRN'] and not str(party).startswith('IND')

def apply_LC_orphan_capture(
    cands_to_add: List[str],
    current_c0: Dict[str, float],
    lc_series: pd.Series,
    party_df: pd.DataFrame
) -> pd.Series:
    """
    Applies Multi-Entrant Capture of LC Orphans: extracts mass from orphans not contesting LA and scales them down proportionally, perfectly preserving the simplex and bloc volumes.
    """
    mod_lc = lc_series.copy()
    
    # Group entrants by their ideological bloc
    entrants_by_bloc = {}
    for e in cands_to_add:
        b = get_party_bloc(e, party_df)
        entrants_by_bloc.setdefault(b, []).append(e)
        
    for b, entrants in entrants_by_bloc.items():
        k_B = len(entrants)
        
        # All minor bloc parties present in the LC baseline - NOTE: GRN is excluded from bloc capture to avoid polluting the 'minor' party average
        minor_bloc_parties = [p for p in mod_lc.index  if get_party_bloc(p, party_df) == b and is_minor_bloc_party(p, party_df) and mod_lc[p] > 1e-9]
        N_B = len(minor_bloc_parties)
        
        if N_B == 0:
            for e in entrants: mod_lc[e] = 0.0
            continue
            
        # Identify total bloc LC mass (Sigma) and the Orphan pool (Omega)
        Sigma_B = sum(mod_lc[p] for p in minor_bloc_parties)
        
        uncontested_minors = [p for p in minor_bloc_parties if current_c0.get(p, 0.0) <= 1e-9]
        Omega_B = sum(mod_lc[p] for p in uncontested_minors)
        
        if Omega_B <= 1e-9:
            for e in entrants: mod_lc[e] = 0.0
            continue

            
        # target entrant LC vote is just under average of bloc share
        target_lc_e = Sigma_B / (k_B + N_B)
        total_target_mass = k_B * target_lc_e
        
        # Cap extraction at the physical orphan limit
        total_extracted = min(total_target_mass, Omega_B)
        lc_e = total_extracted / k_B  # The actual baseline allocated per entrant
        
        # Deflate the orphan pool uniformly to preserve the simplex sum (1.0)
        deflation_factor = max(0.0, (Omega_B - total_extracted) / Omega_B)
        for p in uncontested_minors:
            mod_lc[p] = mod_lc[p] * deflation_factor
            
        # 5. Insert Entrants
        for e in entrants:
            mod_lc[e] = mod_lc.get(e, 0.0) + lc_e
            
    return mod_lc

def _build_state_vars(active_cands: List[str], mod_lc: pd.Series, emp_c0: Dict[str, float], party_df: pd.DataFrame):
    """Extracted utility helper for clean reuse."""
    lc_mass = np.zeros(5)
    for party, share in mod_lc.items():
        b_idx = BLOCS.index(get_party_bloc(party, party_df))
        lc_mass[b_idx] += share
    if lc_mass.sum() > 0:
        lc_mass = lc_mass / lc_mass.sum()

    menu_size, c_cap = np.zeros(5), np.zeros(5)
    cand_to_bloc, cand_demand = {}, {}

    for cand in active_cands:
        b = BLOCS.index(get_party_bloc(cand, party_df))
        cand_to_bloc[cand] = b
        menu_size[b] += 1.0
        
        # Coalition alias handling
        if b in [0, 1]:
            u = mod_lc.get(cand, 0.0)
            if u <= 1e-9:
                u = emp_c0.get(cand, 0.0)
                if u <= 1e-9 and cand in ['LP', 'NP', 'LNP']:
                    coal_lc = sum(mod_lc.get(a, 0.0) for a in ['LP', 'NP', 'LNP'])
                    coal_la = [c for c in active_cands if c in ['LP', 'NP', 'LNP']]
                    if len(coal_la) == 1:
                        u = coal_lc
                    elif len(coal_la) > 1:
                        emp_tot = sum(emp_c0.get(c, 0.0) for c in coal_la)
                        u = coal_lc * (emp_c0.get(cand, 0.0) / emp_tot) if emp_tot > 0 else (coal_lc / len(coal_la))
        else:
            u = mod_lc.get(cand, 0.0)

        cand_demand[cand] = u
        if b in [2, 3, 4]:
            c_cap[b] += u

    c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
    c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
    for b in [2, 3, 4]:
        c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0

    return menu_size, c_cap, cand_to_bloc, cand_demand, lc_mass



def inject_unified_parties_to_dop(
    district: str,
    cands_to_add: List[str],
    current_c0: Dict[str, float],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    calibration_scalars: Dict[str, float] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Unified structural injection for both historical AND new parties simultaneously.
    """

    if calibration_scalars is None:
        calibration_scalars = {c: 1.0 for c in cands_to_add}

    lc_series = lc_df[district].dropna().copy()
    if lc_series.sum() > 1.5:
        lc_series = lc_series / 100.0

    # 1. Provide an LC baseline for Synthetic entrants via Orphan Capture
    zero_lc_entrants = [c for c in cands_to_add if lc_series.get(c, 0.0) <= 1e-9]
    if zero_lc_entrants:
        mod_lc = apply_LC_orphan_capture(zero_lc_entrants, current_c0, lc_series, party_df)
    else:
        mod_lc = lc_series.copy()

    # 2. Define competing Menus
    menu_A = [c for c, v in current_c0.items() if v > 1e-9 and c not in cands_to_add and not str(c).startswith('IND')] # ensures that INDs are not added in for static_high_ind_districts
    menu_B = menu_A + cands_to_add 

    alpha, beta, gamma = model.unpack_params(model.theta)
    
    # State A (Original structure without any entrants)
    ms_A, cap_A, ctb_A, cd_A, lc_mass_A = _build_state_vars(menu_A, lc_series, current_c0, party_df)
    s_hat_A = model.evaluate_state_allocation(lc_mass_A, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)

    # State B (Simultaneous structure WITH ALL entrants)
    ms_B, cap_B, ctb_B, cd_B, lc_mass_B = _build_state_vars(menu_B, mod_lc, current_c0, party_df)
    s_hat_B = model.evaluate_state_allocation(lc_mass_B, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)

    # ---------------------------------------------------------
    # STEP 3: Inter-Bloc MRR (Macro Shift)
    # ---------------------------------------------------------
    emp_bloc_mass, s_hat_A_bloc, s_hat_B_bloc_inc, s_hat_B_bloc_ent = [{b: 0.0 for b in BLOCS} for _ in range(4)]

    for c in menu_A:
        b = get_party_bloc(c, party_df)
        emp_bloc_mass[b] += current_c0.get(c, 0.0)
        s_hat_A_bloc[b] += s_hat_A.get(c, 0.0)
        s_hat_B_bloc_inc[b] += s_hat_B.get(c, 0.0)

    for e in cands_to_add:
        b = get_party_bloc(e, party_df)
        kappa = calibration_scalars.get(e, 1.0)
        s_hat_B_bloc_ent[b] += s_hat_B.get(e, 0.0) * kappa

    s_hat_B_bloc_total = {b: s_hat_B_bloc_inc[b] + s_hat_B_bloc_ent[b] for b in BLOCS}
    
    unnorm_bloc_vol = {}
    for b in BLOCS:
        if emp_bloc_mass[b] > 1e-9 and s_hat_A_bloc[b] > 1e-9:
            macro_retention = s_hat_B_bloc_total[b] / s_hat_A_bloc[b]
            unnorm_bloc_vol[b] = emp_bloc_mass[b] * macro_retention
        else:
            unnorm_bloc_vol[b] = s_hat_B_bloc_ent[b]

    z_bloc = sum(unnorm_bloc_vol.values())
    norm_bloc_vol = {b: v / z_bloc for b, v in unnorm_bloc_vol.items()}

    # ---------------------------------------------------------
    # STEP 4: Entrant Extraction & Empirical Residual Allocation
    # ---------------------------------------------------------
    new_c0 = {}
    for b in BLOCS:
        b_cands_inc = [c for c in menu_A if get_party_bloc(c, party_df) == b]
        b_cands_ent = [c for c in cands_to_add if get_party_bloc(c, party_df) == b]

        v_star = norm_bloc_vol[b]
        b_total_B = s_hat_B_bloc_total[b]

        extracted_entrant_vol = 0.0
        for e in b_cands_ent:
            kappa = calibration_scalars.get(e, 1.0)
            calibrated_s_b = s_hat_B.get(e, 0.0) * kappa
            entrant_share = v_star * (calibrated_s_b / b_total_B) if b_total_B > 1e-9 else 0.0
            new_c0[e] = entrant_share
            extracted_entrant_vol += entrant_share

        v_residual = max(0.0, v_star - extracted_entrant_vol)
        emp_inc_total = emp_bloc_mass[b]

        for c in b_cands_inc:
            cand_intra_share = current_c0[c] / emp_inc_total if emp_inc_total > 1e-9 else (1.0 / len(b_cands_inc) if b_cands_inc else 0.0)
            new_c0[c] = v_residual * cand_intra_share

    # Ensure no incumbent mysteriously grows due to precision drift
    for c in menu_A:
        if new_c0[c] > current_c0[c]: 
            new_c0[c] = current_c0[c]

    z_final = sum(new_c0.values())
    new_c0 = {c: v / z_final for c, v in new_c0.items()}
    shift_vector = {c: new_c0.get(c, 0.0) - current_c0.get(c, 0.0) for c in menu_B}

    return new_c0, shift_vector, s_hat_A, s_hat_B


def precompute_step3_cache(
    det_latent_tables: Dict[tuple, Any],
    next_contesting_parties: Dict[tuple, List[str]],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any
) -> Dict[tuple, Dict[str, Any]]:
    """
    Runs ONCE before the Monte Carlo loop. 
    Precomputes deterministic State A and State B structural allocations.
    """
    alpha, beta, gamma = model.unpack_params(model.theta)
    step3_cache = {}

    for key, table in det_latent_tables.items():
        div = key[0]
        det_c0_party = {k: v for k, v in table.count_0.items() if not str(k).startswith('IND')}

        target_cands = next_contesting_parties.get(key, [])
        has_existing_coal = any(k in det_c0_party for k in ('LP', 'NP', 'LNP', 'COAL')) 
        excluded_majors = {'ALP', 'LP', 'NP', 'LNP'} if has_existing_coal else {'ALP'} 
        missing_cands = [
            c for c in target_cands
            if c not in det_c0_party
            and c not in excluded_majors 
            and not str(c).startswith('IND')
        ]

        lc_series = lc_df[div].dropna().copy()
        if lc_series.sum() > 1.5:
            lc_series = lc_series / 100.0

        zero_lc_entrants = [c for c in missing_cands if lc_series.get(c, 0.0) <= 1e-9]
        if zero_lc_entrants:
            mod_lc = apply_LC_orphan_capture(zero_lc_entrants, det_c0_party, lc_series, party_df)
        else:
            mod_lc = lc_series.copy()

        menu_A = [c for c, v in det_c0_party.items() if v > 1e-9 and c not in missing_cands and not str(c).startswith('IND')]
        menu_B = menu_A + missing_cands

        # Evaluate State A and State B once
        ms_A, cap_A, ctb_A, cd_A, lc_mass_A = _build_state_vars(menu_A, lc_series, det_c0_party, party_df)
        s_hat_A = model.evaluate_state_allocation(lc_mass_A, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)

        ms_B, cap_B, ctb_B, cd_B, lc_mass_B = _build_state_vars(menu_B, mod_lc, det_c0_party, party_df)
        s_hat_B = model.evaluate_state_allocation(lc_mass_B, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)

        # Pre-map candidate blocs
        cand_bloc = {c: get_party_bloc(c, party_df) for c in menu_B}
        s_hat_A_bloc = {b: 0.0 for b in BLOCS}
        s_hat_B_bloc_inc = {b: 0.0 for b in BLOCS}
        for c in menu_A:
            b = cand_bloc[c]
            s_hat_A_bloc[b] += s_hat_A.get(c, 0.0)
            s_hat_B_bloc_inc[b] += s_hat_B.get(c, 0.0)

        b_cands_inc = {b: [c for c in menu_A if cand_bloc[c] == b] for b in BLOCS}
        b_cands_ent = {b: [c for c in missing_cands if cand_bloc[c] == b] for b in BLOCS}

        step3_cache[key] = {
            'div': div,
            'menu_A': menu_A,
            'missing_cands': missing_cands,
            's_hat_A': s_hat_A,
            's_hat_B': s_hat_B,
            's_hat_A_bloc': s_hat_A_bloc,
            's_hat_B_bloc_inc': s_hat_B_bloc_inc,
            'cand_bloc': cand_bloc,
            'b_cands_inc': b_cands_inc,
            'b_cands_ent': b_cands_ent,
        }

    return step3_cache

def anchor_cached_parties_to_dop(
    cache_entry: Dict[str, Any],
    current_c0: Dict[str, float],
    calibration_scalars: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Fast in-loop execution. Anchors precomputed s_hat_A and s_hat_B 
    to the simulation's live empirical current_c0.
    """
    menu_A = cache_entry['menu_A']
    cands_to_add = cache_entry['missing_cands']
    if not cands_to_add:
        return current_c0.copy()

    s_hat_B = cache_entry['s_hat_B']
    s_hat_A_bloc = cache_entry['s_hat_A_bloc']
    s_hat_B_bloc_inc = cache_entry['s_hat_B_bloc_inc']
    cand_bloc = cache_entry['cand_bloc']

    if calibration_scalars is None:
        calibration_scalars = {c: 1.0 for c in cands_to_add}

    # STEP 3: Inter-Bloc MRR (Macro Shift)
    emp_bloc_mass = {b: 0.0 for b in BLOCS}
    s_hat_B_bloc_ent = {b: 0.0 for b in BLOCS}

    for c in menu_A:
        emp_bloc_mass[cand_bloc[c]] += current_c0.get(c, 0.0)

    for e in cands_to_add:
        s_hat_B_bloc_ent[cand_bloc[e]] += s_hat_B.get(e, 0.0) * calibration_scalars.get(e, 1.0)

    s_hat_B_bloc_total = {b: s_hat_B_bloc_inc[b] + s_hat_B_bloc_ent[b] for b in BLOCS}

    unnorm_bloc_vol = {}
    for b in BLOCS:
        if emp_bloc_mass[b] > 1e-9 and s_hat_A_bloc[b] > 1e-9:
            unnorm_bloc_vol[b] = emp_bloc_mass[b] * (s_hat_B_bloc_total[b] / s_hat_A_bloc[b])
        else:
            unnorm_bloc_vol[b] = s_hat_B_bloc_ent[b]

    z_bloc = sum(unnorm_bloc_vol.values())
    norm_bloc_vol = {b: v / z_bloc for b, v in unnorm_bloc_vol.items()}

    # STEP 4: Entrant Extraction & Empirical Residual Allocation
    new_c0 = {}
    for b in BLOCS:
        b_cands_inc = cache_entry['b_cands_inc'][b]
        b_cands_ent = cache_entry['b_cands_ent'][b]
        v_star = norm_bloc_vol[b]
        b_total_B = s_hat_B_bloc_total[b]

        extracted_entrant_vol = 0.0
        for e in b_cands_ent:
            calibrated_s_b = s_hat_B.get(e, 0.0) * calibration_scalars.get(e, 1.0)
            entrant_share = v_star * (calibrated_s_b / b_total_B) if b_total_B > 1e-9 else 0.0
            new_c0[e] = entrant_share
            extracted_entrant_vol += entrant_share

        v_residual = max(0.0, v_star - extracted_entrant_vol)
        emp_inc_total = emp_bloc_mass[b]

        for c in b_cands_inc:
            cand_intra_share = current_c0[c] / emp_inc_total if emp_inc_total > 1e-9 else (1.0 / len(b_cands_inc) if b_cands_inc else 0.0)
            new_c0[c] = v_residual * cand_intra_share

    for c in menu_A:
        if new_c0[c] > current_c0[c]:
            new_c0[c] = current_c0[c]

    z_final = sum(new_c0.values())
    return {c: v / z_final for c, v in new_c0.items()}

def execute_new_minor_injection_intensive( # REPLACED IN MAIN LOOP - STILL USED BY MODEL_TEALS_WEST FOR CONVENIENCE TO AVOID REFACTORING
    latent_tables: Dict[tuple, Any],
    next_contesting_parties: Dict[tuple, List[str]],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    sim_kappas: Dict[str, float],
    static_high_ind_districts: Set[str],
    rng: np.random.Generator,
    anomalous_districts: List = [],
    simulate: bool = True
) -> Tuple[Dict[tuple, Dict[str, float]], Dict[tuple, Dict[str, float]]]:
    """
    Executes Step 3: Simultaneously injects all missing parties into the structural logit. Returns the newly generated counterfactual Count 0 dictionary per electorate. 
    For static_high_ind_districts/anomalous_districts, it uses the model directly for the counterfactual, injecting extra uncertainty
    """
    cf_c0_dict = {}
    s_hat_a_dict = {}  # Store for Step 4
    
    for key, table in latent_tables.items():
        div = key[0]
        emp_c0_full = table.count_0

        # Isolate Party Sub-Space (Strip INDs so model evaluates pure LC party reality)
        emp_c0_party = {k: v for k, v in emp_c0_full.items() if not str(k).startswith('IND')}
        
        target_cands = next_contesting_parties.get(key, [])
        has_existing_coal = any(k in emp_c0_party for k in ('LP', 'NP', 'LNP', 'COAL')) 
        excluded_majors = {'ALP', 'LP', 'NP', 'LNP'} if has_existing_coal else {'ALP'} 
        missing_cands = [
            c for c in target_cands
            if c not in emp_c0_party
            and c not in excluded_majors # NEW
            and not str(c).startswith('IND')
        ] #         Identify who is missing (exclude Majors/Greens and INDs, as they are handled elsewhere)


        new_c0, shift_vec, s_hat_a, s_hat_b = inject_unified_parties_to_dop(district=div, cands_to_add=missing_cands, current_c0=emp_c0_party, lc_df=lc_df, party_df=party_df, model=model, calibration_scalars=sim_kappas)
        s_hat_a_dict[key] = s_hat_a  # IND-free structural baseline for Step 4





        # Direct empirical anchor for delta does not exist -> use model directly; punish with uncertainty
        if missing_cands and (div in static_high_ind_districts or div in anomalous_districts):
            # model's output as 'truth
            override_c0 = copy.deepcopy(s_hat_b)
            
            # calibrate entrants by kappa (with localized uncertainty penalty)
            for e in missing_cands:
                if e in ('LP', 'NP', 'GRN', 'ALP'): # NEW
                    continue # Richmond 2018, keep pure LC model share s_hat_B[e] for major party entrants
                noise_multiplier = max(0.1, rng.normal(1.0, 0.15)) if simulate else 1.0
                penalised_kappa = sim_kappas.get(e, 1.0) * noise_multiplier
                override_c0[e] = s_hat_b.get(e, 0.0) * penalised_kappa
                
            # renormalise (not particularly neat, but simple)
            z = sum(override_c0.values())
            cf_c0_dict[key] = {k: v / z for k, v in override_c0.items()}
            
        else:
            # Standard MRR (Delta) Regime; produces replica of emp_c0_party if not missing_cands
            cf_c0_dict[key] = new_c0
        
        
    return cf_c0_dict, s_hat_a_dict

def execute_new_minor_injection(
    latent_tables: Dict[tuple, Any],
    next_contesting_parties: Dict[tuple, List[str]],
    step3_cache: Dict[tuple, Dict[str, Any]],     # <--- NEW ARGUMENT
    sim_kappas: Dict[str, float],
    static_high_ind_districts: Set[str],
    rng: np.random.Generator,
    anomalous_districts: List = None,
    simulate: bool = True
) -> Tuple[Dict[tuple, Dict[str, float]], Dict[tuple, Dict[str, float]]]:
    """
    Executes Step 3: Instantly anchors precomputed structural states to the live simulation's dictionary.
    """
    if anomalous_districts is None:
        anomalous_districts = []

    cf_c0_dict = {}
    s_hat_a_dict = {}  
    
    for key, table in latent_tables.items():
        div = key[0]
        emp_c0_full = table.count_0
        emp_c0_party = {k: v for k, v in emp_c0_full.items() if not str(k).startswith('IND')}
        
        # Grab the precomputed data for this specific district
        cache_entry = step3_cache[key]
        missing_cands = cache_entry['missing_cands']
        
        # s_hat_A is used strictly by Step 4 later, pass it through directly
        s_hat_a_dict[key] = cache_entry['s_hat_A']

        if missing_cands and (div in static_high_ind_districts or div in anomalous_districts):
            # Anomalous logic: copy the deterministic s_hat_B and penalize it with noise
            override_c0 = cache_entry['s_hat_B'].copy()
            
            for e in missing_cands:
                noise_multiplier = max(0.1, rng.normal(1.0, 0.15)) if simulate else 1.0
                penalised_kappa = sim_kappas.get(e, 1.0) * noise_multiplier
                override_c0[e] = override_c0.get(e, 0.0) * penalised_kappa
                
            z = sum(override_c0.values())
            cf_c0_dict[key] = {k: v / z for k, v in override_c0.items()}
            
        else:
            # Standard MRR logic: Instantly anchor the cache to the live emp_c0_party
            cf_c0_dict[key] = anchor_cached_parties_to_dop(
                cache_entry=cache_entry,
                current_c0=emp_c0_party,
                calibration_scalars=sim_kappas
            )
        
    return cf_c0_dict, s_hat_a_dict



latent_tables = execute_party_removal(
    stripped_tables=stripped_tables,
    static_high_ind_districts=static_high_ind_districts,
    next_contesting_parties=old_contesting_parties,
    LC_FPs_df=LC_FPs_df,
    party_df=party_df,             
    dop_lookup=dop_lookup,                  
    electorate_to_region=electorate_to_region, 
    model = final_model,
    simulate=True
)

print(f"Step 2 Complete: {len(latent_tables)} latent tables generated.")

#import pdb; pdb.set_trace()

# Define synthetic targets with priors AND pseudo-sample sizes (pseudo_N); should be higher than typical as represents the only shrinkage available (heuristic: add 10 to instinctual pseudo-N)!
# High pseudo_N (e.g., 20) = Trusts the prior, weak shrinkage to 1.0
# Low pseudo_N (e.g., 3) = Distrusts the prior, heavy shrinkage to 1.0
NEW_PARTY_PRIORS = {}

NEW_PARTY_PRIORS[2022] = {
    'SAL': {'prior': 0.02, 'pseudo_N': 6+10}, 
    'CEC': {'prior': 0.0103, 'pseudo_N': 19+10},
    'AUD': {'prior': 0.0067, 'pseudo_N': 2+10}, # 2022 and 2025 federal result in NSW
    'MAGA': {'prior': 0.005, 'pseudo_N': 10+10},# 10 added to push parties down despite no results
    'FPAL': {'prior': 0.005, 'pseudo_N': 10+10},
    'ATF': {'prior': 0.005, 'pseudo_N': 10+10},
    'CAN': {'prior': 0.005, 'pseudo_N': 10+10},
}

NEW_PARTY_PRIORS[2018] = {
    'ON': {'prior': 0.039, 'pseudo_N': 38+10}, # upwards adjusted for Higgins
    'FFP': {'prior': 0.024, 'pseudo_N': 40+10}, # 2014 state election
    'HMP': {'prior': 0.03, 'pseudo_N': 10+10}, # 2022 senate in Victoria
    'AVP': {'prior': 0.005, 'pseudo_N': 10+10},
    'CAP': {'prior': 0.005, 'pseudo_N': 10+10},
    'SDA': {'prior': 0.005, 'pseudo_N': 10+10},
    'ND': {'prior': 0.005, 'pseudo_N': 10+10},
} # no prior for FPV given
NEW_PARTY_PRIORS[2014] = {
    'DHJP': {'prior': 0.036, 'pseudo_N': 4+10}, # 2016 federal
    'REAS': {'prior': 0.025, 'pseudo_N': 9+10}, # ASXP 2014
    'ALA': {'prior': 0.03, 'pseudo_N': 9+10}, # 2016 federal
    'SPP': {'prior': 0.007, 'pseudo_N': 1+10}, # 2016 federal
    'VNS': {'prior': 0.022, 'pseudo_N': 2+10}, #  Socialist alliance 2014 VIC (excl. Jolly!)
} # others default
NEW_PARTY_PRIORS[2010] = {
    'AJP': {'prior': 0.011, 'pseudo_N': 2+10}, # 2013 Federal VIC
    'CHR': {'prior': 0.0125, 'pseudo_N': 15+10}, # 2013 Federal VIC
    'CYC': {'prior': 0.01, 'pseudo_N': 0+10}, # vibes
    'RUA': {'prior': 0.007, 'pseudo_N': 31+10}, # 2013 Federal VIC
    'SFF': {'prior': 0.05, 'pseudo_N': 1+10}, # 2011 NSW Toongabbie
    'VEP': {'prior': 0.012, 'pseudo_N': 2+10}, # Lingiari 2013 Federal
}
NEW_PARTY_PRIORS[2006] = {
    'ASXP': {'prior': 0.023, 'pseudo_N': 3+10}, # 2010 Federal VIC
}

# full set of parties that require a kappa i.e. need to be added to at least one electorate
parties_to_add = sorted(list({
    p
    for (old_div, new_div), cands in next_contesting_parties.items()
    if (old_div, old_div) in latent_tables
    for p in cands
    if p not in latent_tables[(old_div, old_div)].count_0 
    and p not in ['ALP', 'LP', 'NP', 'LNP', 'GRN'] and not str(p).startswith('IND') and not str(p)=='TWP'
}))
# OLD: parties_to_add = sorted(list({p for (exist_d, new_d),t in latent_tables.items() if (exist_d, new_d) in next_contesting_parties for p in next_contesting_parties[d] if p not in t.count_0 and p not in ['ALP','LP','NP','LNP','GRN'] and not str(p).startswith('IND')}))
print(f"Parties identified for unified injection: {parties_to_add}")

print("Pre-calibrating structural realization scalars (Kappas)...")
calibration_data = pre_calibrate_all_kappas(
    parties_to_add=parties_to_add,
    latent_tables=latent_tables, 
    lc_df=LC_FPs_df, 
    party_df=party_df, 
    model=final_model, 
    synthetic_priors=NEW_PARTY_PRIORS[data_year]
)



print("Step 3 Complete. Counterfactual baselines generated.")


#################################################### Step 4: Return INDs to Count 0 via defection rates #########################################################################################

def create_augmented_c0_for_defection(stripped_c0: dict, target_cands: list) -> dict:
    """
    Takes the 2022 stripped Count 0 and removes ONLY non-contesting parties (excluding INDs, which are rigidly preserved).Reallocates retiring parties proportionally to isolate IND gravity.
    """
    c0 = stripped_c0.copy()
    
    # 1. Identify who to remove: Not in 2026 AND not an IND
    to_remove = []
    for cand in c0.keys():
        party = cand.split('_')[0] if '_' in cand else cand
        is_contesting = (cand in target_cands) or (party in target_cands)
        
        if not is_contesting and not str(cand).startswith('IND'):
            to_remove.append(cand)
            
    # 2. Pool their votes
    realloc_pool = sum(c0[c] for c in to_remove)
    if realloc_pool <= 1e-9:
        return c0
        
    # 3. Reallocate proportionally to surviving mass (INCLUDING the INDs)
    survivors = {c: v for c, v in c0.items() if c not in to_remove}
    surv_mass = sum(survivors.values())
    
    if surv_mass > 1e-9:
        for c in survivors:
            survivors[c] += realloc_pool * (survivors[c] / surv_mass)
            
    return survivors

def build_and_evaluate_model_allocation(
    district: str,
    menu: List[str],
    lc_series: pd.Series,
    party_df: pd.DataFrame,
    model: Any,
    fallback_c0: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Evaluates model.evaluate_state_allocation for an arbitrary menu in a district.
    Handles Upper House aggregation and Coalition alias fallback logic.
    """
    lc_mass = np.zeros(5)
    for party, share in lc_series.items():
        b_idx = BLOCS.index(get_party_bloc(party, party_df))
        lc_mass[b_idx] += share
    if lc_mass.sum() > 0:
        lc_mass = lc_mass / lc_mass.sum()

    if np.isnan(lc_mass).any():
        import pdb; pdb.set_trace()
    menu_size = np.zeros(5)
    c_cap = np.zeros(5)
    cand_to_bloc = {}
    cand_demand = {}

    for cand in menu:
        b = BLOCS.index(get_party_bloc(cand, party_df))
        cand_to_bloc[cand] = b
        menu_size[b] += 1.0

        # 1. Primary lookup directly in LC series
        u = lc_series.get(cand, 0.0)

        # 2. Fallback resolution if candidate has 0 direct LC vote
        if u <= 1e-9:
            # Check fallback_c0 if provided
            if fallback_c0 is not None and cand in fallback_c0:
                u = fallback_c0.get(cand, 0.0)

            # Coalition resolution (LP / NP / LNP) from joint 'COAL' ticket
            if u <= 1e-9 and cand in ['LP', 'NP', 'LNP']:
                coal_lc = sum(lc_series.get(a, 0.0) for a in ['LP', 'NP', 'LNP', 'COAL'])
                coal_la = [c for c in menu if c in ['LP', 'NP', 'LNP']]
                
                if len(coal_la) == 1:
                    u = coal_lc
                elif len(coal_la) > 1:
                    if fallback_c0 is not None:
                        emp_c = fallback_c0.get(cand, 0.0)
                        emp_tot = sum(fallback_c0.get(c, 0.0) for c in coal_la)
                        u = coal_lc * (emp_c / emp_tot) if emp_tot > 0 else (coal_lc / len(coal_la))
                    else:
                        # Without fallback_c0: equal split across contested Coalition parties
                        u = coal_lc / len(coal_la)

        # 3. Guard against log(0) NaNs in evaluate_state_allocation
        u = max(u, 1e-4)

        cand_demand[cand] = u
        if b in [2, 3, 4]:
            c_cap[b] += u

    c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
    c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
    for b in [2, 3, 4]:
        c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0

    alpha, beta, gamma = model.unpack_params(model.theta)

    return model.evaluate_state_allocation(
        lc_mass, menu, c_cap, menu_size, cand_to_bloc, cand_demand,
        alpha, beta, gamma
    )


def compute_independent_defection_rates(
    empirical_c0_full: Dict[str, float],
    s_hat_a: Dict[str, float]
) -> Tuple[Dict[str, float], float]:
    """
    Calculates lambda_c (the proportion of incumbent c's structural vote 
    that defected to the Independent).
    """
    ind_cands = [c for c, v in empirical_c0_full.items() if str(c).startswith('IND') and v > 1e-9]
    tot_ind_emp = sum(empirical_c0_full[c] for c in ind_cands)
    menu_A = [c for c in empirical_c0_full if c not in ind_cands and empirical_c0_full[c] > 1e-9]

    if tot_ind_emp <= 1e-9:
        return {c: 0.0 for c in menu_A}, 0.0

    raw_stolen = {}
    for c in menu_A:
        pred_a = s_hat_a.get(c, 0.0)
        emp_c = empirical_c0_full.get(c, 0.0)
        raw_stolen[c] = max(0.0, pred_a - emp_c)

    tot_raw_stolen = sum(raw_stolen.values())
    calibrated_stolen = {}

    if tot_raw_stolen > 1e-9:
        scale_factor = tot_ind_emp / tot_raw_stolen
        for c in menu_A:
            calibrated_stolen[c] = min(s_hat_a.get(c, 0.0), raw_stolen[c] * scale_factor)
    else:
        for c in menu_A:
            calibrated_stolen[c] = tot_ind_emp * s_hat_a.get(c, 0.0)

    lambdas = {}
    for c in menu_A:
        pred = s_hat_a.get(c, 0.0)
        lam = calibrated_stolen[c] / pred if pred > 1e-9 else 0.0
        lambdas[c] = min(0.9999, max(0.0, lam))

    return lambdas, tot_ind_emp

def execute_IND_restoration_via_defections(
    district_pair: str,
    latent_c0: Dict[str, float],      # Contains ALL INDs for High-IND seats
    cf_c0_party: Dict[str, float],    # Step 3 Party Counterfactual
    s_hat_a: Dict[str, float],        # Step 3 Pure Party Baseline
    cands_added_in_step_3: List[str], 
    next_contesting_parties: Dict[tuple, List[str]], # Needed for Divergent Futures
    lc_series: pd.Series,
    party_df: pd.DataFrame,
    model: Any,
    global_party_dict: Dict[str, str] = None
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Executes Step 4: Dynamically restores re-contesting Independents.
    Handles divergent futures (e.g., a massive IND retires but a small IND stays)
    by scaling the defection rates proportionally to the surviving IND mass.
    """
    district = district_pair[0]
    # 1. Identify ALL historical INDs residing in the latent baseline
    historical_inds = [c for c in latent_c0 if str(c).startswith('IND') and latent_c0[c] > 1e-9]
    extracted_historical_lambdas = {}
    if not historical_inds:
        return copy.deepcopy(cf_c0_party), extracted_historical_lambdas

    # 2. Check Divergent Futures: Who is actually running in 2026?
    target_cands = next_contesting_parties.get(district_pair, [])
    recontesting_inds = [c for c in historical_inds if c in target_cands]

    # 3. Compute Historical Defection (using the FULL historical IND mass)
    lambdas, tot_ind_emp = compute_independent_defection_rates(latent_c0, s_hat_a)
    all_lambdas = dict(lambdas)
    
    tot_menu_a_mass = sum(s_hat_a.values())
    menu_A = list(s_hat_a.keys())

    # 4. Entrant Defection Rates (Scale-Dampened Affinity)
    for entrant in cands_added_in_step_3:
        s_hat_b_single = build_and_evaluate_model_allocation(
            district=district, menu=menu_A + [entrant], lc_series=lc_series, 
            party_df=party_df, model=model, fallback_c0=latent_c0
        )

        tot_delta = sum(max(0.0, s_hat_a.get(c, 0.0) - s_hat_b_single.get(c, 0.0)) for c in lambdas)
        bar_tau = tot_delta / tot_menu_a_mass if tot_menu_a_mass > 1e-9 else 1.0
        bar_lam = tot_ind_emp / tot_menu_a_mass if tot_menu_a_mass > 1e-9 else 1.0

        scale_tau = bar_tau ** 0.5
        scale_lam = bar_lam ** 0.5

        g_stream = {}
        gamma_stream = {}

        for c, lam_c in lambdas.items():
            pred_a = s_hat_a.get(c, 0.0)
            delta_c = max(0.0, pred_a - s_hat_b_single.get(c, 0.0))
            tau_c = min(0.9999, max(0.0, delta_c / pred_a)) if pred_a > 1e-9 else 0.0

            eff_odds_tau = (tau_c / scale_tau) if scale_tau > 1e-9 else 0.0
            eff_odds_lam = (lam_c / scale_lam) if scale_lam > 1e-9 else 0.0

            gamma_c = (eff_odds_tau / (eff_odds_tau + eff_odds_lam)) if (eff_odds_tau + eff_odds_lam) > 1e-9 else 0.5
            gamma_stream[c] = gamma_c
            g_stream[c] = delta_c * lam_c

        tot_g = sum(g_stream.values())
        ent_mass = cf_c0_party.get(entrant, 0.0)

        if tot_g > 1e-9 and ent_mass > 1e-9:
            gamma_bar = sum((g / tot_g) * gamma_stream[c] for c, g in g_stream.items())
            lam_ent = (tot_g / ent_mass) * (1.0 - gamma_bar)
            all_lambdas[entrant] = min(0.9999, max(0.0, lam_ent))
        else:
            all_lambdas[entrant] = 0.0

    # =====================================================================
    # 5. THE DIVERGENT FUTURES SCALAR
    # =====================================================================
    if recontesting_inds:
        recontesting_mass = sum(latent_c0[c] for c in recontesting_inds)
        retention_ratio = recontesting_mass / tot_ind_emp if tot_ind_emp > 1e-9 else 1.0
    else:
        # All INDs retired. Defection drops to zero.
        retention_ratio = 0.0

    for p in all_lambdas:
        all_lambdas[p] *= retention_ratio

    # 6. Universal Simplex Reintegration
    final_c0 = {}
    total_restored_ind_mass = 0.0

    # Deflate Step 3 party simplex based on the *scaled* defection rates
    for p, v in cf_c0_party.items():
        retention = 1.0 - all_lambdas.get(p, 0.0)
        final_c0[p] = v * retention
        total_restored_ind_mass += v * all_lambdas.get(p, 0.0)

    # Re-allocate stolen votes exclusively to the re-contesting Independents
    if recontesting_inds:
        for ind in recontesting_inds:
            prop = latent_c0[ind] / recontesting_mass if recontesting_mass > 0 else (1.0 / len(recontesting_inds))
            final_c0[ind] = total_restored_ind_mass * prop

    z_final = sum(final_c0.values())


    # =====================================================================
    # Captures macro-bloc defection profile for ALL historical INDs with electorate mass >= 5%
    # =====================================================================
    if global_party_dict is not None and tot_ind_emp >= 0.05:
        BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
        
        bloc_pure = {b: 0.0 for b in BLOCS}
        bloc_stolen = {b: 0.0 for b in BLOCS}
        
        # Aggregate the raw candidate-level stolen votes up to Macro-Blocs
        for c, lam_val in lambdas.items():
            b = global_party_dict.get(c, 'Centre')
            b = b if b in BLOCS else 'Centre'
            
            pure_base = s_hat_a.get(c, 0.0)
            stolen_vol = pure_base * lam_val
            
            bloc_pure[b] += pure_base
            bloc_stolen[b] += stolen_vol
            
        # Calculate the final macro-bloc defection rates
        macro_lambda_profile = {}
        for b in BLOCS:
            if bloc_pure[b] > 1e-9:
                macro_lambda_profile[b] = min(0.9999, max(0.0, bloc_stolen[b] / bloc_pure[b]))
            else:
                macro_lambda_profile[b] = 0.0
                
        # Save this exact profile for every historical IND in this seat, only if there is enough signal here
        if sum(1 for b in BLOCS if bloc_pure.get(b, 0.0) > 1e-9 and min(0.9999, max(0.0, bloc_stolen.get(b, 0.0) / bloc_pure[b])) > 1e-6) >= 3:
            for ind in historical_inds:
                extracted_historical_lambdas[(district, ind)] = macro_lambda_profile

    # =====================================================================

    return {k: v / z_final for k, v in final_c0.items()}, extracted_historical_lambdas

def precompute_step4_entrant_cache(
    det_latent_tables: Dict[tuple, Any],
    global_step3_cache: Dict[tuple, Dict[str, Any]],
    next_contesting_parties: Dict[tuple, List[str]],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    data_year: int
) -> Dict[tuple, Dict[str, Any]]:
    """
    Runs ONCE prior to the Monte Carlo loop.
    Precomputes ONLY the deterministic model shift (delta_c and eff_odds_tau) 
    for each Step 3 entrant, leaving all latent_c0 math live for the simulation.
    """
    step4_cache = {}

    for key, table in det_latent_tables.items():
        old_div, new_div = key
        det_c0 = table.count_0

        # Special cache for Narracan 2022 deterministic LC allocation
        if old_div == 'Narracan' and data_year == 2022:
            lc_series = lc_df[old_div].dropna()
            lc_series = lc_series / 100.0 if lc_series.sum() > 1.5 else lc_series
            narracan_cands = [c for c in next_contesting_parties[key] if not str(c).startswith('IND')]
            narracan_c0 = build_and_evaluate_model_allocation(
                district='Narracan', menu=narracan_cands, lc_series=lc_series,
                party_df=party_df, model=model, fallback_c0=None
            )
            tot_n = sum(narracan_c0.values())
            step4_cache[key] = {
                'narracan_keys': list(narracan_c0.keys()),
                'norm_narracan': [v / tot_n for v in narracan_c0.values()]
            }
            continue

        s_hat_a = global_step3_cache[key]['s_hat_A']
        cands_added = global_step3_cache[key]['missing_cands']
        menu_A = list(s_hat_a.keys())
        tot_menu_a_mass = sum(s_hat_a.values())

        lc_series = lc_df[old_div].dropna()
        lc_series = lc_series / 100.0 if lc_series.sum() > 1.5 else lc_series

        entrant_tau_cache = {}
        for entrant in cands_added:
            # Evaluate single-entrant model once
            s_hat_b_single = build_and_evaluate_model_allocation(
                district=old_div, menu=menu_A + [entrant], lc_series=lc_series,
                party_df=party_df, model=model, fallback_c0=det_c0
            )

            delta_dict = {
                c: max(0.0, s_hat_a.get(c, 0.0) - s_hat_b_single.get(c, 0.0))
                for c in menu_A
            }
            tot_delta = sum(delta_dict.values())
            bar_tau = tot_delta / tot_menu_a_mass if tot_menu_a_mass > 1e-9 else 1.0
            scale_tau = bar_tau ** 0.5

            eff_odds_tau_dict = {}
            for c in menu_A:
                pred_a = s_hat_a.get(c, 0.0)
                tau_c = min(0.9999, max(0.0, delta_dict[c] / pred_a)) if pred_a > 1e-9 else 0.0
                eff_odds_tau_dict[c] = (tau_c / scale_tau) if scale_tau > 1e-9 else 0.0

            entrant_tau_cache[entrant] = {
                'delta_dict': delta_dict,
                'eff_odds_tau_dict': eff_odds_tau_dict
            }

        step4_cache[key] = {
            'tot_menu_a_mass': tot_menu_a_mass,
            'entrant_tau_cache': entrant_tau_cache
        }

    return step4_cache


def execute_IND_restoration_fast_sim(
    district_pair: tuple,
    latent_c0: Dict[str, float],
    cf_c0_party: Dict[str, float],
    s_hat_a: Dict[str, float],
    cands_added_in_step_3: List[str],
    next_contesting_parties: Dict[tuple, List[str]],
    step4_cache_entry: Dict[str, Any]
) -> Dict[str, float]:
    """
    Fast simulation-loop version of Step 4.
    Preserves 100% of live latent_c0 variance while using cached entrant model evaluations.
    """
    # 1. Identify historical INDs and check if any are actually re-contesting
    historical_inds = [c for c in latent_c0 if str(c).startswith('IND') and latent_c0[c] > 1e-9]
    if not historical_inds:
        return cf_c0_party.copy()

    target_cands = next_contesting_parties.get(district_pair, [])
    recontesting_inds = [c for c in historical_inds if c in target_cands]

    # In the simulation loop, if no historical IND is re-contesting, retention_ratio is 0.0,
    # meaning 0 votes are extracted from cf_c0_party. Return immediately!
    if not recontesting_inds:
        return cf_c0_party.copy()

    # 2. Compute LIVE Historical Defection from this simulation's latent_c0
    lambdas, tot_ind_emp = compute_independent_defection_rates(latent_c0, s_hat_a)
    all_lambdas = dict(lambdas)

    tot_menu_a_mass = step4_cache_entry['tot_menu_a_mass']
    bar_lam = tot_ind_emp / tot_menu_a_mass if tot_menu_a_mass > 1e-9 else 1.0
    scale_lam = bar_lam ** 0.5
    entrant_tau_cache = step4_cache_entry['entrant_tau_cache']

    # 3. Compute LIVE Entrant Defection Rates using cached tau & live lambda
    for entrant in cands_added_in_step_3:
        tau_info = entrant_tau_cache[entrant]
        delta_dict = tau_info['delta_dict']
        eff_odds_tau_dict = tau_info['eff_odds_tau_dict']

        g_stream = {}
        gamma_stream = {}

        for c, lam_c in lambdas.items():
            eff_odds_tau = eff_odds_tau_dict.get(c, 0.0)
            eff_odds_lam = (lam_c / scale_lam) if scale_lam > 1e-9 else 0.0

            denom = eff_odds_tau + eff_odds_lam
            gamma_c = (eff_odds_tau / denom) if denom > 1e-9 else 0.5
            gamma_stream[c] = gamma_c
            g_stream[c] = delta_dict.get(c, 0.0) * lam_c

        tot_g = sum(g_stream.values())
        ent_mass = cf_c0_party.get(entrant, 0.0)

        if tot_g > 1e-9 and ent_mass > 1e-9:
            gamma_bar = sum((g / tot_g) * gamma_stream[c] for c, g in g_stream.items())
            lam_ent = (tot_g / ent_mass) * (1.0 - gamma_bar)
            all_lambdas[entrant] = min(0.9999, max(0.0, lam_ent))
        else:
            all_lambdas[entrant] = 0.0

    # 4. LIVE Divergent Futures Scalar & Simplex Reintegration
    recontesting_mass = sum(latent_c0[c] for c in recontesting_inds)
    retention_ratio = recontesting_mass / tot_ind_emp if tot_ind_emp > 1e-9 else 1.0

    for p in all_lambdas:
        all_lambdas[p] *= retention_ratio

    final_c0 = {}
    total_restored_ind_mass = 0.0

    for p, v in cf_c0_party.items():
        lam_p = all_lambdas.get(p, 0.0)
        final_c0[p] = v * (1.0 - lam_p)
        total_restored_ind_mass += v * lam_p

    for ind in recontesting_inds:
        prop = latent_c0[ind] / recontesting_mass if recontesting_mass > 0 else (1.0 / len(recontesting_inds))
        final_c0[ind] = total_restored_ind_mass * prop

    z_final = sum(final_c0.values())
    return {k: v / z_final for k, v in final_c0.items()}


#################################################### Step 5: Add new INDs via pool_model/split_model #########################################################################################


# -----------------------------------------------------------------
# 1. SETUP TARGET ARCHETYPE DISTRICTS & CANDIDATES - Teals and West party
# -----------------------------------------------------------------
def extract_archetype_profile(district_list, target_ind_list, global_party_dict, augmented_c0_dict, s_hat_a_dict):
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    ind_shares = []
    records = []

    for div, target_names in zip(district_list, target_ind_list):
        if (div, div) not in augmented_c0_dict or (div, div) not in s_hat_a_dict:
            continue

        emp_c0 = augmented_c0_dict[(div, div)]
        s_hat = s_hat_a_dict[(div, div)]

        # 1. Target candidate vote volume
        target_ind_vol = sum(emp_c0.get(cand, 0.0) for cand in target_names)
        if target_ind_vol <= 1e-4:
            continue
        ind_shares.append(target_ind_vol)

        # 2. Raw stolen mass: ONLY calculate for parties that existed in the baseline
        raw_stolen = {}
        for cand, latent_vol in s_hat.items():
            emp_vol = emp_c0.get(cand, 0.0)
            if emp_vol > 1e-4:
                raw_stolen[cand] = max(0.0, latent_vol - emp_vol)

        tot_raw_stolen = sum(raw_stolen.values())
        div_stolen_by_bucket = {b: 0.0 for b in BLOCS}

        if tot_raw_stolen > 1e-9:
            scale_factor = target_ind_vol / tot_raw_stolen
            for cand, stolen in raw_stolen.items():
                b = global_party_dict.get(cand, 'Centre')
                b = b if b in BLOCS else 'Centre'
                div_stolen_by_bucket[b] += stolen * scale_factor
        else:
            for cand, latent_vol in s_hat.items():
                b = global_party_dict.get(cand, 'Centre')
                b = b if b in BLOCS else 'Centre'
                div_stolen_by_bucket[b] += latent_vol * target_ind_vol

        # 3. Available pool of each bloc in this district (from s_hat pre-defection baseline)
        div_pool = {b: 0.0 for b in BLOCS}
        for cand, val in s_hat.items():
            b = global_party_dict.get(cand, 'Centre')
            b = b if b in BLOCS else 'Centre'
            div_pool[b] += val

        records.append({
            'district_name': div,
            'target_cand': ', '.join(target_names),
            'ind_primary_vote': target_ind_vol,
            **{f'{b}_points_drained': div_stolen_by_bucket[b] for b in BLOCS},
            **{f'{b}_pool_size': div_pool[b] for b in BLOCS}
        })

    # 4. Final Empirical Defection Rates (lambda_b)
    # This evaluates to ~0.20 for COAL, ~0.03 for Right. It does NOT sum to 1.0.
    lambda_profile = {}
    for b in BLOCS:
        tot_drained = sum(r[f'{b}_points_drained'] for r in records)
        tot_pool = sum(r[f'{b}_pool_size'] for r in records)
        lambda_profile[b] = float(tot_drained / tot_pool) if tot_pool > 1e-4 else 0.0

    mean_vote = float(np.mean(ind_shares)) if ind_shares else 0.15
    std_vote = float(np.std(ind_shares)) if len(ind_shares) > 1 else 0.04

    # Return exactly 4 items to match: teal_profile, teal_records, teal_mean_vote, teal_std_vote
    return lambda_profile, pd.DataFrame(records), mean_vote, std_vote

def model_teals_west(DOP_table_dict, final_model, baseline_theta, inc_df, ia_csv_path, old_contesting_parties, calibration_data, LC_FPs_df, party_df, data_year):

    if data_year == 2022:
        teal_divs = ['Brighton', 'Caulfield', 'Mornington', 'Kew', 'Hawthorn', 'Sandringham', 'Bellarine', 'Benambra']
        teal_inds = [['IND_Felicity_Frederico'], ['IND_Nomi_Kaltmann'], ['IND_Kate_Lardner'], ['IND_Sophie_Torney'], ['IND_Melissa_Lowe'], ['IND_Clarke_Martin'], ['IND_Sarah_Fenton'], ['IND_Jacqui_Hawkins']]
        west_divs = ['Werribee','Point Cook']
        west_inds = [['IND_Paul_Hopper'], ['IND_Joe_Garra']]
    elif data_year == 2018:
        teal_divs = ['Ovens Valley','Euroa','Eildon','Benambra']
        teal_inds = [['IND_Tammy_Atkins'],['IND_Don_Firth'],['IND_Michelle_Dunscombe'],['IND_Jacqui_Hawkins']]
        west_divs = []
        west_inds = []
   

    archetype_divs = [d for d in (teal_divs + west_divs) if d in DOP_table_dict]

    rng_det = np.random.default_rng(seed=42)
    final_model.set_params(baseline_theta)
    archetype_dop = {d: DOP_table_dict[d] for d in archetype_divs}

    # -----------------------------------------------------------------
    # 2. RUN DETERMINISTIC STEPS 1–3 OUTSIDE THE LOOP
    # -----------------------------------------------------------------
    stripped_tables, static_high_inds, _ = execute_strip_incumbency(archetype_dop, inc_df, ia_csv_path)

    stripped_tables = {(div, div): stripped_tables[div] for div in stripped_tables.keys()}

    latent_tables = execute_party_removal(
        stripped_tables, static_high_inds, old_contesting_parties, LC_FPs_df=LC_FPs_df, party_df=party_df, # CHECK - amended to old_contesting_parties
        dop_lookup=dop_lookup, electorate_to_region=electorate_to_region, model = final_model, simulate=False, rng=rng_det
    )

    augmented_c0_dict = {
        (div, div): create_augmented_c0_for_defection(stripped_tables[(div, div)].count_0, old_contesting_parties.get((div, div), []))
        for div in archetype_divs if (div, div) in stripped_tables
    }
    #import pdb; pdb.set_trace() # CHECK correct
    mean_kappas = draw_stochastic_kappas(calibration_data, simulate=False)

    _, s_hat_a_dict = execute_new_minor_injection_intensive(
        latent_tables=latent_tables,
        next_contesting_parties=old_contesting_parties,
        lc_df=LC_FPs_df,
        party_df=party_df,
        model=final_model,
        sim_kappas=mean_kappas,
        static_high_ind_districts=static_high_inds,
        anomalous_districts=[],
        simulate=False,
        rng=rng_det
    )


    # Compute final target-specific profiles and lognormal method of moments fits
    teal_profile ,teal_records, teal_mean_vote, teal_std_vote = extract_archetype_profile(teal_divs, teal_inds, global_party_dict, augmented_c0_dict, s_hat_a_dict)
    west_profile, west_records, west_mean_vote, west_std_vote = extract_archetype_profile(west_divs, west_inds, global_party_dict, augmented_c0_dict, s_hat_a_dict)

    #print(pd.DataFrame(teal_records))
    #print(pd.DataFrame(west_records))

    df_teals = pd.DataFrame(teal_records).set_index('district_name')
    teal_vols = df_teals['ind_primary_vote'].values if not df_teals.empty else np.array([])
    
    if data_year == 2022:
        extra_teal_vols = np.array([16.14, 5.46, 7.23, 19.16, 10.91, 10.28]) / 100.0
        all_teal_vols = np.concatenate([teal_vols, extra_teal_vols])

         # West Volume Arrays
        nq_first_ratios = np.array([4.08, 0.55, 3.03, 2.41]) / 9.37
        other_west_vols = np.array([2.4, 4.9, 1.8, 3.3, 3.1, 0.9, 4.0, 2.4, 3.8, 1.0, 4.4, 3.0, 2.9, 5.5, 7.3, 6.6]) / 100.0 
        all_west_vols = np.append(nq_first_ratios * 10.6 / 100, other_west_vols)
    elif data_year == 2018:
        extra_teal_vols = np.array([10.91, 10.28]) / 100.0
        all_teal_vols = np.concatenate([teal_vols, extra_teal_vols])
        all_west_vols = np.array([])
        teal_profile['Centre'] = 0.25

    if len(all_teal_vols) > 1:
        teal_mean_v = np.mean(all_teal_vols)
        teal_var_v = np.var(all_teal_vols, ddof=1)
        teal_sigma_log = np.sqrt(np.log(1 + (teal_var_v / (teal_mean_v**2))))
        teal_mu_log = np.log(teal_mean_v) - (0.5 * teal_sigma_log**2)
        teal_params = {'mean': teal_mean_v, 'mu_log': teal_mu_log, 'sigma_log': teal_sigma_log, 'profile': teal_profile}
    else:
        # Fallback for years with barely any historical teal data
        import pdb; pdb.set_trace()
        teal_params = {'mean': 0.15, 'mu_log': np.log(0.15), 'sigma_log': 0.1, 'profile': teal_profile}
    
    if len(all_west_vols) > 1:
        west_mean_v = np.mean(all_west_vols)
        west_var_v = np.var(all_west_vols, ddof=1)
        west_sigma_log = np.sqrt(np.log(1 + (west_var_v / (west_mean_v**2))))
        west_mu_log = np.log(west_mean_v) - (0.5 * west_sigma_log**2)
        west_params = {'mean': west_mean_v, 'mu_log': west_mu_log, 'sigma_log': west_sigma_log, 'profile': west_profile}
    else:
        west_params = None

    return teal_params, west_params


def execute_targeted_archetype_injection(
    cf_c0: dict,
    div: str,
    target_teals_dict: dict,
    target_west_list: list,
    teal_params: dict,
    west_params: dict,
    global_party_dict: dict,
    simulate: bool = False,
    rng: np.random.Generator = None
) -> dict:
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    updated_c0 = cf_c0.copy()

    runs = []
    if teal_params is not None and target_teals_dict and div in target_teals_dict:
        runs.append((target_teals_dict[div], teal_params, False))
    if west_params is not None and target_west_list is not None and (div in target_west_list or div == 'Werribee'):
        runs.append(('TWP', west_params, True))

    for cand_name, params, is_west in runs:
        if simulate and rng is not None:
            target_vote = float(np.clip(rng.lognormal(params['mu_log'], params['sigma_log']), 0.005, 0.45))
            # Smoothly perturb each bloc's empirical defection rate independently
            stoc_lambdas = {
                b: float(np.clip(params['profile'].get(b, 0.05) * rng.lognormal(0.0, 0.15), 0.0, 1.0))
                for b in BLOCS
            }
        else:
            target_vote = float(params['mean'])
            stoc_lambdas = params['profile'] # True lambda_b defection rates

        # Hopper Maneuver (2026) preserves existing mass
        if is_west and div == 'Werribee':
            existing_hopper = updated_c0.pop('IND_Paul_Hopper', 0.0)
            updated_c0[cand_name] = existing_hopper + target_vote

        else:
            updated_c0[cand_name] = target_vote

        # -------------------------------------------------------------
        # TWO-STAGE HYBRID DRAIN (Smooth Stage 1 Damping + Shifted Stage 2)
        # -------------------------------------------------------------
        surviving_cands = {c: v for c, v in updated_c0.items() if c != cand_name}
        
        # STAGE 1: Damped Natural Base Defection
        stage1_tax = {}
        for c, v in surviving_cands.items():
            b = global_party_dict.get(c, 'Centre')
            b = b if b in BLOCS else 'Centre'
            
            # Smooth Damping Factor: 0.0 at <=1%, 1.0 at >=2.5%
            # Big parties are unaffected (1.0). Micro parties are shielded.
            damp_factor = min(1.0, max(0.0, (v - 0.01) / 0.015))
            
            lam = stoc_lambdas.get(b, 0.0)
            stage1_tax[c] = (v * lam) * damp_factor

        tot_stage1 = sum(stage1_tax.values())
        final_tax = {}

        if tot_stage1 >= target_vote:
            # Scale down proportionally if natural defection exceeds the target
            scale = target_vote / tot_stage1 if tot_stage1 > 1e-9 else 0.0
            final_tax = {c: stage1_tax[c] * scale for c in surviving_cands}
        else:
            # STAGE 2: Reconcile deficit using shifted linear weighting
            final_tax = stage1_tax.copy()
            deficit = target_vote - tot_stage1
            
            weights = {}
            for c, v in surviving_cands.items():
                b = global_party_dict.get(c, 'Centre')
                b = b if b in BLOCS else 'Centre'
                rem_mass = max(0.0, v - stage1_tax[c])
                
                # Shifted linear weighting: bigger parties take linearly more penalty
                shifted_rem = max(0.0, rem_mass - 0.01)
                
                weights[c] = stoc_lambdas.get(b, 0.0) * shifted_rem
                
            tot_w = sum(weights.values())
            
            if tot_w > 1e-9:
                for c in surviving_cands:
                    final_tax[c] += deficit * (weights[c] / tot_w)
            else:
                # Absolute fallback if all shifted weights equal exactly 0
                rem_sum = sum(max(0.0, v - stage1_tax[c]) for c, v in surviving_cands.items())
                for c in surviving_cands:
                    final_tax[c] += deficit * (max(0.0, surviving_cands[c] - stage1_tax[c]) / rem_sum) if rem_sum > 0 else 0

        # Apply final deductions safely
        for c in surviving_cands:
            updated_c0[c] = max(0.0, surviving_cands[c] - final_tax[c])

        # Strict Normalization
        tot = sum(updated_c0.values())
        updated_c0 = {k: v / tot for k, v in updated_c0.items()}

    return updated_c0

if data_year in [2018,2022]:
    teal_params, west_params = model_teals_west(DOP_table_dict, final_model, baseline_theta, inc_df, ia_csv_path, old_contesting_parties, calibration_data, LC_FPs_df, party_df, data_year)
else:
    teal_params, west_params = None, None

# Get master independent composition dictionaries
master_minor_IND_removal_profiles = {} # Compositions for eliminating minor IND (Un-defecting)


for (old_div, new_div), target_cands in next_contesting_parties.items():
    # We want to capture 'IND' and 'TWP' (Teal/West) candidates
    has_ind = any(c.startswith('IND') for c in target_cands)
    has_twp = any(c == 'TWP' for c in target_cands) # treated as IND structurally
    
    if not has_ind and not has_twp:
        continue

    region = electorate_to_region.get(old_div, 'Unknown_Region')
    assert region != 'Unknown_Region'

    # Get compositions for all the INDs and TWPs
    target_inds = [c for c in target_cands if c.startswith('IND') or c == 'TWP']
    
    for ind_name in target_inds:
        # Even generic INDs will get a locally-tuned composition array if the seat has history.
        comp_array = synthesize_dop_composition(old_div, region, dop_lookup)
        
        # We key it by a tuple of (district, ind_name) to ensure complete safety in case a generic 'IND_John_Smith' runs in multiple seats across simulations.

        if not (new_div, ind_name) in master_minor_IND_removal_profiles: # avoid duplications in redistirbution years
            master_minor_IND_removal_profiles[(new_div, ind_name)] = comp_array


DIRICHLET_NU = 20
print(teal_params)
print(west_params)


def load_step5_simulation_inputs(new_inds: pd.DataFrame, pool: pd.DataFrame, target_year: int = 2026) -> Dict[str, Any]:
    """
    Fuses the candidate-level data and electorate-level data into the dictionary format required by Step 5.
    """
    year_pool = pool[pool['Year'] == target_year].set_index('Electorate')
    year_cands = new_inds[new_inds['Year'] == target_year]
    
    inputs = {}
    for div, group in year_cands.groupby('Electorate'):
        if div not in year_pool.index:
            continue
            
        pool_row = year_pool.loc[div]

        raw_days = float(pool_row['Sum_New_Days'])
        log_days = np.log1p(max(0.0, raw_days))
        
        inputs[div] = {
            'new_candidates': group.to_dict('records'),
            'const': 1.0,
            'Log_Generic_Lag': float(pool_row['Log_Generic_Lag']),
            'Log_Repeat_Past': float(pool_row['Log_Repeat_Past']),
            'Has_New_Councillor': float(pool_row['Has_New_Councillor']),
            'Log_Sum_New_Days': log_days
        }
        
    return inputs


def apply_waterfall_extraction(T_pool: float, p_propensity: np.ndarray, capacities: np.ndarray, max_drain: float = 0.90) -> np.ndarray:
    """
    Extracts votes proportionally from the major party blocs, enforcing a capacity floor 
    and reallocating excess demands to uncapped parties to rigidly conserve total extracted mass.
    """

    # fix type to np.float64
    T_pool = float(T_pool)
    p_propensity = np.asarray(p_propensity, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)

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


def execute_new_IND_injection_old(
    district: str,
    cf_c0_step4: Dict[str, float],
    new_ind_features: Optional[Dict[str, Any]],
    pool_model: Any,
    split_model: Any,
    dop_lookup: Dict[str, Any],
    mu_ind_to_ind_array: np.ndarray,
    global_party_dict: Dict[str, str],
    rng: Optional[np.random.Generator] = None,
    simulate = False
) -> Dict[str, float]:
    """
    Step 5: Generates a brand new Independent insurgency pool using the Gamma model, splits it using OLS (if > 1), and carves the mass out of the Step 4 simplex.
    """
    if rng is None: 
        rng = np.random.default_rng()
    if new_ind_features is None:
        return copy.deepcopy(cf_c0_step4)
        
    new_candidates = new_ind_features['new_candidates']
    num_new_inds = len(new_candidates)
    
    if num_new_inds == 0:
        return copy.deepcopy(cf_c0_step4)

    # -------------------------------------------------------------
    # 1. TRACK 2: GENERATE THE NEW IND POOL (Gamma Draw)
    # -------------------------------------------------------------
    x_pool = np.array([
        new_ind_features.get('const', 1.0),
        new_ind_features.get('Log_Generic_Lag', 0.0),
        new_ind_features.get('Has_New_Councillor', 0.0),
        new_ind_features.get('Log_Sum_New_Days', 0.0),
        new_ind_features.get('Log_Repeat_Past', 0.0),
        new_ind_features.get('Pool_Days_Missing', 0.0)
    ], dtype=np.float64)
    
    linear_pred = np.dot(x_pool, pool_model.params)
    mu_pool = np.exp(linear_pred)
    dispersion = pool_model.scale
    
    # Draw from Gamma & scale from percentage (0-100) to simplex (0.0-1.0); use deterministic expected value if simulate=False

    if simulate:
        T_pool_sim = rng.gamma(shape=1.0/dispersion, scale=mu_pool * dispersion) 
    else:
        T_pool_sim = mu_pool  # The expected value of Gamma(k, theta) is mu
    T_pool_sim = min(T_pool_sim, 0.95)

    # -------------------------------------------------------------
    # 2. TRACK 3: ALLOCATE AMONG MULTIPLE NEW INDS (OLS Draw)
    # -------------------------------------------------------------
    ind_votes = {}
    if num_new_inds == 1:
        c = new_candidates[0]
        formatted_name = f"IND_{str(c['Candidate']).strip().title().replace(' ', '_')}" if election_year != 2026 else clean_ind_name(c['Candidate'])
        ind_votes = {formatted_name: T_pool_sim} 
    else:
        log_scores = []
        for c in new_candidates:
            eff_days = c.get('Campaign_Length_Days', 0)
            is_counc = float(c.get('Is_Former_Councillor', False))
            days_missing = float(c.get('Days_Missing', 0.0))
            x_split = np.array([1.0, is_counc, np.log(eff_days + 1), days_missing])
            eps = rng.normal(0, np.sqrt(split_model.mse_resid)) if simulate else 0.0
            pred_log = np.dot(x_split, split_model.params) + eps
            log_scores.append(pred_log)
            
        max_s = max(log_scores)
        exp_s = np.exp(np.array(log_scores) - max_s)
        split_shares = exp_s / np.sum(exp_s)
        
        for i, c in enumerate(new_candidates):
            formatted_name = f"IND_{str(c['Candidate']).strip().title().replace(' ', '_')}" if election_year != 2026 else clean_ind_name(c['Candidate'])
            ind_votes[formatted_name] = T_pool_sim * split_shares[i]

    # -------------------------------------------------------------
    # 3. MACRO CATEGORY MAPPING & CAPACITY CALCULATION
    # -------------------------------------------------------------
    target_cats = dop_lookup['target_cats']
    initial_caps = {cat: 0.0 for cat in target_cats + ['Repeat_IND']}
    cand_to_cat = {}
    
    has_repeat_ind = False
    for c, v in cf_c0_step4.items():
        if str(c).startswith('IND'):
            cat = 'Repeat_IND'
            has_repeat_ind = True
        else:
            cat = global_party_dict.get(c, 'Centre')
            if cat not in target_cats:
                cat = 'Centre'
                
        initial_caps[cat] += v
        cand_to_cat[c] = cat

    # -------------------------------------------------------------
    # 4. DIRICHLET PRIORS & STOCHASTIC IND->IND DRAW
    # -------------------------------------------------------------
    if district in dop_lookup['electorates']: # CHECK - may fail forredistribtuion years as district = new_div
        mu_base = dop_lookup['electorates'][district].copy()
        S = dop_lookup['S_local']
    else:
        mu_base = dop_lookup['state'].copy()
        S = dop_lookup['S_borrowed']

    if has_repeat_ind:
        # Draw the stochastic IND->IND defection parameter for this specific MC iteration
        if len(mu_ind_to_ind_array) > 0:
            sim_mu_ind = rng.choice(mu_ind_to_ind_array) if simulate else np.mean(mu_ind_to_ind_array)
        else:
            sim_mu_ind = 0.20 # Safe fallback
            
        mu_sim = np.append(mu_base * (1.0 - sim_mu_ind), sim_mu_ind)
        categories_sim = target_cats + ['Repeat_IND']
        S = S * 0.70  # Tier 3 (increased variance)
    else:
        mu_sim = mu_base.copy()
        categories_sim = list(target_cats)
        
    if simulate:
        alpha_dirichlet = np.asarray(S * mu_sim, dtype=np.float64)
        p_draw = rng.dirichlet(alpha_dirichlet)
    else:
        p_draw = mu_sim
    caps_array = np.asarray([initial_caps[cat] for cat in categories_sim], dtype=np.float64)

    # -------------------------------------------------------------
    # 5. WATERFALL EXTRACTION & REAPPLICATION
    # -------------------------------------------------------------
    extracted_v = apply_waterfall_extraction(T_pool_sim, p_draw, caps_array, max_drain=0.90)
    
    cf_c0_step5 = {}
    for c, v in cf_c0_step4.items():
        cat = cand_to_cat[c]
        cat_idx = categories_sim.index(cat)
        bloc_baseline = initial_caps[cat]
        
        if bloc_baseline > 1e-9:
            drain_ratio = extracted_v[cat_idx] / bloc_baseline
            cf_c0_step5[c] = v * (1.0 - drain_ratio)
        else:
            cf_c0_step5[c] = v

    # Inject the New INDs
    for cand_name, ind_vote in ind_votes.items():
        cf_c0_step5[cand_name] = ind_vote
        
    z_final = sum(cf_c0_step5.values())
    return {k: v / z_final for k, v in cf_c0_step5.items()}

def execute_new_IND_injection(
    district: str,
    cf_c0_step4: Dict[str, float],
    new_ind_features: Optional[Dict[str, Any]],
    pool_model: Any,
    split_model: Any,
    dop_lookup: Dict[str, Any],
    mu_ind_to_ind_array: np.ndarray,
    global_party_dict: Dict[str, str],
    election_year: int,
    ind_priors: Optional[Dict[Tuple[int, str, str], Dict[str, float]]] = None,
    rng: Optional[np.random.Generator] = None,
    simulate: bool = False
) -> Dict[str, float]:
    """
    Step 5: Generates a brand new Independent insurgency pool using the Gamma model,
    splits it using OLS (if > 1), shrinks specific candidates toward ind_priors
    with precision scaling, and carves the mass out of the Step 4 simplex.
    """
    if rng is None: 
        rng = np.random.default_rng()
    if new_ind_features is None:
        return copy.deepcopy(cf_c0_step4)
        
    new_candidates = new_ind_features['new_candidates']
    num_new_inds = len(new_candidates)
    
    if num_new_inds == 0:
        return copy.deepcopy(cf_c0_step4)

    formatted_names = [
        f"IND_{str(c['Candidate']).strip().title().replace(' ', '_')}"
        if election_year != 2026 else clean_ind_name(c['Candidate'])
        for c in new_candidates
    ]

    # -------------------------------------------------------------
    # 1. TRACK 2 & 3 BASELINE: DETERMINISTIC MODEL EXPECTATIONS
    # -------------------------------------------------------------
    x_pool = np.array([
        new_ind_features.get('const', 1.0),
        new_ind_features.get('Log_Generic_Lag', 0.0),
        new_ind_features.get('Has_New_Councillor', 0.0),
        new_ind_features.get('Log_Sum_New_Days', 0.0),
        new_ind_features.get('Log_Repeat_Past', 0.0),
        new_ind_features.get('Pool_Days_Missing', 0.0)
    ], dtype=np.float64)
    
    linear_pred = np.dot(x_pool, pool_model.params)
    mu_pool = float(min(np.exp(linear_pred), 0.95))
    dispersion = float(pool_model.scale)

    if num_new_inds == 1:
        s_det = np.array([1.0], dtype=np.float64)
    else:
        det_log_scores = []
        for c in new_candidates:
            eff_days = c.get('Campaign_Length_Days', 0)
            is_counc = float(c.get('Is_Former_Councillor', False))
            days_missing = float(c.get('Days_Missing', 0.0))
            x_split = np.array([1.0, is_counc, np.log(eff_days + 1), days_missing], dtype=np.float64)
            det_log_scores.append(float(np.dot(x_split, split_model.params)))
            
        max_det = max(det_log_scores)
        exp_det = np.exp(np.array(det_log_scores, dtype=np.float64) - max_det)
        s_det = exp_det / np.sum(exp_det)

    # -------------------------------------------------------------
    # 2. CANDIDATE-SPECIFIC PRIOR SHRINKAGE & STOCHASTIC DRAWS
    # -------------------------------------------------------------
    mu_cand = mu_pool * s_det
    lambda_vec = np.zeros(num_new_inds, dtype=np.float64)

    if ind_priors:
        prior_applied = False
        for i, cand_name in enumerate(formatted_names):
            p_entry = ind_priors.get((election_year, district, cand_name))
            if p_entry is not None:
                p_val = float(p_entry['prior'])
                lam = float(np.clip(p_entry.get('info_strength', 0.75), 0.0, 1.0))
                mu_cand[i] = (1.0 - lam) * mu_cand[i] + lam * p_val
                lambda_vec[i] = lam
                prior_applied = True

        if prior_applied:
            tot_mu = float(np.sum(mu_cand))
            mu_pool = float(np.clip(tot_mu, 1e-4, 0.95))
            s_det = mu_cand / tot_mu if tot_mu > 1e-9 else s_det

    # Share-weighted pool information strength in [0, 1] for Gamma dispersion scaling
    lambda_pool = float(np.dot(s_det, lambda_vec))
    eff_dispersion = max(dispersion * (1.0 - 0.75 * lambda_pool), 1e-3)

    if simulate:
        T_pool_sim = float(rng.gamma(shape=1.0 / eff_dispersion, scale=mu_pool * eff_dispersion))
    else:
        T_pool_sim = mu_pool
    T_pool_sim = min(T_pool_sim, 0.95)

    ind_votes = {}
    if num_new_inds == 1:
        ind_votes = {formatted_names[0]: T_pool_sim}
    else:
        if simulate:
            base_mse = float(split_model.mse_resid)
            cand_sigmas = np.sqrt(base_mse * (1.0 - 0.85 * lambda_vec))
            eps_vec = rng.normal(0.0, cand_sigmas)
            log_scores = np.log(np.maximum(s_det, 1e-12)) + eps_vec
            max_s = np.max(log_scores)
            exp_s = np.exp(log_scores - max_s)
            split_shares = exp_s / np.sum(exp_s)
        else:
            split_shares = s_det

        for i, cand_name in enumerate(formatted_names):
            ind_votes[cand_name] = T_pool_sim * float(split_shares[i])

    # -------------------------------------------------------------
    # 3. MACRO CATEGORY MAPPING & CAPACITY CALCULATION
    # -------------------------------------------------------------
    target_cats = dop_lookup['target_cats']
    initial_caps = {cat: 0.0 for cat in target_cats + ['Repeat_IND']}
    cand_to_cat = {}
    
    has_repeat_ind = False
    for c, v in cf_c0_step4.items():
        if str(c).startswith('IND'):
            cat = 'Repeat_IND'
            has_repeat_ind = True
        else:
            cat = global_party_dict.get(c, 'Centre')
            if cat not in target_cats:
                cat = 'Centre'
                
        initial_caps[cat] += v
        cand_to_cat[c] = cat

    # -------------------------------------------------------------
    # 4. DIRICHLET PRIORS & STOCHASTIC IND->IND DRAW
    # -------------------------------------------------------------
    if district in dop_lookup['electorates']:
        mu_base = dop_lookup['electorates'][district].copy()
        S = dop_lookup['S_local']
    else:
        mu_base = dop_lookup['state'].copy()
        S = dop_lookup['S_borrowed']

    if has_repeat_ind:
        if len(mu_ind_to_ind_array) > 0:
            sim_mu_ind = rng.choice(mu_ind_to_ind_array) if simulate else np.mean(mu_ind_to_ind_array)
        else:
            sim_mu_ind = 0.20
            
        mu_sim = np.append(mu_base * (1.0 - sim_mu_ind), sim_mu_ind)
        categories_sim = target_cats + ['Repeat_IND']
        S = S * 0.70  # Tier 3 (increased variance)
    else:
        mu_sim = mu_base.copy()
        categories_sim = list(target_cats)
        
    if simulate:
        alpha_dirichlet = np.asarray(S * mu_sim, dtype=np.float64)
        p_draw = rng.dirichlet(alpha_dirichlet)
    else:
        p_draw = mu_sim
    caps_array = np.asarray([initial_caps[cat] for cat in categories_sim], dtype=np.float64)

    # -------------------------------------------------------------
    # 5. WATERFALL EXTRACTION & REAPPLICATION
    # -------------------------------------------------------------
    extracted_v = apply_waterfall_extraction(T_pool_sim, p_draw, caps_array, max_drain=0.90)
    
    cf_c0_step5 = {}
    for c, v in cf_c0_step4.items():
        cat = cand_to_cat[c]
        cat_idx = categories_sim.index(cat)
        bloc_baseline = initial_caps[cat]
        
        if bloc_baseline > 1e-9:
            drain_ratio = extracted_v[cat_idx] / bloc_baseline
            cf_c0_step5[c] = v * (1.0 - drain_ratio)
        else:
            cf_c0_step5[c] = v

    for cand_name, ind_vote in ind_votes.items():
        cf_c0_step5[cand_name] = ind_vote
        
    z_final = sum(cf_c0_step5.values())
    #if district == 'Pakenham':
    #    import pdb; pdb.set_trace()
    return {k: v / z_final for k, v in cf_c0_step5.items()}

anomalous_districts = ['Narracan'] if data_year == 2022 else ['Richmond'] if data_year == 2018 else []

IND_PRIORS = {(2014, 'Gippsland East', 'IND_Peter_Gardner'): {'prior': 0.0168,
                                                 'info_strength': 0.3},
            (2014, 'Keysborough', 'IND_Michael_Carty'): {'prior': 0.0061,
                                                        'info_strength': 0.3},
            (2014, 'Keysborough', 'IND_Hung_Vo'): {'prior': 0.0357, 'info_strength': 0.3},
            (2014, 'Evelyn', 'IND_Lawrence_Irwin_Mobsby'): {'prior': 0.0077,
                                                            'info_strength': 0.3},
            (2014, 'Sunbury', 'IND_Steve_Jack_Medcraft'): {'prior': 0.0674,
                                                            'info_strength': 0.3},
            (2018, 'Albert Park', 'IND_Joseph_Toscano'): {'prior': 0.004,
                                                        'info_strength': 0.3},
            (2018, 'Geelong', 'IND_Stephen_Juhasz'): {'prior': 0.0128,
                                                    'info_strength': 0.3},
            (2018, 'South-West Coast', 'IND_James_Purcell'): {'prior': 0.1153,
                                                            'info_strength': 0.3},
            (2022, 'Clarinda', 'IND_Hung_Vo'): {'prior': 0.0509, 'info_strength': 0.3},
            (2022, 'Kororoit', 'IND_Joh_Bauch'): {'prior': 0.0291, 'info_strength': 0.3},
            (2022, 'Malvern', 'IND_Steve_Stefanopoulos'): {'prior': 0.0061,
                                                            'info_strength': 0.3},
            (2022, 'Mildura', 'IND_Glenn_Milne'): {'prior': 0.1602, 'info_strength': 0.3},
            (2022, 'Mulgrave', 'IND_Joseph_Toscano'): {'prior': 0.0055,
                                                        'info_strength': 0.3},
            (2022, 'Point Cook', 'IND_Joe_Garra'): {'prior': 0.199, 'info_strength': 0.3},
            (2022, 'Morwell', 'IND_Lisa_Proctor'): {'prior': 0.0893,
                                                    'info_strength': 0.3},
            (2022, 'Northcote', 'IND_Adrian_Whitehead'): {'prior': 0.0103,
                                                        'info_strength': 0.3},
            (2026, 'Essendon', 'IND_Oscar_Yildiz'): {'prior': 0.2351,
                                                    'info_strength': 0.5},
            (2026, 'Footscray', 'IND_Catherine_Cumming'): {'prior': 0.1068,
                                                            'info_strength': 0.7},
            (2026, 'Nepean', 'IND_Joseph_Toscano'): {'prior': 0.0051,
                                                    'info_strength': 0.8},
            (2026, 'Sunbury', 'IND_Steve_Medcraft'): {'prior': 0.0158,
                                                    'info_strength': 0.2},
            (2026, 'Preston', 'IND_William_Fowles'): {'prior': 0.07, 'info_strength': 0.2},
            (2026, 'Preston', 'IND_Aidan_Mclindon'): {'prior': 0.022, 'info_strength': 0.5}
        }


# We will store the results of the 1000 simulations here
mc_results_c0 = []
s_hat_a_dicts = []

new_ind_features = load_step5_simulation_inputs(new_inds_df, pool_df, target_year=election_year) # for Step 5 IND addition
target_districts = list(latent_tables.keys())



############################################ Step 6: Readjust for incumbency ###############################################################



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

# prepare incumbent lookup dictionary for Step 6: Re-adding incumbent
inc_next_lookup = {}
for div, df_div in inc_next_df.groupby('div_nm'):
    # Look for the active incumbent record
    inc_rows = df_div[df_div['incumbent'] == 1]
    if not inc_rows.empty:
        inc_next_lookup[div] = []
        for _, row in inc_rows.iterrows():
            inc_next_lookup[div].append({
                'is_open': False,
                'candidate_name': str(row.get('candidate_name', '')).strip(),
                'inc_clean_name': str(row.get('inc_clean_name', '')).strip().lower(),
                'party': str(row.get('PartyAb', '')).upper().strip(),
                'years_tenure': float(row.get('years_tenure', np.nan)),
            })
    else:
        inc_next_lookup[div] = [{'is_open': True}]


########################## Step 6 Polling ###################

def get_poll_bucket(cand_name: str, data_year: int) -> str:
    party = cand_name.split('_')[0] if '_' in cand_name else cand_name
    if party in ['LP', 'NP', 'LNP']: return 'COAL'
    if party == 'ALP': return 'ALP'
    if party == 'GRN': return 'GRN'
    if party == 'ON' and data_year == 2022: return 'ON' 
    # if str(cand_name).startswith('IND'): return 'IND'
    return 'OTH'

def is_left_minor(cand_name: str) -> bool:
    party = cand_name.split('_')[0] if '_' in cand_name else cand_name
    return party in ['AJP', 'SAL', 'SOC', 'VIC_SOC', 'Reason']


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
        if div == 'Narracan':
            district_total = 45132.0
        elif div in dop_table_dict and 'TOTAL' in dop_table_dict[div].columns:
            district_total = float(dop_table_dict[div]['TOTAL'].iloc[0])
        else:
            district_total = 40000.0

        for cand, vote_prop in step5.items():
            if isinstance(vote_prop, (int, float)) and pd.notna(vote_prop):
                bucket = get_poll_bucket(cand, data_year)
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

def compute_statewide_oth_sub_prior(
    cf_c0_step5_dict: dict, 
    dop_table_dict: dict, 
    data_year: int, 
    global_party_dict: dict
) -> dict:
    """
    Scans the statewide Step 5 outputs to calculate the exact prior composition
    of the OTH category (IND, Left, Right, Centre).
    """
    if data_year != 2022:
        return {}

    sub_raw = {'IND': 0.0, 'Left': 0.0, 'Right': 0.0, 'Centre': 0.0}
    alp, coal, grn, on = 0,0,0,0
    for div, step5 in cf_c0_step5_dict.items():
        if data_year == 2022 and div == 'Narracan':
            turnout = 45132.0
        elif div in dop_table_dict and 'TOTAL' in dop_table_dict[div].columns:
            turnout = float(dop_table_dict[div]['TOTAL'].iloc[0])
        else:
            turnout = 40000.0
            
        for cand, vote_prop in step5.items():
            if pd.isna(vote_prop): continue
            
            bucket = get_poll_bucket(cand, data_year)
            if bucket == 'OTH':
                if str(cand).startswith('IND') or str(cand).startswith('TWP'):
                    sub_cat = 'IND'
                else:
                    party = cand.split('_')[0] if '_' in cand else cand
                    b_ideo = (
                        global_party_dict.get(cand, global_party_dict.get(party, 'Centre'))
                        if global_party_dict else 'Centre'
                    )
                    sub_cat = b_ideo if b_ideo in ('Left', 'Right', 'Centre') else 'Centre'
                sub_raw[sub_cat] += vote_prop * turnout
            if bucket == 'ALP':
                alp += vote_prop * turnout
            if bucket == 'COAL':
                coal += vote_prop * turnout
            if bucket == 'GRN':
                grn += vote_prop * turnout
            if bucket == 'ON':
                on += vote_prop * turnout
                
    tot_oth = sum(sub_raw.values())
    import pdb; pdb.set_trace()
    if tot_oth > 1e-9:
        return {k: v / tot_oth for k, v in sub_raw.items()}
    
    # Failsafe flat prior if OTH is completely empty
    import pdb; pdb.set_trace()
    return {'IND': 0.25, 'Left': 0.25, 'Right': 0.25, 'Centre': 0.25}

def apply_district_alr_shift(
    district_c0: dict,
    alr_shifts: dict,
    data_year: int,
    global_party_dict: dict = None,
    w_poll: float = 0.50,
    S_sub_oth: float = 25.0,
    S_intra_jitter: float = 50.0,
    simulate: bool = False,
    rng: np.random.Generator = None
) -> dict:
    """
    Applies the exponentiated ALR shifts to a single district dictionary, enforcing IND immunity and Left-minor pegging.
    """
    # 1. Floor protection to eliminate log-singularities
    cleaned_c0 = {c: max(v, 0.005) for c, v in district_c0.items()}
    tot_cleaned = sum(cleaned_c0.values())
    cleaned_c0 = {c: v / tot_cleaned for c, v in cleaned_c0.items()}
    shifted_c0 = {}


    # --- SECTION 1: 2022 Sub-OTH Relative Tilt (delta_g) with Shrinkage & Uncertainty ---
    if data_year == 2022:
        # Statewide Step 5 prior proportions within OTH (s_g^(0), sums to 1.0)
        OTH_SUB_PRIOR_2022 = {
            'IND': 0.315,
            'Left': 0.332,
            'Right': 0.310,
            'Centre': 0.043, # {'IND': np.float64(0.31466297961636447), 'Left': np.float64(0.33148935593661005), 'Right': np.float64(0.31031834527406316), 'Centre': np.float64(0.04352931917296243)}
        }
        # 2026 LC/LA poll target proportions within OTH (pi_g, sums to 1.0)
        OTH_SUB_POLL_2026 = {
            'IND': 6.0 / 13.0,                 # ~0.4615
            'Left': (10.0 / 20.0) * (7.0 / 13.0),   # ~0.2692
            'Right': (7.0 / 20.0) * (7.0 / 13.0),   # ~0.1885
            'Centre': (3.0 / 20.0) * (7.0 / 13.0),  # ~0.0808
        }
        sub_keys = ['IND', 'Left', 'Right', 'Centre']

        # Blend prior s_g^(0) and poll pi_g by confidence weight w_poll in [0, 1]
        bar_pi = np.array([
            (1.0 - w_poll) * OTH_SUB_PRIOR_2022[g] + w_poll * OTH_SUB_POLL_2026[g]
            for g in sub_keys
        ], dtype=float)
        bar_pi = bar_pi / np.sum(bar_pi)

        # Apply small-sample uncertainty in simulations via Dirichlet draw
        if simulate and rng is not None and S_sub_oth > 0:
            alphas_sub = np.maximum(bar_pi * S_sub_oth, 1e-3)
            tilde_pi = rng.dirichlet(alphas_sub)
        else:
            tilde_pi = bar_pi

        # Relative ALR tilt delta_g = ln(tilde_pi_g / s_g^(0))
        oth_sub_deltas = {
            g: float(np.log(max(tilde_pi[i], 1e-6) / max(OTH_SUB_PRIOR_2022[g], 1e-6)))
            for i, g in enumerate(sub_keys)
        }
    else:
        oth_sub_deltas = {}


    # --- SECTION 2: Group by Category and Apply ALR Shift ---
    sub_category_priors = {}
    cand_to_sub = {}
    
    for cand, vote in cleaned_c0.items():
        bucket = get_poll_bucket(cand, data_year)
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

    # --- SECTION 3: Disaggregate Down to Candidates via Prior Proportions ---
    shifted_c0 = {}
    for sub_g, B_mass in post_sub_mass.items():
        cands_in_g = [c for c, g in cand_to_sub.items() if g == sub_g]
        
        if len(cands_in_g) == 1:
            shifted_c0[cands_in_g[0]] = B_mass
            continue
            
        # Extract strict intra-category prior proportions
        w_prior = np.array([cleaned_c0[c] for c in cands_in_g], dtype=float)
        w_prior /= w_prior.sum()
        
        # Apply Dirichlet Jitter
        # (Assuming S_coal_jitter is passed in or defined, otherwise fallback to S_intra_jitter)
        S_jitter = S_intra_jitter
        if sub_g == 'COAL' and 'S_coal_jitter' in locals():
            S_jitter = LP_NP_DIRICHLET_ALPHA
            
        if simulate and rng is not None and S_jitter is not None and S_jitter > 0:
            alphas = np.maximum(w_prior * S_jitter, 1e-3)
            props = rng.dirichlet(alphas)
        else:
            props = w_prior
            
        for idx, c in enumerate(cands_in_g):
            shifted_c0[c] = B_mass * float(props[idx])
            
    return shifted_c0

def aggregate_statewide_shares(
    cf_dict: dict, 
    dop_table_dict: dict, 
    valid_buckets: list,
    data_year: int
) -> dict:
    """
    Takes a dictionary of district allocations and returns the statewide 
    proportional shares for the target polling buckets.
    """
    bucket_raw = {b: 0.0 for b in valid_buckets}
    total_votes = 0.0
    
    for div, c0_table in cf_dict.items():
        if data_year == 2022 and div == 'Narracan':
            div_turnout = 45132.0
        elif div in dop_table_dict and 'TOTAL' in dop_table_dict[div].columns:
            div_turnout = float(dop_table_dict[div]['TOTAL'].iloc[0])
        else:
            div_turnout = 40000.0
            
        for cand, prop in c0_table.items():
            b = get_poll_bucket(cand, data_year)
            # Map INDs to OTH for polling aggregation purposes
            check_b = 'OTH' if b == 'IND' else b
            if check_b in bucket_raw:
                raw_v = prop * div_turnout
                bucket_raw[check_b] += raw_v
                total_votes += raw_v
                
    if total_votes > 1e-9:
        return {k: v / total_votes for k, v in bucket_raw.items()}
    return {b: 0.0 for b in valid_buckets}

def append_polling_alr_step(
    deterministic_traces: dict, 
    kalman_poll: dict, 
    dop_table_dict: dict,
    ref_col: str = 'COAL',
    data_year: int = 2022
) -> tuple:
    """
    Runs ALR transform, appends to traces, and returns the modified traces along with the final computed post-ALR polling shares.
    """
    valid_buckets = list(kalman_poll.keys())
    
    # 1. Extract Step 6 dictionaries
    cf_c0_step6_dict = {}
    for div, trace_df in deterministic_traces.items():
        if '6_Step6 (COAL Parties)' in trace_df.index:
            row = trace_df.loc['6_Step6 (COAL Parties)']
            cf_c0_step6_dict[div] = {
                c: float(v) for c, v in row.items() if pd.notna(v) and float(v) > 0
            }

    # 2. Get Step 5 Prior & Calculate Shifts
    prior_shares = aggregate_statewide_shares(cf_c0_step6_dict, dop_table_dict, valid_buckets, data_year)
    poll_norm = sum(kalman_poll.values())
    poll_shares = {k: v / poll_norm for k, v in kalman_poll.items()}

    alr_shifts = {}
    for b in valid_buckets:
        prior_alr = np.log((prior_shares[b] + 1e-5) / (prior_shares[ref_col] + 1e-5))
        poll_alr = np.log((poll_shares[b] + 1e-5) / (poll_shares[ref_col] + 1e-5))
        alr_shifts[b] = poll_alr - prior_alr

    # 3. Apply the shifts to create Step 6
    cf_c0_step7_dict = {}
    for div, step6_c0 in cf_c0_step6_dict.items():
        step7_c0 = apply_district_alr_shift(step6_c0, alr_shifts, data_year, global_party_dict) # Protects INDs and Left Minors
        cf_c0_step7_dict[div] = step7_c0
        deterministic_traces[div].loc['7_Step7 (State Polling ALR)'] = pd.Series(step7_c0)

    # 4. DIAGNOSTIC: Calculate the exact shares AFTER ALR using the same function
    post_shares = aggregate_statewide_shares(cf_c0_step7_dict, dop_table_dict, valid_buckets, data_year)

    # 5. Print the Comparison
    print("\n" + "=" * 52)
    print("      POST-ALR STATEWIDE 5-BLOC DIAGNOSTIC")
    print("=" * 52)
    print(f"{'Bloc':<8} | {'Prior (S6)':<11} | {'Target':<11} | {'Post-ALR (S7)':<11}")
    print("-" * 52)
    for b in valid_buckets:
        prior_pct = prior_shares[b] * 100
        target_pct = poll_shares[b] * 100
        post_pct = post_shares[b] * 100
        print(f"{b:<8} | {prior_pct:>10.2f}% | {target_pct:>10.2f}% | {post_pct:>10.2f}%")
    print("=" * 52 + "\n")

    # Return traces AND the final shares dict so you can access it programmatically
    return deterministic_traces, post_shares


########################## Redistribution step #######################################33

def apply_elimination_flows(df_start: pd.DataFrame, start_c0: dict, reduced_c0: dict) -> pd.DataFrame:
    """
    STAGE 1: start_c0 -> reduced_c0
    Eliminated parties are stripped out, and their voters return to the surviving parties 
    proportional to how those survivors grew in the logit model.
    """
    eliminated = [p for p in start_c0 if p not in reduced_c0]
    survivors = list(reduced_c0.keys())
    
    # 1. Macro Logic: How much of the total eliminated pool did each survivor absorb?
    v_elim_macro = sum(start_c0[p] for p in eliminated)
    
    flow_props = {}
    if v_elim_macro > 0:
        for p in survivors:
            # Survivor's gain = Their reduced share minus their starting share
            gain = max(0, reduced_c0[p] - start_c0.get(p, 0))
            flow_props[p] = gain / v_elim_macro
    else:
        # If no parties were eliminated, survivors just keep their original votes
        flow_props = {p: 0.0 for p in survivors}
    
    # 2. Micro Application (1D Segment Dict)
    df_reduced = {p: float(df_start.get(p, 0.0)) for p in survivors}
    
    if eliminated:
        local_elim_pool = sum( float(df_start.get(p, 0.0)) for p in eliminated )
        for p in survivors:
            df_reduced[p] += local_elim_pool * flow_props[p]
            
    return df_reduced

def apply_entrant_flows(df_reduced: pd.DataFrame, reduced_c0: dict, final_c0: dict) -> pd.DataFrame:
    """
    STAGE 2: reduced_c0 -> final_c0
    Surviving parties shed a fraction of their votes based on their retention rates.
    Those shed votes are pooled and given to the new 2026 insurgents.
    """
    survivors = list(reduced_c0.keys())
    new_entrants = [p for p in final_c0 if p not in reduced_c0]
    
    # 1. Macro Logic: Retention Rates & Insurgent Shares
    # What percentage of their base did each survivor manage to keep?
    retention = {
        p: (final_c0.get(p, 0) / reduced_c0[p]) if reduced_c0[p] > 0 else 0.0 
        for p in survivors
    }
    
    # Of the total votes stolen by new entrants, what was each entrant's share?
    v_added_macro = sum(final_c0[p] for p in new_entrants)
    entrant_shares = {
        p: (final_c0[p] / v_added_macro) if v_added_macro > 0 else 0.0
        for p in new_entrants
    }
    
    # 2. Micro Application (1D Segment Dict)
    df_final = {}
    local_shed_pool = 0.0
    
    for p in survivors:
        v_red = df_reduced[p]
        df_final[p] = v_red * retention[p]
        local_shed_pool += v_red * (1.0 - retention[p])
        
    for p in new_entrants:
        df_final[p] = local_shed_pool * entrant_shares[p]
        
    return df_final

def redistribute_lc_fps_df(
    LC_FPs_df: pd.DataFrame,
    cf_c0_step5_dict: dict,
    rediv_transfer_props_dict: dict
) -> pd.DataFrame:
    """
    Transforms LC_FPs_df from old_div columns to new_div columns using
    rediv_transfer_props_dict weights, aligning LP/LNP across region boundaries
    and preserving single-constituent parties at full strength before renormalizing.
    """
    new_to_old = defaultdict(list)
    for (old_div, new_div) in cf_c0_step5_dict.keys():
        if old_div in LC_FPs_df.columns:
            w = float(rediv_transfer_props_dict.get((old_div, new_div), 1.0))
            if w > 0:
                new_to_old[new_div].append((old_div, w))
    
    rediv_cols = {}
    for new_div, old_weights in new_to_old.items():
        # Fast path: single constituent old_div
        if len(old_weights) == 1:
            old_div = old_weights[0][0]
            s = LC_FPs_df[old_div].copy()
            rediv_cols[new_div] = s / s.sum()
            continue

        # Align LP vs LNP to the dominant (highest-weight) constituent old_div
        dom_old = max(old_weights, key=lambda x: x[1])[0]
        if 'LNP' in LC_FPs_df.index and pd.notna(LC_FPs_df.loc['LNP', dom_old]):
            target_coal, alt_coal = 'LNP', 'LP'
        else:
            target_coal, alt_coal = 'LP', 'LNP'

        weighted_sum = defaultdict(float)
        weight_total = defaultdict(float)

        for old_div, w in old_weights:
            col_s = LC_FPs_df[old_div].dropna()
            for party, val in col_s.items():
                p_key = target_coal if party == alt_coal else party
                weighted_sum[p_key] += float(val) * w
                weight_total[p_key] += w

        # Single-entry parties get (val * w) / w = val; multi-entry get weighted average
        new_s = pd.Series(
            {p: weighted_sum[p] / weight_total[p] for p in weighted_sum if weight_total[p] > 0},
            index=LC_FPs_df.index,
            dtype=float
        )
        rediv_cols[new_div] = new_s / new_s.sum()

    LC_FPs_df_rediv = pd.DataFrame(rediv_cols, index=LC_FPs_df.index)
    LC_FPs_df_rediv.index.name = LC_FPs_df.index.name
    #import pdb; pdb.set_trace()
    return LC_FPs_df_rediv

def execute_final_redistribution(
    rediv_seg_votes_dict: dict, 
    raw_c0_dict: dict, 
    latent_tables: dict, 
    cf_c0_step5_dict: dict, 
    rediv_transfer_props_dict: dict,
    LC_FPs_df: pd.DataFrame = None,
    redistribute_LC: bool = False
):
    """
    Transforms and aggregates raw polling booth votes into final 2026 district totals.
    """
    # Use defaultdict to automatically accumulate vote counts across multiple segments feeding into the same new_div
    redistributed_votes = defaultdict(lambda: defaultdict(float))
    
    for (old_div, new_div) in cf_c0_step5_dict:

        seg_raw_dict = rediv_seg_votes_dict[(old_div, new_div)]
        
        # 1. Fetch the 3 perfectly normalized Macro Dictionaries
        start_c0 = raw_c0_dict[old_div]
        reduced_c0 = latent_tables[(old_div, new_div)].count_0
        final_c0 = cf_c0_step5_dict[(old_div, new_div)]
        
        # Ensure df_raw only has columns that exist in the starting state
        # (Fills missing parties with 0 to prevent KeyError in flow functions)
        for p in start_c0.keys():
            if p not in seg_raw_dict:
                import pdb; pdb.set_trace()
                seg_raw_dict[p] = 0.0
        
        # 2. Apply the 2-Stage Flow Math (these now operate on, and return, 1D dicts)
        df_reduced = apply_elimination_flows(seg_raw_dict, start_c0, reduced_c0)
        df_final = apply_entrant_flows(df_reduced, reduced_c0, final_c0)
        
        # 3 & 4. Accumulate segment votes directly into new_div's running total
        div_totals = redistributed_votes[new_div]
        for p, v in df_final.items():
            div_totals[p] += v

    # 5. Clean up into a standard Python dictionary of raw votes
    redistributed_c0_dict = {}
    
    for new_div, total_dict in redistributed_votes.items():
        # Drop zeroes/floating point artifacts
        clean_dict = {p: v for p, v in total_dict.items() if v > 1e-5}
        
        # Calculate the total formal votes for the new district
        total_votes = sum(clean_dict.values())
        
        if total_votes > 0:
            # Divide each party's votes by the total to get fractions (0.0 to 1.0)
            redistributed_c0_dict[new_div] = {p: v / total_votes for p, v in clean_dict.items()}
        else:
            import pdb; pdb.set_trace()
            # Fallback in case of an empty district
            redistributed_c0_dict[new_div] = {}

    if redistribute_LC:
        LC_FPs_df_rediv = redistribute_lc_fps_df(LC_FPs_df, cf_c0_step5_dict, rediv_transfer_props_dict)
    else:
        LC_FPs_df_rediv = pd.DataFrame()

    return redistributed_c0_dict, LC_FPs_df_rediv

# prepare raw c0 for redistribution:

raw_c0_dict = {}
meta_cols = {'div_nm', 'CandRemaining', 'TOTAL', 'CountType'}

for old_div, raw_df in DOP_table_dict.items():
    
    # extract the primary vote row (Count 0)
    c0_row = raw_df.iloc[0] 
    # extract valid party/candidate votes
    raw_c0 = {
        c: float(c0_row[c]) 
        for c in raw_df.columns 
        if c not in meta_cols and pd.notna(c0_row[c]) and float(c0_row[c]) > 0
    }
    # normalisation
    tot = sum(raw_c0.values())
    raw_c0_dict[old_div] = {k: v / tot for k, v in raw_c0.items()}



######################################### COAL double div reallocation

def estimate_coalition_params_and_flows(
    det_latent_tables: Dict[tuple, Any], # NEW
    lc_df: pd.DataFrame, # NEW
    party_df: pd.DataFrame, # NEW
    static_high_ind_districts: List,
    model: Any # NEW
) -> Tuple[float, float, Dict[str, float]]: # NEW
    """
    Estimates from the current data_year:
    1. tau_coal: symmetric expansion/discount factor comparing empirical LA COAL
       against modelled LC-on-LA-menu COAL across 1-COAL vs 2-COAL seats.
    2. mean_np_share: average NP / (LP + NP) share in 2-COAL seats.
    3. right_leak_profile: model-implied flow weights from Right -> ['ALP', 'Left', 'Right', 'Centre'].
    """
    alpha, beta, gamma = model.unpack_params(model.theta) # NEW
    log_ratios_1_coal = [] # NEW
    log_ratios_2_coal = [] # NEW
    np_shares = [] # NEW

    # --- A. Estimate tau_coal and mean_np_share across det_latent_tables ---
    seen_divs = set() # NEW
    for key, table in det_latent_tables.items(): # NEW
        div = key[0] # NEW
        if div in seen_divs or div not in lc_df: # NEW
            continue # NEW
        seen_divs.add(div) # NEW

        if div in static_high_ind_districts: # INDs corrupt the method
            continue

        raw_party = {k: v for k, v in table.count_0.items() if not str(k).startswith('IND') and v > 1e-9} # NEW
        tot_p = sum(raw_party.values()) # NEW
        if tot_p <= 1e-9: # NEW
            continue # NEW
        norm_party = {k: v / tot_p for k, v in raw_party.items()} # NEW

        coal_present = [c for c in ('LP', 'NP') if norm_party.get(c, 0.0) > 0.00] # NEW
        if not coal_present: # NEW
            continue # NEW

        # Collapse LP and NP into 'LP' so the menu has 1 unified COAL slot on the exact same minor menu
        emp_coal = sum(norm_party.get(c, 0.0) for c in ('LP', 'NP')) # NEW
        collapsed_c0 = {k: v for k, v in norm_party.items() if k not in ('LP', 'NP')} # NEW
        collapsed_c0['LP'] = emp_coal # NEW
        menu = list(collapsed_c0.keys()) # NEW

        lc_series = lc_df[div].dropna().copy() # NEW
        if lc_series.sum() > 1.5: # NEW
            lc_series = lc_series / 100.0 # NEW

        # NEW: Strictly filter for regional joint-ticket LC environments: Skip pure-metro seats where only LP ran in both houses.
        has_regional_lc = lc_series.get('LNP', 0.0) > 1e-4 or lc_series.get('NP', 0.0) > 1e-4
        if not has_regional_lc:
            continue

        ms, cap, ctb, cd, lc_mass = _build_state_vars(menu, lc_series, collapsed_c0, party_df) # NEW
        s_hat = model.evaluate_state_allocation(lc_mass, menu, cap, ms, ctb, cd, alpha, beta, gamma) # NEW
        mod_coal = s_hat.get('LP', 0.0) # NEW

        if emp_coal > 0.05 and mod_coal > 0.05: # NEW
            log_r = np.log(emp_coal / mod_coal) # NEW
            if len(coal_present) == 1: # NEW
                log_ratios_1_coal.append(log_r) # NEW
            elif len(coal_present) == 2: # NEW
                log_ratios_2_coal.append(log_r) # NEW
                np_shares.append(norm_party.get('NP', 0.0) / emp_coal) # NEW
                print(div)
        #print(div)
        #print(emp_coal, mod_coal, lc_series, raw_party)
        #import pdb; pdb.set_trace()

    if log_ratios_1_coal and log_ratios_2_coal: # NEW
        raw_tau = float(np.exp(np.mean(log_ratios_2_coal) - np.mean(log_ratios_1_coal))) # NEW
        tau_coal = float(np.clip(raw_tau, 1.01, 1.25)) # NEW
    else: # NEW
        tau_coal = 1.08  # Safe fallback if a historical election had zero 2-COAL seats # NEW

    mean_np_share = float(np.mean(np_shares)) if np_shares else 0.40 # NEW
    #import pdb; pdb.set_trace()

    # --- B. Extract Right -> ['ALP', 'Left', 'Right', 'Centre'] Flow Profile from Model ---
    # Find one representative party code in party_df for Right and Centre
   
    right_rep = {2022: 'ON',2018: 'SFF',2014:'FFP',2010: 'CYA',2006:'PP'}
    centre_rep = {2022: 'DHJP', 2018: 'DHJP', 2014: 'DLP', 2010: 'DLP', 2006: 'DEM'}

    r_rep = right_rep[data_year]
    c_rep = centre_rep[data_year]
    canon_menu = ['ALP', 'LP', 'GRN', right_rep[data_year], centre_rep[data_year]] # NEW
    canon_c0_dict = {2022: {'ALP': 0.41, 'LP': 0.39, 'GRN': 0.12, right_rep[2022]: 0.05, centre_rep[2022]: 0.03}, 
                2018: {'ALP': 0.44, 'LP': 0.38, 'GRN': 0.11, right_rep[2018]: 0.04, centre_rep[2018]: 0.03},
                2014: {'ALP': 0.39, 'LP': 0.42, 'GRN': 0.12, right_rep[2014]: 0.04, centre_rep[2014]: 0.03},
                2010: {'ALP': 0.37, 'LP': 0.46, 'GRN': 0.12, right_rep[2010]: 0.03, centre_rep[2010]: 0.02},
                2006: {'ALP': 0.44, 'LP': 0.40, 'GRN': 0.10, right_rep[2006]: 0.04, centre_rep[2006]: 0.02},
            }
    canon_c0 = canon_c0_dict[data_year]
    
    # 1. Build canon_lc from lc_df.mean(axis=1), combining LP, LNP, and NP into 'LP'
    raw_lc_mean = lc_df.mean(axis=1).dropna()
    if raw_lc_mean.sum() > 1.5:
        raw_lc_mean = raw_lc_mean / 100.0

    # Combine LP, LNP, NP per electorate first so metro LP (0.27) + rural LNP (0.33)
    # averages to ~0.29 instead of summing to ~0.60, or fallback to raw_lc_mean if 1D
    coal_keys_in_lc = [k for k in ('LP', 'LNP', 'NP', 'COAL') if k in lc_df.index]
    combined_coal_lc = lc_df.loc[coal_keys_in_lc].sum(axis=0).mean()
    if combined_coal_lc > 1.5:
        combined_coal_lc /= 100.0

    canon_lc = raw_lc_mean.drop(labels=[k for k in ('LNP', 'NP', 'COAL') if k in raw_lc_mean.index])
    canon_lc['LP'] = combined_coal_lc
    canon_lc = canon_lc / canon_lc.sum()

    # 2. Identify all available Right donor parties in canon_lc (excluding canon_menu)
    donor_parties = [
        p for p in canon_lc.index
        if p not in canon_menu
        and p not in ('LP', 'LNP', 'NP', 'COAL', 'IND')
        and canon_lc[p] > 1e-6
        and get_party_bloc(p, party_df) == 'Right'
    ]

    # 3. Loop over each Right donor party: evaluate (canon_menu + [rp]) -> canon_menu
    NON_COAL_BLOCS = ['ALP', 'Left', 'Right', 'Centre']
    donor_flow_profiles = []

    for rp in donor_parties:
        menu_A = canon_menu + [rp]

        # Add donor to canon_c0 using its canon_lc share and renormalise to 1.0
        c0_A = canon_c0.copy()
        c0_A[rp] = float(canon_lc[rp])
        tot_c0_A = sum(c0_A.values())
        c0_A = {k: v / tot_c0_A for k, v in c0_A.items()}

        # State A: canon_menu + [rp]
        ms_A, cap_A, ctb_A, cd_A, lc_mass_A = _build_state_vars(menu_A, canon_lc, c0_A, party_df)
        s_hat_A = model.evaluate_state_allocation(lc_mass_A, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)

        # State B: canon_menu (after eliminating rp)
        ms_B, cap_B, ctb_B, cd_B, lc_mass_B = _build_state_vars(canon_menu, canon_lc, c0_A, party_df)
        s_hat_B = model.evaluate_state_allocation(lc_mass_B, canon_menu, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)

        deltas = {c: max(0.0, s_hat_B.get(c, 0.0) - s_hat_A.get(c, 0.0)) for c in canon_menu}
        non_coal_raw = {
            'ALP': deltas.get('ALP', 0.0),
            'Left': deltas.get('GRN', 0.0),
            'Right': deltas.get(r_rep, 0.0),
            'Centre': deltas.get(c_rep, 0.0),
        }

        tot_nc = sum(non_coal_raw.values())
        if tot_nc > 1e-9:
            donor_flow_profiles.append({b: v / tot_nc for b, v in non_coal_raw.items()})

    # 4. Average the normalised flow profiles across all Right donors
    if donor_flow_profiles:
        avg_profile = {
            b: float(np.mean([prof[b] for prof in donor_flow_profiles]))
            for b in NON_COAL_BLOCS
        }
        tot_avg = sum(avg_profile.values())
        right_leak_profile = {b: v / tot_avg for b, v in avg_profile.items()}
    else:
        import pdb; pdb.set_trace()
        right_leak_profile = {'Right': 0.45, 'Centre': 0.30, 'ALP': 0.20, 'Left': 0.05}

    import pdb; pdb.set_trace()
    return tau_coal, mean_np_share, right_leak_profile #


def execute_coalition_reconciliation(
    cf_c0: Dict[str, float],
    pair_key: tuple,
    COAL_party_change_dict: dict,
    tau_coal: float,
    np_share_priors: dict,
    right_leak_profile: Dict[str, float],
    global_party_dict: Dict[str, str],
    simulate: bool = True,
    rng: np.random.Generator = None,
    S_coal_split: float = 4.1783,
    S_leak_split: float = 25.0
) -> Dict[str, float]:
    """
    Reconciles changes in the number/identity of contesting Coalition parties
    using the pre-computed COAL_party_change_dict.
    """
    # Fast exit for standard 1->1 (same party) and 2->2 seats
    if pair_key not in COAL_party_change_dict:
        return cf_c0
        
    changes = COAL_party_change_dict[pair_key]
    removed_coal = changes['removed']
    added_coal = changes['added']
    
    COAL_SET = {'LP', 'NP', 'LNP', 'COAL'}
    old_coal_in_dict = [c for c in COAL_SET if c in cf_c0 and cf_c0[c] > 1e-9]
    
    # Determine the target COAL set we need to end up with
    next_coal = sorted(list((set(old_coal_in_dict) - set(removed_coal)) | set(added_coal)))
    
    updated_c0 = cf_c0.copy()
    v_coal = sum(updated_c0.pop(c, 0.0) for c in old_coal_in_dict) # combined COAL vote
    
    # Failsafe if the historical COAL party had effectively zero votes
    if v_coal <= 1e-9:
        import pdb; pdb.set_trace()
        for c in next_coal:
            updated_c0[c] = 0.0
        return updated_c0

    # Map surviving non-COAL candidates (including INDs -> 'Centre') to the 4 non-COAL blocs
    NON_COAL_BLOCS = ['ALP', 'Left', 'Right', 'Centre']
    bloc_cands = {b: [] for b in NON_COAL_BLOCS}
    for c, v in updated_c0.items():
        if v > 1e-9:
            if str(c).startswith('IND') or str(c).startswith('TWP'):
                b = 'Centre'
            else:
                b = global_party_dict.get(c, 'Centre')
                b = b if b in NON_COAL_BLOCS else 'Centre'
            bloc_cands[b].append(c)

    # Filter right_leak_profile to blocs that actually have active candidates in this seat
    base_bloc_weights = {b: right_leak_profile.get(b, 0.0) for b in NON_COAL_BLOCS if bloc_cands[b]}
    tot_bw = sum(base_bloc_weights.values())
    if tot_bw > 1e-9:
        base_bloc_weights = {b: w / tot_bw for b, w in base_bloc_weights.items()}
    else:
        active_blocs = [b for b in NON_COAL_BLOCS if bloc_cands[b]]
        base_bloc_weights = {b: 1.0 / len(active_blocs) for b in active_blocs}

    # DRAW STOCHASTIC PARAMETERS FOR THIS SIMULATION
    np_share_prior = np_share_priors.get(pair_key[1], 0.5)
    if simulate and rng is not None:
        # A. Stochastic Tau
        tau_sim = float(np.clip(rng.normal(tau_coal, 0.035), 1.01, 1.15))
        
        # B. Stochastic Leak Profile (Dirichlet)
        bloc_keys = list(base_bloc_weights.keys())
        alpha_leak = np.maximum(np.array([base_bloc_weights[b] for b in bloc_keys]) * S_leak_split, 1e-3)
        drawn_leak = rng.dirichlet(alpha_leak)
        active_bloc_weights = {b: float(drawn_leak[i]) for i, b in enumerate(bloc_keys)}
        
        # C. Stochastic LP/NP Split (Dirichlet)
        
        alpha_split = np.maximum(np.array([1.0 - np_share_prior, np_share_prior]) * S_coal_split, 1e-3)
        lp_prop, np_prop = rng.dirichlet(alpha_split)
    else:
        tau_sim = tau_coal
        active_bloc_weights = base_bloc_weights
        lp_prop, np_prop = (1.0 - np_share_prior), np_share_prior

    # Case A: 1 -> 1 Partner Swap (e.g. LP -> NP)
    if len(old_coal_in_dict) == 1 and len(next_coal) == 1:
        updated_c0[next_coal[0]] = v_coal

    # Case B: 2 -> 1 or 1 -> 0 (Consolidation/Withdrawal)
    elif len(old_coal_in_dict) > len(next_coal):
        if len(next_coal) == 1:
            v_coal_new = v_coal / tau_sim
            leaked_mass = v_coal - v_coal_new
            updated_c0[next_coal[0]] = v_coal_new
        else: 
            # 1 -> 0 (e.g. Richmond 2018): 100% of LP vote leaks
            leaked_mass = v_coal
            
        for b, b_weight in active_bloc_weights.items():
            b_mass = leaked_mass * b_weight
            cands_in_b = bloc_cands[b]
            b_tot = sum(updated_c0[c] for c in cands_in_b)
            for c in cands_in_b:
                share = (updated_c0[c] / b_tot) if b_tot > 1e-9 else (1.0 / len(cands_in_b))
                updated_c0[c] += b_mass * share

    # Case C: 1 -> 2 (Expansion Bonus + LP/NP Split, e.g., Ripon)
    elif len(old_coal_in_dict) < len(next_coal):
        needed_gain = v_coal * (tau_sim - 1.0)
        actual_gain = 0.0
        
        for b, b_weight in active_bloc_weights.items():
            b_drain_target = needed_gain * b_weight
            cands_in_b = bloc_cands[b]
            b_tot = sum(updated_c0[c] for c in cands_in_b)
            if b_tot <= 1e-9:
                continue
            for c in cands_in_b:
                share = updated_c0[c] / b_tot
                drain_c = min(updated_c0[c] * 0.50, b_drain_target * share) # cap at maximum of 50% of a candidate's supply FIX: perform this supply-based as in other sections
                updated_c0[c] -= drain_c
                actual_gain += drain_c
                
        v_coal_expanded = v_coal + actual_gain
        
        updated_c0['LP'] = v_coal_expanded * lp_prop
        updated_c0['NP'] = v_coal_expanded * np_prop

        if np_share_prior == 0.5:
            import pdb; pdb.set_trace()

    # Final simplex normalization
    tot = sum(updated_c0.values())
    return {k: v / tot for k, v in updated_c0.items()} if tot > 0 else updated_c0

fresh_coal_params_estimation = 0
if fresh_coal_params_estimation:
    tau_coal, mean_np_share, right_leak_profile = estimate_coalition_params_and_flows(
        det_latent_tables = latent_tables,
        lc_df = LC_FPs_df,
        party_df = party_df,
        static_high_ind_districts=static_high_ind_districts,
        model = final_model
    )

tau_coal_2006 = 1.04744
tau_coal_2010 = 1.06968
tau_coal_2014 = 1.09482
tau_coal_2018 = 1.01
tau_coal_2022 = 1.11951
tau_coal_per_election = {2006: tau_coal_2006, 2010: tau_coal_2010*0.6+tau_coal_2006*0.4, 2014: tau_coal_2014*0.6 + (tau_coal_2010*0.6+tau_coal_2006*0.4)*0.4, 2018:  tau_coal_2014*0.6 + (tau_coal_2010*0.6+tau_coal_2006*0.4)*0.4, 2022: tau_coal_2022*0.6 + (tau_coal_2014*0.6 + (tau_coal_2010*0.6+tau_coal_2006*0.4)*0.4)*0.4}
np_share_prior_dict = {2022: {'Ripon': 0.358}, 
                  2018: {'Mildura': 0.746, 'Euroa': 0.581, 'Bass':0.319,'Morwell': 0.46, 'Shepparton': 0.329},
                  2014: {'Shepparton': 0.498, 'Morwell': 0.663, 'Bendigo East': 0.194},
                  2010: {'Euroa': 0.746, 'Eildon': 0.097, 'Buninyong': 0.194, 'Ripon': 0.249},
                  2006: {'Seymour': 0.319}
                  }
if data_year == 2022:
    right_leak_profile = {'ALP': 0.1839825868761584, 'Left': 0.20281739967023574, 'Right': 0.48157334576346056, 'Centre': 0.13162666769014528}
elif data_year == 2018:
    right_leak_profile = {'ALP': 0.1839825868761584, 'Left': 0.20281739967023574, 'Right': 0.48157334576346056, 'Centre': 0.13162666769014528}
elif data_year <= 2014:
    right_leak_profile = {'ALP': 0.16541718389556379, 'Left': 0.14803161794093067, 'Right': 0.522938124960242, 'Centre': 0.16361307320326365} # CHEAT fro 2010-2006 for convenience
    


def diagnose_ind_volatility(cf_c0_step4_dict, new_ind_features, election_year, n_sims=1000):
    rng_diag = np.random.default_rng(42)
    rows = []
    for key, step4 in cf_c0_step4_dict.items():
        div = key[1]
        print(key)
        if div not in new_ind_features or not new_ind_features[div]['new_candidates']:
            continue

        # Map formatted candidate name -> candidate feature dict
        cand_meta = {}
        for c in new_ind_features[div]['new_candidates']:
            f_name = (
                f"IND_{str(c['Candidate']).strip().title().replace(' ', '_')}"
                if election_year != 2026 else clean_ind_name(c['Candidate'])
            )
            prior_entry = IND_PRIORS.get((election_year, div, f_name)) if IND_PRIORS else None
            cand_meta[f_name] = {
                'Is_Former_Councillor': bool(c.get('Is_Former_Councillor', False)),
                'Campaign_Length_Days': float(c.get('Campaign_Length_Days', np.nan)),
                'ind_prior': float(prior_entry['prior']) if prior_entry is not None else np.nan,
            }

        # 1. Deterministic baseline
        det_s5 = execute_new_IND_injection(
            div, step4, new_ind_features[div], pool_model, split_model,
            dop_lookup, mu_ind_to_ind_array, global_party_dict, election_year, IND_PRIORS, simulate=False
        )
        # 2. 1000 Monte Carlo draws
        sim_records = defaultdict(list)
        for _ in range(n_sims):
            sim_s5 = execute_new_IND_injection(
                div, step4, new_ind_features[div], pool_model, split_model,
                dop_lookup, mu_ind_to_ind_array, global_party_dict, election_year, IND_PRIORS, rng=rng_diag, simulate=True
            )
            for k, v in sim_s5.items():
                if k.startswith('IND_') and k not in step4:
                    sim_records[k].append(v)

        for cand, vals in sim_records.items():
            arr = np.array(vals)
            meta = cand_meta.get(cand, {})
            rows.append({
                'District': div, 'Candidate': cand,
                'Is_Former_Councillor': meta.get('Is_Former_Councillor', False),
                'Campaign_Length_Days': meta.get('Campaign_Length_Days', np.nan),
                'ind_prior': meta.get('ind_prior', np.nan),
                'Det_Mean': det_s5[cand], 'Sim_Mean': arr.mean(),
                'P90': np.percentile(arr, 90), 'P95': np.percentile(arr, 95),
                'P99': np.percentile(arr, 99), 'Max': arr.max(),
                'Prob_gt_18pct': (arr > 0.18).mean() * 100.0,
                'Prob_gt_25pct': (arr > 0.25).mean() * 100.0,
                'Prob_gt_35pct': (arr > 0.35).mean() * 100.0,
                'Prob_gt_50pct': (arr > 0.50).mean() * 100.0
            })
    df_diag = pd.DataFrame(rows).sort_values('P99', ascending=False)
    print(f"pool_model.scale (phi) = {pool_model.scale:.4f}, split RMSE = {np.sqrt(split_model.mse_resid):.4f}")
    return df_diag

############################################# Full simulation loop #######################################################################################

def generate_deterministic_trace_run(
    target_districts: List[str],
    DOP_table_dict: Dict[str, Any],
    raw_c0_dict: Dict[str, Any],
    inc_next_lookup: Dict[str, Any],
    inc_df: pd.DataFrame,
    ia_params_dict: dict,              # <--- Changed from ia_model
    ia_csv_path: str,                  # <--- New path argument for Step 1
    demo_lookup: Dict[str, str],       # <--- New lookup argument for Step 6
    final_model: Any,
    base_theta: np.ndarray,            
    mean_kappas: Dict[str, float],     
    next_contesting_parties: Dict[tuple, List[str]],
    LC_FPs_df: pd.DataFrame,
    party_df: pd.DataFrame,
    new_ind_features: Dict[str, Any],
    pool_model: Any,
    split_model: Any,
    dop_lookup: Dict[str, Any],
    mu_ind_to_ind_array: np.ndarray,
    global_party_dict: Dict[str, str],
    kalman_poll: Dict[str, float],
    data_year: int 
) -> Dict[str, pd.DataFrame]:
    
    debug_traces = {}
    rng_det = np.random.default_rng(seed=42)
    final_model.set_params(base_theta)

    master_4cp_defection_rates = {}

    # -------------------------------------------------------------
    # STEP 1: Deterministic IA Removal (Natively Count 0 via CSV)
    # -------------------------------------------------------------
    stripped_tables, static_high_ind_districts, ia_tracker_dict = execute_strip_incumbency(
        dop_table_dict=DOP_table_dict,
        inc_df = inc_df,
        csv_path=ia_csv_path
    )
    #stripped_tables = {(div, div): stripped_tables[div] for div in stripped_tables.keys()}
    stripped_tables = {(old_div, new_div): stripped_tables[old_div] for old_div, new_div in next_contesting_parties.keys() if old_div in stripped_tables}

    # -------------------------------------------------------------
    # STEP 2: Deterministic Party Removal
    # -------------------------------------------------------------
    latent_tables = execute_party_removal(
        stripped_tables=stripped_tables,
        static_high_ind_districts=static_high_ind_districts,
        next_contesting_parties=next_contesting_parties,
        LC_FPs_df=LC_FPs_df,
        party_df=party_df,             
        dop_lookup=dop_lookup,                  
        electorate_to_region=electorate_to_region, 
        model = final_model,
        simulate=False,
        rng=rng_det
    )

    target_districts = list(latent_tables.keys())

    if (data_year == 2022):
        for old_div, new_div in next_contesting_parties.keys():
            if old_div == 'Narracan' and (old_div, new_div) not in latent_tables:
                sample_key = next(iter(latent_tables.keys()))
                narracan_dummy = copy.deepcopy(latent_tables[sample_key])
                # Blank out count_0 since there is no 2022 LH general baseline
                if hasattr(narracan_dummy, 'count_0'):
                    narracan_dummy.count_0 = pd.Series(dtype=float) if isinstance(narracan_dummy.count_0, pd.Series) else {}
                latent_tables[(old_div, new_div)] = narracan_dummy

    # Build the 88-district dictionary cache once
    step3_cache = precompute_step3_cache(
        det_latent_tables=latent_tables,
        next_contesting_parties=next_contesting_parties,
        lc_df=LC_FPs_df,
        party_df=party_df,
        model=final_model
    )
    # Create perfectly isolated IND baseline

    #augmented_c0_dict = {}
    #for div in target_districts:
    #    if div in stripped_tables:
    #        augmented_c0_dict[div] = create_augmented_c0_for_defection(
    #            stripped_c0=stripped_tables[div].count_0,
    #            target_cands=next_contesting_parties.get(div, [])
    #        )

    augmented_c0_dict = {
        (old_div, new_div): create_augmented_c0_for_defection(
            stripped_c0=table.count_0,
            target_cands=next_contesting_parties.get((old_div, new_div), [])
        ) for (old_div, new_div), table in stripped_tables.items() if (old_div, new_div) in target_districts # CHECK 'if old_div in target_districts'
    }

    

    # Build the 88-district dictionary cache once
    step3_cache = precompute_step3_cache(
        det_latent_tables=latent_tables,
        next_contesting_parties=next_contesting_parties,
        lc_df=LC_FPs_df,
        party_df=party_df,
        model=final_model
    )

    # -------------------------------------------------------------
    # STEP 3: Minor Party Injection
    # -------------------------------------------------------------
    cf_c0_party_dict, s_hat_a_dict = execute_new_minor_injection(
        latent_tables=latent_tables,
        next_contesting_parties=next_contesting_parties,
        #lc_df=LC_FPs_df,
        #party_df=party_df,
        #model=final_model,
        step3_cache=step3_cache,
        sim_kappas=mean_kappas,
        static_high_ind_districts=static_high_ind_districts,
        anomalous_districts=anomalous_districts, # Update if you still use anomalous_districts
        simulate=False,
        rng=rng_det
    )

    cf_c0_step6_dict = {}
    cf_c0_step5_dict = {}
    cf_c0_step4_dict = {}

    for (old_div, new_div) in (target_districts + [('Narracan', 'Narracan')] if data_year == 2022 else target_districts): # CHECK: only 2026 needed? - abusing the fact that there's no redistribution
        lc_series = LC_FPs_df[old_div].dropna()
        lc_series = lc_series / 100.0 if lc_series.sum() > 1.5 else lc_series
        
        if (old_div == 'Narracan') and (data_year == 2022):
            # 1. Evaluate entire 2026 Count 0 directly using the model and LC series
            narracan_cands = [c for c in next_contesting_parties[('Narracan', 'Narracan')] if not str(c).startswith('IND')]
            narracan_c0 = build_and_evaluate_model_allocation(
                district='Narracan',
                menu=narracan_cands,
                lc_series=lc_series,
                party_df=party_df,
                model=final_model,
                fallback_c0=None
            )
            # Normalize to guarantee exact 1.0 sum
            tot = sum(narracan_c0.values())
            cf_c0_step4 = {k: v / tot for k, v in narracan_c0.items()}

        # ---------------------------------------------------------
        # STEP 4: Restore Re-contesting Independents
        # ---------------------------------------------------------
        else:
            cands_added = [c for c in cf_c0_party_dict[(old_div, new_div)] if c not in s_hat_a_dict[(old_div, new_div)]]

            cf_c0_step4, hist_lambdas_for_div = execute_IND_restoration_via_defections(
                district_pair=(old_div, new_div),
                latent_c0=augmented_c0_dict[(old_div, new_div)],
                cf_c0_party=cf_c0_party_dict[(old_div, new_div)],
                s_hat_a=s_hat_a_dict[(old_div, new_div)],
                cands_added_in_step_3=cands_added,
                next_contesting_parties=next_contesting_parties,
                lc_series=lc_series,
                party_df=party_df,
                model=final_model,
                global_party_dict=global_party_dict
            )

            
            if hist_lambdas_for_div:
                master_4cp_defection_rates.update(hist_lambdas_for_div)
       
        # ---------------------------------------------------------
        # STEP 5: Inject New Independent Insurgency
        # ---------------------------------------------------------
        cf_c0_step5 = execute_new_IND_injection(
            district=new_div,
            cf_c0_step4=cf_c0_step4,
            new_ind_features=new_ind_features.get(new_div, None),
            pool_model=pool_model,
            split_model=split_model,
            dop_lookup=dop_lookup,
            mu_ind_to_ind_array=mu_ind_to_ind_array,
            global_party_dict=global_party_dict,
            election_year = data_year + 4,
            ind_priors = IND_PRIORS,
            simulate=False,
            rng=rng_det
        ) 
        # Step 5b: Deterministic Archetype Injection
        if data_year in [2018,2022]:

            cf_c0_step5 = execute_targeted_archetype_injection(
                cf_c0=cf_c0_step5,
                div=new_div,
                target_teals_dict=TARGET_TEALS,
                target_west_list=TARGET_WEST,
                teal_params=teal_params,
                west_params=west_params,
                global_party_dict=global_party_dict,
                simulate=False,
                rng=None
            )

        # apply any COAL double divs changes
        cf_c0_step6 = execute_coalition_reconciliation( 
            cf_c0=cf_c0_step5, 
            pair_key = (old_div, new_div),
            COAL_party_change_dict=COAL_party_change_dict, 
            tau_coal=tau_coal_per_election[data_year], 
            np_share_priors=np_share_prior_dict[data_year], 
            right_leak_profile=right_leak_profile, 
            global_party_dict=global_party_dict, 
            simulate=False, 
            rng=None, 
            S_coal_split = 4.1783,
            S_leak_split = 25
        )
        
        
        cf_c0_step6_dict[(old_div, new_div)] = cf_c0_step6
        cf_c0_step5_dict[(old_div, new_div)] = cf_c0_step5
        cf_c0_step4_dict[(old_div, new_div)] = cf_c0_step4
        
        # =========================================================
        # TRACE CAPTURE (Ends at Step 5)
        # =========================================================
        raw_c0 = {}
        if old_div in DOP_table_dict:
            raw_df = DOP_table_dict[old_div]
            c0_row = raw_df.iloc[0] 
            meta_cols = {'div_nm', 'CandRemaining', 'TOTAL', 'CountType'}
            
            raw_c0 = {
                c: float(c0_row[c]) 
                for c in raw_df.columns 
                if c not in meta_cols and pd.notna(c0_row[c]) and float(c0_row[c]) > 0
            }
            
            tot = sum(raw_c0.values())
            if tot > 1.5:
                raw_c0 = {k: v / tot for k, v in raw_c0.items()}

        if (old_div == 'Narracan') and (data_year == 2022):
            trace_data = {
                '0_Baseline (2022)': {},
                '1_Step1 (Strip IA)': {},
                '2_Step2 (Remove Parties)': {},
                '3_Step3 (Inject Minors)': {},
                '4_Step4 (Restore INDs)': cf_c0_step4,
                '5_Step5 (New INDs)': cf_c0_step5,
                '6_Step6 (COAL Parties)': cf_c0_step6
            }
        else:
            trace_data = {
                '0_Baseline (2022)': raw_c0,
                '1_Step1 (Strip IA)': getattr(stripped_tables.get((old_div, new_div), None), 'count_0', {}),
                '2_Step2 (Remove Parties)': latent_tables[(old_div, new_div)].count_0,
                '3_Step3 (Inject Minors)': cf_c0_party_dict[(old_div, new_div)],
                '4_Step4 (Restore INDs)': cf_c0_step4,
                '5_Step5 (New INDs)': cf_c0_step5,
                '6_Step6 (COAL Parties)': cf_c0_step6
            }
        
        df_trace = pd.DataFrame.from_dict(trace_data, orient='index').fillna(0.0)
        df_trace = df_trace.sort_index()
        debug_traces[(old_div, new_div)] = df_trace.reindex(sorted(df_trace.columns), axis=1)

    #df_diag = diagnose_ind_volatility(cf_c0_step4_dict, new_ind_features, data_year + 4, n_sims=1000)
    #if data_year == 2022:
    #    exact_oth_prior = compute_statewide_oth_sub_prior(cf_c0_step5_dict, DOP_table_dict, data_year, global_party_dict)
    #    import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    # APPLY REDISTRIBUTION QUANTITIES, reduce dict keys to single electorate 

    target_districts = [key[1] for key in target_districts]
    if not is_redistribution:
        debug_traces = {key[1]: debug_traces[key] for key in debug_traces.keys()} # old_div = new_div should persist!  
    else:
        cf_c0_step6_dict, LC_FPs_df_rediv = execute_final_redistribution(
            rediv_seg_votes_dict = rediv_seg_votes_dict, 
            raw_c0_dict = raw_c0_dict, 
            latent_tables = latent_tables, 
            cf_c0_step5_dict = cf_c0_step6_dict, 
            rediv_transfer_props_dict = rediv_transfer_props_dict,
            LC_FPs_df = LC_FPs_df,
            redistribute_LC = True
        )
        assert len(cf_c0_step6_dict) == 88 if data_year != 2018 else  len(cf_c0_step6_dict) == 87# Victoria - ensure abolished divs don't hang around

        debug_traces_rediv = {}
        # keep debug_traces for the (old_div, old_div) one where possible. If fully new electorate, then clear out the starts
        for div in cf_c0_step6_dict.keys():
            if (div, div) in debug_traces:
                debug_traces_rediv[div] = debug_traces[(div, div)]

            else:
                # Brand new division - create empty trace with standard index
                steps = ['0_Baseline (2022)', '1_Step1 (Strip IA)', '2_Step2 (Remove Parties)', 
                         '3_Step3 (Inject Minors)', '4_Step4 (Restore INDs)', '5_Step5 (New INDs)', '6_Step6 (COAL Parties)']
                debug_traces_rediv[div] = pd.DataFrame(index=steps, columns=cf_c0_step6_dict[div].keys(), dtype=float)
            # add redistribution result to trace
            debug_traces_rediv[div].loc['R_StepR (Redistribution)'] = cf_c0_step6_dict[div]
            debug_traces_rediv[div] = debug_traces_rediv[div].fillna(0.0)
        debug_traces = debug_traces_rediv.copy()

    # =========================================================
    # APPLY ALR POLLING SHIFT (STEP 6)
    # =========================================================

    

    print("Applying Kalman ALR Polling Shift to Count 5...")
    debug_traces, post_alr_shares = append_polling_alr_step(
        deterministic_traces=debug_traces,
        kalman_poll=kalman_poll,
        dop_table_dict=DOP_table_dict,
        ref_col='COAL',
        data_year = data_year
    )

    print(post_alr_shares)
    #import pdb; pdb.set_trace()

    # =========================================================
    # SECOND LOOP: RE-APPLY INCUMBENCY ADVANTAGE (STEP 7)
    # =========================================================
    for div in target_districts:
        # Extract the ALR-swung baseline from the trace
        step7_series = debug_traces[div].loc['7_Step7 (State Polling ALR)'].dropna()
        cf_c0_step7 = {k: float(v) for k, v in step7_series.items() if v > 0}

        cf_c0_step8 = execute_reapply_incumbency(
            div=div,
            cf_c0_step5=cf_c0_step7,  # Pass Step 7 in as the structural baseline
            ia_tracker_list=ia_tracker_dict.get(div, None),
            inc_next_info_list=inc_next_lookup.get(div, None),
            params_dict=ia_params_dict,                       
            demo_class=demo_lookup.get(div, 'Inner Metropolitan'), 
            simulate=False,                                   
            rng=rng_det
        )
        #import pdb; pdb.set_trace()

        # Append Step 7 to the trace
        debug_traces[div].loc['8_Step8 (Add 2026 IA)'] = pd.Series(cf_c0_step8)

    return debug_traces, master_4cp_defection_rates, LC_FPs_df_rediv if is_redistribution else []

mean_kappas = draw_stochastic_kappas(calibration_data, simulate=False)

def load_kalman_poll(csv_path: str, election_year: int, ref_col: str = 'COAL') -> Tuple[dict, List[str]]:
    df = pd.read_csv(csv_path, index_col='election_year')
    row = df.loc[election_year].dropna()
    kalman_poll = {k: float(v) for k, v in row.items() if float(v) > 0.0}
    
    # Dynamic ALR columns (4 in 2022, 3 in <=2018), excluding reference 'COAL'
    alr_cols = [k for k in kalman_poll.keys() if k != ref_col]
    return kalman_poll, alr_cols

kalman_poll, alr_cols = load_kalman_poll('kalman_polls.csv', election_year, ref_col = 'COAL')
assert alr_cols == ALR_COLS
print("Running deterministic structural trace...")

deterministic_traces, master_4cp_defection_rates, LC_FPs_df_rediv = generate_deterministic_trace_run(
    target_districts=target_districts,
    DOP_table_dict=DOP_table_dict,
    raw_c0_dict = raw_c0_dict,
    inc_next_lookup=inc_next_lookup,
    inc_df=  inc_df,
    
    # --- Decoupled IA & Demographic Arguments ---
    ia_params_dict=ia_params_dict,      
    ia_csv_path=ia_csv_path,            
    demo_lookup=demo_lookup,            
    
    # --- Model Arguments ---
    final_model=final_model,
    base_theta=baseline_theta,
    mean_kappas=mean_kappas,            # <--- Passed pure expected value here
    
    # --- Simulation State Arguments ---
    next_contesting_parties=next_contesting_parties,
    LC_FPs_df=LC_FPs_df,
    party_df=party_df,
    new_ind_features=new_ind_features,
    pool_model=pool_model,
    split_model=split_model,
    dop_lookup=dop_lookup,
    mu_ind_to_ind_array=mu_ind_to_ind_array,
    global_party_dict=global_party_dict,
    kalman_poll=kalman_poll,
    data_year = data_year
)

print(f"Trace completed for {len(deterministic_traces)} districts.")


print("Step 7 Complete. Traces updated.")

with open("deterministic_traces.pkl", "wb") as f:
    pickle.dump(deterministic_traces, f)

print("Successfully saved deterministic_traces.pkl")



#import pdb; pdb.set_trace()

# diagnostics
# np.array(sorted([np.round(deterministic_traces[div].iloc[-1,:]['ON'], 3)*100 for div in deterministic_traces.keys()]))
# low_on = [div for div in deterministic_traces.keys() if np.round(deterministic_traces[div].iloc[-1,:]['ON'], 3)*100<10]
# high_on = [div for div in deterministic_traces.keys() if np.round(deterministic_traces[div].iloc[-1,:]['ON'], 3)*100>33]





####################### get composition and defection rate dicts ############3

compositions_dict = master_minor_IND_removal_profiles

# Generate the classifications dynamically
ind_classifications = {}
for seat, cand_name in TARGET_TEALS.items():
    ind_classifications[(seat, cand_name)] = 'teal'
for seat in TARGET_WEST:
    ind_classifications[(seat, 'TWP')] = 'west'
if election_year == 2026:
    ind_classifications[('Werribee', 'TWP')] = 'west' # Paul Hopper
    compositions_dict[('Werribee', 'TWP')] = compositions_dict[('Werribee','IND_Paul_Hopper')]


# 2. DEFECTION RATES DICTIONARY
# Use this when modeling preference flows / defection from major INDs/TWPs.
def build_defection_rates(
        master_4cp_rates: dict,
        teal_profile: dict,
        west_profile: dict,
        candidate_archetypes: dict
    ) -> tuple[dict, dict]:
    """
    Builds the target defection rates mapping and calculates a fallback average.
    Priority: Specific empirics > Archetype > Fallback Average.
    """
    defection_rates = {}
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']

    # Step A: Calculate the Fallback (Average of general INDs)
    # We derive this dynamically from all empirical candidate specific rates
    fallback_avg = {k: 0.0 for k in BLOCS}
    if master_4cp_rates:
        for profile in master_4cp_rates.values():
            for k in BLOCS:
                fallback_avg[k] += float(profile.get(k, 0.0))
        fallback_avg = {k: v / len(master_4cp_rates) for k, v in fallback_avg.items()}
    else:
        # Failsafe if the master dictionary is empty
        import pdb; pdb.set_trace()
        fallback_avg = {'ALP': 0.2, 'COAL': 0.2, 'Left': 0.2, 'Right': 0.2, 'Centre': 0.2}

    # Step B: Apply Archetypes (Medium Priority)
    for seat_cand, archetype in candidate_archetypes.items():
        if archetype == 'teal':
            defection_rates[seat_cand] = teal_profile.copy()
        elif archetype == 'west':
            defection_rates[seat_cand] = west_profile.copy()

    # Step C: Overwrite with Specific Empirics (Highest Priority)
    for seat_cand, profile in master_4cp_rates.items():
        defection_rates[seat_cand] = profile.copy()

    # Step D: Standardize formatting (clean floats, ensure identical keys)
    for seat_cand, profile in defection_rates.items():
        clean_profile = {}
        for k in BLOCS:
            clean_profile[k] = float(profile.get(k, 0.0))
        defection_rates[seat_cand] = clean_profile

    return defection_rates, fallback_avg

# Generate the structures
specific_defection_dict, fallback_defection_profile = build_defection_rates(
    master_4cp_rates=master_4cp_defection_rates,
    teal_profile=teal_params['profile'] if teal_params is not None else {},
    west_profile=west_params['profile'] if west_params is not None else {},
    candidate_archetypes=ind_classifications
)

specific_defection_dict = {key: specific_defection_dict[key] for key in specific_defection_dict.keys() if key in compositions_dict.keys()}

#import pdb; pdb.set_trace()
############################ precompute polling and residual errors for simulation loop ############################

# Stop the script here to skip the heavy Monte Carlo simulations
#sys.exit(0)

def predraw_polling_residual_simulation_error(
    days_to_election: float,
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
    with open(Path.home() / "Australian Election/Farrer_byelection_covms.pkl", "rb") as f:
        covm_data = pickle.load(f)

    raw_poll_df = covm_data["Nat_poll_covm"]
    raw_elec_df = covm_data["Electorate_residual_covm"]

    # Map party names to raw 5x5 indices in case pickle stores numpy arrays instead of DataFrames
    # (Adjust RAW_5X5_INDEX if your numpy array uses a different order than [ALP, GRN, ON, IND, OTH])
    RAW_5X5_INDEX = {'ALP': 0, 'GRN': 1, 'ON': 2, 'IND': 3, 'OTH': 4}
    keep_idx = [RAW_5X5_INDEX[p] for p in alr_cols]

    base_poll = (
        raw_poll_df.loc[alr_cols, alr_cols].values
        if isinstance(raw_poll_df, pd.DataFrame)
        else raw_poll_df[np.ix_(keep_idx, keep_idx)]
    )
    C_elec = (
        raw_elec_df.loc[alr_cols, alr_cols].values
        if isinstance(raw_elec_df, pd.DataFrame)
        else raw_elec_df[np.ix_(keep_idx, keep_idx)]
    )

    # 2. Time-decay the statewide polling matrix (4x4 in 2022, 3x3 in <=2018)
    stds_base = np.sqrt(np.diag(base_poll))
    R_poll = base_poll / np.outer(stds_base, stds_base)

    s_diag = [
        OPT_EXP_DECAY_PARAMETERS[p][1]
        + (OPT_EXP_DECAY_PARAMETERS[p][0] - OPT_EXP_DECAY_PARAMETERS[p][1])
        * np.exp(-OPT_EXP_DECAY_PARAMETERS[p][2] * days_to_election)
        for p in alr_cols
    ]

    S = np.diag(s_diag)
    C_poll_unclean = S @ R_poll @ S
    eigvals, eigvecs = np.linalg.eigh(C_poll_unclean)
    C_poll = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T

    # 3. Pre-draw simulation error tensors of shape (num_simulations, n_alr) and (num_simulations, NUM_DISTRICTS, n_alr)
    sim_poll_errors = rng.multivariate_normal(np.zeros(n_alr), C_poll, size=num_simulations)
    sim_elec_errors = rng.multivariate_normal(np.zeros(n_alr), C_elec, size=(num_simulations, NUM_DISTRICTS))

    return sim_poll_errors, sim_elec_errors

num_simulations = 1000
start= time.time()
print(f"Initiating Monte Carlo structural injection ({num_simulations} iterations)...")

days_to_election = 17

sim_poll_errors, sim_elec_errors = predraw_polling_residual_simulation_error(days_to_election, num_simulations=num_simulations, NUM_DISTRICTS=88 - (data_year == 2018), rng=rng, 
                                                                             data_year = data_year, alr_cols = ALR_COLS)

NU_MINOR_PRECISION = {2022: 795, 2018: 748, 2014: 812,2010:762, 2006: 901}

# Lock the model to the single MLE / optimal parameter set
final_model.set_params(baseline_theta)

mc_results_c0 = []
s_hat_a_dicts = []

stripped_tables = {(old_div, new_div): stripped_tables[(old_div, old_div)] for old_div, new_div in next_contesting_parties.keys() if (old_div, old_div) in stripped_tables}


# precompute modelled states
baseline_latent_tables = execute_party_removal(
    stripped_tables=stripped_tables,
    static_high_ind_districts=static_high_ind_districts,
    next_contesting_parties=next_contesting_parties,
    LC_FPs_df=LC_FPs_df,
    party_df=party_df,             
    dop_lookup=dop_lookup,                  
    electorate_to_region=electorate_to_region, 
    model = final_model,
    simulate=False,
    rng=rng 
)
# narracan dummy
if (data_year == 2022):
    for old_div, new_div in next_contesting_parties.keys():
        if old_div == 'Narracan' and (old_div, new_div) not in latent_tables:
            sample_key = next(iter(latent_tables.keys()))
            narracan_dummy = copy.deepcopy(latent_tables[sample_key])
            # Blank out count_0 since there is no 2022 LH general baseline
            if hasattr(narracan_dummy, 'count_0'):
                narracan_dummy.count_0 = pd.Series(dtype=float) if isinstance(narracan_dummy.count_0, pd.Series) else {}
            baseline_latent_tables[(old_div, new_div)] = narracan_dummy

# Build the 88-district dictionary cache once
global_step3_cache = precompute_step3_cache(
    det_latent_tables=baseline_latent_tables, # uses baseline tables - this should have very minor effect on modelled states - CHECK
    next_contesting_parties=next_contesting_parties,
    lc_df=LC_FPs_df,
    party_df=party_df,
    model=final_model
)

global_step4_cache = precompute_step4_entrant_cache(
    det_latent_tables=baseline_latent_tables, 
    global_step3_cache=global_step3_cache,
    next_contesting_parties=next_contesting_parties,
    lc_df=LC_FPs_df,
    party_df=party_df,
    model=final_model,
    data_year=data_year
)


# =========================================================================
# MONTE CARLO SIMULATION LOOP
# =========================================================================
for sim_idx in range(num_simulations):

    if sim_idx % 50 == 1:
        print("Completed", sim_idx - 1)

    next_time = time.time()

    # -------------------------------------------------------------
    # STEP 2: Stochastic Party Removal
    # -------------------------------------------------------------
    latent_tables = execute_party_removal(
        stripped_tables=stripped_tables,
        static_high_ind_districts=static_high_ind_districts,
        next_contesting_parties=next_contesting_parties,
        LC_FPs_df=LC_FPs_df,
        party_df=party_df,             
        dop_lookup=dop_lookup,                  
        electorate_to_region=electorate_to_region, 
        model = final_model,
        simulate=True,
        rng=rng 
    )

    target_districts = list(latent_tables.keys())

    print(time.time() - next_time, 'Step 2 done')
    next_time = time.time()

    if (data_year == 2022):
        latent_tables[('Narracan', 'Narracan')] = narracan_dummy

    # Re-build isolated IND baseline for defections
    #augmented_c0_dict = {}
    #for div in target_districts:
    #    if div in stripped_tables:
    #        augmented_c0_dict[div] = create_augmented_c0_for_defection(
    #            stripped_c0=stripped_tables[div].count_0,
    #            target_cands=next_contesting_parties.get(div, [])
    #        )
    augmented_c0_dict = {
        (old_div, new_div): create_augmented_c0_for_defection(
            stripped_c0=table.count_0,
            target_cands=next_contesting_parties.get((old_div, new_div), [])
        ) for (old_div, new_div), table in stripped_tables.items() if (old_div, new_div) in target_districts # CHECK 'if old_div in target_districts'
    }
    #import pdb; pdb.set_trace()
    print(time.time() - next_time, 'Step 3b')
    next_time = time.time()

    # -------------------------------------------------------------
    # STEP 3: Minor Party Injection + Dirichlet Precision Resampling
    # -------------------------------------------------------------
    sim_kappas = draw_stochastic_kappas(calibration_data, simulate=True, rng=rng)

    cf_c0_party_dict, s_hat_a_dict = execute_new_minor_injection(
        latent_tables=latent_tables,
        next_contesting_parties=next_contesting_parties,
        step3_cache=global_step3_cache,
        #lc_df=LC_FPs_df,
        #party_df=party_df,
        #model=final_model,            
        sim_kappas=sim_kappas,
        static_high_ind_districts=static_high_ind_districts,
        rng=rng,
        anomalous_districts=anomalous_districts,
        simulate = True
    )
    print(time.time() - next_time, 'Step 3c')
    next_time = time.time()

    # --- APPLY DIRICHLET VARIANCE TO STEP 3 PRIMARY SHARES ---
    for key, c0_dict in cf_c0_party_dict.items():
        cands = list(c0_dict.keys())
        expected_shares = [c0_dict[c] for c in cands]
        
        # Scale expected probabilities by calibrated precision
        alphas = [max(1e-3, p * NU_MINOR_PRECISION[data_year]) for p in expected_shares]
        
        # Draw the true election-day distribution on the simplex
        simulated_shares = rng.dirichlet(alphas)
        cf_c0_party_dict[key] = dict(zip(cands, simulated_shares))

    s_hat_a_dicts.append(s_hat_a_dict)

    print(time.time() - next_time, 'Step 3 done')
    next_time = time.time()

    # -------------------------------------------------------------
    # PASS 1: Build Baselines through Step 5 Across All Districts
    # -------------------------------------------------------------
    cf_c0_step5_dict = {}

    for (old_div, new_div) in (target_districts + [('Narracan', 'Narracan')] if data_year == 2022 else target_districts):
        pair_key = (old_div, new_div)

        # --- Narracan Dedicated Clause (Uses cached deterministic model evaluation) ---
        if (old_div == 'Narracan') and (data_year == 2022):
            nar_cache = global_step4_cache[pair_key]
            alphas_nar = [max(1e-3, p * NU_MINOR_PRECISION[data_year]) for p in nar_cache['norm_narracan']]
            sim_nar = rng.dirichlet(alphas_nar)
            cf_c0_step4 = dict(zip(nar_cache['narracan_keys'], sim_nar))

        # --- Standard Step 4 (Fast Cached Model + Live latent_c0 Defections) ---
        else:
            cands_added_in_step_3 = global_step3_cache[pair_key]['missing_cands']
            cf_c0_step4 = execute_IND_restoration_fast_sim(
                district_pair=pair_key,
                latent_c0=augmented_c0_dict[pair_key],
                cf_c0_party=cf_c0_party_dict[pair_key],
                s_hat_a=s_hat_a_dict[pair_key],
                cands_added_in_step_3=cands_added_in_step_3,
                next_contesting_parties=next_contesting_parties,
                step4_cache_entry=global_step4_cache[pair_key]
            )

        # ---------------------------------------------------------
        # STEP 5: Inject New Independent Insurgency
        # ---------------------------------------------------------
        cf_c0_step5 = execute_new_IND_injection(
            district=new_div,
            cf_c0_step4=cf_c0_step4,
            new_ind_features=new_ind_features.get(new_div, None),
            pool_model=pool_model,
            split_model=split_model,
            dop_lookup=dop_lookup,
            mu_ind_to_ind_array=mu_ind_to_ind_array,
            global_party_dict=global_party_dict,
            election_year = election_year,
            ind_priors = IND_PRIORS,
            rng=rng,
            simulate=True
        )

        # Step 5b: Stochastic Archetype Injection
        if data_year in [2018,2022]:
            cf_c0_step5 = execute_targeted_archetype_injection(
                cf_c0=cf_c0_step5,
                div=new_div,
                target_teals_dict=TARGET_TEALS,
                target_west_list=TARGET_WEST,
                teal_params=teal_params,
                west_params=west_params,
                global_party_dict=global_party_dict,
                simulate=True,
                rng=rng 
            )

        # apply any COAL double divs changes
        cf_c0_step5 = execute_coalition_reconciliation( 
            cf_c0=cf_c0_step5, 
            pair_key = (old_div, new_div),
            COAL_party_change_dict=COAL_party_change_dict, 
            tau_coal=tau_coal_per_election[data_year], 
            np_share_priors=np_share_prior_dict[data_year],
            right_leak_profile=right_leak_profile, 
            global_party_dict=global_party_dict, 
            simulate=True, 
            rng=rng, 
            S_coal_split = LP_NP_DIRICHLET_ALPHA,
            S_leak_split = 25
        )
        
        

        cf_c0_step5_dict[(old_div, new_div)] = cf_c0_step5


    print(time.time() - next_time, 'Step 5 done')
    next_time = time.time()

    # APPLY REDISTRIBUTION QUANTITIES

    # reduce dict keys to single electorate 
    if not is_redistribution:
        cf_c0_step5_dict = {key[1]: cf_c0_step5_dict[key] for key in cf_c0_step5_dict.keys()} # old_div = new_div should persist!
        #import pdb; pdb.set_trace()
    else:

        cf_c0_stepR_dict, _ = execute_final_redistribution(
            rediv_seg_votes_dict = rediv_seg_votes_dict, 
            raw_c0_dict = raw_c0_dict, 
            latent_tables = latent_tables, 
            cf_c0_step5_dict = cf_c0_step5_dict, 
            rediv_transfer_props_dict = rediv_transfer_props_dict,
            LC_FPs_df = LC_FPs_df,
            redistribute_LC = False
        )
        cf_c0_step5_dict__copy = cf_c0_step5_dict.copy()
        cf_c0_step5_dict = cf_c0_stepR_dict.copy()
        #import pdb; pdb.set_trace()

    target_districts = [key[1] for key in target_districts]
    target_districts = [d for d in  dict.fromkeys(target_districts).keys()]

    # -------------------------------------------------------------
    # Compute Statewide ALR Shifts (Step 6)
    # -------------------------------------------------------------
    # Uses your existing compute_statewide_alr_shifts helper
    alr_shifts = compute_statewide_alr_shifts(
        cf_c0_step5_dict=cf_c0_step5_dict,
        kalman_poll=kalman_poll,
        dop_table_dict=DOP_table_dict,
        data_year = data_year,
        ref_col='COAL'
    )

    # -------------------------------------------------------------
    # PASS 2: Apply ALR Shift (Step 6) and IA (Step 7)
    # -------------------------------------------------------------
    cf_c0_step7_dict = {}

    for d_idx, div in enumerate(target_districts + ['Narracan'] if data_year == 2022 else target_districts):
        # 1. Total ALR Shift = Deterministic Base + Statewide Error + Electorate Shock
        div_shifts = {b: alr_shifts[b] + sim_poll_errors[sim_idx, k] + sim_elec_errors[sim_idx, d_idx, k] for k, b in enumerate(ALR_COLS)}
        div_shifts['COAL'] = 0.0  # Reference baseline

        # 2. Apply through existing function
        cf_c0_step6 = apply_district_alr_shift(
            district_c0=cf_c0_step5_dict[div],
            alr_shifts=div_shifts,
            data_year = data_year,
            global_party_dict = global_party_dict,
            simulate = True,
            rng = rng
        )
      

        # 2. Step 7: Apply 2026 Incumbency Advantage on the post-ALR baseline
        cf_c0_step7 = execute_reapply_incumbency(
            div=div,
            cf_c0_step5=cf_c0_step6,
            ia_tracker_list=ia_tracker_dict.get(div, None),
            inc_next_info_list=inc_next_lookup.get(div, None),
            params_dict=ia_params_dict,                       
            demo_class=demo_lookup.get(div, 'Inner Metropolitan'), 
            simulate=False,                                   
            rng=rng
        )

        cf_c0_step7_dict[div] = cf_c0_step7

    # Store the complete Step 7 realization for this Monte Carlo draw
    mc_results_c0.append(cf_c0_step7_dict)

    print(time.time() - next_time, 'Step 7 done')
    next_time = time.time()

print(time.time() - start)
# pd.DataFrame([result['Bentleigh'] for result in mc_results_c0])

electorates = mc_results_c0[0].keys()
FP_pred = {e: pd.DataFrame([res[e] for res in mc_results_c0]) for e in electorates}


#import pdb; pdb.set_trace()

#with open(f"FP_pred_{data_year}.pkl", "wb") as f:
#    pickle.dump(FP_pred, f)




# project 2CP results

TCP_COMBINATION_INDEX = {
    ('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, 
    ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, 
    ('COAL','Right'):6, ('LP','NP'): 7, ('IND','IND'): 8, 
    ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, 
    ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14
}



def adjust_TCP_state_dict(TCP_state_dict, party_df, SIGMA_JOINT_BASE):

    # rough notes
    # 0: 115, 1: 14 and Nepean/SA, 2: 8, 3: 9, 4: 1 and Secret Harbour/SA, 5: 1, 6: 1 + SA, 7: 0, 8: 0,9: (8-) and Farrer/SA, 10: (14-), 11: (115-), 12: 0, 13: 0, 14: 0
    # LP/ON shifts: 

    # 6. {'IND': 75}
    # 4. {'LP':34,'GRN':91,'HMP':56,'AJP':62,'Right':31} (and hence 11) w. 'ALP':91
    # 9. {'LP':41,'NP':31,'GRN':92,'HMP':63, 'Centre':70,'Right':39}

    if election_year == 2026:
        UPDATED_FLOWS_ELECTION = {
            6: {'IND': 75.0},
            4: {'LP': 34.0, 'GRN': 91.0, 'HMP': 56.0, 'AJP': 62.0, 'Right': 31.0, 'Left':70, 'Centre':51},
            9: {'LP': 41.0, 'NP': 31.0, 'GRN': 92.0, 'HMP': 63.0, 'Centre': 70.0, 'Right': 39.0, 'Left': 75},
            11: {'LP': 34.0, 'GRN': 91.0, 'HMP': 56.0, 'AJP': 62.0, 'Right': 31.0, 'Left':70, 'Centre':51, 'ALP':90}
        }
    else:
        UPDATED_FLOWS_ELECTION = {}
    # Extracted from non_classic divisions, to approximate how many electorates' data sources are available. 
    if election_year == 2026:
        N_SOURCES = {
            0: 115, 1: 14, 2: 8, 3: 9, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0
        }
    elif election_year == 2022:
        N_SOURCES = { 
            0: 124, 1: 15, 2: 2, 3: 6, 4: 0, 5: 3, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0
        }
    elif election_year == 2018:
        N_SOURCES = { 
            0: 135, 1: 6, 2: 2, 3: 3, 4: 0, 5: 3, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0
        }
    elif election_year == 2014:
        N_SOURCES = { 
            0: 139, 1: 2, 2: 1, 3: 3, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0
        }
    elif election_year == 2010:
        N_SOURCES = { 
            0: 142, 1: 3, 2: 1, 3: 3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0
        }


    EXTERNAL_ROWS = {4, 6, 9, 11} if election_year == 2026 else {}

    SIGMA_JOINT_BASE = 0.045
    EXTERNAL_UNCERTAINTY_PENALTY = 1.25

    def compute_statewide_average_flows(tcp_dict: dict, party_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the statewide average TCP flow for each minor party.
        Calculates aggregated 'Left', 'Right', and 'Centre' columns.
        """
        # 1. Stack all electorate DataFrames and average them (ignores NaNs automatically)
        all_dfs = pd.concat([df.rename(columns=lambda c: 'IND' if str(c).startswith('IND') else c).T.groupby(level=0).mean().T for df in tcp_dict.values()])
        avg_df = all_dfs.groupby(all_dfs.index).mean()
        
        # 2. Create the Ideology columns
        bloc_parties = {'Left': [], 'Right': [], 'Centre': []}
        for col in avg_df.columns:
            if str(col).startswith('IND'):
                bloc = 'Centre'
            else:
                # Assumes get_party_bloc is defined
                bloc = get_party_bloc(col, party_df) 
                
            if bloc in bloc_parties:
                bloc_parties[bloc].append(col)
                
        # 3. Calculate mean across the row for each bloc (ignoring NaNs)
        for bloc, parties in bloc_parties.items():
            if parties:
                # Average the parties belonging to this bloc
                avg_df[bloc] = avg_df[parties].mean(axis=1)
            else:
                avg_df[bloc] = np.nan
                
        return avg_df
    
    def apply_tcp_adjustments(
        tcp_dict: dict, 
        avg_df: pd.DataFrame, 
        updates: dict, 
        party_df: pd.DataFrame
    ) -> dict:
        """
        Computes Deltas from the statewide average and shifts local electorates. Applies bloc-level shifts to all parties inside that bloc.
        """
        adjusted_dict = copy.deepcopy(tcp_dict)
        
        # Compute the Deltas
        # delta_matrix[row][party_or_bloc] = shift_value
        delta_matrix = {row: {} for row in updates.keys()}
        
        for row_idx, target_flows in updates.items():
            for col, target_val in target_flows.items():
                base_val = avg_df.loc[row_idx, col]
                if pd.notna(base_val):
                    delta_matrix[row_idx][col] = target_val - base_val

        # Apply Deltas to all electorates
        for div, df in adjusted_dict.items():
            for row_idx, deltas in delta_matrix.items():
                for col_name, delta_val in deltas.items():
                    
                    # Check if it's a generic bloc
                    if col_name in ['Left', 'Right', 'Centre']:
                        for party_col in df.columns:
                            p_bloc = 'Centre' if str(party_col).startswith('IND') else get_party_bloc(party_col, party_df)
                            
                            # Apply bloc delta ONLY IF the party doesn't have a specific target in this row
                            if p_bloc == col_name and party_col not in deltas:
                                df.loc[row_idx, party_col] = np.clip(df.loc[row_idx, party_col] + delta_val, 0.5, 99.5)
                    
                    # Otherwise apply the specific party delta
                    else:
                        if col_name in df.columns:
                            df.loc[row_idx, col_name] = np.clip(df.loc[row_idx, col_name] + delta_val, 0.5, 99.5)
                            
        return adjusted_dict
    
    def build_joint_noise_vector(
        n_sources: dict, 
        external_rows: set, 
        base_sigma: float, 
        ext_penalty: float
    ) -> np.ndarray:
        """
        Builds the 15-element sigma array for the TCP combinations. Applies small-sample expansion and external data penalties.
        """
        sigmas = np.zeros(15)
        
        for i in range(15):
            n = n_sources.get(i, 0)
            
            # Enforce minimum Phantom N for completely unprecedented matchups (N=0)
            eff_n = max(float(n), 0.5) 
            
            # Base expansion: sqrt((N+1)/N)
            multiplier = np.sqrt((eff_n + 1.0) / eff_n)
            
            # External Penalty Stack
            if i in external_rows:
                multiplier *= ext_penalty
                
            sigmas[i] = base_sigma * multiplier
            
        return sigmas

    # 1. Pre-process the historical data
    avg_tcp_df = compute_statewide_average_flows(TCP_state_dict, party_df)

    # 2. Shift the baselines
    adjusted_TCP_state_dict = apply_tcp_adjustments(TCP_state_dict, avg_tcp_df, UPDATED_FLOWS_ELECTION, party_df)

    # 3. Build the noise matrix
    row_joint_sigmas = build_joint_noise_vector(N_SOURCES, EXTERNAL_ROWS, SIGMA_JOINT_BASE, EXTERNAL_UNCERTAINTY_PENALTY)

    return adjusted_TCP_state_dict, row_joint_sigmas, avg_tcp_df

def get_tcp_category(cand: str, party_df: pd.DataFrame) -> str:
    """Helper to map INDs correctly for the TCP Combination Index."""
    if get_party_bloc(cand, party_df) == 'Centre':
        return 'IND'
    return get_party_bloc(cand, party_df)

def vectorised_alr_shift_to_LC(
    electorate: str,
    LC_FPs_df: pd.DataFrame,
    statewide_alr_array: np.ndarray, # Shape: (1000, 4) -> ['ALP', 'GRN', 'ON', 'OTH']
    local_alr_array: np.ndarray,     # Shape: (1000, 4) -> ['ALP', 'GRN', 'ON', 'OTH']
    data_year: int
) -> pd.DataFrame:
    """
    Vectorised ALR shift of LC votes for (1000) simulations using pre-generated random CovM arrays.
    Re-spreads shifted bucket shares proportionally back to the original LC candidates.
    """
        
    n_sims = statewide_alr_array.shape[0]

    # Combine Statewide and Local Shifts
    total_alr_shift_matrix = statewide_alr_array + local_alr_array # (1000 x 4) matrix of the total ALR shift
    shift_cols = ALR_COLS     # Assuming standard order: ALP, GRN, ON, OTH (removing ON pre-2022)
    total_alr_shift_df = pd.DataFrame(total_alr_shift_matrix, columns=shift_cols)
    
    # group lc into the 5 ALR buckets
    base_lc = LC_FPs_df[electorate].dropna()
    grouped_base = {b: 0.0 for b in ['COAL'] + list(shift_cols)} # generalised for pre-2022 as well
    for p, v in base_lc.items():
        grouped_base[get_poll_bucket(p, data_year)] += v
        
    for b in grouped_base:
        if grouped_base[b] < 1e-6:
            import pdb; pdb.set_trace()
            grouped_base[b] = 1e-6
            
    # Calculate internal proportions for OTH and COAL to perfectly re-spread later
    oth_parties = [p for p in base_lc.index if get_poll_bucket(p, data_year) == 'OTH']
    oth_internal_props = {p: base_lc[p] / grouped_base['OTH'] for p in oth_parties}

    coal_parties = [p for p in base_lc.index if get_poll_bucket(p, data_year) == 'COAL']
    coal_internal_props = {p: base_lc[p] / grouped_base['COAL'] for p in coal_parties}
    
    # Vectorised ALR across all simulations
    # Formula: log(P_i / P_coal) = log(Base_i / Base_coal) + Shift_i
    shifted_shares = pd.DataFrame(index=range(n_sims))
    shifted_shares['COAL'] = 1.0  # Base reference (no shift applied to the reference in ALR math)
    
    for b in shift_cols:
        base_ratio = grouped_base[b] / grouped_base['COAL']
        log_shifted = np.log(base_ratio) + total_alr_shift_df[b]
        shifted_shares[b] = np.exp(log_shifted)
        
    # Renormalize the 5 buckets so each row (simulation) sums to 1.0
    shifted_shares = shifted_shares.div(shifted_shares.sum(axis=1), axis=0)
    
    LC_shifted_df = pd.DataFrame(index=shifted_shares.index)
    
    # Unpack buckets back strictly to the original parties in base_lc
    for p in base_lc.index:
        bucket = get_poll_bucket(p, data_year)
        
        if bucket in ['ALP', 'GRN', 'ON']:
            LC_shifted_df[p] = shifted_shares[bucket]
            
        elif bucket == 'COAL':
            LC_shifted_df[p] = shifted_shares['COAL'] * coal_internal_props[p]
            
        elif bucket == 'OTH':
            LC_shifted_df[p] = shifted_shares['OTH'] * oth_internal_props[p]

    # Final cleanup renormalization to ensure perfectly conserved mass
    LC_shifted_df = LC_shifted_df.div(LC_shifted_df.sum(axis=1), axis=0)
    
    return LC_shifted_df

def process_4cp_and_liquidate_inds_old(
    electorate: str,
    fp_sims_df: pd.DataFrame,
    compositions_dict: dict,
    rng: np.random.Generator,
    ideology_map: dict = {},
    S_param: float = 100.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies Top 4 candidates per simulation. Liquidates non-4CP INDs using Dirichlet-spread compositions, distributing proportionally to the remaining valid candidates.
            
    Returns:
        - updated_fp_df: (N x C) DataFrame with liquidated INDs removed and votes distributed.
        - top4_df: (N x 4) DataFrame of candidate strings representing the 4CP for each simulation.
    """
 
    N, C = fp_sims_df.shape
    cands = np.array(fp_sims_df.columns)
    fp_vals = fp_sims_df.values.copy()
    
    # 1. Identify Top 4 candidates per simulation
    rank_indices = np.argsort(-fp_vals, axis=1) # use -fp_vals for descending order
    top4_indices = rank_indices[:, :4]
    top4_cands = cands[top4_indices]
    top4_df = pd.DataFrame(
        top4_cands, 
        columns=['Rank_1', 'Rank_2', 'Rank_3', 'Rank_4'], 
        index=fp_sims_df.index
    )
    
    # Pre-map all candidates to their structural buckets
    cand_buckets = np.array([ideology_map.get(c, 'Centre') for c in cands])
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    
    # 2. Liquidate non-4CP INDs
    for i in range(N):
        sim_top4_idx = top4_indices[i]
        
        # Find INDs that are NOT in the Top 4
        for j, cand in enumerate(cands):
            if (cand.startswith('IND') or cand.startswith('TWP')) and j not in sim_top4_idx:
                ind_vote = fp_vals[i, j]
                if ind_vote <= 0: continue
                
                # Zero out the liquidated IND
                fp_vals[i, j] = 0.0
                
                # Fetch baseline composition, fallback to equal split if missing
                comp = compositions_dict[(electorate, cand)]
                #comp = compositions_dict.get((electorate, cand), {'ALP': 0.2, 'COAL': 0.2, 'Left': 0.2, 'Right': 0.2, 'Centre': 0.2})

                # Prepare Dirichlet alpha vector (mu * S)
                mu = np.array([comp.get(b, 0.0) for b in BLOCS])
                alpha = np.maximum(mu * S_param, 1e-6) # Protect against alpha=0
                
                # Draw stochastic proportions
                drawn_props = rng.dirichlet(alpha)
                
                # Distribute the drawn mass to the remaining active candidates
                for b_idx, bucket_name in enumerate(BLOCS):
                    bucket_mass = ind_vote * drawn_props[b_idx]
                    if bucket_mass <= 0: continue
                    
                    # Valid candidates = belong to the target bucket AND haven't been liquidated
                    valid_mask = (cand_buckets == bucket_name) & (fp_vals[i] > 0)
                    
                    # Fallback: If no candidate exists in the target bucket (e.g., ALP didn't run), spill the votes proportionally to ALL remaining candidates.
                    if not np.any(valid_mask):
                        valid_mask = fp_vals[i] > 0
                        
                    # Proportional distribution within the valid mask
                    weights = fp_vals[i, valid_mask]
                    weight_sum = np.sum(weights)
                    
                    if weight_sum > 0:
                        alloc = bucket_mass * (weights / weight_sum)
                    else:
                        alloc = bucket_mass / np.sum(valid_mask)
                        
                    fp_vals[i, valid_mask] += alloc

    # Reconstruct the updated DataFrame
    updated_fp_df = pd.DataFrame(fp_vals, columns=cands, index=fp_sims_df.index)
    
    return updated_fp_df, top4_df

def process_4cp_and_liquidate_inds(
    electorate: str,
    fp_sims_df: pd.DataFrame,
    compositions_dict: dict,
    rng: np.random.Generator,
    ideology_map: dict = {},
    S_param: float = 100.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies Top 4 candidates per simulation.
    Groups simulations by their eliminated IND set and vectorises Dirichlet draws
    and proportional bucket allocations across each group.
    """
    N, C = fp_sims_df.shape
    cands = np.array(fp_sims_df.columns)
    fp_vals = fp_sims_df.to_numpy().copy()

    # 1. Identify Top 4 candidates per simulation
    rank_indices = np.argsort(-fp_vals, axis=1)
    top4_indices = rank_indices[:, :4]
    top4_df = pd.DataFrame(
        cands[top4_indices],
        columns=['Rank_1', 'Rank_2', 'Rank_3', 'Rank_4'],
        index=fp_sims_df.index
    )

    # 2. Pre-filter IND/TWP column indices; exit immediately if none exist
    ind_col_indices = np.array([
        j for j, c in enumerate(cands) if str(c).startswith('IND') or str(c).startswith('TWP')
    ], dtype=int)

    if len(ind_col_indices) == 0:
        return fp_sims_df.copy(), top4_df

    # 3. Vectorised check: which IND columns are NOT in top4_indices for each row?
    # Shape: (N, 4, 1) == (1, 1, num_inds) -> .any(axis=1) gives (N, num_inds)
    in_top4_mask = (top4_indices[:, :, None] == ind_col_indices[None, None, :]).any(axis=1)
    is_elim_mask = (~in_top4_mask) & (fp_vals[:, ind_col_indices] > 0)

    # If no IND was eliminated in any simulation, exit immediately
    if not np.any(is_elim_mask):
        return fp_sims_df.copy(), top4_df

    # 4. Group simulations by the exact tuple of eliminated IND column indices
    elim_groups = defaultdict(list)
    for sim_idx in range(N):
        if np.any(is_elim_mask[sim_idx]):
            elim_tuple = tuple(ind_col_indices[is_elim_mask[sim_idx]])
            elim_groups[elim_tuple].append(sim_idx)

    cand_buckets = np.array([ideology_map.get(c, 'Centre') for c in cands])
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']

    # 5. Process each eliminated-IND group in vectorised chunks
    for elim_cols_tuple, sim_indices in elim_groups.items():
        rows = np.array(sim_indices)
        M = len(rows)

        for j in elim_cols_tuple:
            cand = cands[j]
            ind_vote = fp_vals[rows, j].copy()  # Shape: (M,)
            fp_vals[rows, j] = 0.0              # Zero out liquidated IND across the group

            # Prepare Dirichlet alpha vector (mu * S) and draw (M x 5) matrix at once
            comp = compositions_dict[(electorate, cand)]
            mu = np.array([comp.get(b, 0.0) for b in BLOCS])
            alpha = np.maximum(mu * S_param, 1e-6)
            drawn_props = rng.dirichlet(alpha, size=M)  # Shape: (M, 5)

            # Identify active surviving columns in this group
            active_col_mask = (fp_vals[rows] > 0).any(axis=0)

            for b_idx, bucket_name in enumerate(BLOCS):
                bucket_mass = ind_vote * drawn_props[:, b_idx]  # Shape: (M,)
                if np.all(bucket_mass <= 0):
                    continue

                valid_mask = (cand_buckets == bucket_name) & active_col_mask
                if not np.any(valid_mask):
                    valid_mask = active_col_mask

                valid_cols = np.where(valid_mask)[0]
                weights = fp_vals[np.ix_(rows, valid_cols)]      # Shape: (M, V)
                weight_sum = weights.sum(axis=1, keepdims=True)  # Shape: (M, 1)

                shares = np.where(
                    weight_sum > 0,
                    weights / np.maximum(weight_sum, 1e-12),
                    1.0 / len(valid_cols)
                )
                fp_vals[np.ix_(rows, valid_cols)] += bucket_mass[:, None] * shares

    updated_fp_df = pd.DataFrame(fp_vals, columns=cands, index=fp_sims_df.index)
    return updated_fp_df, top4_df

def get_lc_XCP_flow_thetas(
    top_cands: List[str],
    eliminated_cands: List[str],
    lc_series: pd.Series,
    emp_fp: pd.Series,
    party_df: pd.DataFrame,
    model: Any
) -> Dict[str, float]:
    """
    Core wrapper for Procedure A (Anchored LC Flow).
    Evaluates the structural logit for Menu A (Before Elimination) and Menu B (After).
    Converts structural Deltas into empirical flow Thetas.
    """
    # Define the two states
    menu_A = top_cands + eliminated_cands
    menu_B = top_cands

    # ORPHAN CAPTURE: Give synthetic baseline to new LA parties
    zero_lc_entrants = [c for c in menu_A if lc_series.get(c, 0.0) <= 1e-9]
    
    if zero_lc_entrants:
        # Pass the empirical FP votes as the current C0 state for orphan capture
        mod_lc = apply_LC_orphan_capture(cands_to_add=zero_lc_entrants, current_c0=emp_fp.to_dict(), lc_series=lc_series, party_df=party_df)
    else:
        mod_lc = lc_series.copy()
    
    alpha, beta, gamma = model.unpack_params(model.theta)
    
    # 1. Evaluate State A (The full active field before elimination)
    ms_A, cap_A, ctb_A, cd_A, lc_mass_A = _build_state_vars(menu_A, mod_lc, emp_fp.to_dict(), party_df)
    s_hat_A = model.evaluate_state_allocation(lc_mass_A, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)
    
    # 2. Evaluate State B (The restricted field after elimination)
    # The structural mass of eliminated_cands will automatically re-route to top_cands
    ms_B, cap_B, ctb_B, cd_B, lc_mass_B = _build_state_vars(menu_B, mod_lc, emp_fp.to_dict(), party_df)
    s_hat_B = model.evaluate_state_allocation(lc_mass_B, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)
    
    # 3. Calculate Deltas and normalize into empirical Thetas
    # Delta = How much structural mass did this Top N candidate gain?
    deltas = {c: max(0.0, s_hat_B.get(c, 0.0) - s_hat_A.get(c, 0.0)) for c in top_cands}
    total_delta = sum(deltas.values())
    
    # Failsafe: if the structural model predicts zero flow, distribute evenly
    if total_delta <= 1e-9:
        import pdb; pdb.set_trace()
        thetas = {c: 1.0 / len(top_cands) for c in top_cands}
    else:
        thetas = {c: d / total_delta for c, d in deltas.items()}
        
    return thetas

def execute_theta_steal(
    ind_cand: str,
    top_cands_non_ind: List[str],
    base_thetas: Dict[str, float],
    defection_profile: dict,
    party_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Procedure B Math: Generates a 4-Theta composition.
    Extracts a portion of each 3CP party's theoretical flow based on the IND's defection profile, and aggregates the stolen fractions into a Theta for the IND.
    """
    final_thetas = {ind_cand: 0.0}
    
    for c in top_cands_non_ind:
        # 1. Determine the steal rate (extraction fraction) for this specific 3CP party
        # Exact party matches (e.g., 'ALP') take precedence over ideological blocs (e.g., 'Left')
        if c in defection_profile:
            steal_rate = defection_profile[c]
        else:
            bloc = get_party_bloc(c, party_df)
            steal_rate = defection_profile.get(bloc, 0.0)
            
        # Ensure rate bounds (between 0% and 100% extracted)
        steal_rate = max(0.0, min(1.0, steal_rate))
        
        # 2. Extract from the base theta
        base_theta = base_thetas[c]
        stolen_theta = base_theta * steal_rate
        
        # 3. Form the final 4-Theta Composition
        final_thetas[c] = base_theta - stolen_theta
        final_thetas[ind_cand] += stolen_theta
        
    return final_thetas

def get_tcp_thetas(
    non_ind_parties: List[str],
    eliminated_cands: List[str],
    original_fp: pd.Series,  # Use original primary votes to weight the historical flows
    tcp_proportions_df: pd.DataFrame,
    party_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Looks up the historical TCP flows from eliminated minors to the 2 non-IND parties. Aggregates these per-party flows into a single base Theta distribution. Can be reused for the final 2CP stage.
    """
    p1, p2 = non_ind_parties
    
    # 1. Get Ideology Categories (using existing get_party_bloc)
    cat1 = get_party_bloc(p1, party_df)
    cat2 = get_party_bloc(p2, party_df)
    
    # 2. Sort alphabetically to match the tuple index
    cats_with_parties = sorted([(cat1, p1), (cat2, p2)], key=lambda x: x[0])
    top2_category = (cats_with_parties[0][0], cats_with_parties[1][0])
    first_party = cats_with_parties[0][1]
    
    # 3. Lookup the row index (Fallback to 0 'ALP/COAL' if some weird combo occurs)
    row_index = TCP_COMBINATION_INDEX.get(top2_category, 0)
    
    # 4. Fetch the array of transfers to the FIRST category
    transfer_to_first = tcp_proportions_df.iloc[row_index]/100
    
    # 5. Calculate aggregated flow volumes based on original FPs
    flow_p1 = 0.0
    flow_p2 = 0.0
    
    for c in eliminated_cands:
        # Use original FP to perfectly weight the historical flow
        vol = original_fp.get(c, 0.0) 
        # Fallback to 50/50 if a phantom party isn't in the TCP matrix
        prop_to_first = transfer_to_first.get(c, 0.5) 
        
        if p1 == first_party:
            flow_p1 += vol * prop_to_first
            flow_p2 += vol * (1.0 - prop_to_first)
        else:
            flow_p1 += vol * (1.0 - prop_to_first)
            flow_p2 += vol * prop_to_first
            
    # 6. Normalize into Thetas
    total_flow = flow_p1 + flow_p2
    if total_flow > 1e-9:
        return {p1: flow_p1 / total_flow, p2: flow_p2 / total_flow}
    else:
        return {p1: 0.5, p2: 0.5}

def execute_multi_IND_theta_steal(
    ind_cands: List[str],
    non_ind_cands: List[str],
    base_thetas: Dict[str, float],
    defection_profiles: Dict[str, dict],
    party_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Procedure C Math: Handles 2+ INDs stealing from the base TCP thetas simultaneously.
    Applies a discount factor to prevent over-stealing (e.g., negative remaining flow).
    """
    final_thetas = {c: base_thetas.get(c, 0.0) for c in non_ind_cands}
    for ind in ind_cands:
        final_thetas[ind] = 0.0
        
    # Discount factor: 0.6 if 2 INDs, otherwise inversely proportional
    discount = 0.6 if len(ind_cands) == 2 else 1
    
    for c in non_ind_cands:
        bloc = get_party_bloc(c, party_df)
        total_raw_steal = 0.0
        raw_steals = {}
        
        # Calculate what each IND wants to steal
        for ind in ind_cands:
            profile = defection_profiles[ind]
            rate = profile.get(c, profile.get(bloc, 0.0)) * discount
            rate = max(0.0, min(1.0, rate))
            raw_steals[ind] = rate
            total_raw_steal += rate
            
        # Hard cap the combined steal rate at 1.0 (100% of the theta)
        scale_down = 1.0 if total_raw_steal <= 1.0 else (1.0 / total_raw_steal)
        
        # Execute the steal
        for ind in ind_cands:
            actual_steal_rate = raw_steals[ind] * scale_down
            stolen_theta = base_thetas[c] * actual_steal_rate
            
            final_thetas[c] -= stolen_theta
            final_thetas[ind] += stolen_theta
            
    return final_thetas

def process_preference_distribution_old(
    electorate: str,
    original_fp_df: pd.DataFrame,
    updated_fp_df: pd.DataFrame, 
    top_n_df: pd.DataFrame,       
    LC_shifted_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    specific_defection_dict: dict,
    fallback_defection_profile: dict,
    TCP_state_dict: dict
) -> pd.DataFrame:
    """
    Distributes eliminated votes to the Top N to find intermediate totals.
    Eliminates the weakest candidate. 
    Returns: next_top_df (shape: N_sims x (N-1))
    """
    N_sims = len(updated_fp_df)
    cands = updated_fp_df.columns
    
    # Store the surviving candidates for the next round
    next_top_cands = []
    
    # Simulation Stratification (The Masks)
    ind_mask = top_n_df.map(lambda c: str(c).startswith('IND') or str(c).startswith('TWP'))
    ind_counts = ind_mask.sum(axis=1)

    tcp_proportions_df = TCP_state_dict[electorate]

    # use means for bases to calculate thetas - FIX: pre-calculate before loop
    
    
    for sim_idx in range(N_sims):
        top_cands = top_n_df.loc[sim_idx].tolist()
        emp_fp = updated_fp_df.loc[sim_idx]
        orig_fp = original_fp_df.loc[sim_idx]
        
        # Eliminated candidates = anyone with votes > 0 who isn't in Top N
        all_eliminated = [c for c in cands if emp_fp[c] > 1e-9 and c not in top_cands]
        elim_minors = [c for c in all_eliminated if not (str(c).startswith('IND') or str(c).startswith('TWP'))]
        elim_inds = [c for c in all_eliminated if (str(c).startswith('IND') or str(c).startswith('TWP'))] # for IND eliminated at 4CP, apply their own defection profile
        
        adjusted_fp = emp_fp.copy() if elim_inds else emp_fp
        
        if elim_inds:
            for elim_ind in elim_inds:
                ind_vol = adjusted_fp[elim_ind]
                if ind_vol <= 1e-9: continue
                
                profile = specific_defection_dict.get((electorate, elim_ind), fallback_defection_profile)
                weights = {}
                for c in top_cands:
                    lam = max(0.0, min(0.999, profile.get(get_party_bloc(c, party_df), 0.0)))
                    weights[c] = adjusted_fp[c] * (lam / (1.0 - lam))
                    
                sum_weights = sum(weights.values())
                for c in top_cands:
                    adjusted_fp[c] += ind_vol * (weights[c] / sum_weights) if sum_weights > 1e-9 else ind_vol / len(top_cands)
                    
                adjusted_fp[elim_ind] = 0.0
        
        elim_minor_pool = sum(adjusted_fp[c] for c in elim_minors)
        
        # Track simulated totals for this specific step
        totals = {c: adjusted_fp[c] for c in top_cands}
        
        if elim_minor_pool > 1e-9:
            num_inds = ind_counts.loc[sim_idx]
            lc_series = LC_shifted_df.loc[sim_idx]
            
            if num_inds == 0:
                # ---------------------------------------------------------
                # PROCEDURE A: 0 INDs (Anchored LC Model)
                # ---------------------------------------------------------
                thetas = get_lc_XCP_flow_thetas(top_cands, elim_minors, lc_series, adjusted_fp, party_df, model)
                for c in top_cands:
                    totals[c] += elim_minor_pool * thetas[c]
                    
            elif num_inds == 1:
                # ---------------------------------------------------------
                # PROCEDURE B: 1 IND (Scalar Steal)
                # ---------------------------------------------------------
                ind_cand = [c for c in top_cands if str(c).startswith('IND') or str(c).startswith('TWP')][0]
                top_cands_non_ind = [c for c in top_cands if c != ind_cand]
                
                # 1. Get base non-IND flows
                base_thetas = get_lc_XCP_flow_thetas(top_cands_non_ind, elim_minors, lc_series, adjusted_fp, party_df, model)
                
                # 2. Get Defection Profile
                def_profile = specific_defection_dict.get((electorate, ind_cand), fallback_defection_profile)
                
                # 3. Apply the Scalar Steal
                thetas = execute_theta_steal(ind_cand, top_cands_non_ind, base_thetas, def_profile, party_df)
                
                for c in top_cands:
                    totals[c] += elim_minor_pool * thetas[c]
                    
            elif num_inds >= 2 and (len(top_cands) - num_inds) == 2:
                # PROCEDURE C: 2+ INDs, exactly 2 Non-INDs (Historical TCP Steal)

                ind_cands = [c for c in top_cands if str(c).startswith('IND') or str(c).startswith('TWP')]
                non_ind_cands = [c for c in top_cands if c not in ind_cands]
                
                # 1. Base historical thetas using original primary votes
                base_thetas = get_tcp_thetas(
                    non_ind_cands, elim_minors, orig_fp, tcp_proportions_df, party_df
                )

                # 2. Defection profiles for both INDs
                def_profiles = {
                    i: specific_defection_dict.get((electorate, i), fallback_defection_profile) 
                    for i in ind_cands
                }
                
                # 3. Multi-Theta Steal with 0.6 discount
                final_thetas = execute_multi_IND_theta_steal(
                    ind_cands, non_ind_cands, base_thetas, def_profiles, party_df
                )
                for c in top_cands: totals[c] += elim_minor_pool * final_thetas[c]
                

            else:
                
                #import pdb; pdb.set_trace()

                # FALLBACK (e.g., 3CP with 2 INDs -> 1 Non-IND) - split votes proportionally to initial shares - REFINE

                survivor_sum = sum(totals.values())
                for c in top_cands:
                    if survivor_sum > 1e-9: 
                        totals[c] += elim_minor_pool * (totals[c] / survivor_sum)
                    else: 
                        # Failsafe if everyone is exactly at 0
                        totals[c] += elim_minor_pool / len(top_cands)
                
        # Determine who gets eliminated (lowest vote in `totals`)
        weakest_cand = min(totals, key=totals.get)
        survivors = [c for c in top_cands if c != weakest_cand]
        
        next_top_cands.append(survivors)
    
    # Convert output back into a DataFrame (1000 x (N-1))
    col_names = [f'Rank_{i+1}' for i in range(len(next_top_cands[0]))]
    next_top_df = pd.DataFrame(next_top_cands, columns=col_names, index=updated_fp_df.index)
    
    return next_top_df

def process_preference_distribution(
    electorate: str,
    original_fp_df: pd.DataFrame,
    updated_fp_df: pd.DataFrame, 
    top_n_df: pd.DataFrame,       
    LC_shifted_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    specific_defection_dict: dict,
    fallback_defection_profile: dict,
    TCP_state_dict: dict
) -> pd.DataFrame:
    """
    Distributes eliminated votes to the Top N to find intermediate totals.
    Eliminates the weakest candidate. 
    Returns: next_top_df (shape: N_sims x (N-1))
    Now uses average LC shifted df to allow grouped calculation of thetas. Imprecision is acceptable due to it determining the order, not the XCP values.
    """
    # use means for bases to calculate thetas - FIX: pre-calculate before loop
    mean_lc_series = LC_shifted_df.mean(axis=0)
    mean_emp_fp = updated_fp_df.mean(axis=0)
    mean_orig_fp = original_fp_df.mean(axis=0) # for Procedure C

    N_sims, K = top_n_df.shape
    cands = list(updated_fp_df.columns)
    cand_to_col = {c: i for i, c in enumerate(cands)}
    is_ind_map = {c: (str(c).startswith('IND') or str(c).startswith('TWP')) for c in cands}

    fp_mat = updated_fp_df.to_numpy() # Shape: (N_sims, C)
    next_top_mat = np.empty((N_sims, K - 1), dtype=object)

    # Group simulation indices by exact sorted Top-K candidate set (includes specific INDs)
    groups = defaultdict(list)
    for sim_idx, row in enumerate(top_n_df.to_numpy()):
        groups[tuple(sorted(row))].append(sim_idx)

    # Secondary cache in case two different IND groups share the same non-IND parties
    base_theta_cache = {}

    print(f"{electorate}| Number of {K}CP combinations:", len(groups))

    tcp_proportions_df = TCP_state_dict[electorate]

   
    for top_tuple, sim_indices in groups.items():
        rows = np.array(sim_indices)
        top_cands = list(top_tuple) # Fixed length K (4 or 3)
        top_cols = [cand_to_col[c] for c in top_cands]

        elim_minors = [c for c in cands if c not in top_tuple and not is_ind_map[c]]
        elim_minor_cols = [cand_to_col[c] for c in elim_minors]
        elim_inds = [c for c in cands if c not in top_tuple and is_ind_map[c]]

        # Slice ONLY the K columns for the surviving candidates in this group: shape (M, K)
        totals_mat = fp_mat[np.ix_(rows, top_cols)].copy()

        # 1. Vectorized 4th-place IND redistribution (only triggers in top2 stage if an IND was 4th)
        for elim_ind in elim_inds:
            ind_vol = fp_mat[rows, cand_to_col[elim_ind]] # Shape: (M,)
            if np.all(ind_vol <= 1e-9):
                continue
            profile = specific_defection_dict.get((electorate, elim_ind), fallback_defection_profile)
            odds_vec = np.array([
                max(0.0, min(0.999, profile.get(get_party_bloc(c, party_df), 0.0))) for c in top_cands
            ])
            odds_vec = odds_vec / (1.0 - odds_vec) # Shape: (K,)
            
            weights = totals_mat * odds_vec[None, :] # Shape: (M, K)
            sum_w = weights.sum(axis=1, keepdims=True)
            shares = np.where(sum_w > 1e-9, weights / np.maximum(sum_w, 1e-9), 1.0 / K)
            totals_mat += ind_vol[:, None] * shares

        # 2. Vectorized eliminated minor pool: shape (M,)
        elim_minor_pool = fp_mat[np.ix_(rows, elim_minor_cols)].sum(axis=1) if elim_minor_cols else np.zeros(len(rows))

        # FAST-PATH C: Only compute & apply party flow thetas if there are actually eliminated minor-party votes to distribute! (Skips when only INDs were eliminated, e.g. ALP, LP, GRN, IND1, IND2)
        if elim_minors and np.any(elim_minor_pool > 1e-9):

            # 3. Compute Thetas ONCE for the entire group
            ind_cands = [c for c in top_cands if is_ind_map[c]]
            non_ind_cands = [c for c in top_cands if not is_ind_map[c]]
            num_inds = len(ind_cands)

            if num_inds == 0:
                cache_key = (tuple(top_cands), tuple(elim_minors))
                if cache_key not in base_theta_cache:
                    base_theta_cache[cache_key] = get_lc_XCP_flow_thetas(
                        top_cands, elim_minors, mean_lc_series, mean_emp_fp, party_df, model
                    )
                thetas = base_theta_cache[cache_key]

            elif num_inds == 1:
                ind_cand = ind_cands[0]
                cache_key = (tuple(non_ind_cands), tuple(elim_minors))
                if cache_key not in base_theta_cache:
                    base_theta_cache[cache_key] = get_lc_XCP_flow_thetas(
                        non_ind_cands, elim_minors, mean_lc_series, mean_emp_fp, party_df, model
                    )
                base_thetas = base_theta_cache[cache_key]
                def_profile = specific_defection_dict.get((electorate, ind_cand), fallback_defection_profile)
                thetas = execute_theta_steal(ind_cand, non_ind_cands, base_thetas, def_profile, party_df)

            elif num_inds >= 2 and len(non_ind_cands) == 2:
                base_thetas = get_tcp_thetas(non_ind_cands, elim_minors, mean_orig_fp, tcp_proportions_df, party_df)
                def_profiles = {i: specific_defection_dict.get((electorate, i), fallback_defection_profile) for i in ind_cands}
                thetas = execute_multi_IND_theta_steal(ind_cands, non_ind_cands, base_thetas, def_profiles, party_df)

            else:
                thetas = None

            # 4. Broadcast `thetas` across all M simulations in this group. If theta is none, use HACK: proportional split won't change order
            if thetas is not None:
                theta_vec = np.array([thetas[c] for c in top_cands]) # Aligned with top_cands!
                totals_mat += elim_minor_pool[:, None] * theta_vec[None, :]

        # 5. Find weakest candidate per row and assign the K-1 survivors
        weakest_col_idx = np.argmin(totals_mat, axis=1) # Shape: (M,), values in 0..K-1
        for w_idx in range(K):
            mask = (weakest_col_idx == w_idx)
            if np.any(mask):
                survivors = [c for i, c in enumerate(top_cands) if i != w_idx]
                next_top_mat[rows[mask]] = survivors

    col_names = [f'Rank_{i+1}' for i in range(K - 1)]
    return pd.DataFrame(next_top_mat, columns=col_names, index=updated_fp_df.index)


def resolve_final_2cp_old(
    original_fp_df: pd.DataFrame,
    top2_df: pd.DataFrame,
    tcp_proportions_df: pd.DataFrame,
    party_df: pd.DataFrame,
    joint_noise: np.ndarray = None,
    rng: np.random.Generator = None,
    sigma_ind: float = 0.03,
    simulate: bool = True
) -> pd.DataFrame:
    """
    Distributes all eliminated original FP votes to the final 2CP pair
    using stochastic empirical historical TCP flows, bounded between 0 and 1.
    """
    N_sims = len(original_fp_df)
    cands = original_fp_df.columns
    
    # Fallback to zero-matrix if running deterministically
    if joint_noise is None or not simulate:
        joint_noise = np.zeros((N_sims, 15))
        
    results = []
    
    for sim_idx in range(N_sims):
        top2 = top2_df.loc[sim_idx].tolist()
        p1, p2 = top2
        orig_fp = original_fp_df.loc[sim_idx]
        
        # 1. Map to TCP Combination Index
        cat1, cat2 = get_tcp_category(p1, party_df), get_tcp_category(p2, party_df)
        cats_with_parties = sorted([(cat1, p1), (cat2, p2)], key=lambda x: x[0])
        
        top2_category = (cats_with_parties[0][0], cats_with_parties[1][0])
        first_party = cats_with_parties[0][1]
        
        # 2. Fetch Flow Vector
        row_index = TCP_COMBINATION_INDEX.get(top2_category, 0)
        transfer_to_first = tcp_proportions_df.iloc[row_index] / 100.0  # Normalize to 1.0
        
        # Extract the global joint shock for this specific TCP row and simulation
        sim_joint_shock = joint_noise[sim_idx, row_index]
        
        # 3. Calculate Final Flows
        eliminated = [c for c in cands if c not in top2 and orig_fp[c] > 1e-9]
        flow_p1, flow_p2 = 0.0, 0.0
        
        for c in eliminated:
            vol = orig_fp[c]
            base_prop = transfer_to_first.get(c, 0.5) 
            if pd.isna(base_prop):
                base_prop = 0.5
                
            # Draw local idiosyncratic noise for this specific candidate
            ind_shock = rng.normal(0, sigma_ind) if (simulate and rng is not None) else 0.0
            
            # 4. Apply shocks and rigidly bound between 0% and 100%
            prop_first = np.clip(base_prop + sim_joint_shock + ind_shock, 0.0, 1.0)
            
            if p1 == first_party:
                flow_p1 += vol * prop_first
                flow_p2 += vol * (1.0 - prop_first)
            else:
                flow_p1 += vol * (1.0 - prop_first)
                flow_p2 += vol * prop_first
                
        total_p1 = orig_fp[p1] + flow_p1
        total_p2 = orig_fp[p2] + flow_p2
        
        # 5. Declare Winner
        if total_p1 > total_p2:
            results.append({'Winner': p1, 'RunnerUp': p2, 'Winner_2CP': total_p1, 'RunnerUp_2CP': total_p2})
        else:
            results.append({'Winner': p2, 'RunnerUp': p1, 'Winner_2CP': total_p2, 'RunnerUp_2CP': total_p1})
            
    return pd.DataFrame(results, index=original_fp_df.index)

def resolve_final_2cp(
    original_fp_df: pd.DataFrame,
    top2_df: pd.DataFrame,
    tcp_proportions_df: pd.DataFrame,
    party_df: pd.DataFrame,
    joint_noise: np.ndarray = None,
    rng: np.random.Generator = None,
    sigma_ind: float = 0.03,
    simulate: bool = True
) -> pd.DataFrame:
    """
    Groups simulations by their final 2CP pair and vectorises the stochastic
    empirical TCP preference transfer and winner resolution per group.
    """
    N_sims = len(original_fp_df)
    cands = list(original_fp_df.columns)
    cand_to_col = {c: i for i, c in enumerate(cands)}
    orig_mat = original_fp_df.to_numpy()  # Shape: (N_sims, C)

    # Pre-map candidate TCP categories once per electorate
    tcp_cat_map = {c: get_tcp_category(c, party_df) for c in cands}

    # Fallback to zero-matrix if running deterministically
    if joint_noise is None or not simulate:
        joint_noise = np.zeros((N_sims, 15))
    # pre-draw fixed noise matrix
    if simulate and rng is not None:
        ind_shocks_mat = rng.normal(0, sigma_ind, size=(N_sims, len(cands)))
    else:
        ind_shocks_mat = np.zeros((N_sims, len(cands)))

    # Pre-allocate output arrays aligned with original_fp_df.index
    winners_arr = np.empty(N_sims, dtype=object)
    runners_arr = np.empty(N_sims, dtype=object)
    win_2cp_arr = np.empty(N_sims, dtype=float)
    run_2cp_arr = np.empty(N_sims, dtype=float)

    # 1. Group simulation indices by sorted 2CP candidate pair
    groups = defaultdict(list)
    for sim_idx, row in enumerate(top2_df.to_numpy()):
        groups[tuple(sorted(row))].append(sim_idx)

    # 2. Process each unique 2CP pairing in vectorised chunks
    for top2_pair, sim_indices in groups.items():
        rows = np.array(sim_indices)
        M = len(rows)
        p1, p2 = top2_pair

        # Map to TCP Combination Index once for the group
        cat1, cat2 = tcp_cat_map[p1], tcp_cat_map[p2]
        cats_with_parties = sorted([(cat1, p1), (cat2, p2)], key=lambda x: x[0])
        top2_category = (cats_with_parties[0][0], cats_with_parties[1][0])
        first_party = cats_with_parties[0][1]

        row_index = TCP_COMBINATION_INDEX.get(top2_category, 0)
        transfer_to_first = tcp_proportions_df.iloc[row_index] / 100.0

        # Identify eliminated candidates once for the group
        eliminated = [c for c in cands if c != p1 and c != p2]
        elim_cols = [cand_to_col[c] for c in eliminated]
        E = len(elim_cols)

        if E > 0:
            # Build 1D base proportion vector (shape: E,)
            base_props = np.array([
                0.5 if pd.isna(transfer_to_first.get(c, 0.5)) else float(transfer_to_first.get(c, 0.5))
                for c in eliminated
            ])

            # Extract joint shocks (shape: M x 1) and draw idiosyncratic shocks (shape: M x E)
            sim_joint_shocks = joint_noise[rows, row_index][:, None]
            ind_shocks = ind_shocks_mat[np.ix_(rows, elim_cols)]

            # Compute clipped flow rates (shape: M x E) and multiply by eliminated votes
            prop_first_mat = np.clip(base_props[None, :] + sim_joint_shocks + ind_shocks, 0.0, 1.0)
            elim_vols = orig_mat[np.ix_(rows, elim_cols)]  # Shape: (M, E)

            flow_first = (elim_vols * prop_first_mat).sum(axis=1)          # Shape: (M,)
            flow_second = (elim_vols * (1.0 - prop_first_mat)).sum(axis=1) # Shape: (M,)
        else:
            flow_first = np.zeros(M)
            flow_second = np.zeros(M)

        # 3. Add flows to primary votes for p1 and p2
        p1_fp = orig_mat[rows, cand_to_col[p1]]
        p2_fp = orig_mat[rows, cand_to_col[p2]]

        if p1 == first_party:
            total_p1 = p1_fp + flow_first
            total_p2 = p2_fp + flow_second
        else:
            total_p1 = p1_fp + flow_second
            total_p2 = p2_fp + flow_first

        # 4. Vectorised winner assignment into original simulation indices
        p1_wins = total_p1 > total_p2
        winners_arr[rows] = np.where(p1_wins, p1, p2)
        runners_arr[rows] = np.where(p1_wins, p2, p1)
        win_2cp_arr[rows] = np.where(p1_wins, total_p1, total_p2)
        run_2cp_arr[rows] = np.where(p1_wins, total_p2, total_p1)

    return pd.DataFrame({
        'Winner': winners_arr,
        'RunnerUp': runners_arr,
        'Winner_2CP': win_2cp_arr,
        'RunnerUp_2CP': run_2cp_arr
    }, index=original_fp_df.index)

def run_statewide_preference_simulation(
    FP_pred: dict,
    LC_FPs_df: pd.DataFrame,
    sim_statewide_alr: np.ndarray,
    sim_local_alr_dict: dict,
    party_df: pd.DataFrame,
    model: Any,
    compositions_dict: dict,
    specific_defection_dict: dict,
    fallback_defection_profile: dict,
    TCP_state_dict: dict,
    row_joint_sigmas: np.ndarray,
    sigma_ind: float,
    data_year: int,
    rng: np.random.Generator
):
    """
    Executes the full pipeline and constructs summary tracking dictionaries.
    """
    n_sims = sim_statewide_alr.shape[0]
    
    # 1. Initialize Tracking Structures
    per_electorate_winners = defaultdict(list)
    per_electorate_runners_up = defaultdict(list)
    per_electorate_tcp_pair = defaultdict(list)
    per_electorate_winning_tcp = defaultdict(list)
    
    per_simulation_winners = [[] for _ in range(n_sims)]

    # Generate the global (N_sims x 15) joint noise array before looping through electorates
    if rng is not None:
        global_joint_noise = rng.normal(0, row_joint_sigmas, size=(n_sims, 15))
    else:
        global_joint_noise = None

    st = time.time()
    
    # 2. Iterate through all electorates
    for div, fp_sims_df in FP_pred.items():

        num_cands = fp_sims_df.shape[1]

        # FAST-PATH A: Only 3 (or 2) candidates contesting in total: Skip LC shift, 4CP liquidation, 4CP->3CP, and 3CP->2CP!
        if num_cands <= 3:
            cands_arr = np.array(fp_sims_df.columns, dtype=object)
            top2_idx = np.argsort(-fp_sims_df.to_numpy(), axis=1)[:, :2]
            top2_df = pd.DataFrame(cands_arr[top2_idx], columns=['Rank_1', 'Rank_2'], index=fp_sims_df.index)
            print(f"{div} | Fast-path: {num_cands} candidates total -> straight to 2CP")
            st = time.time()

        else:
            # Step 1: ALR Shift to LC Baseline
            local_alr_array = sim_local_alr_dict.get(div, np.zeros_like(sim_statewide_alr))
            lc_shifted_df = vectorised_alr_shift_to_LC(div, LC_FPs_df, sim_statewide_alr, local_alr_array, data_year)
            
            # Step 2: Form 4CP & Liquidate minor INDs
            updated_fp_df, top4_df = process_4cp_and_liquidate_inds(div, fp_sims_df, compositions_dict, rng)
            print(div)
            print(time.time() - st, 'top4')
            st = time.time()

            # FAST-PATH B: Exactly 4 candidates contesting in total - Nobody is outside Top 4 -> read Top 3 straight off updated_fp_df
            if num_cands == 4:
                cands_arr = np.array(updated_fp_df.columns, dtype=object)
                top3_idx = np.argsort(-updated_fp_df.to_numpy(), axis=1)[:, :3]
                top3_df = pd.DataFrame(
                    cands_arr[top3_idx],
                    columns=['Rank_1', 'Rank_2', 'Rank_3'],
                    index=updated_fp_df.index
                )
            else:
                # Step 3: 4CP -> 3CP Elimination
                top3_df = process_preference_distribution(
                    electorate=div, original_fp_df=fp_sims_df, updated_fp_df=updated_fp_df, 
                    top_n_df=top4_df, LC_shifted_df=lc_shifted_df, party_df=party_df, 
                    model=model, specific_defection_dict=specific_defection_dict, 
                    fallback_defection_profile=fallback_defection_profile, 
                    TCP_state_dict=TCP_state_dict
                )
            print(time.time() - st, 'top3')
            st = time.time()
            
            # Step 4: 3CP -> 2CP Elimination
            top2_df = process_preference_distribution(
                electorate=div, original_fp_df=fp_sims_df, updated_fp_df=updated_fp_df, 
                top_n_df=top3_df, LC_shifted_df=lc_shifted_df, party_df=party_df, 
                model=model, specific_defection_dict=specific_defection_dict, 
                fallback_defection_profile=fallback_defection_profile, 
                TCP_state_dict=TCP_state_dict
            )
            print(time.time() - st, 'top2')
            st = time.time()
        # Step 5: Final 2CP Distribution & Winner Identification
        final_2cp_df = resolve_final_2cp(
            original_fp_df=fp_sims_df, top2_df=top2_df, 
            tcp_proportions_df=TCP_state_dict[div], party_df=party_df,
            joint_noise=global_joint_noise,rng=rng,sigma_ind=sigma_ind,simulate=True
        )
        
        
        # Step 6: Populate Trackers
        winners = final_2cp_df['Winner'].tolist()
        runners = final_2cp_df['RunnerUp'].tolist()
        win_tcps = final_2cp_df['Winner_2CP'].tolist()
        
        per_electorate_winners[div] = winners
        per_electorate_runners_up[div] = runners
        per_electorate_winning_tcp[div] = win_tcps

        unique_winners = set(winners)
        winner_bloc_map = {w: ('IND' if w.startswith('IND') else get_party_bloc(w, party_df)) for w in unique_winners}

        for sim_idx in range(n_sims):
            w, r = winners[sim_idx], runners[sim_idx]
            
            # Format the pair alphabetically
            pair = tuple(sorted([w, r]))
            per_electorate_tcp_pair[div].append(pair)
            
            per_simulation_winners[sim_idx].append(w) # specific winner 
        print(time.time() - st, 'winner')
        st = time.time()
    import pdb; pdb.set_trace()
    return {
        'electorate_winners': per_electorate_winners,
        'electorate_runners_up': per_electorate_runners_up,
        'electorate_tcp_pairs': dict(per_electorate_tcp_pair),
        'electorate_winning_tcps': per_electorate_winning_tcp,
        'simulation_winners': per_simulation_winners
    }

ordered_electorates = list(FP_pred.keys()) 
sim_local_alr_dict = {electorate: sim_elec_errors[:, i, :] for i, electorate in enumerate(ordered_electorates)}

# update TCP_state_dict
sigma_joint = 0.045
sigma_ind = 0.03
TCP_state_dict, row_joint_sigmas, avg_tcp_df = adjust_TCP_state_dict(TCP_state_dict, party_df, sigma_joint)

simulation_results = run_statewide_preference_simulation(
    FP_pred=FP_pred, 
    LC_FPs_df=LC_FPs_df_rediv if is_redistribution else LC_FPs_df, 
    sim_statewide_alr=sim_poll_errors,            # Your (1000, 4) numpy array
    sim_local_alr_dict=sim_local_alr_dict,        # Dict mapping div to (1000, 4) numpy array
    party_df=party_df,                            # Your party mapping DataFrame
    model=final_model,                            # The calibrated AggregateLatentLogit instance
    compositions_dict=compositions_dict,    # The pre-calculated Dirichlet compositions
    specific_defection_dict=specific_defection_dict,     # Dict of specific Top-4 IND defection profiles
    fallback_defection_profile=fallback_defection_profile, # Your fallback dictionary
    TCP_state_dict=TCP_state_dict,                # The loaded pickle dictionary
    row_joint_sigmas=row_joint_sigmas,
    sigma_ind=sigma_ind,
    data_year = data_year,
    rng=rng
)

per_electorate_winners = simulation_results['electorate_winners']
per_electorate_runners_up = simulation_results['electorate_runners_up']
per_electorate_tcp_pairs = simulation_results['electorate_tcp_pairs']
per_electorate_winning_tcps = simulation_results['electorate_winning_tcps']
per_simulation_winners = simulation_results['simulation_winners']

simulation_bundle = {
    'FP_pred': FP_pred,
    'electorate_winners': per_electorate_winners,
    'electorate_runners_up': per_electorate_runners_up,
    'electorate_tcp_pairs': per_electorate_tcp_pairs,
    'electorate_winning_tcps': per_electorate_winning_tcps,
    'simulation_winners': per_simulation_winners
}

os.chdir(Path.home() / 'Australian Election/Victorian Election/Website-files/')
with open("vic2026_simulation_bundle.pkl", "wb") as f:
    pickle.dump(simulation_bundle, f)

# 3. Ensure deterministic_traces.pkl is also saved in the same folder
with open("deterministic_traces.pkl", "wb") as f:
    pickle.dump(deterministic_traces, f)

print("Successfully saved vic2026_simulation_bundle.pkl and deterministic_traces.pkl!")
import pdb; pdb.set_trace()

