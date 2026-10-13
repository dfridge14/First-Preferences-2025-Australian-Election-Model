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
from collections import defaultdict



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


base_dir = Path.home() / f"Australian Election/Victorian Election"
os.chdir(base_dir)



LC_results = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LC-Votes-Electorate-Formatted.csv')
LA_results = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Primary-Electorate-Formatted.csv')

LA_Candidates = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-LA-Candidates-Formatted.csv')

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

with open(f'VIC-{data_year}/ia_adjustments_dict.pkl', 'rb') as f:
    ia_adjustments_dict = pickle.load(f)

LC_FPs_df = pd.read_csv(f'VIC-{data_year}/{data_year}-LC-First-Prefs-df.csv', index_col = 'PartyAb')


# FIX: temporarily using only 2022 ia model - should obtain a general one
with open(f"VIC-{data_year}/VIC-{data_year}-IA-model-params.json", 'r') as f:
    ia_params_dict = json.load(f)

ia_csv_path = f"VIC-{data_year}/VIC-{data_year}-IA-adjustments.csv"

# Load the Demographic mappings for the next election (all info for data_year already baked into ia_params)
demo_df = pd.read_csv(f'VIC-{election_year}/Demographic-classification-{election_year}.csv')
demo_lookup = demo_df.set_index('district_name')['classification'].to_dict()

inds_contesting = pd.read_csv("INDs_contesting_2026.csv")
new_inds_df = pd.read_csv("new_inds_features.csv")
pool_df = pd.read_csv("pool_df_features.csv")

# load the new IND models, priors, and party dictionary
with open("add_new_IND_models_and_priors.pkl", "rb") as f:
    add_new_IND_data = pickle.load(f)

pool_model = add_new_IND_data['pool_model']
split_model = add_new_IND_data['split_model']
dop_lookup = add_new_IND_data['dop_lookup']
mu_ind_to_ind_array = add_new_IND_data['mu_ind_to_ind']
global_party_dict = add_new_IND_data['global_party_dict']

ALR_COLS = ['ALP', 'GRN', 'ON', 'OTH'] # for errors


# build the next_contesting_parties Dictionary; rename independents from INDX to IND_First_Last for easier tracking 

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

if election_year != 2026:
    LA_Candidates_next = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
    LA_Candidates_next['PartyAb_Specific'] = LA_Candidates_next.apply(format_specific_party, axis=1)
    next_contesting_parties = LA_Candidates_next.groupby('div_nm')['PartyAb_Specific'].agg(list).to_dict()

else:
    LA_Candidates_temp = LA_Candidates.copy()
    LA_Candidates_temp['PartyAb_Specific'] = LA_Candidates_temp.apply(format_specific_party, axis=1)
    
     # CHANGE for actual parties in 2026
    # Currently, just add ON and VNS in every district, remove deregistered parties
    deregistered_list = ['REAS','DHJP','AVP','HAP','SDA','TMP','FPV', 'HMP', 'DLP','LDP','SFF'] # FPV included as announced not contesting
    base_parties = LA_Candidates_temp.groupby('div_nm')['PartyAb_Specific'].agg(list).to_dict()

    inds_contesting['ind_code'] = 'IND_' + inds_contesting['Candidate'].astype(str).str.strip().str.replace(r'\s+', '_', regex=True)
    inds_2026_by_div = inds_contesting.groupby('Electorate')['ind_code'].agg(list).to_dict()
    

    next_contesting_parties = {}
    for div, parties in base_parties.items():
        # Keep only registered parties, stripping out any 2022 INDs
        retained_parties = [p for p in parties if p not in deregistered_list and not str(p).startswith('IND')]
        
        div_inds_2026 = inds_2026_by_div.get(div, [])
        
        next_contesting_parties[div] = list(dict.fromkeys(retained_parties + ['ON', 'VNS'] + div_inds_2026))
    
    # OLD next_contesting_parties = {div: list(dict.fromkeys([p for p in parties if p not in deregistered_list] + ['ON','VNS'])) for div, parties in base_parties.items()}

   # fix IND additions/removals
    #next_contesting_parties['Hawthorn'] = [p for p in next_contesting_parties['Hawthorn'] if p != 'IND_Melissa_Lowe']
    #next_contesting_parties['Mornington'] = [p for p in next_contesting_parties['Mornington'] if p != 'IND_Kate_Lardner']
    #next_contesting_parties['Macedon'] = ['LP','GRN','ALP','ON','VNS','AJP']
    #next_contesting_parties['Narracan'] = ['LP','GRN','ALP','ON','VNS','FFP']
    next_contesting_parties['Narracan'] = ['LP','GRN','ALP','ON','VNS','FFP']
    next_contesting_parties['Clarinda'].append('IND_Caroline_White')


    TARGET_2026_TEALS = {'Footscray': 'IND_Jess_Marsh', 'Mornington': 'IND_Melissa_Goffin', 'Hawthorn': 'IND_Shima_Ibuki'} # Sophie Torney already there
    TARGET_2026_WEST = {'Footscray','Kororoit','Niddrie','St Albans','Sydneham', 'Tarneit','Point Cook','Williamstown'} # doesn't include Paul Hopper




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
  """Enriches inc_df in-place using the exact Mc/Mac capitalization and

  concordance logic from the original 3CP dataset pipeline.
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
def execute_strip_incumbency(dop_table_dict: dict, csv_path: str = f"VIC-{data_year}/VIC-{data_year}-IA-adjustments.csv"):
    """
    Step 1: Reads deterministic cleansed baselines and applies them to DOP Tables.
    """
    # Load the pre-calculated adjustments
    ia_df = pd.read_csv(csv_path).set_index('electorate')
    
    stripped_tables = {}
    static_high_ind_districts = set()
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
            static_high_ind_districts.add(div)

        # Apply the exact historical strip
        if div in ia_df.index:
            row = ia_df.loc[div]
            inc_cand = row['inc_cand']
            adj = {
                inc_cand: -row['ia_alpha'],
                row['chal_1']: row['chal_1_adj'],
                row['chal_2']: row['chal_2_adj']
            }
            
            table = table.adjust_3cp_incumbency(adj, incumbent_party=inc_cand)
            
            # Save for Step 6 tracking
            ia_tracker_dict[div] = {
                'inc_cand': inc_cand,
                'inc_party': ''.join(filter(str.isalpha, inc_cand)),
                'stripped_ia': row['ia_alpha']
            }

        stripped_tables[div] = table

    return stripped_tables, static_high_ind_districts, ia_tracker_dict


inc_df = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-Incumbents.csv')
inc_next_df = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-Incumbents.csv')
master_inc_df = pd.read_csv('VIC-Incumbents-df-2006-2026.csv')

prepare_incumbents_df(inc_df, master_inc_df, data_year=data_year)
prepare_incumbents_df(inc_next_df, master_inc_df, data_year=election_year)

stripped_tables, static_high_ind_districts, ia_tracker_dict = execute_strip_incumbency(
    dop_table_dict=DOP_table_dict,
    csv_path=f"VIC-{data_year}/VIC-{data_year}-IA-adjustments.csv"
)

#import pdb; pdb.set_trace()



#################################################### Step 2: Remove non-contesting parties and INDs (Simulated) #########################################################################################


def execute_party_removal(
    stripped_tables: Dict[str, Any],
    static_high_ind_districts: Set[str],
    next_contesting_parties: Dict[str, List[str]],
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
    
    for div, table in stripped_tables.items():
        contesting_next = next_contesting_parties.get(div, [])
        cands_to_remove = []
        is_high_ind = div in static_high_ind_districts
        
        for c in table.candidates:
            if str(c).startswith('IND'):
                if is_high_ind:
                    continue  # Protect ALL INDs in high-IND seats
                else:
                    cands_to_remove.append(c)  # Liquidate ALL INDs in <=15% seats
                continue
                
            # Party Handling: Remove obsolete minors
            if c not in contesting_next and c not in ['ALP', 'LP', 'NP', 'LNP', 'GRN', 'COAL']:
                cands_to_remove.append(c)
                
            # Missing Major Alert (e.g., Richmond 2018)
            elif c not in contesting_next and c in ['ALP', 'LP', 'NP', 'LNP', 'GRN', 'COAL']:
                print(f"[ALERT] Major party {c} missing in {div}!")
                import pdb; pdb.set_trace()
                
        if not cands_to_remove:
            latent_tables[div] = copy.deepcopy(table)
            continue
            
        # Execute stochastic backpropagation
        if simulate:
            clean_table = table.remove_candidates_stochastic(cands_to_remove, rng=rng)
        else:
            clean_table = table.remove_candidates(cands_to_remove)
            
        latent_tables[div] = clean_table
        
    return latent_tables

latent_tables = execute_party_removal(
    stripped_tables=stripped_tables,
    static_high_ind_districts=static_high_ind_districts,
    next_contesting_parties=next_contesting_parties,
    simulate=True
)

print(f"Step 2 Complete: {len(latent_tables)} latent tables generated.")

# Quick Diagnostic check
print("\nBefore Cleansing (Bentleigh):")
print(stripped_tables['Bentleigh'].count_0)
print("\nAfter Cleansing (Bentleigh):")
print(latent_tables['Bentleigh'].count_0)

#import pdb; pdb.set_trace()




#################################################### Step 3: Add new parties to Count 0 (Simulated) #########################################################################################

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

        for p_idx in range(self.n_pools):
            o_m = orphan_mass[p_idx]
            if o_m <= 1e-12: continue
            for b_idx in range(self.n_blocs):
                prob_bloc = p_bloc_given_pool[p_idx, b_idx]
                if prob_bloc <= 1e-12: continue
                for c in [c for c in active_cands if cand_to_bloc[c] == b_idx]:
                    s_hat[c] += o_m * prob_bloc * cand_intra_share[c]

        return s_hat



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
        if div not in latent_tables or div not in lc_df.columns:
            continue

        table = latent_tables[div]
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
    full_latent_tables: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit'
) -> float:
    """
    Unilaterally injects new party into LC across all districts, measures mean uncalibrated vote share in LC, and returns the Kappa needed to hit the specified prior_vote_share.
    """
    s_hat_1_list = []
    alpha, beta, gamma = model.unpack_params(model.theta)
    
    for div, table in full_latent_tables.items():
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
    full_latent_tables: Dict[str, Any],
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
                full_latent_tables=full_latent_tables, 
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
            contested_divs = [d for d, t in full_latent_tables.items() if t.count_0.get(party, 0.0) > 1e-9]
            
            if not contested_divs:
                print(f"[Warning] {party} is historical but contested 0 LA seats. Defaulting kappa=1.0")
                k = 1.0
                n_seats = 0
            else:
                _, k = diagnose_and_recalibrate_party(
                    party_abbr=party, 
                    contested_divs=contested_divs,
                    latent_tables=full_latent_tables, 
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

def get_party_bloc(party: str, party_df: pd.DataFrame) -> str:
    """Helper to extract ideological bloc for any party in party_df."""
    if str(party).startswith('IND'):
        return 'Centre'
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    return ideo_map.get(party, 'Centre')



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

def execute_new_minor_injection(
    latent_tables: Dict[str, Any],
    next_contesting_parties: Dict[str, List[str]],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: Any,
    sim_kappas: Dict[str, float],
    static_high_ind_districts: Set[str],
    rng: np.random.Generator,
    anomalous_districts: Set[str] = None,
    simulate: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Executes Step 3: Simultaneously injects all missing parties into the structural logit. Returns the newly generated counterfactual Count 0 dictionary per electorate. 
    For static_high_ind_districts/anomalous_districts, it uses the model directly for the counterfactual, injecting extra uncertainty
    """
    cf_c0_dict = {}
    s_hat_a_dict = {}  # Store for Step 4
    
    for div, table in latent_tables.items():
        emp_c0_full = table.count_0

        # Isolate Party Sub-Space (Strip INDs so model evaluates pure LC party reality)
        emp_c0_party = {k: v for k, v in emp_c0_full.items() if not str(k).startswith('IND')}
        
        target_cands = next_contesting_parties.get(div, [])
        missing_cands = [
            c for c in target_cands 
            if c not in emp_c0_party 
            and c not in ['ALP','LP','NP','LNP','GRN'] 
            and not str(c).startswith('IND')
        ] #         Identify who is missing (exclude Majors/Greens and INDs, as they are handled elsewhere)

        new_c0, shift_vec, s_hat_a, s_hat_b = inject_unified_parties_to_dop(district=div, cands_to_add=missing_cands, current_c0=emp_c0_party, lc_df=lc_df, party_df=party_df, model=model, calibration_scalars=sim_kappas)
        s_hat_a_dict[div] = s_hat_a  # IND-free structural baseline for Step 4





        # Direct empirical anchor for delta does not exist -> use model directly; punish with uncertainty
        if missing_cands and (div in static_high_ind_districts or div in anomalous_districts):
            # model's output as 'truth
            override_c0 = copy.deepcopy(s_hat_b)
            
            # calibrate entrants by kappa (with localized uncertainty penalty)
            for e in missing_cands:
                noise_multiplier = max(0.1, rng.normal(1.0, 0.15)) if simulate else 1.0
                penalised_kappa = sim_kappas.get(e, 1.0) * noise_multiplier
                override_c0[e] = s_hat_b.get(e, 0.0) * penalised_kappa
                
            # renormalise (not particularly neat, but simple)
            z = sum(override_c0.values())
            cf_c0_dict[div] = {k: v / z for k, v in override_c0.items()}
            
        else:
            # Standard MRR (Delta) Regime; produces replica of emp_c0_party if not missing_cands
            cf_c0_dict[div] = new_c0
        
        
    return cf_c0_dict, s_hat_a_dict


# Load minor flow model and bootstrapped realisations
final_model = AggregateLatentLogit()

MINOR_MODEL_FILE = f'VIC-{data_year}/model_theta_saved.npy'
MINOR_MODEL_BOOTSTRAP = f'VIC-{data_year}/bootstrapped_thetas_{data_year}.npy'

if os.path.exists(MINOR_MODEL_FILE):
    print(f"Loading exact fully-estimated baseline parameters from {MINOR_MODEL_FILE}...")
    baseline_theta = np.load(MINOR_MODEL_FILE)
    final_model.set_params(baseline_theta)
else:
    raise FileNotFoundError(f"Missing {MINOR_MODEL_FILE}! Run training script to generate the definitive point estimate.")

if os.path.exists(MINOR_MODEL_BOOTSTRAP):
    print("Loading bootstrapped covariance matrix for epistemic uncertainty...")
    bootstrapped_thetas = np.load(MINOR_MODEL_BOOTSTRAP)
else:
    raise FileNotFoundError(f"Missing {MINOR_MODEL_BOOTSTRAP}! Run training script first.")


# Define synthetic targets with priors AND pseudo-sample sizes (pseudo_N); should be higher than typical as represents the only shrinkage available (heuristic: add 10 to instinctual pseudo-N)!
# High pseudo_N (e.g., 20) = Trusts the prior, weak shrinkage to 1.0
# Low pseudo_N (e.g., 3) = Distrusts the prior, heavy shrinkage to 1.0
NEW_PARTY_PRIORS = {
    'SAL': {'prior': 0.02, 'pseudo_N': 6+10}, 
    'CEC': {'prior': 0.0103, 'pseudo_N': 19+10}
}

# full set of parties that require a kappa i.e. need to be added to at least one electorate
parties_to_add = list({p for d,t in latent_tables.items() if d in next_contesting_parties for p in next_contesting_parties[d] if p not in t.count_0 and p not in ['ALP','LP','NP','LNP','GRN'] and not str(p).startswith('IND')})
print(f"Parties identified for unified injection: {parties_to_add}")



print("Pre-calibrating structural realization scalars (Kappas)...")
calibration_data = pre_calibrate_all_kappas(
    parties_to_add=parties_to_add,
    full_latent_tables=latent_tables, 
    lc_df=LC_FPs_df, 
    party_df=party_df, 
    model=final_model, 
    synthetic_priors=NEW_PARTY_PRIORS
)



print("Step 3 Complete. Counterfactual baselines generated.")


#################################################### Step 4: Return INDs to Count 0 via defection rates #########################################################################################

def create_augmented_c0_for_defection(stripped_c0: dict, target_cands: list) -> dict:
    """
    Takes the 2022 stripped Count 0 and removes ONLY non-contesting parties 
    (excluding INDs, which are rigidly preserved).
    Reallocates retiring parties proportionally to isolate IND gravity.
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
    district: str,
    latent_c0: Dict[str, float],      # Contains ALL INDs for High-IND seats
    cf_c0_party: Dict[str, float],    # Step 3 Party Counterfactual
    s_hat_a: Dict[str, float],        # Step 3 Pure Party Baseline
    cands_added_in_step_3: List[str], 
    next_contesting_parties: Dict[str, List[str]], # Needed for Divergent Futures
    lc_series: pd.Series,
    party_df: pd.DataFrame,
    model: Any
) -> Dict[str, float]:
    """
    Executes Step 4: Dynamically restores re-contesting Independents.
    Handles divergent futures (e.g., a massive IND retires but a small IND stays)
    by scaling the defection rates proportionally to the surviving IND mass.
    """
    # 1. Identify ALL historical INDs residing in the latent baseline
    historical_inds = [c for c in latent_c0 if str(c).startswith('IND') and latent_c0[c] > 1e-9]
    if not historical_inds:
        return copy.deepcopy(cf_c0_party)

    # 2. Check Divergent Futures: Who is actually running in 2026?
    target_cands = next_contesting_parties.get(district, [])
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
    return {k: v / z_final for k, v in final_c0.items()}

def test_step_4_standalone(
    dop_table_dict: dict,
    latent_tables: dict,
    cf_c0_dict: dict,
    s_hat_a_dicts: dict,
    next_contesting_parties: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit'
):
    """
    Tests only Step 4 using pre-calculated pipeline outputs.
    Prints diagnostics for seats containing re-contesting Independents.
    """
    summary_rows = []
    
    for district, original_df in dop_table_dict.items():
        if district not in latent_tables or district not in cf_c0_dict:
            continue
            
        latent_c0 = latent_tables[district].count_0
        
        # 1. Check if there are any re-contesting INDs in this seat
        ind_cands = [c for c in latent_c0 if str(c).startswith('IND') and latent_c0[c] > 1e-9]
        if not ind_cands:
            continue # Skip seats without active INDs for this diagnostic
            
        # 2. Extract inputs for this district
        cf_c0_party = cf_c0_dict[district]
        s_hat_a = s_hat_a_dicts[district]
        target_cands = next_contesting_parties.get(district, [])
        
        # Figure out who was newly added in Step 3
        cands_added_in_step_3 = [
            c for c in target_cands 
            if c not in latent_c0 
            and c not in ['ALP', 'LP', 'NP', 'LNP', 'GRN'] 
            and not str(c).startswith('IND')
        ]
        
        lc_series = lc_df[district].dropna()
        lc_series = lc_series / 100.0 if lc_series.sum() > 1.5 else lc_series

        # =========================================================
        # EXECUTE STEP 4
        # =========================================================
        final_c0_step4 = execute_IND_restoration_via_defections(
            district=district,
            latent_c0=latent_c0,
            cf_c0_party=cf_c0_party,
            s_hat_a=s_hat_a,
            cands_added_in_step_3=cands_added_in_step_3,
            next_contesting_parties=next_contesting_parties,
            lc_series=lc_series,
            party_df=party_df,
            model=model
        )
        
        # =========================================================
        # Diagnostics and Reporting
        # =========================================================
        orig_table = DOPTable(original_df)
        total_votes = sum(orig_table.count_0.values())
        orig_c0_norm = {k: v / total_votes for k, v in orig_table.count_0.items()}
        
        orig_ind_share = sum(orig_c0_norm.get(c, 0.0) for c in ind_cands)
        final_ind_share = sum(final_c0_step4.get(c, 0.0) for c in ind_cands)

        row_data = {
            'District': district,
            'IND_Candidates': ", ".join(ind_cands),
            'Orig_IND_Base': f"{orig_ind_share:.2%}",
            'CF_Restored_IND': f"{final_ind_share:.2%}",
            'IND_Delta': f"{(final_ind_share - orig_ind_share):+.2%}"
        }
        for p in cands_added_in_step_3:
            row_data[f'{p}_Vote'] = f"{final_c0_step4.get(p, 0.0):.2%}"
            
        summary_rows.append(row_data)

        # Print Detailed District Trace
        print(f"\n--- {district.upper()} (IND Restoration Trace) ---")
        detail_data = {}
        all_keys = set(orig_c0_norm.keys()) | set(final_c0_step4.keys())
        
        for p in all_keys:
            detail_data[p] = [
                f"{orig_c0_norm.get(p, 0.0):.2%}",           # 1. Original 2022
                f"{latent_c0.get(p, 0.0):.2%}",              # 2. Cleansed Base (w/ IND)
                f"{s_hat_a.get(p, 0.0):.2%}",                # 3. Model Party Base (S_hat_A)
                f"{cf_c0_party.get(p, 0.0):.2%}",            # 4. CF Party Simplex (Step 3)
                f"{final_c0_step4.get(p, 0.0):.2%}"          # 5. Final CF (Step 4, IA-Free)
            ]
        
        detail_df = pd.DataFrame(
            detail_data,
            index=[
                "1. Empirical 2022 (Normalized)",
                "2. Cleansed & IA-Free (Step 2)",
                "3. Model Party Baseline (S_hat_A)",
                "4. Injected Party Simplex (Step 3)",
                "5. Post-Restoration CF (Step 4)"
            ]
        )
        print(detail_df.to_string())

    print("\n" + "=" * 90)
    print("STEP 4 INDEPENDENT RESTORATION SUMMARY")
    print("=" * 90)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))




#################################################### Step 5: Add new INDs via pool_model/split_model #########################################################################################


# -----------------------------------------------------------------
# 1. SETUP TARGET ARCHETYPE DISTRICTS & CANDIDATES - Teals and West party
# -----------------------------------------------------------------
def extract_archetype_profile(district_list, target_ind_list, global_party_dict, augmented_c0_dict, s_hat_a_dict):
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    ind_shares = []
    records = []

    for div, target_names in zip(district_list, target_ind_list):
        if div not in augmented_c0_dict or div not in s_hat_a_dict:
            continue

        emp_c0 = augmented_c0_dict[div]
        s_hat = s_hat_a_dict[div]

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

def model_teals_west(DOP_table_dict, final_model, baseline_theta, ia_csv_path, next_contesting_parties, calibration_data, LC_FPs_df, party_df):

    teal_divs = ['Brighton', 'Caulfield', 'Mornington', 'Kew', 'Hawthorn', 'Sandringham', 'Bellarine', 'Benambra']
    teal_inds = [['IND_Felicity_Frederico'], ['IND_Nomi_Kaltmann'], ['IND_Kate_Lardner'], ['IND_Sophie_Torney'], ['IND_Melissa_Lowe'], ['IND_Clarke_Martin'], ['IND_Sarah_Fenton'], ['IND_Jacqui_Hawkins']]

    west_divs = ['Werribee','Point Cook']
    west_inds = [['IND_Paul_Hopper'], ['IND_Joe_Garra']]

    archetype_divs = [d for d in (teal_divs + west_divs) if d in DOP_table_dict]
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']

    rng_det = np.random.default_rng(seed=42)
    final_model.set_params(baseline_theta)
    archetype_dop = {d: DOP_table_dict[d] for d in archetype_divs}

    # -----------------------------------------------------------------
    # 2. RUN DETERMINISTIC STEPS 1–3 OUTSIDE THE LOOP
    # -----------------------------------------------------------------
    stripped_tables, static_high_inds, _ = execute_strip_incumbency(archetype_dop, ia_csv_path)

    latent_tables = execute_party_removal(
        stripped_tables, static_high_inds, next_contesting_parties, simulate=False, rng=rng_det
    )

    augmented_c0_dict = {
        div: create_augmented_c0_for_defection(stripped_tables[div].count_0, next_contesting_parties.get(div, []))
        for div in archetype_divs if div in stripped_tables
    }
    mean_kappas = draw_stochastic_kappas(calibration_data, simulate=False)

    _, s_hat_a_dict = execute_new_minor_injection(
        latent_tables=latent_tables,
        next_contesting_parties=next_contesting_parties,
        lc_df=LC_FPs_df,
        party_df=party_df,
        model=final_model,
        sim_kappas=mean_kappas,
        static_high_ind_districts=static_high_inds,
        anomalous_districts=set(),
        simulate=False,
        rng=rng_det
    )


    # Compute final target-specific profiles and lognormal method of moments fits
    teal_profile ,teal_records, teal_mean_vote, teal_std_vote = extract_archetype_profile(teal_divs, teal_inds, global_party_dict, augmented_c0_dict, s_hat_a_dict)
    west_profile, west_records, west_mean_vote, west_std_vote = extract_archetype_profile(west_divs, west_inds, global_party_dict, augmented_c0_dict, s_hat_a_dict)

    print(pd.DataFrame(teal_records))
    print(pd.DataFrame(west_records))

    df_teals = pd.DataFrame(teal_records).set_index('district_name')
    extra_teal_vols = np.array([16.14, 5.46, 7.23, 19.16, 10.91, 10.28]) / 100.0
    all_teal_vols = np.concatenate([df_teals['ind_primary_vote'].values, extra_teal_vols])

    teal_mean_v = np.mean(all_teal_vols)
    teal_var_v = np.var(all_teal_vols, ddof=1)
    teal_sigma_log = np.sqrt(np.log(1 + (teal_var_v / (teal_mean_v**2))))
    teal_mu_log = np.log(teal_mean_v) - (0.5 * teal_sigma_log**2)

    # 3. West Volume Arrays
    nq_first_ratios = np.array([4.08, 0.55, 3.03, 2.41]) / 9.37
    other_west_vols = np.array([2.4, 4.9, 1.8, 3.3, 3.1, 0.9, 4.0, 2.4, 3.8, 1.0, 4.4, 3.0, 2.9, 5.5, 7.3,6.6]) / 100.0 
    all_west_vols = np.append(nq_first_ratios*10.6/100,other_west_vols)

    west_mean_v = np.mean(all_west_vols)
    west_var_v = np.var(all_west_vols, ddof=1)
    west_sigma_log = np.sqrt(np.log(1 + (west_var_v / (west_mean_v**2))))
    west_mu_log = np.log(west_mean_v) - (0.5 * west_sigma_log**2)

    teal_params = {'mean': teal_mean_v, 'mu_log': teal_mu_log, 'sigma_log': teal_sigma_log, 'profile': teal_profile}
    west_params = {'mean': west_mean_v, 'mu_log': west_mu_log, 'sigma_log': west_sigma_log, 'profile': west_profile}

    return teal_params, west_params


def execute_targeted_archetype_injection(
    cf_c0: dict,
    div: str,
    TARGET_2026_TEALS: dict,
    TARGET_2026_WEST: set,
    teal_params: dict,
    west_params: dict,
    global_party_dict: dict,
    simulate: bool = False,
    rng: np.random.Generator = None
) -> dict:
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    updated_c0 = cf_c0.copy()

    runs = []
    if div in TARGET_2026_TEALS:
        runs.append((TARGET_2026_TEALS[div], teal_params, False))
    if div in TARGET_2026_WEST or div == 'Werribee':
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

        # Hopper Maneuver preserves existing mass
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

teal_params, west_params = model_teals_west(DOP_table_dict, final_model, baseline_theta, ia_csv_path, next_contesting_parties, calibration_data, LC_FPs_df, party_df, )


DIRICHLET_NU = 20


def load_step5_simulation_inputs(new_inds: pd.DataFrame, pool: pd.DataFrame, target_year: int = 2026) -> Dict[str, Any]:
    """
    Fuses the candidate-level data and electorate-level data into 
    the dictionary format required by Step 5.
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


def execute_new_IND_injection(
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
    Step 5: Generates a brand new Independent insurgency pool using the Gamma model, 
    splits it using OLS (if > 1), and carves the mass out of the Step 4 simplex.
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
        new_ind_features.get('Log_Repeat_Past', 0.0)
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
        formatted_name = f"IND_{str(c['Candidate']).strip().title().replace(' ', '_')}"
        ind_votes = {formatted_name: T_pool_sim} 
    else:
        log_scores = []
        for c in new_candidates:
            eff_days = c.get('Campaign_Length_Days', 0)
            is_counc = float(c.get('Is_Former_Councillor', False))
            x_split = np.array([1.0, is_counc, np.log(eff_days + 1)])
            eps = rng.normal(0, np.sqrt(split_model.mse_resid)) if simulate else 0.0
            pred_log = np.dot(x_split, split_model.params) + eps
            log_scores.append(pred_log)
            
        max_s = max(log_scores)
        exp_s = np.exp(np.array(log_scores) - max_s)
        split_shares = exp_s / np.sum(exp_s)
        
        for i, c in enumerate(new_candidates):
            formatted_name = f"IND_{str(c['Candidate']).strip().title().replace(' ', '_')}"
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
    if district in dop_lookup['electorates']:
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

anomalous_districts = ['Narracan'] if data_year == 2022 else ['Richmond'] if data_year == 2018 else []
num_simulations = 10

# We will store the results of the 1000 simulations here
mc_results_c0 = []
s_hat_a_dicts = []

new_ind_features = load_step5_simulation_inputs(new_inds_df, pool_df, target_year=election_year) # for Step 5 IND addition
target_districts = list(latent_tables.keys())

print(f"Initiating Monte Carlo structural injection ({num_simulations} iterations)...")


############################################ Step 6: Readjust for incumbency ###############################################################



def execute_reapply_incumbency(
    div: str,
    cf_c0_step5: Dict[str, float],
    ia_tracker: Optional[Dict[str, Any]],
    inc_next_info: Optional[Dict[str, Any]],
    params_dict: dict,          # Loaded from 'VIC-IA-Model-Params.json'
    demo_class: str,            # E.g., 'Inner Metropolitan'
    simulate: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    """Step 6: Re-applies 2026 Incumbency Advantage dynamically at Count 0."""
    if rng is None: rng = np.random.default_rng()
        
    if inc_next_info is None or inc_next_info.get('is_open', True):
        return cf_c0_step5.copy()

    sigma_resid = params_dict['sigma_resid']
    coefs = params_dict['coefficients']
    
    next_party = inc_next_info['party']
    next_name_clean = inc_next_info.get('inc_clean_name', '')
    next_tenure = inc_next_info.get('years_tenure', np.nan)

    orig_name_clean = ia_tracker.get('inc_name', '').lower() if ia_tracker else ''
    is_returning = bool(orig_name_clean and next_name_clean and orig_name_clean == next_name_clean)

    # 1. Calculate Magnitude using JSON coefficients
    if is_returning:
        base_ia = float(ia_tracker['stripped_ia'])
        beta_tenure = float(coefs.get('log_tenure', 0.0))
        old_tenure = max(1.0, float(ia_tracker.get('tenure_2022', 1.0)))
        tenure_2026 = next_tenure if (pd.notna(next_tenure) and next_tenure > 0) else (old_tenure + 4.0)

        tenure_delta = beta_tenure * (np.log(max(1.0, tenure_2026)) - np.log(old_tenure))
        
        campaign_shock = rng.normal(0, sigma_resid) if simulate else 0.0
        target_ia = max(0.0, float(base_ia + tenure_delta + campaign_shock))
        inc_party = ia_tracker['inc_party']
    else:
        tenure_val = next_tenure if (pd.notna(next_tenure) and next_tenure > 0) else 2.0
        x_dict = {"const": 1.0, "log_tenure": np.log(max(1.0, tenure_val)), demo_class: 1.0}
        
        # Build array matching the exact coefficient keys
        x_new = np.array([x_dict.get(k, 0.0) for k in coefs.keys()])
        eps = rng.normal(0, sigma_resid) if simulate else 0.0
        
        pred_ia = float(np.dot(x_new, np.array(list(coefs.values()))) + eps)
        target_ia = max(0.0, pred_ia)
        inc_party = next_party

    if target_ia <= 1e-9:
        return cf_c0_step5.copy()

    # 2. Identify Incumbent and Challengers at Count 0
    coal_aliases = ['LP', 'NP', 'LNP', 'COAL'] if inc_party in ['LP', 'NP', 'COAL'] else [inc_party]
    inc_cand_key = next((c for c in cf_c0_step5.keys() if any(str(c).upper().startswith(p) for p in coal_aliases)), None)
    
    if not inc_cand_key:
        return cf_c0_step5.copy()

    active_challengers = [c for c in cf_c0_step5.keys() if c != inc_cand_key and cf_c0_step5[c] > 1e-9]
    if not active_challengers:
        return cf_c0_step5.copy()

    # 3. Dynamic Square-Root Gravity Split Native to Count 0
    sum_sqrt = sum(np.sqrt(cf_c0_step5[c]) for c in active_challengers)
    cf_c0_step6 = cf_c0_step5.copy()
    
    cf_c0_step6[inc_cand_key] += target_ia
    for c in active_challengers:
        weight = np.sqrt(cf_c0_step5[c]) / sum_sqrt if sum_sqrt > 1e-9 else (1.0 / len(active_challengers))
        cf_c0_step6[c] = max(0.0, cf_c0_step6[c] - (target_ia * weight))

    total_v = sum(cf_c0_step6.values())
    return {k: v / total_v for k, v in cf_c0_step6.items()}

# prepare incumbent lookup dictionary for Step 6: Re-adding incumbent
inc_next_lookup = {}
for div, df_div in inc_next_df.groupby('div_nm'):
    # Look for the active incumbent record
    inc_row = df_div[df_div['incumbent'] == 1]
    if not inc_row.empty:
        row = inc_row.iloc[0]
        inc_next_lookup[div] = {
            'is_open': False,
            'candidate_name': str(row.get('candidate_name', '')).strip(),
            'inc_clean_name': str(row.get('inc_clean_name', '')).strip(),
            'party': str(row.get('PartyAb', '')).upper().strip(),
            'years_tenure': float(row.get('years_tenure', np.nan)),
            'is_coal': 1.0 if str(row.get('PartyAb', '')).upper().strip() in ['LP', 'NP', 'LNP', 'COAL'] else 0.0
        }
    else:
        inc_next_lookup[div] = {'is_open': True}


########################## Step Polling ###################

def get_poll_bucket(cand_name: str) -> str:
    party = cand_name.split('_')[0] if '_' in cand_name else cand_name
    if party in ['LP', 'NP', 'LNP']: return 'COAL'
    if party == 'ALP': return 'ALP'
    if party == 'GRN': return 'GRN'
    if party == 'ON': return 'ON'
    # if str(cand_name).startswith('IND'): return 'IND'
    return 'OTH'

def is_left_minor(cand_name: str) -> bool:
    party = cand_name.split('_')[0] if '_' in cand_name else cand_name
    return party in ['AJP', 'SAL', 'SOC', 'VIC_SOC', 'Reason']


def compute_statewide_alr_shifts(
    cf_c0_step5_dict: dict,
    kalman_poll: dict,
    dop_table_dict: dict,
    ref_col: str = 'COAL'
) -> dict:
    """
    Computes the weighted statewide Step 5 aggregate prior across all districts
    and calculates the ΔALR shift vector against the normalized Kalman targets.
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
                bucket = get_poll_bucket(cand)
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


def apply_district_alr_shift(
    district_c0: dict,
    alr_shifts: dict
) -> dict:
    """
    Applies the exponentiated ALR shifts to a single district dictionary,
    enforcing IND immunity and Left-minor pegging.
    """
    # 1. Floor protection to eliminate log-singularities
    cleaned_c0 = {c: max(v, 0.005) for c, v in district_c0.items()}
    tot_cleaned = sum(cleaned_c0.values())
    cleaned_c0 = {c: v / tot_cleaned for c, v in cleaned_c0.items()}

    # 2. Track immune IND mass
    ind_mass = sum(v for k, v in cleaned_c0.items() if get_poll_bucket(k) == 'IND')

    unnormalized_shifted = {}
    shifted_c0 = {}

    for cand, vote in cleaned_c0.items():
        bucket = get_poll_bucket(cand)
        if bucket == 'IND':
            shift = alr_shifts.get('OTH', 0.0) * 0.20
            #shifted_c0[cand] = vote
            #continue

        if bucket == 'OTH' and is_left_minor(cand):
            shift = alr_shifts.get('GRN', 0.0) * 0.5 + alr_shifts.get('OTH', 0.0) * 0.5
        else:
            shift = alr_shifts.get(bucket, 0.0)

        unnormalized_shifted[cand] = vote * np.exp(shift)

    # Re-normalize non-INDs to fit target mass
    shifted_sum = sum(unnormalized_shifted.values())
    if shifted_sum > 1e-9:
        target_mass = 1.0 - ind_mass
        for cand, raw_v in unnormalized_shifted.items():
            shifted_c0[cand] = raw_v * (target_mass / shifted_sum)
    else:
        shifted_c0.update(unnormalized_shifted)

    return shifted_c0

def aggregate_statewide_shares(
    cf_dict: dict, 
    dop_table_dict: dict, 
    valid_buckets: list
) -> dict:
    """
    Takes a dictionary of district allocations and returns the statewide 
    proportional shares for the target polling buckets.
    """
    bucket_raw = {b: 0.0 for b in valid_buckets}
    total_votes = 0.0
    
    for div, c0_table in cf_dict.items():
        if div == 'Narracan':
            div_turnout = 45132.0
        elif div in dop_table_dict and 'TOTAL' in dop_table_dict[div].columns:
            div_turnout = float(dop_table_dict[div]['TOTAL'].iloc[0])
        else:
            div_turnout = 40000.0
            
        for cand, prop in c0_table.items():
            b = get_poll_bucket(cand)
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
    ref_col: str = 'COAL'
) -> tuple:
    """
    Runs ALR transform, appends to traces, and returns the modified traces 
    along with the final computed post-ALR polling shares.
    """
    valid_buckets = list(kalman_poll.keys())
    
    # 1. Extract Step 5 dictionaries
    cf_c0_step5_dict = {}
    for div, trace_df in deterministic_traces.items():
        if '5_Step5 (New INDs)' in trace_df.index:
            row = trace_df.loc['5_Step5 (New INDs)']
            cf_c0_step5_dict[div] = {
                c: float(v) for c, v in row.items() if pd.notna(v) and float(v) > 0
            }

    # 2. Get Step 5 Prior & Calculate Shifts
    prior_shares = aggregate_statewide_shares(cf_c0_step5_dict, dop_table_dict, valid_buckets)
    poll_norm = sum(kalman_poll.values())
    poll_shares = {k: v / poll_norm for k, v in kalman_poll.items()}

    alr_shifts = {}
    for b in valid_buckets:
        prior_alr = np.log((prior_shares[b] + 1e-5) / (prior_shares[ref_col] + 1e-5))
        poll_alr = np.log((poll_shares[b] + 1e-5) / (poll_shares[ref_col] + 1e-5))
        alr_shifts[b] = poll_alr - prior_alr

    # 3. Apply the shifts to create Step 6
    cf_c0_step6_dict = {}
    for div, step5_c0 in cf_c0_step5_dict.items():
        step6_c0 = apply_district_alr_shift(step5_c0, alr_shifts) # Protects INDs and Left Minors
        cf_c0_step6_dict[div] = step6_c0
        deterministic_traces[div].loc['6_Step6 (State Polling ALR)'] = pd.Series(step6_c0)

    # 4. DIAGNOSTIC: Calculate the exact shares AFTER ALR using the same function
    post_shares = aggregate_statewide_shares(cf_c0_step6_dict, dop_table_dict, valid_buckets)

    # 5. Print the Comparison
    print("\n" + "=" * 52)
    print("      POST-ALR STATEWIDE 5-BLOC DIAGNOSTIC")
    print("=" * 52)
    print(f"{'Bloc':<8} | {'Prior (S5)':<11} | {'Target':<11} | {'Post-ALR (S6)':<11}")
    print("-" * 52)
    for b in valid_buckets:
        prior_pct = prior_shares[b] * 100
        target_pct = poll_shares[b] * 100
        post_pct = post_shares[b] * 100
        print(f"{b:<8} | {prior_pct:>10.2f}% | {target_pct:>10.2f}% | {post_pct:>10.2f}%")
    print("=" * 52 + "\n")

    # Return traces AND the final shares dict so you can access it programmatically
    return deterministic_traces, post_shares

############################################# Full simulation loop #######################################################################################

def generate_deterministic_trace_run(
    target_districts: List[str],
    DOP_table_dict: Dict[str, Any],
    inc_next_lookup: Dict[str, Any],
    ia_params_dict: dict,              # <--- Changed from ia_model
    ia_csv_path: str,                  # <--- New path argument for Step 1
    demo_lookup: Dict[str, str],       # <--- New lookup argument for Step 6
    final_model: Any,
    base_theta: np.ndarray,            
    mean_kappas: Dict[str, float],     
    next_contesting_parties: Dict[str, List[str]],
    LC_FPs_df: pd.DataFrame,
    party_df: pd.DataFrame,
    new_ind_features: Dict[str, Any],
    pool_model: Any,
    split_model: Any,
    dop_lookup: Dict[str, Any],
    mu_ind_to_ind_array: np.ndarray,
    global_party_dict: Dict[str, str],
    kalman_poll: Dict[str, float] 
) -> Dict[str, pd.DataFrame]:
    
    debug_traces = {}
    rng_det = np.random.default_rng(seed=42)
    final_model.set_params(base_theta)

    # -------------------------------------------------------------
    # STEP 1: Deterministic IA Removal (Natively Count 0 via CSV)
    # -------------------------------------------------------------
    stripped_tables, static_high_ind_districts, ia_tracker_dict = execute_strip_incumbency(
        dop_table_dict=DOP_table_dict,
        csv_path=ia_csv_path
    )

    # -------------------------------------------------------------
    # STEP 2: Deterministic Party Removal
    # -------------------------------------------------------------
    latent_tables = execute_party_removal(
        stripped_tables=stripped_tables,
        static_high_ind_districts=static_high_ind_districts,
        next_contesting_parties=next_contesting_parties,
        simulate=False,
        rng=rng_det
    )

    # Create perfectly isolated IND baseline

    augmented_c0_dict = {}
    for div in target_districts:
        if div in stripped_tables:
            augmented_c0_dict[div] = create_augmented_c0_for_defection(
                stripped_c0=stripped_tables[div].count_0,
                target_cands=next_contesting_parties.get(div, [])
            )

    if 'Narracan' not in latent_tables:
            import copy
            sample_key = next(iter(latent_tables.keys()))
            narracan_dummy = copy.deepcopy(latent_tables[sample_key])
            
            # Blank out count_0 since there is no 2022 LH general baseline
            if hasattr(narracan_dummy, 'count_0'):
                if isinstance(narracan_dummy.count_0, pd.Series):
                    narracan_dummy.count_0 = pd.Series(dtype=float)
                else:
                    narracan_dummy.count_0 = {}
            latent_tables['Narracan'] = narracan_dummy

    # -------------------------------------------------------------
    # STEP 3: Minor Party Injection
    # -------------------------------------------------------------
    cf_c0_party_dict, s_hat_a_dict = execute_new_minor_injection(
        latent_tables=latent_tables,
        next_contesting_parties=next_contesting_parties,
        lc_df=LC_FPs_df,
        party_df=party_df,
        model=final_model,
        sim_kappas=mean_kappas,
        static_high_ind_districts=static_high_ind_districts,
        anomalous_districts=set(), # Update if you still use anomalous_districts
        simulate=False,
        rng=rng_det
    )

    for div in target_districts + ['Narracan']: # FIX: only 2026
        lc_series = LC_FPs_df[div].dropna()
        lc_series = lc_series / 100.0 if lc_series.sum() > 1.5 else lc_series


        if (div == 'Narracan') and (data_year == 2022):
            # 1. Evaluate entire 2026 Count 0 directly using the model and LC series
            narracan_cands = [c for c in next_contesting_parties['Narracan'] if not str(c).startswith('IND')]
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
            cands_added = [c for c in cf_c0_party_dict[div] if c not in s_hat_a_dict[div]]

            cf_c0_step4 = execute_IND_restoration_via_defections(
                district=div,
                latent_c0=augmented_c0_dict[div],
                cf_c0_party=cf_c0_party_dict[div],
                s_hat_a=s_hat_a_dict[div],
                cands_added_in_step_3=cands_added,
                next_contesting_parties=next_contesting_parties,
                lc_series=lc_series,
                party_df=party_df,
                model=final_model
            )

        # ---------------------------------------------------------
        # STEP 5: Inject New Independent Insurgency
        # ---------------------------------------------------------
        cf_c0_step5 = execute_new_IND_injection(
            district=div,
            cf_c0_step4=cf_c0_step4,
            new_ind_features=new_ind_features.get(div, None),
            pool_model=pool_model,
            split_model=split_model,
            dop_lookup=dop_lookup,
            mu_ind_to_ind_array=mu_ind_to_ind_array,
            global_party_dict=global_party_dict,
            simulate=False,
            rng=rng_det
        )

        # Step 5b: Deterministic Archetype Injection
        cf_c0_step5 = execute_targeted_archetype_injection(
            cf_c0=cf_c0_step5,
            div=div,
            TARGET_2026_TEALS=TARGET_2026_TEALS,
            TARGET_2026_WEST=TARGET_2026_WEST,
            teal_params=teal_params,
            west_params=west_params,
            global_party_dict=global_party_dict,
            simulate=False,
            rng=None
        )

        # =========================================================
        # TRACE CAPTURE (Ends at Step 5)
        # =========================================================
        raw_c0 = {}
        if div in DOP_table_dict:
            raw_df = DOP_table_dict[div]
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

        if (div == 'Narracan') and (data_year == 2022):
            trace_data = {
                '0_Baseline (2022)': {},
                '1_Step1 (Strip IA)': {},
                '2_Step2 (Remove Parties)': {},
                '3_Step3 (Inject Minors)': {},
                '4_Step4 (Restore INDs)': cf_c0_step4,
                '5_Step5 (New INDs)': cf_c0_step5
            }
        else:
            trace_data = {
                '0_Baseline (2022)': raw_c0,
                '1_Step1 (Strip IA)': getattr(stripped_tables.get(div, None), 'count_0', {}),
                '2_Step2 (Remove Parties)': latent_tables[div].count_0,
                '3_Step3 (Inject Minors)': cf_c0_party_dict[div],
                '4_Step4 (Restore INDs)': cf_c0_step4,
                '5_Step5 (New INDs)': cf_c0_step5
            }
        
        df_trace = pd.DataFrame.from_dict(trace_data, orient='index').fillna(0.0)
        df_trace = df_trace.sort_index()
        debug_traces[div] = df_trace.reindex(sorted(df_trace.columns), axis=1)

    # =========================================================
    # MIDDLE: APPLY ALR POLLING SHIFT (STEP 6)
    # =========================================================

    

    print("Applying Kalman ALR Polling Shift to Count 5...")
    debug_traces, post_alr_shares = append_polling_alr_step(
        deterministic_traces=debug_traces,
        kalman_poll=kalman_poll,
        dop_table_dict=DOP_table_dict,
        ref_col='COAL'
    )

    print(post_alr_shares)
    #import pdb; pdb.set_trace()

    # =========================================================
    # SECOND LOOP: RE-APPLY INCUMBENCY ADVANTAGE (STEP 7)
    # =========================================================
    for div in target_districts:
        # Extract the ALR-swung baseline from the trace
        step6_series = debug_traces[div].loc['6_Step6 (State Polling ALR)'].dropna()
        cf_c0_step6 = {k: float(v) for k, v in step6_series.items() if v > 0}

        cf_c0_step7 = execute_reapply_incumbency(
            div=div,
            cf_c0_step5=cf_c0_step6,  # Pass Step 6 in as the structural baseline
            ia_tracker=ia_tracker_dict.get(div, None),
            inc_next_info=inc_next_lookup.get(div, None),
            params_dict=ia_params_dict,                       
            demo_class=demo_lookup.get(div, 'Inner Metropolitan'), 
            simulate=False,                                   
            rng=rng_det
        )

        # Append Step 7 to the trace
        debug_traces[div].loc['7_Step7 (Add 2026 IA)'] = pd.Series(cf_c0_step7)

    return debug_traces

mean_kappas = draw_stochastic_kappas(calibration_data, simulate=False)

kalman_poll = {
        'COAL': 0.288,
        'ALP': 0.244,
        'GRN': 0.134,
        'ON': 0.226,
        'OTH': 0.108
    }

print("Running deterministic structural trace...")

deterministic_traces = generate_deterministic_trace_run(
    target_districts=target_districts,
    DOP_table_dict=DOP_table_dict,
    inc_next_lookup=inc_next_lookup,
    
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
    kalman_poll=kalman_poll
)

print(f"Trace completed for {len(deterministic_traces)} districts.")


print("Step 7 Complete. Traces updated.")

with open("deterministic_traces.pkl", "wb") as f:
    pickle.dump(deterministic_traces, f)

print("Successfully saved deterministic_traces.pkl")

# Stop the script here to skip the heavy Monte Carlo simulations
sys.exit(0)

#import pdb; pdb.set_trace()

# diagnostics
# np.array(sorted([np.round(deterministic_traces[div].iloc[-1,:]['ON'], 3)*100 for div in deterministic_traces.keys()]))
# low_on = [div for div in deterministic_traces.keys() if np.round(deterministic_traces[div].iloc[-1,:]['ON'], 3)*100<10]
# high_on = [div for div in deterministic_traces.keys() if np.round(deterministic_traces[div].iloc[-1,:]['ON'], 3)*100>33]





############################ precompute polling and residual errors for simulation loop ############################


def predraw_polling_residual_simulation_error(days_to_election, num_simulations, NUM_DISTRICTS):

    OPT_EXP_DECAY_PARAMETERS = {
        'OTH': [0.18562381, 0.28067983, 0.05079430],
        'GRN': [0.15782239, 0.24219055, 0.02555061],
        'ALP': [0.09403604, 0.20026747, 0.02794699],
        'ON':  [0.19570000, 0.28000000, 0.02555000]
    }

    # =============================================================================
    # 2. MINIMAL PICKLE LOAD
    # =============================================================================
    with open(Path.home() / "Australian Election/Farrer_byelection_covms.pkl", "rb") as f:
        covm_data = pickle.load(f)

    # Slice 4x4 base matrices directly (dropping IND)
    raw_poll_df = covm_data["Nat_poll_covm"]
    raw_elec_df = covm_data["Electorate_residual_covm"]

    keep_idx = [0, 1, 2, 4]  # Slices off index 3 (IND)
    base_poll_4x4 = raw_poll_df.loc[ALR_COLS, ALR_COLS].values if isinstance(raw_poll_df, pd.DataFrame) else raw_poll_df[np.ix_(keep_idx, keep_idx)]
    C_elec = raw_elec_df.loc[ALR_COLS, ALR_COLS].values if isinstance(raw_elec_df, pd.DataFrame) else raw_elec_df[np.ix_(keep_idx, keep_idx)]

    # =============================================================================
    # 3. TIME-DECAY THE STATEWIDE POLLING MATRIX (C_poll)
    # =============================================================================
    # A. Isolate base correlation matrix R
    stds_base = np.sqrt(np.diag(base_poll_4x4))
    R_poll = base_poll_4x4 / np.outer(stds_base, stds_base)

    # B. Calculate standard deviations at day t
    s_diag = [
        OPT_EXP_DECAY_PARAMETERS[p][1] + (OPT_EXP_DECAY_PARAMETERS[p][0] - OPT_EXP_DECAY_PARAMETERS[p][1]) * np.exp(-OPT_EXP_DECAY_PARAMETERS[p][2] * days_to_election)
        for p in ALR_COLS
    ]

    # C. Reconstruct dynamic C_poll (S @ R @ S) with eigenvalue floor protection
    S = np.diag(s_diag)
    C_poll_unclean = S @ R_poll @ S
    eigvals, eigvecs = np.linalg.eigh(C_poll_unclean)
    C_poll = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T

    # =============================================================================
    # 4. PRE-DRAW SIMULATION ERROR TENSORS
    # =============================================================================
    # sim_poll_errors: Shape (num_simulations, 4) -> Common statewide shock
    # sim_elec_errors: Shape (num_simulations, n_districts, 4) -> Local district shocks
    sim_poll_errors = rng.multivariate_normal(np.zeros(4), C_poll, size=num_simulations)
    sim_elec_errors = rng.multivariate_normal(np.zeros(4), C_elec, size=(num_simulations, NUM_DISTRICTS))

    return sim_poll_errors, sim_elec_errors

days_to_election = 48

sim_poll_errors, sim_elec_errors = predraw_polling_residual_simulation_error(days_to_election, num_simulations=num_simulations, NUM_DISTRICTS=88)

NU_MINOR_PRECISION = 828.4

# Lock the model to the single MLE / optimal parameter set
final_model.set_params(baseline_theta)

mc_results_c0 = []
s_hat_a_dicts = []

# =========================================================================
# MONTE CARLO SIMULATION LOOP
# =========================================================================
for sim_idx in range(num_simulations):

    # -------------------------------------------------------------
    # STEP 2: Stochastic Party Removal
    # -------------------------------------------------------------
    latent_tables = execute_party_removal(
        stripped_tables=stripped_tables,
        static_high_ind_districts=static_high_ind_districts,
        next_contesting_parties=next_contesting_parties,
        simulate=True,
        rng=rng 
    )

    if 'Narracan' not in latent_tables:
        import copy
        sample_key = next(iter(latent_tables.keys()))
        narracan_dummy = copy.deepcopy(latent_tables[sample_key])
        narracan_dummy.count_0 = {} if not isinstance(getattr(narracan_dummy, 'count_0', {}), pd.Series) else pd.Series(dtype=float)
        latent_tables['Narracan'] = narracan_dummy

    # Re-build isolated IND baseline for defections
    augmented_c0_dict = {}
    for div in target_districts:
        if div in stripped_tables:
            augmented_c0_dict[div] = create_augmented_c0_for_defection(
                stripped_c0=stripped_tables[div].count_0,
                target_cands=next_contesting_parties.get(div, [])
            )

    # -------------------------------------------------------------
    # STEP 3: Minor Party Injection + Dirichlet Precision Resampling
    # -------------------------------------------------------------
    sim_kappas = draw_stochastic_kappas(calibration_data, rng)

    cf_c0_party_dict, s_hat_a_dict = execute_new_minor_injection(
        latent_tables=latent_tables,
        next_contesting_parties=next_contesting_parties,
        lc_df=LC_FPs_df,
        party_df=party_df,
        model=final_model,            # Fixed optimal model parameters
        sim_kappas=sim_kappas,
        static_high_ind_districts=static_high_ind_districts,
        rng=rng,
        anomalous_districts=anomalous_districts
    )

    # --- APPLY DIRICHLET VARIANCE TO STEP 3 PRIMARY SHARES ---
    for div, c0_dict in cf_c0_party_dict.items():
        cands = list(c0_dict.keys())
        expected_shares = [c0_dict[c] for c in cands]
        
        # Scale expected probabilities by calibrated precision
        alphas = [max(1e-3, p * NU_MINOR_PRECISION) for p in expected_shares]
        
        # Draw the true election-day distribution on the simplex
        simulated_shares = rng.dirichlet(alphas)
        cf_c0_party_dict[div] = dict(zip(cands, simulated_shares))

    s_hat_a_dicts.append(s_hat_a_dict)

    # -------------------------------------------------------------
    # PASS 1: Build Baselines through Step 5 Across All Districts
    # -------------------------------------------------------------
    cf_c0_step5_dict = {}

    for div in target_districts + ['Narracan']:
        lc_series = LC_FPs_df[div].dropna()
        lc_series = lc_series / 100.0 if lc_series.sum() > 1.5 else lc_series

        # --- Narracan Dedicated Clause ---
        if (div == 'Narracan') and (data_year == 2022):
            narracan_cands = [c for c in next_contesting_parties['Narracan'] if not str(c).startswith('IND')]
            narracan_c0 = build_and_evaluate_model_allocation(
                district='Narracan',
                menu=narracan_cands,
                lc_series=lc_series,
                party_df=party_df,
                model=final_model,
                fallback_c0=None
            )
            tot_n = sum(narracan_c0.values())
            norm_narracan = [v / tot_n for v in narracan_c0.values()]
            
            # Apply Dirichlet to Narracan's raw LC allocation
            alphas_nar = [max(1e-3, p * NU_MINOR_PRECISION) for p in norm_narracan]
            sim_nar = rng.dirichlet(alphas_nar)
            cf_c0_step4 = dict(zip(narracan_c0.keys(), sim_nar))

        # --- Standard Step 4 (Restore INDs) ---
        else:
            cands_added_in_step_3 = [c for c in cf_c0_party_dict[div] if c not in s_hat_a_dict[div]]
            cf_c0_step4 = execute_IND_restoration_via_defections(
                district=div,
                latent_c0=augmented_c0_dict[div],
                cf_c0_party=cf_c0_party_dict[div],
                s_hat_a=s_hat_a_dict[div],
                cands_added_in_step_3=cands_added_in_step_3,
                next_contesting_parties=next_contesting_parties,
                lc_series=lc_series,
                party_df=party_df,
                model=final_model
            )

        # ---------------------------------------------------------
        # STEP 5: Inject New Independent Insurgency
        # ---------------------------------------------------------
        cf_c0_step5 = execute_new_IND_injection(
            district=div,
            cf_c0_step4=cf_c0_step4,
            new_ind_features=new_ind_features.get(div, None),
            pool_model=pool_model,
            split_model=split_model,
            dop_lookup=dop_lookup,
            mu_ind_to_ind_array=mu_ind_to_ind_array,
            global_party_dict=global_party_dict,
            rng=rng,
            simulate=True
        )

        # Step 5b: Stochastic Archetype Injection
        cf_c0_step5 = execute_targeted_archetype_injection(
            cf_c0=cf_c0_step5,
            div=div,
            TARGET_2026_TEALS=TARGET_2026_TEALS,
            TARGET_2026_WEST=TARGET_2026_WEST,
            teal_params=teal_params,
            west_params=west_params,
            global_party_dict=global_party_dict,
            simulate=True,
            rng=rng 
        )

        cf_c0_step5_dict[div] = cf_c0_step5

    # -------------------------------------------------------------
    # MIDDLE: Compute Statewide ALR Shifts (Step 6)
    # -------------------------------------------------------------
    # Uses your existing compute_statewide_alr_shifts helper
    alr_shifts = compute_statewide_alr_shifts(
        cf_c0_step5_dict=cf_c0_step5_dict,
        kalman_poll=kalman_poll,
        dop_table_dict=DOP_table_dict,
        ref_col='COAL'
    )

    # -------------------------------------------------------------
    # PASS 2: Apply ALR Shift (Step 6) and IA (Step 7)
    # -------------------------------------------------------------
    cf_c0_step7_dict = {}

    for d_idx, div in enumerate(target_districts + ['Narracan']):
        # 1. Total ALR Shift = Deterministic Base + Statewide Error + Electorate Shock
        div_shifts = {
            b: alr_shifts[b] + sim_poll_errors[sim_idx, k] + sim_elec_errors[sim_idx, d_idx, k]
            for k, b in enumerate(ALR_COLS)
        }
        div_shifts['COAL'] = 0.0  # Reference baseline

        # 2. Apply through existing function
        cf_c0_step6 = apply_district_alr_shift(
            district_c0=cf_c0_step5_dict[div],
            alr_shifts=div_shifts
        )

        # 2. Step 7: Apply 2026 Incumbency Advantage on the post-ALR baseline
        cf_c0_step7 = execute_reapply_incumbency(
            div=div,
            cf_c0_step5=cf_c0_step6,
            ia_tracker=ia_tracker_dict.get(div, None),
            inc_next_info=inc_next_lookup.get(div, None),
            params_dict=ia_params_dict,                       
            demo_class=demo_lookup.get(div, 'Inner Metropolitan'), 
            simulate=False,                                   
            rng=rng
        )

        cf_c0_step7_dict[div] = cf_c0_step7

    # Store the complete Step 7 realization for this Monte Carlo draw
    mc_results_c0.append(cf_c0_step7_dict)

# pd.DataFrame([result['Bentleigh'] for result in mc_results_c0])
import pdb; pdb.set_trace()



# Run the test using your pre-existing variables in memory
test_step_4_standalone(
    dop_table_dict = DOP_table_dict,
    latent_tables = latent_tables,            # Output from Step 2
    cf_c0_dict = cf_c0_party_dict,                  # Output from Step 3
    s_hat_a_dicts = s_hat_a_dict,            # Output from Step 3
    next_contesting_parties = next_contesting_parties,
    lc_df = LC_FPs_df,
    party_df = party_df,
    model = final_model
)
