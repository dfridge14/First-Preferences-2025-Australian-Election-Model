import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import re
import time
import copy
import json
import time
import copy
import pickle
import re
import multiprocessing
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from pathlib import Path

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from scipy.optimize import minimize
from collections import defaultdict, Counter

rng = np.random.default_rng(seed=2026)

import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler

data_year = 2006
election_year = data_year + 4

SIGMA_JOINT_BASE = 0.045
EXTERNAL_UNCERTAINTY_PENALTY = 1.25
BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
ALR_COLS = ['ALP','GRN','OTH'] # for errors

TCP_COMBINATION_INDEX = {
    ('ALP','COAL'): 0, ('COAL','IND'):1, ('ALP','IND'):2, 
    ('ALP','Left'):3, ('ALP','Right'):4, ('COAL','Left'):5, 
    ('COAL','Right'):6, ('LP','NP'): 7, ('IND','IND'): 8, 
    ('IND','Right'):9, ('IND','Left'):10, ('Left','Right'):11, 
    ('Left','Left'):12, ('Right','Right'):13, ('COAL','COAL'):14
}

os.chdir(Path.home() / 'Australian Election/Victorian Election')

party_df = pd.read_csv(f'VIC-{data_year}/VIC-{data_year}-Parties.csv')
party_df_next = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-Parties.csv')
party_df = pd.concat([party_df, party_df_next], axis=0)[['PartyAb','Ideo_category']].drop_duplicates()


file_path = Path.home() / f"Australian Election/Victorian Election/VIC-{election_year}/TCP_state_dict_for_{election_year}.pkl"
with open(file_path, "rb") as f:
    TCP_state_dict = pickle.load(f)

ORPHAN_POOLS = ['Left','Right','Centre']

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

       

def canonicalize_ind_to_comp_keys(div: str, cand: str, valid_div_cands: set) -> str:
    """
    Maps an IND_* name (e.g., IND_Clare_Middle_Serve or IND_Clare_Serve)
    to the exact matching IND_* key present in compositions_dict for `div`.
    """
    c_str = str(cand).strip()
    if not c_str.startswith('IND'):
        return c_str
    if c_str in valid_div_cands:
        return c_str

    # Strip trailing digits/punctuation and compare First + Last tokens
    base = re.sub(r'[0-9]+$', '', c_str)
    parts = [p for p in base.split('_') if p]
    if len(parts) >= 3:
        first_last = f"{parts[0]}_{parts[1]}_{parts[-1]}"
        if first_last in valid_div_cands:
            return first_last
        # Match any valid IND candidate in that district sharing the same First and Last name
        for vc in valid_div_cands:
            if vc.startswith('IND'):
                vc_parts = [p for p in re.sub(r'[0-9]+$', '', vc).split('_') if p]
                if len(vc_parts) >= 3 and vc_parts[1] == parts[1] and vc_parts[-1] == parts[-1]:
                    return vc

    return clean_ind_name(c_str)

def get_tcp_category(cand: str, party_df: pd.DataFrame) -> str:
    """Helper to map INDs correctly for the TCP Combination Index."""
    if get_party_bloc(cand, party_df) == 'Centre':
        return 'IND'
    return get_party_bloc(cand, party_df)


def get_poll_bucket(cand_name: str, data_year: int) -> str:
    party = cand_name.split('_')[0] if '_' in cand_name else cand_name
    if party in ['LP', 'NP', 'LNP']: return 'COAL'
    if party == 'ALP': return 'ALP'
    if party == 'GRN': return 'GRN'
    if party == 'ON' and data_year == 2022: return 'ON' 
    # if str(cand_name).startswith('IND'): return 'IND'
    return 'OTH'



def is_minor_bloc_party(party: str, party_df: pd.DataFrame) -> bool:
    """Identifies minor parties. Strictly protects ALP, Coalition, and GRN from orphan capture."""
    return party not in ['ALP', 'LP', 'NP', 'LNP', 'GRN'] and not str(party).startswith('IND')



def align_lc_coalition(lc_series: pd.Series, target_cands: list) -> pd.Series:
    """
    Aligns Coalition aliases (LP, NP, LNP, COAL) between the Upper House (LC) series 
    and the intended Lower House candidates BEFORE checking for orphans.
    """
    mod_lc = lc_series.copy()
    la_coal = [c for c in target_cands if c in ('LP', 'NP', 'LNP', 'COAL')]
    lc_coal_active = [c for c in mod_lc.index if c in ('LP', 'NP', 'LNP', 'COAL') and mod_lc[c] > 1e-9]
    
    # >>> CHANGED: Use strict inequality (!=) instead of intersection.
    # If LA has ['LP', 'NP'] but LC only has ['LP'], they don't match! 
    # Pool the mass and give them both their rightful major-party baseline.
    if la_coal and lc_coal_active and set(la_coal) != set(lc_coal_active):
        lc_coal_mass = sum(mod_lc[c] for c in lc_coal_active)
        for c in lc_coal_active:
            mod_lc[c] = 0.0
        for c in la_coal:
            mod_lc[c] = mod_lc.get(c, 0.0) + (lc_coal_mass / len(la_coal))
            
    return mod_lc

def apply_LC_orphan_capture(
    cands_to_add: List[str],
    current_c0: Dict[str, float],
    lc_series: pd.Series,
    party_df: pd.DataFrame
) -> pd.Series:
    """
    Applies Multi-Entrant Capture of LC Orphans:
      - If same-bloc uncontested minor orphans exist (N_B > 0 and Omega_B > 1e-9),
        executes the exact original orphan capture with zero changes.
      - ONLY when N_B == 0 or Omega_B <= 1e-9 (where original code set mod_lc[e] = 0.0),
        extracts target_lc_e by taking 40% proportionally from same-bloc LC parties and
        60% proportionally from all remaining LC parties, preserving the simplex sum (1.0).
    """
    # Clean NaNs so mod_lc.get(e, 0.0) never returns NaN when an entrant is in lc_series.index
    mod_lc = lc_series.fillna(0.0).copy()
    
    # Group entrants by their ideological bloc
    entrants_by_bloc = {}
    for e in cands_to_add:
        b = get_party_bloc(e, party_df)
        entrants_by_bloc.setdefault(b, []).append(e)
        
    for b, entrants in entrants_by_bloc.items():
        k_B = len(entrants)
        
        # All minor bloc parties present in the LC baseline - NOTE: GRN is excluded from bloc capture to avoid polluting the 'minor' party average
        minor_bloc_parties = [
            p for p in mod_lc.index
            if get_party_bloc(p, party_df) == b and is_minor_bloc_party(p, party_df) and mod_lc[p] > 1e-9
        ]
        N_B = len(minor_bloc_parties)
        Sigma_B = sum(mod_lc[p] for p in minor_bloc_parties)
        
        uncontested_minors = [
            p for p in minor_bloc_parties
            if current_c0.get(p, 0.0) <= 1e-9 and p not in cands_to_add
        ]
        Omega_B = sum(mod_lc[p] for p in uncontested_minors)
        
        # =====================================================================
        # FALLBACK GUARD: ONLY triggers when N_B == 0 or Omega_B <= 1e-9
        # (i.e. the exact cases that previously set mod_lc[e] = 0.0)
        # =====================================================================
        if N_B == 0 or Omega_B <= 1e-9:
            # 1. Determine target_lc_e per entrant:
            #    If N_B > 0 (minors exist in bloc b, but all contested LA), use exact Sigma_B / (k_B + N_B).
            #    If N_B == 0 (no minors in bloc b on LC ballot), use average across all LC minors in seat.
            if N_B > 0 and Sigma_B > 1e-9:
                target_lc_e = Sigma_B / (k_B + N_B)
            else:
                all_lc_minors = [
                    p for p in mod_lc.index
                    if is_minor_bloc_party(p, party_df) and mod_lc[p] > 1e-9
                ]
                if all_lc_minors:
                    target_lc_e = sum(mod_lc[p] for p in all_lc_minors) / (k_B + len(all_lc_minors))
                else:
                    target_lc_e = 0.01  # 1.0% fallback if zero minor parties ran in LC at all

            # Cap total extraction at 10% of the ballot so multiple entrants cannot drain majors excessively
            total_extracted = min(k_B * target_lc_e, 0.10)
            lc_e = total_extracted / k_B

            # 2. Identify same-bloc LC parties (including majors: ALP/GRN for Left, LP/NP/LNP/COAL for Right)
            active_lc_parties = [
                p for p in mod_lc.index
                if p not in cands_to_add and mod_lc[p] > 1e-9 and not str(p).startswith('IND')
            ]
            same_bloc_lc = []
            other_lc = []
            for p in active_lc_parties:
                p_bloc = get_party_bloc(p, party_df)
                is_same = (
                    (p_bloc == b)
                    or (b == 'Left' and (p in ('ALP', 'GRN') or p_bloc in ('ALP', 'GRN', 'Left')))
                    or (b == 'Right' and (p in ('LP', 'NP', 'LNP', 'COAL') or p_bloc in ('LP', 'NP', 'LNP', 'COAL', 'Right')))
                )
                if is_same:
                    same_bloc_lc.append(p)
                else:
                    other_lc.append(p)

            same_mass = sum(mod_lc[p] for p in same_bloc_lc)
            other_mass = sum(mod_lc[p] for p in other_lc)

            # 3. Deduct 40% from same-bloc parties and 60% from remaining parties (with fallback if one side is empty)
            if same_mass > 1e-6 and other_mass > 1e-6:
                take_same = min(0.40 * total_extracted, 0.50 * same_mass)
                take_other = total_extracted - take_same
            elif same_mass > 1e-6:
                take_same, take_other = total_extracted, 0.0
            else:
                take_same, take_other = 0.0, total_extracted

            if take_same > 0 and same_mass > 1e-6:
                for p in same_bloc_lc:
                    mod_lc[p] -= take_same * (mod_lc[p] / same_mass)
            if take_other > 0 and other_mass > 1e-6:
                for p in other_lc:
                    mod_lc[p] -= take_other * (mod_lc[p] / other_mass)

            for e in entrants:
                mod_lc[e] = lc_e

            #div_name = getattr(lc_series, 'name', 'Seat')
            #print(f"[Orphan-40/60-Fallback] {div_name} (bloc={b}, N_B={N_B}, Omega_B={Omega_B:.4f}): assigned {lc_e:.4f} to {entrants}")
            continue

        # =====================================================================
        # ORIGINAL LOGIC (100% UNCHANGED WHEN N_B > 0 AND Omega_B > 1e-9)
        # =====================================================================
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
            
    return mod_lc[mod_lc > 0.0]


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

def process_4cp_and_liquidate_inds(
    electorate: str,
    fp_sims_df: pd.DataFrame,
    compositions_dict: dict,
    rng: np.random.Generator,
    ideology_map: dict = {},
    IND_FLOWS_DIRICHLET_ALPHA: float = 25
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
        elim_cols_arr = np.array(elim_cols_tuple, dtype=int)

        saved_ind_votes = {j: fp_vals[rows, j].copy() for j in elim_cols_tuple} # Snapshot all eliminated IND votes and zero them all out BEFORE distributing so IND_A never dumps 'Centre' votes into an already-eliminated IND
        fp_vals[np.ix_(rows, elim_cols_arr)] = 0.0
        active_col_mask = (fp_vals[rows] > 0).any(axis=0)

        for j in elim_cols_tuple:
            cand = cands[j]
            ind_vote = fp_vals[rows, j].copy()  # Shape: (M,)
            fp_vals[rows, j] = 0.0              # Zero out liquidated IND across the group

            # Prepare Dirichlet alpha vector (mu * S) and draw (M x 5) matrix at once
            comp = compositions_dict[(electorate, cand)]
            mu = np.array([comp.get(b, 0.0) for b in BLOCS])
            alpha = np.maximum(mu * IND_FLOWS_DIRICHLET_ALPHA, 1e-6)
            drawn_props = rng.dirichlet(alpha, size=M)  # Shape: (M, 5)

            # Identify active surviving columns in this group
            #active_col_mask = (fp_vals[rows] > 0).any(axis=0)

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

    # (Aesthetic) Recompute top4_df after IND liquidation so Rank_1..Rank_4 match updated_fp_df
    final_top4_indices = np.argsort(-fp_vals, axis=1)[:, :4]
    top4_df = pd.DataFrame(cands[final_top4_indices], columns=['Rank_1', 'Rank_2', 'Rank_3', 'Rank_4'], index=fp_sims_df.index)

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

    lc_series = align_lc_coalition(lc_series, menu_A)

    # ORPHAN CAPTURE: Give synthetic baseline to new LA parties
    zero_lc_entrants = [c for c in menu_A if lc_series.get(c, 0.0) <= 1e-9]
    
    if zero_lc_entrants:
        contested_la_c0 = {**emp_fp.to_dict(), **{c: 1.0 for c in zero_lc_entrants}}
        mod_lc = apply_LC_orphan_capture(
            cands_to_add=zero_lc_entrants, 
            current_c0=contested_la_c0, 
            lc_series=lc_series, 
            party_df=party_df
        )
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

    #print(f"{electorate}| Number of {K}CP combinations:", len(groups))

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
            #print(f"{div} | Fast-path: {num_cands} candidates total -> straight to 2CP")
            st = time.time()

        else:
            # Step 1: ALR Shift to LC Baseline
            local_alr_array = sim_local_alr_dict.get(div, np.zeros_like(sim_statewide_alr))
            lc_shifted_df = vectorised_alr_shift_to_LC(div, LC_FPs_df, sim_statewide_alr, local_alr_array, data_year)
            
            # Step 2: Form 4CP & Liquidate minor INDs
            updated_fp_df, top4_df = process_4cp_and_liquidate_inds(div, fp_sims_df, compositions_dict, rng)
            #print(div)
            #print(time.time() - st, 'top4')
            #st = time.time()

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
            #print(time.time() - st, 'top3')
            #st = time.time()
            
            # Step 4: 3CP -> 2CP Elimination
            top2_df = process_preference_distribution(
                electorate=div, original_fp_df=fp_sims_df, updated_fp_df=updated_fp_df, 
                top_n_df=top3_df, LC_shifted_df=lc_shifted_df, party_df=party_df, 
                model=model, specific_defection_dict=specific_defection_dict, 
                fallback_defection_profile=fallback_defection_profile, 
                TCP_state_dict=TCP_state_dict
            )
            #print(time.time() - st, 'top2')
            #st = time.time()
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


        for sim_idx in range(n_sims):
            w, r = winners[sim_idx], runners[sim_idx]
            
            # Format the pair alphabetically
            pair = tuple(sorted([w, r]))
            per_electorate_tcp_pair[div].append(pair)
            
            per_simulation_winners[sim_idx].append(w) # specific winner 
        #print(time.time() - st, 'winner')
        #st = time.time()

    return {
        'electorate_winners': per_electorate_winners,
        'electorate_runners_up': per_electorate_runners_up,
        'electorate_tcp_pairs': dict(per_electorate_tcp_pair),
        'electorate_winning_tcps': per_electorate_winning_tcp,
        'simulation_winners': per_simulation_winners
    }


# =============================================================================
# 0. CANDIDATE NAME ALIGNMENT & BLOC HELPERS
# =============================================================================

COAL_PARTIES = {'LP', 'NP', 'LNP', 'COAL', 'CLP'}
EXTERNAL_UNCERTAINTY_PENALTY = 1.25  # Ensure this matches your main pipeline constant

def clean_ind_name(raw_name: str) -> str:
    """Forces any Independent name into the strict IND_First_Last convention."""
    if pd.isna(raw_name) or str(raw_name).strip() == 'nan':
        return "UNKNOWN_PARTY"
    name = re.sub(r'^IND_', '', str(raw_name).strip())
    tokens = re.split(r'[\s_]+', name)
    if len(tokens) > 1:
        return f"IND_{tokens[0].title()}_{tokens[-1].title()}"
    return f"IND_{tokens[0].title()}"


def format_specific_party(row) -> str:
    """Converts 'IND' rows in LA_Candidates into canonical 'IND_First_Last'."""
    ab = str(row['PartyAb']).strip()
    if ab.startswith('IND'):
        raw_full = f"{str(row['first_name']).strip()} {str(row['surname']).strip()}"
        return clean_ind_name(raw_full)
    return ab


def get_party_bloc(party_code: str, party_df: pd.DataFrame) -> str:
    if party_code in ('ALP', 'GRN'):
        return 'Left'
    if party_code in COAL_PARTIES:
        return 'Right'
    try:
        return party_df.loc[party_df['PartyAb'] == party_code, 'Ideology'].iloc[0]
    except (IndexError, KeyError):
        return 'Centre'



# =============================================================================
# 1. TCP STATE DICT ADJUSTMENT & NOISE VECTOR BUILDER
# =============================================================================

def get_tcp_election_metadata(election_year: int) -> Tuple[dict, dict, set]:
    if election_year == 2026:
        updated_flows = {
            6: {'IND': 75.0},
            4: {'LP': 34.0, 'GRN': 91.0, 'HMP': 56.0, 'AJP': 62.0, 'Right': 31.0, 'Left': 70.0, 'Centre': 51.0},
            9: {'LP': 41.0, 'NP': 31.0, 'GRN': 92.0, 'HMP': 63.0, 'Centre': 70.0, 'Right': 39.0, 'Left': 75.0},
            11: {'LP': 34.0, 'GRN': 91.0, 'HMP': 56.0, 'AJP': 62.0, 'Right': 31.0, 'Left': 70.0, 'Centre': 51.0, 'ALP': 90.0}
        }
        n_sources = {0: 115, 1: 14, 2: 8, 3: 9, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
        external_rows = {4, 6, 9, 11}
    elif election_year == 2022:
        updated_flows = {3: {'LP': 30.0, 'NP': 34.0}}
       
        n_sources = {0: 124, 1: 15, 2: 2, 3: 6, 4: 0, 5: 3, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
        external_rows = set()
    elif election_year == 2018:
        updated_flows = {}
        n_sources = {0: 135, 1: 6, 2: 2, 3: 3, 4: 0, 5: 3, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
        external_rows = set()
    elif election_year == 2014:
        updated_flows = {}
        n_sources = {0: 139, 1: 2, 2: 1, 3: 3, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
        external_rows = set()
    elif election_year == 2010:
        updated_flows = {3: {'LP': 62.0, 'NP': 65.0}}
        n_sources = {0: 142, 1: 3, 2: 1, 3: 3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
        external_rows = set()
    else:
        raise ValueError(f"Unsupported election_year: {election_year}")

    return updated_flows, n_sources, external_rows


def compute_statewide_average_flows(tcp_dict: dict, party_df: pd.DataFrame) -> pd.DataFrame:
    all_dfs = pd.concat([
        df.rename(columns=lambda c: 'IND' if str(c).startswith('IND') else c).T.groupby(level=0).mean().T
        for df in tcp_dict.values()
    ])
    avg_df = all_dfs.groupby(all_dfs.index).mean()

    bloc_parties = {'Left': [], 'Right': [], 'Centre': []}
    for col in avg_df.columns:
        bloc = 'Centre' if str(col).startswith('IND') else get_party_bloc(col, party_df)
        if bloc in bloc_parties:
            bloc_parties[bloc].append(col)

    for bloc, parties in bloc_parties.items():
        avg_df[bloc] = avg_df[parties].mean(axis=1) if parties else np.nan

    return avg_df


def apply_tcp_adjustments(
    tcp_dict: dict,
    avg_df: pd.DataFrame,
    updates: dict,
    party_df: pd.DataFrame
) -> dict:
    adjusted_dict = copy.deepcopy(tcp_dict)
    if not updates:
        return adjusted_dict

    delta_matrix = {row: {} for row in updates.keys()}
    for row_idx, target_flows in updates.items():
        for col, target_val in target_flows.items():
            base_val = avg_df.loc[row_idx, col]
            if pd.notna(base_val):
                delta_matrix[row_idx][col] = target_val - base_val

    for div, df in adjusted_dict.items():
        for row_idx, deltas in delta_matrix.items():
            for col_name, delta_val in deltas.items():
                if col_name in ['Left', 'Right', 'Centre']:
                    for party_col in df.columns:
                        p_bloc = 'Centre' if str(party_col).startswith('IND') else get_party_bloc(party_col, party_df)
                        if p_bloc == col_name and party_col not in deltas:
                            df.loc[row_idx, party_col] = np.clip(df.loc[row_idx, party_col] + delta_val, 0.5, 99.5)
                else:
                    if col_name in df.columns:
                        df.loc[row_idx, col_name] = np.clip(df.loc[row_idx, col_name] + delta_val, 0.5, 99.5)

    return adjusted_dict


def build_joint_noise_vector(
    n_sources: dict,
    external_rows: set,
    base_sigma: float,
    ext_penalty: float = EXTERNAL_UNCERTAINTY_PENALTY
) -> np.ndarray:
    sigmas = np.zeros(15, dtype=float)
    for i in range(15):
        n = n_sources.get(i, 0)
        eff_n = max(float(n), 0.5)
        multiplier = np.sqrt((eff_n + 1.0) / eff_n)
        if i in external_rows:
            multiplier *= ext_penalty
        sigmas[i] = base_sigma * multiplier
    return sigmas


def adjust_TCP_state_dict(
    TCP_state_dict: dict,
    party_df: pd.DataFrame,
    SIGMA_JOINT_BASE: float,
    election_year: int,
    ext_penalty: float = EXTERNAL_UNCERTAINTY_PENALTY
) -> Tuple[dict, np.ndarray, pd.DataFrame]:
    updated_flows, n_sources, external_rows = get_tcp_election_metadata(election_year)
    avg_tcp_df = compute_statewide_average_flows(TCP_state_dict, party_df)
    adjusted_TCP_state_dict = apply_tcp_adjustments(TCP_state_dict, avg_tcp_df, updated_flows, party_df)
    row_joint_sigmas = build_joint_noise_vector(n_sources, external_rows, SIGMA_JOINT_BASE, ext_penalty)
    return adjusted_TCP_state_dict, row_joint_sigmas, avg_tcp_df


# =============================================================================
# 2. PURE CALIBRATION METRICS (SOFT-BOUNDARY N_eff-WEIGHTED ECE & RMSE)
# =============================================================================

CALIBRATION_BINS = [
    ("1-10%",   0.01, 0.10, False),
    ("10-25%",  0.10, 0.25, False),
    ("25-40%",  0.25, 0.40, False),
    ("40-60%",  0.40, 0.60, False),
    ("60-75%",  0.60, 0.75, False),
    ("75-90%",  0.75, 0.90, False),
    ("90-99%",  0.90, 0.99, False),
    ("99-100%", 0.99, 1.00, True),
]


def extract_win_probs_from_sim_results(
    simulation_results: Any,
    FP_pred: dict,
    actual_winners_dict: dict,
    election_year: str
) -> pd.DataFrame:
    """
    Extracts each candidate's simulated win probability p_win in [0, 1] and binary
    actual outcome y_win in {0, 1} across all electorates for one election year.
    """

    simulation_results = simulation_results['electorate_winners']

    records = []

    for div, actual_winner_raw in actual_winners_dict.items():
        if div not in FP_pred:
            continue

        actual_winner = (
            clean_ind_name(actual_winner_raw)
            if str(actual_winner_raw).startswith('IND')
            else str(actual_winner_raw).strip()
        )

        if isinstance(FP_pred[div], pd.DataFrame):
            cands = list(FP_pred[div].columns)
        elif isinstance(FP_pred[div], list):
            cands = list(FP_pred[div][0].keys())
        else:
            cands = list(FP_pred[div].keys())

        div_res = simulation_results[div] if isinstance(simulation_results, dict) else getattr(simulation_results, div)

        if isinstance(div_res, dict) and ('win_probs' in div_res or 'winner_probs' in div_res):
            raw_probs = div_res.get('win_probs', div_res.get('winner_probs'))
            norm = 100.0 if max(raw_probs.values(), default=0.0) > 1.05 else 1.0
            win_prob_map = {k: float(v) / norm for k, v in raw_probs.items()}
        elif isinstance(div_res, dict) and 'winners' in div_res:
            winners_list = div_res['winners']
            n_s = len(winners_list)
            counts = Counter(winners_list)
            win_prob_map = {k: counts.get(k, 0) / float(n_s) for k in cands}
        elif isinstance(div_res, pd.DataFrame):
            winner_series = div_res.idxmax(axis=1)
            counts = winner_series.value_counts()
            n_s = len(div_res)
            win_prob_map = {k: float(counts.get(k, 0)) / float(n_s) for k in div_res.columns}
            for k in div_res.columns:
                if k not in cands:
                    cands.append(k)
        elif isinstance(div_res, (list, np.ndarray)):
            n_s = len(div_res)
            if isinstance(div_res[0], str):
                counts = Counter(div_res)
                win_prob_map = {k: counts.get(k, 0) / float(n_s) for k in cands}
            elif isinstance(div_res[0], dict):
                counts = Counter(max(sim_d.items(), key=lambda x: x[1])[0] for sim_d in div_res)
                win_prob_map = {k: counts.get(k, 0) / float(n_s) for k in cands}
            else:
                raise ValueError(f"Unrecognized list element in simulation_results[{div}]: {type(div_res[0])}")
        else:
            raise ValueError(f"Unrecognized structure in simulation_results[{div}]: {type(div_res)}")

        clean_prob_map = {}
        for k, v in win_prob_map.items():
            ck = clean_ind_name(k) if str(k).startswith('IND') else str(k).strip()
            clean_prob_map[ck] = clean_prob_map.get(ck, 0.0) + float(v)

        all_div_cands = set(clean_prob_map.keys()) | {
            clean_ind_name(c) if str(c).startswith('IND') else str(c).strip() for c in cands
        }

        for cand in sorted(all_div_cands):
            p_win = float(np.clip(clean_prob_map.get(cand, 0.0), 0.0, 1.0))
            if cand == actual_winner:
                y_win = 1.0
            elif cand in COAL_PARTIES and actual_winner in COAL_PARTIES and actual_winner not in all_div_cands:
                y_win = 1.0
            else:
                y_win = 0.0

            records.append({
                "year": str(election_year),
                "division": div,
                "candidate": cand,
                "pred_prob": p_win,
                "actual_win": y_win
            })

    return pd.DataFrame(records)


def compute_calibration_table_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the exact 8-bin hard-boundary calibration table for reporting."""
    rows = []
    probs = df["pred_prob"].to_numpy(dtype=float)
    wins = df["actual_win"].to_numpy(dtype=float)

    for label, lo, hi, right_closed in CALIBRATION_BINS:
        if right_closed:
            mask = (probs >= lo) & (probs <= hi + 1e-12)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_cases = int(np.sum(mask))
        if n_cases > 0:
            mean_pred = float(np.mean(probs[mask]))
            obs_rate = float(np.mean(wins[mask]))
            diff = obs_rate - mean_pred
        else:
            mean_pred = np.nan
            obs_rate = np.nan
            diff = np.nan

        rows.append({
            "bin": label,
            "n_cases": n_cases,
            "mean_pred_prob": mean_pred,
            "observed_win_rate": obs_rate,
            "obs_minus_pred": diff
        })

    return pd.DataFrame(rows).set_index("bin")


def compute_soft_binned_calibration_loss(
    df: pd.DataFrame,
    bandwidth_scale: float = 0.25
) -> Tuple[float, float, float, pd.DataFrame]:
    """
    Pure Calibration Objective (Equal per-case N_eff weighting across all bins):
      1. Computes soft Gaussian-tapered bin memberships over the 8 bins in [1%, 100%]
         so candidates crossing a bin boundary (e.g. 24.9% -> 25.0%) change the
         objective smoothly rather than causing a discontinuous step-function jump.
      2. Weights each bin strictly by its effective sample size N_eff (no sqrt(N)
         down-weighting of the 1-10% or 99-100% outlier bins).
    Returns: (combined_cal_loss_pct, soft_weighted_ece_pct, hard_weighted_ece_pct, hard_cal_table)
    """
    hard_cal_table = compute_calibration_table_for_df(df)

    active = df[df["pred_prob"] >= 0.01].copy()
    if len(active) == 0:
        return 1e6, 1e6, 1e6, hard_cal_table

    probs = active["pred_prob"].to_numpy(dtype=float)
    wins = active["actual_win"].to_numpy(dtype=float)

    soft_abs_errors = []
    soft_sq_errors = []
    n_eff_weights = []

    for _, lo, hi, right_closed in CALIBRATION_BINS:
        width = hi - lo
        tau = max(0.003, bandwidth_scale * width)

        dist_below = np.maximum(0.0, lo - probs)
        dist_above = np.maximum(0.0, probs - hi) if not right_closed else np.zeros_like(probs)
        dist_out = dist_below + dist_above

        w = np.exp(-0.5 * ((dist_out / tau) ** 2))
        w = np.where(dist_out > 2.5 * tau, 0.0, w)

        n_eff = float(np.sum(w))
        if n_eff > 1e-6:
            soft_mean_pred = float(np.sum(w * probs) / n_eff)
            soft_obs_rate = float(np.sum(w * wins) / n_eff)
            err = abs(soft_obs_rate - soft_mean_pred)

            soft_abs_errors.append(err)
            soft_sq_errors.append(err ** 2)
            n_eff_weights.append(n_eff)

    n_eff_weights = np.asarray(n_eff_weights, dtype=float)
    soft_abs_errors = np.asarray(soft_abs_errors, dtype=float)
    soft_sq_errors = np.asarray(soft_sq_errors, dtype=float)

    # 1. N_eff-weighted Soft Mean Absolute Calibration Error (all candidates weighted equally)
    soft_weighted_ece = float(np.sum(n_eff_weights * soft_abs_errors) / np.sum(n_eff_weights))
    # 2. N_eff-weighted Soft Root Mean Squared Calibration Error
    soft_weighted_rmse = float(np.sqrt(np.sum(n_eff_weights * soft_sq_errors) / np.sum(n_eff_weights)))

    # Standard hard-boundary N_b-weighted ECE for reporting
    valid_hard = hard_cal_table.dropna(subset=["obs_minus_pred"])
    if len(valid_hard) > 0:
        hw = valid_hard["n_cases"].to_numpy(dtype=float)
        herr = np.abs(valid_hard["obs_minus_pred"].to_numpy(dtype=float))
        hard_ece = float(np.sum(hw * herr) / np.sum(hw))
    else:
        hard_ece = 1.0

    combined_cal_loss_pct = (0.5 * soft_weighted_ece + 0.5 * soft_weighted_rmse) * 100.0
    return combined_cal_loss_pct, soft_weighted_ece * 100.0, hard_ece * 100.0, hard_cal_table


def format_side_by_side_calibration_table(df_all: pd.DataFrame, elections=('2022', '2018', '2014', '2010')) -> pd.DataFrame:
    """Builds the exact multi-column Side-by-Side & Pooled Calibration Table."""
    tables = {}
    for yr in elections:
        sub = df_all[df_all["year"] == str(yr)]
        tables[str(yr)] = compute_calibration_table_for_df(sub)

    tables["Pooled (All 4)"] = compute_calibration_table_for_df(df_all)
    return pd.concat(tables, axis=1)


# =============================================================================
# 3. MULTIPROCESSING WORKER & LOO-CV OPTIMIZER
# =============================================================================

GLOBAL_PART2_BUNDLES = {}

def _init_part2_worker(part2_bundles):
    global GLOBAL_PART2_BUNDLES
    GLOBAL_PART2_BUNDLES = part2_bundles


def evaluate_part2_grid_point(args):
    sigma_tot, joint_ind_ratio, sigma_joint, sigma_ind, yr_str = args
    bundle = GLOBAL_PART2_BUNDLES[yr_str]

    row_joint_sigmas = build_joint_noise_vector(
        n_sources=bundle["n_sources"],
        external_rows=bundle["external_rows"],
        base_sigma=sigma_joint,
        ext_penalty=EXTERNAL_UNCERTAINTY_PENALTY
    )

    rng_sim = np.random.default_rng(2026 + int(yr_str))

    simulation_results = run_statewide_preference_simulation(
        FP_pred=bundle["FP_pred"],
        LC_FPs_df=bundle["LC_FPs_df"],
        sim_statewide_alr=bundle["sim_poll_errors"],
        sim_local_alr_dict=bundle["sim_local_alr_dict"],
        party_df=bundle["party_df"],
        model=bundle["final_model"],
        compositions_dict=bundle["compositions_dict"],
        specific_defection_dict=bundle["specific_defection_dict"],
        fallback_defection_profile=bundle["fallback_defection_profile"],
        TCP_state_dict=bundle["adjusted_TCP_state_dict"],
        row_joint_sigmas=row_joint_sigmas,
        sigma_ind=sigma_ind,
        data_year=bundle["data_year"],
        rng=rng_sim
    )

    cand_df = extract_win_probs_from_sim_results(
        simulation_results=simulation_results,
        FP_pred=bundle["FP_pred"],
        actual_winners_dict=bundle["actual_winners_dict"],
        election_year=yr_str
    )

    return {
        "sigma_tot": sigma_tot,
        "ratio": joint_ind_ratio,
        "sigma_joint": sigma_joint,
        "sigma_ind": sigma_ind,
        "year": yr_str,
        "cand_df": cand_df
    }


def validate_part2_tcp_params(
    PART2_BUNDLES_BY_YEAR: dict,
    *,
    # Total preference std dev sqrt(sigma_joint^2 + sigma_ind^2); 0.0541 corresponds to (0.045, 0.030)
    sigma_tot_grid: list = [0.020, 0.030, 0.040, 0.048, 0.0541, 0.062, 0.070, 0.080, 0.095, 0.110],
    # Fixed ratio sigma_joint / sigma_ind = 1.5 (from 0.045 / 0.030) to resolve collinearity
    joint_to_ind_ratio_grid: list = [1.5],
    n_processes: int = 24
):
    elections = ['2010', '2014', '2018', '2022']

    param_pairs = []
    for s_tot in sigma_tot_grid:
        for r in joint_to_ind_ratio_grid:
            s_ind = round(float(s_tot / np.sqrt(1.0 + r ** 2)), 5)
            s_joint = round(float(r * s_ind), 5)
            param_pairs.append((s_tot, r, s_joint, s_ind))

    all_tasks = [
        (s_tot, r, s_joint, s_ind, yr)
        for (s_tot, r, s_joint, s_ind) in param_pairs
        for yr in elections
    ]

    print(f"Starting Part 2 TCP precomputation for {len(all_tasks)} tasks across {n_processes} cores...")
    t0 = time.time()

    with multiprocessing.Pool(
        processes=n_processes,
        initializer=_init_part2_worker,
        initargs=(PART2_BUNDLES_BY_YEAR,)
    ) as pool:
        raw_results = list(tqdm(
            pool.imap_unordered(evaluate_part2_grid_point, all_tasks, chunksize=1),
            total=len(all_tasks)
        ))

    print(f"Part 2 precomputation complete in {time.time() - t0:.2f} seconds.")

    df_lookup = defaultdict(dict)
    for res in raw_results:
        key = (res["sigma_tot"], res["ratio"], res["sigma_joint"], res["sigma_ind"])
        df_lookup[key][res["year"]] = res["cand_df"]

    # --- LEAVE-ONE-ELECTION-OUT CROSS-VALIDATION (3 NON-HOLDOUT YEARS COMBINED) ---
    best_loo_by_heldout = {}
    oos_heldout_dfs = []

    for heldout in elections:
        train_elections = [e for e in elections if e != heldout]
        candidate_scores = []

        for key, yr_dfs in df_lookup.items():
            train_df = pd.concat([yr_dfs[e] for e in train_elections], ignore_index=True)
            val_loss, val_soft_ece, val_hard_ece, _ = compute_soft_binned_calibration_loss(train_df)
            candidate_scores.append((key, val_loss, val_soft_ece, val_hard_ece))

        best_key, best_val_loss, best_val_soft_ece, best_val_hard_ece = min(candidate_scores, key=lambda x: x[1])
        s_tot, r, best_sj, best_si = best_key

        heldout_df = df_lookup[best_key][heldout]
        oos_heldout_dfs.append(heldout_df)
        test_loss, test_soft_ece, test_hard_ece, _ = compute_soft_binned_calibration_loss(heldout_df)

        best_loo_by_heldout[heldout] = {
            "best_sigma_tot": s_tot,
            "best_sigma_joint": best_sj,
            "best_sigma_ind": best_si,
            "train_cal_loss": best_val_loss,
            "train_soft_ece_pct": best_val_soft_ece,
            "train_hard_ece_pct": best_val_hard_ece,
            "test_cal_loss": test_loss,
            "test_soft_ece_pct": test_soft_ece,
            "test_hard_ece_pct": test_hard_ece
        }

        print(
            f"[Held-out {heldout}] Best sigma_tot={s_tot:.4f} (sigma_joint={best_sj:.4f}, sigma_ind={best_si:.4f}) | "
            f"3-Yr Train ECE={best_val_hard_ece:.2f}% (Soft ECE={best_val_soft_ece:.2f}%) | "
            f"Test ECE={test_hard_ece:.2f}%"
        )

    # --- POOLED RANKING ACROSS ALL 4 ELECTIONS ---
    full_ranking = []
    for key, yr_dfs in df_lookup.items():
        s_tot, r, s_joint, s_ind = key
        pooled_df = pd.concat([yr_dfs[e] for e in elections], ignore_index=True)
        tot_loss, soft_ece, hard_ece, cal_tbl = compute_soft_binned_calibration_loss(pooled_df)
        full_ranking.append({
            "sigma_tot": s_tot,
            "ratio": r,
            "sigma_joint": s_joint,
            "sigma_ind": s_ind,
            "cal_obj_loss": tot_loss,
            "soft_weighted_ece_pct": soft_ece,
            "hard_weighted_ece_pct": hard_ece,
            "max_bin_err_pct": float(cal_tbl["obs_minus_pred"].abs().max() * 100.0)
        })

    ranking_df = pd.DataFrame(full_ranking).sort_values("cal_obj_loss", ascending=True).reset_index(drop=True)
    print("\n=== TOP PARAMETER COMBINATIONS (POOLED 2010-2022) ===")
    print(ranking_df.to_string(index=False))

    best_row = ranking_df.iloc[0]
    best_key = (best_row["sigma_tot"], best_row["ratio"], best_row["sigma_joint"], best_row["sigma_ind"])
    best_pooled_df = pd.concat([df_lookup[best_key][e] for e in elections], ignore_index=True)
    cal_table_best = format_side_by_side_calibration_table(best_pooled_df)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.3f}'.format)

    print(
        f"\n=== SIDE-BY-SIDE & POOLED CALIBRATION TABLE "
        f"(Best: sigma_joint={best_row['sigma_joint']:.4f}, sigma_ind={best_row['sigma_ind']:.4f}) ==="
    )
    print(cal_table_best)

    oos_pooled_df = pd.concat(oos_heldout_dfs, ignore_index=True)
    cal_table_oos = format_side_by_side_calibration_table(oos_pooled_df)
    print("\n=== OUT-OF-SAMPLE (LOO-CV) SIDE-BY-SIDE & POOLED CALIBRATION TABLE ===")
    print(cal_table_oos)

    return best_loo_by_heldout, ranking_df, cal_table_best, cal_table_oos


# =============================================================================
# 4. STAGE A INPUT LOADER & EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    PART2_BUNDLES_BY_YEAR = {}

    for election_year in [2010, 2014, 2018, 2022]:
        yr_str = str(election_year)
        data_year = election_year - 4

        # 1. Load Part 1 First Preference Simulation Output (FP_pred_for_{election_year}.pkl)
        with open(f'FP_pred_for_{election_year}.pkl', 'rb') as f:
            FP_results_c0 = pickle.load(f)

        # Convert list[sim_idx][div] -> dict[div] of DataFrames (num_sims x num_cands)
        if isinstance(FP_results_c0, list):
            electorates = list(FP_results_c0[0].keys())
            FP_pred = {
                e: pd.DataFrame([res[e] for res in FP_results_c0]).fillna(0.0)
                for e in electorates
            }
        else:
            FP_pred = FP_results_c0

        # Canonicalize any IND_* columns inside FP_pred to match actual_winners_dict
        FP_pred = {
            div: df.rename(columns=lambda c: clean_ind_name(c) if str(c).startswith('IND') else c)
                   .T.groupby(level=0).sum().T
            for div, df in FP_pred.items()
        }
        

        ordered_electorates = list(FP_pred.keys())
        num_sims = len(next(iter(FP_pred.values())))

        # 2. Build Actual Winners Dict from LA_Candidates_next
        LA_Candidates_next = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
        if election_year == 2022:
            LA_Candidates_next = LA_Candidates_next.loc[~(LA_Candidates_next['div_nm'] == 'Narracan')]
        LA_Candidates_next['PartyAb'] = LA_Candidates_next.apply(format_specific_party, axis=1)

        div_col = 'div_nm' if 'div_nm' in LA_Candidates_next.columns else 'Electorate'
        elected_df = LA_Candidates_next.loc[LA_Candidates_next['elected'] == 1]
        actual_winners_dict = dict(
            zip(
                elected_df[div_col].astype(str).str.strip(),
                elected_df['PartyAb'].astype(str).str.strip()
            )
        )

        with open(f'ALR_errors_for_{election_year}.pkl', 'rb') as f:
            alr_error_bundle = pickle.load(f)

        sim_poll_errors = alr_error_bundle["sim_poll_errors"]
        sim_local_alr_dict = alr_error_bundle["sim_local_alr_dict"]

        # 4. Load per-election preference model inputs
        #    (Update file paths below if your preference objects are stored under slightly different names)
        with open(f'VIC-{election_year}/TCP_state_dict_for_{election_year}.pkl', 'rb') as f:
            TCP_state_dict = pickle.load(f)
        with open(f'pref_model_bundle_for_{election_year}.pkl', 'rb') as f:
            pref_bundle = pickle.load(f)

        party_df = pref_bundle["party_df"]
        LC_FPs_df_active = pref_bundle["LC_FPs_df_rediv"] if pref_bundle.get("is_redistribution", False) else pref_bundle["LC_FPs_df"]
        compositions_dict = pref_bundle["compositions_dict"]
        specific_defection_dict = pref_bundle["specific_defection_dict"]
        fallback_defection_profile = pref_bundle["fallback_defection_profile"]

         # Load minor flow model and bootstrapped realisations
        final_model = AggregateLatentLogit()
        MINOR_MODEL_FILE = f'VIC-{data_year}/{data_year}_LC_flow_model_theta.npy'
        baseline_theta = np.load(MINOR_MODEL_FILE) if data_year != 2006 else np.array([-0.94748286, 0.05, -1, 0.32609389, 3, 2, 3, 1.20926125, -0.1977211 , -1.55463882,0.01473377,  0.55844421,  0.42967291,  0.75830687,  1.91614176, 0.2]) # HIJACKED SOME PARAMETERS FROM 2010-2022
        final_model.set_params(baseline_theta)

        def _norm_ind(c):
            if not str(c).startswith('IND'):
                return c
            parts = clean_ind_name(c).split('_')
            return f"{parts[0]}_{parts[1]}_{parts[-1]}" if len(parts) >= 3 else clean_ind_name(c)

        compositions_dict = {
            (div, _norm_ind(cand)): val
            for (div, cand), val in pref_bundle["compositions_dict"].items()
        }

        FP_pred = {
            div: df.rename(columns=_norm_ind).T.groupby(level=0).sum().T
            for div, df in FP_pred.items()
        }

        for key in compositions_dict:
            if key[1] not in FP_pred[key[0]].columns:
                print(key)
                print([p for p in FP_pred[key[0]].columns if p.startswith(f'IND_{key[1][4:5]}')])
                import pdb; pdb.set_trace()

        # 5. Precompute adjusted_TCP_state_dict ONCE for this election year
        adjusted_TCP_state_dict, _, _ = adjust_TCP_state_dict(
            TCP_state_dict=TCP_state_dict,
            party_df=party_df,
            SIGMA_JOINT_BASE=0.045,
            election_year=election_year
        )
        _, n_sources, external_rows = get_tcp_election_metadata(election_year)

        PART2_BUNDLES_BY_YEAR[yr_str] = {
            "FP_pred": FP_pred,
            "LC_FPs_df": LC_FPs_df_active,
            "sim_poll_errors": sim_poll_errors,
            "sim_local_alr_dict": sim_local_alr_dict,
            "party_df": party_df,
            "final_model": final_model,
            "compositions_dict": compositions_dict,
            "specific_defection_dict": specific_defection_dict,
            "fallback_defection_profile": fallback_defection_profile,
            "adjusted_TCP_state_dict": adjusted_TCP_state_dict,
            "n_sources": n_sources,
            "external_rows": external_rows,
            "data_year": data_year,
            "actual_winners_dict": actual_winners_dict
        }
        print(f"Loaded Part 2 bundle for {election_year} ({len(ordered_electorates)} electorates, {num_sims} sims)")

    best_loo, ranking_df, cal_best, cal_oos = validate_part2_tcp_params(
        PART2_BUNDLES_BY_YEAR,
        sigma_tot_grid=[0.020, 0.030, 0.040, 0.048, 0.0541, 0.062, 0.070, 0.080, 0.095, 0.110],
        joint_to_ind_ratio_grid=[1.5],  # Locks ratio at 0.045 / 0.030 = 1.5
        n_processes=24
    )


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
    elif election_year == 2022: # assuming 40% HTV loyalty
        UPDATED_FLOWS_ELECTION = { 
            3: {'LP': 30.0, 'NP': 34.0},
          }
    elif election_year == 2010:
        UPDATED_FLOWS_ELECTION = {
            3: {'LP': 62.0, 'NP': 65.0},
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
    import pdb; pdb.set_trace()
    # 3. Build the noise matrix
    row_joint_sigmas = build_joint_noise_vector(N_SOURCES, EXTERNAL_ROWS, SIGMA_JOINT_BASE, EXTERNAL_UNCERTAINTY_PENALTY)

    return adjusted_TCP_state_dict, row_joint_sigmas, avg_tcp_df

sigma_joint = 0.045
adjust_TCP_state_dict(TCP_state_dict, party_df, sigma_joint)

import pdb; pdb.set_trace()