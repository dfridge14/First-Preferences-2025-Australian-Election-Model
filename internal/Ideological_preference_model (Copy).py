import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import time
import copy

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize
from collections import defaultdict

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

from DOPTable import DOPTable


LC_results = pd.read_csv(f'VIC-{data_year}-LC-Votes-Electorate-Formatted.csv')
LA_results = pd.read_csv(f'VIC-{data_year}-LA-Primary-Electorate-Formatted.csv')

LA_Candidates = pd.read_csv(f'VIC-{data_year}-LA-Candidates-Formatted.csv')

# create df matching party code to party name
party_df = pd.read_csv(f'VIC-{data_year}-Parties.csv')
Ideology_map = party_df.set_index('PartyAb')['Ideo_category']
IDEO_CATEGORIES = ['ALP','COAL','Left','Right','Centre']

DEEPEST_CAND_NO = 4

with open(f'{data_year}_complete_DOP_table_dict.pkl', 'rb') as f:
    DOP_table_dict = pickle.load(f)
    if data_year == '2022':
        del DOP_table_dict['Narracan']

with open('ia_adjustments_dict.pkl', 'rb') as f:
    ia_adjustments_dict = pickle.load(f)

LC_FPs_df = pd.read_csv(f'{data_year}-LC-First-Prefs-df.csv', index_col = 'PartyAb')


def get_party_bloc(party: str, party_df: pd.DataFrame) -> str:
    """Helper to extract ideological bloc for any party in party_df."""
    if str(party).startswith('IND'):
        return 'Centre'
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    return ideo_map.get(party, 'Centre')

def generate_latent_training_tables(DOP_table_dict, ia_adjustments_dict):
    """
    Executes Phase 1: Cleansing
    """
    latent_tables = {}
    
    for div, df in DOP_table_dict.items():
        table = DOPTable(df)
        
        # 1. Convert to percentages if not already
        total_votes = sum(table.count_0.values())
        table.count_0 = {k: v / total_votes for k, v in table.count_0.items()}
        for rnd in table.rounds:
            rnd['V_elim'] /= total_votes
            rnd['transfers'] = {k: v / total_votes for k, v in rnd['transfers'].items()}
            
        # 2. Filter Electorates with massive Independent disruptions
        ind_cands = [c for c in table.candidates if str(c).startswith('IND')]
        ind_share = sum(table.count_0.get(c, 0) for c in ind_cands)
        
        if ind_share > 0.15:
            print(f"Skipping {div}: IND share {ind_share*100:.1f}% exceeds 15% threshold.")
            continue
            
        # 3. Strip Incumbency (if applicable to this seat)
        if div in ia_adjustments_dict:
            adjustments = ia_adjustments_dict[div] 
            # e.g., {'ALP_Smith': -0.03, 'LP_Jones': 0.02, 'GRN_Doe': 0.01}
            table = table.strip_incumbency(adjustments)
            
        # 4. Cleanse INDs symmetrically
        if ind_cands:
            table = table.remove_candidates(ind_cands)
            
        latent_tables[div] = table
        
    return latent_tables


if 0:

    print("\n--- TRACKING BASS COUNT 0 MUTATION ---")

    # 1. Raw VEC Data
    bass_df = DOP_table_dict['Bass']
    test_table = DOPTable(bass_df)
    print("\n1. RAW VEC VOTES (Absolute):")
    print(test_table.count_0)

    # 2. Percentage Conversion
    total = sum(test_table.count_0.values())
    test_table.count_0 = {k: v / total for k, v in test_table.count_0.items()}
    for rnd in test_table.rounds:
        rnd['V_elim'] /= total
        rnd['transfers'] = {k: v / total for k, v in rnd['transfers'].items()}
        
    print("\n2. AFTER PERCENTAGE CONVERSION:")
    print({k: f"{v:.4%}" for k, v in test_table.count_0.items()})

    # 3. Strip Incumbency
    if 'Bass' in ia_adjustments_dict:
        test_table = test_table.strip_incumbency(ia_adjustments_dict['Bass'])
        print("\n3. AFTER INCUMBENCY STRIPPING:")
        print({k: f"{v:.4%}" for k, v in test_table.count_0.items()})
    else:
        print("\n3. AFTER INCUMBENCY STRIPPING: (No adjustments for Bass)")

    # 4. Remove Independents
    ind_cands = [c for c in test_table.candidates if str(c).startswith('IND')]
    if ind_cands:
        test_table = test_table.remove_candidates(ind_cands)
        print(f"\n4. AFTER REMOVING {ind_cands}:")
        print({k: f"{v:.4%}" for k, v in test_table.count_0.items()})
    else:
        print("\n4. AFTER REMOVING INDS: (No INDs in Bass)")
        
    print("------------------------------------------")

    import pdb; pdb.set_trace()




# Define our fixed ideological arrays
BLOCS = ["ALP", "COAL", "Left", "Right", "Centre"]
ORPHAN_POOLS = ["Left", "Right", "Centre"]

@dataclass
class CumulativeObservation:
    district: str
    count_index: int
    active_cands: List[str]
    empirical_shares: Dict[str, float]
    lc_mass: np.ndarray          # 5-element array (ALP, COAL, Left, Right, Centre)
    c_cap: np.ndarray            # 5-element array
    menu_size: np.ndarray        # 5-element array
    cand_to_bloc: Dict[str, int]
    cand_demand: Dict[str, float]

def build_cumulative_lc_sequences(
    latent_tables: dict,
    party_df: pd.DataFrame,
    lc_df: pd.DataFrame,
    deepest_cand_no: int = 4
) -> List[CumulativeObservation]:
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    def get_bloc(c): return 'Centre' if str(c).startswith('IND') else ideo_map.get(c, 'Centre')

    observations = []

    for district, table in latent_tables.items():
        if district not in lc_df.columns:
            continue

        lc_series = lc_df[district].dropna().copy()
        if lc_series.sum() > 1.5:
            lc_series = lc_series / 100.0

        lc_mass = np.zeros(5)
        for party, share in lc_series.items():
            lc_mass[BLOCS.index(get_bloc(party))] += share
        if lc_mass.sum() > 0:
            lc_mass = lc_mass / lc_mass.sum()

        # --- STATE RECONSTRUCTION EXTRACTION ---
        counts_dict = {}
        if hasattr(table, 'count_0') and isinstance(table.count_0, dict):
            current_state = table.count_0.copy()
            counts_dict[0] = current_state.copy()
            
            if hasattr(table, 'rounds') and isinstance(table.rounds, list):
                for i, rnd in enumerate(table.rounds):
                    elim_cand = rnd.get('eliminated')
                    transfers = rnd.get('transfers', {})
                    
                    if elim_cand in current_state:
                        del current_state[elim_cand]
                        
                    for cand, val in transfers.items():
                        current_state[cand] = current_state.get(cand, 0.0) + val
                        
                    counts_dict[i + 1] = current_state.copy()
        
        if not counts_dict: 
            continue

        for count_idx in sorted(counts_dict.keys()):
            raw_shares = counts_dict[count_idx]
            active_cands = [c for c, v in raw_shares.items() if v > 1e-9]
            
            if len(active_cands) < deepest_cand_no:
                continue

            tot = sum(raw_shares[c] for c in active_cands)
            if tot <= 1e-9:
                continue
            emp_shares = {c: raw_shares[c] / tot for c in active_cands}

            menu_size = np.zeros(5)
            c_cap = np.zeros(5)
            cand_to_bloc = {}
            cand_demand = {}

            for cand in active_cands:
                b_idx = BLOCS.index(get_bloc(cand))
                cand_to_bloc[cand] = b_idx
                menu_size[b_idx] += 1.0

                u_m = lc_series.get(cand, 0.0)
                
                if u_m <= 1e-9 and cand in ['LP', 'NP', 'LNP']:
                    coal_lc_mass = sum(lc_series.get(a, 0.0) for a in ['LP', 'NP', 'LNP'])
                    coal_la_cands = [c for c in active_cands if c in ['LP', 'NP', 'LNP']]
                    
                    if len(coal_la_cands) == 1:
                        u_m = coal_lc_mass
                    elif len(coal_la_cands) > 1:
                        emp_c = emp_shares.get(cand, 0.0)
                        emp_tot = sum(emp_shares.get(c, 0.0) for c in coal_la_cands)
                        u_m = coal_lc_mass * (emp_c / emp_tot) if emp_tot > 0 else (coal_lc_mass / len(coal_la_cands))

                cand_demand[cand] = u_m
                if b_idx in [2, 3, 4]:
                    c_cap[b_idx] += u_m

            c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
            c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
            for b in [2, 3, 4]:
                c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0

            observations.append(CumulativeObservation(
                district=district,
                count_index=count_idx,
                active_cands=active_cands,
                empirical_shares=emp_shares,
                lc_mass=lc_mass,
                c_cap=c_cap,
                menu_size=menu_size,
                cand_to_bloc=cand_to_bloc,
                cand_demand=cand_demand
            ))

    return observations






ORPHAN_POOLS = ['Left', 'Right', 'Centre']
BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']

# =====================================================================
# 1. TEST SCRIPT BLOCK
# =====================================================================
# Define the test subset
test_divs = ['Eureka', 'Bass', 'Bentleigh', 'Morwell', 'Bayswater']
test_dop_dict = {d: DOP_table_dict[d] for d in test_divs if d in DOP_table_dict}

print(f"--- STARTING TEST RUN ON {len(test_dop_dict)} ELECTORATES ---")

test_latent_tables = generate_latent_training_tables(
    DOP_table_dict=test_dop_dict,  
    ia_adjustments_dict=ia_adjustments_dict
)

print("\n[Extracting Cumulative State Observations...]")
test_observations = build_cumulative_lc_sequences(
    latent_tables=test_latent_tables, 
    party_df=party_df, 
    lc_df=LC_FPs_df
)

print(f"\n--- TEST COMPLETE: Successfully extracted {len(test_observations)} observations. ---")

# Deeply inspect the first observation to verify formatting
if test_observations:
    t = test_observations[0]
    
    print(f"\n==================================================")
    print(f"OBSERVATION INSPECTION: {t.district} (Count {t.count_index})")
    print(f"Global LC Base (True Demand): {np.round(t.lc_mass, 4)}")
    print(f"==================================================")
    
    print("\n--- STATE MENU ---")
    print(f"Active Menu ({len(t.active_cands)}): {t.active_cands}")
    print(f"Bloc Capacities: {np.round(t.c_cap, 4)}")
    print(f"Menu Sizes:      {t.menu_size}")
    
    print("\n--- GROUND TRUTH (The Cumulative Target) ---")
    print(f"Empirical LA Shares at this Count:")
    for cand, val in t.empirical_shares.items():
        if val > 0:
            print(f"  -> {cand:<6}: {val:.6%}")
    print(f"==================================================")


# =====================================================================
# 2. MODEL CLASS
# =====================================================================
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



def compute_cumulative_l2_loss(
    theta: np.ndarray,
    observations: List[CumulativeObservation],
    model: 'AggregateLatentLogit',
    l2_reg: float = 1e-4
) -> float:
    """
    Direct L2 Loss over cumulative LC -> Count X allocations.
    Algebraically identical to penalizing error in net transfer vectors:
    || S_hat(M_X) - S_emp(M_X) ||^2 == || Delta_hat(LC->X) - Delta_emp(LC->X) ||^2.
    """
    alpha, beta, gamma = model.unpack_params(theta)
    total_loss = 0.0

    for obs in observations:
        s_hat = model.evaluate_state_allocation(
            obs.lc_mass,
            obs.active_cands,
            obs.c_cap,
            obs.menu_size,
            obs.cand_to_bloc,
            obs.cand_demand,
            alpha,
            beta,
            gamma
        )

        emp = obs.empirical_shares.copy()
        pred = s_hat.copy()

        # Coalition Layer 3 Firewall (handles joint LP/NP dynamics)
        if ('LP' in emp or 'LP' in pred) and ('NP' in emp or 'NP' in pred):
            emp['COAL_COMBINED'] = emp.pop('LP', 0.0) + emp.pop('NP', 0.0)
            pred['COAL_COMBINED'] = pred.pop('LP', 0.0) + pred.pop('NP', 0.0)

        # Candidate-level L2 error
        cand_keys = set(emp.keys()) | set(pred.keys())
        obs_l2 = sum((pred.get(c, 0.0) - emp.get(c, 0.0)) ** 2 for c in cand_keys)
        total_loss += obs_l2

    # L2 regularization on affinity deviations
    total_loss += l2_reg * np.sum(alpha ** 2)
    return float(total_loss)


def compute_flow_proportional_l2_loss(
    theta: np.ndarray,
    observations: List[CumulativeObservation],
    model: 'AggregateLatentLogit',
    l2_reg: float = 1e-4
) -> float:
    """
    Flow-Proportional L2 Loss.
    Evaluates routing error as a percentage of the available orphan pool (1/O^2 weighting).
    Algebraically cancels out the LC baseline, seamlessly handling negative empirical flows.
    """
    alpha, beta, gamma = model.unpack_params(theta)
    total_loss = 0.0

    for obs in observations:
        s_hat = model.evaluate_state_allocation(
            obs.lc_mass,
            obs.active_cands,
            obs.c_cap,
            obs.menu_size,
            obs.cand_to_bloc,
            obs.cand_demand,
            alpha,
            beta,
            gamma
        )

        emp = obs.empirical_shares.copy()
        pred = s_hat.copy()

        # Coalition Layer 3 Firewall (handles joint LP/NP dynamics)
        if ('LP' in emp or 'LP' in pred) and ('NP' in emp or 'NP' in pred):
            emp['COAL_COMBINED'] = emp.pop('LP', 0.0) + emp.pop('NP', 0.0)
            pred['COAL_COMBINED'] = pred.pop('LP', 0.0) + pred.pop('NP', 0.0)

        # 1. Calculate the true volume of the Orphan Pool (O) for this count
        # (The mass from Left, Right, Centre that is orphaned due to capacity limits)
        orphan_vol = 0.0
        for b in [2, 3, 4]:
            orphan_vol += obs.lc_mass[b] * max(0.0, 1.0 - obs.c_cap[b])
        
        # Floor O at 1% to prevent scaling explosions in near-perfect Count 0 states
        O_safe = max(orphan_vol, 0.01)

        # 2. Evaluate candidate-level error relative to the Orphan Pool size
        cand_keys = set(emp.keys()) | set(pred.keys())
        obs_l2 = 0.0
        
        for c in cand_keys:
            diff = pred.get(c, 0.0) - emp.get(c, 0.0)
            # The error is (Absolute Difference / Orphan Pool Size)^2
            obs_l2 += (diff / O_safe) ** 2
            
        total_loss += obs_l2

    # Ridge regularization on affinity deviations
    total_loss += l2_reg * np.sum(alpha ** 2)
    return float(total_loss)
# =====================================================================
# 3. OPTIMIZATION & FITTING FUNCTIONS
# =====================================================================
def fit_preference_model(
    observations: list,
    n_starts: int = 15,
    l2_reg: float = 1e-4,
    random_seed: int = 42,
) -> Tuple[AggregateLatentLogit, Tuple[pd.DataFrame, pd.Series]]:
    np.random.seed(random_seed)
    model = AggregateLatentLogit()

    bounds = []
    for _ in range(12): bounds.append((-6.0, 6.0))
    for _ in range(3): bounds.append((0.0, 5.0))
    bounds.append((0.1, 1.0)) # Gamma

    priors = model.get_default_priors()
    starting_points = [priors]

    for _ in range(n_starts - 1):
        jitter_alphas = np.random.uniform(-1.0, 1.0, size=12)
        jitter_betas  = np.random.uniform(-0.1, 0.1, size=3)
        jitter_gamma  = np.random.uniform(-0.05, 0.05, size=1)
        
        jitter = np.concatenate([jitter_alphas, jitter_betas, jitter_gamma])
        start = np.clip(priors + jitter, [b[0] for b in bounds], [b[1] for b in bounds])
        starting_points.append(start)

    best_loss = np.inf
    best_theta = None

    print(f"--- Beginning Multi-Start Optimization ({n_starts} starts, {len(observations)} observations) ---")
    start_time = time.time()

    for idx, x0 in enumerate(starting_points):
        res = minimize(
            fun=compute_flow_proportional_l2_loss,
            x0=x0,
            args=(observations, model, l2_reg),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-7, "disp": False},
        )

        status_str = "CONVERGED" if res.success else "ITER_LIMIT"
        print(f"  Start {idx+1:02d}/{n_starts:02d} | Loss: {res.fun:.6f} | Status: {status_str}")

        if res.fun < best_loss:
            best_loss = res.fun
            best_theta = res.x

    elapsed = time.time() - start_time
    print(f"--- Optimization Complete in {elapsed:.2f}s | Best Loss: {best_loss:.6f} ---\n")

    model.set_params(best_theta)
    alpha_df, global_params = format_parameter_summary(best_theta)

    return model, (alpha_df, global_params)

def format_parameter_summary(theta: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
    model = AggregateLatentLogit()
    alpha, beta, gamma = model.unpack_params(theta)

    rows = []
    for p_idx, pool_name in enumerate(ORPHAN_POOLS):
        row = {
            "Orphan_Pool": pool_name,
            "Alpha_ALP (Ref)": 0.0,
            "Alpha_COAL": alpha[p_idx, 1],
            "Alpha_Left": alpha[p_idx, 2],
            "Alpha_Right": alpha[p_idx, 3],
            "Alpha_Centre": alpha[p_idx, 4],
        }
        rows.append(row)
    alpha_df = pd.DataFrame(rows).set_index("Orphan_Pool")

    global_params = pd.Series({
        "Beta_Mass (Attraction to LC Base)": beta[0],
        "Beta_Cap (Sensitivity to Truncation)": beta[1],
        "Beta_Menu (Attraction to Options)": beta[2],
        "Gamma (Sub-proportionality)": gamma
    })

    return alpha_df, global_params


# =====================================================================
# 4. DIAGNOSTICS & BOOTSTRAP
# =====================================================================
def run_clustered_bootstrap(observations: list, n_iterations: int = 100, n_starts: int = 3):
    print(f"\n=== INITIATING CLUSTERED BOOTSTRAP ({n_iterations} Iterations) ===")
    
    electorate_clusters = defaultdict(list)
    for obs in observations:
        electorate_clusters[obs.district].append(obs)
        
    districts = list(electorate_clusters.keys())
    n_districts = len(districts)
    print(f"Clustering complete: {n_districts} unique electorates found.")

    bootstrap_results = []
    start_time = time.time()
    
    for i in range(n_iterations):
        sampled_districts = np.random.choice(districts, size=n_districts, replace=True)
        
        resampled_observations = []
        for d in sampled_districts:
            resampled_observations.extend(electorate_clusters[d])
            
        model, _ = fit_preference_model(
            observations=resampled_observations, 
            n_starts=n_starts, 
            random_seed=np.random.randint(0, 100000)
        )
        
        bootstrap_results.append(model.theta)
        if (i + 1) % 10 == 0:
            print(f"  Bootstrap Progress: {i + 1}/{n_iterations} completed ({time.time() - start_time:.1f}s)")

    bootstrap_array = np.vstack(bootstrap_results) 
    means = np.mean(bootstrap_array, axis=0)
    ci_lower = np.percentile(bootstrap_array, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_array, 97.5, axis=0)
    
    param_names = [
        "Left_Alpha_COAL", "Left_Alpha_Left", "Left_Alpha_Right", "Left_Alpha_Centre",
        "Right_Alpha_COAL", "Right_Alpha_Left", "Right_Alpha_Right", "Right_Alpha_Centre",
        "Centre_Alpha_COAL", "Centre_Alpha_Left", "Centre_Alpha_Right", "Centre_Alpha_Centre",
        "Beta_Mass", "Beta_Cap", "Beta_Menu", "Gamma"
    ]
    
    results_df = pd.DataFrame({
        "Parameter": param_names,
        "Mean": means,
        "95%_CI_Lower": ci_lower,
        "95%_CI_Upper": ci_upper
    }).set_index("Parameter")
    
    print("\n=== FINAL BOOTSTRAP PARAMETER ESTIMATES ===")
    print(results_df.round(4))
    
    return results_df

def inspect_electorate_fit(district_name: str, model: AggregateLatentLogit, observations: list):
    """
    Prints a count-by-count audit for an electorate across all available elimination counts.
    """
    dist_obs = [o for o in observations if o.district == district_name]
    dist_obs.sort(key=lambda x: x.count_index)
    
    if not dist_obs:
        print(f"\n[!] No observation data found for '{district_name}'.")
        return

    alpha, beta, gamma = model.unpack_params(model.theta)

    print(f"\n==========================================================================")
    print(f"  CUMULATIVE FIT AUDIT (LC -> ALL COUNTS): {district_name.upper()}")
    print(f"  Total Counts Extracted: {len(dist_obs)} (Count 0 through Count {dist_obs[-1].count_index})")
    print(f"==========================================================================")

    for obs in dist_obs:
        s_hat = model.evaluate_state_allocation(
            obs.lc_mass, obs.active_cands, obs.c_cap, obs.menu_size,
            obs.cand_to_bloc, obs.cand_demand, alpha, beta, gamma
        )
        
        print(f"\n--- [COUNT {obs.count_index}] Active Candidates: {len(obs.active_cands)} ---")
        print(f"{'Candidate':<10} | {'Empirical LA':<16} | {'Predicted (Ŝ)':<16} | {'Residual':<10}")
        print("-" * 60)
        
        total_sq_err = 0.0
        for cand in sorted(obs.active_cands):
            act = obs.empirical_shares.get(cand, 0.0)
            pred = s_hat.get(cand, 0.0)
            err = pred - act
            total_sq_err += err ** 2
            print(f"{cand:<10} | {act:>16.4%} | {pred:>16.4%} | {err:>+10.4%}")
            
        rmse = np.sqrt(total_sq_err / len(obs.active_cands))
        print(f"-> Count {obs.count_index} Overall RMSE: {rmse:.4%}")
        
    print("\n" + "=" * 74 + "\n")





# --- SNEAK PEEK USING EXISTING TRANSITION DATA ---
full_latent_tables = generate_latent_training_tables(
            DOP_table_dict=DOP_table_dict, 
            ia_adjustments_dict=ia_adjustments_dict
        )

full_transitions = build_cumulative_lc_sequences(
    latent_tables=full_latent_tables, 
    party_df=party_df, 
    lc_df=LC_FPs_df
)



# =====================================================================
# EXECUTION BLOCK
# =====================================================================
if 1:
    print("--- EXTRACTING ALL ELECTORATES ---")
    full_latent_tables = generate_latent_training_tables(
        DOP_table_dict=DOP_table_dict, 
        ia_adjustments_dict=ia_adjustments_dict
    )

    full_obs = build_cumulative_lc_sequences(
        latent_tables=full_latent_tables, 
        party_df=party_df, 
        lc_df=LC_FPs_df
    )

    THETA_FILE = 'model_theta_saved.npy'

    # Instantiate the empty model
    final_model = AggregateLatentLogit()

    if os.path.exists(THETA_FILE):
        print(f"Loading pre-trained parameters from {THETA_FILE}...")
        final_model.theta = np.load(THETA_FILE)
    else:
        print("No saved parameters found. Running L-BFGS-B optimization...")
        
        # Run the optimizer
        final_model, (alpha_df, globals_df) = fit_preference_model(
            observations=full_obs,
            n_starts=10, 
            random_seed=42
        )
    
        # Save the parameters to disk for next time
        np.save(THETA_FILE, final_model.theta)
        print(f"Optimization complete. Parameters saved to {THETA_FILE}.")


        print("\n=== IDEOLOGICAL AFFINITIES (ALPHAS) ===")
        print(alpha_df.round(4))
        print("\n=== SHARED STRUCTURAL PARAMETERS ===")
        print(globals_df.round(4))

    # Run the diagnostics on the final model using the full dataset!
    inspect_electorate_fit('Eureka', final_model, full_obs)
    inspect_electorate_fit('Bass', final_model, full_obs)
    inspect_electorate_fit('Bentleigh', final_model, full_obs)
    inspect_electorate_fit('Macedon', final_model, full_obs)
    inspect_electorate_fit('Melbourne', final_model, full_obs)



    print("\n--- RUNNING CLUSTERED BOOTSTRAP ---")
    # Taking it down to 50 iterations with 2 starts just to make it run faster for your first full test
    #bootstrap_summary = run_clustered_bootstrap(full_transitions, n_iterations=50, n_starts=2)

    # Save everything
    #with open("final_victorian_model.pkl", "wb") as f:
    #    pickle.dump(final_model, f)
    #bootstrap_summary.to_csv("parameter_confidence_intervals.csv")






def inject_lc_party_to_dop_old(
    district: str,
    cand_to_add: str,
    target_bloc: str,
    empirical_c0: Dict[str, float],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
    calibration_scalar: float = 1.0
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Injects an LC-contesting party into an existing LA Count 0 table.
    Uses Multiplicative Retention Ratios to ensure strict non-negativity,
    compositional coherence, and preservation of local candidate baselines.
    """
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    def get_bloc(c): return 'Centre' if str(c).startswith('IND') else ideo_map.get(c, 'Centre')

    lc_series = lc_df[district].dropna().copy()
    if lc_series.sum() > 1.5:
        lc_series = lc_series / 100.0

    lc_mass = np.zeros(5)
    for party, share in lc_series.items():
        lc_mass[BLOCS.index(get_bloc(party))] += share
    if lc_mass.sum() > 0:
        lc_mass = lc_mass / lc_mass.sum()

    cand_pure_lc = lc_series.get(cand_to_add, 0.0)
    if cand_pure_lc <= 1e-9:
        raise ValueError(f"Candidate {cand_to_add} has 0.0 LC demand in {district}.")

    menu_A = [c for c, v in empirical_c0.items() if v > 1e-9 and c != cand_to_add]
    menu_B = menu_A + [cand_to_add]

    def build_state_vars(active_cands: List[str]):
        menu_size = np.zeros(5)
        c_cap = np.zeros(5)
        cand_to_bloc = {}
        cand_demand = {}

        for cand in active_cands:
            b = BLOCS.index(get_bloc(cand))
            cand_to_bloc[cand] = b
            menu_size[b] += 1.0
            u = cand_pure_lc if cand == cand_to_add else lc_series.get(cand, 0.0)
            if u <= 1e-9 and b in [0, 1]:
                u = empirical_c0.get(cand, 0.0)
            cand_demand[cand] = u
            if b in [2, 3, 4]:
                c_cap[b] += u

        c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
        c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
        for b in [2, 3, 4]:
            c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0

        return menu_size, c_cap, cand_to_bloc, cand_demand

    ms_A, cap_A, ctb_A, cd_A = build_state_vars(menu_A)
    ms_B, cap_B, ctb_B, cd_B = build_state_vars(menu_B)

    alpha, beta, gamma = model.unpack_params(model.theta)
    s_hat_A = model.evaluate_state_allocation(lc_mass, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)
    s_hat_B = model.evaluate_state_allocation(lc_mass, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)

    # 1. Compute Multiplicative Retention Ratios: R[c] = S_hat_B[c] / S_hat_A[c]
    # 2. Scale Injected Party by Calibration Scalar
    unnorm_shares = {}
    for c in menu_A:
        retention_ratio = s_hat_B.get(c, 0.0) / s_hat_A[c] if s_hat_A.get(c, 0.0) > 0 else 1.0
        # Retention ratio is strictly capped at 1.0 (parties only lose mass on entrant addition)
        retention_ratio = min(1.0, max(0.0, retention_ratio))
        unnorm_shares[c] = empirical_c0[c] * retention_ratio

    # Apply calibrated entrant share
    unnorm_shares[cand_to_add] = s_hat_B.get(cand_to_add, 0.0) * calibration_scalar

    # 3. Close the simplex (normalize to 1.0)
    total_unnorm = sum(unnorm_shares.values())
    new_c0 = {c: v / total_unnorm for c, v in unnorm_shares.items()}

    # Compute net change relative to empirical baseline
    shift_vector = {}
    for c in menu_B:
        shift_vector[c] = new_c0[c] - empirical_c0.get(c, 0.0)

    return new_c0, shift_vector, s_hat_A, s_hat_B


# ==============================================================================
# 2. RECALIBRATE / DIAGNOSE A SPECIFIC PARTY (e.g. One Nation)
# ==============================================================================

def diagnose_and_recalibrate_party_old(
    party_abbr: str,
    contested_divs: List[str],
    latent_tables: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit'
) -> Tuple[pd.DataFrame, float]:
    
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    def get_bloc(c): return 'Centre' if str(c).startswith('IND') else ideo_map.get(c, 'Centre')

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
            lc_mass[BLOCS.index(get_bloc(party))] += share
        if lc_mass.sum() > 0:
            lc_mass = lc_mass / lc_mass.sum()

        menu = list(table.count_0.keys())
        menu_size = np.zeros(5)
        c_cap = np.zeros(5)
        cand_to_bloc = {}
        cand_demand = {}

        for cand in menu:
            b = BLOCS.index(get_bloc(cand))
            cand_to_bloc[cand] = b
            menu_size[b] += 1.0
            
            # --- CORRECTED COALITION ALIAS FIX ---
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
    calibration_scalar = float(diag_df['Realization_Ratio'].median()) if not diag_df.empty else 1.0
    return diag_df, calibration_scalar

def trace_electorate_mechanics(district: str, model: 'AggregateLatentLogit', transitions: list):
    """
    Exposes the step-by-step internal math of the model for a given electorate.
    Shows exactly how Alpha, Beta, and Gamma are applied at every elimination round.
    """
    dist_trans = [t for t in transitions if t.district == district]
    if not dist_trans:
        print(f"No transitions found for {district}.")
        return
        
    alpha, beta, gamma = model.unpack_params(model.theta)
    b_mass, b_cap, b_menu = beta
    
    print(f"=========================================================")
    print(f" DEEP DIAGNOSTIC TRACE: {district.upper()}")
    print(f" Parameters: b_mass={b_mass:.4f}, b_cap={b_cap:.4f}, b_menu={b_menu:.4f}, gamma={gamma:.4f}")
    print(f"=========================================================\n")

    for t in dist_trans:
        print(f"--- ROUND {t.count_index} (Eliminating {t.eliminated_cand}) ---")
        
        # We trace State X (Pre-Elimination) to see how the model calculates S_hat_X
        active = t.active_cands
        cd = t.cand_demand_X
        ctb = t.cand_to_bloc_X
        
        print("1. INTRA-BLOC SHARES (The Gamma Effect):")
        for b_idx, b_name in enumerate(['ALP', 'COAL', 'Left', 'Right', 'Centre']):
            bloc_cands = [c for c in active if ctb[c] == b_idx]
            if not bloc_cands: continue
            
            eff_g = 1.0 if b_idx in [0, 1] else gamma
            sum_d = sum(cd[c] ** eff_g for c in bloc_cands)
            
            print(f"   Bloc {b_name} (eff_gamma={eff_g:.2f}):")
            for c in bloc_cands:
                share = (cd[c] ** eff_g) / sum_d
                print(f"     -> {c}: LC Demand = {cd[c]:.4f} | Intra-Share = {share:.2%}")

        print("\n2. ROUTING PROBABILITIES (Orphan Pools -> Blocs):")
        # Calculate exactly as the model does
        log_mass = np.log(np.maximum(t.lc_mass, 1e-6))
        log_cap  = np.log(np.maximum(t.c_cap_X, 1e-6))
        log_menu = np.log(np.maximum(t.menu_size_X, 1e-6))
        
        u_struc = (b_mass * log_mass) + (b_cap * log_cap) + (b_menu * log_menu)
        
        for p_idx, pool in enumerate(['Left', 'Right', 'Centre']):
            # Orphan pool index is 0, 1, 2 representing Left, Right, Centre
            v_matrix = alpha[p_idx] + u_struc
            unavailable = t.menu_size_X <= 0
            v_matrix[unavailable] = -1e9
            
            # Softmax
            exp_v = np.exp(v_matrix - np.max(v_matrix))
            exp_v[unavailable] = 0.0
            probs = exp_v / np.sum(exp_v)
            
            print(f"   {pool} Orphans route to:")
            for b_idx, b_name in enumerate(['ALP', 'COAL', 'Left', 'Right', 'Centre']):
                if not unavailable[b_idx]:
                    print(f"     -> {b_name:<6}: {probs[b_idx]:>6.2%} (Utility: {v_matrix[b_idx]:.2f})")
                    
        print("\n3. FINAL PREDICTED DELTA VS EMPIRICAL DELTA:")
        delta_hat = model.predict_transition_delta(t, alpha, beta, gamma)
        sum_pred = sum(delta_hat.values())
        sum_act = t.empirical_total_transferred
        
        for c in t.active_cands_X1:
            pred_share = delta_hat.get(c, 0.0) / sum_pred if sum_pred > 0 else 0.0
            act_share = t.empirical_transfers.get(c, 0.0) / sum_act if sum_act > 0 else 0.0
            print(f"   {c:<8} | Pred: {pred_share:>6.2%} | Act: {act_share:>6.2%} | Err: {pred_share-act_share:>+6.2%}")
        print("\n")


def trace_injection_mechanics_detailed(
    district: str,
    cand_to_add: str,
    target_bloc: str,
    empirical_c0: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
    calibration_scalar: float = 1.0
):
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    alpha, beta, gamma = model.unpack_params(model.theta)
    b_mass, b_cap, b_menu = beta
    
    # 1. Parse Upper House (LC) primary votes
    lc_series = lc_df[district].dropna().copy()
    if lc_series.sum() > 1.5: 
        lc_series = lc_series / 100.0
        
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    def get_bloc(c): return 'Centre' if str(c).startswith('IND') else ideo_map.get(c, 'Centre')

    # Sum total LC mass by bloc (strictly sums to 1.0)
    lc_mass = np.zeros(5)
    for party, share in lc_series.items():
        lc_mass[BLOCS.index(get_bloc(party))] += share
    lc_mass = lc_mass / lc_mass.sum()

    raw_cand_lc = lc_series.get(cand_to_add, 0.0)
    cand_effective_lc = raw_cand_lc * calibration_scalar
    
    menu_A = [c for c, v in empirical_c0.items() if v > 1e-9 and c != cand_to_add]
    menu_B = menu_A + [cand_to_add]

    print("=" * 80)
    print(f" MATHEMATICAL AUDIT TRACE: {district.upper()} (+ {cand_to_add})")
    print(f" Model Parameters: b_mass={b_mass:.4f}, b_cap={b_cap:.4f}, b_menu={b_menu:.4f}, gamma={gamma:.4f}")
    print(f" Candidate Demand: Raw LC={raw_cand_lc:.4%} * Scalar={calibration_scalar:.4f} -> Effective={cand_effective_lc:.4%}")
    print(" Total LC Mass by Bloc (sums to 100%):")
    for b_name, m in zip(BLOCS, lc_mass):
        print(f"   {b_name:<8}: {m:.4%}")
    print("=" * 80 + "\n")

    def run_pure_latent_state(menu, state_label):
        print(f"--- EVALUATING STATE {state_label} (Candidates: {len(menu)}) ---")
        menu_size = np.zeros(5)
        c_cap = np.zeros(5)
        cd = {}
        ctb = {}
        
        # Candidate Demand is strictly LC demand
        for cand in menu:
            b = BLOCS.index(get_bloc(cand))
            ctb[cand] = b
            menu_size[b] += 1.0
            
            # Identify injected candidate vs baseline candidates
            if cand == cand_to_add:
                u = cand_effective_lc # (Or cand_pure_lc in inject_function)
            else:
                u = lc_series.get(cand, 0.0)
                
                # --- THE STRICT COALITION ALIAS FIX ---
                if u <= 1e-9 and cand in ['LP', 'NP', 'LNP']:
                    coal_lc_mass = sum(lc_series.get(a, 0.0) for a in ['LP', 'NP', 'LNP'])
                    coal_la_cands = [c for c in menu if c in ['LP', 'NP', 'LNP']]
                    
                    if len(coal_la_cands) == 1:
                        u = coal_lc_mass
                    elif len(coal_la_cands) > 1:
                        emp_c = empirical_c0.get(cand, 0.0)
                        emp_tot = sum(empirical_c0.get(c, 0.0) for c in coal_la_cands)
                        u = coal_lc_mass * (emp_c / emp_tot) if emp_tot > 0 else coal_lc_mass / len(coal_la_cands)
                        
            cd[cand] = u
            if b in [2, 3, 4]:
                c_cap[b] += u
            
        c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
        c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
        for b in [2, 3, 4]:
            c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0
            
        # 1. Native vs Orphan Breakdown
        orphan_pools = np.zeros(5)
        for b in [2, 3, 4]:
            orphan_pools[b] = lc_mass[b] * max(0.0, 1.0 - c_cap[b])
            
        print("1. Capacity & Orphan Volumes (sums to 100%):")
        for b, name in enumerate(BLOCS):
            active_vol = lc_mass[b] * c_cap[b]
            orph_vol = orphan_pools[b]
            print(f"   {name:<8}: Active Base = {active_vol:>8.4%} | Orphan Pool = {orph_vol:>8.4%} (Total LC = {lc_mass[b]:>8.4%})")

        # 2. Softmax Routing Matrix
        log_mass = np.log(np.maximum(lc_mass, 1e-6))
        log_cap  = np.log(np.maximum(c_cap, 1e-6))
        log_menu = np.log(np.maximum(menu_size, 1e-6))
        u_struc = (b_mass * log_mass) + (b_cap * log_cap) + (b_menu * log_menu)
        
        routing_probs = np.zeros((3, 5))
        incoming_orphans = np.zeros(5)
        
        print("\n2. Orphan Routing Probabilities & Absolute Flows:")
        for p_idx, pool_name in enumerate(['Left', 'Right', 'Centre']):
            p_bloc = p_idx + 2
            v = orphan_pools[p_bloc]
            if v <= 1e-9: 
                continue
            
            u_vec = alpha[p_idx] + u_struc
            u_vec[menu_size <= 0] = -1e9
            exp_u = np.exp(u_vec - np.max(u_vec))
            exp_u[menu_size <= 0] = 0.0
            probs = exp_u / np.sum(exp_u)
            routing_probs[p_idx] = probs
            
            print(f"   From {pool_name} Pool ({v:.4%}):")
            for dest_b, dest_name in enumerate(BLOCS):
                if menu_size[dest_b] > 0:
                    flow = v * probs[dest_b]
                    incoming_orphans[dest_b] += flow
                    print(f"     -> {dest_name:<8}: {probs[dest_b]:>6.2%} (Flow = +{flow:.4%})")

        # 3. Candidate Allocation (Native + Incoming Orphans)
        s_hat = {}
        print("\n3. Intra-Bloc Dispersal (Gamma Application):")
        for b, name in enumerate(BLOCS):
            bloc_cands = [c for c in menu if ctb[c] == b]
            if not bloc_cands: 
                continue
            
            eff_g = 1.0 if b in [0, 1] else gamma
            sum_d = sum(cd[c] ** eff_g for c in bloc_cands)
            
            for c in bloc_cands:
                share = (cd[c] ** eff_g) / sum_d
                native_part = cd[c] if b in [2, 3, 4] else lc_mass[b]
                orphan_part = incoming_orphans[b] * share
                total_cand = native_part + orphan_part
                s_hat[c] = total_cand
                print(f"   {c:<8}: Share={share:>6.2%} | Native={native_part:>8.4%} + Orphans={orphan_part:>8.4%} -> S_hat={total_cand:>8.4%}")
                
        print(f" Total State Allocation Sum: {sum(s_hat.values()):.6f} (Must equal 1.0)\n")
        return s_hat

    s_hat_A = run_pure_latent_state(menu_A, "A (WITHOUT INJECTED CANDIDATE)")
    print("-" * 80 + "\n")
    s_hat_B = run_pure_latent_state(menu_B, "B (WITH INJECTED CANDIDATE)")

    # 4. Final Differencing & Application to Empirical Table
    print("=" * 80)
    print(" RECONSTRUCTED COUNTERFACTUAL DOP TABLE")
    print(" Formula: New C0 = Empirical C0 + (S_hat_B - S_hat_A)")
    print(f" {'Candidate':<10} | {'Empirical C0':<14} | {'S_hat_A':<12} | {'S_hat_B':<12} | {'Delta':<12} | {'New C0':<12}")
    print("-" * 80)
    
    total_delta = 0.0
    total_new = 0.0
    for c in menu_B:
        emp = empirical_c0.get(c, 0.0)
        sha_a = s_hat_A.get(c, 0.0)
        sha_b = s_hat_B.get(c, 0.0)
        delta = sha_b - sha_a
        new_c0 = emp + delta
        total_delta += delta
        total_new += new_c0
        print(f" {c:<10} | {emp:>14.4%} | {sha_a:>12.4%} | {sha_b:>12.4%} | {delta:>+12.4%} | {new_c0:>12.4%}")
        
    print("-" * 80)
    print(f" Check: Sum of Deltas = {total_delta:>+.6f} | Sum of New C0 = {total_new:.6f}")
    print("=" * 80)
    
       




































def inject_parties_to_dop(
    district: str,
    cands_to_add: List[str],
    empirical_c0: Dict[str, float],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
    calibration_scalars: Dict[str, float] = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Injects one or more LC-contesting parties into an existing LA Count 0 table.
    
    Architecture:
      1. Inter-Bloc MRR: Scales empirical bloc volumes by model-predicted structural shifts.
      2. Entrant Extraction: Entrants extract their kappa-scaled share from their bloc.
      3. Empirical-Residual Distribution: Remaining bloc volume is distributed to incumbents
         strictly proportional to their original empirical baseline.
    
    Guarantees:
      - Simplex closure (sum = 1.0)
      - Strict incumbent non-expansion (S_CF[c] <= S_emp[c])
      - Invariant empirical ratios among incumbents
    """
    if calibration_scalars is None:
        calibration_scalars = {c: 1.0 for c in cands_to_add}

    # Clean and normalize Upper House (LC) series
    lc_series = lc_df[district].dropna().copy()
    if lc_series.sum() > 1.5:
        lc_series = lc_series / 100.0

    lc_mass = np.zeros(5)
    for party, share in lc_series.items():
        b_idx = BLOCS.index(get_party_bloc(party, party_df))
        lc_mass[b_idx] += share
    if lc_mass.sum() > 0:
        lc_mass = lc_mass / lc_mass.sum()

    # Verify all entrants have non-zero Upper House demand
    for cand in cands_to_add:
        if lc_series.get(cand, 0.0) <= 1e-9:
            raise ValueError(f"Candidate '{cand}' has 0.0 LC demand in district '{district}'.")

    # Define Menus A (baseline) and B (counterfactual)
    menu_A = [c for c, v in empirical_c0.items() if v > 1e-9 and c not in cands_to_add]
    menu_B = menu_A + [c for c in cands_to_add if c not in menu_A]

    def build_state_vars(active_cands: List[str]):
        menu_size = np.zeros(5)
        c_cap = np.zeros(5)
        cand_to_bloc = {}
        cand_demand = {}

        for cand in active_cands:
            b = BLOCS.index(get_party_bloc(cand, party_df))
            cand_to_bloc[cand] = b
            menu_size[b] += 1.0
            
            # Coalition alias handling
            if b in [0, 1]:
                u = lc_series.get(cand, 0.0)
                if u <= 1e-9:
                    u = empirical_c0.get(cand, 0.0)
                    if u <= 1e-9 and cand in ['LP', 'NP', 'LNP']:
                        coal_lc = sum(lc_series.get(a, 0.0) for a in ['LP', 'NP', 'LNP'])
                        coal_la = [c for c in active_cands if c in ['LP', 'NP', 'LNP']]
                        if len(coal_la) == 1:
                            u = coal_lc
                        elif len(coal_la) > 1:
                            emp_c = empirical_c0.get(cand, 0.0)
                            emp_tot = sum(empirical_c0.get(c, 0.0) for c in coal_la)
                            u = coal_lc * (emp_c / emp_tot) if emp_tot > 0 else (coal_lc / len(coal_la))
            else:
                u = lc_series.get(cand, 0.0)

            cand_demand[cand] = u
            if b in [2, 3, 4]:
                c_cap[b] += u

        c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
        c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
        for b in [2, 3, 4]:
            c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0

        return menu_size, c_cap, cand_to_bloc, cand_demand

    ms_A, cap_A, ctb_A, cd_A = build_state_vars(menu_A)
    ms_B, cap_B, ctb_B, cd_B = build_state_vars(menu_B)

    alpha, beta, gamma = model.unpack_params(model.theta)
    s_hat_A = model.evaluate_state_allocation(lc_mass, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)
    s_hat_B = model.evaluate_state_allocation(lc_mass, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)

    # ---------------------------------------------------------
    # STEP 1: Inter-Bloc MRR (Macro Shift)
    # ---------------------------------------------------------
    emp_bloc_mass = {b: 0.0 for b in BLOCS}
    s_hat_A_bloc = {b: 0.0 for b in BLOCS}
    s_hat_B_bloc_inc = {b: 0.0 for b in BLOCS}
    s_hat_B_bloc_ent = {b: 0.0 for b in BLOCS}

    for c in menu_A:
        b = get_party_bloc(c, party_df)
        emp_bloc_mass[b] += empirical_c0.get(c, 0.0)
        s_hat_A_bloc[b] += s_hat_A.get(c, 0.0)
        s_hat_B_bloc_inc[b] += s_hat_B.get(c, 0.0)

    for e in cands_to_add:
        b = get_party_bloc(e, party_df)
        kappa = calibration_scalars.get(e, 1.0)
        s_hat_B_bloc_ent[b] += s_hat_B.get(e, 0.0) * kappa

    # Total modeled mass in State B per bloc (incumbents + calibrated entrants)
    s_hat_B_bloc_total = {b: s_hat_B_bloc_inc[b] + s_hat_B_bloc_ent[b] for b in BLOCS}

    # Unnormalized counterfactual bloc volume
    unnorm_bloc_vol = {}
    for b in BLOCS:
        if emp_bloc_mass[b] > 1e-9 and s_hat_A_bloc[b] > 1e-9:
            macro_retention = s_hat_B_bloc_total[b] / s_hat_A_bloc[b]
            unnorm_bloc_vol[b] = emp_bloc_mass[b] * macro_retention
        else:
            # Handles entrant into a bloc with zero empirical incumbents
            unnorm_bloc_vol[b] = s_hat_B_bloc_ent[b]

    # Close macro simplex across all blocs
    z_bloc = sum(unnorm_bloc_vol.values())
    norm_bloc_vol = {b: v / z_bloc for b, v in unnorm_bloc_vol.items()}

    # ---------------------------------------------------------
    # STEP 2 & 3: Entrant Extraction & Empirical Residual Allocation
    # ---------------------------------------------------------
    new_c0 = {}

    for b in BLOCS:
        b_cands_inc = [c for c in menu_A if get_party_bloc(c, party_df) == b]
        b_cands_ent = [c for c in cands_to_add if get_party_bloc(c, party_df) == b]

        v_star = norm_bloc_vol[b]
        b_total_B = s_hat_B_bloc_total[b]

        # 2. Extract calibrated entrant shares
        extracted_entrant_vol = 0.0
        for e in b_cands_ent:
            kappa = calibration_scalars.get(e, 1.0)
            calibrated_s_b = s_hat_B.get(e, 0.0) * kappa
            entrant_share = v_star * (calibrated_s_b / b_total_B) if b_total_B > 1e-9 else 0.0
            new_c0[e] = entrant_share
            extracted_entrant_vol += entrant_share

        # 3. Distribute residual mass strictly by empirical baseline ratios
        v_residual = max(0.0, v_star - extracted_entrant_vol)
        emp_inc_total = emp_bloc_mass[b]

        for c in b_cands_inc:
            if emp_inc_total > 1e-9:
                cand_intra_share = empirical_c0[c] / emp_inc_total
            else:
                cand_intra_share = 1.0 / len(b_cands_inc) if len(b_cands_inc) > 0 else 0.0
            
            new_c0[c] = v_residual * cand_intra_share

    # Final assertion of strict monotonicity (guard against floating point precision drift)
    for c in menu_A:
        if new_c0[c] > empirical_c0[c]:
            new_c0[c] = empirical_c0[c]

    # Final simplex closure check
    z_final = sum(new_c0.values())
    new_c0 = {c: v / z_final for c, v in new_c0.items()}

    shift_vector = {c: new_c0[c] - empirical_c0.get(c, 0.0) for c in menu_B}

    return new_c0, shift_vector, s_hat_A, s_hat_B


def inject_lc_party_to_dop(
    district: str,
    cand_to_add: str,
    target_bloc: str,
    empirical_c0: Dict[str, float],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
    calibration_scalar: float = 1.0
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Single-party backwards-compatible wrapper around inject_parties_to_dop."""
    return inject_parties_to_dop(
        district=district,
        cands_to_add=[cand_to_add],
        empirical_c0=empirical_c0,
        lc_df=lc_df,
        party_df=party_df,
        model=model,
        calibration_scalars={cand_to_add: calibration_scalar}
    )


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

    # Apply Bayesian Shrinkage towards 1.0 
    N = len(diag_df)
    # Shrinkage hyperparametera : 5 electorates fives ~25% weight to empirical kappa. 20 electorates gives 80% weight. 50 gives 97%.
    K = 9.0   # Midpoint parameter (where W = 0.5)
    p = 2    # Curvature exponent (higher p = steeper transition)
    
    N_p = N ** p
    K_p = K ** p
    W = N_p / (N_p + K_p) # ADD: make W probabilistic 
    
    shrunk_kappa = (W * raw_kappa) + ((1 - W) * 1.0)

    return diag_df, shrunk_kappa


def universally_add_party_to_DOPs(
    full_latent_tables: Dict[str, object],
    parties_to_add: List[str],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
    known_kappas: Dict[str, float] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Evaluates all electorates across full_latent_tables and applies counterfactual injection.
    
    Logic:
      - If all parties in parties_to_add already exist in district Count 0: do nothing.
      - If some exist: inject only the missing ones.
      - If none exist: inject all.
      - Automatically computes kappa via diagnose_and_recalibrate_party if not provided.
    
    Returns:
      counterfactual_c0_tables: Dict[district, new_c0_dict]
      calibrated_kappas: Dict[party, kappa]
    """
    calibrated_kappas = {} if known_kappas is None else copy.deepcopy(known_kappas)
    
    # 1. Calibrate missing kappas across contested seats
    for party in parties_to_add:
        if party not in calibrated_kappas:
            contested_divs = [
                d for d, tbl in full_latent_tables.items() 
                if tbl.count_0.get(party, 0.0) > 1e-9
            ]
            if contested_divs:
                _, kappa = diagnose_and_recalibrate_party(
                    party_abbr=party,
                    contested_divs=contested_divs,
                    latent_tables=full_latent_tables,
                    lc_df=lc_df,
                    party_df=party_df,
                    model=model
                )
                calibrated_kappas[party] = kappa
                if verbose:
                    print(f"[Calibration] Party '{party}' calibrated across {len(contested_divs)} seats: kappa = {kappa:.4f}")
            else:
                calibrated_kappas[party] = 1.0
                if verbose:
                    print(f"[Calibration] Party '{party}' has no contested baseline. Defaulted kappa = 1.0000")

    counterfactual_c0_tables = {}
    skipped_all_present = 0
    injected_count = 0

    # 2. Iterate across districts and perform injections
    for district, table in full_latent_tables.items():
        if district not in lc_df.columns:
            continue

        emp_c0 = table.count_0
        
        # Determine missing candidates
        missing_cands = [p for p in parties_to_add if emp_c0.get(p, 0.0) <= 1e-9]

        if not missing_cands:
            # Case 1: All parties already exist
            counterfactual_c0_tables[district] = copy.deepcopy(emp_c0)
            skipped_all_present += 1
            continue

        # Filter missing candidates who have non-zero Upper House demand
        viable_missing = []
        for p in missing_cands:
            u_lc = lc_df[district].get(p, 0.0)
            if u_lc > 1e-9:
                viable_missing.append(p)
            elif verbose:
                print(f"[Notice] Skipping {p} in '{district}': 0.0 Upper House demand.")

        if not viable_missing:
            counterfactual_c0_tables[district] = copy.deepcopy(emp_c0)
            continue

        # Case 2 & 3: Inject the subset of missing parties
        new_c0, _, _, _ = inject_parties_to_dop(
            district=district,
            cands_to_add=viable_missing,
            empirical_c0=emp_c0,
            lc_df=lc_df,
            party_df=party_df,
            model=model,
            calibration_scalars={p: calibrated_kappas[p] for p in viable_missing}
        )

        counterfactual_c0_tables[district] = new_c0
        injected_count += 1

    if verbose:
        print(f"\nExecution Complete:")
        print(f"  - Districts unchanged (already had all parties): {skipped_all_present}")
        print(f"  - Districts counterfactually injected: {injected_count}")

    return counterfactual_c0_tables, calibrated_kappas





counterfactual_c0_dict = {}
shift_vectors_dict = {}
model_pred_A_dict = {}
model_pred_B_dict = {}

PARTIES_TO_ADD = ['ON','VNS']
kappas = {}

for party in PARTIES_TO_ADD:
    # Programmatically find all districts where the party empirically contested LA
    contested_divs = [dist for dist, table in full_latent_tables.items() if table.count_0.get(party, 0.0) > 1e-9]
    
    if contested_divs:
        _, k = diagnose_and_recalibrate_party(
            party_abbr=party,
            contested_divs=contested_divs,
            latent_tables=full_latent_tables,
            lc_df=LC_FPs_df,
            party_df=party_df,
            model=final_model
        )
        kappas[party] = k
        print(f"[{party}] Contested {len(contested_divs)} seats. Calculated Kappa: {k:.4f}")
    else:
        kappas[party] = 1.0
        print(f"[{party}] Contested 0 seats. Defaulting Kappa to 1.000")

# ==========================================================
# 1. Multi-Party Diagnostic Execution Loop
# ==========================================================
counterfactual_c0_dict = {}
shift_vectors_dict = {}
model_pred_A_dict = {}
model_pred_B_dict = {}

for district, table in full_latent_tables.items():
    if district not in LC_FPs_df.columns:
        continue
        
    emp_c0 = table.count_0
    
    # Identify which of our target parties are missing from this district
    missing_cands = [p for p in PARTIES_TO_ADD if emp_c0.get(p, 0.0) <= 1e-9]
    
    # Filter out any missing party that has 0.0 Upper House base in this specific district
    viable_missing = [
        p for p in missing_cands 
        if LC_FPs_df[district].get(p, 0.0) > 1e-9
    ]

    # Skip district entirely if there's no one valid left to inject
    if not viable_missing:
        continue

    # Inject multiple parties simultaneously using the plural function
    new_c0, shift_vector, s_hat_a, s_hat_b = inject_parties_to_dop(
        district=district,
        cands_to_add=viable_missing,
        empirical_c0=emp_c0,
        lc_df=LC_FPs_df,
        party_df=party_df,
        model=final_model,
        calibration_scalars={p: kappas[p] for p in viable_missing}
    )

    counterfactual_c0_dict[district] = new_c0
    shift_vectors_dict[district] = shift_vector
    model_pred_A_dict[district] = s_hat_a
    model_pred_B_dict[district] = s_hat_b


# ==========================================================
# 2. Updated Diagnostic Table Generator
# ==========================================================
def generate_injection_comparison_table(
    district: str, 
    counterfactual_c0_dict: dict, 
    model_pred_A_dict: dict,
    model_pred_B_dict: dict,
    lc_df: pd.DataFrame,
    latent_tables: dict
) -> pd.DataFrame:
    """
    Renders a 5-row diagnostic table showing the structural transition.
    """
    if district not in counterfactual_c0_dict:
        print(f"District '{district}' not found in counterfactual results.")
        return pd.DataFrame()
        
    cf_c0 = counterfactual_c0_dict[district]
    s_hat_A = model_pred_A_dict[district]
    s_hat_B = model_pred_B_dict[district]
    orig_c0 = latent_tables[district].count_0
    
    lc_series = lc_df[district].dropna().copy()
    if lc_series.sum() > 1.5:
        lc_series = lc_series / 100.0
        
    table_data = {}
    
    # Use menu_B (all parties in the counterfactual) to ensure we see the entrant
    for party in cf_c0.keys():
        table_data[party] = [
            lc_series.get(party, 0.0),
            s_hat_A.get(party, 0.0),
            s_hat_B.get(party, 0.0),
            cf_c0.get(party, 0.0),
            orig_c0.get(party, 0.0)
        ]
        
    df = pd.DataFrame(
        table_data, 
        index=[
            "Raw LC Base",
            "Predicted LA C0 (Ŝ_A)",
            "Predicted LA C0_ON (Ŝ_B)",
            "Counterfactual LA C0",
            "Original Empirical LA C0"
        ]
    )
    
    return df.map(lambda x: f"{x:.2%}")

# ==========================================================
# 3. Print Output
# ==========================================================
simulated_districts = list(counterfactual_c0_dict.keys())

for dist in simulated_districts: # Limit to first 3 for rapid inspection
    print(f"\n--- {dist.upper()} ---")
    comp_table = generate_injection_comparison_table(
        dist, 
        counterfactual_c0_dict, 
        model_pred_A_dict, 
        model_pred_B_dict,
        LC_FPs_df, 
        full_latent_tables
    )
    print(comp_table.to_string())


import pdb; pdb.set_trace()




















######################################################################### Add IND back in ############################################################################################################


def extract_count_0(table_obj: Any) -> Dict[str, float]:
    """Cleanly extracts count_0 dictionary whether object is a Table or dict."""
    if hasattr(table_obj, 'count_0'):
        raw = table_obj.count_0
    elif isinstance(table_obj, dict):
        raw = table_obj
    else:
        raise TypeError("Expected dict or object with a 'count_0' attribute.")
    
    tot = sum(raw.values())
    if tot > 1.5:
        return {k: v / 100.0 for k, v in raw.items()}
    return dict(raw)


def build_and_evaluate_model_allocation(
    district: str,
    menu: List[str],
    lc_series: pd.Series,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
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

    menu_size = np.zeros(5)
    c_cap = np.zeros(5)
    cand_to_bloc = {}
    cand_demand = {}

    for cand in menu:
        b = BLOCS.index(get_party_bloc(cand, party_df))
        cand_to_bloc[cand] = b
        menu_size[b] += 1.0

        if b in [0, 1]:
            u = lc_series.get(cand, 0.0)
            if u <= 1e-9 and fallback_c0 is not None:
                u = fallback_c0.get(cand, 0.0)
                if u <= 1e-9 and cand in ['LP', 'NP', 'LNP']:
                    coal_lc = sum(lc_series.get(a, 0.0) for a in ['LP', 'NP', 'LNP'])
                    coal_la = [c for c in menu if c in ['LP', 'NP', 'LNP']]
                    if len(coal_la) == 1:
                        u = coal_lc
                    elif len(coal_la) > 1:
                        emp_c = fallback_c0.get(cand, 0.0)
                        emp_tot = sum(fallback_c0.get(c, 0.0) for c in coal_la)
                        u = coal_lc * (emp_c / emp_tot) if emp_tot > 0 else (coal_lc / len(coal_la))
        else:
            u = lc_series.get(cand, 0.0)

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
    district: str,
    empirical_c0_full: Dict[str, float],
    s_hat_a: Dict[str, float]
) -> Tuple[Dict[str, float], List[str], float]:
    """
    Calculates lambda_c (the proportion of incumbent c's structural vote 
    that defected to the Independent), strictly floored at 0 and calibrated 
    to match the total empirical Independent vote share.
    """
    ind_cands = [c for c, v in empirical_c0_full.items() if str(c).startswith('IND') and v > 1e-9]
    tot_ind_emp = sum(empirical_c0_full[c] for c in ind_cands)
    menu_A = [c for c in empirical_c0_full if c not in ind_cands and empirical_c0_full[c] > 1e-9]

    if tot_ind_emp <= 1e-9:
        return {c: 0.0 for c in menu_A}, [], 0.0

    # Raw votes drawn by the Independent from each incumbent
    raw_stolen = {}
    for c in menu_A:
        pred_a = s_hat_a.get(c, 0.0)
        emp_c = empirical_c0_full.get(c, 0.0)
        raw_stolen[c] = max(0.0, pred_a - emp_c)

    tot_raw_stolen = sum(raw_stolen.values())
    calibrated_stolen = {}

    if tot_raw_stolen > 1e-9:
        # Scale proportionally to match observed Independent vote share
        scale_factor = tot_ind_emp / tot_raw_stolen
        for c in menu_A:
            calibrated_stolen[c] = min(s_hat_a.get(c, 0.0), raw_stolen[c] * scale_factor)
    else:
        # Fallback proportional to predicted size if model underpredicts all incumbents
        for c in menu_A:
            calibrated_stolen[c] = tot_ind_emp * s_hat_a.get(c, 0.0)

    # Compute defection rate lambda_c = stolen / model_prediction
    lambdas = {}
    for c in menu_A:
        pred = s_hat_a.get(c, 0.0)
        lam = calibrated_stolen[c] / pred if pred > 1e-9 else 0.0
        lambdas[c] = min(0.9999, max(0.0, lam))

    return lambdas, ind_cands, tot_ind_emp


def inject_parties_with_ind_fork(
    district: str,
    cands_to_add: List[str],
    empirical_c0_full: Dict[str, float],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
    latent_table_obj: Optional[Any] = None,
    calibration_scalars: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Executes the Complete 6-Step Injection & Independent Restoration Pipeline.
    Includes Square-Root Scale Dampening for endogenous IND vs entrant party affinity.
    """
    if calibration_scalars is None:
        calibration_scalars = {c: 1.0 for c in cands_to_add}

    # Clean Upper House series
    lc_series = lc_df[district].dropna().copy()
    if lc_series.sum() > 1.5:
        lc_series = lc_series / 100.0

    # Separate incumbents and Independents
    ind_cands = [c for c, v in empirical_c0_full.items() if str(c).startswith('IND') and v > 1e-9]
    menu_A = [c for c in empirical_c0_full if c not in ind_cands and empirical_c0_full[c] > 1e-9]
    menu_B = menu_A + [c for c in cands_to_add if c not in menu_A]

    # -------------------------------------------------------------
    # STEP 1 & 2: Structural Baseline (S_hat_A) and Defection Rates
    # -------------------------------------------------------------
    s_hat_a = build_and_evaluate_model_allocation(
        district=district,
        menu=menu_A,
        lc_series=lc_series,
        party_df=party_df,
        model=model,
        fallback_c0=empirical_c0_full
    )

    lambdas, _, tot_ind_emp = compute_independent_defection_rates(
        district=district,
        empirical_c0_full=empirical_c0_full,
        s_hat_a=s_hat_a
    )

    # -------------------------------------------------------------
    # STEP 3 & 4: The Counterfactual Without IND (The Fork)
    # -------------------------------------------------------------
    if latent_table_obj is not None:
        regime = "latent_table_mrr"
        clean_latent_c0 = extract_count_0(latent_table_obj)
        
        cf_c0_no_ind, _, _, s_hat_b = inject_parties_to_dop(
            district=district,
            cands_to_add=cands_to_add,
            empirical_c0=clean_latent_c0,
            lc_df=lc_df,
            party_df=party_df,
            model=model,
            calibration_scalars=calibration_scalars
        )
    else:
        regime = "structural_pure"
        
        s_hat_b = build_and_evaluate_model_allocation(
            district=district,
            menu=menu_B,
            lc_series=lc_series,
            party_df=party_df,
            model=model,
            fallback_c0=empirical_c0_full
        )
        
        unnorm_b = {}
        for c in menu_B:
            if c in cands_to_add:
                kappa = calibration_scalars.get(c, 1.0)
                unnorm_b[c] = s_hat_b.get(c, 0.0) * kappa
            else:
                unnorm_b[c] = s_hat_b.get(c, 0.0)
                
        tot_unnorm = sum(unnorm_b.values())
        cf_c0_no_ind = {c: v / tot_unnorm for c, v in unnorm_b.items()}

    # -------------------------------------------------------------
    # STEP 5: Entrant Defection Rates via Scale-Dampened Affinity
    # -------------------------------------------------------------
    all_lambdas = dict(lambdas)
    gamma_scalars = {}

    for entrant in cands_to_add:
        if len(cands_to_add) == 1:
            s_hat_b_single = s_hat_b
        else:
            s_hat_b_single = build_and_evaluate_model_allocation(
                district=district,
                menu=menu_A + [entrant],
                lc_series=lc_series,
                party_df=party_df,
                model=model,
                fallback_c0=empirical_c0_full
            )

        # 1. District-wide baseline defection rates (denominators)
        tot_menu_a_mass = sum(s_hat_a.values())
        tot_delta = sum(max(0.0, s_hat_a.get(c, 0.0) - s_hat_b_single.get(c, 0.0)) for c in lambdas)
        
        bar_tau = tot_delta / tot_menu_a_mass if tot_menu_a_mass > 1e-9 else 1.0
        bar_lam = tot_ind_emp / tot_menu_a_mass if tot_menu_a_mass > 1e-9 else 1.0

        # Scale dampeners (square-root scaling balances affinity vs size)
        scale_tau = (bar_tau ** 0.5)
        scale_lam = (bar_lam ** 0.5)

        g_stream = {}
        gamma_stream = {}

        for c, lam_c in lambdas.items():
            pred_a = s_hat_a.get(c, 0.0)
            delta_c = max(0.0, pred_a - s_hat_b_single.get(c, 0.0))
            tau_c = min(0.9999, max(0.0, delta_c / pred_a)) if pred_a > 1e-9 else 0.0

            # Scale-dampened odds
            eff_odds_tau = (tau_c / scale_tau) if scale_tau > 1e-9 else 0.0
            eff_odds_lam = (lam_c / scale_lam) if scale_lam > 1e-9 else 0.0

            if (eff_odds_tau + eff_odds_lam) > 1e-9:
                gamma_c = eff_odds_tau / (eff_odds_tau + eff_odds_lam)
            else:
                gamma_c = 0.5

            gamma_stream[c] = gamma_c
            g_stream[c] = delta_c * lam_c

        tot_g = sum(g_stream.values())

        if tot_g > 1e-9:
            gamma_bar = sum((g / tot_g) * gamma_stream[c] for c, g in g_stream.items())
            
            ent_mass = cf_c0_no_ind.get(entrant, 0.0)
            lambda_donor_bar = tot_g / ent_mass if ent_mass > 1e-9 else 0.0
            
            lam_ent = lambda_donor_bar * (1.0 - gamma_bar)
            all_lambdas[entrant] = min(0.9999, max(0.0, lam_ent))
            gamma_scalars[entrant] = gamma_bar
        else:
            all_lambdas[entrant] = 0.0
            gamma_scalars[entrant] = 1.0

    # -------------------------------------------------------------
    # STEP 6: Universal Simplex Reintegration
    # -------------------------------------------------------------
    final_c0 = {}
    total_restored_ind_mass = 0.0

    # Retain non-defected share across all parties
    for p in cf_c0_no_ind.keys():
        retention = 1.0 - all_lambdas.get(p, 0.0)
        final_c0[p] = cf_c0_no_ind[p] * retention
        total_restored_ind_mass += cf_c0_no_ind[p] * all_lambdas.get(p, 0.0)

    # Distribute restored mass across Independents proportional to original vote
    if ind_cands:
        for ind in ind_cands:
            prop = empirical_c0_full[ind] / tot_ind_emp if tot_ind_emp > 0 else (1.0 / len(ind_cands))
            final_c0[ind] = total_restored_ind_mass * prop

    # Exact simplex closure
    z_final = sum(final_c0.values())
    final_c0 = {k: v / z_final for k, v in final_c0.items()}

    # -------------------------------------------------------------
    # Calculate Pure Extraction Rates (tau) for IA Reintegration
    # -------------------------------------------------------------
    extraction_rates = {}
    
    if latent_table_obj is not None:
        baseline_no_ind_no_on = extract_count_0(latent_table_obj)
    else:
        baseline_no_ind_no_on = s_hat_a
    
    for c in menu_A:
        base_val = baseline_no_ind_no_on.get(c, 0.0)
        cf_val = cf_c0_no_ind.get(c, 0.0)
        extraction_rates[c] = min(0.9999, max(0.0, 1.0 - (cf_val / base_val))) if base_val > 1e-9 else 0.0

    if tot_ind_emp > 1e-9:
        ind_loss_frac = max(0.0, (tot_ind_emp - total_restored_ind_mass) / tot_ind_emp)
    else:
        ind_loss_frac = 0.0
        
    for ind in ind_cands:
        extraction_rates[ind] = ind_loss_frac

    diagnostics = {
        'regime': regime,
        'baseline_no_ind_no_on': baseline_no_ind_no_on,
        'cf_c0_no_ind': cf_c0_no_ind,
        'gamma_scalars': gamma_scalars,
        'tot_ind_emp': tot_ind_emp,
        'tot_ind_final': total_restored_ind_mass,
        'extraction_rates': extraction_rates
    }

    return final_c0, all_lambdas, diagnostics











############################################ Re-inject incumbency ###############

def reapply_incumbency_to_counterfactual(
    cf_c0_no_ia: Dict[str, float],
    extraction_rates: Dict[str, float], 
    ia_adjustments: Dict[str, float],
    dop_table: Any,
    entrants: List[str],
    emp_c0_with_ia: Optional[Dict[str, float]] = None,
    debug: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Complete Standalone IA Restoration Pipeline.
    """
    # -------------------------------------------------------------------------
    # STAGE 1: 3CP -> Count 0 via DOP Reverse-Transfers
    # -------------------------------------------------------------------------
    table_copy = copy.deepcopy(dop_table)
    
    if emp_c0_with_ia is None:
        if hasattr(table_copy, 'apply_incumbency'):
            adjusted_table = table_copy.apply_incumbency(ia_adjustments)
        elif hasattr(table_copy, 'strip_incumbency'):
            inverted_adjustments = {k: -v for k, v in ia_adjustments.items()}
            adjusted_table = table_copy.strip_incumbency(inverted_adjustments)
        else:
            raise AttributeError("dop_table must have 'apply_incumbency' or 'strip_incumbency' method.")
            
        emp_c0_with_ia = extract_count_0(adjusted_table)

    # -------------------------------------------------------------------------
    # STAGE 2: Absolute Extraction & Entrant Mapping (Count 0 -> Count -1)
    # -------------------------------------------------------------------------
    final_c0 = {}
    pooled_entrant_mass = 0.0

    if debug:
        print(f"\n--- STAGE 2: Count -1 (Entrant Extraction) ---")

    for cand, ia_adjusted_mass in emp_c0_with_ia.items():
        if cand in entrants:
            continue

        ext_frac = extraction_rates.get(cand, 0.0)
        retention_frac = 1.0 - ext_frac

        final_c0[cand] = ia_adjusted_mass * retention_frac
        stolen_vol = ia_adjusted_mass * ext_frac
        pooled_entrant_mass += stolen_vol

        if debug:
            print(f"  {cand:5s} | Retained {retention_frac:6.2%} of {ia_adjusted_mass:.4%} | Donated {stolen_vol:.4%} to pool")

    # Distribute pooled entrant mass
    entrant_pre_ia_masses = {e: cf_c0_no_ia.get(e, 0.0) for e in entrants}
    tot_entrant_pre_ia = sum(entrant_pre_ia_masses.values())

    if debug:
        print(f"\n  [Pool Total]: {pooled_entrant_mass:.4%}")

    if tot_entrant_pre_ia > 1e-9:
        for e, pre_mass in entrant_pre_ia_masses.items():
            prop = pre_mass / tot_entrant_pre_ia
            final_c0[e] = pooled_entrant_mass * prop
            if debug:
                print(f"  {e:5s} | Receives {prop:5.1%} of pool -> {final_c0[e]:.4%}")
    else:
        if entrants:
            split = pooled_entrant_mass / len(entrants)
            for e in entrants:
                final_c0[e] = split

    # Exact Simplex Closure
    z_final = sum(final_c0.values())
    final_c0 = {k: v / z_final for k, v in final_c0.items()}
    
    if debug:
        print(f"{'='*70}\n")

    return final_c0, emp_c0_with_ia


def test_high_ind_electorates(
    dop_table_dict: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit',
    ia_adjustments_dict: dict,
    latent_tables: dict = None,
    parties_to_add: list = None,
    kappas: dict = None
):
    if latent_tables is None:
        latent_tables = {}
    if parties_to_add is None:
        parties_to_add = ['ON']
    if kappas is None:
        kappas = {p: 0.985 for p in parties_to_add}
        
    summary_rows = []
    
    for district, df in dop_table_dict.items():
        if district not in lc_df.columns:
            continue
            
        # =========================================================
        # 1. Initialize DOPTable and Normalize
        # =========================================================
        table = DOPTable(df)
        total_votes = sum(table.count_0.values())
        table.count_0 = {k: v / total_votes for k, v in table.count_0.items()}
        for rnd in table.rounds:
            rnd['V_elim'] /= total_votes
            rnd['transfers'] = {k: v / total_votes for k, v in rnd['transfers'].items()}
            
        # =========================================================
        # 2. High-IND Check
        # =========================================================
        ind_cands = [c for c in table.candidates if str(c).startswith('IND')]
        ind_share = sum(table.count_0.get(c, 0) for c in ind_cands)
        
        if ind_share < 0.15:
            continue
            
        orig_ind_share = ind_share
        
        # =========================================================
        # 3. Strip Incumbency & Filter Valid Entrants
        # =========================================================
        ia_adj = ia_adjustments_dict.get(district, {})
        stripped_table = table.strip_incumbency(ia_adj) if ia_adj else table
        emp_c0_stripped = stripped_table.count_0
        
        valid_parties = []
        for p in parties_to_add:
            raw_lc = lc_df[district].get(p, 0.0)
            if raw_lc > 1.5:
                raw_lc /= 100.0
            if raw_lc > 1e-9:
                valid_parties.append(p)
            else:
                print(f"[Notice] {p} has 0.0 LC demand in '{district}'. Skipping {p}.")
                
        if not valid_parties:
            print(f"[Notice] No valid entrants for '{district}'. Skipping district entirely.")
            continue
            
        valid_kappas = {p: kappas.get(p, 1.0) for p in valid_parties}

        # =========================================================
        # 4. Inject Parties & Compute Entrant/IND Tug-of-War
        # =========================================================
        tbl_obj = latent_tables.get(district, None)
        
        cf_c0_no_ia, lambdas, diag = inject_parties_with_ind_fork(
            district=district,
            cands_to_add=valid_parties,
            empirical_c0_full=emp_c0_stripped,
            lc_df=lc_df,
            party_df=party_df,
            model=model,
            latent_table_obj=tbl_obj,
            calibration_scalars=valid_kappas
        )
        
        # =========================================================
        # 5. Re-apply Incumbency Advantage
        # =========================================================
        final_c0, abs_ia_c0 = reapply_incumbency_to_counterfactual(
            cf_c0_no_ia=cf_c0_no_ia,
            extraction_rates=diag['extraction_rates'],
            ia_adjustments=ia_adj,
            dop_table=stripped_table,
            entrants=valid_parties
        )
        
        # =========================================================
        # 6. Diagnostics and Reporting
        # =========================================================
        ind_keys = [c for c in final_c0 if str(c).startswith('IND')]
        final_ind_share = sum(final_c0[c] for c in ind_keys)

        # Base row data
        row_data = {
            'District': district,
            'Regime': diag['regime'],
            'Orig_IND_with_IA': f"{orig_ind_share:.2%}",
            'Final_IND_with_IA': f"{final_ind_share:.2%}",
            'IND_Delta': f"{(final_ind_share - orig_ind_share):+.2%}"
        }
        
        # Dynamically append columns for each entrant
        for p in valid_parties:
            row_data[f'{p}_Vote'] = f"{final_c0.get(p, 0.0):.2%}"
            row_data[f'{p}_λ'] = f"{lambdas.get(p, 0.0):.2%}"
            row_data[f'{p}_γ'] = f"{diag['gamma_scalars'].get(p, 1.0):.2%}"
            
        summary_rows.append(row_data)

        # Print Detailed District Trace
        print(f"\n--- {district.upper()} ({diag['regime']}) ---")
        detail_data = {}
        for p in final_c0.keys():
            detail_data[p] = [
                f"{table.count_0.get(p, 0.0):.2%}",
                f"{emp_c0_stripped.get(p, 0.0):.2%}",
                f"{diag['baseline_no_ind_no_on'].get(p, 0.0):.2%}",
                f"{diag['cf_c0_no_ind'].get(p, 0.0):.2%}",
                f"{cf_c0_no_ia.get(p, 0.0):.2%}",
                f"{final_c0.get(p, 0.0):.2%}"
            ]
        
        detail_df = pd.DataFrame(
            detail_data,
            index=[
                "1. Empirical (With IA)",
                "2. Empirical (IA Free)",
                "3. CF w/o IND, w/o Entrants (IA Free)",
                "4. CF w/o IND, WITH Entrants (IA Free)",
                "5. CF WITH IND, WITH Entrants (IA Free)",
                "6. Final Counterfactual (With IA)"
            ]
        )
        print(detail_df.to_string())

    print("\n" + "=" * 90)
    print("HIGH-IND SUMMARY TABLE")
    print("=" * 90)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

test_high_ind_electorates(
    dop_table_dict = DOP_table_dict,
    lc_df = LC_FPs_df,
    party_df = party_df,
    model = final_model,
    ia_adjustments_dict = ia_adjustments_dict,
    latent_tables = None,
    parties_to_add = PARTIES_TO_ADD,
    kappas = kappas
)

import pdb; pdb.set_trace()
























def test_removal_from_DOP():
    # remove DOP table
    file = Path(f'{data_year}_DOP_table_dict.pkl')

    if file.exists():
        with open(file, 'rb') as f:
            DOP_table_dict = pickle.load(f)


    table_df = DOP_table_dict['Pakenham']

    print(table_df)
    curr_3CP = table_df.loc[table_df['CandRemaining'].eq(3)].iloc[:,2:-2].dropna(axis=1).columns.tolist()


    table = DOPTable(table_df)

    for p in [q for q in table_df.columns[2:-2] if q not in curr_3CP]:

        print(p, 'removed')
        cf_table = table.remove_candidate(p, lambda_val=0.9, gamma=0.5)

        print(cf_table.to_dataframe().to_string())

        import pdb; pdb.set_trace()