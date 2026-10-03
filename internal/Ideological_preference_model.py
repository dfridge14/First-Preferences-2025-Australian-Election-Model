import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import time

from scipy.optimize import minimize

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

with open('ia_adjustments_dict.pkl', 'rb') as f:
    ia_adjustments_dict = pickle.load(f)

LC_FPs_df = pd.read_csv(f'{data_year}-LC-First-Prefs-df.csv', index_col = 'PartyAb')


import pdb; pdb.set_trace()

def generate_latent_training_tables(DOP_table_dict, party_df, LC_FPs_df, ia_adjustments_dict):
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

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Define our fixed ideological arrays
BLOCS = ["ALP", "COAL", "Left", "Right", "Centre"]
ORPHAN_POOLS = ["Left", "Right", "Centre"]

@dataclass
class RoundTransition:
    """Captures the First-Difference transition between two discrete choice environments (M_X -> M_X+1)."""
    district: str
    round_index: int
    eliminated_cand: str
    
    # Static District Base
    lc_mass: np.ndarray          # Shape (5,) - LC baseline of each bloc (True Demand)
    
    # State X (Pre-Elimination)
    active_cands_X: List[str]
    menu_size_X: np.ndarray      # Shape (5,)
    c_cap_X: np.ndarray          # Shape (5,)
    cand_to_bloc_X: Dict[str, int]
    cand_demand_X: Dict[str, float]
    
    # State X+1 (Post-Elimination)
    active_cands_X1: List[str]
    menu_size_X1: np.ndarray
    c_cap_X1: np.ndarray
    cand_to_bloc_X1: Dict[str, int]
    cand_demand_X1: Dict[str, float]
    
    # Empirical Ground Truth
    empirical_transfers: Dict[str, float] # The actual vote fractions distributed
    empirical_total_transferred: float    # Sum of transfers in this round


def build_training_sequences(latent_tables: dict, party_df: pd.DataFrame, lc_df: pd.DataFrame) -> List[RoundTransition]:
    """
    Executes Phase 2: Converts latent DOP tables into discrete choice state transitions.
    """
    # 1. Build Taxonomy Mappings
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    
    def get_bloc(cand: str) -> str:
        if str(cand).startswith('IND'): return 'Centre'
        return ideo_map.get(cand, 'Centre')

    def build_menu_state(active_cands: set, lc_series: pd.Series, lc_mass: np.ndarray, count_0: dict) -> Tuple:
        """Helper to compute Menu Size, Capacities, and Demand for a given set of candidates."""
        menu_size = np.zeros(5)
        c_cap = np.zeros(5)
        cand_to_bloc = {}
        cand_demand = {}

        for cand in active_cands:
            b_idx = BLOCS.index(get_bloc(cand))
            cand_to_bloc[cand] = b_idx
            menu_size[b_idx] += 1.0

            if b_idx == 1:  
                # COAL: Peak at LA Count 0 to establish internal ticket ratios
                u_m = count_0.get(cand, 0.0)
            else: 
                # otherwise strictly rely on LC base demand
                u_m = lc_series.get('IND', 0.0) if str(cand).startswith('IND') else lc_series.get(cand, 0.0)
            
            cand_demand[cand] = u_m
            
            # Accumulate minor party capacity
            if b_idx != 1:
                c_cap[b_idx] += u_m

        # Finalize Capacities (Majors always have C_j = 1.0 if present)
        c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
        c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
        
        for b in [2, 3, 4]:
            if lc_mass[b] > 0:
                c_cap[b] = min(1.0, c_cap[b] / lc_mass[b])
            else:
                c_cap[b] = 0.0
                
        return menu_size, c_cap, cand_to_bloc, cand_demand


    transitions = []

    # 2. Iterate through cleansed electorates
    for div, table in latent_tables.items():
        if div not in lc_df.columns:
            continue
            
        lc_series = lc_df[div].dropna()
        if lc_series.sum() > 1.5: 
            lc_series = lc_series / 100.0

        # Calculate total LC mass per bloc (LC_j)
        lc_mass = np.zeros(5)
        for party, share in lc_series.items():
            bloc_idx = BLOCS.index(get_bloc(party))
            lc_mass[bloc_idx] += share
            
        # Normalize to exactly 1.0 to handle minor informal leaks
        if lc_mass.sum() > 0:
            lc_mass = lc_mass / lc_mass.sum()

        # Track surviving candidates (starts with all in count_0)
        active_cands = set(c for c, v in table.count_0.items() if v > 1e-9)

        # 3. Step through elimination rounds
        for r_idx, rnd in enumerate(table.rounds):
            
            # Strict stop condition: Do not model the 4CP -> 3CP elimination (or anything smaller).
            # We only evaluate transitions where Menu X has 5 or more candidates.
            if len(active_cands) <= DEEPEST_CAND_NO:
                break
                
            elim_cand = rnd['eliminated']
            
            # Build State X (Pre-Elimination)
            ms_X, cap_X, ctb_X, cd_X = build_menu_state(active_cands, lc_series, lc_mass, table.count_0)
            
            # Remove eliminated candidate to form State X+1
            active_cands_X1 = active_cands.copy()
            active_cands_X1.remove(elim_cand)
            
            # Build State X+1 (Post-Elimination)
            ms_X1, cap_X1, ctb_X1, cd_X1 = build_menu_state(active_cands_X1, lc_series, lc_mass, table.count_0)
            
            # Update the running set for the next loop iteration
            active_cands = active_cands_X1
            
            # We do not use major party eliminations to train the orphan preference flow parameters
            if get_bloc(elim_cand) in ["ALP", "COAL"]:
                continue
                
            transitions.append(RoundTransition(
                district=div,
                round_index=r_idx,
                eliminated_cand=elim_cand,
                lc_mass=lc_mass,
                
                # State X
                active_cands_X=list(ctb_X.keys()),
                menu_size_X=ms_X,
                c_cap_X=cap_X,
                cand_to_bloc_X=ctb_X,
                cand_demand_X=cd_X,
                
                # State X+1
                active_cands_X1=list(ctb_X1.keys()),
                menu_size_X1=ms_X1,
                c_cap_X1=cap_X1,
                cand_to_bloc_X1=ctb_X1,
                cand_demand_X1=cd_X1,
                
                # Targets
                empirical_transfers=rnd['transfers'].copy(),
                empirical_total_transferred=rnd.get('V_elim', sum(rnd['transfers'].values()))
            ))

    print(f"Extracted {len(transitions)} First-Difference state transitions for model training.")
    return transitions

# 1. Define the test subset (change names if needed based on your dataset)
test_divs = ['Eureka', 'Bass', 'Bentleigh', 'Morwell','Bayswater']
test_dop_dict = {d: DOP_table_dict[d] for d in test_divs if d in DOP_table_dict}

print(f"--- STARTING TEST RUN ON {len(test_dop_dict)} ELECTORATES ---")


test_latent_tables = generate_latent_training_tables(
    DOP_table_dict=test_dop_dict, 
    party_df=party_df, 
    LC_FPs_df=LC_FPs_df, 
    ia_adjustments_dict=ia_adjustments_dict
)

print("\n[Extracting First-Difference State Transitions...]")
test_transitions = build_training_sequences(
    latent_tables=test_latent_tables, 
    party_df=party_df, 
    lc_df=LC_FPs_df
)

print(f"\n--- TEST COMPLETE: Successfully extracted {len(test_transitions)} transitions. ---")

# 2. Deeply inspect the first transition to verify formatting
if test_transitions:
    t = test_transitions[0]
    
    print(f"\n==================================================")
    print(f"TRANSITION INSPECTION: {t.district} (Round {t.round_index})")
    print(f"Eliminated Candidate: {t.eliminated_cand}")
    print(f"Global LC Base (True Demand): {np.round(t.lc_mass, 4)}")
    print(f"==================================================")
    
    print("\n--- STATE X (Before Elimination) ---")
    print(f"Active Menu ({len(t.active_cands_X)}): {t.active_cands_X}")
    print(f"Bloc Capacities: {np.round(t.c_cap_X, 4)}")
    print(f"Menu Sizes:      {t.menu_size_X}")
    
    print("\n--- STATE X+1 (After Elimination) ---")
    print(f"Active Menu ({len(t.active_cands_X1)}): {t.active_cands_X1}")
    print(f"Bloc Capacities: {np.round(t.c_cap_X1, 4)}")
    print(f"Menu Sizes:      {t.menu_size_X1}")
    
    print("\n--- GROUND TRUTH (The Delta Target) ---")
    print(f"Total Volume Transferred: {t.empirical_total_transferred:.5f}")
    print(f"Actual Transfer Vector:")
    for cand, val in t.empirical_transfers.items():
        if val > 0:
            print(f"  -> {cand}: {val:.6f}")
    print(f"==================================================")

import pdb; pdb.set_trace()




class AggregateLatentLogit:
    """Aggregate Latent Class Logit Model with First-Difference State Differencing.

    Parameter vector theta has length 15:
      For each of the 3 orphan pools i in {Left, Right, Centre}:
        - 4 destination log-affinities: alpha_{i, COAL}, alpha_{i, Left},
        alpha_{i, Right}, alpha_{i, Centre}
          (alpha_{i, ALP} is constrained to 0.0 as the identification reference)
        - 1 capacity elasticity: beta_i >= 0
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
        """Sensible empirical priors:

        Left pool favors Left/ALP; Right pool favors Right/COAL; Centre pool is
        balanced. All beta_i initialized to 1.0 (linear capacity scaling).
        """
        # [alpha_COAL, alpha_Left, alpha_Right, alpha_Centre, beta]
        prior_left = np.array([-1.5, 2.0, -2.0, -0.5])
        prior_right = np.array([1.5, -1.5, 2.0, 1])
        prior_centre = np.array([0, 0.5, 0.5, 1.5])

        # betas: [beta_mass, beta_cap, beta_menu]
        prior_betas = np.array([1, 0.5, 0.1]) # ADD: prior parameter interpretability

        prior_gamma = np.array([0.8])

        return np.concatenate([prior_left, prior_right, prior_centre, prior_betas, prior_gamma])

    def unpack_params(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpacks theta (15,) into alpha matrix (3, 5) and shared beta vector (3,)."""
        alpha = np.zeros((self.n_pools, self.n_blocs), dtype=np.float64)
        
        # Unpack Alphas
        alpha[0, 1:] = theta[0:4]   # Left pool
        alpha[1, 1:] = theta[4:8]   # Right pool
        alpha[2, 1:] = theta[8:12]  # Centre pool
        
        # Unpack Shared Betas
        beta = theta[12:15]  # [beta_mass, beta_cap, beta_menu]

        gamma = float(theta[15])
        
        return alpha, beta, gamma

    def set_params(self, theta: np.ndarray):
        if len(theta) != self.n_params:
            raise ValueError(
                f"Expected theta of length {self.n_params}, got {len(theta)}"
            )
        self.theta = np.asarray(theta, dtype=np.float64)

    def compute_choice_probabilities(
        self,
        lc_mass: np.ndarray,      # <--- Now required for beta_mass
        c_cap: np.ndarray,
        menu_size: np.ndarray,
        alpha: np.ndarray,
        beta_shared: np.ndarray,
    ) -> np.ndarray:
        """Computes P(j | i, M) using shared structural betas."""
        b_mass, b_cap, b_menu = beta_shared
        
        # Apply safe logarithms
        log_mass = np.log(np.maximum(lc_mass, 1e-6))
        log_cap  = np.log(np.maximum(c_cap, 1e-6))
        log_menu = np.log(np.maximum(menu_size, 1e-6))
        
        # Calculate shared structural utility: U_j = b_m*ln(LC) + b_c*ln(C) + b_n*ln(|M|)
        # Shape: (5,)
        u_structural = (b_mass * log_mass) + (b_cap * log_cap) + (b_menu * log_menu)
        
        # V_{i, j} = alpha_{i, j} + U_structural_j
        # Broadcasts structural utility across all 3 orphan pools
        v_matrix = alpha + u_structural[np.newaxis, :]

        # Mask out blocs that have no candidates on the ballot
        unavailable_mask = menu_size <= 0
        v_matrix[:, unavailable_mask] = -1e9

        # Numerically stable softmax
        v_max = np.max(v_matrix, axis=1, keepdims=True)
        exp_v = np.exp(v_matrix - v_max) * (~unavailable_mask)[np.newaxis, :]
        sum_exp_v = np.sum(exp_v, axis=1, keepdims=True)
        sum_exp_v = np.where(sum_exp_v <= 1e-12, 1.0, sum_exp_v)

        return exp_v / sum_exp_v

    def evaluate_state_allocation(
        self,
        lc_mass: np.ndarray,
        active_cands: List[str],
        c_cap: np.ndarray,
        menu_size: np.ndarray,
        cand_to_bloc: Dict[str, int],
        cand_demand: Dict[str, float],
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: float,
    ) -> Dict[str, float]:
        """Calculates global predicted vote share S_m(M) for each candidate on menu M."""
        # 1. Routing probabilities P(bloc j | orphan pool i) -> shape (3, 5)
        p_bloc_given_pool = self.compute_choice_probabilities(lc_mass, c_cap, menu_size, alpha, beta)

        # 2. Intra-bloc demand shares P(candidate m | bloc j)
        cand_intra_share = {}
        for b_idx in range(self.n_blocs):
            bloc_cands = [c for c in active_cands if cand_to_bloc[c] == b_idx]
            if not bloc_cands:
                continue

            eff_gamma = 1.0 if b_idx in [1] else gamma # ignore gamma for COAL parties

            # Apply Gamma to the primary vote demand
            sum_demand = sum((cand_demand[c] ** eff_gamma) for c in bloc_cands)
            
            if sum_demand > 1e-9:
                for c in bloc_cands:
                    cand_intra_share[c] = (cand_demand[c] ** eff_gamma) / sum_demand
            else:
                for c in bloc_cands:
                    cand_intra_share[c] = 1.0 / len(bloc_cands)

        # 3. Base / Native Demand Allocation
        s_hat = {c: 0.0 for c in active_cands}

        # Major parties receive their full LC mass (or proportional split if double-div)
        for b_idx in [0, 1]:  # ALP, COAL
            bloc_cands = [c for c in active_cands if cand_to_bloc[c] == b_idx]
            for c in bloc_cands:
                s_hat[c] += lc_mass[b_idx] * cand_intra_share[c]

        # Minor parties receive their specific un-orphaned LC demand: u_m
        for b_idx in [2, 3, 4]:  # Left, Right, Centre
            bloc_cands = [c for c in active_cands if cand_to_bloc[c] == b_idx]
            for c in bloc_cands:
                s_hat[c] += cand_demand[c]

        # 4. Orphan Pool Distribution
        # Unrepresented mass per pool i: O_i = LC_i * max(0, 1 - C_i)
        orphan_mass = np.zeros(self.n_pools, dtype=np.float64)
        for p_idx, b_idx in enumerate([2, 3, 4]):
            orphan_mass[p_idx] = lc_mass[b_idx] * max(0.0, 1.0 - c_cap[b_idx])

        # Route orphan pools: Pool i -> Bloc j -> Candidate m
        for p_idx in range(self.n_pools):
            o_m = orphan_mass[p_idx]
            if o_m <= 1e-12:
                continue
            for b_idx in range(self.n_blocs):
                prob_bloc = p_bloc_given_pool[p_idx, b_idx]
                if prob_bloc <= 1e-12:
                    continue
                bloc_cands = [
                    c for c in active_cands if cand_to_bloc[c] == b_idx
                ]
                for c in bloc_cands:
                    s_hat[c] += o_m * prob_bloc * cand_intra_share[c]

        return s_hat

    def predict_transition_delta(
        self, transition, alpha: np.ndarray, beta: np.ndarray, gamma: float
    ) -> Dict[str, float]:
        """Calculates Delta S_hat = S_hat(M_{X+1}) - S_hat(M_X) for all surviving candidates."""
        # Evaluate state X (pre-elimination)
        s_X = self.evaluate_state_allocation(
            lc_mass=transition.lc_mass,
            active_cands=transition.active_cands_X,
            c_cap=transition.c_cap_X,
            menu_size=transition.menu_size_X,
            cand_to_bloc=transition.cand_to_bloc_X,
            cand_demand=transition.cand_demand_X,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        # Evaluate state X+1 (post-elimination)
        s_X1 = self.evaluate_state_allocation(
            lc_mass=transition.lc_mass,
            active_cands=transition.active_cands_X1,
            c_cap=transition.c_cap_X1,
            menu_size=transition.menu_size_X1,
            cand_to_bloc=transition.cand_to_bloc_X1,
            cand_demand=transition.cand_demand_X1,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        # Delta for candidates surviving into M_{X+1}
        delta_hat = {}
        for c in transition.active_cands_X1:
            delta_hat[c] = s_X1[c] - s_X.get(c, 0.0)

        return delta_hat
    

def compute_objective_loss(
    theta: np.ndarray,
    transitions: list,
    model: AggregateLatentLogit,
    l2_reg: float = 1e-4,
) -> float:
    """
    Computes Weighted Squared Error on transfer shares. Pools LP and NP into a unified 'COAL_COMBINED' entity 
    during error calculation to shield the optimizer from intra-COAL splits irrelevant to the model.
    """
    alpha, beta, gamma = model.unpack_params(theta)
    total_loss = 0.0

    for t in transitions:
        # 1. Model predicts raw deltas using naive Count 0 ratios for LP/NP
        delta_hat = model.predict_transition_delta(t, alpha, beta, gamma)
        
        sum_pred = sum(delta_hat.values())
        sum_actual = t.empirical_total_transferred

        if sum_pred <= 1e-9 or sum_actual <= 1e-9:
            total_loss += 1.0 * sum_actual
            continue

        # 2. Copy dictionaries so empirical transition data remains untouched
        actual_transfers = t.empirical_transfers.copy()
        pred_transfers = delta_hat.copy()

        # 3. THE OPTIMIZER SHIELD: Pool LP & NP strictly for the error evaluation
        has_lp = ('LP' in actual_transfers or 'LP' in pred_transfers)
        has_np = ('NP' in actual_transfers or 'NP' in pred_transfers)

        if has_lp and has_np:
            # Fuse empirical actuals
            coal_actual = actual_transfers.pop('LP', 0.0) + actual_transfers.pop('NP', 0.0)
            actual_transfers['COAL_COMBINED'] = coal_actual

            # Fuse model predictions
            coal_pred = pred_transfers.pop('LP', 0.0) + pred_transfers.pop('NP', 0.0)
            pred_transfers['COAL_COMBINED'] = coal_pred

        # 4. Compute Weighted Squared Error on the pooled shares
        t_loss = 0.0
        all_eval_cands = set(actual_transfers.keys()) | set(pred_transfers.keys())

        for cand in all_eval_cands:
            actual_vol = actual_transfers.get(cand, 0.0)
            pred_vol = pred_transfers.get(cand, 0.0)

            actual_share = actual_vol / sum_actual
            pred_share = pred_vol / sum_pred

            t_loss += (pred_share - actual_share) ** 2

        # Weight by empirical round volume
        total_loss += t_loss * sum_actual

    # Apply L2 Regularization to Alphas
    total_loss += l2_reg * np.sum(alpha**2)
    return float(total_loss)


def fit_preference_model(
    transitions: list,
    n_starts: int = 15,
    l2_reg: float = 1e-4,
    random_seed: int = 42,
) -> Tuple[AggregateLatentLogit, Tuple[pd.DataFrame, pd.Series]]:
    """Calibrates the 15 parameters using a Multi-Start L-BFGS-B optimization routine."""
    np.random.seed(random_seed)
    model = AggregateLatentLogit()

    # Parameter Bounds:
    bounds = []
    for _ in range(12):
        bounds.append((-6.0, 6.0))
    # force beta bounds to remain positive drivers)
    for _ in range(3):
        bounds.append((0.0, 5.0))
    # gamma bound prevents zero-division and restricts to sub-proportionality
    bounds.append((0.1, 1.0))

    # Generate Initial Starting Points
    priors = model.get_default_priors()
    starting_points = [priors]  # Start 0 is the informed prior

    for _ in range(n_starts - 1):
        # Perturb informed priors with uniform random noise
        jitter_alphas = np.random.uniform(-1.0, 1.0, size=12)
        jitter_betas  = np.random.uniform(-0.5, 0.5, size=3)  # CHECK: should maybe be different scale for each
        jitter_gamma  = np.random.uniform(-0.1, 0.1, size=1)
        
        jitter = np.concatenate([jitter_alphas, jitter_betas, jitter_gamma])
        # Ensure beta stays non-negative
        start = np.clip(priors + jitter, [b[0] for b in bounds], [b[1] for b in bounds])
        starting_points.append(start)

    best_loss = np.inf
    best_theta = None
    best_result = None

    print(
        f"--- Beginning Multi-Start Optimization ({n_starts} starts, {len(transitions)} transitions) ---"
    )
    start_time = time.time()

    for idx, x0 in enumerate(starting_points):
        res = minimize(
            fun=compute_objective_loss,
            x0=x0,
            args=(transitions, model, l2_reg),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-7, "disp": False},
        )

        status_str = "CONVERGED" if res.success else "ITER_LIMIT"
        print(
            f"  Start {idx+1:02d}/{n_starts:02d} | Loss: {res.fun:.6f} | Status: {status_str}"
        )

        if res.fun < best_loss:
            best_loss = res.fun
            best_theta = res.x
            best_result = res

    elapsed = time.time() - start_time
    print(
        f"--- Optimization Complete in {elapsed:.2f}s | Best Loss: {best_loss:.6f} ---\n"
    )

    model.set_params(best_theta)
    alpha_df, global_params = format_parameter_summary(best_theta)

    return model, (alpha_df, global_params)



def format_parameter_summary(theta: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
    """Formats the optimal 16 parameters into interpretable tables."""
    model = AggregateLatentLogit()
    alpha, beta, gamma = model.unpack_params(theta)

    # 1. Pool-Specific Ideological Affinities (Alphas)
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

    # 2. Shared Structural & Intra-Bloc Parameters
    global_params = pd.Series({
        "Beta_Mass (Attraction to LC Base)": beta[0],
        "Beta_Cap (Sensitivity to Truncation)": beta[1],
        "Beta_Menu (Attraction to Options)": beta[2],
        "Gamma (Sub-proportionality)": gamma
    })

    return alpha_df, global_params

from collections import defaultdict

def run_clustered_bootstrap(transitions: list, n_iterations: int = 100, n_starts: int = 3):
    """
    Runs a Clustered Bootstrap (sampling whole electorates with replacement)
    to estimate parameter uncertainty (95% Confidence Intervals).
    """
    print(f"\n=== INITIATING CLUSTERED BOOTSTRAP ({n_iterations} Iterations) ===")
    
    # 1. Cluster transitions by electorate
    electorate_clusters = defaultdict(list)
    for t in transitions:
        electorate_clusters[t.district].append(t)
        
    districts = list(electorate_clusters.keys())
    n_districts = len(districts)
    print(f"Clustering complete: {n_districts} unique electorates found.")

    # 2. Storage for the 16 parameters across all runs
    bootstrap_results = []
    
    start_time = time.time()
    
    for i in range(n_iterations):
        # Sample electorates with replacement
        sampled_districts = np.random.choice(districts, size=n_districts, replace=True)
        
        # Build the resampled dataset
        resampled_transitions = []
        for d in sampled_districts:
            resampled_transitions.extend(electorate_clusters[d])
            
        # Fit the model on the resample (using fewer multi-starts for speed)
        model, _ = fit_preference_model(
            transitions=resampled_transitions, 
            n_starts=n_starts, 
            random_seed=np.random.randint(0, 100000)
        )
        
        bootstrap_results.append(model.theta)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Bootstrap Progress: {i + 1}/{n_iterations} completed ({elapsed:.1f}s)")

    # 3. Calculate Mean and 95% Confidence Intervals
    bootstrap_array = np.vstack(bootstrap_results) # Shape: (n_iterations, 16)
    
    means = np.mean(bootstrap_array, axis=0)
    ci_lower = np.percentile(bootstrap_array, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_array, 97.5, axis=0)
    
    # Format the output into a clean DataFrame
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

def inspect_electorate_fit(district_name: str, model: AggregateLatentLogit, transitions: list, latent_tables: dict):
    """
    Compares the full Latent Class Model fit using Shares and Count 0 Formation.
    """
    dist_transitions = [t for t in transitions if t.district == district_name]
    
    if not dist_transitions or district_name not in latent_tables:
        print(f"\n[!] Data for '{district_name}' missing.")
        return

    alpha, beta, gamma = model.unpack_params(model.theta)
    table = latent_tables[district_name]

    print(f"\n==========================================================================")
    print(f"  FULL STRUCTURAL FIT REPORT: {district_name.upper()}")
    print(f"==========================================================================")

    # ---------------------------------------------------------
    # PART 1: COUNT 0 FORMATION (Raw Volume)
    # ---------------------------------------------------------
    t0 = dist_transitions[0]
    s_hat_count0 = model.evaluate_state_allocation(
        t0.lc_mass, t0.active_cands_X, t0.c_cap_X, t0.menu_size_X,
        t0.cand_to_bloc_X, t0.cand_demand_X, alpha, beta, gamma
    )

    print(f"\n[ROUND LC] -> LA Count 0 Formation (Raw Electorate %)")
    print(f"{'Candidate':<10} | {'Empirical C0':<12} | {'Predicted (Ŝ_0)':<15} | {'Error':<10}")
    print("-" * 60)
    
    c0_actuals = table.count_0
    for cand in sorted(list(set(c0_actuals.keys()) | set(s_hat_count0.keys()))):
        act = c0_actuals.get(cand, 0.0)
        pred = s_hat_count0.get(cand, 0.0)
        print(f"{cand:<10} | {act:>12.4%} | {pred:>15.4%} | {pred - act:>+10.4%}")

    # ---------------------------------------------------------
    # PART 2: ELIMINATION ROUNDS (Flow Shares)
    # ---------------------------------------------------------
    total_sq_error = 0.0
    total_flow_points = 0

    for t in dist_transitions:
        delta_hat = model.predict_transition_delta(t, alpha, beta, gamma)
        sum_pred = sum(delta_hat.values())
        sum_actual = t.empirical_total_transferred
        
        print(f"\n[ROUND {t.round_index}] | Eliminated: {t.eliminated_cand} (Empirical Vol: {sum_actual:.2%})")
        print(f"{'Candidate':<10} | {'Actual Share':<12} | {'Predicted Share':<15} | {'Share Error':<10}")
        print("-" * 60)
        
        all_cands = sorted(list(set(t.empirical_transfers.keys()) | set(delta_hat.keys())))
        
        for cand in all_cands:
            actual_vol = t.empirical_transfers.get(cand, 0.0)
            pred_vol = delta_hat.get(cand, 0.0)
            
            actual_share = (actual_vol / sum_actual) if sum_actual > 0 else 0.0
            pred_share = (pred_vol / sum_pred) if sum_pred > 0 else 0.0
            
            diff = pred_share - actual_share
            
            if actual_share > 0 or pred_share > 0.01:
                total_sq_error += diff ** 2
                total_flow_points += 1
                
            print(f"{cand:<10} | {actual_share:>12.2%} | {pred_share:>15.2%} | {diff:>+10.2%}")

    rmse = np.sqrt(total_sq_error / max(1, total_flow_points))

    print("\n" + "=" * 60)
    print(f"Transfer Routing Accuracy (Shares) for {district_name}:")
    print(f"  Rounds Evaluated: {len(dist_transitions)}")
    print(f"  Share RMSE: {rmse:.4%} (Average routing error per candidate)")
    print("=" * 60)







# --- SNEAK PEEK USING EXISTING TRANSITION DATA ---
full_latent_tables = generate_latent_training_tables(
            DOP_table_dict=DOP_table_dict, 
            party_df=party_df, 
            LC_FPs_df=LC_FPs_df, 
            ia_adjustments_dict=ia_adjustments_dict
        )

full_transitions = build_training_sequences(
    latent_tables=full_latent_tables, 
    party_df=party_df, 
    lc_df=LC_FPs_df
)

print("--- SNEAK PEEK: BASS COALITION DEMANDS ---")
bass_transitions = [t for t in full_transitions if t.district == 'Bass']

if bass_transitions:

    t0 = bass_transitions[0]  # First round transition for Bass
    print(f"District: {t0.district}")
    print(f"Active Candidates (State X): {t0.active_cands_X}")
    print("\nCandidate Demands (cand_demand_X stored in transition):")
    
    for cand, demand in t0.cand_demand_X.items():
        bloc_idx = t0.cand_to_bloc_X.get(cand)
        print(f"  -> {cand} (Bloc Index {bloc_idx}): Demand = {demand:.4f} ({demand*100:.2f}%)")
        
    # Check Coalition intra-bloc share directly from the cached state
    bloc_cands = [c for c in t0.active_cands_X if t0.cand_to_bloc_X.get(c) == 1] # 1 is COAL
    sum_demand = sum(t0.cand_demand_X[c] for c in bloc_cands)
    
    print("\nResulting Coalition Intra-Bloc Shares:")
    for c in bloc_cands:
        share = t0.cand_demand_X[c] / sum_demand if sum_demand > 0 else 0.0
        print(f"  -> {c}: {share:.2%}")
else:
    print("[!] No transitions found for Bass.")




import pdb; pdb.set_trace()


# =====================================================================
# EXECUTION BLOCK
# =====================================================================
if 1:
    print("--- EXTRACTING ALL ELECTORATES ---")
    full_latent_tables = generate_latent_training_tables(
        DOP_table_dict=DOP_table_dict, 
        party_df=party_df, 
        LC_FPs_df=LC_FPs_df, 
        ia_adjustments_dict=ia_adjustments_dict
    )

    full_transitions = build_training_sequences(
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
            transitions=full_transitions, 
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
    inspect_electorate_fit('Eureka', final_model, full_transitions, full_latent_tables)
    inspect_electorate_fit('Bass', final_model, full_transitions, full_latent_tables)
    inspect_electorate_fit('Bentleigh', final_model, full_transitions, full_latent_tables)
    inspect_electorate_fit('Macedon', final_model, full_transitions, full_latent_tables)
    inspect_electorate_fit('Melbourne', final_model, full_transitions, full_latent_tables)



    print("\n--- RUNNING CLUSTERED BOOTSTRAP ---")
    # Taking it down to 50 iterations with 2 starts just to make it run faster for your first full test
    #bootstrap_summary = run_clustered_bootstrap(full_transitions, n_iterations=50, n_starts=2)

    # Save everything
    #with open("final_victorian_model.pkl", "wb") as f:
    #    pickle.dump(final_model, f)
    #bootstrap_summary.to_csv("parameter_confidence_intervals.csv")



def inject_lc_party_to_dop(
    district: str,
    cand_to_add: str,
    target_bloc: str,
    empirical_c0: Dict[str, float],
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit'
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Injects an LC-contesting party that did NOT run in the LA into an existing LA DOP table.
    Uses the First-Difference injection vector to cannibalize the existing field.
    
    Returns:
        new_c0: The updated empirical Count 0 with cand_to_add included.
        delta_vector: The exact vote shifts applied to the field (sums to 0.0).
    """
    BLOCS = ['ALP', 'COAL', 'Left', 'Right', 'Centre']
    b_idx_inject = BLOCS.index(target_bloc)
    
    # 1. Extract district LC mass
    lc_series = lc_df[district].dropna()
    if lc_series.sum() > 1.5:
        lc_series = lc_series / 100.0
        
    ideo_map = dict(zip(party_df['PartyAb'], party_df['Ideo_category']))
    ideo_map.update({'LNP': 'COAL', 'LP': 'COAL', 'NP': 'COAL', 'ALP': 'ALP'})
    
    def get_bloc(c):
        if str(c).startswith('IND'): return 'Centre'
        return ideo_map.get(c, 'Centre')

    lc_mass = np.zeros(5)
    for party, share in lc_series.items():
        lc_mass[BLOCS.index(get_bloc(party))] += share
    if lc_mass.sum() > 0:
        lc_mass = lc_mass / lc_mass.sum()

    cand_lc_demand = lc_series.get(cand_to_add, 0.0)
    if cand_lc_demand <= 1e-9:
        raise ValueError(f"Candidate {cand_to_add} has 0.0 or missing LC demand in {district}.")

    # 2. Build Menus A (Current) and B (Injected)
    menu_A = [c for c, v in empirical_c0.items() if v > 1e-9 and c != cand_to_add]
    menu_B = menu_A + [cand_to_add]

    def build_local_state(active_cands: List[str]):
        menu_size = np.zeros(5)
        c_cap = np.zeros(5)
        cand_to_bloc = {}
        cand_demand = {}
        
        for cand in active_cands:
            b = BLOCS.index(get_bloc(cand))
            cand_to_bloc[cand] = b
            menu_size[b] += 1.0
            
            if b == 1:
                u_m = empirical_c0.get(cand, 0.0)
            else:
                u_m = cand_lc_demand if cand == cand_to_add else (
                    lc_series.get('IND', 0.0) if str(cand).startswith('IND') else lc_series.get(cand, 0.0)
                )
            cand_demand[cand] = u_m
            if b != 1:
                c_cap[b] += u_m
                
        c_cap[0] = 1.0 if menu_size[0] > 0 else 0.0
        c_cap[1] = 1.0 if menu_size[1] > 0 else 0.0
        for b in [2, 3, 4]:
            c_cap[b] = min(1.0, c_cap[b] / lc_mass[b]) if lc_mass[b] > 0 else 0.0
            
        return menu_size, c_cap, cand_to_bloc, cand_demand

    ms_A, cap_A, ctb_A, cd_A = build_local_state(menu_A)
    ms_B, cap_B, ctb_B, cd_B = build_local_state(menu_B)

    # 3. Compute First-Difference: Delta = State(B) - State(A)
    alpha, beta, gamma = model.unpack_params(model.theta)
    s_hat_A = model.evaluate_state_allocation(lc_mass, menu_A, cap_A, ms_A, ctb_A, cd_A, alpha, beta, gamma)
    s_hat_B = model.evaluate_state_allocation(lc_mass, menu_B, cap_B, ms_B, ctb_B, cd_B, alpha, beta, gamma)

    delta_vector = {}
    for c in menu_B:
        delta_vector[c] = s_hat_B.get(c, 0.0) - s_hat_A.get(c, 0.0)

    # 4. Apply delta to empirical Count 0
    new_c0 = {}
    for c in menu_A:
        new_c0[c] = max(0.0, empirical_c0[c] + delta_vector[c])
    new_c0[cand_to_add] = max(0.0, delta_vector[cand_to_add])

    # Re-normalize to exact 1.0
    total = sum(new_c0.values())
    new_c0 = {c: v / total for c, v in new_c0.items()}

    return new_c0, delta_vector


# ==============================================================================
# 2. RECALIBRATE / DIAGNOSE A SPECIFIC PARTY (e.g. One Nation)
# ==============================================================================

def diagnose_and_recalibrate_party(
    party_abbr: str,
    contested_divs: List[str],
    latent_tables: dict,
    lc_df: pd.DataFrame,
    party_df: pd.DataFrame,
    model: 'AggregateLatentLogit'
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluates systematic over/under-estimation of a specific party in seats where it contested both LC & LA.
    Calculates a multiplicative elasticity scalar (k_party) for subsequent counterfactual injections.
    """
    results = []
    alpha, beta, gamma = model.unpack_params(model.theta)

    for div in contested_divs:
        if div not in latent_tables or div not in lc_df.columns:
            continue
            
        table = latent_tables[div]
        actual_c0_party = table.count_0.get(party_abbr, 0.0)
        if actual_c0_party <= 1e-9:
            continue

        # Reconstruct base menu allocation
        lc_series = lc_df[div].dropna()
        if lc_series.sum() > 1.5: lc_series = lc_series / 100.0
        
        # Build menu without party (Counterfactual M_without)
        menu_with = list(table.count_0.keys())
        menu_without = [c for c in menu_with if c != party_abbr]
        
        # Pull empirical baseline
        empirical_without = {c: table.count_0[c] for c in menu_without}
        tot = sum(empirical_without.values())
        empirical_without = {c: v / tot for c, v in empirical_without.items()}

        # Run model injection
        pred_c0, delta = inject_lc_party_to_dop(
            district=div,
            cand_to_add=party_abbr,
            target_bloc='Right',
            empirical_c0=empirical_without,
            lc_df=lc_df,
            party_df=party_df,
            model=model
        )

        pred_party_vote = pred_c0.get(party_abbr, 0.0)
        error = pred_party_vote - actual_c0_party
        ratio = actual_c0_party / pred_party_vote if pred_party_vote > 0 else 1.0

        results.append({
            'District': div,
            'Actual_LA_C0': actual_c0_party,
            'Model_Pred_C0': pred_party_vote,
            'Residual': error,
            'Actual_to_Pred_Ratio': ratio
        })

    diag_df = pd.DataFrame(results)
    
    # Calibration scalar: median actual/pred ratio across tested districts
    calibration_scalar = float(diag_df['Actual_to_Pred_Ratio'].median()) if not diag_df.empty else 1.0
    
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
        print(f"--- ROUND {t.round_index} (Eliminating {t.eliminated_cand}) ---")
        
        # We trace State X (Pre-Elimination) to see how the model calculates S_hat_X
        active = t.active_cands_X
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
            u = cand_effective_lc if cand == cand_to_add else lc_series.get(cand, 0.0)
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
    
       
# 1. Grab the cleaned empirical Count 0 for Eureka
eureka_table = full_latent_tables['Eureka']
eureka_empirical_c0 = eureka_table.count_0.copy()

# 2. Run the injection function
new_eureka_c0, eureka_deltas = inject_lc_party_to_dop(
    district='Eureka',
    cand_to_add='ON',
    target_bloc='Right',
    empirical_c0=eureka_empirical_c0,
    lc_df=LC_FPs_df,
    party_df=party_df,
    model=final_model
)

# 3. Print the results to see the cannibalization
print("=== INJECTING ONE NATION INTO EUREKA ===")
print(f"{'Candidate':<10} | {'Base C0':<12} | {'New C0':<12} | {'Delta (Shift)':<12}")
print("-" * 55)
for cand in sorted(set(eureka_empirical_c0.keys()) | {'ON'}):
    base_val = eureka_empirical_c0.get(cand, 0.0)
    new_val = new_eureka_c0.get(cand, 0.0)
    shift = eureka_deltas.get(cand, 0.0)
    print(f"{cand:<10} | {base_val:>12.4%} | {new_val:>12.4%} | {shift:>+12.4%}")


on_contested_divs = ['Macedon', 'Morwell', 'Pakenham','Bendigo East'] 

diag_df, on_scalar = diagnose_and_recalibrate_party(
    party_abbr='ON',
    contested_divs=on_contested_divs,
    latent_tables=full_latent_tables,
    lc_df=LC_FPs_df,
    party_df=party_df,
    model=final_model
)

print("\n=== ONE NATION DIAGNOSTIC REPORT ===")
print(diag_df.to_string(index=False))
print(f"=> ON Calibration Scalar (Median Actual/Pred): {on_scalar:.4f}\n")

# 2. Run the detailed trace on Eureka, applying the extracted scalar
trace_injection_mechanics_detailed(
    district='Eureka',
    cand_to_add='ON',
    target_bloc='Right',
    empirical_c0=full_latent_tables['Eureka'].count_0,
    lc_df=LC_FPs_df,
    party_df=party_df,
    model=final_model,
    calibration_scalar=on_scalar
)


#trace_electorate_mechanics('Eureka', final_model, full_transitions)

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