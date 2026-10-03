import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle

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



class DOPTable:
    def __init__(self, df):
        """
        Initializes the DOPTable from a standard pandas DataFrame.
        """
        self.df = df.copy()
        self.div_nm = self.df['div_nm'].iloc[0]
        
        # Identify candidate columns
        skip_cols = {'div_nm', 'CandRemaining', 'TOTAL', 'CountType'}
        self.candidates = [c for c in self.df.columns if c not in skip_cols]
        
        self.count_0 = {}
        self.rounds = []
        self._parse_dataframe()

    @classmethod
    def _from_raw(cls, candidates, count_0, rounds, div_nm):
        """Internal constructor for generating new counterfactual tables."""
        obj = cls.__new__(cls)
        obj.candidates = candidates
        obj.count_0 = count_0
        obj.rounds = rounds
        obj.div_nm = div_nm
        return obj

    def _parse_dataframe(self):
        """Parses the DataFrame into a chronologically ordered sequence of eliminations."""
        votes_rows = self.df[self.df['CountType'] == 'Votes'].sort_values('CandRemaining', ascending=False)
        trans_rows = self.df[self.df['CountType'] == 'TransferredVotes'].sort_values('CandRemaining', ascending=False)
        
        self.count_0 = votes_rows.iloc[0][self.candidates].dropna().to_dict()
        
        prev_votes = self.count_0
        for _, t_row in trans_rows.iterrows():
            cand_rem = t_row['CandRemaining']
            
            # Find the eliminated candidate (present in prev_votes, but NaN in transferred row)
            # Get the 'Votes' totals for the new state AFTER elimination
            v_row = votes_rows[votes_rows['CandRemaining'] == cand_rem].iloc[0]
            curr_votes = v_row[self.candidates].dropna().to_dict()
            
            # The true eliminated candidate is the one who disappeared from the totals
            eliminated = None
            for c in prev_votes.keys():
                if c not in curr_votes:
                    eliminated = c
                    break
                    
            if eliminated is None:
                continue
                
            # Extract transfers ONLY for surviving candidates (safely handling 0s/NaNs)
            transfers = {}
            for c in curr_votes.keys():
                val = t_row.get(c, np.nan)
                if pd.notna(val) and float(val) > 1e-9:
                    transfers[c] = float(val)
            
            self.rounds.append({
                'eliminated': eliminated,
                'V_elim': prev_votes[eliminated],
                'transfers': transfers
            })
            
            prev_votes = curr_votes

    def _wallenius_extract(self, amount, weights, bounds):
        """
        True Wallenius batch-sequential sampler.
        Executes discrete draws around safe mean weights while dynamically preventing 0-collisions.
        """
        amount = int(np.round(amount))
        C = {j: float(bounds.get(j, 0.0)) for j in weights}
        m = {j: int(np.floor(C[j])) for j in weights}
        drawn = {j: 0 for j in weights}
        
        remaining_to_draw = min(amount, sum(m.values()))
        
        while remaining_to_draw > 0:
            intensities = {}
            for j in weights:
                if m[j] <= 0 or C[j] <= 1e-9 or weights.get(j, 0) <= 1e-9:
                    intensities[j] = 0.0
                else:
                    # Wallenius Depletion Factor: (Remaining / Capacity)
                    # Acts as the localized shock-absorber during the Monte Carlo draw
                    intensities[j] = weights[j] * (m[j] / C[j])
            
            sum_int = sum(intensities.values())
            
            if sum_int <= 1e-12:
                # Fallback: physical density if ideological weights are totally exhausted
                active = [j for j in weights if m[j] > 0]
                if not active: break
                probs = np.array([m[j] for j in active], dtype=float)
            else:
                active = [j for j in weights if intensities[j] > 0]
                probs = np.array([intensities[j] for j in active], dtype=float)
                
            probs /= probs.sum()
            
            # Adaptive batching: scales to the smallest active pile to ensure smooth probability updates
            safe_batch = min(remaining_to_draw, max(1, min([m[j] for j in active]) // 4))
            batch_draws = np.random.multinomial(safe_batch, probs)
            
            for idx, j in enumerate(active):
                if batch_draws[idx] > 0:
                    actual = min(batch_draws[idx], m[j])
                    drawn[j] += actual
                    m[j] -= actual
                    remaining_to_draw -= actual
                    
        return drawn

    def _safe_extract(self, amount, weights, max_bounds):
        """
        Bounded Water-Filling: Extracts 'amount' according to 'weights', 
        ensuring no extraction exceeds the available 'max_bounds'.
        """
        extraction = {j: 0.0 for j in weights}
        remaining_amount = amount
        current_bounds = max_bounds.copy()
        active_candidates = [j for j in weights if current_bounds.get(j, 0) > 1e-9 and weights.get(j, 0) > 1e-9]

        while remaining_amount > 1e-9 and active_candidates:
            w_sum = sum(weights[j] for j in active_candidates)
            if w_sum < 1e-12: 
                break
            
            norm_weights = {j: weights[j] / w_sum for j in active_candidates}
            proposed = {j: remaining_amount * norm_weights[j] for j in active_candidates}
            
            # Find the bottleneck scale to prevent breaching zero
            scale = 1.0
            for j in active_candidates:
                if proposed[j] > 1e-12:
                    scale = min(scale, current_bounds[j] / proposed[j])
                    
            for j in list(active_candidates):
                ext = proposed[j] * scale
                extraction[j] += ext
                current_bounds[j] -= ext
                remaining_amount -= ext
                
                if current_bounds[j] <= 1e-9:
                    print(f"\n[BOUNDARY ALERT] The terminal pile for {j} just hit 0!")
                    print(f"Target extraction remainder: {remaining_amount:.2f} votes.")
                    print("Dropping into debugger so you can inspect the state...")
                    import pdb; pdb.set_trace()

                    active_candidates.remove(j)
                    
        return extraction

    def remove_candidate(self, target_X, lambda_val=0.75, gamma = 0.5):
        """
        Executes the Reverse-Chronological extraction algorithm for a single candidate.
        Returns a new DOPTable object.
        """
        r_X_idx = next((i for i, rnd in enumerate(self.rounds) if rnd['eliminated'] == target_X), -1)
        if r_X_idx == -1:
            raise ValueError(f"{target_X} not found in elimination history.")

        rnd_X = self.rounds[r_X_idx]
        T_X = rnd_X['transfers']

        M = list(T_X.keys())
        pile_X = {j: T_X[j] for j in M}  # Tracks the terminal votes of X backwards

        cf_rounds = []

        # 1. Backpropagate up to Count 0
        for i in range(r_X_idx - 1, -1, -1):
            rnd_k = self.rounds[i]
            k = rnd_k['eliminated']

            #print(f"\n[DEBUG] Un-mixing candidate: {k}")
            #import pdb; pdb.set_trace()


            V_k = rnd_k['V_elim'] # candidate k vote in round k
            T_k = rnd_k['transfers']

            T_k_X = T_k.get(target_X, 0.0) # transfer from k to X
            M_k = [j for j in T_k.keys() if j != target_X] # excludes X from list

            # flows from X
            sum_pile = sum(pile_X.values()) # X votes when excluded
            pi_X = {j: pile_X.get(j, 0) / sum_pile if sum_pile > 0 else 0 for j in M} # proportions
            # flows from k (excluding X)
            sum_eta = sum(T_k[j] for j in M_k)
            eta_k = {j: T_k[j] / sum_eta if sum_eta > 0 else 0 for j in M_k}

            # Ideological mixture vectors
            w_prior_2a = {j: (1 - lambda_val) * pi_X.get(j, 0) + lambda_val * eta_k.get(j, 0) for j in M} # mostly eta_k
            w_prior_2b = {j: lambda_val * pi_X.get(j, 0) + (1 - lambda_val) * eta_k.get(j, 0) for j in M} # mostly pi_X

            # exponential shrinkage away from 0
            base_vol_X = max(0, sum_pile - T_k_X) # The true base volume of X before receiving k's transfer

            # demand by symmetry, scaled (geometrically) by vote share ratio to avoid bias against big parties
            theta = 0.5
            if V_k > 0 and base_vol_X > 0:
                T_hat_X_k = T_k_X * ((base_vol_X / V_k)**theta)
            else:
                T_hat_X_k = 0
            
            gamma = gamma  # continuous decay rate
            alpha = 3.0  # The geometric steepness of the barrier cliff
            
            log_barriers = {}
            for j in M:
                C_j = pile_X.get(j, 0)
                if C_j <= 1e-9:
                    log_barriers[j] = -float('inf')
                else:
                    # Combined Expected Demand
                    D_j = (w_prior_2a.get(j, 0) * T_k_X) + (w_prior_2b.get(j, 0) * T_hat_X_k)
                    
                    # Depletion Ratio
                    rho_j = D_j / C_j
                    
                    # Immediate continuous penalty (tau = 0)
                    log_barriers[j] = -gamma * rho_j**alpha
            
            # Log-sum-exp trick for numerical stability
            max_log_b = max(log_barriers.values()) if log_barriers else 0
            
            w_2a_safe, w_2b_safe = {}, {}
            for j in M:
                if log_barriers[j] == -float('inf'):
                    w_2a_safe[j] = 0.0
                    w_2b_safe[j] = 0.0
                else:
                    barrier = np.exp(log_barriers[j] - max_log_b)
                    w_2a_safe[j] = w_prior_2a.get(j, 0) * barrier
                    w_2b_safe[j] = w_prior_2b.get(j, 0) * barrier
                    
            # Normalize
            sum_w_2a = sum(w_2a_safe.values())
            sum_w_2b = sum(w_2b_safe.values())
            
            w_2a = {j: w_2a_safe[j] / sum_w_2a if sum_w_2a > 0 else 0 for j in M}
            w_2b = {j: w_2b_safe[j] / sum_w_2b if sum_w_2b > 0 else 0 for j in M}

            #import pdb; pdb.set_trace()

            # 2a: Extract historical T_{k -> X}
            withdrawal_2a = self._safe_extract(T_k_X, w_2a, pile_X)
            for j in M: 
                pile_X[j] -= withdrawal_2a.get(j, 0)

            # 2b: Extract counterfactual symmetry T_{X -> k}
            withdrawal_2b = self._safe_extract(T_hat_X_k, w_2b, pile_X)
            for j in M: 
                pile_X[j] -= withdrawal_2b.get(j, 0)

            # Re-integrate k into the latent pool
            pile_X[k] = sum(withdrawal_2b.values()) # total votes from X to k
            M.append(k)

            # Build counterfactual transfers from k
            T_star_k = {}
            for j in M:
                if j == k: continue
                val = T_k.get(j, 0) + withdrawal_2a.get(j, 0) + withdrawal_2b.get(j, 0)
                if val > 1e-9:
                    T_star_k[j] = val

            cf_rounds.insert(0, {
                'eliminated': k,
                'V_elim': sum(T_star_k.values()),
                'transfers': T_star_k
            })

            #print(pile_X)
            #print(cf_rounds)
            #import pdb; pdb.set_trace()

        # 2. Form Latent Count 0 (X's Primary vote is gracefully returned to M)
        cf_count_0 = {}
        for c, v in self.count_0.items():
            if c != target_X:
                cf_count_0[c] = v + pile_X.get(c, 0)

        # 3. Append subsequent elimination rounds (unchanged terminal transfers)
        for i in range(r_X_idx + 1, len(self.rounds)):
            rnd = self.rounds[i]
            cf_rounds.append({
                'eliminated': rnd['eliminated'],
                'V_elim': rnd['V_elim'],
                'transfers': rnd['transfers'].copy()
            })

        new_candidates = [c for c in self.candidates if c != target_X]
        return DOPTable._from_raw(new_candidates, cf_count_0, cf_rounds, self.div_nm)

    def remove_candidates(self, targets, lambda_val=0.75, gamma = 0.5):
        """
        Removes multiple candidates symmetrically. 
        Automatically processes them in reverse-elimination order (outermost doll first)
        to prevent matrix cross-contamination.
        """
        elim_indices = {rnd['eliminated']: i for i, rnd in enumerate(self.rounds) if rnd['eliminated'] in targets}
        sorted_targets = sorted(elim_indices.keys(), key=lambda x: elim_indices[x], reverse=True)

        current_table = self
        for t in sorted_targets:
            current_table = current_table.remove_candidate(t, lambda_val, gamma = 0.5)

        return current_table

    def remove_candidate_stochastic(self, target_X, lambda_base=0.75, gamma_base=0.5, dirichlet_K=500):
        """
        Executes a single stochastic Monte Carlo run of the Reverse-Chronological extraction algorithm.
        Returns a new DOPTable object.
        """
        import numpy as np
        
        r_X_idx = next((i for i, rnd in enumerate(self.rounds) if rnd['eliminated'] == target_X), -1)
        if r_X_idx == -1:
            raise ValueError(f"{target_X} not found in elimination history.")

        rnd_X = self.rounds[r_X_idx]
        T_X = rnd_X['transfers']

        M = list(T_X.keys())
        pile_X = {j: T_X[j] for j in M}

        cf_rounds = []
        
        # =====================================================================
        # 1. TIER 1: SYSTEMIC EPISTEMIC UNCERTAINTY 
        # Set the "political climate" parameters for this specific simulation
        # =====================================================================
        
        # lambda: Ideological mixture (Mean ~0.75). Direct beta natively bounds [0, 1].
        # a=15, b=5 naturally peaks at 0.75 and provides realistic density down to ~0.50.
        sampled_lambda = np.random.beta(15, 5) 
        
        # theta (Gravity): Uniformly sample to allow massive structural swings in T_hat.
        # 0.1 = Strong major party protection (low flow). 0.6 = High rate symmetry (massive flow).
        sampled_theta = np.random.uniform(0.1, 0.6)
        
        # gamma (Barrier Strength): Soft variance +/- 20% around the base to prevent algorithmic rigidity
        sampled_gamma = gamma_base * np.random.uniform(0.8, 1.2)

        # =====================================================================
        # 2. THE BACKPROPAGATION LOOP
        # =====================================================================
        for i in range(r_X_idx - 1, -1, -1):
            rnd_k = self.rounds[i]
            k = rnd_k['eliminated']
            V_k = rnd_k['V_elim']
            T_k = rnd_k['transfers']

            T_k_X = T_k.get(target_X, 0.0)
            M_k = [j for j in T_k.keys() if j != target_X]

            sum_pile = sum(pile_X.values())
            sum_eta = sum(T_k[j] for j in M_k)
            
            # --- DIRICHLET JITTER ON HISTORICAL FLOWS ---
            # Mimics natural behavioral polling variance (Margin of Error)
            base_pi = np.array([pile_X.get(j, 0) / sum_pile if sum_pile > 0 else 0 for j in M])
            base_eta = np.array([T_k[j] / sum_eta if sum_eta > 0 else 0 for j in M_k])
            
            if dirichlet_K and sum_pile > 0 and sum_eta > 0:
                # Add 1e-3 to prevent log(0) errors in the Dirichlet sampler for empty piles
                jittered_pi = np.random.dirichlet(np.maximum(base_pi * dirichlet_K, 1e-3))
                jittered_eta = np.random.dirichlet(np.maximum(base_eta * dirichlet_K, 1e-3))
                pi_X = {j: jittered_pi[idx] for idx, j in enumerate(M)}
                eta_k = {j: jittered_eta[idx] for idx, j in enumerate(M_k)}
            else:
                pi_X = {j: base_pi[idx] for idx, j in enumerate(M)}
                eta_k = {j: base_eta[idx] for idx, j in enumerate(M_k)}

            w_prior_2a = {j: (1 - sampled_lambda) * pi_X.get(j, 0) + sampled_lambda * eta_k.get(j, 0) for j in M}
            w_prior_2b = {j: sampled_lambda * pi_X.get(j, 0) + (1 - sampled_lambda) * eta_k.get(j, 0) for j in M}

            base_vol_X = max(0, sum_pile - T_k_X)

            # --- STOCHASTIC SYMMETRY (T_hat) ---
            # Completely deterministic calculation based on the radically uniform sampled_theta
            if V_k > 0 and base_vol_X > 0:
                T_hat_X_k = T_k_X * ((base_vol_X / V_k) ** sampled_theta)
            else:
                T_hat_X_k = 0.0
            
            # --- TIER 2: GLOBAL REGULARIZER (The Expected Mean Calibrator) ---
            # Throttles ideological demand prior to execution to protect shrinking capacities
            alpha = 3.0  
            log_barriers = {}
            for j in M:
                C_j = pile_X.get(j, 0)
                if C_j <= 1e-9:
                    log_barriers[j] = -float('inf')
                else:
                    D_j = (w_prior_2a.get(j, 0) * T_k_X) + (w_prior_2b.get(j, 0) * T_hat_X_k)
                    rho_j = D_j / C_j
                    log_barriers[j] = -sampled_gamma * (rho_j ** alpha)
            
            max_log_b = max(log_barriers.values()) if log_barriers else 0
            
            w_2a_safe, w_2b_safe = {}, {}
            for j in M:
                if log_barriers[j] == -float('inf'):
                    w_2a_safe[j] = 0.0
                    w_2b_safe[j] = 0.0
                else:
                    barrier = np.exp(log_barriers[j] - max_log_b)
                    w_2a_safe[j] = w_prior_2a.get(j, 0) * barrier
                    w_2b_safe[j] = w_prior_2b.get(j, 0) * barrier
                    
            sum_w_2a = sum(w_2a_safe.values())
            sum_w_2b = sum(w_2b_safe.values())
            
            # These are the mathematically safe expected target weights
            w_2a = {j: w_2a_safe[j] / sum_w_2a if sum_w_2a > 0 else 0 for j in M}
            w_2b = {j: w_2b_safe[j] / sum_w_2b if sum_w_2b > 0 else 0 for j in M}

            # --- TIER 3: ALEATORIC EXECUTION (Wallenius) ---
            # Discrete, physically aware sampling that adds natural variance around the safe mean
            withdrawal_2a = self._wallenius_extract(T_k_X, w_2a, pile_X)
            for j in M: 
                pile_X[j] -= withdrawal_2a.get(j, 0)

            withdrawal_2b = self._wallenius_extract(T_hat_X_k, w_2b, pile_X)
            for j in M: 
                pile_X[j] -= withdrawal_2b.get(j, 0)

            # --- RE-INTEGRATION ---
            pile_X[k] = sum(withdrawal_2b.values())
            M.append(k)

            T_star_k = {}
            for j in M:
                if j == k: continue
                val = T_k.get(j, 0) + withdrawal_2a.get(j, 0) + withdrawal_2b.get(j, 0)
                if val > 1e-9:
                    T_star_k[j] = val

            cf_rounds.insert(0, {
                'eliminated': k,
                'V_elim': sum(T_star_k.values()),
                'transfers': T_star_k
            })

        # =====================================================================
        # 3. FORM THE COUNTERFACTUAL TABLE
        # =====================================================================
        cf_count_0 = {}
        for c, v in self.count_0.items():
            if c != target_X:
                cf_count_0[c] = v + pile_X.get(c, 0)

        for i in range(r_X_idx + 1, len(self.rounds)):
            rnd = self.rounds[i]
            cf_rounds.append({
                'eliminated': rnd['eliminated'],
                'V_elim': rnd['V_elim'],
                'transfers': rnd['transfers'].copy()
            })

        new_candidates = [c for c in self.candidates if c != target_X]
        return self.__class__._from_raw(new_candidates, cf_count_0, cf_rounds, self.div_nm)

    def to_dataframe(self):
        """Reconstructs the standard DataFrame format from internal state."""
        rows = []
        cand_remaining = len(self.candidates)

        # Count 0 row
        row0 = {'div_nm': self.div_nm, 'CandRemaining': cand_remaining, 'CountType': 'Votes'}
        for c in self.candidates: 
            row0[c] = self.count_0.get(c, np.nan)
        row0['TOTAL'] = sum(self.count_0.values())
        rows.append(row0)

        current_votes = self.count_0.copy()

        # Elimination rounds
        for rnd in self.rounds:
            cand_remaining -= 1
            elim = rnd['eliminated']
            trans = rnd['transfers']

            # TransferredVotes row
            t_row = {'div_nm': self.div_nm, 'CandRemaining': cand_remaining, 'CountType': 'TransferredVotes'}
            for c in self.candidates:
                if c == elim or current_votes.get(c) is None:
                    t_row[c] = np.nan
                else:
                    t_row[c] = trans.get(c, 0.0)
            t_row['TOTAL'] = sum(trans.values())
            rows.append(t_row)

            # Apply transfers to Cumulative Votes
            current_votes[elim] = None
            for c, v in trans.items():
                if current_votes.get(c) is not None:
                    current_votes[c] += v

            # Votes row
            v_row = {'div_nm': self.div_nm, 'CandRemaining': cand_remaining, 'CountType': 'Votes'}
            v_total = sum(v for v in current_votes.values() if v is not None)
            for c in self.candidates:
                v_row[c] = current_votes.get(c, np.nan)
            v_row['TOTAL'] = v_total
            rows.append(v_row)

        return pd.DataFrame(rows)

    def strip_incumbency(self, ia_adjustments: dict):
        """
        Strips Incumbency Advantage at the 3CP stage and back-propagates it 
        proportionally to all minor parties by reversing the historical donation fractions.
        """
        # 1. Forward pass to map the exact historical vote tallies at every round
        history = [self.count_0.copy()]
        for rnd in self.rounds:
            prev = history[-1].copy()
            prev[rnd['eliminated']] = 0.0  
            for c, v in rnd['transfers'].items():
                prev[c] += v
            history.append(prev)

        # 2. Locate the 3CP round index
        r_3cp_idx = -1
        for i, h in enumerate(history):
            active_cands = sum(1 for v in h.values() if v > 1e-9)
            if active_cands == 3:
                r_3cp_idx = i
                break
                
        if r_3cp_idx == -1:
            raise ValueError("Could not locate 3CP stage in this DOP Table.")

        # 3. Apply the IA Adjustments to the 3CP State
        state_3cp_cf = history[r_3cp_idx].copy()
        
        # Track any vote mass lost due to flooring tiny parties at 1e-9
        total_mass_deficit = 0.0

        for cand, adj in ia_adjustments.items():
            if cand in state_3cp_cf:
                desired_val = state_3cp_cf[cand] + adj
                
                if desired_val < 1e-9:
                    # Calculate how much vote mass couldn't be subtracted because of the floor
                    deficit = 1e-9 - desired_val
                    total_mass_deficit += deficit
                    state_3cp_cf[cand] = 1e-9
                else:
                    state_3cp_cf[cand] = desired_val

        # If flooring a minor party slightly altered the total sum, 
        # let the largest party (ALP or LP) absorb the tiny residual to preserve 100.0% sum.
        if total_mass_deficit > 1e-12:
            largest_cand = max(state_3cp_cf, key=state_3cp_cf.get)
            state_3cp_cf[largest_cand] -= total_mass_deficit

        # 4. Reverse-Propagate: Un-mix the table backwards to Count 0
        cf_state = state_3cp_cf.copy()
        cf_rounds = []
        
        for i in range(r_3cp_idx - 1, -1, -1):
            rnd = self.rounds[i]
            k = rnd['eliminated']
            orig_transfers = rnd['transfers']
            orig_V_elim = rnd['V_elim']
            orig_state_after = history[i + 1]

            cf_transfers = {}
            cf_state_before = cf_state.copy()

            for j, t_vol in orig_transfers.items():
                if orig_state_after[j] > 1e-9:
                    donation_frac = t_vol / orig_state_after[j]
                else:
                    donation_frac = 0.0

                cf_t = cf_state[j] * donation_frac
                cf_transfers[j] = cf_t
                cf_state_before[j] -= cf_t

            sum_orig_t = sum(orig_transfers.values())
            sum_cf_t = sum(cf_transfers.values())

            if sum_orig_t > 1e-9:
                cf_V_elim = sum_cf_t * (orig_V_elim / sum_orig_t)
            else:
                cf_V_elim = sum_cf_t

            cf_state_before[k] = cf_V_elim
            cf_state = cf_state_before 
            
            cf_rounds.insert(0, {
                'eliminated': k,
                'V_elim': cf_V_elim,
                'transfers': cf_transfers
            })

        # --- FIX IS HERE: Save the reconstructed Count 0 immediately! ---
        cf_count_0 = cf_state.copy()

        # 5. Append downstream rounds (post-3CP) using standard forward proportional flow
        # --- FIX: Reset cf_state back to 3CP before simulating forwards! ---
        cf_state = state_3cp_cf.copy() 
        
        for i in range(r_3cp_idx, len(self.rounds)):
            rnd = self.rounds[i]
            k = rnd['eliminated']
            orig_transfers = rnd['transfers']
            orig_V_elim = rnd['V_elim']
            
            cf_V_elim = cf_state[k]
            cf_transfers = {}
            sum_orig_t = sum(orig_transfers.values())
            
            if sum_orig_t > 1e-9:
                for j, t_vol in orig_transfers.items():
                    prop = t_vol / sum_orig_t
                    cf_transfers[j] = cf_V_elim * prop
            
            cf_state[k] = 0.0
            for j, v in cf_transfers.items():
                cf_state[j] += v
                
            cf_rounds.append({
                'eliminated': k,
                'V_elim': cf_V_elim,
                'transfers': cf_transfers
            })

        # 6. Instantiate the cleansed table
        for c in history[0].keys():
            if c not in cf_count_0:
                cf_count_0[c] = history[0][c]

        return DOPTable._from_raw(self.candidates.copy(), cf_count_0, cf_rounds, self.div_nm)

