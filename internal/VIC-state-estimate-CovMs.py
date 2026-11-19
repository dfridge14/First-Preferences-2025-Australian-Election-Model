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


os.chdir(Path.home() / 'Australian Election/Victorian Election')



DOP_table_dict_year = {}
for data_year in [2006,2010,2014,2018]:
    with open(f'VIC-{data_year}/{data_year}_complete_DOP_table_dict.pkl', 'rb') as f:
        DOP_table_dict = pickle.load(f)
        if data_year == 2022:
            del DOP_table_dict['Narracan'] # FIX: Narracan dealt with separately!
        DOP_table_dict_year[data_year] = DOP_table_dict


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

def get_poll_bucket(cand_name: str, data_year: int = 2002) -> str:
    party = cand_name.split('_')[0] if '_' in cand_name else cand_name
    if party in ['LP', 'NP', 'LNP']: return 'COAL'
    if party == 'ALP': return 'ALP'
    if party == 'GRN': return 'GRN'
    if party == 'ON' and data_year == 2022: return 'ON' 
    # if str(cand_name).startswith('IND'): return 'IND'
    return 'OTH'


def compute_statewide_alr_shifts(
    cf_c0_step5_dict: dict,
    kalman_poll: dict,
    dop_table_dict: dict,
    data_year: int = 2002,
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

def format_observed_la_results(
    la_results_df: pd.DataFrame,
    alr_blocs: List[str] = ['ALP', 'GRN', 'OTH'], # Numerator blocs (K-1)
    ref_bloc: str = 'COAL',                       # Reference denominator bloc
    eps_alr: float = 1e-4
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame, Dict[str, float]]:
    """
    1. Excludes INFORMAL rows and normalizes votes to percentages per electorate.
    2. Replaces IND1, IND2, etc. in PartyAb with clean_ind_name("First Last").
    3. Returns:
       - clean_df: DataFrame with columns ['div_nm', 'PartyAb', 'pct']
       - obs_c0_dict: {div_nm: {PartyAb: pct}} (matches simulation Step 6 c0 format)
       - Y_obs_df: DataFrame of shape (88, K-1) in ALR space (Input 2)
       - true_statewide_shares: {PartyAb/Bloc: statewide_share} for zero-error polling calibration
    """
    # 1. Exclude INFORMAL
    df = la_results_df[
        la_results_df['PartyAb'].astype(str).str.upper().str.strip() != 'INFORMAL'
    ].copy()
    df['votes'] = df['votes'].astype(float)

    # 2. Replace IND* in PartyAb with clean_ind_name(first_name + surname)
    is_ind = df['PartyAb'].astype(str).str.upper().str.startswith('IND')
    raw_full_names = (
        df.loc[is_ind, 'first_name'].fillna('').astype(str).str.strip()
        + ' '
        + df.loc[is_ind, 'surname'].fillna('').astype(str).str.strip()
    )
    df.loc[is_ind, 'PartyAb'] = [clean_ind_name(nm) for nm in raw_full_names]

    # 3. Compute within-electorate percentages and statewide formal shares
    div_totals = df.groupby('div_nm')['votes'].transform('sum')
    df['pct'] = np.where(div_totals > 0, df['votes'] / div_totals, 0.0)

    clean_df = df[['div_nm', 'PartyAb', 'pct']].copy()

    # Build nested dictionary {div_nm: {PartyAb: pct}} (easiest for pipeline lookup)
    obs_c0_dict = {
        div: dict(zip(grp['PartyAb'], grp['pct']))
        for div, grp in clean_df.groupby('div_nm')
    }

    # 4. Aggregate into ALR blocs (ALP, GRN, OTH vs COAL) to build Input 2: Y_obs (88, K-1)
    alr_rows = {}
    for div, c0_map in sorted(obs_c0_dict.items()):
        bloc_shares = {'ALP': 0.0, 'COAL': 0.0, 'GRN': 0.0, 'OTH': 0.0}
        for p, share in c0_map.items():
            p_up = str(p).upper()
            if p_up == 'ALP':
                bloc_shares['ALP'] += share
            elif p_up in ('LP', 'NP', 'LNP', 'COAL'):
                bloc_shares['COAL'] += share
            elif p_up == 'GRN':
                bloc_shares['GRN'] += share
            else:
                bloc_shares['OTH'] += share

        denom = max(bloc_shares[ref_bloc], eps_alr)
        alr_rows[div] = {
            f"ALR_{b}_{ref_bloc}": np.log(max(bloc_shares[b], eps_alr) / denom)
            for b in alr_blocs
        }

    Y_obs_df = pd.DataFrame.from_dict(alr_rows, orient='index')

    # 5. Compute true statewide primary vote shares (to act as zero-error 'kalman_poll')
    total_state_votes = df['votes'].sum()
    df['poll_bucket'] = [get_poll_bucket(p) for p in df['PartyAb']]
    state_bucket_totals = df.groupby('poll_bucket')['votes'].sum() / total_state_votes
    true_statewide_shares = state_bucket_totals.to_dict()

    return clean_df, obs_c0_dict, Y_obs_df, true_statewide_shares

def c0_dict_to_alr_vec(
    c0_map: Dict[str, float],
    alr_blocs: List[str] = ['ALP', 'GRN', 'OTH'],
    ref_bloc: str = 'COAL',
    eps_alr: float = 1e-4
) -> np.ndarray:
    """Maps a single seat's {party: share} dict to a (K-1,) ALR vector."""
    bloc_shares = {'ALP': 0.0, 'COAL': 0.0, 'GRN': 0.0, 'OTH': 0.0}
    for p, share in c0_map.items():
        p_up = str(p).upper()
        if p_up == 'ALP':
            bloc_shares['ALP'] += float(share)
        elif p_up in ('LP', 'NP', 'LNP', 'COAL'):
            bloc_shares['COAL'] += float(share)
        elif p_up == 'GRN':
            bloc_shares['GRN'] += float(share)
        else:
            bloc_shares['OTH'] += float(share)

    denom = max(bloc_shares[ref_bloc], eps_alr)
    return np.array(
        [np.log(max(bloc_shares[b], eps_alr) / denom) for b in alr_blocs],
        dtype=np.float64
    )

def apply_deterministic_poll_shift_wrapper(
    step6_c0_dict: dict,
    target_statewide_poll: dict,
    simulate: bool,          # Ignored: forced to False in this context
    # Additional kwargs you might need:
    dop_table_dict: dict = {},
    data_year: int = 2002
) -> dict:
    """
    Computes the deterministic ALR shifts to hit the historical target, 
    and applies them via Softmax equivalent to produce final post-shift electorates.
    """
    # 1. Calculate the ALR shifts using your existing function
    alr_shifts = compute_statewide_alr_shifts(
        cf_c0_step5_dict=step6_c0_dict, 
        kalman_poll=target_statewide_poll, 
        dop_table_dict=dop_table_dict, 
        data_year=data_year, 
        ref_col='COAL'
    )
    
    # 2. Apply the shifts back to every electorate (Softmax / Inverse-ALR)
    post_shift_dict = {}
    for div, step6_shares in step6_c0_dict.items():
        shifted_c0 = {}
        for cand, prop in step6_shares.items():
            if prop > 1e-9:
                bucket = get_poll_bucket(cand, data_year)
                shift = alr_shifts.get(bucket, 0.0)
                
                # Math trick: Adding Delta in ALR space is mathematically identical 
                # to multiplying by exp(Delta) in proportion space!
                shifted_c0[cand] = prop * np.exp(shift)
            else:
                shifted_c0[cand] = 0.0
                
        # Renormalize the electorate so it sums to 1.0 again
        tot = sum(shifted_c0.values())
        if tot > 0:
            post_shift_dict[div] = {c: v / tot for c, v in shifted_c0.items()}
        else:
            post_shift_dict[div] = step6_shares.copy()
            
    return post_shift_dict



def collect_deconvolution_inputs_for_year(
    target_year: int,
    pre_poll_sims_step6: List[Dict[str, Dict[str, float]]], # Length S=100 list of {div: c0_step6_dict}
    ind_glm_meta_by_div: Dict[str, Dict[str, float]],       # {div: {'k': k, 'theta': theta_i}} for seats with IND GLM active
    la_results_df: pd.DataFrame,
    alr_blocs: List[str] = ['ALP', 'GRN', 'OTH'],
    ref_bloc: str = 'COAL'
) -> Dict[str, Any]:
    """
    Takes the 100 pre-polling Step 6 simulation dicts for `target_year`, applies the
    zero-error historical statewide polling shift to each draw, and packages the 5 inputs.
    """
    # 1. Format LA_results to get ordered electorates, Y_obs (Input 2), and true statewide target
    clean_df, obs_c0_dict, Y_obs_df, true_statewide_shares = format_observed_la_results(
        la_results_df, alr_blocs=alr_blocs, ref_bloc=ref_bloc
    )
    divs_sorted = list(Y_obs_df.index) # Canonical 88-seat order
    N_seats = len(divs_sorted)
    S = len(pre_poll_sims_step6)
    D = len(alr_blocs)

    # 2. Build Input 1: is_clean_seat (shape: (88,))
    is_clean_seat = np.array(
        [div not in ind_glm_meta_by_div for div in divs_sorted],
        dtype=bool
    )
    clean_divs = [d for d, is_c in zip(divs_sorted, is_clean_seat) if is_c]
    ind_divs   = [d for d, is_c in zip(divs_sorted, is_clean_seat) if not is_c]

    # 3. Build Input 2: Y_obs (shape: (88, K-1))
    Y_obs = Y_obs_df.loc[divs_sorted].to_numpy(dtype=np.float64)

    # 4. Apply deterministic statewide polling calibration (Sigma_poll = 0) to each of the 100 Step-6 draws
    Y_sim_clean = np.zeros((S, len(clean_divs), D), dtype=np.float64)
    Y_sim_IND   = np.zeros((S, len(ind_divs), D), dtype=np.float64)

    for s_idx, step6_dict_s in enumerate(pre_poll_sims_step6):
        # Push draw s_idx to match true_statewide_shares with zero polling noise (simulate=False)
        post_shift_dict_s = apply_deterministic_poll_shift_wrapper(
            step6_c0_dict=step6_dict_s,
            target_statewide_poll=true_statewide_shares,
            simulate=False,
            dop_table_dict=DOP_table_dict_year[target_year-4]
        )

        for c_i, div in enumerate(clean_divs):
            Y_sim_clean[s_idx, c_i, :] = c0_dict_to_alr_vec(
                post_shift_dict_s[div], alr_blocs=alr_blocs, ref_bloc=ref_bloc
            )

        for ind_i, div in enumerate(ind_divs):
            Y_sim_IND[s_idx, ind_i, :] = c0_dict_to_alr_vec(
                post_shift_dict_s[div], alr_blocs=alr_blocs, ref_bloc=ref_bloc
            )

    # 5. Build Input 5: GLM_params for the N_IND seats in matching order
    GLM_params = {
        'divs': ind_divs,
        'k': np.array([ind_glm_meta_by_div[d]['k'] for d in ind_divs], dtype=np.float64),
        'theta': np.array([ind_glm_meta_by_div[d]['theta'] for d in ind_divs], dtype=np.float64),
    }

    return {
        'year': target_year,
        'divs_sorted': divs_sorted,
        'is_clean_seat': is_clean_seat, # Input 1: (88,) bool
        'Y_obs': Y_obs,                 # Input 2: (88, K-1) float
        'Y_sim_clean': Y_sim_clean,     # Input 3: (100, N_clean, K-1) float
        'Y_sim_IND': Y_sim_IND,         # Input 4: (100, N_IND, K-1) float
        'GLM_params': GLM_params,       # Input 5: dict of (N_IND,) arrays for k and theta_i
        'results_df': clean_df
    }

def run_variance_deconvolution_and_ind_calibration(
    deconvolution_inputs: Dict[int, Dict[str, Any]],
    alr_blocs: List[str] = ['ALP', 'GRN', 'OTH'],
    oth_bloc_name: str = 'OTH',
    eig_floor: float = 1e-6,
    min_disp: float = 0.02,
    seat_filter: str = 'clean',  
    mu_threshold: float = 0.03   
) -> Dict[str, Any]:
    """
    Pools clean-seat and IND-seat inputs across all historical years in `deconvolution_inputs`
    to estimate:
      1. S_r_bar: Raw ensemble-mean residual covariance (Context 1 fixed-prior equivalent)
      2. Omega_bar_clean: Mean propagated prior covariance across clean seats
      3. Sigma_hat: Deconvolved, PSD-floored within-electorate ALR residual covariance
      4. lambda_star_macro: Scalar ALR variance retention ratio for IND seats
      5. precomp_ind_table: DataFrame of per-seat deflated (k_prime, theta_prime, lambda_i_star)
    """
    # -------------------------------------------------------------------------
    # STEP 0: Pool Clean Seats and IND Seats Across All Historical Years
    # -------------------------------------------------------------------------
    Y_obs_clean_list = []
    Y_sim_clean_list = []
    Y_sim_ind_list = []
    ind_meta_rows = []

    for yr, data in sorted(deconvolution_inputs.items()):
        mask_clean = np.asarray(data['is_clean_seat'], dtype=bool)
        obs_clean = data['Y_obs'][mask_clean]    # (N_clean_yr, D)
        sim_clean = data['Y_sim_clean']          # (S, N_clean_yr, D)
        obs_ind   = data['Y_obs'][~mask_clean]   # (N_IND_yr, D)
        sim_ind   = data['Y_sim_IND']            # (S, N_IND_yr, D)

        glm_p = data['GLM_params']
        mu_vec = np.asarray(glm_p['k'], dtype=float) * np.asarray(glm_p['theta'], dtype=float)

        # >>> CHANGED: Select seat subset based on seat_filter
        if seat_filter == 'clean':
            obs_sel, sim_sel = obs_clean, sim_clean
        elif seat_filter == 'all':
            obs_sel = np.concatenate([obs_clean, obs_ind], axis=0)
            sim_sel = np.concatenate([sim_clean, sim_ind], axis=1)
        elif seat_filter == 'ind':
            obs_sel, sim_sel = obs_ind, sim_ind
        elif seat_filter == 'low_ind':
            m = mu_vec < mu_threshold
            obs_sel, sim_sel = obs_ind[m], sim_ind[:, m, :]
        elif seat_filter == 'high_ind':
            m = mu_vec >= mu_threshold
            obs_sel, sim_sel = obs_ind[m], sim_ind[:, m, :]
        elif seat_filter == 'clean_plus_low_ind':
            m = mu_vec < mu_threshold
            obs_sel = np.concatenate([obs_clean, obs_ind[m]], axis=0)
            sim_sel = np.concatenate([sim_clean, sim_ind[:, m, :]], axis=1)
        else:
            raise ValueError(f"Unknown seat_filter: {seat_filter}")

        if len(obs_sel) > 0:
            Y_obs_clean_list.append(obs_sel)
            Y_sim_clean_list.append(sim_sel)
        # <<< END CHANGED

        Y_sim_ind_list.append(sim_ind)
        for d_name, k_val, th_val in zip(glm_p['divs'], glm_p['k'], glm_p['theta']):
            ind_meta_rows.append({
                'year': yr,
                'div_nm': d_name,
                'k_orig': float(k_val),
                'theta_orig': float(th_val),
                'mu_orig': float(k_val * th_val),
                'phi_orig': float(1.0 / k_val)
            })

    # Stacked clean arrays: Y_obs_clean is (N_clean_total, D); Y_sim_clean is (S, N_clean_total, D)
    Y_obs_clean = np.concatenate(Y_obs_clean_list, axis=0)
    Y_sim_clean = np.concatenate(Y_sim_clean_list, axis=1)
    S, N_clean_total, D = Y_sim_clean.shape

    # -------------------------------------------------------------------------
    # STEP 1: Post-Shift Ensemble Mean (m_bar_i) & Prior Covariance (Omega_hat_i)
    # Math:
    #   m_bar_i = (1/S) * sum_s Y_sim_clean[s, i, :]
    #   Omega_hat_i = (1/(S-1)) * sum_s (Y_sim_clean[s, i] - m_bar_i)(...)^T
    # -------------------------------------------------------------------------
    m_bar_clean = np.mean(Y_sim_clean, axis=0)  # Shape: (N_clean_total, D)
    centered_clean = Y_sim_clean - m_bar_clean[None, :, :]  # Shape: (S, N_clean_total, D)

    # Compute (D x D) covariance matrix for each of the N_clean_total seat-elections
    Omega_hat_all = np.einsum('snd,sne->nde', centered_clean, centered_clean) / (S - 1)
    Omega_bar_clean = np.mean(Omega_hat_all, axis=0)  # Shape: (D, D)

    # -------------------------------------------------------------------------
    # STEP 2: Ensemble-Mean Residual Covariance (S_r_bar)
    # Math:
    #   r_bar_i = y_obs_i - m_bar_i
    #   S_r_bar = (1 / N_clean_total) * sum_i (r_bar_i @ r_bar_i.T)
    # -------------------------------------------------------------------------
    r_bar = Y_obs_clean - m_bar_clean  # Shape: (N_clean_total, D)
    S_r_bar = (r_bar.T @ r_bar) / N_clean_total  # Shape: (D, D)

    # -------------------------------------------------------------------------
    # STEP 3: Spectral-Floored Moment Subtraction to Estimate Sigma_hat
    # Math:
    #   M = S_r_bar - (1 + 1/S) * Omega_bar_clean
    #   Sigma_hat = V @ max(Lambda, eig_floor) @ V.T
    # -------------------------------------------------------------------------
    finite_sim_factor = 1.0 + (1.0 / S)
    M_raw = S_r_bar - finite_sim_factor * Omega_bar_clean

    eigvals, eigvecs = np.linalg.eigh(M_raw)
    eigvals_clipped = np.maximum(eigvals, eig_floor)
    Sigma_hat = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # -------------------------------------------------------------------------
    # STEP 4: Estimate Macro lambda_star & Per-Seat Pre-Compensated Gamma Params
    # Math:
    #   V_bar_GLM_shift = Mean variance of Y_sim_IND on the OTH coordinate across S draws
    #   V_bar_shock     = Sigma_hat[oth_idx, oth_idx]
    #   lambda_star     = 1.0 - (V_bar_shock / V_bar_GLM_shift)
    # -------------------------------------------------------------------------
    oth_idx = alr_blocs.index(oth_bloc_name)
    V_bar_shock = float(Sigma_hat[oth_idx, oth_idx])

    Y_sim_ind_all = np.concatenate(Y_sim_ind_list, axis=1)  # Shape: (S, N_IND_total, D)
    oth_sim_draws = Y_sim_ind_all[:, :, oth_idx]            # Shape: (S, N_IND_total)
    per_seat_alr_var = np.var(oth_sim_draws, axis=0, ddof=1)
    V_bar_GLM_shift = float(np.mean(per_seat_alr_var))

    lambda_star_macro = 1.0 - (V_bar_shock / V_bar_GLM_shift) if V_bar_GLM_shift > 0 else np.nan

    # Compute closed-form proportion-space pre-compensation for each IND seat
    precomp_df = pd.DataFrame(ind_meta_rows)
    if not precomp_df.empty:
        mu_vec = precomp_df['mu_orig'].to_numpy()
        phi_vec = precomp_df['phi_orig'].to_numpy()

        # 1. Downstream relative variance (CV^2) injected in proportion space: eta_i
        eta_vec = np.exp(((1.0 - mu_vec) ** 2) * V_bar_shock) - 1.0

        # 2. Deflated Gamma dispersion (phi'_i) and shape (k'_i = 1 / phi'_i)
        phi_prime_vec = np.maximum(min_disp, (phi_vec - eta_vec) / (1.0 + eta_vec))
        k_prime_vec = 1.0 / phi_prime_vec

        # 3. Effective per-seat dispersion retention ratio: lambda*_i = phi'_i / phi_i
        lambda_seat_vec = phi_prime_vec / phi_vec

        # 4. Jensen's inequality convexity correction for mean (mu'_i) and scale (theta'_i)
        jensen_corr = np.exp(-0.5 * (1.0 - mu_vec) * (1.0 - 2.0 * mu_vec) * V_bar_shock)
        mu_prime_vec = mu_vec * jensen_corr
        theta_prime_vec = mu_prime_vec / k_prime_vec

        precomp_df['eta_i'] = eta_vec
        precomp_df['phi_prime'] = phi_prime_vec
        precomp_df['k_prime'] = k_prime_vec
        precomp_df['theta_prime'] = theta_prime_vec
        precomp_df['lambda_i_star'] = lambda_seat_vec

    return {
        'alr_blocs': alr_blocs,
        'N_clean_total': N_clean_total,
        'S_r_bar': pd.DataFrame(S_r_bar, index=alr_blocs, columns=alr_blocs),
        'Omega_bar_clean': pd.DataFrame(Omega_bar_clean, index=alr_blocs, columns=alr_blocs),
        'Sigma_hat': pd.DataFrame(Sigma_hat, index=alr_blocs, columns=alr_blocs),
        'raw_eigenvalues': eigvals,
        'V_bar_GLM_shift': V_bar_GLM_shift,
        'V_bar_shock': V_bar_shock,
        'lambda_star_macro': lambda_star_macro,
        'precomp_ind_table': precomp_df
    }


def compare_seat_tiers(deconvolution_inputs: dict, mu_threshold: float = 0.03):
    """Compares S_r_bar and Omega_bar across Clean, Low-IND (<3%), and High-IND (>=3%) seats."""
    tiers = {'Clean (No New IND)': [], f'Low-IND (<{mu_threshold*100:.0f}%)': [], f'High-IND (>={mu_threshold*100:.0f}%)': []}

    for yr, data in deconvolution_inputs.items():
        # Clean seats
        m_clean = np.mean(data['Y_sim_clean'], axis=0)
        r_clean = data['Y_obs'][data['is_clean_seat']] - m_clean
        omega_clean = np.var(data['Y_sim_clean'], axis=0, ddof=1)
        for i in range(len(r_clean)):
            tiers['Clean (No New IND)'].append((r_clean[i]**2, omega_clean[i]))

        # IND seats split by mu_orig = k * theta
        ind_obs = data['Y_obs'][~data['is_clean_seat']]
        m_ind = np.mean(data['Y_sim_IND'], axis=0)
        r_ind = ind_obs - m_ind
        omega_ind = np.var(data['Y_sim_IND'], axis=0, ddof=1)
        mu_vec = data['GLM_params']['k'] * data['GLM_params']['theta']

        for i, mu_val in enumerate(mu_vec):
            key = f'Low-IND (<{mu_threshold*100:.0f}%)' if mu_val < mu_threshold else f'High-IND (>={mu_threshold*100:.0f}%)'
            tiers[key].append((r_ind[i]**2, omega_ind[i]))

    rows = []
    for t_name, items in tiers.items():
        if not items: continue
        r2_mean = np.mean([x[0] for x in items], axis=0)
        om_mean = np.mean([x[1] for x in items], axis=0)
        rows.append({
            'Tier': t_name, 'N_seats': len(items),
            'Sr_ALP': r2_mean[0], 'Om_ALP': om_mean[0], 'Net_ALP': r2_mean[0] - 1.01*om_mean[0],
            'Sr_GRN': r2_mean[1], 'Om_GRN': om_mean[1], 'Net_GRN': r2_mean[1] - 1.01*om_mean[1],
            'Sr_OTH': r2_mean[2], 'Om_OTH': om_mean[2], 'Net_OTH': r2_mean[2] - 1.01*om_mean[2],
        })
    return pd.DataFrame(rows).set_index('Tier').round(4)




deconvolution_inputs = {}
for election_year in [2010,2014,2018,2022]:

    with open(f'pre_poll_sims_step6_for_{election_year}.pkl', 'rb') as f:
        pre_poll_sims_step6 = pickle.load(f)
    with open(f'ind_glm_for_{election_year}.pkl', 'rb') as f:
        global_ind_glm_meta = pickle.load(f)
    LA_results =pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Primary-Electorate-Formatted.csv')

    if election_year == 2022:
        LA_results = LA_results.loc[LA_results['div_nm'] != 'Narracan']

    deconvolution_inputs[election_year] = collect_deconvolution_inputs_for_year(
        target_year=election_year,
        pre_poll_sims_step6=pre_poll_sims_step6,
        ind_glm_meta_by_div=global_ind_glm_meta,
        la_results_df = LA_results,
    )


# 1. Run Pooled Estimation across [2006, 2010, 2014, 2018]
pooled_res = run_variance_deconvolution_and_ind_calibration(
    deconvolution_inputs=deconvolution_inputs,
    alr_blocs=['ALP', 'GRN', 'OTH'],
    oth_bloc_name='OTH',
    seat_filter = 'all', 
    mu_threshold = 0.03 
)

print("=== 1. POOLED MATRICES (ALL 4 ELECTIONS, N_clean =", pooled_res['N_clean_total'], ") ===")
print("\nRaw Residual Covariance S_r_bar (Fixed-Prior Equivalent):")
print(pooled_res['S_r_bar'].round(4))

print("\nPropagated Prior Covariance Omega_bar_clean (Prior Uncertainty):")
print(pooled_res['Omega_bar_clean'].round(4))

print("\nDeconvolved True Residual Covariance Sigma_hat:")
print(pooled_res['Sigma_hat'].round(4))

# Percentage reduction on diagonals from removing prior uncertainty
diag_reduction_pct = (
    1.0 - np.diag(pooled_res['Sigma_hat']) / np.diag(pooled_res['S_r_bar'])
) * 100
print("\nDiagonal Variance Reduction (%):", dict(zip(['ALP', 'GRN', 'OTH'], np.round(diag_reduction_pct, 1))))
print("Raw Eigenvalues of M (check all > 0):", np.round(pooled_res['raw_eigenvalues'], 5))
print(f"Macro IND Retention Ratio (lambda*): {pooled_res['lambda_star_macro']:.3f}")

# 2. Year-by-Year Comparison
print("\n=== 2. YEAR-BY-YEAR DIAGONAL COMPARISON ===")
yr_rows = []
for yr in sorted(deconvolution_inputs.keys()):
    yr_res = run_variance_deconvolution_and_ind_calibration({yr: deconvolution_inputs[yr]})
    yr_rows.append({
        'year': yr,
        'N_clean': yr_res['N_clean_total'],
        'S_r_ALP': yr_res['S_r_bar'].loc['ALP', 'ALP'],
        'Sigma_ALP': yr_res['Sigma_hat'].loc['ALP', 'ALP'],
        'S_r_GRN': yr_res['S_r_bar'].loc['GRN', 'GRN'],
        'Sigma_GRN': yr_res['Sigma_hat'].loc['GRN', 'GRN'],
        'S_r_OTH': yr_res['S_r_bar'].loc['OTH', 'OTH'],
        'Sigma_OTH': yr_res['Sigma_hat'].loc['OTH', 'OTH'],
        'lambda_star': yr_res['lambda_star_macro']
    })
print(pd.DataFrame(yr_rows).set_index('year').round(4))

for filt in ['clean', 'all', 'clean_plus_low_ind', 'ind']:
    pooled_res = run_variance_deconvolution_and_ind_calibration(
        deconvolution_inputs=deconvolution_inputs,
        seat_filter=filt
    )
    print(f"\n================ FILTER: {filt.upper()} (N = {pooled_res['N_clean_total']}) ================")
    print("Deconvolved Sigma_hat (Pooled):")
    print(pooled_res['Sigma_hat'].round(4))

    yr_rows = []
    for yr in sorted(deconvolution_inputs.keys()):
        yr_res = run_variance_deconvolution_and_ind_calibration(
            {yr: deconvolution_inputs[yr]},
            seat_filter=filt
        )
        yr_rows.append({
            'year': yr,
            'N_seats': yr_res['N_clean_total'],
            'S_r_ALP': yr_res['S_r_bar'].loc['ALP', 'ALP'],
            'Sigma_ALP': yr_res['Sigma_hat'].loc['ALP', 'ALP'],
            'S_r_GRN': yr_res['S_r_bar'].loc['GRN', 'GRN'],
            'Sigma_GRN': yr_res['Sigma_hat'].loc['GRN', 'GRN'],
            'S_r_OTH': yr_res['S_r_bar'].loc['OTH', 'OTH'],
            'Sigma_OTH': yr_res['Sigma_hat'].loc['OTH', 'OTH'],
        })
    print(pd.DataFrame(yr_rows).set_index('year').round(4))

print(compare_seat_tiers(deconvolution_inputs, mu_threshold=0.03))


import pdb; pdb.set_trace()


def build_production_sigma_2026(
    res_pooled: dict, 
    res_2022: dict, 
    w_realign: float = 0.60
) -> pd.DataFrame:
    """
    Constructs the 4x4 ALR residual covariance matrix for the 2026 forecast.
    Blends the high-volatility 2022 regime with the 4-year pooled baseline.
    """
    # 1. Extract base variances from deconvolved Victorian matrices
    v_alp_pooled = res_pooled['Sigma_hat'].loc['ALP', 'ALP']
    v_alp_2022   = res_2022['Sigma_hat'].loc['ALP', 'ALP']
    
    v_grn_pooled = res_pooled['Sigma_hat'].loc['GRN', 'GRN']
    v_grn_2022   = res_2022['Sigma_hat'].loc['GRN', 'GRN']
    
    # 2. Blend ALP/GRN, Anchor OTH, and synthesize ON
    var_alp = (w_realign * v_alp_2022) + ((1.0 - w_realign) * v_alp_pooled)
    var_grn = (w_realign * v_grn_2022) + ((1.0 - w_realign) * v_grn_pooled)
    var_oth = res_pooled['Sigma_hat'].loc['OTH', 'OTH']  # Anchored to 0.1230
    var_on  = 0.1538 * 0.78  # 0.12 Informed by Federal (0.1538) & SA 2026 (0.1380) minus prior var

    # Standard deviations for the diagonal matrix
    stds = np.sqrt(np.array([var_alp, var_grn, var_on, var_oth], dtype=np.float64))

    # 3. Construct the Correlation Matrix (R)
    # Order: ['ALP', 'GRN', 'ON', 'OTH']
    R = np.array([
        [1.00,  0.42,  0.18,  0.22],  # ALP
        [0.42,  1.00,  0.22,  0.30],  # GRN
        [0.18,  0.22,  1.00,  0.10],  # ON
        [0.22,  0.30,  0.10,  1.00]   # OTH
    ], dtype=np.float64)

    # 4. Reconstruct Covariance: Sigma = diag(S) * R * diag(S)
    Sigma_2026 = np.outer(stds, stds) * R
    
    blocs = ['ALP', 'GRN', 'ON', 'OTH']
    return pd.DataFrame(Sigma_2026, index=blocs, columns=blocs)

def get_historical_backtest_sigma(
    target_year: int, 
    deconvolution_inputs: dict, 
    use_trailing_pool: bool = True
) -> pd.DataFrame:
    """
    Returns the appropriate 3x3 (ALP, GRN, OTH) covariance matrix for historical backtesting.
    
    If use_trailing_pool=True, it builds a matrix using ONLY data available up to that year 
    (to prevent forward data leakage). If False, it just uses that specific year's Sigma.
    """
    
    if use_trailing_pool:
        # Build a temporary dict containing only the years <= target_year
        valid_inputs = {yr: data for yr, data in deconvolution_inputs.items() if yr <= target_year}
        # If we are at 2010 and have no prior data, just use 2010
        if not valid_inputs:
            valid_inputs = {target_year: deconvolution_inputs[target_year]}
            
        res = run_variance_deconvolution_and_ind_calibration(
            deconvolution_inputs=valid_inputs,
            alr_blocs=['ALP', 'GRN', 'OTH'],
            oth_bloc_name='OTH',
            seat_filter = 'all', 
            mu_threshold = 0.03 
        )
        return res['Sigma_hat']
    
    else:
        # Just use the single-year matrix (less stable, but purely localized to that year)
        res = run_variance_deconvolution_and_ind_calibration(
            {target_year: deconvolution_inputs[target_year]},
            alr_blocs=['ALP', 'GRN', 'OTH'],
            oth_bloc_name='OTH',
            seat_filter = 'all', 
            mu_threshold = 0.03 
        )
        return res['Sigma_hat']
    


# A. GENERATE THE 2026 PRODUCTION MATRIX (4x4: ALP, GRN, ON, OTH)
# =========================================================================
# 1. Run deconvolution across all 4 historical elections (N_clean = 142)
res_pooled = run_variance_deconvolution_and_ind_calibration(
    deconvolution_inputs=deconvolution_inputs,
    alr_blocs=['ALP', 'GRN', 'OTH'],
    oth_bloc_name='OTH'
)

# 2. Run deconvolution on the 2022 realignment election alone (N_clean = 26)
res_2022 = run_variance_deconvolution_and_ind_calibration(
    deconvolution_inputs={2022: deconvolution_inputs[2022]},
    alr_blocs=['ALP', 'GRN', 'OTH'],
    oth_bloc_name='OTH'
)

# 3. Build the 4x4 production matrix for 2026 (60% 2022 realignment + 40% pooled)
Sigma_residual_2026 = build_production_sigma_2026(
    res_pooled=res_pooled,
    res_2022=res_2022,
    w_realign=0.60
)
print("Production 2026 Sigma_residual (4x4):")
print(Sigma_residual_2026.round(4))


# =========================================================================
# B. GENERATE HISTORICAL BACKTEST MATRICES FOR [2010, 2014, 2018, 2022]
# =========================================================================
def get_historical_backtest_sigma(
    target_year: int,
    deconvolution_inputs: dict,
    mode: str = 'leave_one_out'  # 'leave_one_out', 'pooled', or 'single_year'
) -> pd.DataFrame:
    """
    Returns the 3x3 (ALP, GRN, OTH) Sigma_hat matrix for a historical backtest year.
      - 'leave_one_out': Pools the 3 OTHER elections (~100-116 clean seats) for true out-of-sample testing.
      - 'pooled': Uses all 4 elections (142 clean seats).
      - 'single_year': Uses only `target_year` (~26-42 clean seats).
    """
    if mode == 'leave_one_out':
        sub_inputs = {yr: d for yr, d in deconvolution_inputs.items() if yr != target_year}
    elif mode == 'single_year':
        sub_inputs = {target_year: deconvolution_inputs[target_year]}
    else:
        sub_inputs = deconvolution_inputs

    res = run_variance_deconvolution_and_ind_calibration(
        deconvolution_inputs=sub_inputs,
        alr_blocs=['ALP', 'GRN', 'OTH'],
        oth_bloc_name='OTH'
    )
    return res['Sigma_hat']

# Dictionary of 3x3 covariance matrices to feed into Step 8 of each backtest year
backtest_sigmas_by_year = {
    yr: get_historical_backtest_sigma(yr, deconvolution_inputs, mode='leave_one_out')
    for yr in [2010, 2014, 2018, 2022]
}

import pdb; pdb.set_trace()