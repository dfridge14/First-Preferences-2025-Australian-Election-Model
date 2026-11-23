

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import re

import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb) 
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()

sys.excepthook = exception_handler


YEARS = [2010, 2014, 2018, 2022]
THETAS = [0.0, 0.5, 0.75, 1.0]
EXP_DIR = Path.home() /"Australian Election/Victorian Election/"
COAL_SET = {'LP', 'NP', 'LNP', 'COAL'}

META_COLS = {'div_nm', 'CandRemaining', 'TOTAL', 'CountType'}

os.chdir(Path.home() /"Australian Election/Victorian Election/")


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



def load_actual_c0(year: int) -> tuple[dict, dict]:
    """
    Loads actual election-night Count 0 shares (0..1) and formal-vote seat weights
    directly from VIC-{year}/{year}_complete_DOP_table_dict.pkl.
    """
    with open(f"VIC-{year}/{year}_complete_DOP_table_dict.pkl", "rb") as f:
        dop_next = pickle.load(f)
    if year == 2022 and "Narracan" in dop_next:
        del dop_next["Narracan"]

    la_cands_next = pd.read_csv(f"VIC-{year}/VIC-{year}-LA-Candidates-Formatted.csv")
    dop_next = sync_dop_table_inds(dop_next, la_cands_next)

    actual_c0 = {}
    div_weights = {}
    for div, raw_df in dop_next.items():
        if year == 2022 and div == "Narracan":
            continue
        c0_row = raw_df.iloc[0]
        raw_votes = {
            c: float(c0_row[c])
            for c in raw_df.columns
            if c not in META_COLS and pd.notna(c0_row[c]) and float(c0_row[c]) > 0
        }
        tot = sum(raw_votes.values())
        if tot > 0:
            actual_c0[div] = {c: v / tot for c, v in raw_votes.items()}
            div_weights[div] = tot

    return actual_c0, div_weights


def _to_5bloc(code: str) -> str:
    c = str(code).strip()
    if c in COAL_SET:
        return 'COAL'
    return c if c in ('ALP', 'GRN', 'ON') else 'OTH'


def sims_list_to_seat_means(fp_list: list) -> dict:
    """Converts saved FP_results_c0 (list of {div: {cand: share}}) -> {div: mean Series}."""
    by_div = {}
    for sim_dict in fp_list:
        for div, cands in sim_dict.items():
            by_div.setdefault(div, []).append(cands)
    return {div: pd.DataFrame(rows).fillna(0.0).mean() for div, rows in by_div.items()}


def extract_inc_lookups(inc_next_lookup: dict, ia_tracker_dict: dict = None) -> tuple[dict, dict]:
    """Extracts {div: set(recontesting_inc_codes)} and {div: retiring_party} from your saved pickles."""
    recontesting_inc, retiring_party = {}, {}
    for div, inc_list in inc_next_lookup.items():
        if not inc_list or inc_list[0].get('is_open', True):
            if ia_tracker_dict and div in ia_tracker_dict and ia_tracker_dict[div]:
                retiring_party[div] = str(ia_tracker_dict[div][0].get('inc_party', '')).upper().strip()
            continue

        codes = set()
        for inc in inc_list:
            p = str(inc.get('party', '')).upper().strip()
            if p.startswith('IND'):
                cname = str(inc.get('inc_clean_name', '')).strip().title().replace(' ', '_')
                if cname:
                    codes.add(f"IND_{cname}")
            elif p:
                codes.add(p)
        recontesting_inc[div] = codes
    return recontesting_inc, retiring_party


def evaluate_theta_backtest(
    seat_means: dict,
    actual_c0: dict,
    recontesting_inc: dict,
    retiring_party: dict,
    div_weights: dict = None,
    year: int = None,
    theta: float = 0.0,
) -> dict:
    districts = sorted(set(seat_means.keys()) & set(actual_c0.keys()))
    if not districts:
        raise ValueError(f"No overlapping districts for year={year}, theta={theta}")

    if div_weights:
        avail = [float(div_weights[d]) for d in districts if d in div_weights]
        fallback = float(np.mean(avail)) if avail else 1.0
        raw_w = {d: float(div_weights.get(d, fallback)) for d in districts}
    else:
        raw_w = {d: 1.0 for d in districts}
    w_tot = sum(raw_w.values())
    w = {d: raw_w[d] / w_tot for d in districts}

    sw_pred = {b: 0.0 for b in ('COAL', 'ALP', 'GRN', 'ON', 'OTH')}
    sw_act  = {b: 0.0 for b in ('COAL', 'ALP', 'GRN', 'ON', 'OTH')}
    all_err, inc_err, open_err, chal_err, oth_err = [], [], [], [], []

    for d in districts:
        pred_s = pd.Series(seat_means[d], dtype=float)
        act_s = pd.Series(actual_c0[d], dtype=float)
        if pred_s.sum() > 1.5: pred_s = pred_s / 100.0
        if act_s.sum() > 1.5:  act_s = act_s / 100.0

        for c, v in pred_s.items(): sw_pred[_to_5bloc(c)] += w[d] * float(v) * 100.0
        for c, v in act_s.items():  sw_act[_to_5bloc(c)]  += w[d] * float(v) * 100.0

        pred_oth = sum(float(v) for c, v in pred_s.items() if _to_5bloc(c) == 'OTH') * 100.0
        act_oth  = sum(float(v) for c, v in act_s.items()  if _to_5bloc(c) == 'OTH') * 100.0
        oth_err.append(pred_oth - act_oth)

        inc_set = recontesting_inc.get(d, set())
        ret_p = retiring_party.get(d)

        for c in set(pred_s.index) | set(act_s.index):
            p_val = float(pred_s.get(c, 0.0)) * 100.0
            a_val = float(act_s.get(c, 0.0)) * 100.0
            if max(p_val, a_val) < 0.1:
                continue
            err = p_val - a_val
            all_err.append(err)

            if c in inc_set:
                inc_err.append(err)
            elif ret_p and c == ret_p:
                open_err.append(err)
            else:
                chal_err.append(err)

    def _stats(arr):
        if not arr:
            return np.nan, np.nan
        a = np.asarray(arr, dtype=float)
        return round(float(a.mean()), 3), round(float(np.abs(a).mean()), 3)

    _, all_mae = _stats(all_err)
    inc_bias, inc_mae = _stats(inc_err)
    open_bias, open_mae = _stats(open_err)
    chal_bias, _ = _stats(chal_err)
    oth_bias, oth_mae = _stats(oth_err)

    row = {
        'Year': year, 'Theta': theta,
        'All_MAE': all_mae,
        'All_RMSE': round(float(np.sqrt(np.mean(np.square(all_err)))), 3),
        'Inc_Bias': inc_bias, 'Inc_MAE': inc_mae,
        'Open_Bias': open_bias, 'Open_MAE': open_mae,
        'Chal_Bias': chal_bias,
        'OTH_Bias': oth_bias, 'OTH_MAE': oth_mae,
    }
    for b in ('COAL', 'ALP', 'GRN', 'ON', 'OTH'):
        row[f'SW_{b}'] = round(sw_pred[b] - sw_act[b], 2)
    return row


if __name__ == "__main__":
    results = []

    for year in YEARS:
        actual_c0, div_weights = load_actual_c0(year)

        with open(f"inc_next_lookup_for_{year}.pkl", "rb") as f:
            inc_next_lookup = pickle.load(f)
        ia_path = f"ia_tracker_dict_for_{year}.pkl"
        ia_tracker = pickle.load(open(ia_path, "rb")) if os.path.exists(ia_path) else {}
        recontesting_inc, retiring_party = extract_inc_lookups(inc_next_lookup, ia_tracker)

        for theta in THETAS:
            tag = f"{theta:.2f}".replace(".", "p")
            fp_path = os.path.join(EXP_DIR, f"ia_theta_experiments/FP_pred_for_{year}_theta_{tag}.pkl")
            if not os.path.exists(fp_path):
                print(f"[SKIP] Missing {fp_path}")
                continue

            with open(fp_path, "rb") as f:
                fp_list = pickle.load(f)

            seat_means = sims_list_to_seat_means(fp_list)
            results.append(evaluate_theta_backtest(
                seat_means=seat_means,
                actual_c0=actual_c0,
                recontesting_inc=recontesting_inc,
                retiring_party=retiring_party,
                div_weights=div_weights,
                year=year,
                theta=theta,
            ))

    df = pd.DataFrame(results)
    print("\n=== PER-YEAR RESULTS ===")
    print(df.to_string(index=False))

    print("\n=== 4-YEAR POOLED AVERAGE BY THETA ===")
    pooled = df.drop(columns=['Year']).groupby('Theta').mean().round(3)
    print(pooled.to_string())