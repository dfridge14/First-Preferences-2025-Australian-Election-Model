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
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
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


from collections import Counter
import numpy as np
import pandas as pd

os.chdir(Path.home() / 'Australian Election/Victorian Election')

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

def true_coverage(
    per_electorate_winners: dict,
    actual_winners_dict: dict,
    coverage_intervals: list = [50, 60, 70, 80, 90, 95, 99]
) -> pd.DataFrame:
    """
    Computes cumulative coverage for the modal favorite at each confidence threshold,
    returning seat counts, mean predicted confidence, and observed win rate.
    """
    results = []
    for electorate, sims in per_electorate_winners.items():
        if electorate not in actual_winners_dict or not sims:
            continue
        n_sims = len(sims)
        counts = Counter(sims)
        fav_party, count = counts.most_common(1)[0]
        conf = count / n_sims
        correct = float(fav_party == actual_winners_dict[electorate])
        results.append((conf, correct))

    rows = []
    for cov in coverage_intervals:
        thresh = cov / 100.0
        filtered = [(conf, corr) for conf, corr in results if conf >= thresh]
        if filtered:
            n_seats = len(filtered)
            mean_conf = np.mean([c for c, _ in filtered])
            obs_rate  = np.mean([corr for _, corr in filtered])
        else:
            n_seats, mean_conf, obs_rate = 0, np.nan, np.nan

        rows.append({
            'threshold_pct': f">={cov}%",
            'n_seats': n_seats,
            'mean_pred_prob': mean_conf,
            'observed_win_rate': obs_rate,
            'obs_minus_pred': obs_rate - mean_conf if n_seats > 0 else np.nan
        })

    return pd.DataFrame(rows).set_index('threshold_pct')


def calibration_curve(
    per_electorate_winners: dict,
    actual_winners_dict: dict,
    bins: list = None,
    include_underdogs: bool = False
) -> Tuple[pd.DataFrame, float]:
    """
    Computes bucketed probability calibration and multi-class Brier score.
    
    Parameters:
      - include_underdogs=False: Evaluates only the modal favorite per seat (bins from 40% to 100%).
      - include_underdogs=True: Evaluates ALL contenders with >0 simulated wins (bins from 1% to 100%).
    """
    if bins is None:
        if include_underdogs:
            bins = [(1, 10), (10, 25), (25, 40), (40, 60), (60, 75), (75, 90), (90, 99), (99, 100)]
        else:
            bins = [(40, 60), (60, 70), (70, 80), (80, 90), (90, 95), (95, 99), (99, 100)]

    results = []
    brier_seat_scores = []

    for electorate, sims in per_electorate_winners.items():
        if electorate not in actual_winners_dict or not sims:
            continue
        n_sims = len(sims)
        counts = Counter(sims)
        actual = actual_winners_dict[electorate]

        # Multi-class Brier score for this seat: sum_p (prob_p - I(p == actual))^2
        all_parties = set(counts.keys()) | {actual}
        seat_brier = sum(((counts.get(p, 0) / n_sims) - float(p == actual)) ** 2 for p in all_parties)
        brier_seat_scores.append(seat_brier)

        if include_underdogs:
            for party, count in counts.items():
                prob = count / n_sims
                results.append((prob, float(party == actual)))
        else:
            fav_party, count = counts.most_common(1)[0]
            prob = count / n_sims
            results.append((prob, float(fav_party == actual)))

    rows = []
    for low, high in bins:
        low_p, high_p = low / 100.0, high / 100.0
        # Fix boundary bug: include p == 1.0 when high == 100
        if high == 100:
            selected = [(p, c) for (p, c) in results if low_p <= p <= high_p]
        else:
            selected = [(p, c) for (p, c) in results if low_p <= p < high_p]

        if selected:
            n_obs = len(selected)
            mean_pred = np.mean([p for p, _ in selected])
            obs_rate  = np.mean([c for _, c in selected])
        else:
            n_obs, mean_pred, obs_rate = 0, np.nan, np.nan

        rows.append({
            'bin': f"{low}-{high}%",
            'n_cases': n_obs,
            'mean_pred_prob': mean_pred,
            'observed_win_rate': obs_rate,
            'obs_minus_pred': obs_rate - mean_pred if n_obs > 0 else np.nan
        })

    mean_brier = float(np.mean(brier_seat_scores)) if brier_seat_scores else np.nan
    return pd.DataFrame(rows).set_index('bin'), mean_brier







for election_year in [2010,2014,2018,2022]:

    LA_Candidates_next = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
    if election_year == 2022:
        LA_Candidates_next =  LA_Candidates_next.loc[~(LA_Candidates_next['div_nm']=='Narracan'),] # remove Narracan due to no election
    LA_Candidates_next['PartyAb'] = LA_Candidates_next.apply(format_specific_party, axis=1)
        
    div_col = 'div_nm' if 'div_nm' in LA_Candidates_next.columns else 'Electorate'

    elected_df = LA_Candidates_next.loc[LA_Candidates_next['elected'] == 1]

    actual_winners_dict = dict(
        zip(
            elected_df[div_col].astype(str).str.strip(),
            elected_df['PartyAb'].astype(str).str.strip()
        )
    )

    # Sanity check: should be 88 electorates (or 87 in 2022 if Narracan supplementary is separate)
    print(f"Loaded {len(actual_winners_dict)} actual electorate winners.")


    with open(Path.home() / f"Australian Election/Victorian Election/Past-simulations/{election_year}_simulation_bundle.pkl", "rb") as f:
        sim_bundle = pickle.load(f)

        per_electorate_winners = sim_bundle['electorate_winners']

    calib_df, brier = calibration_curve(per_electorate_winners, actual_winners_dict, bins=None, include_underdogs=True)
    print(election_year)
    print(calib_df)
    print(brier)


pooled_sim_winners = {}
pooled_actual_winners = {}
yearly_calib_dfs = {}
yearly_briers = {}

for election_year in [ 2022, 2018, 2014, 2010]:
    LA_Candidates_next = pd.read_csv(f'VIC-{election_year}/VIC-{election_year}-LA-Candidates-Formatted.csv')
    if election_year == 2022:
        LA_Candidates_next = LA_Candidates_next.loc[~(LA_Candidates_next['div_nm'] == 'Narracan'),]
    LA_Candidates_next['PartyAb'] = LA_Candidates_next.apply(format_specific_party, axis=1)
        
    div_col = 'div_nm' if 'div_nm' in LA_Candidates_next.columns else 'Electorate'
    elected_df = LA_Candidates_next.loc[LA_Candidates_next['elected'] == 1]

    actual_winners_dict = dict(
        zip(
            elected_df[div_col].astype(str).str.strip(),
            elected_df['PartyAb'].astype(str).str.strip()
        )
    )

    with open(Path.home() / f"Australian Election/Victorian Election/Past-simulations/{election_year}_simulation_bundle.pkl", "rb") as f:
        sim_bundle = pickle.load(f)
        per_electorate_winners = sim_bundle['electorate_winners']

    # 1. Per-year calibration (optional, for side-by-side comparison)
    calib_df_yr, brier_yr = calibration_curve(
        per_electorate_winners, actual_winners_dict, bins=None, include_underdogs=True
    )
    yearly_calib_dfs[election_year] = calib_df_yr
    yearly_briers[election_year] = (len(actual_winners_dict), brier_yr)

    # 2. Accumulate into pooled dictionaries using "{year}_{div}" keys so seats don't overwrite
    for div, winner in actual_winners_dict.items():
        if div in per_electorate_winners:
            pooled_key = f"{election_year}_{div}"
            pooled_actual_winners[pooled_key] = winner
            pooled_sim_winners[pooled_key] = per_electorate_winners[div]

# --- A. SINGLE POOLED CALIBRATION TABLE (ALL 4 ELECTIONS, N = 351 SEATS) ---
pooled_calib_df, pooled_brier = calibration_curve(
    pooled_sim_winners, pooled_actual_winners, bins=None, include_underdogs=True
)

print(f"\n=== POOLED CALIBRATION CURVE (2010-2022, N = {len(pooled_actual_winners)} seats) ===")
print(pooled_calib_df.round(4))
print(f"Pooled Multi-Class Brier Score: {pooled_brier:.4f}")

# --- B. COMBINED SIDE-BY-SIDE + POOLED SUMMARY TABLE ---
combined_table = pd.concat(
    {**yearly_calib_dfs, 'Pooled (All 4)': pooled_calib_df},
    axis=1
)
print("\n=== SIDE-BY-SIDE & POOLED CALIBRATION TABLE ===")
print(combined_table.round(3))

brier_summary = pd.DataFrame(
    [{'Year': yr, 'N_seats': n_s, 'Brier': br} for yr, (n_s, br) in yearly_briers.items()]
    + [{'Year': 'Pooled', 'N_seats': len(pooled_actual_winners), 'Brier': pooled_brier}]
).set_index('Year')
print("\n=== BRIER SCORE SUMMARY ===")
print(brier_summary.round(4))




# 90%+ predicted probability but lost
lost_90 = []
# 1-10% predicted probability but won
won_1_10 = []

for seat, sims in pooled_sim_winners.items():
    n_sims = len(sims)
    counts = Counter(sims)
    actual = pooled_actual_winners[seat]

    for party, count in counts.items():
        prob = count / n_sims

        if prob >= 0.90 and party != actual:
            lost_90.append((seat, party, prob, actual))

        elif 0.01 <= prob < 0.10 and party == actual:
            won_1_10.append((seat, party, prob))

print("\n=== 90%+ PREDICTED BUT LOST ===")
for seat, party, prob, actual in sorted(lost_90, key=lambda x: -x[2]):
    print(f"{seat}: {party} ({prob:.1%}) lost to {actual}")

print("\n=== 1-10% PREDICTED BUT WON ===")
for seat, party, prob in sorted(won_1_10, key=lambda x: x[2]):
    print(f"{seat}: {party} ({prob:.1%})")




party_name_dict = {
    'ALP': 'Labor', 'LP': 'Liberal', 'NP': 'National', 'COAL': 'Coalition',
    'LNP': 'Liberal National', 'GRN': 'Greens', 'VNS': 'Victorian Socialists',
    'VSOC': 'Victorian Socialists', 'HMP': 'Legalise Cannabis', 'LCV': 'Legalise Cannabis',
    'AJP': 'Animal Justice', 'FFP': 'Family First', 'FFPA': 'Family First',
    'LDP': 'Libertarian', 'LTP': 'Libertarian', 'SFF': 'Shooters F.& F.',
    'ASP': 'Shooters F.& F.', 'AUD': 'Australian Democrats',
    'ON': 'One Nation', 'TOP': 'Trumpet of Patriots', 'CYA': 'Trumpet of Patriots',
    'FUSN': 'FUSION', 'SOPA': 'FUSION', 'SAP': 'Sustainable Australia',
    'SPP': 'Sustainable Australia', 'CIT': 'Citizens Party', 'CEC': 'Citizens Party',
    'IND': 'Independent', 'OTH': 'Other', 'DLP': 'Democratic Labour', 'SAL': 'Socialist Allicance', 
    'FPAL': 'Free Palestine', 'TWP': 'West Party', 'MAGA': 'Republican', 'CAN': 'Climate Action', 'ATF':'Avoid the Fine',
    'AVP': 'Angry Victorians', 'DHJP': 'Justice Party', 'HAP': 'Health Australia', 'REAS': 'Reason', # 2022 
    'SDA' : 'Sack Dan Andrews', 'TMP': 'Transport Matters','UAPP':'United Australia', 
    'ND' : 'New Democrats', 'CAP' : 'Companions and Pets', 'FPV': 'Freedom Party'
}

site_abbreviation_dict = {
    'Labor': 'LAB', 'Liberal': 'LIB', 'National': 'NAT', 'Coalition': 'COAL',
    'Greens': 'GRN', 'One Nation': 'ON', 'Legalise Cannabis': 'LCV',
    'Family First': 'FFP', 'Shooters F.& F.': 'SFF',
    'Trumpet of Patriots': 'TOP', 'Independent': 'IND', 'Animal Justice': 'AJP',
    'FUSION': 'FUSN', 'Libertarian': 'LDP', 'Australian Democrats': 'AUD',
    'Citizens Party': 'CIT', 'Victorian Socialists': 'VSOC',
    'Sustainable Australia': 'SAP', 'Other': 'OTH','West Party': 'WEST',
    'Democratic Labour': 'DLP', 'Socialist Allicance': 'SAL', 'Free Palestine':'FPAL',  
    'Republican': 'MAGA', 'Climate Action': 'CA', 'Avoid the Fine':'ATF',
    'Angry Victorians': 'AVP','Justice Party': 'DHJP','Health Australia': 'HAP', 'Reason': 'REAS',
    'Sack Dan Andrews': 'SDA','Transport Matters': 'TMP','United Australia':'UAPP',
    'Companions and Pets': 'CAP','New Democrats': 'ND', 'Freedom Party': 'FPV'
}

party_colors = {
    'Labor': '#F04143', 'Liberal': '#4053B2', 'Liberal National': '#4053B2',
    'Greens': '#6BBE6C', 'One Nation': '#EB8846', 'Legalise Cannabis': '#91D06A',
    'Libertarian': '#F4D04C', 'Trumpet of Patriots': '#FAEE76',
    'Animal Justice': '#764E9A', 'Family First': '#9B86A8', "People's First": '#5679A8',
    'Independent': '#999999', 'National': '#057e0d', 'Coalition': '#377eb8',
    'Teal Independent': '#1ed6db', 'Australian Democrats': '#f6ca06',
    'FUSION': '#cd18fc', 'Shooters F.& F.': '#f3144d',
    'Citizens Party': '#496332', 'Victorian Socialists': '#070706',
    'Socialist Allicance':'#91120d', 'West Party': '#131c13', 'Democratic Labour': '#7a6733',
    'Sustainable Australia': '#65A868', 'Justice Party': '#002f5d','Freedom Party': '#9e0000','Other': '#7F8C8D'
}

def make_slug(district_name):
    return str(district_name).strip().lower().replace(" ", "_").replace("'", "")

def format_party_name(raw_party):
    raw_str = str(raw_party).strip()
    if raw_str.startswith('IND_'):
        parts = raw_str.split('_')
        return f"Ind. {parts[-1]}" if len(parts) > 1 else "Independent"
    return party_name_dict.get(raw_str, raw_str)

def get_abbreviation(display_name, raw_party):
    if display_name.startswith('Ind. '):
        return display_name
    return site_abbreviation_dict.get(display_name, str(raw_party)[:4].upper())

def get_color(display_name, raw_party=""):
    if display_name.startswith('Ind.'):
        return party_colors.get(display_name, '#999999')
    return party_colors.get(display_name, party_colors.get(raw_party, '#7F8C8D'))


def parse_independent(col_name):
    """ Converts 'IND_Zoe_Daniel' or 'IND_Daniel' to 'Ind. Daniel' """
    parts = str(col_name).strip().split('_')
    last_name = parts[-1] if len(parts) > 1 else 'Unknown'
    display_name = f"Ind. {last_name}"
    abbr = f"IND {last_name[:3].title()}."
    return display_name, abbr

import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def generate_win_bar_json(winners_data, district_name, output_dir="Past-simulations"):
    os.makedirs(output_dir, exist_ok=True)

    # Convert simulation winners list/Series/dict into sorted Win Percentage DataFrame
    if isinstance(winners_data, pd.DataFrame) and 'Win Percentage' in winners_data.columns:
        df = winners_data.copy()
    elif isinstance(winners_data, dict):
        s = pd.Series(winners_data, dtype=float)
        if s.sum() > 1.05:
            s = s / s.sum()
        df = s.reset_index()
        df.columns = ['Party', 'Win Percentage']
    else:
        s = pd.Series(list(winners_data)).value_counts(normalize=True)
        df = s.reset_index()
        df.columns = ['Party', 'Win Percentage']

    df = df[df['Win Percentage'] > 0.001].sort_values('Win Percentage', ascending=False)

    fig = go.Figure()
    for _, row in df.iterrows():
        raw_p = str(row['Party'])
        party = format_party_name(raw_p)
        pct = float(row['Win Percentage'])
        color = get_color(party, raw_p)

        fig.add_trace(go.Bar(
            y=[district_name],
            x=[pct],
            name=party,
            orientation='h',
            marker=dict(color=color, line=dict(color='white', width=2)),
            hovertemplate=f"<b>{party}</b>: {pct:.1%}<extra></extra>",
            text=f"{party} {pct:.0%}" if pct >= 0.06 else "",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=15, family="Poppins, sans-serif", weight=600),
            width=0.65
        ))

    fig.update_layout(
        barmode='stack',
        height=95,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1], fixedrange=True),
        yaxis=dict(showticklabels=False, fixedrange=True),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    slug = make_slug(district_name)
    with open(os.path.join(output_dir, f"{slug}_bar.json"), "w", encoding="utf-8") as f:
        f.write(fig.to_json())

#

def generate_fp_violin_json(df_raw, district_name, output_dir="Past-simulations", preview=False):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    df_plot = df_raw.copy()

    if len(df_plot) > 1000:
        df_plot = df_plot.sample(n=1000, random_state=42)

    clean_cols = []
    display_to_abbr = {}
    for c in df_plot.columns:
        c_str = str(c).strip()
        if c_str.startswith('IND_'):
            display_name, abbr = parse_independent(c_str)
            clean_cols.append(display_name)
            display_to_abbr[display_name] = abbr
        elif c_str == 'IND':
            clean_cols.append('Independent')
            display_to_abbr['Independent'] = 'IND'
        else:
            full_name = party_name_dict.get(c_str, c_str)
            clean_cols.append(full_name)
            display_to_abbr[full_name] = site_abbreviation_dict.get(full_name, full_name[:4].upper())
            
    df_plot.columns = clean_cols

    numeric_cols = df_plot.select_dtypes(include=[np.number])
    if numeric_cols.mean().max() <= 1.0:
        df_pct = df_plot * 100
    else:
        df_pct = df_plot.copy()

    numeric_pct = df_pct.select_dtypes(include=[np.number])
    party_order = numeric_pct.mean().sort_values(ascending=False).index.tolist()
    if 'Other' in party_order:
        party_order.remove('Other')
        party_order.append('Other')
        
    # Check max value across all traces to dynamically adjust Y-axis max limit
    max_y_val = df_pct[party_order].max().max()
    y_max_limit = 75 if max_y_val > 67 else 65

    summary_stats = {
        party: {
            'mean': df_pct[party].mean(),
            'lower_95': np.percentile(df_pct[party], 2.5),
            'upper_95': np.percentile(df_pct[party], 97.5)
        }
        for party in party_order
    }

    fig = go.Figure()

    for party in party_order:
        party_data = df_pct[party].dropna().to_numpy()
        if len(party_data) == 0: continue

        stats = summary_stats[party]
        display_name = party
        x_label = display_to_abbr.get(display_name, display_name[:4].upper())
        party_color = party_colors.get(display_name, '#999999' if display_name.startswith('Ind') else '#7F8C8D')

        dynamic_bw = 0.7 if stats['mean'] > 4.0 else 0.2
        dynamic_bw = 2.0 if stats['mean'] > 10.0 else dynamic_bw

        # 1. Base Violin Shape (Opacity set to 1.0)
        fig.add_trace(go.Violin(
            x=[x_label] * len(party_data),
            y=party_data.tolist(),
            name=display_name,
            hoveron='violins',
            hoverinfo='none',
            hovertemplate=(
                f"<b>{display_name}</b><br>"
                f"Mean: {stats['mean']:.1f}%<br>"
                f"2.5% CI: {stats['lower_95']:.1f}%<br>"
                f"97.5% CI: {stats['upper_95']:.1f}%<extra></extra>"
            ),
            box_visible=False,
            meanline_visible=True,
            meanline=dict(color='#333333'),
            line_color='rgba(0,0,0,0)',
            fillcolor=party_color,
            opacity=1.0, 
            width=0.7,
            bandwidth=dynamic_bw,
            points=False
        ))

        # 2. Hover Interaction Dots
        n_pts = len(party_data)
        if n_pts >= 200:
            sorted_data = np.sort(party_data)
            lowest_40 = sorted_data[:40]
            highest_30 = sorted_data[-30:]
            middle_vals = sorted_data[40:-30]
            middle_130 = rng.choice(middle_vals, min(130, len(middle_vals)), replace=False)
            hover_y = np.concatenate([lowest_40, highest_30, middle_130])
        else:
            hover_y = party_data

        sim_templates = [f"<b>{display_name}</b><br>%{{y:.1f}}%<extra></extra>"] * len(hover_y)
        sim_bgs = ['white'] * len(hover_y)
        sim_fonts = [party_color] * len(hover_y)

        fig.add_trace(go.Scatter(
            x=[x_label] * len(hover_y),
            y=hover_y.tolist(),
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=10),
            hovertemplate=sim_templates,
            hoverlabel=dict(bgcolor=sim_bgs, font=dict(color=sim_fonts)),
            name=f"{display_name} sims",
            showlegend=False
        ))

    # Layout configuration
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        legend=dict(
            font=dict(size=13),
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        ),
        violinmode="group",
        violingap=0.1,
        margin=dict(l=35, r=25, t=15, b=90),
        xaxis_tickangle=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(
        fixedrange=True,
        tick0=0,
        dtick=10,                      # Grid every 10%
        showgrid=True,                 # Turn on grid
        gridcolor='rgba(0,0,0,0.1)',   # Very thin, faint line
        gridwidth=1,
        tickformat='.0f',
        ticksuffix='%',
        tickfont=dict(size=13),
        range=[-1, y_max_limit]        # Dynamic top limit
    )

    slug = district_name.lower().replace(" ", "_").replace("'", "")
    
    if preview:
        fig.show()

    slug = district_name.lower().replace(" ", "_").replace("'", "")
    
    json_path = os.path.join(output_dir, f"{slug}_violin.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fig.to_dict(), f)

    return slug

with open(f"Past-simulations/{election_year}_simulation_bundle.pkl", "rb") as f:
        sim_bundle = pickle.load(f)
with open(f"Past-simulations/{election_year}_deterministic_traces.pkl", "rb") as f:
    all_district_steps = pickle.load(f)

fp_pred = sim_bundle['FP_pred']
electorate_winners = sim_bundle['electorate_winners']

manifest = []
districts = sorted(list(all_district_steps.keys()))

import pdb; pdb.set_trace()
for district in ['Morwell']:

    # 2. Generate FP Violin
    if district in fp_pred:
        generate_fp_violin_json(fp_pred[district], district, preview=True)# (district in['Ashwood', 'Bentleigh','Macedon','Footscray','Morwell']))

    if district in electorate_winners:
        generate_win_bar_json(electorate_winners[district], district)
        
import pdb; pdb.set_trace()

















mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

VIC_PARTY_NAME_DICT = {
    "ALP": "Labor",
    "LP": "Liberal",
    "NP": "National",
    "COAL": "Coalition",
    "GRN": "Greens",
    "IND": "Independent",
    "ON": "One Nation",
    "FFP": "Family First",
    "AJP": "Animal Justice",
    "VNS": "Victorian Socialists",
    "DHJP": "Justice Party",
    "LBT": "Libertarian / LDP",
    "SFF": "Shooters, Fishers & Farmers",
    "DLP": "Democratic Labour",
    "HMP": "Legalise Cannabis",
    "OTH": "Other parties (avg)"
}

VIC_DEFAULT_COLOURS = {
    "ALP": "#E13940",
    "LP": "#1C4F9C",
    "NP": "#006644",
    "COAL": "#1C4F9C",
    "GRN": "#10C25B",
    "IND": "#888888",
    "ON": "#F36C21",
    "FFP": "#00AEEF",
    "AJP": "#5B2C6F",
    "VNS": "#990000",
    "DHJP": "#1F618D",
    "LBT": "#F1C40F",
    "OTH": "#888888"
}


def canonical_party_label(cand_code: str) -> str:
    """Maps candidate keys like IND_Melissa_Lowe or TWP to 'IND', preserving party codes."""
    c_str = str(cand_code).strip()
    if c_str.startswith("IND") or c_str.startswith("TWP"):
        return "IND"
    return c_str.split("_")[0] if "_" in c_str else c_str


def extract_actual_fp_from_dop(DOP_table_dict: dict) -> dict:
    """
    Extracts normalized actual Count 0 first-preference percentages (0..100)
    from DOP_table_dict for each electorate.
    """
    actual_fp_dict = {}
    for div, table in DOP_table_dict.items():
        c0 = table.count_0
        tot = sum(c0.values())
        if tot > 0:
            actual_fp_dict[div] = {c: (v / tot) * 100.0 for c, v in c0.items()}
    return actual_fp_dict


def analyse_vic_fp_coverage_and_mae(
    fp_pred_by_year: dict,
    actual_fp_by_year: dict,
    title_suffix: str = "2010–2022 Pooled",
    plot_parties: list = ["ALP", "COAL", "GRN", "IND", "ON", "FFP", "AJP", "VNS"],
    top_mae_parties: list = ["ALP", "LP", "NP", "GRN", "IND", "ON", "FFP", "AJP", "VNS", "DHJP", "LBT"],
    party_colours_csv: str = "Party Colours.csv",
    save_prefix: str = None
):
    """
    Computes vectorized 1%..99% posterior predictive quantile coverage and MAE
    across one or multiple Victorian elections.

    Parameters
    ----------
    fp_pred_by_year : dict
        Either `{year: {div: fp_sims_df}}` for multiple years, or `{div: fp_sims_df}` for a single year.
    actual_fp_by_year : dict
        Either `{year: {div: {cand: actual_pct}}}` for multiple years, or `{div: {cand: actual_pct}}`.
    """
    # Normalize single-year inputs into {year: ...} format
    first_val = next(iter(fp_pred_by_year.values()))
    if isinstance(first_val, pd.DataFrame):
        fp_pred_by_year = {"Single": fp_pred_by_year}
        actual_fp_by_year = {"Single": actual_fp_by_year}

    ci_levels = np.arange(1, 100, dtype=int)         # 1..99%
    lo_pcts = (100.0 - ci_levels) / 2.0              # 49.5 down to 0.5
    hi_pcts = 100.0 - lo_pcts                        # 50.5 up to 99.5

    mae_records = []
    ci_records = []

    for yr, fp_pred_dict in fp_pred_by_year.items():
        actual_dict = actual_fp_by_year[yr]

        for div, fp_sims_df in fp_pred_dict.items():
            if div not in actual_dict:
                continue

            actual_seat = actual_dict[div]
            # Keep candidates present in both simulated DataFrame and actual Count 0
            common_cands = [c for c in fp_sims_df.columns if c in actual_seat]
            if not common_cands:
                continue

            sim_mat = fp_sims_df[common_cands].to_numpy(dtype=float)
            # Convert proportions [0, 1] to percentages [0, 100] if needed
            if np.nanmax(sim_mat) <= 1.05:
                sim_mat = sim_mat * 100.0

            actual_vec = np.array([float(actual_seat[c]) for c in common_cands], dtype=float)
            if actual_vec.sum() <= 1.05:
                actual_vec = actual_vec * 100.0

            # 1. MAE Records
            sim_means = np.mean(sim_mat, axis=0)
            abs_errs = np.abs(sim_means - actual_vec)
            for idx, cand in enumerate(common_cands):
                p_label = canonical_party_label(cand)
                mae_records.append({
                    "year": yr,
                    "div_nm": f"{yr}_{div}" if len(fp_pred_by_year) > 1 else div,
                    "Candidate": cand,
                    "Party": p_label,
                    "sim_mean": sim_means[idx],
                    "actual_pct": actual_vec[idx],
                    "signed_error": actual_vec[idx] - sim_means[idx],
                    "abs_error": abs_errs[idx]
                })

            # 2. Vectorized Central Credible Intervals (1%..99%) in one call
            #    method='weibull' (k/(S+1)) prevents finite-sample truncation bias when N_sims=100
            try:
                lo_mat = np.percentile(sim_mat, lo_pcts, axis=0, method="weibull")  # Shape: (99, C)
                hi_mat = np.percentile(sim_mat, hi_pcts, axis=0, method="weibull")  # Shape: (99, C)
            except TypeError:
                lo_mat = np.percentile(sim_mat, lo_pcts, axis=0)
                hi_mat = np.percentile(sim_mat, hi_pcts, axis=0)

            inside_mat = (actual_vec[None, :] >= lo_mat) & (actual_vec[None, :] <= hi_mat)  # Shape: (99, C)

            for idx, cand in enumerate(common_cands):
                p_label = canonical_party_label(cand)
                flags = inside_mat[:, idx]
                for ci_idx, ci_val in enumerate(ci_levels):
                    ci_records.append({
                        "year": yr,
                        "div_nm": f"{yr}_{div}" if len(fp_pred_by_year) > 1 else div,
                        "Candidate": cand,
                        "Party": p_label,
                        "ci": int(ci_val),
                        "inside": int(flags[ci_idx]),
                        "no_cands": len(common_cands)
                    })

    mae_df = pd.DataFrame(mae_records)
    ci_df = pd.DataFrame(ci_records)

    # -------------------------------------------------------------------------
    # A. Plot 1: Posterior Quantile Calibration Curve (Nominal CI vs Empirical)
    # -------------------------------------------------------------------------
    overall_cov = ci_df.groupby("ci")["inside"].mean() * 100.0
    by_party = ci_df.groupby(["Party", "ci"])["inside"].mean().unstack("Party") * 100.0

    # Combine LP, NP, LNP into COAL for the coverage curve
    coal_mask = ci_df["Party"].isin(["LP", "NP", "LNP", "COAL"])
    if coal_mask.any():
        by_party["COAL"] = ci_df.loc[coal_mask].groupby("ci")["inside"].mean() * 100.0

    # Dynamic N = total candidate-seat instances evaluated
    n_obs_total = int((ci_df["ci"] == 50).sum())
    z = 1.96
    p0 = ci_levels / 100.0
    den = 1.0 + (z ** 2) / n_obs_total
    center = (p0 + (z ** 2) / (2.0 * n_obs_total)) / den
    half = (z / den) * np.sqrt(p0 * (1.0 - p0) / n_obs_total + (z ** 2) / (4.0 * (n_obs_total ** 2)))
    lower_band = (center - half) * 100.0
    upper_band = (center + half) * 100.0

    plt.figure(figsize=(9, 6))
    plt.fill_between(ci_levels, lower_band, upper_band, color="cyan", alpha=0.12, label=f"95% Wilson band (n={n_obs_total})")
    plt.plot(ci_levels, ci_levels, linewidth=2, linestyle="--", color="black", label="Perfect calibration (y = x)")
    plt.plot(ci_levels, overall_cov.values, linewidth=2.5, color="violet", label="Overall empirical coverage")

    for p in plot_parties:
        if p in by_party.columns:
            plt.plot(
                ci_levels,
                by_party[p].values,
                linewidth=1.2,
                alpha=0.45,
                label=p,
                color=VIC_DEFAULT_COLOURS.get(p, "#555555")
            )

    plt.xlabel("Nominal CI level (%)")
    plt.ylabel("Empirical coverage (% inside CI)")
    plt.title(f"FP Posterior Calibration ({title_suffix})")
    plt.ylim(0, 100)
    plt.xlim(1, 99)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_fp_coverage.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.show()

    # -------------------------------------------------------------------------
    # B. Plot 2: Horizontal MAE Bar Chart by Party (With "Other parties (avg)")
    # -------------------------------------------------------------------------
    mae_overall = float(mae_df["abs_error"].mean())
    mae_df["PlotParty"] = mae_df["Party"].where(mae_df["Party"].isin(top_mae_parties), "OTH")
    mae_party_series = mae_df.groupby("PlotParty")["abs_error"].mean()

    bar_df = pd.DataFrame({
        "Party": mae_party_series.index,
        "MAE": mae_party_series.values
    })
    bar_df["Label"] = bar_df["Party"].map(lambda x: VIC_PARTY_NAME_DICT.get(x, x))
    bar_df = bar_df.sort_values("MAE", ascending=True).reset_index(drop=True)

    def soften_color(color, amount=0.35):
        rgb = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        return tuple((1.0 - amount) * rgb + amount * white)

    colors_map = {VIC_PARTY_NAME_DICT.get(k, k): v for k, v in VIC_DEFAULT_COLOURS.items()}
    csv_path = Path(party_colours_csv)
    if csv_path.exists():
        colour_df = pd.read_csv(csv_path)
        colour_df["Party"] = colour_df["Party"].astype(str).str.replace("’", "'", regex=False)
        colors_map.update(colour_df.set_index("Party")["Colour"].to_dict())

    bar_colors = [
        soften_color(colors_map.get(lbl, "#888888"), amount=0.35)
        for lbl in bar_df["Label"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    y = np.arange(len(bar_df))
    bars = ax.barh(y, bar_df["MAE"], color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(bar_df["Label"], fontsize=11)
    ax.set_xlabel("Mean Absolute Error (%)", fontsize=13)
    ax.set_title(f"First Preference Mean Absolute Error by Party ({title_suffix})", fontsize=14)

    ax.axvline(mae_overall, linestyle="--", linewidth=1.3, color="purple")
    ax.text(
        mae_overall,
        max(0.5, len(bar_df) * 0.45),
        f" Overall MAE = {mae_overall:.2f}%",
        rotation=90, va="top", ha="right", color="purple", fontsize=13
    )

    for rect in bars:
        w = rect.get_width()
        ax.text(w + 0.05, rect.get_y() + rect.get_height() / 2.0, f"{w:.2f}", va="center", fontsize=10)

    ax.tick_params(axis="x", labelsize=11)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_mae_by_party.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.show()

    # -------------------------------------------------------------------------
    # C. Diagnostic Summary Table (50%, 80%, 90%, 95%, 99% Coverage + MAE)
    # -------------------------------------------------------------------------
    key_cis = [50, 80, 90, 95, 99]
    cov_summary = by_party.loc[key_cis].T
    cov_summary.columns = [f"CI_{c}%" for c in key_cis]
    cov_summary["MAE (%)"] = mae_df.groupby("Party")["abs_error"].mean()
    cov_summary["N_obs"] = mae_df.groupby("Party")["abs_error"].count()
    cov_summary.loc["OVERALL"] = (
        [overall_cov.loc[c] for c in key_cis] + [mae_overall, n_obs_total]
    )

    # 99% CI Extreme Misses Table
    extreme_misses = ci_df.loc[(ci_df["ci"] == 99) & (ci_df["inside"] == 0)].merge(
        mae_df[["div_nm", "Candidate", "actual_pct", "sim_mean", "abs_error"]],
        on=["div_nm", "Candidate"],
        how="left"
    ).sort_values("abs_error", ascending=False)

    print(f"\n=== FP COVERAGE & MAE SUMMARY ({title_suffix}) ===")
    print(cov_summary.dropna(subset=["N_obs"]).sort_values("N_obs", ascending=False).round(2))
    print(f"\n=== 99% CI EXTREME MISSES (Count = {len(extreme_misses)} / {n_obs_total}) ===")
    print(extreme_misses[["div_nm", "Candidate", "actual_pct", "sim_mean", "abs_error"]].head(20).round(2))

    return cov_summary, mae_df, ci_df, extreme_misses