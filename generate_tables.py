import os
import sys
import pickle
import numpy as np
import pandas as pd

# ==========================================
# 1. DICTIONARIES & COLOR CONFIGURATION
# ==========================================
party_name_dict = {
    'ALP': 'Labor', 'LP': 'Liberal', 'NP': 'National', 'COAL': 'Coalition',
    'GRN': 'Greens', 'VNS': 'Victorian Socialists', 'VSOC': 'Victorian Socialists',
    'HMP': 'Legalise Cannabis', 'LCV': 'Legalise Cannabis',
    'AJP': 'Animal Justice', 'FFP': 'Family First', 'FFPA': 'Family First',
    'LDP': 'Libertarian', 'LTP': 'Libertarian', 'SFF': 'Shooters Fishers & Farmers',
    'ASP': 'Shooters Fishers & Farmers', 'AUD': 'Australian Democrats',
    'ON': 'One Nation', 'TOP': 'Trumpet of Patriots', 'CYA': 'Trumpet of Patriots',
    'FUSN': 'FUSION', 'SOPA': 'FUSION', 'SAP': 'Sustainable Australia',
    'SPP': 'Sustainable Australia', 'CIT': 'Citizens Party', 'CEC': 'Citizens Party',
    'OTH': 'Other'
}

party_colors = {
    'Labor': '#F04143', 'Liberal': '#4053B2', 'Liberal National': '#4053B2',
    'Greens': '#6BBE6C', 'One Nation': '#EB8846', 'Legalise Cannabis': '#91D06A',
    'Libertarian': '#F4D04C', 'Trumpet of Patriots': '#FAEE76',
    'Animal Justice': '#764E9A', 'Family First': '#9B86A8', 'People\'s First': '#5679A8',
    'Independent': '#999999', 'National': '#057e0d', 'Coalition': '#377eb8',
    'Country Liberal': '#fa8c03', 'Teal Independent': '#1ed6db', 'HEART': '#dc7f2b',
    'Australian Democrats': '#f6ca06', 'FUSION': '#cd18fc',
    'Shooters Fishers & Farmers': '#f3144d', 'Citizens Party': '#496332',
    'Socialist Alliance': '#91120d', 'Indigenous-Aboriginal': '#284820',
    'Katter\'s Australian': '#b30f0a', 'Centre Alliance': '#f15705',
    'South Australia National': '#057e0d', 'Victorian Socialists': '#070706',
    'Christians': '#fddd02', 'WA Nationals': '#057e0d', 'Great Australian Party': '#030562',
    'Other': '#7F8C8D'
}

def parse_independent(col_name):
    parts = str(col_name).strip().split('_')
    last_name = parts[-1] if len(parts) > 1 else 'Unknown'
    return f"Ind. {last_name}"

# ==========================================
# 2. HTML GENERATOR FUNCTION
# ==========================================
def generate_step_table_html(df_steps, district_name, output_dir="output_tables"):
    os.makedirs(output_dir, exist_ok=True)
    
    if len(df_steps) < 8:
        return None

    numeric_cols = df_steps.select_dtypes(include=[np.number])
    if numeric_cols.mean().max() <= 1.0:
        df_steps = df_steps * 100

    r0 = df_steps.iloc[0].fillna(0)
    r1 = df_steps.iloc[1].fillna(0)
    r5 = df_steps.iloc[5].fillna(0)
    r6 = df_steps.iloc[6].fillna(0)
    r7 = df_steps.iloc[7].fillna(0)

    shifts = {
        'Baseline': r0,
        'Inc_Rem': r1 - r0,
        'Cand_Change': r5 - r1,
        'Poll_Shift': r6 - r5,
        'Inc_Adv': r7 - r6,
        'Final': r7
    }

    cumulatives = { 0: r0, 1: r1, 2: r5, 3: r6, 4: r7 }

    col_dead = {
        1: shifts['Inc_Rem'].abs().sum() < 0.05,
        2: shifts['Cand_Change'].abs().sum() < 0.05,
        3: shifts['Poll_Shift'].abs().sum() < 0.05,
        4: shifts['Inc_Adv'].abs().sum() < 0.05
    }

    party_order = r7.sort_values(ascending=False).index.tolist()
    if 'Other' in party_order:
        party_order.remove('Other')
        party_order.append('Other')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; background: transparent; overflow-x: hidden; }}
        .model-flow-container {{ font-family: 'Poppins', sans-serif; max-width: 1250px; margin: 0 auto; overflow-x: auto; padding-bottom: 20px; display: flex; flex-direction: column; align-items: center; }}
        .model-flow-container::-webkit-scrollbar {{ height: 8px; -webkit-appearance: none; }}
        .model-flow-container::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}
        
        .model-flow-grid {{ display: grid; grid-template-columns: 150px repeat(6, minmax(85px, 1fr)); gap: 6px; min-width: 800px; margin: 0 auto; align-items: center; justify-content: start; }}
        .flow-header {{ font-weight: 700; color: #475569; font-size: 0.85rem; text-align: center; padding-bottom: 10px; border-bottom: 2px solid #cbd5e1; margin-bottom: 10px; line-height: 1.25; }}
        
        /* Sticky left column with #2596be background and mask shadow */
        .party-label-header, .party-label {{ 
            position: -webkit-sticky; 
            position: sticky; 
            left: 0; 
            background-color: #2596be; 
            z-index: 20; 
            height: 100%; 
            margin: 0; 
            padding: 1px 4px; 
            display: flex; 
            align-items: center; 
            box-sizing: border-box; 
            box-shadow: 2px 0 4px -2px rgba(0,0,0,0.15); 
        }}
        .party-label-header {{ border-bottom: 2px solid #94a3b8 !important; font-size: 0.95rem; font-weight: 700; color: #ffffff; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); justify-content: flex-start; padding-left: 8px; }}
        .party-label {{ font-weight: 700; font-size: 0.95rem; line-height: 1.2; text-shadow: 1px 1px 0 #fff, -1px 1px 0 #fff, 1px -1px 0 #fff, -1px -1px 0 #fff; padding-left: 8px; }}
        
        .data-cell {{ display: flex; justify-content: center; }}
        .vote-pill {{ box-sizing: border-box; border-radius: 8px; padding: 6px 12px; font-weight: 600; font-size: 0.9rem; width: 70px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        
        .pill-ghost {{ background-color: rgba(255, 255, 255, 0.4); color: #94a3b8; box-shadow: none; border: 1px dashed #cbd5e1; }}
        .spacer-right {{ margin-right: 16px; }}

        @media (max-width: 768px) {{
            .model-flow-container {{ margin: 0 5px; }}
            .model-flow-grid {{ grid-template-columns: 85px repeat(6, minmax(55px, 65px)); min-width: 450px; gap: 2px; }}
            .vote-pill {{ width: 50px; padding: 3px 4px; font-size: 0.75rem; border-radius: 6px; }}
            .flow-header {{ font-size: 0.65rem; padding-bottom: 4px; letter-spacing: -0.2px; line-height: 1.1; margin-bottom: 6px; }}
            .party-label {{ font-size: 0.8rem; background-color: #2596be; padding-left: 4px; text-shadow: 1px 1px 0 #fff, -1px 1px 0 #fff, 1px -1px 0 #fff, -1px -1px 0 #fff; }}
            .party-label-header {{ padding-left: 4px; }}
            .spacer-right {{ margin-right: 8px; }}
        }}
    </style>
</head>
<body>
    <div style="max-width: 1250px; margin: 1rem auto 1rem auto; font-family: 'Poppins', sans-serif;">
        <h2 style="color: #6419a8; font-weight: 800; font-size: 1.8rem; margin-bottom: 8px; margin-top: 0; text-shadow: 1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff;">Deconstructing the Model</h2>
        <div style="display: inline-block; padding: 6px 16px; background: rgba(255,255,255,0.7); border-radius: 6px; color: #475569; font-size: 0.9rem; font-weight: 500; border: 1px solid #e2e8f0;">
            Tracking how incumbency, candidate changes, and polling swings shape each party's final projected vote.
        </div>
    </div>
    
    <div class="model-flow-container">
        <div class="model-flow-grid">
            <div class="flow-header party-label-header">Party</div>
            <div class="flow-header spacer-right" style="color: #475569;">2022<br>Result</div>
            <div class="flow-header" style="color: #475569;">Incumbency<br>Removed</div>
            <div class="flow-header" style="color: #475569;">Parties<br>Contesting</div>
            <div class="flow-header" style="color: #475569;">Poll<br>Swing</div>
            <div class="flow-header spacer-right" style="color: #475569;">Incumbency<br>Add</div>
            <div class="flow-header" style="color: #9333ea; background: rgba(255,255,255,0.8); border-radius: 4px; padding: 2px;">Model<br>Average</div>
    """

    def format_shift_pill(val, color, step_k, party):
        col_is_dead = col_dead[step_k]
        cumuls = [cumulatives[i].get(party, 0) for i in range(5)]
        active_indices = [i for i, c_val in enumerate(cumuls) if c_val > 0.05]
        party_is_alive = (active_indices[0] <= step_k <= active_indices[-1]) if active_indices else False

        if abs(val) < 0.05:
            if col_is_dead or not party_is_alive:
                return '<div class="vote-pill pill-ghost">-</div>'
            else:
                return f'<div class="vote-pill" style="background-color: {color}; color: white; opacity: 0.6;">-</div>'
        else:
            sign = "+" if val > 0 else ""
            return f'<div class="vote-pill" style="background-color: {color}; color: white; opacity: 0.6;">{sign}{val:.1f}</div>'

    def format_base_pill(val, color, is_final=False):
        if abs(val) < 0.05:
            extra_style = ' style="border: 2px solid #9333ea;"' if is_final else ''
            return f'<div class="vote-pill pill-ghost"{extra_style}>0.0%</div>'
        else:
            border_style = " border: 2px solid #9333ea;" if is_final else ""
            return f'<div class="vote-pill" style="background-color: {color}; color: white;{border_style}">{val:.1f}%</div>'

    for raw_party in party_order:
        display_name = party_name_dict.get(raw_party, raw_party)
        if str(raw_party).startswith('IND_'):
            display_name = parse_independent(raw_party)
            
        color = party_colors.get(display_name, party_colors.get(raw_party, '#999999' if display_name.startswith('Ind') else '#7F8C8D'))

        if shifts['Final'][raw_party] < 0.05 and shifts['Baseline'][raw_party] < 0.05:
            continue

        html += f"""
            <div class="party-label" style="color: {color};">{display_name}</div>
            <div class="data-cell spacer-right">{format_base_pill(shifts['Baseline'][raw_party], color, False)}</div>
            <div class="data-cell">{format_shift_pill(shifts['Inc_Rem'][raw_party], color, 1, raw_party)}</div>
            <div class="data-cell">{format_shift_pill(shifts['Cand_Change'][raw_party], color, 2, raw_party)}</div>
            <div class="data-cell">{format_shift_pill(shifts['Poll_Shift'][raw_party], color, 3, raw_party)}</div>
            <div class="data-cell spacer-right">{format_shift_pill(shifts['Inc_Adv'][raw_party], color, 4, raw_party)}</div>
            <div class="data-cell">{format_base_pill(shifts['Final'][raw_party], color, True)}</div>
        """

    html += """
        </div>
    </div>
</body>
</html>
    """
    
    slug = district_name.lower().replace(" ", "_").replace("'", "")
    file_path = os.path.join(output_dir, f"{slug}_table.html")
    with open(file_path, "w") as f:
        f.write(html)
        
    return slug

if __name__ == "__main__":
    pkl_file = "deterministic_traces.pkl"
    if not os.path.exists(pkl_file):
        sys.exit(1)

    with open(pkl_file, "rb") as f:
        all_district_steps = pickle.load(f)

    for district in all_district_steps.keys():
        generate_step_table_html(all_district_steps[district], district, output_dir="output_tables")
        
