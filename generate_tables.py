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
    """ Converts 'IND_Zoe_Daniel' or 'IND_Daniel' to 'Ind. Daniel' """
    parts = str(col_name).strip().split('_')
    last_name = parts[-1] if len(parts) > 1 else 'Unknown'
    return f"Ind. {last_name}"

# ==========================================
# 2. HTML GENERATOR FUNCTION
# ==========================================
def generate_step_table_html(df_steps, district_name, output_dir="output_tables"):
    os.makedirs(output_dir, exist_ok=True)
    
    if len(df_steps) < 8:
        print(f"Skipping {district_name}: DataFrame has fewer than 8 rows ({len(df_steps)}).")
        return None

    # Convert to percentage if raw decimals
    numeric_cols = df_steps.select_dtypes(include=[np.number])
    if numeric_cols.mean().max() <= 1.0:
        df_steps = df_steps * 100

    # Extract exact requested rows
    r0 = df_steps.iloc[0].fillna(0)  # Row 0: Last Election
    r1 = df_steps.iloc[1].fillna(0)  # Row 1: Incumbency Removal
    r5 = df_steps.iloc[5].fillna(0)  # Row 5: Candidate Change
    r6 = df_steps.iloc[6].fillna(0)  # Row 6: Polling Shift
    r7 = df_steps.iloc[7].fillna(0)  # Row 7: Final Projection / Incumbency Adv

    # Mathematically exact step shifts
    shifts = {
        'Baseline': r0,
        'Inc_Rem': r1 - r0,
        'Cand_Change': r5 - r1,
        'Poll_Shift': r6 - r5,
        'Inc_Adv': r7 - r6,
        'Final': r7
    }

    # Sort parties descending based on Final Result
    party_order = r7.sort_values(ascending=False).index.tolist()

    # Move 'Other' to the bottom
    if 'Other' in party_order:
        party_order.remove('Other')
        party_order.append('Other')

    # Farrer Style CSS + 6-Data-Column Grid
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&display=swap" rel="stylesheet">
    <style>
        .model-flow-container {{ font-family: 'Poppins', sans-serif; max-width: 1250px; margin: 0 auto; overflow-x: auto; padding-bottom: 10px; display: flex; flex-direction: column; align-items: center; }}
        .model-flow-container::-webkit-scrollbar {{ height: 8px; -webkit-appearance: none; }}
        .model-flow-container::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}
        
        /* Desktop Grid: Party (150px) + 6 Data Columns */
        .model-flow-grid {{ display: grid; grid-template-columns: 150px repeat(6, minmax(85px, 1fr)); gap: 6px; min-width: 800px; margin: 0 auto; align-items: center; justify-content: start; }}
        
        .flow-header {{ font-weight: 700; color: #475569; font-size: 0.85rem; text-align: center; padding-bottom: 10px; border-bottom: 2px solid #cbd5e1; margin-bottom: 10px; line-height: 1.25; }}
        
        /* Sticky Left Column */
        .party-label-header, .party-label {{ position: sticky; left: 0; background-color: #ffffff; z-index: 10; height: 100%; margin: 0 !important; padding: 1px 1px; display: flex; align-items: center; box-sizing: border-box; border-right: 1px solid #f1f5f9; }}
        .party-label-header {{ border-bottom: 2px solid #94a3b8 !important; font-size: 0.95rem; font-weight: 700; color: #334155; }}
        .party-label {{ font-weight: 700; font-size: 0.95rem; line-height: 1.2; }}
        
        /* Pill Styling */
        .data-cell {{ display: flex; justify-content: center; }}
        .vote-pill {{ box-sizing: border-box; border-radius: 8px; padding: 6px 12px; font-weight: 600; font-size: 0.9rem; width: 70px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .pill-ghost {{ background-color: #f1f5f9; color: #94a3b8; box-shadow: none; border: 1px dashed #cbd5e1; }}
        .spacer-right {{ margin-right: 16px; }}

        /* Mobile Responsiveness */
        @media (max-width: 768px) {{
            .model-flow-container {{ margin: 0 5px; }}
            .model-flow-grid {{ grid-template-columns: 85px repeat(6, minmax(55px, 65px)); min-width: 450px; gap: 2px; }}
            .vote-pill {{ width: 50px; padding: 3px 4px; font-size: 0.75rem; border-radius: 6px; }}
            .flow-header {{ font-size: 0.65rem; padding-bottom: 4px; letter-spacing: -0.2px; line-height: 1.1; margin-bottom: 6px; }}
            .party-label {{ font-size: 0.8rem; }}
            .spacer-right {{ margin-right: 8px; }}
        }}
    </style>
</head>
<body>
    <div style="max-width: 1250px; margin: 1rem auto 1rem auto; font-family: 'Poppins', sans-serif;">
        <h2 style="color: #6419a8; font-weight: 800; font-size: 1.8rem; margin-bottom: 8px; margin-top: 0;">Deconstructing the Model: {district_name}</h2>
        <div style="display: inline-block; padding: 6px 16px; background: #f8fafc; border-radius: 6px; color: #475569; font-size: 0.9rem; font-weight: 500; letter-spacing: 0.2px; border: 1px solid #e2e8f0;">
            Tracking how incumbency, candidate changes, and polling swings shape each party's final projected vote.
        </div>
    </div>
    
    <div class="model-flow-container">
        <div class="model-flow-grid">
             <div class="flow-header party-label-header">Party</div>
            <div class="flow-header spacer-right">2022<br>Result</div>
            <div class="flow-header">Incumbency<br>Removed</div>
            <div class="flow-header">Parties<br>Contesting</div>
            <div class="flow-header">Poll<br>Swing</div>
            <div class="flow-header spacer-right">Incumbency<br>Added</div>
            <div class="flow-header" style="color: #9333ea;">Model<br>Average</div>
    """

    def format_pill(val, color, is_shift=False, is_final=False):
        # Handle zeros cleanly using ghost pills
        if abs(val) < 0.05:
            if is_shift:
                return '<div class="vote-pill pill-ghost">-</div>'
            else:
                return '<div class="vote-pill pill-ghost">0.0%</div>'
        
        # Build text format
        sign = "+" if is_shift and val > 0 else ""
        text_val = f"{sign}{val:.1f}" + ("%" if not is_shift else "")
        
        # Build inline styles
        style = f"background-color: {color}; color: white;"
        if is_shift:
            style += " opacity: 0.6;"
        if is_final:
            style += " border: 2px solid #9333ea;"
            
        return f'<div class="vote-pill" style="{style}">{text_val}</div>'

    # Generate party rows
    for raw_party in party_order:
        # Standardize party name and colors
        display_name = party_name_dict.get(raw_party, raw_party)
        if str(raw_party).startswith('IND_'):
            display_name = parse_independent(raw_party)
            
        color = party_colors.get(display_name, party_colors.get(raw_party, '#999999' if display_name.startswith('Ind') else '#7F8C8D'))

        # Skip parties that never cross 0.05% at baseline OR final
        if shifts['Final'][raw_party] < 0.05 and shifts['Baseline'][raw_party] < 0.05:
            continue

        html += f"""
            <div class="party-label" style="color: {color};">{display_name}</div>
            <div class="data-cell spacer-right">{format_pill(shifts['Baseline'][raw_party], color, is_shift=False)}</div>
            <div class="data-cell">{format_pill(shifts['Inc_Rem'][raw_party], color, is_shift=True)}</div>
            <div class="data-cell">{format_pill(shifts['Cand_Change'][raw_party], color, is_shift=True)}</div>
            <div class="data-cell">{format_pill(shifts['Poll_Shift'][raw_party], color, is_shift=True)}</div>
            <div class="data-cell spacer-right">{format_pill(shifts['Inc_Adv'][raw_party], color, is_shift=True)}</div>
            <div class="data-cell">{format_pill(shifts['Final'][raw_party], color, is_shift=False, is_final=True)}</div>
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

# ==========================================
# 3. BATCH RUNNER
# ==========================================
if __name__ == "__main__":
    pkl_file = "deterministic_traces.pkl"
    if not os.path.exists(pkl_file):
        print(f"Error: Could not locate {pkl_file}. Run the simulation script to step 2 first.")
        sys.exit(1)

    print(f"Loading {pkl_file}...")
    with open(pkl_file, "rb") as f:
        all_district_steps = pickle.load(f)

    districts = list(all_district_steps.keys())
    print(f"Loaded step-by-step data for {len(districts)} districts.")
    print("Generating HTML tables...")

    for district in districts:
        df_steps = all_district_steps[district]
        generate_step_table_html(df_steps, district, output_dir="output_tables")
        print(f"✓ Generated table for {district}")

    print("\nCompleted! Tables saved to output_tables/ folder.")
      
