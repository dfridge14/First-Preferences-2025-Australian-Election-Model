import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Initialize RNG for deterministic hover sampling
rng = np.random.default_rng(42)

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

site_abbreviation_dict = {
    'Labor': 'LAB', 'Liberal': 'LIB', 'National': 'NAT', 'Coalition': 'COAL',
    'Greens': 'GRN', 'Victorian Socialists': 'VSOC', 'Legalise Cannabis': 'LCP',
    'Animal Justice': 'AJP', 'Family First': 'FFP', 'Libertarian': 'LBT',
    'Shooters Fishers & Farmers': 'SFF', 'Australian Democrats': 'AUD',
    'One Nation': 'ON', 'Trumpet of Patriots': 'TOP', 'FUSION': 'FUSN',
    'Sustainable Australia': 'SAP', 'Citizens Party': 'CIT', 'Other': 'OTH'
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

# ==========================================
# 2. INDEPENDENT PARSER FUNCTION
# ==========================================
def parse_independent(col_name):
    """ Converts 'IND_Zoe_Daniel' or 'IND_Daniel' to 'Ind. Daniel' """
    parts = str(col_name).strip().split('_')
    last_name = parts[-1] if len(parts) > 1 else 'Unknown'
    display_name = f"Ind. {last_name}"
    # Farrer legacy style: "IND Dan."
    abbr = f"IND {last_name[:3].title()}."
    return display_name, abbr

# ==========================================
# 3. PLOT GENERATION FUNCTION
# ==========================================
def make_single_electorate_violin_plot(df_raw, district_name, colors, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    df_plot = df_raw.copy()

    if len(df_plot) > 1000:
        df_plot = df_plot.sample(n=1000, random_state=42)

    # Clean columns and apply Independent regex
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

    # Convert to percentage
    numeric_cols = df_plot.select_dtypes(include=[np.number])
    if numeric_cols.mean().max() <= 1.0:
        df_pct = df_plot * 100
    else:
        df_pct = df_plot.copy()

    # Sort parties
    numeric_pct = df_pct.select_dtypes(include=[np.number])
    party_order = numeric_pct.mean().sort_values(ascending=False).index.tolist()

    if 'Other' in party_order:
        party_order.remove('Other')
        party_order.append('Other')

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
        
        # Pull color from dict, or force #999999 if it is a dynamically generated Independent
        party_color = colors.get(display_name, '#999999' if display_name.startswith('Ind') else '#7F8C8D')

        dynamic_bw = 0.7 if stats['mean'] > 4.0 else 0.2
        dynamic_bw = 2.0 if stats['mean'] > 10.0 else dynamic_bw

        # 1. Base Violin Shape
        fig.add_trace(go.Violin(
            x=[x_label] * len(party_data),
            y=party_data,
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
            meanline=dict(color='#5d3fd3'),
            line_color='rgba(0,0,0,0)',
            fillcolor=party_color,
            opacity=0.7,
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
            y=hover_y,
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=10),
            hovertemplate=sim_templates,
            hoverlabel=dict(bgcolor=sim_bgs, font=dict(color=sim_fonts)),
            name=f"{display_name} sims",
            showlegend=False
        ))

    # Layout configuration
    fig.update_layout(
        title=dict(text=f"{district_name} - Primary Vote Projection", x=0.05, font=dict(size=18)),
        showlegend=True,
        hovermode="closest",
        legend=dict(
            font=dict(size=14),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        violinmode="group",
        violingap=0.1,
        margin=dict(l=35, r=25, t=50, b=40),
        xaxis_tickangle=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(
        tick0=0,
        dtick=5,
        tickformat='.0f',
        ticksuffix='%',
        tickfont=dict(size=13),
        range=[-1, 65]
    )

    slug = district_name.lower().replace(" ", "_").replace("'", "")
    
    # JSON export for WP
    json_path = os.path.join(output_dir, f"{slug}_violin.json")
    with open(json_path, "w") as f:
        f.write(fig.to_json())

    # HTML export for instant mobile viewing
    html_path = os.path.join(output_dir, f"{slug}_violin.html")
    fig.write_html(html_path, include_plotlyjs='cdn')

    return fig

# ==========================================
# 4. BATCH RUNNER
# ==========================================
if __name__ == "__main__":
    pkl_file = "sim_results.pkl"
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Could not locate {pkl_file} in current working directory.")

    print(f"Loading {pkl_file}...")
    with open(pkl_file, "rb") as f:
        sim_results = pickle.load(f)

    colors = party_colors
    districts = list(sim_results.keys())
    print(f"Loaded simulation data for {len(districts)} districts.")

    # Run a test batch first (3 districts). Once you confirm it works, 
    # change "districts[:3]" to just "districts" to run all 88 at once.
    test_batch = districts[:3]
    print(f"Generating test figures for: {test_batch}...")

    for district in test_batch:
        df_district = sim_results[district]
        make_single_electorate_violin_plot(df_district, district, colors, output_dir="output")
        print(f"Generated output/ for {district}")

    print("\nCompleted. Files saved to output/ folder.")
  
