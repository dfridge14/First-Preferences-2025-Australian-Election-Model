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
    'Greens': 'GRN', 'Victorian Socialists': 'VS', 'Legalise Cannabis': 'LCP',
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

def parse_independent(col_name):
    """ Converts 'IND_Zoe_Daniel' or 'IND_Daniel' to 'Ind. Daniel' """
    parts = str(col_name).strip().split('_')
    last_name = parts[-1] if len(parts) > 1 else 'Unknown'
    display_name = f"Ind. {last_name}"
    abbr = f"IND {last_name[:3].title()}."
    return display_name, abbr

# ==========================================
# 2. PLOT GENERATION FUNCTION
# ==========================================
def make_single_electorate_violin_plot(df_raw, district_name, colors, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
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
        party_color = colors.get(display_name, '#999999' if display_name.startswith('Ind') else '#7F8C8D')

        dynamic_bw = 0.7 if stats['mean'] > 4.0 else 0.2
        dynamic_bw = 2.0 if stats['mean'] > 10.0 else dynamic_bw

        # 1. Base Violin Shape (Opacity set to 1.0)
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
        height=420,
        title=dict(
            text="<b>Projected First Preference Votes</b>", 
            x=0.5, 
            xanchor='center', 
            font=dict(size=18)
        ),
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
        margin=dict(l=35, r=25, t=50, b=90),
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
    
    json_path = os.path.join(output_dir, f"{slug}_violin.json")
    with open(json_path, "w") as f:
        f.write(fig.to_json())

    html_path = os.path.join(output_dir, f"{slug}_violin.html")
    fig.write_html(html_path, include_plotlyjs='cdn', config={'displayModeBar': False, 'scrollZoom': False})

    return slug

# ==========================================
# 3. BATCH RUNNER & DASHBOARD BUILDER
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
    districts.sort()
    print(f"Loaded simulation data for {len(districts)} districts.")

    print("Generating figures for all districts...")
    
    generated_slugs = {}
    for district in districts:
        df_district = sim_results[district]
        slug = make_single_electorate_violin_plot(df_district, district, colors, output_dir="output")
        generated_slugs[district] = slug
        print(f"✓ Generated {district}")

    print("Building mobile dashboard...")
    index_path = os.path.join("output", "index.html")
    
    options_html = ""
    for name, slug in generated_slugs.items():
        options_html += f'        <option value="{slug}">{name}</option>\n'

    # Notice the second iframe and the updated Javascript function
    dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Electorate Dashboard</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px 10px; background: transparent; text-align: center; }}
        select {{ font-size: 16px; padding: 12px; margin-bottom: 15px; width: 95%; max-width: 400px; border-radius: 8px; border: 1px solid #ccc; background: white; }}
        iframe {{ width: 100%; border: none; background: transparent; }}
        .plot-frame {{ height: 450px; margin-bottom: 10px; }}
        .table-frame {{ height: 500px; }}
    </style>
</head>
<body>
    <select id="districtSelect" onchange="updateDashboard(this.value)">
{options_html}
    </select>
    <br>
    
    <!-- Top Plotly Chart -->
    <iframe id="plotFrame" class="plot-frame" src="" allowtransparency="true"></iframe>
    
    <!-- Bottom Data Table -->
    <iframe id="tableFrame" class="table-frame" src="" allowtransparency="true" scrolling="no" style="width: 100%; border: none; overflow: hidden;"></iframe>

    <script>
        function updateDashboard(slug) {{
            // Loads the plot from the current folder
            document.getElementById('plotFrame').src = slug + '_violin.html';
            
            // Loads the table from the sibling output_tables folder
            document.getElementById('tableFrame').src = '../output_tables/' + slug + '_table.html';
        }}
        
        // Initialize on page load
        var sel = document.getElementById('districtSelect');
        if (sel.options.length > 0) {{
            updateDashboard(sel.options[0].value);
        }}
    </script>
</body>
</html>"""

    with open(index_path, "w") as f:
        f.write(dashboard_html)

    print("\nCompleted! 2-in-1 Dashboard")
