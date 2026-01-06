import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1) Common bins
# -----------------------
bin_edges = [
    0.40,   # <60%
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
    1   # 99%+
]

midpoints = np.array([0.508, 0.65, 0.75, 0.85, 0.925, 0.975, 0.995])

# -----------------------
# 2) Your model
# -----------------------
obs_mine = np.array([0.363, 0.615, 0.765, 0.818, 0.923, 1.0, 1.0])
n_mine   = np.array([11, 13, 17, 22, 26, 34, 27])

# -----------------------
# 3) YouGov MRP (reordered)
# -----------------------
obs_yougov = np.array([0.5, 1,1,0.75, 0.777777777777778,
                        0.954545454545455, 0.942857142857143])
n_yougov   = np.array([6, 2, 7, 12, 9, 44, 70])

obs_aef = np.array([0.5, 7/11,7/12,18/23,10/12,1,1])

n_aef = np.array([10,11,12,23,12,52,30])

# -----------------------
# 4) Weighted linear fit
# -----------------------
def weighted_fit(x, y, w):
    # Fit y = a + b x
    coef = np.polyfit(x, y, deg=1, w=w)
    return coef  # [slope, intercept]

coef_mine   = weighted_fit(midpoints, obs_mine, n_mine)
coef_yougov = weighted_fit(midpoints, obs_yougov, n_yougov)

x_fit = np.linspace(0.5, 1.0, 200)
y_fit_mine   = coef_mine[0] * x_fit + coef_mine[1]
y_fit_yougov = coef_yougov[0] * x_fit + coef_yougov[1]

# -----------------------
# 5) Marker sizes
# -----------------------
def size_from_n(n):
    return 5 + 5* n # 35 * np.sqrt(n)

s_mine = size_from_n(n_mine)
s_you  = size_from_n(n_yougov)
s_aef = size_from_n(n_aef)
s_aef[6] = 220


# -----------------------
# 6) Plot
# -----------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))

for i in range(len(bin_edges) - 1):
    if i % 2 == 0:  # shade every second bin
        ax.axvspan(
            bin_edges[i],
            bin_edges[i + 1],
            color="violet",
            alpha=0.06,
            zorder=0
        )


# Faint vertical lines at bin midpoints
#for x in midpoints:
#    ax.axvline(x=x, linestyle=":", linewidth=1, alpha=0.35, zorder=0)


# Perfect calibration
ax.plot([0.508, 1.0], [0.508, 1.0], linestyle="--", linewidth=1, label="Perfect calibration")

# Scatter points
ax.scatter(midpoints, obs_yougov, s=s_you, label="YouGov MRP", color = 'salmon')
ax.scatter(midpoints, obs_aef, s=s_aef, label="AEForecasts", color = 'mediumseagreen')
ax.scatter(midpoints, obs_mine, s=s_mine, label="First Preference", color = 'purple')


# Best-fit lines
#ax.plot(x_fit, y_fit_mine, linewidth=2, color = 'purple')
#ax.plot(x_fit, y_fit_yougov, linewidth=2, color = 'salmon')

# Formatting
ax.set_xlim(0.48, 1.01)
ax.set_ylim(0.30, 1.03)
ax.set_xlabel("Predicted favourite win probability (bin midpoint)", fontsize = 16)
ax.set_ylabel("Observed favourite win proportion", fontsize = 16)
ax.set_title("'Favourite' win calibration: First Preference and YouGov MRP", fontsize = 16)

ax.legend(loc="lower right", fontsize = 16)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
    "mathtext.fontset": "cm",   # Computer / Latin Modern–style math
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

import matplotlib.font_manager as fm

print(any("Latin Modern" in f.name for f in fm.fontManager.ttflist))


# -----------------------
# Inputs
# -----------------------
mae_overall = 2.384141176895219

mae_party = {
   "ALP": 4.335079, "CYA": 0.768155,
    "FFPA": 1.261584, "GRN": 1.912917,
    "HMP": 1.127135,
    "IND":3.110154,"LNP": 3.546618, "LP": 3.650339,
   "NP": 4.396615, "ON": 1.975933,"OTH": 1.450949
}

# Top 10 parties by vote share (your definition)
top_vote_share = ["ALP", "LP", "NP", "LNP", "GRN", "ON", "CYA", "FFPA", "HMP"]

party_name_dict = {
    "ALP": "Labor", "LP": "Liberal", "NP": "National",
    "LNP": "Liberal National", "GRN": "Greens",
    "ON": "One Nation", "CYA": "Trumpet of Patriots",
    "FFPA": "Family First", "IND":"Independent","HMP": "Legalise Cannabis",
    "OTH": "Other parties (avg)"
}

# -----------------------
# Build DataFrame
# -----------------------
df = pd.DataFrame({"Party": mae_party.keys(), "MAE": mae_party.values()})


df["Label"] = df["Party"].map(party_name_dict)

# Order by MAE
df = df.sort_values("MAE", ascending=True)

# -----------------------
# Load party colours if available
# -----------------------
import matplotlib.colors as mcolors

def soften_color(color, amount=0.5):
    """
    Blend a color with white.
    amount=0.5 means 50% softer.
    """
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * rgb + amount * white)

colors = {}
csv_path = Path("Party Colours.csv")
if csv_path.exists():
    colour_df = pd.read_csv(csv_path)
    colour_df["Party"] = colour_df["Party"].astype(str).str.replace("’", "'", regex=False)
    colors = colour_df.set_index("Party")["Colour"].to_dict()

bar_colors = [soften_color(colors.get(p, "#888888"), amount=0.35) if p != "OTH" else soften_color("#888888", 0.35) for p in df["Label"]]

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(9, 5.8))

fig.patch.set_alpha(0)      # transparent figure background
ax.set_facecolor("none")    # transparent axes background

y = np.arange(len(df))
bars = ax.barh(y, df["MAE"], color=bar_colors)

ax.set_yticks(y)
ax.set_yticklabels(df["Label"], fontsize=11)

ax.set_xlabel("Mean Absolute Error (%)", fontsize=13)
ax.set_title(
    "Mean Absolute Error by party",
    fontsize=14
)

# Overall MAE reference line
ax.axvline(mae_overall, linestyle="--", linewidth=1.3, color = 'purple')
ax.text(
    mae_overall, len(df) - 6.5,
    f" Overall MAE = {mae_overall:.2f}",
    rotation=90, va="top", ha="right", color = 'purple', fontsize=14
)

# Value labels
for rect in bars:
    w = rect.get_width()
    ax.text(
        w + 0.05,
        rect.get_y() + rect.get_height() / 2,
        f"{w:.2f}",
        va="center",
        fontsize=10
    )

ax.tick_params(axis="x", labelsize=11)
plt.tight_layout()
plt.show()

plt.savefig(
    "mae_by_party.png",
    dpi=200,
    bbox_inches="tight",
    transparent=True
)