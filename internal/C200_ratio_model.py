import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import time
import matplotlib.pyplot as plt

import sys
import pdb
import traceback


import statsmodels.api as sm




def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler


base_dir = Path.home() / "Necessary CSV Files"
os.chdir(base_dir)

os.chdir('/home/dania-freidgeim/Australian Election')
#multiple_INDs_df = pd.read_csv(f"{election_year}_Multiple_INDs_divs.csv", index_col=None)
C200_IND_splits = pd.read_csv(f"Independent_splits_multiple.csv", index_col=None).iloc[:-4,]

rows = []

for election_year in [2016,2019,2022,2025,2026]:
    C200_IND_splits_curr = C200_IND_splits.loc[C200_IND_splits['Election'].astype(int) < election_year,]


    IND_total = C200_IND_splits_curr['IND'] + C200_IND_splits_curr['IND_Other_1']
    Ratio = C200_IND_splits_curr['Ratio']


    y = np.log(Ratio / (1 - Ratio))

    X = np.log(IND_total)

    # --- 3. Add intercept ---
    X_design = sm.add_constant(X)

    model = sm.OLS(y, X_design).fit()


    print(model.summary())


    # parameters
    n_sims = 500
    sigma = np.std(model.resid)

    # simulate in logit space
    X_sim = np.random.choice(X.values, size=n_sims, replace=True)
    X_design_sim = sm.add_constant(X_sim)

    y_sim_logit = (
        model.params['const']
        + model.params[0] * X_sim
        + np.random.normal(0, sigma, n_sims)
    )

    # back-transform
    ratio_sim = 1 / (1 + np.exp(-y_sim_logit))
    ind_sim = np.exp(X_sim)

    # plot
    plt.scatter(ind_sim, ratio_sim, alpha=0.2, color = 'orange')
    plt.scatter(IND_total, Ratio, alpha=0.8)


    x_grid = np.linspace(0.01, max(IND_total), 200)
    y_pred_logit = model.predict(sm.add_constant(np.log(x_grid)))
    y_pred = 1 / (1 + np.exp(-y_pred_logit))

    beta0 = model.params['const']
    beta1 = model.params[0]
    sigma = np.std(model.resid)

    rows.append({
        "election_year": election_year,
        "beta0": beta0,
        "beta1": beta1,
        "sigma": sigma,
        "n_obs": len(C200_IND_splits_curr)
    })

    plt.plot(x_grid, y_pred, color='black')
    plt.show()




parameters = pd.DataFrame(rows)
parameters.loc[parameters['election_year']==2026,'election_year'] = 'Byelection'

parameters.to_csv("C200_ratio_lm_params.csv")
import pdb; pdb.set_trace()


