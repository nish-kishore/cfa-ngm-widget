#sandbox to run

import importlib.metadata

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

import sandbox_internal as si

params_default = pl.DataFrame(
    {
        "Group name": ["Core", "Children", "Adults"],
        "Pop. size": np.array([0.05, 0.45, 0.5]) * int(1e7),
        "No. vaccines": [2.5e5, 2.5e5, 5e5],
        "Prob. severe": [0.02, 0.06, 0.02],
    }
)

params = params_default

VE = 0.74

m_def_np = np.array([[3.0, 0.0, 0.2], [0.10, 1.0, 0.5], [0.25, 1.0, 1.5]])


M_default = (
    pl.DataFrame(
        {
            f"from {grp}": m_def_np[:, i]
            for i, grp in enumerate(params["Group name"])
        }
    )
    .with_columns(pl.Series("", [f"to {grp}" for grp in params["Group name"]]))
    .select(["", *[f"from {grp}" for grp in params["Group name"]]])
)

M_df = st.data_editor(M_default, disabled=["to"], hide_index=True)
M_novax = M_df.drop("").to_numpy()
G = 10
sigdigs = 2

group_names = params["Group name"]
N = params["Pop. size"].to_numpy()
V = params["No. vaccines"].to_numpy()
p_severe = params["Prob. severe"].to_numpy()

scenario = {
        "scenario_title": "Scenario: Vaccination",
        "group_names": group_names,
        "n_total": N.sum(),
        "pop_props": N / N.sum(),
        "M_novax": M_novax,
        "p_severe": p_severe,
        "n_vax": V,
        "ve": VE,
        "G": G,
    }

counterfactual = scenario.copy()
counterfactual["scenario_title"] = "Scenario: Counterfactual (no vaccination)"
counterfactual["n_vax"] = 0 * V
counterfactual["p_vax"] = 0.0 * V
scenarios = [
    scenario,
    counterfactual,
]

#setting to default scenario
params = scenarios[0]
groups = params["Group name"]
display=[
        "infections_",
        "deaths_per_prior_infection_",
        "deaths_after_G_generations_",
    ]
display_names=[
        "Percent of infections",
        "Severe infections per prior infection",
        "Severe infections after G generations",
    ]

p_vax = params["n_vax"] / (params["n_total"] * params["pop_props"])

scenario1 = si.simulate_scenario(scenarios[0], distributions_as_percents=True)
counterfactual = si.simulate_scenario(scenarios[1], distributions_as_percents=True)