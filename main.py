import sys
import utils.dal as dal
import polars as pl
import numpy as np
import sandbox_internal as si


if len(sys.argv) > 1:
    yaml_file = sys.argv[1]
else:
    print("You must define a yaml file to be read.")


param_set = dal.read_yaml_file(yaml_file)
pop = param_set['population_size']
n_vacc = param_set['n_vaccines']

params = pl.DataFrame(
    {
        "Group name": [group['group_name'] for group in param_set['groups'].values()],
        "Pop. size": [round(group['pop_frac']*pop) for group in param_set['groups'].values()],
        "No. vaccines": [round(group['vaccine_frac']*n_vacc) for group in param_set['groups'].values()],
        "Prob. severe": [group['prob_severe'] for group in param_set['groups'].values()],
    }
)

ng = [group['infections_generated'] for group in param_set['groups'].values()]

m_def_np = np.array([ [group_inf[i] for i in group_inf.keys()] for group_inf in ng]).T

M_df = (
    pl.DataFrame(
        {
            f"from {grp}": m_def_np[:, i]
            for i, grp in enumerate(params["Group name"])
        }
    )
    .with_columns(pl.Series("", [f"to {grp}" for grp in params["Group name"]]))
    .select(["", *[f"from {grp}" for grp in params["Group name"]]])
)

M_novax = M_df.drop("").to_numpy()
G = 10
sigdigs = 2

group_names = params["Group name"]
N = params["Pop. size"].to_numpy()
V = params["No. vaccines"].to_numpy()
p_severe = params["Prob. severe"].to_numpy()
