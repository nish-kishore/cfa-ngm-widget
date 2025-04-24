import sys
import utils.dal as dal
import utils.data_munging
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
VE = param_set['vaccine_efficacy']
scenarios = {}
for x in param_set['scenarios'].keys():
    scenarios[x] = {
        "scenario_title": param_set['scenarios'][x]['title'],
        "group_names": group_names,
        "n_total": N.sum(),
        "pop_props": N / N.sum(),
        "M_novax": M_novax,
        "p_severe": p_severe,
        "n_vax": V,
        "ve": VE,
        "G": G,
    }

results = {}
for x in scenarios.keys():
    results[x] = si.simulate_scenario(scenarios[x], distributions_as_percents=True)


for x in results.keys():
    results[x] = results[x].with_columns(
    pl.lit(scenarios[x]['scenario_title']).alias("Title")
)
    
results = [df for df in results.values()]
results = pl.concat(results)

results = results.reorder(0, "Title")

out_file_path = "results/"+param_set["evaluation_id"]+".csv"

results.write_csv(out_file_path)

print("Results from " + yaml_file + " written to " + out_file_path)