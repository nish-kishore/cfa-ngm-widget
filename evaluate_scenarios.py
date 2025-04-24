import sys
import utils.dal as dal
import utils.data_munging
import polars as pl
import numpy as np
import utils.simulate_scenario as si

#read in the argument from the command line
if len(sys.argv) > 1:
    yaml_file = sys.argv[1]
else:
    print("You must define a yaml file to be read.")

#read in yaml file
param_set = dal.read_yaml_file(yaml_file)

#extract key population level parameters
pop = param_set['population_size']
n_vacc = param_set['n_vaccines']
n_groups = param_set['n_groups']

#extract key simulation parameters
G = param_set['generations']
sigdigs = param_set['sigdigs']

#generate parameters dataframe
params = pl.DataFrame(
    {
        "Group name": [group['group_name'] for group in param_set['groups'].values()],
        "Pop. size": [round(group['pop_frac']*pop) for group in param_set['groups'].values()],
        "No. vaccines": [round(group['vaccine_frac']*n_vacc) for group in param_set['groups'].values()],
        "Prob. severe": [group['prob_severe'] for group in param_set['groups'].values()],
    }
)

#develop list for the next generation matrix
ng = [group['infections_generated'] for group in param_set['groups'].values()]

#transpose matrix to match ngm function format
m_def_np = np.array([ [group_inf[i] for i in group_inf.keys()] for group_inf in ng]).T

#generate visual representation of the next generation matrix for error checking 
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

#generate numeric value for next generation matrix 
M_novax = M_df.drop("").to_numpy()

#extract general simulation parameters 
group_names = params["Group name"]
N = params["Pop. size"].to_numpy()
V = params["No. vaccines"].to_numpy()
p_severe = params["Prob. severe"].to_numpy()
VE = param_set['vaccine_efficacy']

#check if the # of groups and the # of transitions and # of names match
if not(len(group_names) == n_groups and M_novax.shape[0] == n_groups and M_novax.shape[1] == n_groups):
    raise ValueError(
        f"Number of groups ({len(group_names)}) does not match the number of transitions in the next generation matrix."
    )

#initiatlize a dictionary of scenarios to be simulated
scenarios = {}

#loop through scenarios and extract key values for simulation
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

#create a dictionary to hold the results of each simulation
results = {}
for x in scenarios.keys():
    results[x] = si.simulate_scenario(scenarios[x], distributions_as_percents=True)

#add title to each scenario result
for x in results.keys():
    results[x] = results[x].with_columns(
    pl.lit(scenarios[x]['scenario_title']).alias("Title")
)

#convert dictionary of results into list
results = [df for df in results.values()]

#concatenate list of dataframe into single dataframe
results = pl.concat(results)

#place title in the beginning of the dataframe
results = results.reorder(0, "Title")

#write out results
out_file_path = "results/"+param_set["evaluation_id"]+".csv"

results.write_csv(out_file_path)

print("Results from " + yaml_file + " written to " + out_file_path)