import wandb
import pandas as pd

api = wandb.Api()

# Specify the username and project name
runs = api.runs("trevenl/OfflineRLSimulationWithDelayBIGEXP")

summary_list = []
config_list = []
name_list = []
group_list = []

for index, run in enumerate(runs):
    print(f"Run #{index+1}")
    # Run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # Run.config is the input metrics.  We remvoe special values that start with _
    config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})

    # Run.name is the name of the run
    name_list.append(run.name)

    # Run.group is the group of the run (useful if you are using group parameter in wandb.init)
    group_list.append(run.group)

# Create a pandas DataFrame
summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame(name_list, columns=["Name"])
group_df = pd.DataFrame(group_list, columns=["Group"])
all_df = pd.concat([name_df, group_df, config_df, summary_df], axis=1)

print(all_df.head())

# Save the DataFrame to a CSV file
all_df.to_csv('wandb_runs.csv', index=False)