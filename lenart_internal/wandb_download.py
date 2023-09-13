import os

import wandb

api = wandb.Api()

# specify your project name
project_name = "trevenl/OfflineTrainingSimVsNoSimComparisonN4"
group_name = "horizon_len=100_diff_w=1.0_cost_w=0.005_num_offline_trans=1000_use_sim_prior=1_use_sim_norm_stats=1"

# Specify local directory to download the image
# specify the directory to check/create
local_dir = "saved_data"
dir_to_save = 'models'

# check if directory exists
if not os.path.exists(local_dir):
    # if directory does not exist, create it
    os.makedirs(local_dir)

runs = api.runs(project_name)
for run in runs:
    if run.group == group_name:
        print(run.name)
        run_name = run.name
        for file in run.files():
            if file.name.startswith(dir_to_save):
                file.download(replace=True, root=os.path.join(local_dir, run_name))
