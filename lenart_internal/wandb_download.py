import os

import wandb

api = wandb.Api()

# specify your project name
project_name = "sukhijab/OfflineRLLikelihood0_test"
group_name = "use_sim_prior=0_use_grey_box=0_high_fidelity=0_num_offline_data=2500_share_of_x0s=0.5_0.5"

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
