import pickle
from sim_transfer.sims.util import plot_rc_trajectory

ENCODE_ANGLE = False
RECORDING_NAME = f'recording_check_car2_sep29.pickle'

with open(RECORDING_NAME, 'rb') as f:
    rec_traj = pickle.load(f)
observations = rec_traj.observation[:200]
actions = rec_traj.action[:200]

plot_rc_trajectory(observations[..., :(7 if ENCODE_ANGLE else 6)], actions,
                   encode_angle=ENCODE_ANGLE, show=True)