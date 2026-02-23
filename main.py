#%%
from sp_feature_ext import extract_spike_features
import numpy as np
import neo
#%% Settings
Animal = '2483'
Chunk = 13
probe = 'L32'
file_struct = 'dat_t_s'
#%%
# Probe
match probe:
    case 'CNtch_H9':
        xPosition = np.tile([0, 30], 32)  # Repeat [0, 30] 32 times = 64 elements
        yPosition = np.arange(64) * 23    # [0, 23, 46, 69, ..., 1449]
    case 'NN_Poly2':
        yPosition = np.arange(64)*23    # [0, 23, 46, 69, ..., 1449]  64 elements
        xPosition = np.tile([0, 30], 32)  # [0, 30, 0, 30, ..., 0, 30] 64 elements
    case 'M64':
        yPosition = np.arange(64) * 25    # [0, 25, 50, 75, ..., 1500]
        xPosition = np.tile([0, 25], 32)  # [0, 25, 0, 25, ..., 0, 25] 64 elements
    case 'L32':
        yPosition=np.arange(32)*30;
        xPosition=np.ones((32,1));

exp_folder = rf'Z:\Augusto\EphysData\DATFiles\F{Animal}'  # Use raw string for Windows paths
reader = neo.io.NeuralynxIO(dirname=exp_folder)
last_event_ch = reader.event_channels_count()-1
star_end=reader.get_event_timestamps(0,0,last_event_ch)[0]
start_time = star_end[2*(Chunk-1)]
end_time = star_end[2*(Chunk-1)+1]

ch_map = np.column_stack([xPosition, yPosition])
out_folder = rf'Z:\Augusto\EphysData\SpikesV2\F{Animal}\{Chunk}'  # Output folder for plots and saved data

time_interval = (start_time/1e6, end_time/1e6)
extract_spike_features(exp_folder, file_struct, ch_map, out_folder, time_interval,make_plots=True)

# %%
