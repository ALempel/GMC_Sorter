#%%
Animal_prefix = 'F'
Animal = '1807'
Chunk_list = [1] # Empty list to do all chunks
probe = 'CNtch_H9'
file_struct = 'dat_t_s'
#%%
from sp_feature_ext import extract_spike_features
import numpy as np
import neo
import glob
import scipy.io
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

def do_chunk():
    start_time = star_end[2*(Chunk-1)]
    end_time = star_end[2*(Chunk-1)+1]

    ch_map = np.column_stack([xPosition, yPosition])
    out_folder = rf'Z:\Augusto\EphysData\SpikesV2\{Animal_prefix}{Animal}\{Chunk}'  # Output folder for plots and saved data

    time_interval = (start_time/1e6, end_time/1e6)
    extract_spike_features(exp_folder, file_struct, ch_map, out_folder, time_interval,make_plots=True)

    # Loas and concatenate all the properties files
    properties_files = glob.glob(rf'Z:\Augusto\EphysData\SpikesV2\{Animal_prefix}{Animal}\{Chunk}\spike_prop\*properties.npz')
    properties = np.concatenate([np.load(file)['Properties'] for file in properties_files])
    # Load the property titles
    prop_titles_files = glob.glob(rf'Z:\Augusto\EphysData\SpikesV2\{Animal_prefix}{Animal}\{Chunk}\spike_prop\property_titles.npz')
    prop_titles = np.load(prop_titles_files[0])['PropTitles']# Save the properties matriz as a matlab file
    scipy.io.savemat(rf'Z:\Augusto\EphysData\SpikesV2\{Animal_prefix}{Animal}\{Chunk}\properties.mat', {'properties': properties, 'prop_titles': prop_titles})

    import matplotlib.pyplot as pl
    # read binary ttl.dat file and timestamps from RawData_timestamps.dat
    ttl_dat_file = rf'Z:\Augusto\EphysData\DATFiles\{Animal_prefix}{Animal}\RawData_ttl.dat'
    with open(ttl_dat_file, 'rb') as f:
        ttl_dat = f.read()
    ttl_timestamps_file = rf'Z:\Augusto\EphysData\DATFiles\{Animal_prefix}{Animal}\RawData_timestamps.dat'
    with open(ttl_timestamps_file, 'rb') as f:
        ttl_timestamps = f.read()
    # conert to numpy array
    ttl_timestamps = np.frombuffer(ttl_timestamps, dtype=np.int64)
    ttl_dat = np.frombuffer(ttl_dat, dtype=np.int32)
    # Find the index of the start and end of the time interval in the ttl_timestamps
    start_idx = np.searchsorted(ttl_timestamps, start_time)
    end_idx = np.searchsorted(ttl_timestamps, end_time)
    # Extract the ttl_dat for the time interval
    ttl_dat = ttl_dat[start_idx:end_idx]
    # Compute the difference between consecutive ttl_dat
    ttl_dat_diff = np.diff(ttl_dat)
    # Get the timestamps of ttl diffs for whic diff%2 = 1
    ttl_dat_diff_idx = np.where((abs(ttl_dat_diff)%2) == 1)[0]
    ch_1_timestamps = ttl_timestamps[ttl_dat_diff_idx + start_idx]
    # Get the timestamps of ttl diffs for which diff//2 = 1
    ttl_dat_diff_idx = np.where((abs(ttl_dat_diff)//2) == 1)[0]
    ch_2_timestamps = ttl_timestamps[ttl_dat_diff_idx + start_idx]

    stim_star_times = ch_2_timestamps[::2]
    WF_times = ch_1_timestamps[::2]
    # Save the stim_star_times and WF_times as one matlab file
    scipy.io.savemat(rf'Z:\Augusto\EphysData\SpikesV2\{Animal_prefix}{Animal}\{Chunk}\stim_and_WF_times.mat', {'stim_star_times': stim_star_times/1e6, 'WF_times': WF_times/1e6})

exp_folder = rf'Z:\Augusto\EphysData\DATFiles\{Animal_prefix}{Animal}'  # Use raw string for Windows paths
reader = neo.io.NeuralynxIO(dirname=exp_folder)
last_event_ch = reader.event_channels_count()-1
star_end=reader.get_event_timestamps(0,0,last_event_ch)[0]
if len(Chunk_list) == 0:
    total_chunks = len(star_end)//2
    Chunk_list = list(range(1,total_chunks+1))
for Chunk in Chunk_list:
    print(f'Processing chunk {Chunk} of {total_chunks}')
    do_chunk()
# %%
