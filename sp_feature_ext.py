import numpy as np
import matplotlib.pyplot as plt
from raw_data import raw_data
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Try to use PyTorch for GPU filtering (FFT-based, mainstream approach)
try:
    import torch
    from helper_functions import _bandpass_fft_torch
    from extract_spike_features_func import extract_spike_properties_GPU
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"GPU available: {DEVICE}")
    else:
        DEVICE = torch.device('cpu')
        print(f"CPU available: {DEVICE}")
        TORCH_AVAILABLE = False
except ImportError:
    print("PyTorch not available")
    TORCH_AVAILABLE = False
    DEVICE = None


def filter_data_GPU(batch_data, sample_rate, low_fr, high_fr, sos_bandpass):
    """
    Filter data using GPU FFT-based bandpass filter.
    
    Parameters
    ----------
    batch_data : np.ndarray
        Input data, shape (n_channels, n_samples)
    sample_rate : float
        Sample rate in Hz
    low_fr : float
        Low frequency cutoff in Hz
    high_fr : float
        High frequency cutoff in Hz
    sos_bandpass : array-like or None
        SOS filter coefficients (not used for GPU, but kept for API consistency)
    
    Returns
    -------
    np.ndarray
        Filtered data, shape (n_channels, n_samples)
    """
    batch_data_torch = torch.from_numpy(batch_data.astype(np.float32)).to(DEVICE)
    filtered_data_torch = _bandpass_fft_torch(
        batch_data_torch, 
        sample_rate, 
        low_fr, 
        high_fr
    )
    filtered_data_cpu = filtered_data_torch.cpu().numpy()
    # Delete GPU tensors to free GPU memory
    del batch_data_torch, filtered_data_torch
    torch.cuda.empty_cache()
    return filtered_data_cpu


def filter_data_CPU(batch_data, sos_bandpass):
    """
    Filter data using CPU scipy bandpass filter.
    
    Parameters
    ----------
    batch_data : np.ndarray
        Input data, shape (n_channels, n_samples)
    sos_bandpass : array-like or None
        SOS filter coefficients
    
    Returns
    -------
    np.ndarray
        Filtered data, shape (n_channels, n_samples)
    """
    if sos_bandpass is not None:
        filtered_data_cpu = signal.sosfiltfilt(sos_bandpass, batch_data.astype(np.float32), axis=1)
    else:
        filtered_data_cpu = batch_data.astype(np.float32)
    return filtered_data_cpu


def denoise_data_GPU(filtered_data_cpu, noise_neighbors, bad_ch, batch_idx, noise_signals, spike_median_threshold):
    """
    Denoise data using GPU PyTorch operations.
    
    Parameters
    ----------
    filtered_data_cpu : np.ndarray
        Filtered data, shape (n_channels, n_samples)
    noise_neighbors : dict
        Dictionary mapping channel index to list of neighbor channel indices
    bad_ch : set
        Set of bad channel indices
    batch_idx : int
        Batch index (0 for first batch)
    noise_signals : np.ndarray or None
        Array to store noise signals for first batch, shape (n_channels, n_samples)
    spike_median_threshold : float
        Multiplier for median abs (over first 10k samples) to form clip threshold; neighbor signals are clipped before PCA.
    
    Returns
    -------
    torch.Tensor
        Denoised data on GPU, shape (n_channels, n_samples)
    """
    n_channels, n_samples = filtered_data_cpu.shape
    
    # Move data to GPU
    filtered_data_torch = torch.from_numpy(filtered_data_cpu).to(DEVICE)
    denoised_data_torch = torch.zeros_like(filtered_data_torch)
    
    for ch_idx in range(n_channels):
        # Skip bad channels - fill with zeros
        if ch_idx in bad_ch:
            denoised_data_torch[ch_idx, :] = 0.0
            if batch_idx == 0 and noise_signals is not None:
                noise_signals[ch_idx, :] = 0.0
            continue
        
        # Get neighboring channels for this channel (exclude bad channels)
        neighbors = [n for n in noise_neighbors[ch_idx] if n not in bad_ch]
        
        if len(neighbors) > 0:
            # Stack current channel and its neighbors: shape (n_neighbors+1, n_samples)
            neighbor_indices = [ch_idx] + neighbors
            neighbor_signals = filtered_data_torch[neighbor_indices, :]  # Shape: (n_neighbors+1, n_samples)
            # Clip per channel by +/- (median abs over 10k evenly spaced samples * spike_median_threshold) per neighbor channel
            n_use = min(10000, n_samples)
            idx = torch.linspace(0, n_samples - 1, n_use, device=neighbor_signals.device).long()
            med_per_ch = torch.median(torch.abs(neighbor_signals[:, idx]), dim=1, keepdim=True)[0]  # (n_neighbors+1, 1)
            thresh = (med_per_ch * spike_median_threshold).clamp(min=1e-9)
            neighbor_signals = torch.clamp(neighbor_signals, -thresh, thresh)
            
            # Transpose to (n_samples, n_channels) for PCA
            neighbor_signals_T = neighbor_signals.T  # Shape: (n_samples, n_neighbors+1)
            
            # Center the data (subtract mean)
            neighbor_signals_T_centered = neighbor_signals_T - neighbor_signals_T.mean(dim=0, keepdim=True)
            
            # Extract first principal component using SVD (faster on GPU)
            U, S, V = torch.linalg.svd(neighbor_signals_T_centered, full_matrices=False)
            # First principal component is first column of U scaled by first singular value
            shared_noise = (U[:, 0:1] * S[0]).flatten()  # Shape: (n_samples,)
            
            # Regress shared noise out of the channel using least squares
            channel_signal = filtered_data_torch[ch_idx, :]  # Shape: (n_samples,)
            # Least squares: beta = (X^T X)^(-1) X^T y
            X = shared_noise.reshape(-1, 1)
            y = channel_signal
            beta = torch.linalg.lstsq(X, y.reshape(-1, 1)).solution.flatten()
            predicted_noise = (X @ beta.reshape(-1, 1)).flatten()
            denoised_data_torch[ch_idx, :] = channel_signal - predicted_noise
            
            # Store noise signal for first batch
            if batch_idx == 0 and noise_signals is not None:
                noise_signals[ch_idx, :] = predicted_noise.cpu().numpy()
        else:
            # No neighbors, keep original signal
            denoised_data_torch[ch_idx, :] = filtered_data_torch[ch_idx, :]
            if batch_idx == 0 and noise_signals is not None:
                noise_signals[ch_idx, :] = 0.0  # No noise removed
    
    return denoised_data_torch, filtered_data_torch


def denoise_data_CPU(filtered_data_cpu, noise_neighbors, bad_ch, batch_idx, noise_signals, spike_median_threshold):
    """
    Denoise data using CPU sklearn operations.
    
    Parameters
    ----------
    filtered_data_cpu : np.ndarray
        Filtered data, shape (n_channels, n_samples)
    noise_neighbors : dict
        Dictionary mapping channel index to list of neighbor channel indices
    bad_ch : set
        Set of bad channel indices
    batch_idx : int
        Batch index (0 for first batch)
    noise_signals : np.ndarray or None
        Array to store noise signals for first batch, shape (n_channels, n_samples)
    spike_median_threshold : float
        Multiplier for median abs (over first 10k samples) to form clip threshold; neighbor signals are clipped before PCA.
    
    Returns
    -------
    np.ndarray
        Denoised data, shape (n_channels, n_samples)
    """
    n_channels, n_samples = filtered_data_cpu.shape
    denoised_data = np.zeros_like(filtered_data_cpu)
    
    for ch_idx in range(n_channels):
        # Skip bad channels - fill with zeros
        if ch_idx in bad_ch:
            denoised_data[ch_idx, :] = 0.0
            if batch_idx == 0 and noise_signals is not None:
                noise_signals[ch_idx, :] = 0.0
            continue
        
        # Get neighboring channels for this channel (exclude bad channels)
        neighbors = [n for n in noise_neighbors[ch_idx] if n not in bad_ch]
        
        if len(neighbors) > 0:
            # Stack current channel and its neighbors: shape (n_neighbors+1, n_samples)
            neighbor_signals = np.vstack([filtered_data_cpu[ch_idx:ch_idx+1, :], 
                                         filtered_data_cpu[neighbors, :]])
            # Clip per channel by +/- (median abs over 10k evenly spaced samples * spike_median_threshold) per neighbor channel
            n_use = min(10000, n_samples)
            idx = np.linspace(0, n_samples - 1, n_use, dtype=np.intp)
            med_per_ch = np.median(np.abs(neighbor_signals[:, idx]), axis=1, keepdims=True)
            thresh = np.maximum(med_per_ch * spike_median_threshold, 1e-9)
            neighbor_signals = np.clip(neighbor_signals, -thresh, thresh)
            
            # Transpose to (n_samples, n_channels) for PCA
            neighbor_signals_T = neighbor_signals.T
            
            # Extract first principal component (shared noise)
            pca = PCA(n_components=1)
            shared_noise = pca.fit_transform(neighbor_signals_T).flatten()  # Shape: (n_samples,)
            
            # Regress shared noise out of the channel
            reg = LinearRegression()
            channel_signal = filtered_data_cpu[ch_idx, :].reshape(-1, 1)
            reg.fit(shared_noise.reshape(-1, 1), channel_signal)
            predicted_noise = reg.predict(shared_noise.reshape(-1, 1)).flatten()
            denoised_data[ch_idx, :] = filtered_data_cpu[ch_idx, :] - predicted_noise
            
            # Store noise signal for first batch
            if batch_idx == 0 and noise_signals is not None:
                noise_signals[ch_idx, :] = predicted_noise
        else:
            # No neighbors, keep original signal
            denoised_data[ch_idx, :] = filtered_data_cpu[ch_idx, :]
            if batch_idx == 0 and noise_signals is not None:
                noise_signals[ch_idx, :] = 0.0  # No noise removed
    
    return denoised_data


def find_spikes_GPU(denoised_data_inverted_torch, refractory_neighbors, fix_rect_mask, n_temporal_samples, n_channels, n_samples, spike_median_threshold):
    """
    Find local minima (spikes) using GPU PyTorch operations.
    
    Parameters
    ----------
    denoised_data_inverted_torch : torch.Tensor
        Inverted denoised data on GPU, shape (n_channels, n_samples) (data should be negated: -denoised_data)
    refractory_neighbors : dict
        Dictionary mapping channel index to list of neighbor channel indices
    fix_rect_mask : bool
        Whether to use fixed rectangular mask
    n_temporal_samples : int
        Number of temporal samples in the mask
    n_channels : int
        Number of channels
    n_samples : int
        Number of time samples
    spike_median_threshold : float
        Threshold multiplier for median absolute value (spikes must be below -median * threshold)
    
    Returns
    -------
    list
        List of (channel, sample) tuples for local minima
    """
    # Find local minima using refractory neighbors (keep on GPU for efficiency)
    # Use max_pool2d: once if fix_rect_mask, otherwise per-channel
    # Data is already inverted, so we can use it directly
    if fix_rect_mask:
        # Use efficient max_pool2d for rectangular kernel on whole data
        # For minimum filter, use -max(-x) trick: data is already inverted
        data_inverted = denoised_data_inverted_torch
        
        # Get kernel size from channel with maximum neighbors
        max_neighbors = 0
        max_neighbors_ch = 0
        for ch_idx in range(n_channels):
            if len(refractory_neighbors[ch_idx]) > max_neighbors:
                max_neighbors = len(refractory_neighbors[ch_idx])
                max_neighbors_ch = ch_idx
        
        all_channels = sorted([max_neighbors_ch] + refractory_neighbors[max_neighbors_ch])
        n_mask_channels = len(all_channels)
        
        # Pad data for boundary handling
        pad_temporal = n_temporal_samples // 2
        pad_spatial = n_mask_channels // 2
        
        data_4d = data_inverted.unsqueeze(0).unsqueeze(0)  # (1, 1, n_channels, n_samples)
        data_padded = torch.nn.functional.pad(
            data_4d,
            (pad_temporal, pad_temporal, pad_spatial, pad_spatial),
            mode='constant',
            value=-float('inf')
        )
        # Apply max_pool2d with rectangular kernel
        filtered_inverted = torch.nn.functional.max_pool2d(
            data_padded,
            kernel_size=(n_mask_channels, n_temporal_samples),
            stride=1,
            padding=0
        )
        filtered_inverted = filtered_inverted.squeeze(0).squeeze(0)  # (n_channels, n_samples)
    else:
        # Process per channel (works for both fix_mask_map True/False)
        # For minimum filter, use -max(-x) trick: data is already inverted
        data_inverted = denoised_data_inverted_torch
        
        # Pad data for boundary handling (use max temporal size)
        pad_temporal = n_temporal_samples // 2
        # For spatial padding, we'll pad enough for any channel's neighbors
        max_neighbors = max(len(refractory_neighbors[ch_idx]) for ch_idx in range(n_channels))
        pad_spatial = max_neighbors // 2
        
        filtered_inverted = torch.full((n_channels, n_samples), -float('inf'), device=DEVICE, dtype=data_inverted.dtype)
        
        # Pad data once
        data_padded = torch.nn.functional.pad(
            data_inverted.unsqueeze(0).unsqueeze(0),
            (pad_temporal, pad_temporal, pad_spatial, pad_spatial),
            mode='constant',
            value=-float('inf')
        )
        data_padded = data_padded.squeeze(0).squeeze(0)  # (n_channels + 2*pad_spatial, n_samples + 2*pad_temporal)
        
        # For each channel, find neighbors and apply max_pool2d
        for ch_idx in range(n_channels):
            # Get neighbors for this channel (including itself)
            neighbors = refractory_neighbors[ch_idx]
            neighbor_channels = sorted([ch_idx] + neighbors)
            n_neighbor_channels = len(neighbor_channels)
            
            if n_neighbor_channels == 0:
                continue
            
            # Extract neighbor channels from padded data
            # Adjust indices for spatial padding offset
            neighbor_channels_padded = [ch + pad_spatial for ch in neighbor_channels]
            neighbor_data = data_padded[neighbor_channels_padded, :]  # (n_neighbor_channels, n_samples + 2*pad_temporal)
            
            # Reshape for max_pool2d: (1, 1, n_neighbor_channels, n_samples + 2*pad_temporal)
            neighbor_data_4d = neighbor_data.unsqueeze(0).unsqueeze(0)
            
            # Apply max_pool2d with rectangular kernel (n_neighbor_channels x n_temporal_samples)
            # This finds max over spatiotemporal block for each time position
            ch_filtered = torch.nn.functional.max_pool2d(
                neighbor_data_4d,
                kernel_size=(n_neighbor_channels, n_temporal_samples),
                stride=1,
                padding=0
            )
            ch_filtered = ch_filtered.squeeze(0).squeeze(0)  # (1, n_samples)
            
            # Store result for this channel (extract valid region, excluding temporal padding)
            filtered_inverted[ch_idx, :] = ch_filtered[0, pad_temporal:pad_temporal+n_samples]
        
    # Compute median absolute value per channel (across time)
    # Shape: (n_channels, 1) - broadcastable to (n_channels, n_samples)
    # Note: denoised_data_inverted_torch is -denoised_data, so abs is the same
    abs_denoised = torch.abs(denoised_data_inverted_torch)
    median_abs_per_channel = torch.median(abs_denoised, dim=1, keepdim=True)[0]  # Shape: (n_channels, 1)
    threshold_per_channel = median_abs_per_channel * spike_median_threshold  # Shape: (n_channels, 1) - positive threshold
    
    # Check where data equals the local minimum (strict equality) AND is above threshold
    # Note: denoised_data_inverted_torch = -denoised_data; we check inverted data == filtered (max of inverted = min of original)
    # and inverted data > positive threshold (original < -threshold)
    is_peak = (data_inverted == filtered_inverted) & (denoised_data_inverted_torch > threshold_per_channel)
    
    pad_temporal = n_temporal_samples // 2
    nRef = pad_temporal

    # 0. Extract peak indices before masking padding (needed for dedup)
    peak_indices = torch.nonzero(is_peak, as_tuple=False)

    if peak_indices.shape[0] > 0:
        dev = is_peak.device
        # Refractory channel set (same for all when using fixed mask logic for dedup)
        max_neighbors = max(len(refractory_neighbors[ch_idx]) for ch_idx in range(n_channels))
        max_neighbors_ch = next(c for c in range(n_channels) if len(refractory_neighbors[c]) == max_neighbors)
        all_channels = sorted([max_neighbors_ch] + refractory_neighbors[max_neighbors_ch])
        n_refr_ch = len(all_channels)

        # 1. Tensor (n_peaks, n_refr_ch, nRef): is_peak at (all_channels, t-k) for k=1..nRef; out-of-bounds -> 0
        t_peak = peak_indices[:, 1]
        k_offsets = torch.arange(1, nRef + 1, device=dev, dtype=torch.long)
        t_idx = (t_peak[:, None, None] - k_offsets[None, None, :]).clamp(min=0)
        valid_t = (t_peak[:, None, None] - k_offsets[None, None, :] >= 0)
        ch_idx = torch.tensor(all_channels, device=dev, dtype=torch.long).view(1, -1, 1).expand(peak_indices.shape[0], n_refr_ch, nRef)
        prev_vals = is_peak[ch_idx, t_idx]
        prev_vals = torch.where(valid_t, prev_vals, torch.zeros_like(prev_vals))

        # 2–3. late_peaks: any preceding peak in refr window
        late_peaks = prev_vals.any(dim=(1, 2))

        # 4. deep_peaks: any more superficial channel (index < ch) had a peak at same time
        is_peak_at_peaks = is_peak[:, peak_indices[:, 1]].T
        ch_peak = peak_indices[:, 0]
        mask_superficial = torch.arange(n_channels, device=dev, dtype=torch.long)[None, :] < ch_peak[:, None]
        deep_peaks = (is_peak_at_peaks & mask_superficial).any(dim=1)

        # 5. Keep peaks that are not late and not deep
        keep = ~(late_peaks | deep_peaks)
        peak_indices = peak_indices[keep]

        # 6. Remove peaks in temporal padding
        keep_pad = (peak_indices[:, 1] >= pad_temporal) & (peak_indices[:, 1] < n_samples - pad_temporal)
        peak_indices = peak_indices[keep_pad]

    local_minima_indices_cpu = [(int(peak_indices[i, 0].item()), int(peak_indices[i, 1].item())) for i in range(peak_indices.shape[0])]

    # Clean up all GPU/CPU tensors except denoised_data_inverted_torch
    del data_inverted, filtered_inverted, is_peak, peak_indices, abs_denoised, median_abs_per_channel, threshold_per_channel
    if 'data_padded' in locals():
        del data_padded
    if 'data_4d' in locals():
        del data_4d
    if 'neighbor_data' in locals():
        del neighbor_data
    if 'neighbor_data_4d' in locals():
        del neighbor_data_4d
    if 'ch_filtered' in locals():
        del ch_filtered
    torch.cuda.empty_cache()  # Clear GPU cache
    
    return local_minima_indices_cpu


def find_spikes_CPU(denoised_data, refractory_neighbors, fix_rect_mask, n_temporal_samples, n_channels, n_samples):
    """
    Find local minima (spikes) using CPU operations.
    
    Parameters
    ----------
    denoised_data : np.ndarray
        Denoised data, shape (n_channels, n_samples)
    refractory_neighbors : dict
        Dictionary mapping channel index to list of neighbor channel indices
    fix_rect_mask : bool
        Whether to use fixed rectangular mask
    n_temporal_samples : int
        Number of temporal samples in the mask
    n_channels : int
        Number of channels
    n_samples : int
        Number of time samples
    
    Returns
    -------
    list
        List of (channel, sample) tuples for local minima
    """
    # TODO: Implement CPU version of local minima finding
    # For now, return empty list
    # This would need scipy.ndimage.maximum_filter or equivalent CPU implementation
    return []


def extract_spike_features(exp_folder, file_struct, ch_map, out_folder, time_interval=(0, float('inf')), make_plots=True, **kwargs):
    """
    Extract spike features from experimental data.
    
    Parameters
    ----------
    exp_folder : str
        Path to the experimental folder containing the data files.
    file_struct : str
        File structure identifier (e.g., 'NPx', 'dat_t_s').
    ch_map : array-like
        Array of (x, y) spatial positions for each channel.
        Shape should be (n_channels, 2) where each row is [x, y].
    out_folder : str
        Path to the output folder where plots and other saved data will be stored.
    time_interval : tuple, optional
        Time interval to process, default=(0, inf).
        Tuple of (start_time, end_time) in seconds.
    make_plots : bool, optional
        Whether to create and save plots, default=True.
    **kwargs : dict, optional
        Optional parameters with default values:
        - refr_time : float, default=0.75
            Refractory period time in milliseconds.
        - refr_space : float, default=79
            Refractory period space in micrometers.
        - wvf_space : float, default=79
            Waveform space in micrometers.
        - noise_space : float, default=100
            Noise space in micrometers.
        - low_fr : float, default=250
            Low frequency threshold in Hz.
        - high_fr : float, default=5000
            High frequency threshold in Hz.
        - overlap_time : float, default=2
            Overlap time in milliseconds.
        - LP_filter : float, default=5000
            Low-pass filter cutoff frequency in Hz.
        - data_buffer_size : float, default=512
            Data buffer size in MB for batch processing.
        - init_time_delay : float, default=0.5
            Initial time delay in seconds to add to the start of the time interval.
        - fix_mask_map : bool, default=True
            If True, compute a fixed spatial mask for refractory neighbors that works for all channels.
            If False, use channel-specific masks (not yet implemented).
    
    Returns
    -------
    None
        Function processes data and saves plots.
    """
    # Set default values
    params = {
        'init_time_delay': 0.5,      # in seconds
        'spike_median_threshold': 7,      # in medians
        'refr_time': 0.75,      # in msec
        'refr_space': 100,        # in um
        'wvf_space': 100,         # in um
        'wvf_time': 0.75,         # in msec
        'noise_space': 201,      # in um
        'low_fr': 250,
        'high_fr': 5000,
        'overlap_time': 2,       # in msec
        'LP_filter': 5000,       # in Hz
        'data_buffer_size': 128,  # in MB
        'fix_mask_map': True,    # Use fixed spatial mask for refractory neighbors
        'bad_ch_fill_distance': 50.0,  # in um; replace bad channel trace with mean of valid channels within this distance
        'noise_median_threshold': 3,      # in medians
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    # Convert ch_map to numpy array if needed
    ch_map = np.array(ch_map)
    n_channels = ch_map.shape[0]
    
    # Create neighbor channel dictionaries
    # Calculate pairwise distances between all channels
    # ch_map shape: (n_channels, 2) where each row is [x, y]
    distances = np.sqrt(((ch_map[:, np.newaxis, :] - ch_map[np.newaxis, :, :]) ** 2).sum(axis=2))
    # distances[i, j] = distance between channel i and channel j
    
    # Create noise_neighbors dictionary
    noise_neighbors = {}
    for ch_idx in range(n_channels):
        # Find channels within noise_space distance (excluding self)
        neighbors = np.where((distances[ch_idx, :] <= params['noise_space']) & 
                              (distances[ch_idx, :] > 0))[0].tolist()
        noise_neighbors[ch_idx] = neighbors
    
    # Create refractory_neighbors dictionary
    refractory_neighbors = {}
    for ch_idx in range(n_channels):
        # Find channels within refr_space distance (excluding self)
        neighbors = np.where((distances[ch_idx, :] <= params['refr_space']) & 
                             (distances[ch_idx, :] > 0))[0].tolist()
        refractory_neighbors[ch_idx] = neighbors

    # Create wvf_neighbors dictionary
    wvf_neighbors = {}
    for ch_idx in range(n_channels):
        # Find channels within wvf_space distance (excluding self)
        neighbors = np.where((distances[ch_idx, :] <= params['wvf_space']) & 
                             (distances[ch_idx, :] > 0))[0].tolist()
        wvf_neighbors[ch_idx] = neighbors
    
    # Create raw_data object
    data_obj = raw_data(exp_folder, file_struct)
    
    # Get metadata
    n_saved_chans, n_time_samples, sample_rate = data_obj.meta_info()
    
    # Compute temporal window size for refractory period
    # refr_time is in msec, convert to samples
    refr_time_seconds = params['refr_time'] / 1000.0  # Convert msec to seconds
    refr_time_samples = int(np.round(refr_time_seconds * sample_rate))
    # Total temporal samples: +/- refr_time_samples around center sample
    n_temporal_samples = 2 * refr_time_samples + 1
    
    # Check if we can use a fixed rectangular mask (fix_mask_map AND continuous channel range)
    fix_rect_mask = False
    if params['fix_mask_map']:
        # Find channel with maximum number of refractory neighbors
        max_neighbors = 0
        max_neighbors_ch = 0
        for ch_idx in range(n_channels):
            if len(refractory_neighbors[ch_idx]) > max_neighbors:
                max_neighbors = len(refractory_neighbors[ch_idx])
                max_neighbors_ch = ch_idx
        
        # Get all channels (channel itself + neighbors)
        all_channels = sorted([max_neighbors_ch] + refractory_neighbors[max_neighbors_ch])
        
        # Check if channels form a continuous range
        # Continuous means: [min, min+1, min+2, ..., max] with no gaps
        if len(all_channels) > 0:
            min_ch = all_channels[0]
            max_ch = all_channels[-1]
            expected_range = list(range(min_ch, max_ch + 1))
            fix_rect_mask = (all_channels == expected_range)
    
    # Calculate actual time interval (handle infinity)
    if time_interval[1] == float('inf'):
        # Calculate total recording time from samples
        total_time = n_time_samples / sample_rate
        time_interval = (time_interval[0], total_time)
    
    start_time, end_time = time_interval
    # Add initial time delay to start time
    start_time = start_time + params['init_time_delay']
    total_duration = end_time - start_time
    
    # Calculate batch size in seconds
    # data_buffer_size is in MB, convert to bytes
    buffer_bytes = params['data_buffer_size'] * 1024 * 1024
    # Get bytes per sample per channel from raw_data object
    bytes_per_sample_per_channel = data_obj.bytes_per_sample_per_channel
    bytes_per_time_sample = n_saved_chans * bytes_per_sample_per_channel
    # Number of time samples that fit in buffer
    samples_per_batch = int(buffer_bytes / bytes_per_time_sample)
    # Time duration per batch (in seconds)
    batch_duration = samples_per_batch / sample_rate
    
    # Account for overlap - each batch extends by overlap_seconds
    overlap_seconds = params['overlap_time'] / 1000.0  # Convert ms to seconds
    # Each batch includes overlap time, so actual batch duration is longer
    actual_batch_duration = batch_duration + overlap_seconds
    # Step size between batch starts (without overlap)
    batch_step = batch_duration
    
    # Calculate number of batches
    if batch_step <= 0:
        raise ValueError("Batch duration must be greater than zero")
    
    n_batches = int(np.ceil(total_duration / batch_step))
    
    # Calculate time intervals for each batch
    batch_intervals = []
    for i in range(n_batches):
        batch_start = start_time + i * batch_step
        batch_end = min(batch_start + actual_batch_duration, end_time)
        batch_intervals.append((batch_start, batch_end))
    
    # Use actual number of batches created (may differ from n_batches due to rounding)
    n_batches_actual = len(batch_intervals)
    
    # Create output directory
    os.makedirs(out_folder, exist_ok=True)
    # Create subfolder for spike properties (always needed)
    spike_prop_dir = os.path.join(out_folder, 'spike_prop')
    os.makedirs(spike_prop_dir, exist_ok=True)
    # Create subfolders for plots only if make_plots is True
    if make_plots:
        plot_dir = os.path.join(out_folder, 'filt_data_plots')
        os.makedirs(plot_dir, exist_ok=True)
        den_plot_dir = os.path.join(out_folder, 'den_data_plots')
        os.makedirs(den_plot_dir, exist_ok=True)
        noise_plot_dir = os.path.join(out_folder, 'noise_signal_plots')
        os.makedirs(noise_plot_dir, exist_ok=True)
    
    # Design filters
    nyquist = sample_rate / 2.0
    low_cutoff = params['low_fr'] / nyquist
    high_cutoff = params['high_fr'] / nyquist
    
    # Bandpass filter: low_fr to high_fr
    if low_cutoff < 1.0 and high_cutoff < 1.0:
        sos_bandpass = signal.butter(4, [low_cutoff, high_cutoff], btype='band', output='sos')
    else:
        sos_bandpass = None
    
    # Calculate wvf_samples from wvf_time (convert time window to samples)
    # wvf_time is in milliseconds, convert to seconds then to samples
    wvf_samples = int(np.round(sample_rate * params['wvf_time'] / 1000.0))
    # Also calculate wvf_time in seconds for timestamp-based filtering
    wvf_time_seconds = params['wvf_time'] / 1000.0
    # Calculate overlap_time in seconds
    overlap_time_seconds = params['overlap_time'] / 1000.0
    
    # Process all batches in a single loop
    if len(batch_intervals) > 0:
        # Initialize progress bar (will be created starting from batch 1)
        pbar = None
        # Initialize bad_ch set (will be populated on first batch)
        bad_ch = set()
        
        for batch_idx, (batch_start, batch_end) in enumerate(batch_intervals):
            # Print status for first batch
            if batch_idx == 0:
                print(f"Processing batch {batch_idx + 1}/{n_batches_actual}: {batch_start:.2f}s - {batch_end:.2f}s")
            # Start progress bar from second batch
            elif batch_idx == 1 and TQDM_AVAILABLE and len(batch_intervals) > 1:
                pbar = tqdm(total=len(batch_intervals) - 1, desc="Processing batches", initial=0)
                pbar.update(1)  # Update for batch 1
            elif pbar is not None:
                pbar.update(1)
            elif not TQDM_AVAILABLE:
                print(f"Processing batch {batch_idx + 1}/{n_batches_actual}: {batch_start:.2f}s - {batch_end:.2f}s")
            
            # Load data and timestamps
            batch_data, batch_timestamps = data_obj.extract_raw_data((batch_start, batch_end))
            # batch_data shape: (n_channels, n_samples)
            # batch_timestamps shape: (n_samples,) in seconds
            
            # Check for empty batches - error out if this happens
            if batch_data.shape[1] == 0:
                raise ValueError(f"Empty batch detected at batch {batch_idx + 1}/{n_batches_actual} (time: {batch_start:.2f}s - {batch_end:.2f}s)")
            # Apply bandpass filter
            if TORCH_AVAILABLE and DEVICE.type == 'cuda':
                filtered_data_cpu = filter_data_GPU(batch_data, sample_rate, params['low_fr'], params['high_fr'], sos_bandpass)
            else:
                filtered_data_cpu = filter_data_CPU(batch_data, sos_bandpass)
            
            # Get dimensions
            n_channels = filtered_data_cpu.shape[0]
            n_samples = filtered_data_cpu.shape[1]
            
            # Compute spike_index and detect bad channels only on first batch
            if batch_idx == 0:
                # Compute spike_index for each channel (before denoising)
                normal_ratio = np.sqrt(2 / np.pi) / 0.6744897501960817  # ≈ 1.183 for normal distribution
                spike_index = np.zeros(n_channels)
                for ch_idx in range(n_channels):
                    abs_signal = np.abs(filtered_data_cpu[ch_idx, :])
                    avg_abs = np.mean(abs_signal)
                    median_abs = np.median(abs_signal)
                    if median_abs > 0:
                        ratio = avg_abs / median_abs
                        spike_index[ch_idx] = ratio - normal_ratio
                    else:
                        spike_index[ch_idx] = np.nan
                
                # Detect bad channels based on spike_index
                bad_ch = set()
                for ch_idx in range(n_channels):
                    if np.isnan(spike_index[ch_idx]):
                        bad_ch.add(ch_idx)
                        continue
                    
                    # Get noise_space neighbors
                    neighbors = noise_neighbors[ch_idx]
                    if len(neighbors) > 0:
                        # Get spike_index values for neighbors (excluding bad channels)
                        neighbor_indices = [n for n in neighbors if n not in bad_ch and not np.isnan(spike_index[n])]
                        if len(neighbor_indices) > 0:
                            median_neighbor_index = np.median(spike_index[neighbor_indices])
                            # Tag as bad if: spike_index < 0 OR spike_index < (median_neighbors / 5)
                            if spike_index[ch_idx] < 0 or spike_index[ch_idx] < (median_neighbor_index / 5):
                                bad_ch.add(ch_idx)
                        else:
                            # No valid neighbors, tag as bad if spike_index < 0
                            if spike_index[ch_idx] < 0:
                                bad_ch.add(ch_idx)
                    else:
                        # No neighbors, tag as bad if spike_index < 0
                        if spike_index[ch_idx] < 0:
                            bad_ch.add(ch_idx)
            
            # Denoise each channel using PCA on neighboring channels
            # Store noise signals for first batch only (for plotting, if make_plots is True)
            if batch_idx == 0 and make_plots:
                noise_signals = np.zeros_like(filtered_data_cpu)
            else:
                noise_signals = None
            
            if TORCH_AVAILABLE and DEVICE.type == 'cuda':
                # GPU version
                denoised_data_torch, filtered_data_torch = denoise_data_GPU(
                    filtered_data_cpu, noise_neighbors, bad_ch, batch_idx, noise_signals,
                    params['noise_median_threshold']
                )
                
                # Invert denoised data once (both find_spikes and extract_spike_properties expect inverted data)
                denoised_data_inverted_torch = -denoised_data_torch
                # Delete non-inverted version immediately to save memory
                del denoised_data_torch
                torch.cuda.empty_cache()
                
                # Find local minima using refractory neighbors
                local_minima_indices_cpu = find_spikes_GPU(
                    denoised_data_inverted_torch, refractory_neighbors, fix_rect_mask, 
                    n_temporal_samples, n_channels, n_samples, params['spike_median_threshold']
                )
                
                # Convert spike sample indices to timestamps for accurate filtering
                # local_minima_indices_cpu contains (ch, sample) tuples
                spike_timestamps = []
                spike_indices_with_timestamps = []
                for ch, sample in local_minima_indices_cpu:
                    if sample < len(batch_timestamps):
                        spike_time = batch_timestamps[sample]
                        spike_timestamps.append(spike_time)
                        spike_indices_with_timestamps.append((ch, sample, spike_time))
                
                # Calculate exclusion regions based on timestamps (non-overlapping)
                # For first batch: exclude spikes before batch_start + wvf_time
                if batch_idx == 0:
                    min_time = batch_start + wvf_time_seconds
                else:
                    # For batches after first: exclude before batch_start + overlap_time - wvf_time
                    # (to catch spikes that were too close to end of previous batch)
                    min_time = batch_start + overlap_time_seconds - wvf_time_seconds
                
                # For all batches: exclude spikes after batch_end - wvf_time
                max_time = batch_end - wvf_time_seconds
                
                # Filter spikes based on timestamps
                local_minima_indices_cpu = [(ch, sample) for ch, sample, spike_time in spike_indices_with_timestamps 
                                           if spike_time >= min_time and spike_time < max_time]
                
                # Extract spike properties using GPU
                if len(local_minima_indices_cpu) == 0:
                    raise ValueError(f"No spikes found in batch {batch_idx + 1}/{n_batches_actual} (time: {batch_start:.2f}s - {batch_end:.2f}s)")
                
                Properties, PropTitles, ch_noise_str = extract_spike_properties_GPU(
                    denoised_data_inverted_torch,
                    local_minima_indices_cpu,
                    wvf_neighbors,  # Use waveform neighbors for waveform extraction
                    wvf_samples,
                    params['spike_median_threshold'],
                    sample_rate,
                    ch_map,
                    bad_channels=bad_ch,
                    bad_ch_fill_distance=float(params.get('bad_ch_fill_distance', 50.0))
                )
                
                # Save property titles once for note keeping (on first batch)
                if batch_idx == 0:
                    prop_titles_filename = os.path.join(spike_prop_dir, 'property_titles.npz')
                    np.savez(prop_titles_filename, PropTitles=PropTitles)
                    # Also save as text file for easy reading
                    prop_titles_txt_filename = os.path.join(spike_prop_dir, 'property_titles.txt')
                    with open(prop_titles_txt_filename, 'w') as f:
                        f.write("Property Titles (column order):\n")
                        for i, title in enumerate(PropTitles):
                            f.write(f"  Column {i}: {title}\n")
                
                # Replace sample numbers with timestamps in the column titled 't'
                # local_minima_indices_cpu contains (ch, sample) tuples
                # Extract sample indices and use them to index into batch_timestamps
                sample_indices = np.array([sample for ch, sample in local_minima_indices_cpu])
                t_col = list(PropTitles).index('t') if 't' in PropTitles else 6
                Properties[:, t_col] = batch_timestamps[sample_indices]
                
                # Remove spikes with NaN in any property (invalid spikes with both weights = 0)
                valid_mask = ~np.any(np.isnan(Properties), axis=1)
                Properties = Properties[valid_mask]
                
                # Save spike properties for this batch
                spike_prop_filename = os.path.join(spike_prop_dir, f'batch_{batch_idx + 1:04d}_spike_properties.npz')
                np.savez(spike_prop_filename, Properties=Properties, PropTitles=PropTitles, ch_noise_str=ch_noise_str)
                
                # Copy to CPU for plotting (if make_plots is True)
                if make_plots:
                    denoised_data = denoised_data_inverted_torch.cpu().numpy()
                else:
                    denoised_data = None
                
                # Clean up filtered_data_torch (already used)
                del filtered_data_torch
                torch.cuda.empty_cache()
                
                # For non-first batches, we can delete denoised_data_inverted_torch after copying
                if batch_idx > 0:
                    del denoised_data_inverted_torch
                    torch.cuda.empty_cache()
            else:
                # CPU version
                denoised_data = denoise_data_CPU(
                    filtered_data_cpu, noise_neighbors, bad_ch, batch_idx, noise_signals,
                    params['spike_median_threshold']
                )
                
                # Find local minima using refractory neighbors
                local_minima_indices_cpu = find_spikes_CPU(
                    denoised_data, refractory_neighbors, fix_rect_mask, 
                    n_temporal_samples, n_channels, n_samples
                )

                # Convert spike sample indices to timestamps for accurate filtering
                # local_minima_indices_cpu contains (ch, sample) tuples
                spike_timestamps = []
                spike_indices_with_timestamps = []
                for ch, sample in local_minima_indices_cpu:
                    if sample < len(batch_timestamps):
                        spike_time = batch_timestamps[sample]
                        spike_timestamps.append(spike_time)
                        spike_indices_with_timestamps.append((ch, sample, spike_time))
                
                # Calculate exclusion regions based on timestamps (non-overlapping)
                # For first batch: exclude spikes before batch_start + wvf_time
                if batch_idx == 0:
                    min_time = batch_start + wvf_time_seconds
                else:
                    # For batches after first: exclude before batch_start + overlap_time - wvf_time
                    # (to catch spikes that were too close to end of previous batch)
                    min_time = batch_start + overlap_time_seconds - wvf_time_seconds
                
                # For all batches: exclude spikes after batch_end - wvf_time
                max_time = batch_end - wvf_time_seconds
                
                # Filter spikes based on timestamps
                local_minima_indices_cpu = [(ch, sample) for ch, sample, spike_time in spike_indices_with_timestamps 
                                           if spike_time >= min_time and spike_time < max_time]
            
            # Plot filtered data only for first batch (before deleting)
            if batch_idx == 0 and make_plots:
                # Only plot first 5% of batch data
                n_plot_samples = int(n_samples * 0.05)
                plot_end_idx = min(n_plot_samples, n_samples)
                
                # Extract the portion to plot (first 5% only)
                plot_data = filtered_data_cpu[:, :plot_end_idx]
                plot_time_axis = np.arange(plot_end_idx) / sample_rate + batch_start
                
                # Create one plot per channel
                for ch_idx in range(n_channels):
                    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                    ax.plot(plot_time_axis, plot_data[ch_idx, :])
                    ax.set_ylabel(f'Amplitude (Ch {ch_idx})')
                    ax.set_xlabel('Time (s)')
                    ax.set_title(f'Filtered Data - Channel {ch_idx} (Batch {batch_idx + 1}/{n_batches_actual})\n'
                                f'Time: {plot_time_axis[0]:.2f}s - {plot_time_axis[-1]:.2f}s | '
                                f'Filter: {params["low_fr"]}-{params["high_fr"]} Hz')
                    ax.grid(True)
                    plt.tight_layout()
                    
                    # Save individual plot for this channel
                    plot_filename = os.path.join(plot_dir, f'batch_{batch_idx + 1:04d}_ch_{ch_idx:03d}_filtered.png')
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                
                # Plot denoised data for first batch (first 5% only)
                if denoised_data is not None:
                    plot_denoised_data = denoised_data[:, :plot_end_idx]
                    plot_time_axis_den = np.arange(plot_end_idx) / sample_rate + batch_start
                    
                    # Calculate threshold per channel: median absolute value * spike_median_threshold
                    # denoised_data is inverted, so we take absolute value to get the magnitude
                    abs_denoised_plot = np.abs(plot_denoised_data)
                    median_abs_per_channel = np.median(abs_denoised_plot, axis=1)  # Shape: (n_channels,)
                    threshold_per_channel = median_abs_per_channel * params['spike_median_threshold']  # Shape: (n_channels,)
                    
                    # Create one plot per channel for denoised data
                    for ch_idx in range(n_channels):
                        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                        ax.plot(plot_time_axis_den, plot_denoised_data[ch_idx, :])
                        # Add dashed threshold line (positive value since data is inverted)
                        ax.axhline(y=threshold_per_channel[ch_idx], color='r', linestyle='--', 
                                  linewidth=1.5, label=f'Threshold ({threshold_per_channel[ch_idx]:.2f})')
                        ax.set_ylabel(f'Amplitude (Ch {ch_idx})')
                        ax.set_xlabel('Time (s)')
                        ax.set_title(f'Denoised Data - Channel {ch_idx} (Batch {batch_idx + 1}/{n_batches_actual})\n'
                                    f'Time: {plot_time_axis_den[0]:.2f}s - {plot_time_axis_den[-1]:.2f}s | '
                                    f'Filter: {params["low_fr"]}-{params["high_fr"]} Hz')
                        ax.grid(True)
                        ax.legend()
                        plt.tight_layout()
                        
                        # Save individual plot for this channel
                        plot_filename = os.path.join(den_plot_dir, f'batch_{batch_idx + 1:04d}_ch_{ch_idx:03d}_denoised.png')
                        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    print(f"  Saved {n_channels} denoised channel plots for batch {batch_idx + 1} (first 5%)")
                
                # Plot noise signals for first batch (first 5% only)
                if noise_signals is not None:
                    plot_noise_data = noise_signals[:, :plot_end_idx]
                    plot_time_axis_noise = np.arange(plot_end_idx) / sample_rate + batch_start
                    
                    # Create one plot per channel for noise signals
                    for ch_idx in range(n_channels):
                        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                        ax.plot(plot_time_axis_noise, plot_noise_data[ch_idx, :])
                        ax.set_ylabel(f'Amplitude (Ch {ch_idx})')
                        ax.set_xlabel('Time (s)')
                        ax.set_title(f'Noise Signal (Regressed Out) - Channel {ch_idx} (Batch {batch_idx + 1}/{n_batches_actual})\n'
                                    f'Time: {plot_time_axis_noise[0]:.2f}s - {plot_time_axis_noise[-1]:.2f}s | '
                                    f'Filter: {params["low_fr"]}-{params["high_fr"]} Hz')
                        ax.grid(True)
                        plt.tight_layout()
                        
                        # Save individual plot for this channel
                        plot_filename = os.path.join(noise_plot_dir, f'batch_{batch_idx + 1:04d}_ch_{ch_idx:03d}_noise.png')
                        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    print(f"  Saved {n_channels} noise signal plots for batch {batch_idx + 1} (first 5%)")
                
                # Create scatter plot: channel number vs spike_index (bad channels in red, good in green)
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                # Filter out NaN values for plotting
                valid_mask = ~np.isnan(spike_index)
                valid_channels = np.where(valid_mask)[0]
                
                # Separate good and bad channels
                good_channels = [ch for ch in valid_channels if ch not in bad_ch]
                bad_channels = [ch for ch in valid_channels if ch in bad_ch]
                
                # Plot good channels in green
                if len(good_channels) > 0:
                    ax.scatter(good_channels, spike_index[good_channels], alpha=0.6, s=50, color='green', label='Good channels')
                
                # Plot bad channels in red
                if len(bad_channels) > 0:
                    ax.scatter(bad_channels, spike_index[bad_channels], alpha=0.6, s=50, color='red', label='Bad channels')
                
                ax.set_xlabel('Channel Number')
                ax.set_ylabel('Spike Index (Avg(|Signal|) / Median(|Signal|) - Normal Ratio)')
                ax.set_title(f'Spike Index - Batch {batch_idx + 1}/{n_batches_actual}\n'
                            f'Time: {batch_start:.2f}s - {batch_end:.2f}s | Bad channels: {len(bad_ch)}')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='Normal distribution (0.0)')
                ax.legend()
                plt.tight_layout()
                
                # Save spike index plot
                ratio_plot_filename = os.path.join(out_folder, f'batch_{batch_idx + 1:04d}_spike_index.png')
                plt.savefig(ratio_plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved spike index scatter plot for batch {batch_idx + 1} ({len(bad_ch)} bad channels detected)")
                
                # Delete filtered data to free memory (after plotting, only for first batch)
                del filtered_data_cpu
                # Also clean up denoised_data_inverted_torch from GPU after plotting
                if TORCH_AVAILABLE and DEVICE.type == 'cuda':
                    if 'denoised_data_inverted_torch' in locals():
                        del denoised_data_inverted_torch
                    torch.cuda.empty_cache()
                print(f"  Saved {n_channels} channel plots for batch {batch_idx + 1} (first 5%)")
            elif batch_idx == 0:
                # First batch but make_plots is False - still need to clean up
                del filtered_data_cpu
                if TORCH_AVAILABLE and DEVICE.type == 'cuda':
                    if 'denoised_data_inverted_torch' in locals():
                        del denoised_data_inverted_torch
                    torch.cuda.empty_cache()
            else:
                # For batches after the first, filtered_data_cpu already deleted above
                pass
        
        # Close progress bar if it was created
        if pbar is not None:
            pbar.close()
    
    if make_plots:
        print(f"\nProcessing complete! Plots saved to: {plot_dir}")
        print(f"Denoised plots saved to: {den_plot_dir}")
        print(f"Noise signal plots saved to: {noise_plot_dir}")
    else:
        print(f"\nProcessing complete! (Plots skipped)")
    return None
