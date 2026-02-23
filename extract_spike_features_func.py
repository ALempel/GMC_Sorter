import numpy as np

# Try to use PyTorch for GPU operations
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None


def estimate_peak_location(
    S, depth, rows, reg=1e-9, x_anchor_d=240.0, anchor_weight=0.5, eps=1e-9
):
    """
    Estimate peak location from signal strength and channel positions.
    
    Parameters
    ----------
    S : np.ndarray, shape (N, K+1)
        Signal strength for each spike's channels
    depth : np.ndarray, shape (N, K+1)
        Depth coordinates for each spike's channels (gets quadratic fit)
    rows : np.ndarray, shape (N, K+1)
        Row coordinates for each spike's channels (gets weighted average)
    """
    N = S.shape[0]
    valid = np.isfinite(depth) & np.isfinite(S) & np.isfinite(rows)

    # Anchor & max-row (col 0 defines the row)
    depth_ref = depth[:, 0]
    row0    = rows[:, 0]
    same_row0 = valid & (np.abs(rows - row0[:, None]) < 1e-6)
    row_other = valid & ~same_row0
    row_other_for_sum = row_other.copy()
    # Add detected channel (column 0) to other-row fit
    row_other[:, 0] = valid[:, 0]

    # --- masks & design (as you have) ---

    depth_s = np.where(valid, depth, 0.0)
    depth_c = depth_s - depth_ref[:, None]
    Phi = np.stack([np.ones_like(depth_c), depth_c, depth_c**2], axis=2)

    Slog = np.where(valid, np.log(np.clip(S, eps, np.inf)), 0.0)
    
    # Compute quadratic fit for same-row channels
    W0_same = same_row0.astype(float)[..., None]
    WX0_same = W0_same * Phi
    A_base_same = np.einsum('nij,nik->njk', WX0_same, Phi) + eps*np.eye(3)[None, :, :]
    b_base_same = np.einsum('nij,ni->nj', WX0_same, Slog)
    
    # Compute quadratic fit for other-row channels
    W0_other = row_other.astype(float)[..., None]
    WX0_other = W0_other * Phi
    A_base_other = np.einsum('nij,nik->njk', WX0_other, Phi) + eps*np.eye(3)[None, :, :]
    b_base_other = np.einsum('nij,ni->nj', WX0_other, Slog)
    
    # Count valid channels for each fit
    n_valid_same = np.sum(same_row0, axis=1)  # (N,)
    n_valid_other = np.sum(row_other, axis=1)  # (N,)
    has_two_same = (n_valid_same == 2)
    has_two_other = (n_valid_other == 2)
    
    # Center of mass for 2 valid channels (depth_hat = sum(depth*S)/sum(S))
    sum_S_same = np.sum(np.where(same_row0, S, 0.0), axis=1)
    sum_dS_same = np.sum(np.where(same_row0, depth * S, 0.0), axis=1)
    depth_hat_same_com = np.divide(sum_dS_same, np.maximum(sum_S_same, eps), out=np.full_like(sum_S_same, np.nan), where=(has_two_same & (sum_S_same > eps)))
    sum_S_other = np.sum(np.where(row_other_for_sum, S, 0.0), axis=1)
    sum_dS_other = np.sum(np.where(row_other_for_sum, depth * S, 0.0), axis=1)
    depth_hat_other_com = np.divide(sum_dS_other, np.maximum(sum_S_other, eps), out=np.full_like(sum_S_other, np.nan), where=(has_two_other & (sum_S_other > eps)))
    
    # Solve for same-row fit (>= 3 points)
    has_enough_same = n_valid_same >= 3
    a0_same = np.zeros((N,))
    b0_same = np.zeros((N,))
    c0_same = np.zeros((N,))
    
    if np.any(has_enough_same):
        a0_same_fit, b0_same_fit, c0_same_fit = np.linalg.solve(
            A_base_same[has_enough_same], 
            b_base_same[has_enough_same, ..., None]
        )[..., 0].T
        a0_same[has_enough_same] = a0_same_fit
        b0_same[has_enough_same] = b0_same_fit
        c0_same[has_enough_same] = c0_same_fit
    
    # Solve for other-row fit
    has_enough_other = n_valid_other >= 3
    a0_other = np.zeros((N,))
    b0_other = np.zeros((N,))
    c0_other = np.zeros((N,))
    
    if np.any(has_enough_other):
        a0_other_fit, b0_other_fit, c0_other_fit = np.linalg.solve(
            A_base_other[has_enough_other], 
            b_base_other[has_enough_other, ..., None]
        )[..., 0].T
        a0_other[has_enough_other] = a0_other_fit
        b0_other[has_enough_other] = b0_other_fit
        c0_other[has_enough_other] = c0_other_fit
    
    # Compute depth_hat for same-row fit (quadratic when >=3, else COM when 2, else ref)
    denom_same = np.where(np.abs(c0_same) > reg, 2.0*c0_same, -2.0*reg)
    depth_hat_same_quad = depth_ref - (b0_same / denom_same)
    depth_hat_same = np.where(has_enough_same, depth_hat_same_quad,
                              np.where(has_two_same, depth_hat_same_com, depth_ref))
    
    # Compute depth_hat for other-row fit (quadratic when >=3, else COM when 2, else ref)
    denom_other = np.where(np.abs(c0_other) > reg, 2.0*c0_other, -2.0*reg)
    depth_hat_other_quad = depth_ref - (b0_other / denom_other)
    depth_hat_other = np.where(has_enough_other, depth_hat_other_quad,
                                np.where(has_two_other, depth_hat_other_com, depth_ref))
    
    # Compute signal strength means for both rows (used for depth weighting and row_hat)
    # Use mean instead of sum so it doesn't depend on number of channels
    n_valid_same_for_mean = np.sum(same_row0, axis=1)  # (N,)
    n_valid_other_for_sum = np.sum(row_other_for_sum, axis=1)  # (N,)
    # Compute mean_S_same - avoid division warning by using np.divide with where
    sum_S_same = np.sum(np.where(same_row0, S, 0.0), axis=1)
    mean_S_same = np.divide(sum_S_same, n_valid_same_for_mean,
                           out=np.zeros_like(sum_S_same),
                           where=(n_valid_same_for_mean > 0))  # (N,)
    
    # Compute mean_S_other - n_valid_other_for_sum can be 0 if ALL valid channels are in same row as row0
    sum_S_other = np.sum(np.where(row_other_for_sum, S, 0.0), axis=1)
    mean_S_other = np.divide(sum_S_other, n_valid_other_for_sum,
                            out=np.zeros_like(sum_S_other),
                            where=(n_valid_other_for_sum > 0))  # (N,)
    
    # Weight for same-row: 1.0 if >= 2 valid channels (quadratic or COM), else 0
    weight_same = np.where(n_valid_same >= 2, 1.0, 0.0)
    
    # Weight for other-row: mean(other) / mean(same) if >= 2 valid channels, else 0
    # Avoid division by zero
    weight_other = np.where(
        (n_valid_other >= 2) & (mean_S_same > eps),
        mean_S_other / mean_S_same,
        0.0
    )
    
    # Normalize weights (handle case where both are 0)
    # total_weight = 0 when n_valid_same < 2 and (n_valid_other < 2 or mean_S_same <= eps).
    # So spikes with only 1 valid channel get invalid_spikes = True and depth_hat = NaN (discarded later).
    total_weight = weight_same + weight_other
    invalid_spikes = total_weight <= eps  # Spikes with both weights = 0
    
    weight_same_norm = np.divide(weight_same, total_weight,
                                out=np.zeros_like(weight_same),
                                where=(total_weight > eps))
    weight_other_norm = np.divide(weight_other, total_weight,
                                 out=np.zeros_like(weight_other),
                                 where=(total_weight > eps))
    
    # Weighted average of the two depth_hat values
    depth_hat_raw = weight_same_norm * depth_hat_same + weight_other_norm * depth_hat_other
    
    # For Pass 1, we'll use the combined result
    # Store coefficients for Pass 2 (we'll use same-row for Pass 2 logic)
    a0, b0, c0 = a0_same, b0_same, c0_same

    # rows that are not concave (c >= 0); for 2-channel COM only check finite
    bad_same = (has_enough_same & (c0_same >= 0.0)) | (has_two_same & ~np.isfinite(depth_hat_same_com))
    bad_other = (has_enough_other & (c0_other >= 0.0)) | (has_two_other & ~np.isfinite(depth_hat_other_com))
    bad = bad_same | bad_other  # Bad if either fit is bad

    # ---- Build anchor (only used for bad rows) ----
    L = float(x_anchor_d)
    L2 = L * L
    L4 = L2 * L2
    
    # anchor amplitude: use your chosen scalar level (e.g., x_min_amp)
    # per-row minimum over samples on the same row0
    smin_row0 = np.maximum(eps, np.nanmin(np.where(same_row0, S, np.nan), axis=1)) / 2
    y_fake    = np.log(smin_row0)

    A_add = anchor_weight * np.array([[2.0,    0.0,   2.0*L2],
                                      [0.0, 2.0*L2,      0.0],
                                      [2.0*L2, 0.0,   2.0*L4]], dtype=A_base_same.dtype)
    b_add = anchor_weight * y_fake[:, None] * np.array([2.0, 0.0, 2.0*L2], dtype=b_base_same.dtype)

    # Pass 2 for same-row fit
    aw_same = np.where(bad_same & has_enough_same, float(anchor_weight), 0.0)
    A1_same = A_base_same + aw_same[:, None, None] * A_add[None, :, :]
    b1_same = b_base_same + aw_same[:, None] * b_add
    
    # Pass 2 for other-row fit
    aw_other = np.where(bad_other & has_enough_other, float(anchor_weight), 0.0)
    A1_other = A_base_other + aw_other[:, None, None] * A_add[None, :, :]
    b1_other = b_base_other + aw_other[:, None] * b_add

    # ---- Pass 2: re-solve with conditional anchor ----
    # Same-row Pass 2
    aL_same = a0_same.copy()  # Start with Pass 1 values
    bL_same = b0_same.copy()
    cL_same = c0_same.copy()
    if np.any(aw_same > 0):
        mask_same = aw_same > 0
        # Solve for rows that need anchor
        coeffs_same = np.linalg.solve(
            A1_same[mask_same],
            b1_same[mask_same, ..., None]
        )  # Shape: (n_bad, 3, 1)
        aL_same[mask_same] = coeffs_same[:, 0, 0]
        bL_same[mask_same] = coeffs_same[:, 1, 0]
        cL_same[mask_same] = coeffs_same[:, 2, 0]
    
    # Other-row Pass 2
    aL_other = a0_other.copy()  # Start with Pass 1 values
    bL_other = b0_other.copy()
    cL_other = c0_other.copy()
    if np.any(aw_other > 0):
        mask_other = aw_other > 0
        # Solve for rows that need anchor
        coeffs_other = np.linalg.solve(
            A1_other[mask_other],
            b1_other[mask_other, ..., None]
        )  # Shape: (n_bad, 3, 1)
        aL_other[mask_other] = coeffs_other[:, 0, 0]
        bL_other[mask_other] = coeffs_other[:, 1, 0]
        cL_other[mask_other] = coeffs_other[:, 2, 0]
    
    # Recompute depth_hat for Pass 2 (only for quadratic; keep COM or ref for 2-channel / 1-channel)
    denom_same_L = np.where(np.abs(cL_same) > reg, 2.0*cL_same, -2.0*reg)
    depth_hat_same_L = np.where(has_enough_same,
                                depth_ref - (bL_same / denom_same_L),
                                depth_hat_same)
    
    denom_other_L = np.where(np.abs(cL_other) > reg, 2.0*cL_other, -2.0*reg)
    depth_hat_other_L = np.where(has_enough_other,
                                  depth_ref - (bL_other / denom_other_L),
                                  depth_hat_other)
    
    # Weighted average for Pass 2 (using same weights as Pass 1)
    depth_hat_raw = weight_same_norm * depth_hat_same_L + weight_other_norm * depth_hat_other_L
    
    # Store for final check
    cL = cL_same  # Use same-row for final check


    # if still non-concave (or non-finite), declare failure; convexity only applies when quadratic was used (>=3 ch)
    # Also mark invalid_spikes (both weights = 0, e.g. only 1 valid channel) as NaN
    bad_final = (~np.isfinite(depth_hat_raw.flatten())) | (has_enough_same & (cL.flatten() >= 0.0)) | invalid_spikes
    depth_hat     = np.where(~bad_final, depth_hat_raw, np.nan)

    # ==== (3) Sharpness from log-linear fit vs absolute distance (no baseline ops) ====
    # Model: log(S) ~ alpha - lam * d, with d = |depth - depth_hat|.
    # Only use samples with S > eps to avoid log negatives; fully vectorized 2x2 solve.
    d = np.where(valid, np.abs(depth - depth_hat[:, None]), 0.0)
    use = same_row0
    use[:,0] = True # always use detected channel

    d_use  = np.where(use, d, 0.0)
    logS   = np.where(use, np.log(S), 0.0)

    sw   = np.sum(use.astype(float), axis=1)               # (N,)
    swd  = np.sum(d_use, axis=1)
    swd2 = np.sum(d_use * d_use, axis=1)
    swy  = np.sum(logS, axis=1)
    swdy = np.sum(d_use * logS, axis=1)

    det = sw * swd2 - swd * swd
    det_safe = np.where(np.abs(det) > reg, det, np.where(det >= 0, det + reg, det - reg))

    alpha = (swy * swd2 - swd * swdy) / det_safe
    lam   = (sw * swdy - swd * swy)   / det_safe
    lam   = np.where(sw > 1, lam, np.nan)    # rows with <2 pts → lam=0

    sharpness = lam
    # Mark invalid spikes (both weights = 0) as NaN
    sharpness[invalid_spikes] = np.nan
    
    # row_hat weighting (if you want no negatives at all):
    # Use means directly as weights (independent of number of channels)
    row1 = np.where(row_other_for_sum, rows, -np.inf).max(axis=1)
    den  = np.maximum(mean_S_same + mean_S_other, reg)
    # Replace NaN/inf so multiply does not raise or produce invalid values
    mean_S_same_safe = np.nan_to_num(mean_S_same, nan=0.0, posinf=0.0, neginf=0.0)
    mean_S_other_safe = np.nan_to_num(mean_S_other, nan=0.0, posinf=0.0, neginf=0.0)
    row0_safe = np.nan_to_num(row0, nan=0.0, posinf=0.0, neginf=0.0)
    row1_safe = np.nan_to_num(row1, nan=0.0, posinf=0.0, neginf=0.0)
    numerator = mean_S_same_safe * row0_safe + mean_S_other_safe * row1_safe
    row_hat = np.divide(numerator, den,
                       out=np.zeros_like(numerator),
                       where=(den > 0))
    # Mark invalid spikes (both weights = 0) as NaN
    row_hat[invalid_spikes] = np.nan

    return depth_hat, row_hat, sharpness


def extract_spike_properties_GPU(denoise_data_inverted, spike_indices, neighbor_map, wvf_samples, spike_median_threshold, fs, ch_map, **kwargs):
    """
    Extract spike properties from detected spikes using GPU acceleration.
    
    Parameters
    ----------
    denoise_data_inverted : np.ndarray or torch.Tensor
        Inverted denoised data, shape (n_channels, n_samples) (data should be negated: -denoised_data).
        If numpy array, will be moved to GPU.
    spike_indices : array-like
        List of (channel, sample) tuples for detected spikes
    neighbor_map : dict
        Dictionary mapping channel index to list of neighbor channel indices
    wvf_samples : int
        Number of samples for waveform window (±wvf_samples around peak)
    spike_median_threshold : float
        Threshold multiplier for median absolute value (used for clipping)
    fs : float
        Sample rate in Hz
    ch_map : np.ndarray
        Channel map, shape (n_channels, 2) with columns [x, y]
    **kwargs : dict, optional
        Optional parameters with default values:
        - str_window_ms : float, default=0.05
            Signal strength window in milliseconds (±str_window_ms around peak)
        - max_d_for_pos : float, default=refr_space (or 100 if refr_space not in params)
            Soft boundary: depth position is clamped to this distance from the detected channel depth (same units as ch_map). Defaults to refr_space when provided.
        - rel_strength_threshold : float, default=0.2
            For quadratic-fit position: strengths are normalized by the detected channel strength, then channels with relative strength below this are set to 0 (excluded from fit)
        - bad_channels : set or list, optional
            Channel indices to treat as bad; their rows are replaced with mean of valid channels within bad_ch_fill_distance so position/strength use filled data.
        - bad_ch_fill_distance : float, default=50.0
            Max distance (same units as ch_map) for valid channels used to fill bad channel rows.
    
    Returns
    -------
    Properties : np.ndarray
        Array of spike properties, shape (n_spikes, 8)
        Columns: [y_pos, x_pos, sig_str, th_pk, half_w, spread, t, ch]
        - y_pos: Depth/vertical position (interpolated from channel Y coordinates)
        - x_pos: Horizontal position (interpolated from channel X coordinates)
        - sig_str: Signal strength
        - th_pk: Threshold peak
        - half_w: Half width in milliseconds
        - spread: Spatial spread
        - t: Timestamp
        - ch: Channel number where spike was detected
    PropTitles : list
        List of property names
    ch_noise_str : np.ndarray
        Channel noise strength estimates
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU operations")
    
    if DEVICE is None:
        raise RuntimeError("No device available for GPU operations")
    
    # Set default values
    params = {
        'str_window_ms': 0.05,  # ±0.05 ms for signal strength window
        # max_d_for_pos defaults to refr_space when provided (see below)
        'rel_strength_threshold': 0.2,  # Relative strength below this (after normalizing by detected channel) is set to 0 and excluded from quadratic fit
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    # Convert inputs to torch tensors and move to GPU
    if isinstance(denoise_data_inverted, np.ndarray):
        denoise_data_torch = torch.from_numpy(denoise_data_inverted.astype(np.float32)).to(DEVICE)
    else:
        denoise_data_torch = denoise_data_inverted.to(DEVICE)
    
    # Fill bad channel rows in place so position/strength use replaced data (same tensor we gather from)
    bad_channels = params.get('bad_channels', ())
    bad_ch_fill_distance = params.get('bad_ch_fill_distance', 50.0)
    if bad_channels and bad_ch_fill_distance > 0 and ch_map is not None:
        ch_map = np.asarray(ch_map)
        distances = np.sqrt(((ch_map[:, np.newaxis, :] - ch_map[np.newaxis, :, :]) ** 2).sum(axis=2))
        n_channels = denoise_data_torch.shape[0]
        for bad_ch_idx in bad_channels:
            valid_mask = (distances[bad_ch_idx, :] <= bad_ch_fill_distance) & np.isin(np.arange(n_channels), list(bad_channels), invert=True)
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 0:
                valid_t = torch.from_numpy(valid_idx).long().to(denoise_data_torch.device)
                mean_signal = denoise_data_torch.index_select(0, valid_t).mean(dim=0)
                denoise_data_torch[bad_ch_idx, :] = mean_signal
    
    spike_indices = np.array(spike_indices)
    ch = torch.from_numpy(spike_indices[:, 0]).long().to(DEVICE)  # (N,)
    t = torch.from_numpy(spike_indices[:, 1]).long().to(DEVICE)  # (N,)
    N = len(spike_indices)
    
    # Build neighbor array on CPU first (small, no need for GPU)
    max_nei = max(len(sublist) for sublist in neighbor_map.values())
    neigh_arr = np.full((len(neighbor_map), max_nei), len(neighbor_map))
    for C in range(len(neighbor_map)):
        n_ar_c = np.array(neighbor_map[C])
        neigh_arr[C, 0:len(n_ar_c)] = n_ar_c
    
    neigh_arr_torch = torch.from_numpy(neigh_arr).long().to(DEVICE)
    K = neigh_arr_torch.shape[1]

    # Full list of channels: center + neighbors
    full_channels = torch.cat([
        ch[:, None],            # (N, 1)
        neigh_arr_torch[ch]     # (N, K)
    ], dim=1)                   # → shape (N, K+1)

    # Time window for each (broadcasted)
    wvf_w = wvf_samples
    time_offsets = torch.arange(-wvf_w, wvf_w + 1, device=DEVICE, dtype=torch.long)  # (W,)
    time_windows = t[:, None] + time_offsets  # (N, W)

    # Gather waveforms using advanced indexing
    # For each spike n, channel k, time w: get denoise_data[full_channels[n,k], time_windows[n,w]]
    N_w = time_windows.shape[1]  # W = 2*wvf_w + 1
    
    # Create expanded indices: (N, W, K+1)
    ch_idx_expanded = full_channels.unsqueeze(1).expand(N, N_w, K+1)  # (N, W, K+1)
    t_idx_expanded = time_windows.unsqueeze(2).expand(N, N_w, K+1)   # (N, W, K+1)
    
    # Check for sentinel positions (where channel index equals n_channels)
    # Sentinels occur when neighbor_map has fewer neighbors than max_nei, 
    # and the remaining slots are filled with n_channels (out-of-bounds value)
    sentinel_mask = (ch_idx_expanded == denoise_data_torch.shape[0])
    
    # Clip channel indices only for sentinels (to avoid index errors), but time indices should be valid
    # If time indices are out of bounds, that's a bug and should error
    ch_safe = torch.clamp(ch_idx_expanded, 0, denoise_data_torch.shape[0] - 1)
    
    # PyTorch advanced indexing: flatten and gather
    ch_flat = ch_safe.reshape(-1).long()
    t_flat = t_idx_expanded.reshape(-1).long()  # No clipping - let it error if out of bounds
    
    # Gather values using flat indices (will error if t_flat is out of bounds)
    gathered_flat = denoise_data_torch[ch_flat, t_flat]
    spike_wvf = gathered_flat.reshape(N, N_w, K+1)
    
    # Set NaN at sentinel positions (invalid neighbor channels)
    nan_val = torch.tensor(float('nan'), device=DEVICE, dtype=spike_wvf.dtype)
    spike_wvf = torch.where(sentinel_mask, nan_val, spike_wvf)
    # Get signal strength
    ssJump = int(np.floor(denoise_data_torch.shape[1] / 2000))  # SS 2000 time samples
    subsampled_data = denoise_data_torch[:, ::ssJump]
    
    # Compute median absolute value per channel (ignoring NaNs)
    abs_subsampled = torch.abs(subsampled_data)
    median_abs_subsampled = torch.nanmedian(abs_subsampled, dim=1)[0]
    
    clip_threshold = median_abs_subsampled[:, None] * spike_median_threshold
    clipped_data = torch.clamp(subsampled_data, -clip_threshold, clip_threshold)
    ch_noise_str = torch.sqrt(torch.nanmean(torch.square(clipped_data), dim=1))
    
    # Extract windowed data
    str_w = int(params['str_window_ms'] * 1e-3 * fs)  # ±str_window_ms in samples
    window_offsets = torch.arange(-str_w, str_w + 1, device=DEVICE, dtype=torch.long) + wvf_w
    windowed_wvf = spike_wvf[:, window_offsets, 0]  # (N, window_size)

    # Strength: raw RMS then smooth ReLU (str - k) with epsilon = h*k
    sig_str = torch.sqrt(torch.nanmean(torch.square(windowed_wvf), dim=1))
    k = ch_noise_str[full_channels[:, 0]]
    h = float(params.get('smooth_relu_h', 0.3))
    eps = h * k
    sig_str = 0.5 * ((sig_str - k) + torch.sqrt((sig_str - k) ** 2 + eps ** 2))
    
    # Get through over peak
    peak_region = spike_wvf[:, wvf_w+1:-str_w, 0]  # (N, region_size)
    th_loc = torch.argmin(peak_region, dim=1) + wvf_w + 1  # (N,)
    th_offsets = torch.arange(-str_w, str_w + 1, device=DEVICE, dtype=torch.long)
    th_locs = (th_loc[:, None] + th_offsets[None, :]).long()  # (N, 2*str_w+1)
    # Gather th_locs values using gather
    th_locs_clamped = torch.clamp(th_locs, 0, spike_wvf.shape[1] - 1)  # (N, 2*str_w+1)
    # Use gather to select values: spike_wvf[n, th_locs_clamped[n, i], 0]
    spike_wvf_ch0 = spike_wvf[:, :, 0]  # (N, W)
    # Create index tensor for gather: (N, 2*str_w+1)
    th_pk_numerator = torch.gather(spike_wvf_ch0, 1, th_locs_clamped).sum(dim=1)
    th_pk_denominator = torch.sum(spike_wvf[:, window_offsets, 0], dim=1)
    th_pk = th_pk_numerator / th_pk_denominator
    
    # Get peak position using quadratic fits
    # ch_map is (n_channels, 2) with columns [x, y] where:
    # - x is horizontal position (only 0 or 30 - discrete)
    # - y is depth/vertical position (0, 23, 46, 69, ... - continuous variation)
    # estimate_peak_location does:
    # - Quadratic fit on X coordinate (first parameter) → returns x_hat
    # - Weighted average on Y coordinate (second parameter) → returns y_hat
    # We want:
    # - Quadratic fit on Y (depth, continuous variation)
    # - Weighted average on X (horizontal, discrete: 0 or 30)
    # So we must SWAP coordinates when calling:
    # - Pass depth (y) as X parameter (first) to get quadratic fit
    # - Pass horizontal (x) as Y parameter (second) to get weighted average
    channel_xy = ch_map.copy()  # Keep [x, y] order: x=horizontal, y=depth
    nan_row = np.full((1, 2), np.nan)
    channel_xy = np.vstack([channel_xy, nan_row])  # (n_channels+1, 2)
    
    # Convert full_channels to numpy and ensure correct shape
    full_channels_np = full_channels.cpu().numpy().astype(np.int64)  # (N, K+1) - must be integer for indexing
    
    # NumPy advanced indexing: channel_xy[full_channels_np] gives (N, K+1, 2)
    # Each element of full_channels_np indexes into channel_xy's first dimension
    channel_pos = channel_xy[full_channels_np]  # (N, K+1, 2)
    
    str_by_ch = torch.sqrt(torch.nanmean(torch.square(spike_wvf[:, window_offsets, :]), dim=1))
    full_channels_safe = torch.clamp(full_channels, 0, denoise_data_torch.shape[0] - 1)
    k = ch_noise_str[full_channels_safe]
    nan_val_k = torch.tensor(float('nan'), device=DEVICE, dtype=k.dtype)
    k = torch.where(full_channels == denoise_data_torch.shape[0], nan_val_k, k)
    eps_ch = h * k
    str_by_ch = 0.5 * ((str_by_ch - k) + torch.sqrt((str_by_ch - k) ** 2 + eps_ch ** 2))

    # Noise-subtracted strengths (for spread and as base for position)
    str_by_ch_np = str_by_ch.cpu().numpy()  # (N, K+1)
    str_by_ch_np[str_by_ch_np <= 0] = np.nan

    # Spread (sharpness): use noise-subtracted strengths
    N = str_by_ch_np.shape[0]
    det_str = str_by_ch_np[:, 0]
    sum_all = np.nansum(str_by_ch_np, axis=1)
    spread = np.divide(det_str, sum_all, out=np.full(N, np.nan), where=(sum_all > 0))



    Y_np = channel_pos[:, :, 1]  # (N, K+1) - Depth positions (y from ch_map)
    X_np = channel_pos[:, :, 0]  # (N, K+1) - Horizontal positions (x from ch_map)

    # When all channels have the same X, we compute y_pos only; x_pos is not computed/assigned/saved (left NaN)
    all_same_x = (np.unique(ch_map[:, 0]).size == 1)

    quadratic_depth = bool(params.get('quadratic_depth', False))
    if quadratic_depth:
        depth_hat, row_hat, _ = estimate_peak_location(
            str_by_ch_np,
            Y_np,
            X_np
        )
        if all_same_x:
            row_hat = np.full(N, np.nan)
    else:
        row0 = X_np[:, 0]
        valid_row = np.isfinite(str_by_ch_np)
        same_row0 = valid_row & (np.abs(X_np - row0[:, None]) < 1e-6)
        row_other = valid_row & ~same_row0
        n_same = np.sum(same_row0.astype(float), axis=1)
        n_other = np.sum(row_other.astype(float), axis=1)
        # Depth COM: same row only
        sum_S_same = np.sum(np.where(same_row0, str_by_ch_np, 0.0), axis=1)
        sum_Y_S_same = np.sum(np.where(same_row0, Y_np * str_by_ch_np, 0.0), axis=1)
        depth_hat = np.divide(
            sum_Y_S_same, sum_S_same,
            out=np.full(N, np.nan), where=(sum_S_same > 0)
        )
        # row_hat: only when not all_same_x; otherwise leave unassigned (NaN)
        if all_same_x:
            row_hat = np.full(N, np.nan)
            invalid_com = (n_same <= 1)
        else:
            sum_S_other = np.sum(np.where(row_other, str_by_ch_np, 0.0), axis=1)
            mean_S_same = np.divide(sum_S_same, n_same, out=np.zeros_like(sum_S_same), where=(n_same > 0))
            mean_S_other = np.divide(sum_S_other, n_other, out=np.zeros_like(sum_S_other), where=(n_other > 0))
            row1 = np.max(np.where(row_other, X_np, -np.inf), axis=1)
            row1 = np.where(n_other > 0, row1, row0)
            den_row = np.maximum(mean_S_same + mean_S_other, 1e-12)
            row_hat = np.divide(
                mean_S_same * row0 + mean_S_other * row1, den_row,
                out=np.full(N, np.nan), where=(den_row > 1e-12)
            )
            invalid_com = (n_same <= 1) | (n_other == 0)
        depth_hat = np.where(invalid_com, np.nan, depth_hat)
        row_hat = np.where(invalid_com, np.nan, row_hat)

    # For spikes over the limit: replace position with COM (no hard clamp). Others unchanged.
    max_d = float(params.get('max_d_for_pos', params.get('refr_space', 100.0)))
    depth_ref = Y_np[:, 0]
    over_limit = np.isfinite(depth_hat) & (np.abs(depth_hat - depth_ref) > max_d)
    sum_S_com = np.nansum(str_by_ch_np, axis=1)
    depth_com = np.divide(
        np.nansum(Y_np * str_by_ch_np, axis=1), sum_S_com,
        out=np.full(N, np.nan), where=(sum_S_com > 0)
    )
    depth_hat = np.where(over_limit, depth_com, depth_hat)
    if not all_same_x:
        row_com = np.divide(
            np.nansum(X_np * str_by_ch_np, axis=1), sum_S_com,
            out=np.full(N, np.nan), where=(sum_S_com > 0)
        )
        row_hat = np.where(over_limit, row_com, row_hat)
    # when all_same_x, row_hat stays NaN (x_pos not computed/assigned)

    y_pos = depth_hat
    x_pos = row_hat
    
    # Get half width
    baseline = torch.max(spike_wvf[:, [0, -1], 0], dim=1)[0]  # (N,)
    peak = spike_wvf[:, wvf_w, 0]  # (N,)
    norm_wvf = (spike_wvf[:, :, 0] - baseline[:, None]) / ((peak - baseline)[:, None])
    
    # Find down crossing
    down_region = norm_wvf[:, wvf_w+1:]
    down_mask = down_region < 0.5
    down_cross = wvf_w + 1 + torch.argmax(down_mask.float(), dim=1)
    # Handle case where no crossing found
    down_cross = torch.where(torch.sum(down_mask.float(), dim=1) > 0, down_cross, 
                             torch.tensor(spike_wvf.shape[1] - 1, device=DEVICE))
    
    # Find up crossing (reversed)
    up_region = torch.flip(norm_wvf[:, :wvf_w], dims=[1])
    up_mask = up_region < 0.5
    up_cross = wvf_w - 1 - torch.argmax(up_mask.float(), dim=1)
    up_cross = torch.where(torch.sum(up_mask.float(), dim=1) > 0, up_cross,
                          torch.tensor(0, device=DEVICE))
    
    # Refine crossings with linear interpolation
    down_cross_clamped = torch.clamp(down_cross, 1, norm_wvf.shape[1] - 1)
    down_cross_v = norm_wvf[torch.arange(N, device=DEVICE), down_cross_clamped]
    down_cross_vp = norm_wvf[torch.arange(N, device=DEVICE), down_cross_clamped - 1]
    down_cross_refined = down_cross.float() - (0.5 - down_cross_v) / (down_cross_vp - down_cross_v)
    
    up_cross_clamped = torch.clamp(up_cross, 0, norm_wvf.shape[1] - 2)
    up_cross_v = norm_wvf[torch.arange(N, device=DEVICE), up_cross_clamped]
    up_cross_vn = norm_wvf[torch.arange(N, device=DEVICE), up_cross_clamped + 1]
    up_cross_refined = up_cross.float() + (0.5 - up_cross_v) / (up_cross_vn - up_cross_v)
    
    # Convert half width to milliseconds
    half_w = (down_cross_refined - up_cross_refined) * 1000.0 / fs
    
    # Convert all results to numpy and create Properties array
    # When all_same_x: omit x_pos entirely (7 columns). Otherwise: [y_pos, x_pos, sig_str, ...] (8 columns)
    if all_same_x:
        Properties = np.zeros((len(spike_indices), 7))
        Properties[:, 0] = y_pos
        Properties[:, 1] = sig_str.cpu().numpy()
        Properties[:, 2] = th_pk.cpu().numpy()
        Properties[:, 3] = half_w.cpu().numpy()
        Properties[:, 4] = spread
        Properties[:, 5] = t.cpu().numpy()
        Properties[:, 6] = ch.cpu().numpy()
        PropTitles = ['y_pos', 'sig_str', 'th_pk', 'half_w', 'spread', 't', 'ch']
    else:
        Properties = np.zeros((len(spike_indices), 8))
        Properties[:, 0] = y_pos
        Properties[:, 1] = x_pos
        Properties[:, 2] = sig_str.cpu().numpy()
        Properties[:, 3] = th_pk.cpu().numpy()
        Properties[:, 4] = half_w.cpu().numpy()
        Properties[:, 5] = spread
        Properties[:, 6] = t.cpu().numpy()
        Properties[:, 7] = ch.cpu().numpy()
        PropTitles = ['y_pos', 'x_pos', 'sig_str', 'th_pk', 'half_w', 'spread', 't', 'ch']
    
    # Note: Filtering of invalid spikes (NaN) will be done after timestamp replacement in sp_feature_ext.py
    
    # Clean up GPU memory
    del denoise_data_torch, spike_wvf, ch, t, full_channels, time_windows
    torch.cuda.empty_cache()

    return Properties, PropTitles, ch_noise_str.cpu().numpy()