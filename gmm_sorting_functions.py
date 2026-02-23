import numpy as np
import gc
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
# Try to use PyTorch for GPU acceleration
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


def gm_seed_local_max(properties, clust_ind, spikes_per_batch_thousands, minimum_cluster_index,
                       feature_distances, sorted_feature_idx, progress_callback=None, bounds=None):
    """
    Find local maxima seeds for Gaussian Mixture Model sorting.
    Caller (e.g. visualizer) must pass feature_distances and bounds keyed by column index
    into properties, and sorted_feature_idx; no feature names/titles.
    
    Parameters
    ----------
    properties : np.ndarray, shape (N, n_features)
        Spike properties array
    clust_ind : np.ndarray, shape (N,)
        Cluster index for each spike (cluster_den / background_den)
    spikes_per_batch_thousands : float
        Number of spikes per batch (in thousands)
    minimum_cluster_index : float
        Minimum cluster index threshold
    feature_distances : dict
        Dictionary mapping column index (int) to distance value
    sorted_feature_idx : int
        Column index of the feature used for sorting/batching
    progress_callback : callable, optional
        Callback function(current_batch, total_batches) for progress updates
    bounds : dict, optional
        If provided, maps column index (int) -> (min_val, max_val). Seeds outside these bounds are removed.
        
    Returns
    -------
    seeds : list of int
        List of global spike indices that are local maxima seeds
        
    Note
    ----
    Memory usage: For N spikes per batch, each feature comparison requires
    a (N, N) boolean matrix (~381 MB for 20,000 spikes). Features are processed
    incrementally to minimize peak memory usage.
    
    Spikes are filtered by minimum_cluster_index before batching, reducing the
    total number of batches to process. Batch size remains constant, but fewer
    batches means less total computation time.
    
    GPU acceleration: If PyTorch and CUDA are available, pairwise comparisons
    are automatically performed on GPU for significant speedup, especially
    for large batches. Falls back to CPU if GPU is not available.
    """
    N = properties.shape[0]
    n_cols = properties.shape[1]
    sorted_feature_idx = int(sorted_feature_idx)
    if sorted_feature_idx < 0 or sorted_feature_idx >= n_cols:
        raise ValueError(f"sorted_feature_idx must be in [0, n_cols-1]; got {sorted_feature_idx}, n_cols={n_cols}")

    # Filter spikes by minimum cluster index threshold
    valid_mask = clust_ind >= minimum_cluster_index
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return []  # No spikes pass the threshold
    
    # Copy filtered data to float32 arrays for contiguous memory and speed
    # This also ensures data is contiguous which accelerates operations
    n_valid = len(valid_indices)
    properties_filtered = properties[valid_indices, :].astype(np.float32)
    clust_ind_filtered = clust_ind[valid_indices].astype(np.float32)
    
    # Store mapping from filtered indices back to global indices
    # valid_indices[i] gives the global index of the i-th filtered spike

    # Get the sorted feature values from filtered data
    sorted_feature_values = properties_filtered[:, sorted_feature_idx]
    
    # Get distance for the sorted feature
    sorted_feature_distance = feature_distances.get(sorted_feature_idx, 0.0)
    
    # Convert spikes_per_batch_thousands to actual number
    max_dots_per_batch = int(spikes_per_batch_thousands * 1000)
    
    batches = []
    
    # First batch (using filtered data, so N is now n_valid)
    N_filtered = n_valid
    start = 0
    padded_start = 0
    # Calculate padded_end first: padded_start + max_dots_per_batch
    padded_end = min(padded_start + max_dots_per_batch, N_filtered)
    
    # Calculate end: last spike with sorted feature value below that of padded_end spike - distance
    if padded_end < N_filtered:
        padded_end_spike_feature_value = sorted_feature_values[padded_end - 1]
        threshold = padded_end_spike_feature_value - sorted_feature_distance
        
        # Find last spike where feature < threshold
        threshold_indices = np.where(sorted_feature_values < threshold)[0]
        if len(threshold_indices) > 0:
            end = threshold_indices[-1] + 1  # +1 to make it exclusive
        else:
            end = padded_start  # If no spikes below threshold, end equals padded_start
        del threshold_indices
    else:
        # If padded_end is at the end, end is also at the end
        end = N_filtered
    
    batches.append({
        'start': start,
        'end': end,
        'padded_start': padded_start,
        'padded_end': padded_end
    })
    
    # If we've reached the end, we're done with batch definition
    if end >= N_filtered or padded_end >= N_filtered:
        # Only one batch, continue to processing
        pass
    else:
        # Iteratively create subsequent batches
        while True:
            # Start of next batch is end of previous batch
            prev_batch = batches[-1]
            start = prev_batch['end']
            
            # Padded start: first spike where sorted_feature > feature[start] - distance
            start_spike_feature_value = sorted_feature_values[start]
            threshold_low = start_spike_feature_value - sorted_feature_distance
            
            # Find first spike where feature > threshold_low
            threshold_indices = np.where(sorted_feature_values > threshold_low)[0]
            if len(threshold_indices) > 0:
                padded_start = threshold_indices[0]
            else:
                padded_start = start
            del threshold_indices
            
            # Padded end: padded_start + max_dots_per_batch (or last spike)
            padded_end = min(padded_start + max_dots_per_batch, N_filtered)
            
            # Calculate end: last spike with sorted feature value below that of padded_end spike - distance
            if padded_end < N_filtered:
                padded_end_spike_feature_value = sorted_feature_values[padded_end - 1]
                threshold = padded_end_spike_feature_value - sorted_feature_distance
                
                # Find last spike where feature < threshold
                threshold_indices = np.where(sorted_feature_values < threshold)[0]
                if len(threshold_indices) > 0:
                    end = threshold_indices[-1] + 1  # +1 to make it exclusive
                else:
                    end = padded_start  # If no spikes below threshold, end equals padded_start
                del threshold_indices
            else:
                # If padded_end is at the end, end is also at the end
                end = N_filtered
            
            # Check if this is the last batch before running the assertion
            is_last_batch = (end >= N_filtered or padded_end >= N_filtered)
            
            # Assert that end > start + spikes_per_batch/4 to ensure batch is large enough
            # Skip this check for the last batch
            if not is_last_batch:
                min_required_end = start + (max_dots_per_batch // 4)
                if end <= min_required_end:
                    raise ValueError(
                        f"Batch definition error: end ({end}) must be > start + spikes_per_batch/4 "
                        f"({min_required_end}). This may indicate insufficient data or incorrect "
                        f"feature distance settings. start={start}, padded_start={padded_start}, "
                        f"padded_end={padded_end}, max_dots_per_batch={max_dots_per_batch}"
                    )
            
            batches.append({
                'start': start,
                'end': end,
                'padded_start': padded_start,
                'padded_end': padded_end
            })
            
            # Check if we've reached the end
            if is_last_batch:
                break
    
    # Initialize seeds list
    seeds = []
    
    # Get included feature indexes (all features that have distances defined)
    included_feature_indexes = sorted(feature_distances.keys())
    n_included_features = len(included_feature_indexes)
    
    # Pre-allocate reusable matrices for memory efficiency
    # We'll allocate them on the first batch and reuse for subsequent batches
    # Only reallocate if the last batch is smaller
    diff_matrix = None
    feature_comparisons = None
    max_batch_size = 0  # Track maximum batch size seen
    
    # Process batches to find seeds
    n_batches = len(batches)
    
    for batch_idx, batch in enumerate(batches):
        # 0- Display progress
        if progress_callback is not None:
            progress_callback(batch_idx + 1, n_batches)
        
        # Get batch indices
        padded_start = batch['padded_start']
        padded_end = batch['padded_end']
        start = batch['start']
        end = batch['end']
        
        # Get spikes in padded batch (from filtered data, already float32)
        # Views are fine - they don't copy data, just reference it
        batch_properties = properties_filtered[padded_start:padded_end, :]
        batch_clust_ind = clust_ind_filtered[padded_start:padded_end]
        spikes_in_batch = batch_properties.shape[0]
        
        if spikes_in_batch == 0:
            continue
        
        # Memory optimization: Instead of storing full (N, N, features+1) tensor,
        # we compute the result incrementally. However, we still need the full pairwise
        # matrices for each feature to check "for all s2" condition.
        
        # Initialize comparisons matrix: (spikes_in_batch, spikes_in_batch)
        # comparisons[i, j] will be True if spike j is within all feature distances of spike i
        # AND spike j has clust_ind >= spike i's clust_ind
        # A spike is a local max if there are NO other spikes satisfying these conditions
        if TORCH_AVAILABLE and DEVICE.type == 'cuda':
            comparisons = torch.ones((spikes_in_batch, spikes_in_batch), dtype=torch.bool, device=DEVICE)
        else:
            comparisons = np.ones((spikes_in_batch, spikes_in_batch), dtype=bool)
        
        # 2- Loop through included features
        for feat_idx_pos, feat_idx in enumerate(included_feature_indexes):
            # batch_properties is already float32, so no conversion needed
            feature_values = batch_properties[:, feat_idx]
            feature_distance = feature_distances[feat_idx]
            
            # Fast compute pairwise comparisons: abs(feature[s2] - feature[s1]) < feature_distance
            # Shape: (spikes_in_batch, spikes_in_batch)
            # feature_comparisons[i, j] is True if |feature[j] - feature[i]| < distance
            # Memory: spikes_in_batch^2 booleans (e.g., 20k^2 = 400M = 400 MB per feature)
            # Use GPU if available for faster computation
            if TORCH_AVAILABLE and DEVICE.type == 'cuda':
                # Transfer to GPU and compute on GPU
                feature_values_gpu = torch.from_numpy(feature_values).to(DEVICE)
                diff_matrix_gpu = torch.abs(feature_values_gpu[:, None] - feature_values_gpu[None, :])
                feature_comparisons_gpu = diff_matrix_gpu < feature_distance
                # Update comparisons: AND with feature_comparisons
                comparisons = comparisons & feature_comparisons_gpu
                # Free GPU memory (but keep feature_values_gpu for potential reuse if same size)
                del diff_matrix_gpu, feature_comparisons_gpu
                del feature_values_gpu
                torch.cuda.empty_cache()  # Clear GPU cache
            else:
                # CPU fallback - reuse allocated matrices if size matches
                # Allocate or resize matrices if needed
                if diff_matrix is None or diff_matrix.shape[0] != spikes_in_batch:
                    # Free old matrices if they exist and are wrong size
                    if diff_matrix is not None:
                        del diff_matrix
                    if feature_comparisons is not None:
                        del feature_comparisons
                    # Allocate new matrices
                    diff_matrix = np.empty((spikes_in_batch, spikes_in_batch), dtype=np.float32)
                    feature_comparisons = np.empty((spikes_in_batch, spikes_in_batch), dtype=bool)
                    max_batch_size = spikes_in_batch
                elif spikes_in_batch < max_batch_size:
                    # Last batch is smaller - use slice of pre-allocated matrices
                    diff_matrix_slice = diff_matrix[:spikes_in_batch, :spikes_in_batch]
                    feature_comparisons_slice = feature_comparisons[:spikes_in_batch, :spikes_in_batch]
                    # Compute pairwise differences into slice
                    diff_matrix_slice[:] = np.abs(feature_values[:, np.newaxis] - feature_values[np.newaxis, :])
                    feature_comparisons_slice[:] = diff_matrix_slice < feature_distance
                    # Update comparisons: AND with feature_comparisons
                    comparisons = comparisons & feature_comparisons_slice
                    continue  # Skip the rest of this iteration
                
                # Compute pairwise differences (reuse allocated matrix)
                diff_matrix[:] = np.abs(feature_values[:, np.newaxis] - feature_values[np.newaxis, :])
                feature_comparisons[:] = diff_matrix < feature_distance
                # Update comparisons: AND with feature_comparisons
                comparisons = comparisons & feature_comparisons
        
        # 3- Compute cluster index comparison: clust_ind[s2] >= clust_ind[s1]
        # Shape: (spikes_in_batch, spikes_in_batch)
        # clust_ind_matrix[i, j] is True if clust_ind[j] >= clust_ind[i]
        # Use GPU if available for faster computation
        if TORCH_AVAILABLE and DEVICE.type == 'cuda':
            # Transfer to GPU and compute on GPU
            batch_clust_ind_gpu = torch.from_numpy(batch_clust_ind).to(DEVICE)
            # clust_ind_matrix[i, j] = True if clust_ind[j] >= clust_ind[i]
            # This is: row vector >= column vector = (1, N) >= (N, 1)
            clust_ind_matrix_gpu = batch_clust_ind_gpu[None, :] >= batch_clust_ind_gpu[:, None]
            # Update comparisons: AND with clust_ind_matrix
            comparisons = comparisons & clust_ind_matrix_gpu
            # Free GPU memory
            del batch_clust_ind_gpu, clust_ind_matrix_gpu, batch_clust_ind
            torch.cuda.empty_cache()  # Clear GPU cache
        else:
            # CPU fallback
            # clust_ind_matrix[i, j] = True if clust_ind[j] >= clust_ind[i]
            # This is: row vector >= column vector = (1, N) >= (N, 1)
            clust_ind_matrix = batch_clust_ind[np.newaxis, :] >= batch_clust_ind[:, np.newaxis]
            # Update comparisons: AND with clust_ind_matrix
            comparisons = comparisons & clust_ind_matrix
            # Free memory
            del clust_ind_matrix, batch_clust_ind
        
        # 4- A spike is a local max if there are NO OTHER spikes (excluding itself) within all 
        # feature distances that have clust_ind >= the current spike's clust_ind
        # We need to exclude the diagonal (i == j) because a spike is always within distance of itself
        if TORCH_AVAILABLE and DEVICE.type == 'cuda':
            # Convert to numpy for final computation
            comparisons_cpu = comparisons.cpu().numpy()
            del comparisons
            torch.cuda.empty_cache()
            # Set diagonal to False (exclude self-comparison)
            np.fill_diagonal(comparisons_cpu, False)
            # is_local_max[i] = True if all comparisons[i, :] are False (no competing spikes)
            is_local_max = np.all(~comparisons_cpu, axis=1)
            del comparisons_cpu
        else:
            # Set diagonal to False (exclude self-comparison)
            np.fill_diagonal(comparisons, False)
            # is_local_max[i] = True if all comparisons[i, :] are False (no competing spikes)
            is_local_max = np.all(~comparisons, axis=1)
            del comparisons
        
        # 5- Set padded region spikes to False
        # Spikes in the padded region (but not in core) should be excluded
        # Core region: [start - padded_start : end - padded_start]
        core_start_local = start - padded_start
        core_end_local = end - padded_start
        
        # Set all spikes outside core region to False
        is_local_max[:core_start_local] = False
        is_local_max[core_end_local:] = False
        
        # 6- Find True spikes and add global indices to seeds
        # local_seed_indices are indices in the batch (relative to padded_start)
        local_seed_indices = np.where(is_local_max)[0]
        if len(local_seed_indices) > 0:
            # Map to filtered indices (indices in properties_filtered)
            filtered_seed_indices = padded_start + local_seed_indices
            # Map back to global indices using valid_indices mapping
            global_seed_indices = valid_indices[filtered_seed_indices]
            # Extend seeds list efficiently (avoid .tolist() if possible, but need list for return type)
            seeds.extend(global_seed_indices.tolist())
            del local_seed_indices, filtered_seed_indices, global_seed_indices
        
        # Clean up batch-specific arrays
        del batch_properties, is_local_max
        
        # Force garbage collection every few batches to prevent memory accumulation
        if (batch_idx + 1) % 10 == 0:
            gc.collect()
    
    # Clean up reusable matrices after all batches are processed
    if diff_matrix is not None:
        del diff_matrix
    if feature_comparisons is not None:
        del feature_comparisons
    
    if bounds is not None and len(bounds) > 0:
        def in_bounds(s):
            for fid in bounds:
                if fid >= properties.shape[1]:
                    continue
                min_val, max_val = bounds[fid]
                if min_val is not None and properties[s, fid] < min_val:
                    return False
                if max_val is not None and properties[s, fid] > max_val:
                    return False
            return True
        seeds = [s for s in seeds if in_bounds(s)]
    return seeds


def whiten_data(data, cov, center=None):
    """
    Whiten data using covariance matrix.
    
    Parameters
    ----------
    data : np.ndarray, shape (N, D)
        Data to whiten
    cov : np.ndarray, shape (D, D)
        Covariance matrix
    center : np.ndarray, shape (D,), optional
        Center to subtract before whitening. If None, uses mean of data.
        
    Returns
    -------
    whitened_data : np.ndarray, shape (N, D)
        Whitened data
    """
    if center is None:
        center = np.mean(data, axis=0)
    
    # Center the data
    centered_data = data - center
    
    # Compute eigendecomposition of covariance
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Ensure positive eigenvalues
    eigvals = np.maximum(eigvals, 1e-12)
    
    # Whitening transform: W = diag(1/sqrt(eigvals)) @ eigvecs.T
    # Then whitened_data = centered_data @ W.T
    inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
    W = eigvecs * inv_sqrt_eigvals[np.newaxis, :]
    
    whitened_data = centered_data @ W.T
    
    return whitened_data


def apply_bic_refinement(points, center, cov, in_bounds, bic_threshold, n_features, min_points_for_cluster=100, max_iter=10, seed_row=None):
    """
    Optionally refine in_bounds by GMM 1 vs 2 BIC; return refined in_bounds (indices into points).
    Data is whitened with the current covariance (and center) so the BIC comparison is in spherical space.
    If seed_row is not None, keep the cluster containing the seed; else keep the cluster containing the model center.
    """
    if len(in_bounds) < min_points_for_cluster:
        return in_bounds
    iteration_count = 0
    while iteration_count < max_iter:
        data = whiten_data(points[in_bounds, :], cov, center=center)
        gmm_1 = GaussianMixture(n_components=1, random_state=42, max_iter=100)
        gmm_2 = GaussianMixture(n_components=2, random_state=42, max_iter=100)
        gmm_1.fit(data)
        gmm_2.fit(data)
        bic_1 = gmm_1.bic(data)
        bic_2 = gmm_2.bic(data)
        dBIC = (bic_1 - bic_2) / len(data)
        if dBIC <= bic_threshold:
            break
        labels = gmm_2.predict(data)
        if seed_row is not None:
            seed_in_data = np.where(in_bounds == seed_row)[0]
            if len(seed_in_data) > 0:
                keep_label = gmm_2.predict(data[seed_in_data[0]:seed_in_data[0] + 1])[0]
                in_bounds = in_bounds[labels == keep_label]
            else:
                center_label = gmm_2.predict(np.zeros((1, n_features)))
                in_bounds = in_bounds[labels == center_label[0]]
        else:
            center_label = gmm_2.predict(np.zeros((1, n_features)))
            in_bounds = in_bounds[labels == center_label[0]]
        iteration_count += 1
    return in_bounds


def iterate_GM_model(properties, weights, model, sort_feature_idx, max_range_sorted,
                     seed_idx, min_change=0.01, min_points_for_cluster=100, curr_max_prob=None):
    """
    Iteratively refine a Gaussian model (COM and covariance). Caller passes a copy of the model
    so the original is only updated depending on outcome. The window (points) and valid_indices
    are derived from model.data_range; seed_row is derived from seed_idx (global index in properties).
    Range checks apply only to the sorted feature dimension.

    Parameters
    ----------
    properties : np.ndarray, shape (M, n_features)
        Full features tensor (must be sorted by model.sort_feature_idx).
    weights : np.ndarray, shape (M,)
        Cluster density (e.g. HC) for each row of properties.
    model : GaussianModel
        A copy of the current model; model.data_range defines the window [first_idx, last_idx].
    sort_feature_idx : int
        Column index of the sorted feature; range is checked only for this dimension.
    max_range_sorted : float
        Maximum allowed model range (2*bounds*std) for the sorted feature; if exceeded, outcome 'exploded'.
    seed_idx : int
        Global row index of the seed in properties (used by BIC refinement to select which cluster to keep).
    min_change : float, optional
        Relative change threshold for stable vs grow/shrink (default 0.01).
    min_points_for_cluster : int, optional
        Minimum points for BIC refinement (default 100).
    curr_max_prob : np.ndarray, shape (M,)
        Per-spike current_max_prob (background density or previous prob_density). in_bounds uses
        prob_density > curr_max_prob (prob_density from density_curve or linear decay). Required.

    Returns
    -------
    outcome : str
        'collapsed', 'exploded', or 'stable'.
    model : GaussianModel
        On stable: model with updated mean and covariance. On collapsed/exploded: model with mean and covariance set to NaN.
    assignment : np.ndarray or float
        On stable: in_bounds indices into the window (row indices 0..len(points)-1). On collapsed/exploded: np.nan.
    reason : str or None
        On exploded: reason string ('repeated BIC refinement' or 'max range exceeded (sorted feature)'). Otherwise None.
    """
    if model.data_range is None:
        raise ValueError("model.data_range must be set (e.g. via model.compute_data_range(properties)) before calling iterate_GM_model")
    if curr_max_prob is None or len(curr_max_prob) < properties.shape[0]:
        raise ValueError("curr_max_prob is required and must have same length as spikes (number of rows in properties)")
    curr_max_prob = np.asarray(curr_max_prob, dtype=float)
    if np.any(np.isnan(curr_max_prob)) or np.any((curr_max_prob < 0) & np.isfinite(curr_max_prob)):
        raise ValueError("curr_max_prob must contain only valid (non-negative finite) values or inf")
    first_idx, last_idx = model.data_range
    points = properties[first_idx:last_idx + 1, :]
    weights_window = weights[first_idx:last_idx + 1]
    seed_row = (seed_idx - first_idx) if first_idx <= seed_idx <= last_idx else None

    bounds = model.mah_threshold
    bic_threshold = model.bic_threshold
    n_features = points.shape[1]
    collapsed = False
    exploded = False
    explode_reason = None
    BIC_refined = False

    assignment_global = model.in_bounds_indices(properties, curr_max_prob)
    assignment = assignment_global - first_idx

    max_inner_iter = 200
    inner_iter = 0
    while True:
        inner_iter += 1
        if inner_iter > max_inner_iter:
            collapsed = True
            break
        # Save current assignment for length comparison (no index transfer needed)
        previous_assignment = np.copy(assignment)

        if len(assignment) == 0:
            collapsed = True
            break
        # Refresh mean, covariance, and core/fringe densities from in-bounds data
        model.refresh_covs_and_densities(points[assignment], weights_window[assignment])
        new_center = model.mean
        new_cov = model.covariance
        if np.any(np.isnan(new_center)):
            collapsed = True
            break
        # Model range check: only sorted feature dimension; range = 2*bounds*std
        stds = np.sqrt(np.maximum(np.diag(new_cov), 0.0))
        range_sorted = 2.0 * bounds * stds[sort_feature_idx]
        if range_sorted < 1e-9:
            collapsed = True
            break
        if range_sorted > max_range_sorted:
            exploded = True
            explode_reason = "max range exceeded (sorted feature)"
            break

        model.compute_data_range(properties)
        first_idx, last_idx = model.data_range
        points = properties[first_idx:last_idx + 1, :]
        weights_window = weights[first_idx:last_idx + 1]
        seed_row = (seed_idx - first_idx) if first_idx <= seed_idx <= last_idx else None

        # Recompute in-bounds (assignment) on new points, including BIC refinement
        assignment_global = model.in_bounds_indices(properties, curr_max_prob)
        assignment = assignment_global - first_idx

        n_before_bic = len(assignment)
        assignment = apply_bic_refinement(
            points, new_center, new_cov, assignment,
            bic_threshold, n_features,
            min_points_for_cluster=min_points_for_cluster,
            max_iter=10, seed_row=seed_row
        )
        bic_did_refine = len(assignment) < n_before_bic

        if bic_did_refine:
            if BIC_refined:
                exploded = True
                explode_reason = "repeated BIC refinement"
                break
            BIC_refined = True
            continue

        n_prev = len(previous_assignment)
        n_after = len(assignment)

        if n_after < n_prev - min_change * n_prev:
            collapsed = True
            break
        if n_after > n_prev + min_change * n_prev:
            continue
        break

    if collapsed or exploded:
        outcome = 'collapsed' if collapsed else 'exploded'
        model.mean = np.full(n_features, np.nan)
        model.covariance = np.full((n_features, n_features), np.nan)
        assignment = np.nan
    else:
        outcome = 'stable'

    reason = explode_reason if exploded else None
    return outcome, model, assignment, reason


def prob_density_from_curve_or_formula(mahal_d, mah_th, density_curve=None):
    """Compute prob_density: if density_curve is set, interpolate from curve at rel=mahal_d/mah_th; else linear decay max(0, 1 - rel). Anything with mahal_d > mah_th is 0. Returns array or scalar."""
    mahal_d = np.asarray(mahal_d, dtype=float)
    if mah_th <= 0:
        return (np.zeros_like(mahal_d) if np.ndim(mahal_d) else 0.0)
    rel = np.clip(mahal_d / mah_th, 0.0, 1.0)
    if density_curve is not None and len(density_curve) > 0:
        x_curve = np.linspace(0, 1, len(density_curve))
        out = np.interp(rel, x_curve, np.asarray(density_curve, dtype=float))
        out = np.maximum(out, 0.0)
    else:
        out = np.maximum(1.0 - rel, 0.0)
    # Above threshold: probability 0
    out = np.where(mahal_d <= mah_th, out, 0.0)
    return out


class GaussianModel:
    """
    Gaussian model (mean, covariance, Mahalanobis threshold, BIC threshold) with optional
    data range and sort range for the sorted dimension.

    Attributes
    ----------
    mean : np.ndarray, shape (n_features,)
        Mean vector of the Gaussian distribution
    covariance : np.ndarray, shape (n_features, n_features)
        Covariance matrix of the Gaussian distribution
    bic_threshold : float
        BIC threshold for multi-cluster refinement
    mah_threshold : float
        Mahalanobis distance threshold (boundary)
    data_range : tuple (first_idx, last_idx), optional
        First and last row index (inclusive) of data in sorted-dim range
    sort_range : tuple (sort_lo, sort_hi), optional
        Sorted-dim value range of the model
    sort_feature_idx : int, optional
        Index of the sorted feature in the feature tensor used for fitting
    density_curve : np.ndarray, optional
        If set, shape (101,) density at relative Mahal 0, 0.01, ..., 1.0; prob_density is computed by linear interpolation from this.
    """
    def __init__(self, mean, covariance, bic_threshold, mah_threshold,
                 data_range=None, sort_range=None, sort_feature_idx=None, density_curve=None):
        self.mean = mean
        self.covariance = covariance
        self.bic_threshold = bic_threshold
        self.mah_threshold = mah_threshold
        self.data_range = data_range
        self.sort_range = sort_range
        self.sort_feature_idx = sort_feature_idx
        self.density_curve = density_curve

    def refresh_covs_and_densities(self, points, cluster_densities):
        """
        Update mean and covariance from in-bound points and their cluster densities.
        Center = cluster-density-weighted COM; covariance = unweighted sample covariance (regularized);
        Parameters
        ----------
        points : np.ndarray, shape (n, n_features)
            In-bound points (same feature space as self.mean).
        cluster_densities : np.ndarray, shape (n,)
            Cluster density for each point.

        Returns
        -------
        None (updates self in place).
        """
        points = np.asarray(points, dtype=float)
        cluster_densities = np.asarray(cluster_densities, dtype=float)
        n = points.shape[0]
        n_features = points.shape[1]
        if n == 0:
            return
        if n == 1:
            self.mean = np.copy(points[0])
            return
        w = cluster_densities
        Nk = np.sum(w)
        if Nk <= 0:
            Nk = 1.0
            w = np.ones(n, dtype=float)
        weighted_sum = np.sum(w[:, np.newaxis] * points, axis=0)
        new_center = weighted_sum / Nk
        diff_c = points - new_center
        new_cov = np.einsum('ni,nj->ij', diff_c, diff_c) / n
        new_cov = 0.5 * (new_cov + new_cov.T)
        reg = max(1e-10, 1e-8 * np.trace(new_cov) / n_features)
        new_cov = new_cov + reg * np.eye(n_features)
        self.mean = np.copy(new_center)
        self.covariance = np.copy(new_cov)

    def copy(self):
        """Return a copy of this model (same params; caller may pass copy to iterate without mutating this)."""
        return GaussianModel(
            mean=np.copy(self.mean),
            covariance=np.copy(self.covariance),
            bic_threshold=self.bic_threshold,
            mah_threshold=self.mah_threshold,
            data_range=self.data_range,
            sort_range=self.sort_range,
            sort_feature_idx=self.sort_feature_idx,
            density_curve=np.copy(self.density_curve) if self.density_curve is not None else None,
        )

    def in_bounds_indices(self, features, current_max_prob):
        """
        Return row indices into the full feature tensor that lie within this model's boundary.
        Uses self.data_range to define the window; only rows in that range are considered.
        If self.data_range is None, calls self.compute_data_range(features) first (requires
        sort_feature_idx to be set; features must be sorted by that column).
        Indices returned are global (into the full features array).

        Row i is in bounds iff prob_density > current_max_prob[i].
        prob_density from density_curve (interpolation) or linear decay max(0, 1 - mahal_d/mah_th).

        Parameters
        ----------
        features : np.ndarray, shape (N, n_features)
            Full feature tensor; n_features must match len(self.mean).
        current_max_prob : np.ndarray, shape (N,)
            Per-spike current max probability (or background density); same length as features. Required.

        Returns
        -------
        np.ndarray
            Integer array of row indices into features (global indices) that are in bounds.
        """
        features = np.asarray(features, dtype=float)
        n_full = features.shape[0]
        if features.shape[1] != len(self.mean):
            raise ValueError("features second dimension must match model mean length")
        if current_max_prob is None:
            raise ValueError("current_max_prob is required")
        current_max_prob = np.asarray(current_max_prob, dtype=float)
        if len(current_max_prob) < n_full:
            raise ValueError("current_max_prob length must be >= number of rows in features")
        if self.data_range is not None:
            first_idx, last_idx = self.data_range
            last_idx = min(last_idx, n_full - 1)
            first_idx = min(first_idx, last_idx)
            window = features[first_idx:last_idx + 1, :]
            cmp_slice = current_max_prob[first_idx:last_idx + 1]
        else:
            self.compute_data_range(features)
            first_idx, last_idx = self.data_range
            last_idx = min(last_idx, n_full - 1)
            first_idx = min(first_idx, last_idx)
            window = features[first_idx:last_idx + 1, :]
            cmp_slice = current_max_prob[first_idx:last_idx + 1]
        n = window.shape[0]
        diff = window - self.mean
        inv_cov = np.linalg.pinv(self.covariance)
        mahal_d = np.einsum('ij,jk,ik->i', diff, inv_cov, diff) ** 0.5
        mah_th = float(self.mah_threshold)
        within_mah = mahal_d <= mah_th
        prob_density = prob_density_from_curve_or_formula(
            mahal_d, mah_th, density_curve=getattr(self, 'density_curve', None)
        )
        local_in = np.where(within_mah & (prob_density > cmp_slice[:n]))[0]
        return first_idx + local_in

    def compute_data_range(self, properties):
        """
        Compute (first_idx, last_idx) and (sort_lo, sort_hi) from current model params using sort search.
        properties must be sorted by sort_feature_idx. Updates self.data_range and self.sort_range;
        uses existing self.data_range and self.sort_range as previous state when present.
        Returns (data_range, sort_range).
        """
        if self.sort_feature_idx is None:
            raise ValueError("GaussianModel.sort_feature_idx must be set to compute_data_range")
        sorted_feat_values = np.asarray(properties[:, self.sort_feature_idx], dtype=float)
        n = len(sorted_feat_values)
        std = np.sqrt(max(0.0, float(self.covariance[self.sort_feature_idx, self.sort_feature_idx])))
        if std <= 0:
            std = 1.0
        center_sorted = float(self.mean[self.sort_feature_idx])
        sort_lo = center_sorted - std * self.mah_threshold
        sort_hi = center_sorted + std * self.mah_threshold

        prev_data_range = self.data_range
        prev_sort_range = self.sort_range
        if prev_data_range is not None and prev_sort_range is not None:
            prev_first, prev_last = prev_data_range
            prev_sort_lo, prev_sort_hi = prev_sort_range
            prev_last = min(prev_last, n - 1)
            if prev_first <= prev_last:
                if sort_lo >= prev_sort_lo:
                    first_idx = prev_first + np.searchsorted(sorted_feat_values[prev_first:], sort_lo, side='left')
                else:
                    first_idx = np.searchsorted(sorted_feat_values[:prev_first + 1], sort_lo, side='left')
                if sort_hi <= prev_sort_hi:
                    last_idx = first_idx + np.searchsorted(sorted_feat_values[first_idx:prev_last + 1], sort_hi, side='right') - 1
                else:
                    last_idx = first_idx + np.searchsorted(sorted_feat_values[first_idx:], sort_hi, side='right') - 1
            else:
                first_idx = np.searchsorted(sorted_feat_values, sort_lo, side='left')
                last_idx = first_idx + np.searchsorted(sorted_feat_values[first_idx:], sort_hi, side='right') - 1
        else:
            first_idx = np.searchsorted(sorted_feat_values, sort_lo, side='left')
            last_idx = first_idx + np.searchsorted(sorted_feat_values[first_idx:], sort_hi, side='right') - 1

        last_idx = min(max(last_idx, first_idx), n - 1)
        first_idx = min(first_idx, last_idx)
        self.data_range = (int(first_idx), int(last_idx))
        self.sort_range = (sort_lo, sort_hi)
        return self.data_range, self.sort_range

    def pdf(self, points):
        """
        Compute multivariate normal PDF for given points.
        
        Parameters
        ----------
        points : np.ndarray, shape (N, n_features)
            Points to evaluate PDF at
            
        Returns
        -------
        pdf_values : np.ndarray, shape (N,)
            PDF values for each point
        """
        mvn = multivariate_normal(mean=self.mean, cov=self.covariance, allow_singular=True)
        return mvn.pdf(points)


def _create_init_cluster_from_seed(properties, seed_idx, cluster_densities, curr_max_prob, settings):
    """
    Build initial Gaussian model from a seed: spatial window, COM refinement, covariance, and boundary (mah_th/size).
    Returns either a failure dict or a success dict with 'model' and 'in_bounds' for use by _fit_model_size.
    """
    n_features = properties.shape[1]
    n_spikes = properties.shape[0]

    sorted_feature_idx = int(settings.get('sorted_feature_idx', 0))
    if sorted_feature_idx < 0 or sorted_feature_idx >= n_features:
        sorted_feature_idx = 0
    min_points_for_cluster = int(settings.get('min_points_for_cluster', 100))
    initial_stds = settings.get('initial_stds')
    if initial_stds is None or len(np.atleast_1d(initial_stds)) != n_features:
        initial_stds = np.ones(n_features, dtype=float)
    initial_stds = np.asarray(initial_stds, dtype=float).ravel()[:n_features]
    if len(initial_stds) < n_features:
        initial_stds = np.ones(n_features, dtype=float)
    init_mah_d = float(settings.get('init_mah_th', 1.0))
    com_iteration_threshold = settings.get('com_iteration_threshold', 0.25)
    max_iterations = int(settings.get('com_iteration_max_iterations', 50))
    density_threshold_for_init_distance = settings.get('density_threshold_for_init_distance', 0.1)
    gaussian_filter_sigma = settings.get('gaussian_filter_sigma', 50.0)
    dist_step = settings.get('dist_step', 0.1)
    min_change = settings.get('min_change', 0.01)
    multi_cluster_threshold = settings.get('multi_cluster_threshold', 0.2)
    max_range_sorted = float(settings.get('max_range_sorted', initial_stds[sorted_feature_idx] * 4.0))
    n_samples_density_curve = int(settings.get('n_samples_density_curve', 101))
    max_dist_scale = max_range_sorted / (2.0 * initial_stds[sorted_feature_idx])

    curr_max_prob = np.asarray(curr_max_prob, dtype=float)
    if len(curr_max_prob) != n_spikes:
        return {'success': False, 'message': f'curr_max_prob length ({len(curr_max_prob)}) must match n_spikes ({n_spikes})'}
    if np.any(np.isnan(curr_max_prob)) or np.any((curr_max_prob < 0) & np.isfinite(curr_max_prob)):
        return {'success': False, 'message': 'curr_max_prob must contain only valid (non-negative finite) values or inf'}

    if seed_idx < 0 or seed_idx >= n_spikes:
        return {'success': False, 'message': f'Invalid seed index: {seed_idx}'}
    if len(initial_stds) != n_features:
        return {'success': False, 'message': f'initial_stds length ({len(initial_stds)}) does not match number of features ({n_features})'}
    if sorted_feature_idx < 0 or sorted_feature_idx >= n_features:
        return {'success': False, 'message': f'Invalid sorted_feature_idx: {sorted_feature_idx} (must be in [0, {n_features-1}])'}

    seed_sorted_feat_value = properties[seed_idx, sorted_feature_idx]
    half_window = max_range_sorted / 2.0
    lower_bound = seed_sorted_feat_value - half_window
    upper_bound = seed_sorted_feat_value + half_window
    sorted_feat_values = properties[:, sorted_feature_idx]
    start_idx = np.searchsorted(sorted_feat_values, lower_bound, side='left')
    end_idx = np.searchsorted(sorted_feat_values, upper_bound, side='right')
    valid_indices = np.arange(start_idx, end_idx)
    valid_indices_all = valid_indices.copy()

    if len(valid_indices) == 0:
        return {'success': False, 'message': f'No points within max range (sorted feature) of seed'}

    seed_in_valid = np.searchsorted(valid_indices, seed_idx)
    if seed_in_valid >= len(valid_indices) or valid_indices[seed_in_valid] != seed_idx:
        raise RuntimeError(
            'Seed point not found in valid range. Properties must be sorted by sorted_feature_idx. '
            f'seed_idx={seed_idx}, valid_indices=[{valid_indices[0]}..{valid_indices[-1]}]'
        )

    points = properties[valid_indices, :]
    HC = cluster_densities[valid_indices]

    m = np.copy(properties[seed_idx, :])
    covs = np.zeros((n_features, n_features))
    np.fill_diagonal(covs, initial_stds**2)
    inv_cov = np.linalg.pinv(covs)

    diff = points - m
    mahal_d = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)**0.5
    min_mahal_init = np.min(mahal_d)
    if min_mahal_init > max_dist_scale * 2:
        return {'success': False, 'message': 'Local vicinity too sparse'}

    init_model = GaussianModel(mean=m, covariance=covs, bic_threshold=multi_cluster_threshold, mah_threshold=init_mah_d, data_range=None, sort_feature_idx=sorted_feature_idx)
    init_model.compute_data_range(properties)
    first_idx, last_idx = init_model.data_range
    in_bounds_global = init_model.in_bounds_indices(properties, curr_max_prob)
    in_bounds = np.where(np.isin(valid_indices, in_bounds_global))[0]
    if len(in_bounds) < min_points_for_cluster:
        if len(mahal_d) >= min_points_for_cluster:
            top_k = np.partition(mahal_d, min_points_for_cluster - 1)[min_points_for_cluster - 1]
            in_bounds = np.where(mahal_d <= top_k)[0]
        else:
            return {'success': False, 'message': f'Insufficient points within initial threshold: {len(in_bounds)} < {min_points_for_cluster}'}

    debug_init_cb = settings.get('debug_init_callback')
    if debug_init_cb is not None:
        debug_init_cb({
            'points': points.copy(), 'center': m.copy(), 'covariance': covs.copy(),
            'valid_indices': valid_indices.copy(), 'in_bounds': in_bounds.copy(),
            'mahal_d': mahal_d.copy(), 'seed_idx': seed_idx, 'properties': properties,
            'init_mah_d': init_mah_d, 'curr_max_prob': curr_max_prob[valid_indices].copy(),
        })

    Nk = np.sum(HC[in_bounds])
    weighted_sum = np.sum(HC[in_bounds, np.newaxis] * points[in_bounds, :], axis=0)
    COM = weighted_sum / Nk
    diff_COM = (COM - m)[np.newaxis, :]
    mahal_COM = np.einsum('ij,jk,ik->i', diff_COM, inv_cov, diff_COM)**0.5
    iteration_count = 0

    com_thresh_abs = com_iteration_threshold * init_mah_d
    com_model = GaussianModel(mean=COM.copy(), covariance=covs, bic_threshold=multi_cluster_threshold, mah_threshold=init_mah_d, data_range=(first_idx, last_idx), sort_feature_idx=sorted_feature_idx)
    while mahal_COM[0] > com_thresh_abs and iteration_count < max_iterations:
        m = COM
        com_model.mean[:] = m
        com_model.compute_data_range(properties)
        first_idx, last_idx = com_model.data_range
        in_bounds_global = com_model.in_bounds_indices(properties, curr_max_prob)
        in_bounds = np.where(np.isin(valid_indices, in_bounds_global))[0]
        if len(in_bounds) == 0:
            break
        Nk = np.sum(HC[in_bounds])
        weighted_sum = np.sum(HC[in_bounds, np.newaxis] * points[in_bounds, :], axis=0)
        COM = weighted_sum / Nk
        diff_COM = (COM - m)[np.newaxis, :]
        mahal_COM = np.einsum('ij,jk,ik->i', diff_COM, inv_cov, diff_COM)**0.5
        iteration_count += 1

    if len(in_bounds) < min_points_for_cluster:
        return {'success': False, 'message': f'Insufficient points after COM iteration: {len(in_bounds)} < {min_points_for_cluster}'}
    in_bounds = apply_bic_refinement(
        points, m, covs, in_bounds, multi_cluster_threshold, n_features,
        min_points_for_cluster=min_points_for_cluster, seed_row=seed_in_valid
    )
    if len(in_bounds) < min_points_for_cluster:
        return {'success': False, 'message': f'Insufficient points after BIC refinement (COM): {len(in_bounds)} < {min_points_for_cluster}'}
    Nk = np.sum(HC[in_bounds])
    weighted_sum = np.sum(HC[in_bounds, np.newaxis] * points[in_bounds, :], axis=0)
    m = weighted_sum / Nk
    diff_c = points[in_bounds, :] - m
    n_in = len(in_bounds)
    covs = np.einsum('ni,nj->ij', diff_c, diff_c) / n_in
    covs = 0.5 * (covs + covs.T)
    inv_cov = np.linalg.pinv(covs)
    diff = points - m
    mahal_d = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)**0.5
    stds = np.sqrt(np.maximum(np.diag(covs), 1e-12))
    max_dist_for_init_bound = max_range_sorted / (2.0 * stds[sorted_feature_idx])

    in_max = np.where(mahal_d < max_dist_for_init_bound)[0]
    if len(in_max) == 0:
        min_mahal = np.min(mahal_d) if len(mahal_d) > 0 else float('inf')
        max_mahal = np.max(mahal_d) if len(mahal_d) > 0 else float('inf')
        return {
            'success': False,
            'message': f'No points within max distance ({max_dist_for_init_bound}) for boundary detection. Mahalanobis distances: min={min_mahal:.3f}, max={max_mahal:.3f}, mean={np.mean(mahal_d):.3f}'
        }

    points_w = whiten_data(points[in_max, :], covs, center=m)
    mah_sort_ind = np.argsort(mahal_d[in_max])
    mahal_d_s = mahal_d[in_max[mah_sort_ind]]
    points_w_sort = points_w[mah_sort_ind, :]
    HC_sorted = HC[in_max[mah_sort_ind]]
    rolling_HC = gaussian_filter1d(HC_sorted, gaussian_filter_sigma)
    min_HC = np.min(rolling_HC)
    crossing_threshold = (rolling_HC[0] * density_threshold_for_init_distance) + min_HC
    stops = np.where(rolling_HC < crossing_threshold)[0]
    mah_th = mahal_d_s[stops[0]]
    if mah_th < mahal_d_s[0]:
        mah_th = mahal_d_s[0]
    mah_th = np.round(mah_th / dist_step) * dist_step

    mah_th_max = max_range_sorted / (2.0 * stds[sorted_feature_idx])
    mah_th = min(mah_th, float(mah_th_max))
    mah_th = np.round(mah_th / dist_step) * dist_step

    if mah_th > 0 and len(mahal_d_s) > 0:
        x_rel = mahal_d_s / mah_th
        x_query = np.linspace(0, 1, n_samples_density_curve)
        density_curve = np.interp(x_query, x_rel, rolling_HC).astype(float)
        density_curve = np.maximum(density_curve, 0.0)
    else:
        density_curve = None

    model_after_mah_th = GaussianModel(mean=m, covariance=covs, bic_threshold=multi_cluster_threshold, mah_threshold=mah_th, data_range=(first_idx, last_idx), density_curve=density_curve)
    in_bounds_mah_th_global = model_after_mah_th.in_bounds_indices(properties, curr_max_prob)
    in_bounds_mah_th = np.where(np.isin(valid_indices, in_bounds_mah_th_global))[0]
    in_bounds_mah_th = apply_bic_refinement(
        points, m, covs, in_bounds_mah_th, multi_cluster_threshold, n_features,
        min_points_for_cluster=min_points_for_cluster, seed_row=seed_in_valid
    )
    in_bounds_mah_th_global = valid_indices[in_bounds_mah_th]

    debug_cb = settings.get('debug_after_com_callback')
    if debug_cb is not None:
        debug_cb({
            'points': points, 'center': m.copy(), 'covariance': covs.copy(),
            'valid_indices': valid_indices.copy(), 'in_bounds': in_bounds_mah_th.copy(),
            'mahal_d': mahal_d.copy(), 'seed_idx': seed_idx, 'properties': properties,
            'init_mah_d': init_mah_d, 'mah_th': float(mah_th), 'mahal_d_s': mahal_d_s.copy(),
            'rolling_HC': rolling_HC.copy(), 'density_curve': density_curve.copy() if density_curve is not None else None,
        })

    in_bounds = in_bounds_mah_th
    if len(in_bounds) < min_points_for_cluster:
        min_mahal = np.min(mahal_d) if len(mahal_d) > 0 else float('inf')
        max_mahal = np.max(mahal_d) if len(mahal_d) > 0 else float('inf')
        fail_model = GaussianModel(mean=m, covariance=covs, bic_threshold=multi_cluster_threshold, mah_threshold=mah_th, density_curve=density_curve)
        fail_viz = {
            'points': points.copy(), 'mahal_d': mahal_d.copy(), 'seed_point': np.copy(properties[seed_idx, :]),
            'valid_indices': valid_indices.copy(), 'in_bounds': in_bounds.copy(),
            'iteration_history_full': [], 'all_points': points.copy(), 'all_valid_indices': valid_indices.copy(),
            'success': False,
        }
        debug_gm_cb = settings.get('debug_after_gm_callback')
        if debug_gm_cb is not None:
            debug_gm_cb(fail_viz, fail_model)
        return {
            'success': False,
            'message': f'Insufficient points within boundary threshold: {len(in_bounds)} < {min_points_for_cluster}. mah_th={mah_th:.3f}, Mahalanobis distances: min={min_mahal:.3f}, max={max_mahal:.3f}, mean={np.mean(mahal_d):.3f}',
            'debug_data': {'points': points, 'mahal_d': mahal_d, 'mah_th': mah_th, 'center': m, 'covariance': covs}
        }

    model = GaussianModel(
        mean=m.copy(), covariance=covs.copy(), bic_threshold=multi_cluster_threshold, mah_threshold=mah_th,
        data_range=None, sort_range=None, sort_feature_idx=sorted_feature_idx, density_curve=density_curve,
    )
    model.compute_data_range(properties)
    first_idx, last_idx = model.data_range
    valid_indices = np.arange(first_idx, last_idx + 1)
    if seed_idx < first_idx or seed_idx > last_idx:
        return {'success': False, 'message': f'Seed index {seed_idx} outside model range after window update (data_range={model.data_range})'}
    valid_indices_all = valid_indices.copy()
    points = properties[valid_indices, :]
    HC = cluster_densities[valid_indices]
    seed_row = np.where(valid_indices == seed_idx)[0]
    seed_row = int(seed_row[0]) if len(seed_row) > 0 else None

    in_bounds_global = model.in_bounds_indices(properties, curr_max_prob)
    in_bounds = (in_bounds_global - valid_indices[0]).astype(int)
    return {'success': True, 'model': model, 'in_bounds': in_bounds}


def _create_init_cluster_from_cluster_data(properties, cluster_densities, curr_max_prob, cluster_indices, settings):
    """
    Build initial Gaussian model from cluster seed using the FULL properties tensor.
    1) COM and covariance from in_bounds only (cluster_indices = dots inside cluster seed).
    2) Mahalanobis distance for ALL dots in properties; find range of data within max distance (settings).
    3) Rolling HC curve on ALL that data (sorted by Mahal).
    4) Model size (mah_th) from the rolling HC curve.
    5) Probability curve from that size and HC curve.
    6) Initial model: COM/cov from cluster seed, size/density from full-window data, BIC from defaults.
    properties/cluster_densities/curr_max_prob are full arrays (N,); cluster_indices are row indices of the seed.
    Returns dict with 'model', 'in_bounds' (indices into data window) for _fit_model_size.
    """
    cluster_indices = np.asarray(cluster_indices, dtype=int).ravel()
    n_features = properties.shape[1]
    n_points = properties.shape[0]
    points = np.asarray(properties, dtype=float)
    HC = np.asarray(cluster_densities, dtype=float)
    curr_max_prob = np.asarray(curr_max_prob, dtype=float)
    if len(HC) != n_points or len(curr_max_prob) != n_points:
        return {'success': False, 'message': 'cluster_densities and curr_max_prob must match properties length'}
    if len(cluster_indices) < 2:
        return {'success': False, 'message': 'Cluster seed must have at least 2 points'}
    if np.any(cluster_indices < 0) or np.any(cluster_indices >= n_points):
        return {'success': False, 'message': 'cluster_indices out of range'}

    sorted_feature_idx = int(settings.get('sorted_feature_idx', 0))
    if sorted_feature_idx < 0 or sorted_feature_idx >= n_features:
        sorted_feature_idx = 0
    min_points_for_cluster = int(settings.get('min_points_for_cluster', 100))
    multi_cluster_threshold = float(settings.get('multi_cluster_threshold', 0.2))
    gaussian_filter_sigma = float(settings.get('gaussian_filter_sigma', 50.0))
    dist_step = float(settings.get('dist_step', 0.1))
    density_threshold_for_init_distance = float(settings.get('density_threshold_for_init_distance', 0.1))
    initial_stds = np.asarray(settings.get('initial_stds', np.ones(n_features)), dtype=float).ravel()[:n_features]
    if len(initial_stds) < n_features:
        initial_stds = np.ones(n_features, dtype=float)
    max_range_sorted = float(settings.get('max_range_sorted', initial_stds[sorted_feature_idx] * 4.0))
    n_samples_density_curve = int(settings.get('n_samples_density_curve', 101))

    # 1) COM and covariance from in_bounds only (cluster seed points)
    pts_seed = points[cluster_indices]
    HC_seed = HC[cluster_indices]
    Nk = np.sum(HC_seed)
    if Nk <= 0:
        return {'success': False, 'message': 'Cluster seed densities sum must be positive'}
    weighted_sum = np.sum(HC_seed[:, np.newaxis] * pts_seed, axis=0)
    m = weighted_sum / Nk
    diff_c = pts_seed - m
    covs = np.einsum('ni,nj->ij', diff_c, diff_c) / len(cluster_indices)
    covs = 0.5 * (covs + covs.T)
    inv_cov = np.linalg.pinv(covs)
    diff = points - m
    mahal_d = np.einsum('ij,jk,ik->i', diff, inv_cov, diff) ** 0.5
    stds = np.sqrt(np.maximum(np.diag(covs), 1e-12))
    # Max Mahal distance such that range in sorted dimension = 2*d*std_sorted equals max_range_sorted
    max_dist_for_init_bound = max_range_sorted / (2.0 * stds[sorted_feature_idx])

    # 2) Restrict to dots within max distance in the FULL properties for rolling HC / size / density curve
    in_max = np.where(mahal_d < max_dist_for_init_bound)[0]
    if len(in_max) == 0:
        min_mahal = float(np.min(mahal_d)) if len(mahal_d) > 0 else float('inf')
        max_mahal = float(np.max(mahal_d)) if len(mahal_d) > 0 else float('inf')
        return {
            'success': False,
            'message': f'No points within max distance ({max_dist_for_init_bound}) for boundary detection. Mahalanobis distances: min={min_mahal:.3f}, max={max_mahal:.3f}, mean={np.mean(mahal_d):.3f}'
        }
    # 3) Rolling HC curve on ALL that data (full tensor subset) sorted by Mahal
    points_w = whiten_data(points[in_max, :], covs, center=m)
    mah_sort_ind = np.argsort(mahal_d[in_max])
    mahal_d_s = mahal_d[in_max[mah_sort_ind]]
    HC_sorted = HC[in_max[mah_sort_ind]]
    rolling_HC = gaussian_filter1d(HC_sorted.astype(float), gaussian_filter_sigma, mode='nearest')
    min_HC = np.min(rolling_HC)
    crossing_threshold = (rolling_HC[0] * density_threshold_for_init_distance) + min_HC
    stops = np.where(rolling_HC < crossing_threshold)[0]
    if len(stops) == 0:
        mah_th = np.max(mahal_d_s)
    else:
        mah_th = float(mahal_d_s[stops[0]])
    if mah_th < mahal_d_s[0]:
        mah_th = float(mahal_d_s[0])
    mah_th = np.round(mah_th / dist_step) * dist_step

    mah_th_max = max_range_sorted / (2.0 * stds[sorted_feature_idx])
    mah_th = min(mah_th, float(mah_th_max))
    mah_th = np.round(mah_th / dist_step) * dist_step

    # 5) Probability curve from that size and HC curve
    if mah_th > 0 and len(mahal_d_s) > 0:
        x_rel = mahal_d_s / mah_th
        x_query = np.linspace(0, 1, n_samples_density_curve)
        density_curve = np.interp(x_query, x_rel, rolling_HC).astype(float)
        density_curve = np.maximum(density_curve, 0.0)
    else:
        density_curve = None

    # 6) Initial model; data window is determined by sort_range on full properties
    model = GaussianModel(
        mean=m.copy(), covariance=covs.copy(), bic_threshold=multi_cluster_threshold, mah_threshold=float(mah_th),
        data_range=None, sort_range=None, sort_feature_idx=sorted_feature_idx, density_curve=density_curve,
    )
    model.compute_data_range(properties)
    first_idx, last_idx = model.data_range
    in_bounds_global = model.in_bounds_indices(properties, curr_max_prob)
    in_bounds = (in_bounds_global - first_idx).astype(int)
    window_points = points[first_idx:last_idx + 1]
    in_bounds = apply_bic_refinement(
        window_points, m, covs, in_bounds, multi_cluster_threshold, n_features,
        min_points_for_cluster=min_points_for_cluster, seed_row=None
    )
    if len(in_bounds) < min_points_for_cluster:
        return {'success': False, 'message': f'Insufficient points after BIC refinement: {len(in_bounds)} < {min_points_for_cluster}'}
    diff_window = window_points - m
    mahal_d_window = np.einsum('ij,jk,ik->i', diff_window, inv_cov, diff_window) ** 0.5
    seed_idx_debug = int(cluster_indices[0]) if len(cluster_indices) > 0 else first_idx
    debug_data = {
        'points': np.asarray(window_points, dtype=float),
        'center': m.copy(),
        'covariance': covs.copy(),
        'mahal_d': mahal_d_window,
        'in_bounds': in_bounds,
        'valid_indices': np.arange(first_idx, last_idx + 1, dtype=int),
        'mah_th': float(mah_th),
        'mahal_d_s': mahal_d_s,
        'rolling_HC': rolling_HC,
        'density_curve': density_curve,
        'properties': properties,
        'seed_idx': seed_idx_debug,
        'init_mah_d': 1.0,
    }
    return {'success': True, 'model': model, 'in_bounds': in_bounds, 'debug_data': debug_data}


def make_gaussian_model_from_cluster(properties, cluster_densities, curr_max_prob, cluster_indices, settings):
    """
    Build a Gaussian model from a cluster seed using the FULL properties tensor.
    Init from COM/cov (cluster seed only) and HC/size/density on full-window data; then _fit_model_size
    on full properties. Returns (result_dict) with 'success', 'model', 'visualization_data'
    (window-based: valid_indices are 0-based into data window), 'window_indices' (global indices of window),
    'cluster_indices', 'seed_idx_local' (index into window for seed), or 'message' on failure.
    """
    cluster_indices = np.asarray(cluster_indices, dtype=int).ravel()
    n_cluster = len(cluster_indices)
    if n_cluster < 2:
        return {'success': False, 'message': 'Cluster must have at least 2 points'}
    n_spikes = properties.shape[0]
    if np.any(cluster_indices < 0) or np.any(cluster_indices >= n_spikes):
        return {'success': False, 'message': 'cluster_indices out of range'}

    init_result = _create_init_cluster_from_cluster_data(
        properties, cluster_densities, curr_max_prob, cluster_indices, settings
    )
    if not init_result.get('success'):
        return init_result
    debug_init_cb = settings.get('debug_init_cluster_callback')
    if callable(debug_init_cb):
        debug_init_cb(init_result['debug_data'])
    model = init_result['model']
    in_bounds = init_result['in_bounds']
    # Seed = closest to COM among cluster seed points (global index)
    pts_seed = np.asarray(properties[cluster_indices, :], dtype=float)
    diff_seed = pts_seed - model.mean
    inv_cov = np.linalg.pinv(model.covariance)
    dist_to_com = np.einsum('ij,jk,ik->i', diff_seed, inv_cov, diff_seed) ** 0.5
    seed_idx_global = int(cluster_indices[np.argmin(dist_to_com)])

    # Size optimization on FULL properties tensor
    model, in_bounds, _hist, _hist_full, stability_failed, iter_count = _fit_model_size(
        properties, cluster_densities, model, in_bounds, seed_idx_global, curr_max_prob, settings
    )
    first_idx, last_idx = model.data_range
    window_indices = np.arange(first_idx, last_idx + 1)
    points = np.asarray(properties[window_indices, :], dtype=float)
    inv_cov = np.linalg.pinv(model.covariance)
    diff_new = points - model.mean
    mahal_d = np.einsum('ij,jk,ik->i', diff_new, inv_cov, diff_new) ** 0.5
    sort_lo, sort_hi = model.sort_range
    width = sort_hi - sort_lo
    half_extra = 0.25 * width
    sorted_col = np.asarray(properties[:, model.sort_feature_idx], dtype=float)
    viz_first = np.searchsorted(sorted_col, sort_lo - half_extra, side='left')
    viz_last = np.searchsorted(sorted_col, sort_hi + half_extra, side='right') - 1
    viz_last = min(max(viz_last, viz_first), n_spikes - 1)
    valid_indices_all = np.arange(viz_first, viz_last + 1)
    points_all = np.asarray(properties[valid_indices_all, :], dtype=float)
    in_bounds_global = model.in_bounds_indices(properties, curr_max_prob)
    in_bounds_all = np.where(np.isin(valid_indices_all, in_bounds_global))[0]
    seed_row_in_points = (seed_idx_global - viz_first) if (viz_first <= seed_idx_global <= viz_last) else None
    if seed_row_in_points is None and len(valid_indices_all) > 0:
        seed_row_in_points = int(np.argmin(np.abs(valid_indices_all - seed_idx_global)))
    in_bounds_all = apply_bic_refinement(
        points_all, model.mean, model.covariance,
        in_bounds_all, model.bic_threshold, properties.shape[1],
        min_points_for_cluster=settings.get('min_points_for_cluster', 100),
        seed_row=seed_row_in_points
    )
    # valid_indices = 0-based into data window (for dialog: valid_indices_global = window_indices[valid_indices])
    valid_indices_0based = np.arange(0, last_idx - first_idx + 1)
    seed_idx_in_window = (seed_idx_global - first_idx) if (first_idx <= seed_idx_global <= last_idx) else 0
    viz = {
        'points': points,
        'mahal_d': mahal_d,
        'valid_indices': valid_indices_0based,
        'in_bounds': in_bounds,
        'all_points': points_all,
        'all_valid_indices': valid_indices_all,
        'HC': cluster_densities[valid_indices_all],
        'seed_point': np.copy(properties[seed_idx_global, :]) if 0 <= seed_idx_global < n_spikes else np.copy(points[0, :]),
        'seed_row_in_points': seed_row_in_points,
        'multi_cluster_threshold': model.bic_threshold,
        'success': not stability_failed,
        'iteration_history_full': _hist_full,
        'window_indices': window_indices,
    }
    debug_after_cb = settings.get('debug_after_gm_callback')
    if callable(debug_after_cb):
        debug_after_cb(viz, model)
    return {
        'success': True, 'model': model, 'visualization_data': viz,
        'window_indices': window_indices, 'cluster_indices': cluster_indices,
        'seed_idx_local': seed_idx_in_window, 'seed_idx_global': seed_idx_global,
    }


def _round_step(x, step):
    """Round x to the closest multiple of step."""
    return round(float(x) / step) * step


def _fit_model_size(properties, cluster_densities, model, in_bounds, seed_idx, curr_max_prob, settings):
    """
    Run the model-size iteration loop: adjust mah_th (size) until outcome is stable or max iterations.
    S1 = largest tested size that collapsed or is stable; S2 = min size that explodes (initialized as
    the size at which 2*mah_th*std equals max_range_sorted for the sorted feature, then updated when an explosion is seen).
    New size is bisected between S1 and S2, capped below S2 so max_range_sorted is never violated.
    Terminates when S2 <= S1+step. Returns (model, in_bounds, iteration_history, iteration_history_full, stability_failed, iter_count).
    """
    max_range_sorted = float(settings.get('max_range_sorted', settings['initial_stds'].ravel()[model.sort_feature_idx] * 4.0))
    sort_feature_idx = int(getattr(model, 'sort_feature_idx', 0))
    dist_step = settings.get('dist_step', 0.1)
    min_change = settings.get('min_change', 0.01)
    min_points_for_cluster = int(settings.get('min_points_for_cluster', 100))
    max_iter_for_model = int(settings.get('max_iter_for_model', 500))

    stability_failed = False
    iter_count = 0
    iteration_history = []
    iteration_history_full = []

    S1 = 0.0
    S1_is_stable = False
    stds = np.sqrt(np.maximum(np.diag(model.covariance), 1e-12))
    S2_init = max_range_sorted / (2.0 * stds[sort_feature_idx])
    S2 = _round_step(S2_init, dist_step)

    model.mah_threshold = _round_step(model.mah_threshold, dist_step)
    model.compute_data_range(properties)

    while iter_count < max_iter_for_model:
        iter_count += 1
        outcome, model_out, assignment_out, _ = iterate_GM_model(
            properties, cluster_densities, model.copy(), sort_feature_idx, max_range_sorted,
            seed_idx, min_change=min_change, min_points_for_cluster=min_points_for_cluster,
            curr_max_prob=curr_max_prob
        )
        current_size = _round_step(model.mah_threshold, dist_step)
        iteration_history_full.append((iter_count, float(current_size), outcome))
        if outcome in ('collapsed', 'exploded'):
            iteration_history.append((iter_count, float(current_size), outcome))

        if outcome in ('collapsed', 'stable'):
            S1 = current_size
            S1_is_stable = (outcome == 'stable')
            if S2 <= S1 + dist_step:
                if S1_is_stable:
                    model.mah_threshold = S1
                    model.compute_data_range(properties)
                    outcome_s1, model_out_s1, assignment_out_s1, _ = iterate_GM_model(
                        properties, cluster_densities, model.copy(), sort_feature_idx, max_range_sorted,
                        seed_idx, min_change=min_change, min_points_for_cluster=min_points_for_cluster,
                        curr_max_prob=curr_max_prob
                    )
                    model = model_out_s1
                    in_bounds = assignment_out_s1
                else:
                    stability_failed = True
                break
            if S1 <= 0:
                new_size = S2 * 0.5
            else:
                new_size = (S1 + S2) / 2.0
            new_size = _round_step(new_size, dist_step)
            if new_size == S1:
                new_size += dist_step
            new_size = min(new_size, S2 - dist_step)
            new_size = _round_step(new_size, dist_step)
            if outcome == 'stable':
                model = model_out
                in_bounds = assignment_out
            model.mah_threshold = new_size
            model.compute_data_range(properties)
            continue

        if outcome == 'exploded':
            S2 = min(S2, current_size)
            S2 = _round_step(S2, dist_step)
            if S2 <= S1 + dist_step:
                if S1_is_stable:
                    model.mah_threshold = S1
                    model.compute_data_range(properties)
                    outcome_s1, model_out_s1, assignment_out_s1, _ = iterate_GM_model(
                        properties, cluster_densities, model.copy(), sort_feature_idx, max_range_sorted,
                        seed_idx, min_change=min_change, min_points_for_cluster=min_points_for_cluster,
                        curr_max_prob=curr_max_prob
                    )
                    model = model_out_s1
                    in_bounds = assignment_out_s1
                else:
                    stability_failed = True
                break
            new_size = (S1 + S2) / 2.0
            new_size = _round_step(new_size, dist_step)
            new_size = min(new_size, S2 - dist_step)
            new_size = _round_step(new_size, dist_step)
            model.mah_threshold = new_size
            model.compute_data_range(properties)
            continue

    if iter_count >= max_iter_for_model:
        stability_failed = True

    return (model, in_bounds, iteration_history, iteration_history_full, stability_failed, iter_count)


def fit_gaussian_model_from_seed(properties, seed_idx, cluster_densities, curr_max_prob, settings):
    """
    Fit a Gaussian model from a seed point, implementing steps 4-9 of the iterative clustering algorithm.
    
    This function:
    1. Extracts spatial window around seed
    2. Refines center iteratively using weighted COM
    3. Detects initial cluster boundary
    4. Checks for multi-cluster and refines if needed
    5. Refines covariance matrix
    6. Detects final cluster boundary
    7. Returns Gaussian model with center, covariance, and points within final boundary
    
    Parameters
    ----------
    properties : np.ndarray, shape (N, n_features)
        Spike properties array
    seed_idx : int
        Index of the seed spike in properties array
    cluster_densities : np.ndarray, shape (N,)
        Cluster density values (not cluster index) for each spike, used as weights.
    curr_max_prob : np.ndarray, shape (N,)
        Per-spike current_max_prob: background density where estimated, inf otherwise. Assignment uses
        prob_density > curr_max_prob[spike] (prob_density from density_curve or linear decay).
    settings : dict
        Dictionary containing:
        - 'initial_stds': np.ndarray, shape (n_features,)
            Initial standard deviations for each feature (used to initialize covariance)
        - 'com_iteration_threshold': float
            COM accuracy relative to model size: iteration stops when center shift (Mahal) is below this fraction of init_mah_th (default: 0.25)
        - 'com_iteration_max_iterations': int, optional
            Maximum COM refinement iterations (default: 50)
        - 'max_iter_for_model': int, optional
            Maximum outer iterations for model refinement (default: 500)
        - 'init_mah_d': float
            Mahalanobis distance threshold for defining points to recompute covariance (default: 5.0)
        - 'density_threshold_for_init_distance': float, optional
            Density fraction (relative to peak) at which to stop when finding init boundary (default: 0.1)
        - 'dist_step': float, optional
            Step for adjusting mah_th in the outer iteration loop (default: 0.1)
        - 'sorted_feature_idx': int
            Index of the sorted feature in Properties (used for initial filtering)
        - 'max_range_sorted': float, optional
            Maximum allowed model range (2*mah_th*std) for the sorted feature (default: 4*initial_std of sorted feature)
        - 'n_samples_density_curve': int, optional
            Number of samples for the probability density curve (default: 101)
        - 'min_points_for_cluster': int, optional
            Minimum number of points required for a valid cluster (default: 100)
        - 'multi_cluster_threshold': float, optional
            BIC difference threshold for multi-cluster detection (default: 0.2)
        - 'min_change': float, optional
            Relative change threshold for stable vs grow/shrink in model iteration (default: 0.01)
        - 'gaussian_filter_sigma': float, optional
            Sigma for Gaussian smoothing in boundary detection (default: 50.0)

    Returns
    -------
    result : GaussianModel or dict
        If successful, returns a GaussianModel object containing:
        - mean: Final center of mass (mean) of the Gaussian model
        - covariance: Final covariance matrix
        - bic_threshold: BIC threshold used for refinement
        - mah_threshold: Mahalanobis distance threshold (boundary)
        - density_curve: optional N-point curve for prob_density interpolation (N = n_samples_density_curve)
        
        If unsuccessful, returns a dict with 'success': False and 'message' describing the error
    """
    init_result = _create_init_cluster_from_seed(properties, seed_idx, cluster_densities, curr_max_prob, settings)
    if not init_result['success']:
        return init_result
    initial_model = init_result['model'].copy()
    model, in_bounds, iteration_history, iteration_history_full, stability_failed, iter_count = _fit_model_size(
        properties, cluster_densities, init_result['model'], init_result['in_bounds'], seed_idx, curr_max_prob, settings
    )
    n_features = properties.shape[1]
    multi_cluster_threshold = settings.get('multi_cluster_threshold', 0.2)
    min_points_for_cluster = int(settings.get('min_points_for_cluster', 100))
    max_iter_for_model = int(settings.get('max_iter_for_model', 500))

    # Derive window from model for final indices and viz
    first_idx, last_idx = model.data_range
    valid_indices = np.arange(first_idx, last_idx + 1)
    # Wider range for visualization only: sort_range + 50% value range, then searchsorted
    sort_lo, sort_hi = model.sort_range
    width = sort_hi - sort_lo
    half_extra = 0.25 * width
    viz_lo = sort_lo - half_extra
    viz_hi = sort_hi + half_extra
    sorted_col = np.asarray(properties[:, model.sort_feature_idx], dtype=float)
    viz_first = np.searchsorted(sorted_col, viz_lo, side='left')
    viz_last = np.searchsorted(sorted_col, viz_hi, side='right') - 1
    viz_last = min(max(viz_last, viz_first), len(properties) - 1)
    valid_indices_all = np.arange(viz_first, viz_last + 1)
    points = properties[valid_indices, :]

    # Map back to original indices; use model for viz and return
    m = model.mean
    covs = model.covariance
    mah_th = model.mah_threshold
    gaussian_model = model

    # Visualization and seed check using final model parameters
    inv_cov_final = np.linalg.pinv(covs)
    diff_final = points - m
    mahal_d_final = np.einsum('ij,jk,ik->i', diff_final, inv_cov_final, diff_final)**0.5
    seed_row_check = np.where(valid_indices == seed_idx)[0]
    if len(seed_row_check) == 0:
        points_all = properties[valid_indices_all, :]
        HC_all = cluster_densities[valid_indices_all]
        mahal_d_all = np.einsum('ij,jk,ik->i', points_all - m, inv_cov_final, points_all - m) ** 0.5
        in_bounds_global = gaussian_model.in_bounds_indices(properties, curr_max_prob)
        in_bounds_all = (in_bounds_global - valid_indices_all[0]).astype(int)
        seed_row_in_points = np.where(valid_indices_all == seed_idx)[0]
        seed_row_in_points = int(seed_row_in_points[0]) if len(seed_row_in_points) > 0 else None
        fail_viz = {
            'points': points_all, 'mahal_d': mahal_d_all, 'seed_point': np.copy(properties[seed_idx, :]),
            'valid_indices': valid_indices_all.copy(), 'in_bounds': in_bounds_all,
            'HC': HC_all.copy(), 'multi_cluster_threshold': multi_cluster_threshold,
            'seed_row_in_points': seed_row_in_points, 'iteration_history_full': iteration_history_full,
            'all_points': points_all.copy(), 'all_valid_indices': valid_indices_all.copy(),
            'success': False,
        }
        debug_gm_cb = settings.get('debug_after_gm_callback')
        if debug_gm_cb is not None:
            debug_gm_cb(fail_viz, gaussian_model)
        return {
            'success': False,
            'message': 'Seed not found in valid indices at final boundary check'
        }
    seed_mahal = mahal_d_final[seed_row_check[0]]
    if seed_mahal > mah_th:
        points_all = properties[valid_indices_all, :]
        HC_all = cluster_densities[valid_indices_all]
        mahal_d_all = np.einsum('ij,jk,ik->i', points_all - m, inv_cov_final, points_all - m) ** 0.5
        in_bounds_global = gaussian_model.in_bounds_indices(properties, curr_max_prob)
        in_bounds_all = (in_bounds_global - valid_indices_all[0]).astype(int)
        seed_row_in_points = np.where(valid_indices_all == seed_idx)[0]
        seed_row_in_points = int(seed_row_in_points[0]) if len(seed_row_in_points) > 0 else None
        fail_viz = {
            'points': points_all, 'mahal_d': mahal_d_all, 'seed_point': np.copy(properties[seed_idx, :]),
            'valid_indices': valid_indices_all.copy(), 'in_bounds': in_bounds_all,
            'HC': HC_all.copy(), 'multi_cluster_threshold': multi_cluster_threshold,
            'seed_row_in_points': seed_row_in_points, 'iteration_history_full': iteration_history_full,
            'all_points': points_all.copy(), 'all_valid_indices': valid_indices_all.copy(),
            'success': False,
        }
        debug_gm_cb = settings.get('debug_after_gm_callback')
        if debug_gm_cb is not None:
            debug_gm_cb(fail_viz, gaussian_model)
        return {
            'success': False,
            'message': f'Seed outside final boundary: Mahalanobis distance {seed_mahal:.3f} > threshold {mah_th:.3f}'
        }

    points_all = properties[valid_indices_all, :]
    HC_all = cluster_densities[valid_indices_all]
    diff_all = points_all - m
    mahal_d_all = np.einsum('ij,jk,ik->i', diff_all, inv_cov_final, diff_all) ** 0.5
    in_bounds_global = gaussian_model.in_bounds_indices(properties, curr_max_prob)
    in_bounds_all = (in_bounds_global - valid_indices_all[0]).astype(int)
    seed_row_in_points = np.where(valid_indices_all == seed_idx)[0]
    seed_row_in_points = int(seed_row_in_points[0]) if len(seed_row_in_points) > 0 else None
    in_bounds_all = apply_bic_refinement(
        points_all, m, covs,
        in_bounds_all, multi_cluster_threshold, n_features,
        min_points_for_cluster=min_points_for_cluster,
        seed_row=seed_row_in_points
    )
    # visualization_data: arrays for viz; model params (center/covariance) come from result['model']
    viz = {
        'points': points_all,
        'mahal_d': mahal_d_all,
        'valid_indices': valid_indices_all.copy(),
        'in_bounds': in_bounds_all,
        'HC': HC_all.copy(),
        'seed_point': np.copy(properties[seed_idx, :]),
        'seed_row_in_points': seed_row_in_points,
        'multi_cluster_threshold': multi_cluster_threshold,
        'iteration_history_full': iteration_history_full,
        'all_points': points_all.copy(),
        'all_valid_indices': valid_indices_all.copy(),
        'success': not stability_failed,
    }
    result = {
        'model': gaussian_model,
        'visualization_data': viz,
        'initial_model': initial_model,
    }
    if stability_failed:
        result['success'] = False
        result['message'] = 'Maximum iterations reached without convergence' if iter_count >= max_iter_for_model else 'model did not reach stability'
        result['stability_iteration_history'] = iteration_history

    # Optional debug callback 3: final model after GM iterations (always show, including on failure)
    debug_gm_cb = settings.get('debug_after_gm_callback')
    if debug_gm_cb is not None:
        debug_gm_cb(viz, gaussian_model)

    return result
