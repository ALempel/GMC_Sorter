import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

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


def get_raw_densities(properties, grid_boundaries, step_sizes, reg_density, use_gpu=False):
    """
    Initialize a grid tensor and count points in each voxel.
    
    Parameters:
    -----------
    properties : np.ndarray or torch.Tensor
        Properties tensor of shape (N_points, D_dimensions)
    grid_boundaries : list of tuples
        List of (min, max) tuples for each dimension in order
    step_sizes : list
        List of step sizes for each dimension in order
    reg_density : float
        Regularization density to add to each voxel count
    use_gpu : bool
        If True and GPU available, use GPU acceleration
    
    Returns:
    --------
    grid_counts : np.ndarray or torch.Tensor
        D-dimensional array of point counts per voxel (with reg_density added)
    bin_edges : list
        List of bin edge arrays for each dimension
    voxel_mins : list
        List of arrays with minimum values for each voxel per dimension
    voxel_maxs : list
        List of arrays with maximum values for each voxel per dimension
    """
    # Convert to numpy if needed
    if TORCH_AVAILABLE and isinstance(properties, torch.Tensor):
        properties_np = properties.cpu().numpy()
        is_torch = True
    else:
        properties_np = np.asarray(properties)
        is_torch = False
    
    N_points, D = properties_np.shape
    
    # Validate inputs
    assert len(grid_boundaries) == D, f"Number of dimensions in properties ({D}) must match grid_boundaries length"
    assert len(step_sizes) == D, f"Number of dimensions in properties ({D}) must match step_sizes length"
    
    # Build bin edges for each dimension
    bin_edges = []
    grid_shape = []
    
    for d in range(D):
        min_val, max_val = grid_boundaries[d]
        step = step_sizes[d]
        
        # Calculate number of bins
        range_size = max_val - min_val
        n_bins = int(range_size / step)
        
        # Create bin edges (centered grid)
        grid_size = n_bins * step
        grid_start = min_val + (range_size - grid_size) / 2.0
        grid_end = grid_start + grid_size
        
        edges = np.linspace(grid_start, grid_end, n_bins + 1)
        bin_edges.append(edges)
        grid_shape.append(n_bins)
    
    grid_shape = tuple(grid_shape)
    
    # Initialize grid as double tensor
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        grid_counts = torch.zeros(grid_shape, dtype=torch.float64, device='cuda')
        properties_tensor = torch.from_numpy(properties_np).float().to('cuda')
    else:
        grid_counts = np.zeros(grid_shape, dtype=np.float64)
        properties_tensor = properties_np
    
    # Calculate voxel min/max for each dimension
    # For each dimension, voxel mins are the bin edges[:-1] and maxs are edges[1:]
    voxel_mins_list = []
    voxel_maxs_list = []
    for edges in bin_edges:
        voxel_mins_list.append(edges[:-1])  # All but last edge
        voxel_maxs_list.append(edges[1:])   # All but first edge
    
    # Count points in each voxel - EFFICIENT IMPLEMENTATION
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        grid_counts = _count_points_gpu(properties_tensor, bin_edges, grid_shape, reg_density)
        voxel_mins = [torch.from_numpy(m).to('cuda') for m in voxel_mins_list]
        voxel_maxs = [torch.from_numpy(m).to('cuda') for m in voxel_maxs_list]
    else:
        grid_counts = _count_points_cpu(properties_np, bin_edges, grid_shape, reg_density)
        voxel_mins = [np.array(m) for m in voxel_mins_list]
        voxel_maxs = [np.array(m) for m in voxel_maxs_list]
    
    return grid_counts, bin_edges, voxel_mins, voxel_maxs


def _count_points_cpu(points, bin_edges, grid_shape, reg_density):
    """
    CPU-optimized point counting using numpy.
    This is the fast implementation from density_calc.py.
    """
    N, D = points.shape
    assert len(bin_edges) == D
    assert len(grid_shape) == D
    
    # Initialize grid
    grid_counts = np.zeros(grid_shape, dtype=np.float64)
    
    # Digitize each dimension to get bin indices
    inds = np.empty((N, D), dtype=np.int32)
    for d in range(D):
        # Use searchsorted to find bin index for each point
        # side='right' means: if point == edge, it goes to the right bin
        # Then subtract 1 to get 0-indexed bin
        inds[:, d] = np.searchsorted(bin_edges[d], points[:, d], side='right') - 1
    
    # Mask invalid bins (out of bounds)
    valid = np.ones(N, dtype=bool)
    for d in range(D):
        valid &= (inds[:, d] >= 0) & (inds[:, d] < grid_shape[d])
    
    # Only count valid points
    if np.any(valid):
        inds_valid = inds[valid]
        
        # Convert multi-dimensional bin indices to flat index
        flat_inds = np.ravel_multi_index(inds_valid.T, grid_shape)
        
        # Count points per voxel using bincount (very fast!)
        bincount = np.bincount(flat_inds, minlength=np.prod(grid_shape))
        grid_counts = bincount.reshape(grid_shape).astype(np.float64)
    
    # Add regularization density
    grid_counts += reg_density
    
    return grid_counts


def _count_points_gpu(points_tensor, bin_edges, grid_shape, reg_density):
    """
    GPU-optimized point counting using PyTorch.
    
    This implementation uses PyTorch's searchsorted and scatter operations
    which can be faster for large datasets on GPU.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available for GPU implementation")
    
    N, D = points_tensor.shape
    device = points_tensor.device
    
    # Convert bin_edges to tensors on GPU
    bin_edges_tensors = [torch.from_numpy(edges).float().to(device) for edges in bin_edges]
    
    # Initialize grid on GPU
    grid_counts = torch.zeros(grid_shape, dtype=torch.float64, device=device)
    
    # Digitize each dimension
    inds = torch.empty((N, D), dtype=torch.long, device=device)
    for d in range(D):
        # PyTorch's searchsorted (available in PyTorch 1.7+)
        # Returns rightmost insertion point
        inds[:, d] = torch.searchsorted(bin_edges_tensors[d], points_tensor[:, d], right=True) - 1
    
    # Mask invalid bins
    valid = torch.ones(N, dtype=torch.bool, device=device)
    for d in range(D):
        valid &= (inds[:, d] >= 0) & (inds[:, d] < grid_shape[d])
    
    if torch.any(valid):
        inds_valid = inds[valid]
        
        # Convert multi-dimensional indices to flat index
        # PyTorch equivalent of ravel_multi_index
        flat_inds = _ravel_multi_index_torch(inds_valid, grid_shape)
        
        # Count points per voxel using scatter_add (GPU-optimized)
        # This is faster than bincount on GPU for large arrays
        grid_counts_flat = torch.zeros(np.prod(grid_shape), dtype=torch.float64, device=device)
        grid_counts_flat.scatter_add_(0, flat_inds, torch.ones(len(flat_inds), dtype=torch.float64, device=device))
        grid_counts = grid_counts_flat.reshape(grid_shape)
    
    # Add regularization density
    grid_counts += reg_density
    
    return grid_counts


def _ravel_multi_index_torch(multi_index, shape):
    """
    PyTorch equivalent of np.ravel_multi_index.
    Converts multi-dimensional indices to flat indices.
    """
    if len(shape) == 1:
        return multi_index[:, 0]
    
    # Calculate strides
    strides = torch.tensor([np.prod(shape[i+1:]) for i in range(len(shape))], 
                           dtype=torch.long, device=multi_index.device)
    
    # Compute flat index
    flat_index = (multi_index * strides.unsqueeze(0)).sum(dim=1)
    return flat_index


def _gaussian_filter_torch(input_tensor, sigma, mode='constant'):
    """
    Apply multi-dimensional Gaussian filter using PyTorch (GPU-compatible).
    
    Uses separable 1D Gaussian filters applied sequentially along each dimension.
    This is equivalent to scipy.ndimage.gaussian_filter but works on GPU.
    
    Parameters:
    -----------
    input_tensor : torch.Tensor
        Input tensor of arbitrary shape
    sigma : list or tuple
        Standard deviation for Gaussian kernel for each dimension
    mode : str
        Boundary mode ('constant', 'reflect', 'replicate', 'circular')
        Currently only 'constant' is fully supported
    
    Returns:
    --------
    torch.Tensor
        Filtered tensor with same shape as input
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available for GPU filtering")
    
    result = input_tensor.clone()
    device = result.device
    dtype = result.dtype
    
    # Apply 1D Gaussian filter along each dimension sequentially
    for dim, sig in enumerate(sigma):
        if sig <= 0:
            continue  # Skip filtering for this dimension
        
        # Calculate kernel size (use 4*sigma for good coverage)
        kernel_size = int(2 * np.ceil(4 * sig) + 1)
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
        
        # Create 1D Gaussian kernel
        center = kernel_size // 2
        x = torch.arange(kernel_size, dtype=dtype, device=device) - center
        kernel_1d = torch.exp(-0.5 * (x / sig) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize
        
        # Apply padding for boundary handling
        pad_size = kernel_size // 2
        pad_tuple = [0] * (2 * len(result.shape))
        pad_tuple[2 * dim] = pad_size  # Before
        pad_tuple[2 * dim + 1] = pad_size  # After
        # Reverse for F.pad (it expects reverse order: last dim first)
        pad_tuple = pad_tuple[::-1]
        
        padded = torch.nn.functional.pad(result, pad_tuple, mode='constant', value=0.0)
        
        # Apply convolution along current dimension
        # Move dimension to filter to the last position
        n_dims = len(result.shape)
        permute_order = list(range(n_dims))
        permute_order[dim], permute_order[-1] = permute_order[-1], permute_order[dim]
        result_permuted = padded.permute(*permute_order)
        
        # Reshape to (batch, channels, length) for conv1d
        other_dims = result_permuted.shape[:-1]
        length = result_permuted.shape[-1]
        batch_size = int(np.prod(other_dims))
        result_reshaped = result_permuted.reshape(batch_size, 1, length)
        
        # Apply conv1d
        kernel_1d_reshaped = kernel_1d.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size)
        filtered = torch.nn.functional.conv1d(result_reshaped, kernel_1d_reshaped, padding=0)
        
        # Reshape back
        filtered_reshaped = filtered.reshape(*other_dims, filtered.shape[-1])
        
        # Permute back to original order
        result = filtered_reshaped.permute(*permute_order)
    
    return result


def separate_cluster_background_densities(raw_densities, spatial_filter, spatial_dim_indexes,
                                           spatial_dim_names=None, it_number=3, cluster_sig_th=1.5, 
                                           use_gpu=False, reg_density=1.0):
    """
    Separate cluster and background densities from raw densities using iterative 
    Gaussian filtering.
    
    This function implements the iterative filtering approach from density_calc.py:
    1. Apply small Gaussian filter (lowpass) to smooth raw densities
    2. Iteratively compute background using large Gaussian filter (highpass)
    3. Extract cluster signal by subtracting background and thresholding
    
    Parameters:
    -----------
    raw_densities : np.ndarray
        Raw density grid of shape (n_bins_dim0, n_bins_dim1, ...)
        This should be the output from get_raw_densities
    spatial_filter : dict or tuple
        If dict: maps spatial dimension names (e.g., 'y_pos', 'x_pos') to (lowpass, highpass) tuples in grid units
        If tuple: (lowpass, highpass) in grid units (backward compatibility)
        - lowpass: small sigma for initial smoothing (typically ~1 grid step)
        - highpass: large sigma for background estimation (typically ~12 grid steps)
    spatial_dim_indexes : list
        List of dimension indexes that are spatial (e.g., [0, 1] for x_pos, y_pos)
        Only these dimensions will be filtered; others will have sigma=0
    spatial_dim_names : list, optional
        List of spatial dimension names corresponding to spatial_dim_indexes
        Required if spatial_filter is a dict
    it_number : int, default=3
        Number of iterations for background/cluster separation
    cluster_sig_th : float, default=1.5
        Threshold multiplier: cluster_signal must be > background_signal * cluster_sig_th
    use_gpu : bool, default=False
        If True and GPU available, use GPU acceleration for filtering
    reg_density : float, default=1.0
        Regularization density - background_density will be capped to be at least this value
    
    Returns:
    --------
    background_density : np.ndarray
        Background density grid, same shape as raw_densities
    cluster_density : np.ndarray
        Cluster density grid, same shape as raw_densities
    """
    raw_densities = np.asarray(raw_densities, dtype=np.float64)
    raw_densities += reg_density
    D = len(raw_densities.shape)
    
    # Validate inputs and handle backward compatibility
    if isinstance(spatial_filter, dict):
        if spatial_dim_names is None:
            raise ValueError("spatial_dim_names must be provided when spatial_filter is a dict")
        if len(spatial_dim_names) != len(spatial_dim_indexes):
            raise ValueError("spatial_dim_names length must match spatial_dim_indexes length")
    elif isinstance(spatial_filter, (tuple, list)) and len(spatial_filter) == 2:
        # Backward compatibility: convert tuple to dict with same values for all spatial dims
        lowpass, highpass = spatial_filter
        spatial_filter = {name: (lowpass, highpass) for name in (spatial_dim_names or ['y_pos', 'x_pos'])}
    else:
        raise ValueError("spatial_filter must be a dict or tuple of (lowpass, highpass)")
    
    assert isinstance(spatial_dim_indexes, (list, tuple)), "spatial_dim_indexes must be a list or tuple"
    
    # Build sigma arrays: apply filter only to spatial dimensions
    sigma_lowpass = []
    sigma_highpass = []
    for d in range(D):
        if d in spatial_dim_indexes:
            # Spatial dimension: get filter values for this dimension
            dim_idx_in_spatial = spatial_dim_indexes.index(d)
            if spatial_dim_names is not None and dim_idx_in_spatial < len(spatial_dim_names):
                dim_name = spatial_dim_names[dim_idx_in_spatial]
                if dim_name in spatial_filter:
                    lowpass, highpass = spatial_filter[dim_name]
                else:
                    # Default values if not specified
                    lowpass, highpass = 2.0, 25.0
            else:
                # Fallback: use first available filter or default
                if len(spatial_filter) > 0:
                    lowpass, highpass = list(spatial_filter.values())[0]
                else:
                    lowpass, highpass = 2.0, 25.0
            sigma_lowpass.append(lowpass)
            sigma_highpass.append(highpass)
        else:
            # Non-spatial dimension: no filtering
            sigma_lowpass.append(0.0)
            sigma_highpass.append(0.0)
    
    # Determine if we should use GPU
    use_gpu_actual = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
    
    if use_gpu_actual:
        # Convert to PyTorch tensor on GPU
        device = torch.device('cuda')
        raw_densities_t = torch.from_numpy(raw_densities.astype(np.float32)).to(device)
        ones_t = torch.ones_like(raw_densities_t)
        
        # Step 1: Apply small Gaussian filter (lowpass) to smooth raw densities
        # Create normalization mask for small filter
        mask_c = _gaussian_filter_torch(ones_t, sigma_lowpass, mode='constant')
        
        # Smooth raw densities with small filter and normalize
        density_smoothed_t = _gaussian_filter_torch(raw_densities_t, sigma_lowpass, mode='constant') / mask_c
        
        # Step 2: Create normalization mask for large filter (highpass)
        mask_s = _gaussian_filter_torch(ones_t, sigma_highpass, mode='constant')
        
        # Step 3: Iteratively separate cluster and background
        cluster_signal_t = torch.zeros_like(raw_densities_t)
        
        if it_number == 0:
            # No iterations: background = highpass of raw, cluster = full smoothed
            background_signal_t = _gaussian_filter_torch(raw_densities_t, sigma_highpass, mode='constant') / mask_s
            cluster_signal_t = density_smoothed_t.clone()
        else:
            for it in range(it_number):
                # Compute background by filtering (smoothed_density - current_cluster_signal)
                background_signal_t = _gaussian_filter_torch(density_smoothed_t - cluster_signal_t, 
                                                             sigma_highpass, mode='constant') / mask_s
                
                # Cluster signal is the difference between smoothed and background
                cluster_signal_t = density_smoothed_t - background_signal_t
                
                # Threshold: keep only cluster signal that is significantly above background
                threshold_mask = cluster_signal_t < (background_signal_t * cluster_sig_th)
                cluster_signal_t[threshold_mask] = 0
        
        # Convert back to numpy
        background_density = background_signal_t.cpu().numpy().astype(np.float64)
        cluster_density = cluster_signal_t.cpu().numpy().astype(np.float64)
    else:
        # CPU implementation using scipy
        # Step 1: Apply small Gaussian filter (lowpass) to smooth raw densities
        # Create normalization mask for small filter
        mask_c = gaussian_filter(np.ones_like(raw_densities.astype(float)), 
                                  sigma=sigma_lowpass, mode='constant')
        
        # Smooth raw densities with small filter and normalize
        density_smoothed = gaussian_filter(raw_densities.astype(float), 
                                           sigma=sigma_lowpass, mode='constant') / mask_c
        
        # Step 2: Create normalization mask for large filter (highpass)
        mask_s = gaussian_filter(np.ones_like(raw_densities.astype(float)), 
                                  sigma=sigma_highpass, mode='constant')
        
        # Step 3: Iteratively separate cluster and background
        cluster_signal = np.zeros(raw_densities.shape, dtype=np.float64)
        
        if it_number == 0:
            # No iterations: background = highpass of raw, cluster = full smoothed
            background_signal = gaussian_filter(raw_densities.astype(float), 
                                                 sigma=sigma_highpass, mode='constant') / mask_s
            cluster_signal = density_smoothed.copy() - background_signal.copy()
        else:
            for it in range(it_number):
                # Compute background by filtering (smoothed_density - current_cluster_signal)
                background_signal = gaussian_filter(density_smoothed - cluster_signal, 
                                                     sigma=sigma_highpass, mode='constant') / mask_s
                
                # Cluster signal is the difference between smoothed and background
                cluster_signal = density_smoothed - background_signal
                
                # Threshold: keep only cluster signal that is significantly above background
                cluster_signal[cluster_signal < (background_signal * cluster_sig_th)] = 0
        
        # Return background and cluster densities
        background_density = background_signal
        cluster_density = cluster_signal
    
    
    return background_density, cluster_density


def _interpolate_grid_to_spikes(properties, density_grid, bin_edges, grid_shape):
    """
    Interpolate density grid values to spike positions using linear interpolation.
    
    Uses scipy.interpolate.RegularGridInterpolator for multi-dimensional linear interpolation.
    Spikes outside the grid boundaries get density = 0 (will be handled by caller).
    
    Parameters:
    -----------
    properties : np.ndarray
        Spike properties (N_spikes, D_dimensions)
    density_grid : np.ndarray
        Density grid of shape grid_shape
    bin_edges : list
        List of bin edge arrays for each dimension
    grid_shape : tuple
        Shape of the density grid
    
    Returns:
    --------
    spike_densities : np.ndarray
        Density values for each spike (N_spikes,)
    """
    N, D = properties.shape
    assert len(bin_edges) == D
    assert len(grid_shape) == D
    
    # Compute bin centers for each dimension
    centers = [0.5 * (e[1:] + e[:-1]) for e in bin_edges]
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        centers,
        density_grid,
        bounds_error=False,
        fill_value=None  # allows extrapolation
    )
    
    # Interpolate for all spike positions
    spike_densities = interpolator(properties)
    
    # Convert to float64 and handle any NaN/None values from extrapolation
    spike_densities = np.asarray(spike_densities, dtype=np.float64)
    spike_densities = np.nan_to_num(spike_densities, nan=0.0, posinf=0.0, neginf=0.0)
    
    return spike_densities


def estimate_grid_batches(prop_titles, included_feature_indexes, grid_ranges, grid_steps, max_memory_mb,
                          spatial_filter, padding_sigma_multiplier=2.0):
    """
    Estimate number of batches and total voxels using the same logic as calculate_densities_batch.
    Used by the grid settings UI so the displayed batch count matches the actual run.
    spatial_filter : dict mapping batching dimension name to (lowpass, highpass) in grid units,
        or (lowpass, highpass) tuple. Must provide highpass for the batching dimension ('y_pos').
    padding_sigma_multiplier : float, default=2.0
        Padding (in bins) = ceil(padding_sigma_multiplier * highpass) on each side of batch core.
    Returns (n_batches, total_voxels).
    """
    if spatial_filter is None:
        raise ValueError("spatial_filter is required for estimate_grid_batches")
    batching_dim = 'y_pos'
    if batching_dim not in prop_titles:
        return 1, 0
    batching_dim_global_idx = prop_titles.index(batching_dim)
    if batching_dim_global_idx not in included_feature_indexes:
        return 1, 0
    batching_dim_local_idx = included_feature_indexes.index(batching_dim_global_idx)
    if isinstance(spatial_filter, dict) and batching_dim in spatial_filter:
        highpass_batching = spatial_filter[batching_dim][1]
    elif isinstance(spatial_filter, (tuple, list)) and len(spatial_filter) == 2:
        highpass_batching = spatial_filter[1]
    else:
        raise ValueError("spatial_filter must provide (lowpass, highpass) for batching dimension 'y_pos'")
    padding_bins = max(1, int(np.ceil(padding_sigma_multiplier * highpass_batching)))
    sorted_feat_indices = sorted(grid_ranges.keys())
    full_grid_shape = []
    for feat_idx in sorted_feat_indices:
        min_val, max_val = grid_ranges[feat_idx]
        step_val = grid_steps[feat_idx]
        range_size = max_val - min_val
        n_steps = int(range_size / step_val)
        n_voxels_dim = n_steps + 1
        full_grid_shape.append(n_voxels_dim)
    other_dims_memory = 1
    for i, feat_idx in enumerate(sorted_feat_indices):
        if i != batching_dim_local_idx:
            other_dims_memory *= full_grid_shape[i]
    slice_memory_bytes = other_dims_memory * 8
    max_memory_bytes = max_memory_mb * 1024 * 1024
    slices_per_batch = max(1, int(max_memory_bytes / slice_memory_bytes) - 2 * padding_bins)
    batching_min, batching_max = grid_ranges[batching_dim_global_idx]
    batching_step = grid_steps[batching_dim_global_idx]
    batching_range_size = batching_max - batching_min
    n_batching_bins = int(batching_range_size / batching_step)
    total_batching_slices = n_batching_bins + 1
    if slices_per_batch >= total_batching_slices:
        n_batches = 1
    else:
        n_batches = max(1, int(np.ceil(total_batching_slices / slices_per_batch)))
    total_voxels = 1
    for n in full_grid_shape:
        total_voxels *= n
    return n_batches, total_voxels


def calculate_densities_batch(properties, prop_titles, included_feature_indexes, grid_ranges, grid_steps, 
                               max_memory_mb, spatial_filter, it_number=3, cluster_sig_th=1.5, 
                               use_gpu=False, progress_callback=None, reg_density=1.0, padding_sigma_multiplier=2.0):
    """
    Calculate densities in batches with spatial filtering.
    
    Parameters:
    -----------
    properties : np.ndarray
        Properties tensor (N_spikes, N_features) for included features only
    prop_titles : list
        List of all property titles (full list, not just included)
    included_feature_indexes : list
        List of feature indices (in original prop_titles order) that are included
    grid_ranges : dict
        Dictionary mapping feature index to (min, max) tuple
    grid_steps : dict
        Dictionary mapping feature index to step size
    max_memory_mb : float
        Maximum memory in MB per batch
    spatial_filter : dict
        Dictionary mapping spatial dimension names (e.g., 'y_pos', 'x_pos') to (lowpass, highpass) tuples
        in grid units (already converted from physical units)
    it_number : int, default=3
        Number of iterations for background/cluster separation
    cluster_sig_th : float, default=1.5
        Threshold multiplier for cluster signal
    use_gpu : bool, default=False
        If True and GPU available, use GPU acceleration
    progress_callback : callable, optional
        Function called with (current_batch, total_batches) for progress updates
    reg_density : float, default=1.0
        Regularization density added to each voxel count
    padding_sigma_multiplier : float, default=2.0
        Padding (in bins) = ceil(padding_sigma_multiplier * highpass) on each side of batch core.
    
    Returns:
    --------
    cluster_den : np.ndarray
        Cluster density for each spike
    background_den : np.ndarray
        Background density for each spike
    """
    # Find spatial dimensions (those ending in "_pos")
    spatial_dim_indexes = []
    spatial_dim_names = []
    for feat_idx in included_feature_indexes:
        if feat_idx < len(prop_titles) and prop_titles[feat_idx].endswith('_pos'):
            spatial_dim_indexes.append(included_feature_indexes.index(feat_idx))
            spatial_dim_names.append(prop_titles[feat_idx])
    
    if len(spatial_dim_indexes) == 0:
        raise ValueError("No spatial dimensions (ending in '_pos') found in included features")
    
    # Find batching dimension index (default: 'y_pos')
    batching_dim = 'y_pos'
    if batching_dim not in prop_titles:
        raise ValueError(f"Batching dimension '{batching_dim}' not found in properties")
    
    # Find batching dimension in included features
    batching_dim_global_idx = prop_titles.index(batching_dim)
    if batching_dim_global_idx not in included_feature_indexes:
        raise ValueError(f"Batching dimension '{batching_dim}' must be in included features")
    
    # Get local index of batching dimension in included features
    batching_dim_local_idx = included_feature_indexes.index(batching_dim_global_idx)
    
    # Padding on batching axis: match the larger (highpass) sigma so spatial filters have enough data
    if spatial_filter is None:
        raise ValueError("spatial_filter is required for density calculation")
    if isinstance(spatial_filter, dict) and batching_dim in spatial_filter:
        highpass_batching = spatial_filter[batching_dim][1]
    elif isinstance(spatial_filter, (tuple, list)) and len(spatial_filter) == 2:
        highpass_batching = spatial_filter[1]
    else:
        raise ValueError("spatial_filter must provide (lowpass, highpass) for batching dimension 'y_pos'")
    padding_bins = max(1, int(np.ceil(padding_sigma_multiplier * highpass_batching)))
    
    # Get sorted feature indices for consistent ordering
    sorted_feat_indices = sorted(grid_ranges.keys())
    
    # Calculate full grid shape (all dimensions)
    full_grid_shape = []
    for feat_idx in sorted_feat_indices:
        min_val, max_val = grid_ranges[feat_idx]
        step_val = grid_steps[feat_idx]
        range_size = max_val - min_val
        n_steps = int(range_size / step_val)
        n_voxels_dim = n_steps + 1
        full_grid_shape.append(n_voxels_dim)
    
    # Calculate memory per "slice" (one position in batching dimension)
    # Memory = product of all other dimensions * 8 bytes
    other_dims_memory = 1
    for i, feat_idx in enumerate(sorted_feat_indices):
        if i != batching_dim_local_idx:
            other_dims_memory *= full_grid_shape[i]
    slice_memory_bytes = other_dims_memory * 8
    
    # Max slices per batch: leave room for padding on both sides (full grid = core + 2*padding_bins)
    max_memory_bytes = max_memory_mb * 1024 * 1024
    slices_per_batch = max(1, int(max_memory_bytes / slice_memory_bytes) - 2 * padding_bins)
    
    # Get batching dimension range and step
    batching_min, batching_max = grid_ranges[batching_dim_global_idx]
    batching_step = grid_steps[batching_dim_global_idx]
    batching_range_size = batching_max - batching_min
    n_batching_bins = int(batching_range_size / batching_step)
    total_batching_slices = n_batching_bins + 1
    
    # Calculate number of batches with padding (padding_bins on each side of core)
    if slices_per_batch >= total_batching_slices:
        n_batches = 1
        batch_starts = [0]
        batch_ends = [total_batching_slices]
        batch_pad_starts = [0]
        batch_pad_ends = [total_batching_slices]
    else:
        n_batches = 1
        batch_starts = [0]
        batch_ends = [slices_per_batch]
        batch_pad_starts = [0]
        batch_pad_ends = [min(slices_per_batch + padding_bins, total_batching_slices)]
        
        current_end = slices_per_batch
        while current_end < total_batching_slices:
            next_start = current_end
            next_end = min(next_start + slices_per_batch, total_batching_slices)
            batch_starts.append(next_start)
            batch_ends.append(next_end)
            pad_start = max(0, next_start - padding_bins)
            pad_end = min(next_end + padding_bins, total_batching_slices)
            batch_pad_starts.append(pad_start)
            batch_pad_ends.append(pad_end)
            current_end = next_end
            n_batches += 1
        # First batch: pad only at top (no batch below)
        batch_pad_starts[0] = 0
        batch_pad_ends[0] = min(slices_per_batch + padding_bins, total_batching_slices)
        # Last batch: pad only at bottom
        batch_pad_starts[-1] = max(0, batch_starts[-1] - padding_bins)
        batch_pad_ends[-1] = total_batching_slices
    
    # Initialize output arrays
    n_spikes = properties.shape[0]
    cluster_den = np.zeros(n_spikes, dtype=np.float64)
    background_den = np.zeros(n_spikes, dtype=np.float64)
    
    # Process each batch
    for batch_idx in range(n_batches):
        if progress_callback:
            progress_callback(batch_idx + 1, n_batches)
        
        # Get batching dimension range for this batch (with padding)
        batching_grid_size = n_batching_bins * batching_step
        batching_grid_start = batching_min + (batching_range_size - batching_grid_size) / 2.0
        
        batch_pad_start_bin = batch_pad_starts[batch_idx]
        batch_pad_end_bin = batch_pad_ends[batch_idx]
        
        batch_binning_min = batching_grid_start + batch_pad_start_bin * batching_step
        batch_binning_max = batching_grid_start + batch_pad_end_bin * batching_step
        
        # Create grid boundaries for this batch
        batch_grid_boundaries = []
        batch_step_sizes = []
        for feat_idx in sorted_feat_indices:
            if feat_idx == batching_dim_global_idx:
                batch_grid_boundaries.append((batch_binning_min, batch_binning_max))
                batch_step_sizes.append(batching_step)
            else:
                min_val, max_val = grid_ranges[feat_idx]
                batch_grid_boundaries.append((min_val, max_val))
                batch_step_sizes.append(grid_steps[feat_idx])
        
        # Slice properties to only include points within padded batch range (for grid calculation)
        batching_values = properties[:, batching_dim_local_idx]
        start_idx_padded = np.searchsorted(batching_values, batch_binning_min, side='left')
        end_idx_padded = np.searchsorted(batching_values, batch_binning_max, side='right')
        
        batch_properties_padded = properties[start_idx_padded:end_idx_padded]
        
        # Get raw densities for this batch (using padded properties)
        batch_raw_densities, batch_bin_edges, batch_voxel_mins, batch_voxel_maxs = get_raw_densities(
            batch_properties_padded,
            batch_grid_boundaries,
            batch_step_sizes,
            reg_density,
            use_gpu=False
        )
        
        # Apply spatial filtering to get cluster and background densities (using padded grid)
        batch_background_density, batch_cluster_density = separate_cluster_background_densities(
            batch_raw_densities,
            spatial_filter,
            spatial_dim_indexes,
            spatial_dim_names=spatial_dim_names,
            it_number=it_number,
            cluster_sig_th=cluster_sig_th,
            use_gpu=use_gpu,
            reg_density=reg_density
        )
        
        # Delete raw_densities to save memory
        del batch_raw_densities
        
        # Calculate interpolation range
        # For first batch: include spikes below core min (extend to beginning)
        # For last batch: include spikes above core max (extend to end)
        # For middle batches: use core range only
        batch_start_bin = batch_starts[batch_idx]
        batch_end_bin = batch_ends[batch_idx]
        
        # Convert bin indices to actual values
        batching_grid_size = n_batching_bins * batching_step
        batching_grid_start = batching_min + (batching_range_size - batching_grid_size) / 2.0
        batch_core_min = batching_grid_start + batch_start_bin * batching_step
        batch_core_max = batching_grid_start + batch_end_bin * batching_step
        
        # For first batch: extend interpolation range downward (include all spikes below core min)
        if batch_idx == 0:
            start_idx_interp = 0  # Include all spikes from the beginning
            end_idx_interp = np.searchsorted(batching_values, batch_core_max, side='right')
        # For last batch: extend interpolation range upward (include all spikes above core max)
        elif batch_idx == n_batches - 1:
            start_idx_interp = np.searchsorted(batching_values, batch_core_min, side='left')
            end_idx_interp = len(batching_values)  # Include all spikes to the end
        # For middle batches: use core range only
        else:
            start_idx_interp = np.searchsorted(batching_values, batch_core_min, side='left')
            end_idx_interp = np.searchsorted(batching_values, batch_core_max, side='right')
        
        batch_properties_interp = properties[start_idx_interp:end_idx_interp]
        
        # Interpolate grid densities back to spike positions using linear interpolation
        batch_cluster_den = _interpolate_grid_to_spikes(
            batch_properties_interp,
            batch_cluster_density,
            batch_bin_edges,
            batch_cluster_density.shape
        )
        batch_background_den = _interpolate_grid_to_spikes(
            batch_properties_interp,
            batch_background_density,
            batch_bin_edges,
            batch_background_density.shape
        )
        
        # Handle spikes outside grid: background = reg_density, cluster = 0
        # Check which spikes are outside grid boundaries
        out_of_bounds = np.zeros(len(batch_properties_interp), dtype=bool)
        for d in range(len(batch_bin_edges)):
            dim_values = batch_properties_interp[:, d]
            min_edge = batch_bin_edges[d][0]
            max_edge = batch_bin_edges[d][-1]
            out_of_bounds |= (dim_values < min_edge) | (dim_values > max_edge)
        
        # Set default values for out-of-bounds spikes
        batch_background_den[out_of_bounds] = reg_density
        batch_cluster_den[out_of_bounds] = 0.0
        
        # Cap background_den to be at least reg_density (for in-bounds spikes that might be very low)
        batch_background_den = np.maximum(batch_background_den, reg_density)
        
        del batch_background_density, batch_cluster_density
        
        # Store results for interpolated spikes
        cluster_den[start_idx_interp:end_idx_interp] = batch_cluster_den
        background_den[start_idx_interp:end_idx_interp] = batch_background_den
    
    return cluster_den, background_den
