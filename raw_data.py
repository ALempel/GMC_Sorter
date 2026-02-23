import re
import os
import numpy as np
import struct


def parse_npx_metadata(file_path):
    """
    Parse Neuropixels metadata from .meta file.
    
    Parameters
    ----------
    file_path : str
        Path to the .meta file.
    
    Returns
    -------
    tuple
        Tuple containing (in order):
        - n_saved_chans : int
            Number of saved channels
        - n_time_samples : int
            Length of recording in samples
        - sample_rate : float
            Sample rate in Hz
    """
    with open(file_path, 'r') as f:
        header = f.read()

    # Get sample frequency
    match_samp = re.search(r'imSampRate=(\d+)', header)
    samp_freq = float(match_samp.group(1)) if match_samp else None

    # Get file size in bytes
    match_size = re.search(r'fileSizeBytes=(\d+)', header)
    file_size_bytes = int(match_size.group(1)) if match_size else None

    # Get total number of saved channels (including AP and SY channels)
    match_nch = re.search(r'nSavedChans=(\d+)', header)
    n_saved_chans = int(match_nch.group(1)) if match_nch else None
    
    # Calculate total number of time samples
    n_time_samples = None
    if file_size_bytes and n_saved_chans:
        bytes_per_sample = 2  # int16
        n_time_samples = file_size_bytes // (n_saved_chans * bytes_per_sample)
    
    # Return in order: n_saved_chans, n_time_samples, sample_rate
    return (n_saved_chans, n_time_samples, samp_freq)


def parse_dat_t_s_metadata(exp_folder):
    """
    Parse dat_t_s metadata from timestamps and samples files.
    
    Parameters
    ----------
    exp_folder : str
        Path to the experimental folder containing the data files.
    
    Returns
    -------
    tuple
        Tuple containing (in order):
        - n_saved_chans : int
            Number of saved channels
        - n_time_samples : int
            Length of recording in samples
        - sample_rate : float
            Sample rate in Hz
    """
    # Find timestamps and samples files
    files = os.listdir(exp_folder)
    timestamps_file = None
    samples_file = None
    
    for f in files:
        if f.endswith('_timestamps.dat'):
            timestamps_file = os.path.join(exp_folder, f)
        elif f.endswith('_samples.dat'):
            samples_file = os.path.join(exp_folder, f)
    
    if timestamps_file is None:
        raise FileNotFoundError(f"No *_timestamps.dat file found in {exp_folder}")
    if samples_file is None:
        raise FileNotFoundError(f"No *_samples.dat file found in {exp_folder}")
    
    # Calculate n_time_samples from timestamps file size (int64 = 8 bytes per timestamp)
    timestamps_file_size = os.path.getsize(timestamps_file)
    bytes_per_timestamp = 8  # int64
    n_time_samples = timestamps_file_size // bytes_per_timestamp
    
    # Read only first 1000 timestamps to calculate sample rate
    n_timestamps_to_read = min(1000, n_time_samples)
    timestamps = np.fromfile(timestamps_file, dtype=np.int64, count=n_timestamps_to_read)
    
    # Calculate sample rate from timestamps (convert microseconds to seconds)
    if len(timestamps) > 1:
        # Calculate median interval between samples
        intervals_us = np.diff(timestamps)
        median_interval_us = np.median(intervals_us)
        sample_rate = 1e6 / median_interval_us  # Convert microseconds to Hz
    else:
        sample_rate = None
    
    # Calculate n_saved_chans from samples file size
    # Samples are stored as int16, organized as: [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ...]
    # So file_size = n_time_samples * n_saved_chans * 2 bytes
    samples_file_size = os.path.getsize(samples_file)
    bytes_per_sample = 2  # int16
    n_saved_chans = samples_file_size // (n_time_samples * bytes_per_sample)
    
    # Return in order: n_saved_chans, n_time_samples, sample_rate
    return (n_saved_chans, n_time_samples, sample_rate)


class raw_data:
    """
    Class to handle raw data from different acquisition systems.
    
    Parameters
    ----------
    exp_folder : str
        Path to the experimental folder containing the data files.
    file_struct : str
        File structure identifier (e.g., 'NPx', 'dat_t_s').
    """
    
    def __init__(self, exp_folder, file_struct):
        self.exp_folder = exp_folder
        self.file_struct = file_struct
        self._meta_info_cache = None
    
    @property
    def bytes_per_sample_per_channel(self):
        """
        Get the number of bytes per sample per channel based on file structure.
        
        Returns
        -------
        int
            Bytes per sample per channel.
        """
        if self.file_struct == 'dat_t_s':
            return 2  # int16
        elif self.file_struct == 'NPx':
            return 2  # int16 - TODO: Double-check this is correct for NPx
        else:
            raise ValueError(f"Unknown file structure: {self.file_struct}")
    
    def meta_info(self):
        """
        Read metadata information from the data files.
        
        Returns
        -------
        tuple
            Tuple containing (in order):
            - n_saved_chans : int
                Number of saved channels
            - n_time_samples : int
                Length of recording in samples
            - sample_rate : float
                Sample rate in Hz
        """
        if self._meta_info_cache is not None:
            return self._meta_info_cache
        
        if self.file_struct == 'NPx':
            # Find meta file in exp_folder
            meta_files = [f for f in os.listdir(self.exp_folder) if f.endswith('.meta')]
            if not meta_files:
                raise FileNotFoundError(f"No .meta file found in {self.exp_folder}")
            meta_file = os.path.join(self.exp_folder, meta_files[0])
            
            # Parse metadata - returns (n_saved_chans, n_time_samples, sample_rate)
            result = parse_npx_metadata(meta_file)
            
        elif self.file_struct == 'dat_t_s':
            # Parse metadata - returns (n_saved_chans, n_time_samples, sample_rate)
            result = parse_dat_t_s_metadata(self.exp_folder)
        else:
            raise ValueError(f"Unsupported file structure: {self.file_struct}")
        
        # Cache the result
        self._meta_info_cache = result
        return result
    
    def extract_raw_data(self, time_interval, channels=None):
        """
        Extract raw data samples over a time interval and optionally for specific channels.
        
        Parameters
        ----------
        time_interval : tuple
            Time interval to extract, (start_time, end_time) in seconds.
        channels : list, optional
            List of channel indices to extract. If None, extracts all channels.
        
        Returns
        -------
        tuple
            Tuple containing:
            - data : numpy.ndarray
                Extracted data with shape (n_channels, n_samples) if channels specified,
                or (n_saved_chans, n_samples) if all channels.
            - timestamps : numpy.ndarray
                Timestamps for each sample in seconds, shape (n_samples,).
        """
        if self.file_struct == 'NPx':
            # TODO: Implement NPx data extraction
            raise NotImplementedError(f"extract_raw_data for {self.file_struct} is not yet implemented")
        
        elif self.file_struct == 'dat_t_s':
            return self._extract_dat_t_s_data(time_interval, channels)
        
        else:
            raise ValueError(f"Unsupported file structure: {self.file_struct}")
    
    def _extract_dat_t_s_data(self, time_interval, channels=None):
        """
        Extract data from dat_t_s file structure.
        
        Parameters
        ----------
        time_interval : tuple
            Time interval (start_time, end_time) in seconds.
        channels : list, optional
            List of channel indices to extract.
        
        Returns
        -------
        tuple
            Tuple containing:
            - data : numpy.ndarray
                Extracted data with shape (n_channels, n_samples).
            - timestamps : numpy.ndarray
                Timestamps for each sample in seconds, shape (n_samples,).
        """
        # Find timestamps and samples files
        files = os.listdir(self.exp_folder)
        timestamps_file = None
        samples_file = None
        
        for f in files:
            if f.endswith('_timestamps.dat'):
                timestamps_file = os.path.join(self.exp_folder, f)
            elif f.endswith('_samples.dat'):
                samples_file = os.path.join(self.exp_folder, f)
        
        if timestamps_file is None or samples_file is None:
            raise FileNotFoundError(f"Required dat_t_s files not found in {self.exp_folder}")
        
        # Get metadata
        n_saved_chans, n_time_samples, _ = self.meta_info()
        
        # Convert time interval from seconds to microseconds
        start_time_us = int(time_interval[0] * 1e6)
        end_time_us = int(time_interval[1] * 1e6)
        
        # Binary search on timestamps file without loading entire file
        # Read only the specific bytes needed for binary search
        bytes_per_timestamp = 8  # int64
        
        def read_timestamp_at_index(idx):
            """Read a single timestamp at given index without loading entire file."""
            with open(timestamps_file, 'rb') as f:
                f.seek(idx * bytes_per_timestamp)
                return np.frombuffer(f.read(bytes_per_timestamp), dtype=np.int64)[0]
        
        # Binary search for start_idx (first index where timestamp > start_time_us)
        left, right = 0, n_time_samples - 1
        start_idx = n_time_samples
        while left <= right:
            mid = (left + right) // 2
            mid_timestamp = read_timestamp_at_index(mid)
            if mid_timestamp > start_time_us:
                start_idx = mid
                right = mid - 1
            else:
                left = mid + 1
        
        # Binary search for end_idx (last index where timestamp < end_time_us)
        left, right = 0, n_time_samples - 1
        end_idx = 0
        while left <= right:
            mid = (left + right) // 2
            mid_timestamp = read_timestamp_at_index(mid)
            if mid_timestamp < end_time_us:
                end_idx = mid + 1
                left = mid + 1
            else:
                right = mid - 1
        
        # Ensure valid range
        start_idx = max(0, start_idx)
        end_idx = min(n_time_samples, end_idx)
        
        if start_idx >= end_idx:
            # No samples in the interval
            n_extracted_samples = 0
            if channels is not None:
                empty_data = np.empty((len(channels), 0), dtype=np.int16)
            else:
                empty_data = np.empty((n_saved_chans, 0), dtype=np.int16)
            empty_timestamps = np.empty(0, dtype=np.float64)
            return empty_data, empty_timestamps
        
        n_extracted_samples = end_idx - start_idx
        
        # Read timestamps for the extracted range
        timestamps_mmap = np.memmap(timestamps_file, dtype=np.int64, mode='r')
        extracted_timestamps_us = timestamps_mmap[start_idx:end_idx]
        # Convert from microseconds to seconds
        extracted_timestamps = extracted_timestamps_us.astype(np.float64) / 1e6
        
        # Read samples from the samples file
        # Samples are organized as: [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ...]
        # For sample i, data starts at byte offset: i * n_saved_chans * 2 (int16 = 2 bytes)
        samples_mmap = np.memmap(samples_file, dtype=np.int16, mode='r')
        
        # Reshape to (n_time_samples, n_saved_chans)
        samples_reshaped = samples_mmap.reshape(n_time_samples, n_saved_chans)
        
        # Extract the time range
        extracted_samples = samples_reshaped[start_idx:end_idx, :]
        
        # Extract specific channels if requested
        if channels is not None:
            extracted_samples = extracted_samples[:, channels]
            
        extracted_samples = np.array(extracted_samples, copy=True)
        # Transpose to (n_channels, n_samples) format
        return extracted_samples.T, extracted_timestamps