# Baseline feature extraction (RMS, entropy, etc.)
import numpy as np
import scipy.signal as signal
import scipy.stats as stats

def compute_time_domain(window):
    """Calculates basic statistical features over a window of EEG data."""
    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2))
    
    # Variance, Skewness, Kurtosis
    variance = np.var(window)
    skewness = float(stats.skew(window))
    kurtosis = float(stats.kurtosis(window))
    
    # Shannon Entropy approximation
    # Create a histogram of the data to estimate probability density
    hist, _ = np.histogram(window, density=True, bins='auto')
    hist = hist[hist > 0] # Remove zeros to avoid log(0) errors
    entropy = float(stats.entropy(hist))
    
    return [rms, variance, skewness, kurtosis, entropy]

def compute_frequency_domain(window, sfreq=256.0):
    """Calculates absolute bandpower for standard EEG frequency bands."""
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 40.0)
    }
    
    # Calculate Power Spectral Density (PSD) using Welch's method
    # nperseg defines the length of each segment for the FFT (using 2-second segments here)
    nperseg = min(int(sfreq * 2), len(window))
    freqs, psd = signal.welch(window, fs=sfreq, nperseg=nperseg)
    
    bandpowers = []
    for band, (low, high) in bands.items():
        # Find the indices of frequencies that fall within our band
        idx = np.logical_and(freqs >= low, freqs <= high)
        # Integrate the area under the curve to get the total power in that band
        power = np.trapezoid(psd[idx], freqs[idx])
        bandpowers.append(power)
        
    return bandpowers

def compute_hjorth_parameters(window):
    """Calculates Activity, Mobility, and Complexity of the EEG signal."""
    first_deriv = np.diff(window)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(window)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    # 1. Activity (same as variance)
    activity = var_zero
    
    # 2. Mobility
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0.0
    
    # 3. Complexity
    complexity = np.sqrt(var_d2 / var_d1) / mobility if mobility > 0 and var_d1 > 0 else 0.0

    return [activity, mobility, complexity]

def extract_features_from_window(window, sfreq=256.0):
    """
    Master function to run all feature extractions on a single window of data.
    Returns a single flat 1D numpy array of all features combined.
    """
    time_feats = compute_time_domain(window)
    freq_feats = compute_frequency_domain(window, sfreq)
    hjorth_feats = compute_hjorth_parameters(window)
    
    # Combine them all into one flat array
    all_features = np.array(time_feats + freq_feats + hjorth_feats)
    return all_features

if __name__ == "__main__":
    # Quick sanity check with some fake random EEG data (10 seconds at 256Hz)
    dummy_window = np.random.normal(0, 1, 256 * 10)
    features = extract_features_from_window(dummy_window)
    
    print(f"Success! Extracted {len(features)} features per channel.")
    print("Feature array:", features)