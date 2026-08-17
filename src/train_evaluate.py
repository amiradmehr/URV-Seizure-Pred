# Training loops, AUC, and FPR/hour calculations
import numpy as np
from feature_engineering import extract_features_from_window

def convert_3d_eeg_to_features(X_3d_array):
    """
    Takes the 3D array from preprocessing (windows, channels, samples)
    and converts it into a 2D array of engineered features (windows, features).
    """
    n_windows, n_channels, n_samples = X_3d_array.shape
    
    # We have 13 features per channel, so total features = n_channels * 13
    all_engineered_windows = []
    
    print(f"Extracting features for {n_windows} windows...")
    
    for i in range(n_windows):
        window_features = []
        
        # Extract features for each channel separately and combine them
        for ch in range(n_channels):
            single_channel_data = X_3d_array[i, ch, :]
            # Using 256.0 Hz since preprocessing enforces the native rate
            ch_feats = extract_features_from_window(single_channel_data, sfreq=256.0)
            window_features.extend(ch_feats)
            
        all_engineered_windows.append(window_features)
        
    return np.array(all_engineered_windows)