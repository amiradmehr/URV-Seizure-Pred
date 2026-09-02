import numpy as np
def normalize_window_orig(window, availability_columns):
    """Verbatim reimplementation of the pre-fix features.normalize_window
    (IQR floor 1e-6, no clipping, no dead-channel zeroing)."""
    if window.size == 0:
        return window
    q = np.percentile(window, [25, 50, 75], axis=0)
    median = q[1]
    spread = np.maximum(q[2] - q[0], 1e-6)
    normalized = (window - median) / spread
    normalized[:, ~np.asarray(availability_columns, dtype=bool)] = 0.0
    return normalized.astype(np.float32, copy=False)
