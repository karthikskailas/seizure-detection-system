# config.py
"""
Seizure Detection System Configuration
---------------------------------------
All tunable parameters in one place for easy adjustment.
"""

# --- Camera & Processing ---
RESIZE_WIDTH = 320          # Processing width (320px is faster than 200px while maintaining accuracy)
FPS_ASSUMED = 30            # Target FPS for FFT calculations (match your camera)
BUFFER_SECONDS = 4.0        # Seconds of history for FFT analysis (4s = good spectral resolution)
BUFFER_SIZE = int(FPS_ASSUMED * BUFFER_SECONDS)  # 120 frames at 30fps

# --- Noise Gate (CRITICAL for false positive prevention) ---
# Motion below this threshold is treated as "no motion" (sensor noise)
# This prevents false detections when the person is sitting still
MOTION_NOISE_FLOOR = 0.5    # Minimum optical flow magnitude to consider as real motion
ENERGY_NOISE_FLOOR = 10.0   # Minimum absolute FFT energy to consider as real activity

# --- Spectral Analysis (The Physics) ---
# Frequencies in Hertz (Hz)
# Normal human movement: slow, irregular (0.1 - 2 Hz)
# Clonic seizure shaking: fast, rhythmic (2 - 7 Hz)
FREQ_NORMAL_LOW = 0.1       # Background motion band (walking, swaying)
FREQ_NORMAL_HIGH = 2.0
FREQ_SEIZURE_LOW = 2.0      # Seizure/Convulsive frequency band
FREQ_SEIZURE_HIGH = 7.0     # Clonic seizures rarely exceed 7 Hz

# --- Decision Logic ---
MOTION_THRESHOLD = 0.60     # Confidence score (0.0 - 1.0) required to trigger shaking alert
FALL_TRIGGER_SENSITIVITY = 0.7 
DURATION_THRESHOLD = 20     # Consecutive frames required (~0.67s at 30fps)
COOLDOWN_FRAMES = 90        # 3 seconds cooldown after an alert

# --- Foreground Isolation ---
FG_HISTORY = 500              # MOG2 history frames (higher = more stable background model)
FG_VAR_THRESHOLD = 16         # Variance threshold for MOG2 (lower = more sensitive)
FG_MIN_AREA_RATIO = 0.01      # Minimum contour area as fraction of frame (1%)
FG_STABILITY_THRESHOLD = 0.3  # IoU threshold for tracking stability
FG_LOST_FRAMES_MAX = 30       # Frames before resetting tracker (~1 second)