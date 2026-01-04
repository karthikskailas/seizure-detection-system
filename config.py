# config.py
"""
Seizure Detection System Configuration
---------------------------------------
BALANCED TUNING - Fast but accurate detection.
"""

# --- Camera & Processing ---
RESIZE_WIDTH = 320          # Processing width
FPS_ASSUMED = 30            # Target FPS
BUFFER_SECONDS = 2.0        # 2 seconds - good balance
BUFFER_SIZE = int(FPS_ASSUMED * BUFFER_SECONDS)  # 60 frames

# --- Noise Gate (BALANCED) ---
MOTION_NOISE_FLOOR = 0.3    # Moderate noise floor
ENERGY_NOISE_FLOOR = 5.0    # Moderate energy threshold

# --- Spectral Analysis ---
FREQ_NORMAL_LOW = 0.1       
FREQ_NORMAL_HIGH = 2.0
FREQ_SEIZURE_LOW = 2.0      # Seizure band: 2-7 Hz
FREQ_SEIZURE_HIGH = 7.0     

# --- Decision Logic (BALANCED) ---
MOTION_THRESHOLD = 0.50     # 50% confidence to trigger
FALL_TRIGGER_SENSITIVITY = 0.7 
DURATION_THRESHOLD = 15     # ~0.5 seconds sustained detection
COOLDOWN_FRAMES = 90        # 3 seconds cooldown

# --- Foreground Isolation ---
FG_HISTORY = 300              
FG_VAR_THRESHOLD = 20         
FG_MIN_AREA_RATIO = 0.005     
FG_STABILITY_THRESHOLD = 0.2  
FG_LOST_FRAMES_MAX = 15       

# --- ROI Validation ---
MIN_ROI_RATIO = 0.05          # ROI must be at least 5% of frame