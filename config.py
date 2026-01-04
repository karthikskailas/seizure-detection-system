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

# --- Noise Gate (STRICT - prevents idle false positives) ---
MOTION_NOISE_FLOOR = 0.5    # Higher = ignores small movements
ENERGY_NOISE_FLOOR = 8.0    # Higher = needs stronger rhythmic signal

# --- Spectral Analysis ---
FREQ_NORMAL_LOW = 0.1       
FREQ_NORMAL_HIGH = 2.0
FREQ_SEIZURE_LOW = 2.0      # Seizure band: 2-7 Hz
FREQ_SEIZURE_HIGH = 7.0     

# --- Decision Logic (FASTER but STRICTER) ---
MOTION_THRESHOLD = 0.60     # 60% confidence to trigger (higher = less false positives)
FALL_TRIGGER_SENSITIVITY = 0.7 
DURATION_THRESHOLD = 10     # ~0.3 seconds - FASTER detection
COOLDOWN_FRAMES = 90        # 3 seconds cooldown

# --- Foreground Isolation ---
FG_HISTORY = 300              
FG_VAR_THRESHOLD = 20         
FG_MIN_AREA_RATIO = 0.005     
FG_STABILITY_THRESHOLD = 0.2  
FG_LOST_FRAMES_MAX = 15       

# --- ROI Validation ---
MIN_ROI_RATIO = 0.05          # ROI must be at least 5% of frame