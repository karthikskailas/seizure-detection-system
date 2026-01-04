# config.py
"""
Seizure Detection System Configuration
---------------------------------------
STRICT TUNING - Prioritizes accuracy over speed to reduce false alarms.
"""

# --- Camera & Processing ---
RESIZE_WIDTH = 320          # Processing width
FPS_ASSUMED = 30            # Target FPS
BUFFER_SECONDS = 3.0        # 3 seconds - more data for FFT accuracy
BUFFER_SIZE = int(FPS_ASSUMED * BUFFER_SECONDS)  # 90 frames

# --- Signal Processing (STRICT - heavy smoothing) ---
MOTION_EMA_ALPHA = 0.15          # Lower = heavier smoothing, fewer spikes
VELOCITY_SMOOTH_FRAMES = 7       # More frames for velocity averaging
BASELINE_CALIBRATION_SECONDS = 8 # Longer calibration for better baseline

# --- Noise Gate (STRICT - filters out normal movement) ---
MOTION_NOISE_FLOOR = 0.8    # High - ignores normal fidgeting
ENERGY_NOISE_FLOOR = 15.0   # High - needs strong rhythmic signal

# --- Fall Detection (STRICT - prevents false falls) ---
FALL_VELOCITY_THRESHOLD = 0.08   # Higher - needs faster downward motion
FALL_CONFIRMATION_FRAMES = 5     # More frames to confirm fall

# --- Spectral Analysis ---
FREQ_NORMAL_LOW = 0.1       
FREQ_NORMAL_HIGH = 2.0
FREQ_SEIZURE_LOW = 2.5      # Narrower seizure band: 2.5-6 Hz (more specific)
FREQ_SEIZURE_HIGH = 6.0     

# --- Decision Logic (STRICT - requires strong sustained signal) ---
MOTION_THRESHOLD = 0.75     # 75% confidence required (was 60%)
FALL_TRIGGER_SENSITIVITY = 0.8 
DURATION_THRESHOLD = 25     # ~0.8 seconds sustained (was 10 frames)
COOLDOWN_FRAMES = 150       # 5 seconds cooldown (was 3)

# --- Foreground Isolation ---
FG_HISTORY = 300              
FG_VAR_THRESHOLD = 20         
FG_MIN_AREA_RATIO = 0.005     
FG_STABILITY_THRESHOLD = 0.2  
FG_LOST_FRAMES_MAX = 15       

# --- ROI Validation ---
MIN_ROI_RATIO = 0.05          # ROI must be at least 5% of frame

# --- Alert System ---
ALERT_COOLDOWN_SECONDS = 30   # Minimum time between alerts (prevents spam)
ALERT_SOUND_DURATION = 5      # Audio alarm duration in seconds
ALERT_CONFIG_FILE = "data/alert_config.json"  # User-editable alert settings
