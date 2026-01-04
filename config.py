# config.py

# --- Camera & Processing ---
RESIZE_WIDTH = 200          # Processing width (Keep low for CPU speed)
FPS_ASSUMED = 30            # Target FPS for FFT calculations
BUFFER_SECONDS = 2.0        # How many seconds of history to keep
BUFFER_SIZE = int(FPS_ASSUMED * BUFFER_SECONDS) # 60 frames

# --- Spectral Analysis (The Math) ---
# Frequencies in Hertz (Hz)
FREQ_NORMAL_LOW = 0.5       # Normal movement (waving, walking)
FREQ_NORMAL_HIGH = 2.5
FREQ_SEIZURE_LOW = 3.0      # Seizure/Convulsive range
FREQ_SEIZURE_HIGH = 12.0

# --- Decision Logic ---
MOTION_THRESHOLD = 0.55     # Confidence score (0.0 - 1.0) to trigger shaking
FALL_TRIGGER_SENSITIVITY = 0.7 
DURATION_THRESHOLD = 15     # Consecutive frames required to trigger alert (approx 0.5s)
COOLDOWN_FRAMES = 90        # 3 seconds cooldown after an alert