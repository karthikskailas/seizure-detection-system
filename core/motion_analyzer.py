# core/motion_analyzer.py
import cv2
import numpy as np
from scipy.fft import fft, fftfreq
from collections import deque
import config

class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None
        # Circular buffer to store motion magnitude history
        self.motion_buffer = deque(maxlen=config.BUFFER_SIZE)
        
    def get_motion_score(self, frame):
        """
        Input: Raw video frame
        Output: Float 0.0 -> 1.0 (Seizure Confidence Score)
        """
        # 1. Resize & Grayscale (Optimization for Speed)
        h, w = frame.shape[:2]
        scale = config.RESIZE_WIDTH / w
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # 2. Optical Flow (Farneback)
        # Optimized for CPU: reduced iterations and window size
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 
            pyr_scale=0.5, levels=1, winsize=15, 
            iterations=1, poly_n=5, poly_sigma=1.1, flags=0
        )

        # 3. Calculate Magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = np.mean(magnitude)
        
        # 4. Update Buffer
        self.motion_buffer.append(avg_motion)
        self.prev_gray = gray

        # 5. Physics Check (Spectral Analysis)
        return self._analyze_spectrum()

    def _analyze_spectrum(self):
        """Internal: Performs FFT on the motion buffer"""
        if len(self.motion_buffer) < config.BUFFER_SIZE:
            return 0.0

        # Prepare Data
        data = np.array(self.motion_buffer)
        data = data - np.mean(data) # Remove DC offset (steady state)
        
        # Windowing (Hanning) to reduce spectral leakage
        windowed_data = data * np.hanning(len(data))
        
        # FFT Calculation
        N = config.BUFFER_SIZE
        yf = fft(windowed_data)
        xf = fftfreq(N, 1 / config.FPS_ASSUMED)

        # Power Spectrum
        power = np.abs(yf[:N//2])
        freqs = xf[:N//2]

        # Calculate Energy in Bands
        mask_normal = (freqs >= config.FREQ_NORMAL_LOW) & (freqs <= config.FREQ_NORMAL_HIGH)
        mask_seizure = (freqs >= config.FREQ_SEIZURE_LOW) & (freqs <= config.FREQ_SEIZURE_HIGH)

        energy_normal = np.sum(power[mask_normal])
        energy_seizure = np.sum(power[mask_seizure])
        
        # Avoid division by zero
        total_relevant_energy = energy_normal + energy_seizure + 1e-6
        
        # Ratio: Seizure Energy / Total Energy
        ratio = energy_seizure / total_relevant_energy
        
        # Scale for detection (Heuristic boost)
        confidence = np.clip(ratio * 2.5, 0.0, 1.0)
        
        return confidence