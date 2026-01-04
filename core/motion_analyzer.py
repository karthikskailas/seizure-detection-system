# core/motion_analyzer.py
"""
Motion Physics Engine
---------------------
Extracts rhythmic motion patterns using optical flow and FFT analysis.
Designed to detect clonic seizure oscillations (2-7 Hz range).

Key Features:
- Noise gate: Prevents false positives when person is idle
- Farneback optical flow: CPU-efficient motion extraction
- FFT spectral analysis: Identifies seizure-frequency energy
"""

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
        Main API: Analyze frame motion and return seizure confidence.
        
        Input: BGR video frame (can be full frame or ROI crop)
        Output: Float 0.0 -> 1.0 (Seizure Confidence Score)
        
        The confidence represents how much the motion pattern resembles
        rhythmic clonic seizure activity (2-7 Hz oscillations).
        """
        # 1. Resize & Grayscale (Optimization for Speed)
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return 0.0
            
        scale = config.RESIZE_WIDTH / w
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce camera sensor noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Handle first frame or size change
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return 0.0

        # 2. Optical Flow (Farneback)
        # Optimized for CPU: reduced iterations and window size
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 
            pyr_scale=0.5, levels=1, winsize=15, 
            iterations=1, poly_n=5, poly_sigma=1.1, flags=0
        )

        # 3. Calculate Motion Magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = np.mean(magnitude)
        
        # ========================================
        # NOISE GATE (Critical for false positives)
        # ========================================
        # If motion is below threshold, the person is essentially still.
        # Push zero to buffer and return zero confidence.
        # This prevents "detecting seizures" in idle people.
        if avg_motion < config.MOTION_NOISE_FLOOR:
            self.motion_buffer.append(0.0)
            self.prev_gray = gray
            return 0.0
        
        # 4. Update Buffer with real motion
        self.motion_buffer.append(avg_motion)
        self.prev_gray = gray

        # 5. Physics Check (Spectral Analysis)
        return self._analyze_spectrum()

    def _analyze_spectrum(self):
        """
        Perform FFT on the motion buffer to detect seizure frequencies.
        
        Clonic seizures produce strong, rhythmic oscillations in the 2-7 Hz 
        range. We compute the ratio of energy in this band vs total energy.
        """
        if len(self.motion_buffer) < config.BUFFER_SIZE:
            return 0.0

        # Prepare Data
        data = np.array(self.motion_buffer)
        
        # Remove DC offset (the "average" motion level)
        # This isolates the oscillatory component we care about
        data = data - np.mean(data)
        
        # Check if there's any meaningful variation
        if np.std(data) < 0.01:
            return 0.0
        
        # Windowing (Hanning) to reduce spectral leakage
        windowed_data = data * np.hanning(len(data))
        
        # FFT Calculation
        N = config.BUFFER_SIZE
        yf = fft(windowed_data)
        xf = fftfreq(N, 1 / config.FPS_ASSUMED)

        # Power Spectrum (only positive frequencies)
        power = np.abs(yf[:N//2])
        freqs = xf[:N//2]

        # ========================================
        # ENERGY BAND CALCULATIONS
        # ========================================
        # Total relevant energy: everything from 0.1 Hz to 10 Hz
        mask_total = (freqs >= 0.1) & (freqs <= 10.0)
        
        # Seizure band: 2-7 Hz (clonic oscillation range)
        mask_seizure = (freqs >= config.FREQ_SEIZURE_LOW) & (freqs <= config.FREQ_SEIZURE_HIGH)

        energy_total = np.sum(power[mask_total])
        energy_seizure = np.sum(power[mask_seizure])
        
        # ========================================
        # ABSOLUTE ENERGY GATE
        # ========================================
        # Even if the ratio is high, if absolute energy is low,
        # it's probably just noise, not a real seizure.
        if energy_seizure < config.ENERGY_NOISE_FLOOR:
            return 0.0
        
        # Avoid division by zero
        if energy_total < 1e-6:
            return 0.0
        
        # Ratio: Seizure Energy / Total Energy
        ratio = energy_seizure / energy_total
        
        # Scale and clip for output
        # A ratio > 0.4 is very significant for seizure-like motion
        confidence = np.clip(ratio * 2.0, 0.0, 1.0)
        
        return float(confidence)
    
    def reset(self):
        """Reset analyzer state (useful when switching targets)."""
        self.prev_gray = None
        self.motion_buffer.clear()