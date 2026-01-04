# core/motion_analyzer.py
"""
Motion Physics Engine (Improved Signal Processing)
---------------------------------------------------
Extracts rhythmic motion patterns using optical flow and FFT analysis.
Designed to detect clonic seizure oscillations (2-7 Hz range).

Key Features:
- Timestamp-based sampling: Accurate FFT regardless of frame rate jitter
- Signal normalization: Consistent values across different camera setups
- Bandpass filtering: Removes camera jitter and lighting flicker
- EMA smoothing: Prevents false positives from transient spikes
- Noise gate: Prevents false positives when person is idle
"""

import cv2
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from collections import deque
import time
import config


class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None
        
        # Circular buffers for motion and timestamps
        self.motion_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.timestamp_buffer = deque(maxlen=config.BUFFER_SIZE)
        
        # Temporal smoothing state (EMA)
        self.smoothed_score = 0.0
        self.ema_alpha = config.MOTION_EMA_ALPHA
        
        # Frame size for normalization
        self.reference_pixels = 320 * 240  # Normalize to this reference size
        
    def get_motion_score(self, frame):
        """
        Main API: Analyze frame motion and return seizure confidence.
        
        Input: BGR video frame (can be full frame or ROI crop)
        Output: Float 0.0 -> 1.0 (Seizure Confidence Score)
        
        The confidence represents how much the motion pattern resembles
        rhythmic clonic seizure activity (2-7 Hz oscillations).
        """
        current_time = time.time()
        
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
            self.timestamp_buffer.append(current_time)
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
        
        # 4. Normalize motion by frame size
        # This ensures consistent values regardless of camera distance/resolution
        current_pixels = gray.shape[0] * gray.shape[1]
        normalization_factor = self.reference_pixels / max(current_pixels, 1)
        normalized_motion = avg_motion * normalization_factor
        
        # ========================================
        # NOISE GATE (Critical for false positives)
        # ========================================
        # If motion is below threshold, the person is essentially still.
        # Push zero to buffer and return zero confidence.
        if normalized_motion < config.MOTION_NOISE_FLOOR:
            self.motion_buffer.append(0.0)
            self.timestamp_buffer.append(current_time)
            self.prev_gray = gray
            # Decay smoothed score when idle
            self.smoothed_score = self.smoothed_score * 0.9
            return max(0.0, self.smoothed_score)
        
        # 5. Update Buffers with real motion
        self.motion_buffer.append(normalized_motion)
        self.timestamp_buffer.append(current_time)
        self.prev_gray = gray

        # 6. Physics Check (Spectral Analysis with improved processing)
        raw_score = self._analyze_spectrum()
        
        # 7. Apply Exponential Moving Average for temporal smoothing
        # This prevents single-frame spikes from causing false detections
        self.smoothed_score = self.ema_alpha * raw_score + (1 - self.ema_alpha) * self.smoothed_score
        
        return float(self.smoothed_score)

    def _bandpass_filter(self, data, fs):
        """
        Apply bandpass filter to remove DC drift and high-frequency noise.
        
        Filters out:
        - DC component (0 Hz) - average motion level
        - Very low frequencies (< 0.5 Hz) - slow drifts
        - High frequencies (> 10 Hz) - camera noise, sensor jitter
        
        Keeps:
        - 0.5 Hz to 10 Hz - includes seizure band (2-7 Hz) with margin
        """
        if len(data) < 12:  # Need minimum samples for filter
            return data
            
        nyq = 0.5 * fs
        lowcut = 0.5
        highcut = min(10.0, nyq * 0.9)  # Stay below Nyquist
        
        if highcut <= lowcut:
            return data
            
        low = lowcut / nyq
        high = highcut / nyq
        
        try:
            b, a = butter(2, [low, high], btype='band')
            # Use shorter padlen for small buffers
            padlen = min(len(data) - 1, 10)
            if padlen < 1:
                return data
            filtered = filtfilt(b, a, data, padlen=padlen)
            return filtered
        except Exception:
            # Fall back to raw data if filter fails
            return data

    def _estimate_actual_fps(self):
        """
        Estimate actual FPS from timestamp buffer.
        
        This is critical for accurate FFT frequency mapping.
        If frames arrive at variable rates, using assumed FPS
        would map frequencies incorrectly.
        """
        if len(self.timestamp_buffer) < 2:
            return config.FPS_ASSUMED
            
        timestamps = np.array(self.timestamp_buffer)
        time_diffs = np.diff(timestamps)
        
        if len(time_diffs) == 0 or np.mean(time_diffs) == 0:
            return config.FPS_ASSUMED
            
        # Remove outliers (frames that took too long)
        median_diff = np.median(time_diffs)
        valid_diffs = time_diffs[time_diffs < median_diff * 3]
        
        if len(valid_diffs) == 0:
            return config.FPS_ASSUMED
            
        actual_fps = 1.0 / np.mean(valid_diffs)
        
        # Clamp to reasonable range
        return np.clip(actual_fps, 10, 60)

    def _analyze_spectrum(self):
        """
        Perform FFT on the motion buffer to detect seizure frequencies.
        
        Improvements over original:
        1. Uses actual FPS from timestamps (not assumed)
        2. Applies bandpass filtering before FFT
        3. Better spectral leakage reduction
        
        Clonic seizures produce strong, rhythmic oscillations in the 2-7 Hz 
        range. We compute the ratio of energy in this band vs total energy.
        """
        if len(self.motion_buffer) < config.BUFFER_SIZE:
            return 0.0

        # Prepare Data
        data = np.array(self.motion_buffer)
        
        # Remove DC offset (the "average" motion level)
        data = data - np.mean(data)
        
        # Check if there's any meaningful variation
        if np.std(data) < 0.01:
            return 0.0
        
        # Get actual FPS from timestamps
        actual_fps = self._estimate_actual_fps()
        
        # Apply bandpass filter to remove noise BEFORE FFT
        filtered_data = self._bandpass_filter(data, actual_fps)
        
        # Windowing (Hanning) to reduce spectral leakage
        windowed_data = filtered_data * np.hanning(len(filtered_data))
        
        # FFT Calculation using actual FPS
        N = len(windowed_data)
        yf = fft(windowed_data)
        xf = fftfreq(N, 1 / actual_fps)

        # Power Spectrum (only positive frequencies)
        power = np.abs(yf[:N//2])
        freqs = xf[:N//2]

        # ========================================
        # ENERGY BAND CALCULATIONS
        # ========================================
        # Total relevant energy: everything from 0.5 Hz to 10 Hz
        mask_total = (freqs >= 0.5) & (freqs <= 10.0)
        
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
        
        # Also consider the absolute energy magnitude
        # A high ratio with weak signal is less reliable
        energy_confidence = min(1.0, energy_seizure / (config.ENERGY_NOISE_FLOOR * 3))
        
        # Combined confidence: ratio weighted by energy significance
        combined_confidence = ratio * 0.7 + energy_confidence * 0.3
        
        # Scale and clip for output
        confidence = np.clip(combined_confidence * 1.5, 0.0, 1.0)
        
        return float(confidence)
    
    def reset(self):
        """Reset analyzer state (useful when switching targets)."""
        self.prev_gray = None
        self.motion_buffer.clear()
        self.timestamp_buffer.clear()
        self.smoothed_score = 0.0