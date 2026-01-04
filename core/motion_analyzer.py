# core/motion_analyzer.py
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
        self.motion_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.timestamp_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.smoothed_score = 0.0
        self.ema_alpha = config.MOTION_EMA_ALPHA
        self.reference_pixels = 320 * 240
        
    def get_motion_score(self, frame):
        current_time = time.time()
        
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return 0.0
            
        scale = config.RESIZE_WIDTH / w
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.timestamp_buffer.append(current_time)
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 
            pyr_scale=0.5, levels=1, winsize=15, 
            iterations=1, poly_n=5, poly_sigma=1.1, flags=0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = np.mean(magnitude)
        
        current_pixels = gray.shape[0] * gray.shape[1]
        normalization_factor = self.reference_pixels / max(current_pixels, 1)
        normalized_motion = avg_motion * normalization_factor
        
        if normalized_motion < config.MOTION_NOISE_FLOOR:
            self.motion_buffer.append(0.0)
            self.timestamp_buffer.append(current_time)
            self.prev_gray = gray
            self.smoothed_score = self.smoothed_score * 0.9
            return max(0.0, self.smoothed_score)
        
        self.motion_buffer.append(normalized_motion)
        self.timestamp_buffer.append(current_time)
        self.prev_gray = gray

        raw_score = self._analyze_spectrum()
        self.smoothed_score = self.ema_alpha * raw_score + (1 - self.ema_alpha) * self.smoothed_score
        
        return float(self.smoothed_score)

    def _bandpass_filter(self, data, fs):
        if len(data) < 12:
            return data
            
        nyq = 0.5 * fs
        lowcut = 0.5
        highcut = min(10.0, nyq * 0.9)
        
        if highcut <= lowcut:
            return data
            
        low = lowcut / nyq
        high = highcut / nyq
        
        try:
            b, a = butter(2, [low, high], btype='band')
            padlen = min(len(data) - 1, 10)
            if padlen < 1:
                return data
            filtered = filtfilt(b, a, data, padlen=padlen)
            return filtered
        except Exception:
            return data

    def _estimate_actual_fps(self):
        if len(self.timestamp_buffer) < 2:
            return config.FPS_ASSUMED
            
        timestamps = np.array(self.timestamp_buffer)
        time_diffs = np.diff(timestamps)
        
        if len(time_diffs) == 0 or np.mean(time_diffs) == 0:
            return config.FPS_ASSUMED
            
        median_diff = np.median(time_diffs)
        valid_diffs = time_diffs[time_diffs < median_diff * 3]
        
        if len(valid_diffs) == 0:
            return config.FPS_ASSUMED
            
        actual_fps = 1.0 / np.mean(valid_diffs)
        return np.clip(actual_fps, 10, 60)

    def _analyze_spectrum(self):
        if len(self.motion_buffer) < config.BUFFER_SIZE:
            return 0.0

        data = np.array(self.motion_buffer)
        data = data - np.mean(data)
        
        if np.std(data) < 0.01:
            return 0.0
        
        actual_fps = self._estimate_actual_fps()
        filtered_data = self._bandpass_filter(data, actual_fps)
        windowed_data = filtered_data * np.hanning(len(filtered_data))
        
        N = len(windowed_data)
        yf = fft(windowed_data)
        xf = fftfreq(N, 1 / actual_fps)

        power = np.abs(yf[:N//2])
        freqs = xf[:N//2]

        mask_total = (freqs >= 0.5) & (freqs <= 10.0)
        mask_seizure = (freqs >= config.FREQ_SEIZURE_LOW) & (freqs <= config.FREQ_SEIZURE_HIGH)

        energy_total = np.sum(power[mask_total])
        energy_seizure = np.sum(power[mask_seizure])
        
        if energy_seizure < config.ENERGY_NOISE_FLOOR:
            return 0.0
        
        if energy_total < 1e-6:
            return 0.0
        
        ratio = energy_seizure / energy_total
        energy_confidence = min(1.0, energy_seizure / (config.ENERGY_NOISE_FLOOR * 3))
        combined_confidence = ratio * 0.7 + energy_confidence * 0.3
        confidence = np.clip(combined_confidence * 1.5, 0.0, 1.0)
        
        return float(confidence)
    
    def reset(self):
        self.prev_gray = None
        self.motion_buffer.clear()
        self.timestamp_buffer.clear()
        self.smoothed_score = 0.0