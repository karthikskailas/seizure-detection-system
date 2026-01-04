# core/decision_engine.py
"""
Decision Engine (The Brain) - Improved Signal Processing
---------------------------------------------------------
Combines motion analysis and fall detection signals to make final seizure decision.

Improvements:
- Baseline calibration: Adapts to individual patient's normal motion level
- Adaptive thresholds: Adjusts sensitivity based on baseline
- Slower frame counter decay: More robust temporal consistency
- Combined signal weighting: Better fusion of motion + fall signals
"""

import config

# 3 seconds display timer at 30fps
DISPLAY_FRAMES = 90


class DecisionEngine:
    def __init__(self):
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0  # Shows "DETECTED" status
        self.is_alerting = False
        
        # Baseline calibration state
        self.baseline_motion = 0.0
        self.baseline_samples = 0
        self.calibration_frames = config.FPS_ASSUMED * config.BASELINE_CALIBRATION_SECONDS
        self.is_calibrated = False
        
        # Motion history for adaptive thresholding
        self.recent_motion = []
        self.motion_history_size = 30  # 1 second at 30fps
        
    def _update_baseline(self, motion_confidence):
        """
        Update baseline during calibration period.
        
        The baseline represents the patient's normal motion level when not seizing.
        This allows the system to adapt to different patients and setups.
        """
        if self.baseline_samples < self.calibration_frames:
            # Incremental mean calculation
            self.baseline_motion = (
                (self.baseline_motion * self.baseline_samples + motion_confidence) / 
                (self.baseline_samples + 1)
            )
            self.baseline_samples += 1
            
            if self.baseline_samples >= self.calibration_frames:
                self.is_calibrated = True
                print(f"[DecisionEngine] Calibration complete. Baseline motion: {self.baseline_motion:.3f}")
    
    def _get_adaptive_threshold(self, fall_detected):
        """
        Calculate adaptive threshold based on baseline and conditions.
        
        The threshold adjusts based on:
        1. Calibrated baseline (if available)
        2. Whether a fall was detected (lower threshold after fall)
        3. Fixed minimum to prevent over-sensitivity
        """
        base_threshold = config.MOTION_THRESHOLD
        
        # If calibrated, adjust threshold relative to baseline
        if self.is_calibrated and self.baseline_motion > 0:
            # Threshold should be at least 2x baseline for significance
            adaptive_threshold = max(base_threshold, self.baseline_motion * 2.5)
            # But cap it to prevent requiring impossibly high motion
            adaptive_threshold = min(adaptive_threshold, 0.85)
        else:
            adaptive_threshold = base_threshold
        
        # Lower threshold after fall detected (more paranoid)
        if fall_detected:
            adaptive_threshold = min(adaptive_threshold, 0.35)
        
        return adaptive_threshold
    
    def _update_motion_history(self, motion_confidence):
        """Track recent motion for context-aware decisions."""
        self.recent_motion.append(motion_confidence)
        if len(self.recent_motion) > self.motion_history_size:
            self.recent_motion.pop(0)
    
    def _get_motion_trend(self):
        """
        Analyze recent motion trend.
        
        Returns:
        - 'rising': Motion is increasing (potential seizure onset)
        - 'falling': Motion is decreasing
        - 'stable': Motion is relatively constant
        """
        if len(self.recent_motion) < 10:
            return 'stable'
        
        first_half = sum(self.recent_motion[:len(self.recent_motion)//2])
        second_half = sum(self.recent_motion[len(self.recent_motion)//2:])
        
        ratio = second_half / max(first_half, 0.01)
        
        if ratio > 1.3:
            return 'rising'
        elif ratio < 0.7:
            return 'falling'
        else:
            return 'stable'

    def process(self, motion_confidence: float, fall_detected: bool) -> tuple:
        """
        Main decision logic with improved signal processing.
        
        Inputs:
        - motion_confidence (float): 0.0-1.0 from MotionAnalyzer
        - fall_detected (bool): From PoseAnalyzer
        
        Returns:
        - is_alert (bool): Should we trigger the alarm?
        - metadata (dict): Debug data for UI and logging
        """
        
        # Update baseline during calibration
        self._update_baseline(motion_confidence)
        
        # Track motion history
        self._update_motion_history(motion_confidence)
        
        # 1. Display Timer (Show "DETECTED" for 3 seconds)
        if self.display_timer > 0:
            self.display_timer -= 1
            remaining_secs = self.display_timer // config.FPS_ASSUMED
            return False, {
                "status": f"🚨 DETECTED (reset in {remaining_secs}s)", 
                "risk": motion_confidence,
                "counter": 0,
                "calibrated": self.is_calibrated
            }
        
        # 2. Cooldown Check (after display timer ends)
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False, {
                "status": "MONITORING", 
                "risk": motion_confidence,
                "counter": 0,
                "calibrated": self.is_calibrated
            }

        # 3. Get Adaptive Threshold
        current_threshold = self._get_adaptive_threshold(fall_detected)
        
        # 4. Frame Evaluation with STRICT logic
        detected_this_frame = False
        detection_reason = None
        
        # Rule A: High Intensity Rhythmic Shaking (PRIMARY RULE)
        # Only trigger if motion is significantly above threshold
        if motion_confidence >= current_threshold:
            detected_this_frame = True
            detection_reason = "high_motion"
            
        # Rule B: Fall + Strong Shaking (both signals must be strong)
        # Raised from 0.25 to 0.50 - needs real shaking, not just fidgeting
        if fall_detected and motion_confidence >= 0.50:
            detected_this_frame = True
            detection_reason = "fall_plus_motion"
        
        # REMOVED: Rising trend rule - caused too many false positives
        # Rising motion without sustained high levels is usually just normal activity

        # 5. Temporal Consistency (Improved Debouncing)
        if detected_this_frame:
            self.consecutive_frames += 1
        else:
            # Slower decay - require sustained absence of detection
            # This prevents brief normal breaks from resetting the counter
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

        # 6. Alert Trigger
        if self.consecutive_frames >= config.DURATION_THRESHOLD:
            self.is_alerting = True
            self.display_timer = DISPLAY_FRAMES
            self.cooldown_timer = config.COOLDOWN_FRAMES
            self.consecutive_frames = 0
            return True, {
                "status": "🚨 DETECTED (reset in 3s)", 
                "risk": motion_confidence,
                "counter": config.DURATION_THRESHOLD,
                "reason": detection_reason,
                "calibrated": self.is_calibrated
            }

        # 7. Status Reporting with more detail
        status = "MONITORING"
        if not self.is_calibrated:
            remaining = self.calibration_frames - self.baseline_samples
            status = f"CALIBRATING ({remaining // config.FPS_ASSUMED}s)"
        elif self.consecutive_frames > 5:
            status = "ANALYZING..."
        elif fall_detected:
            status = "⚠️ FALL WARNING"
            
        return False, {
            "status": status, 
            "risk": motion_confidence, 
            "counter": self.consecutive_frames,
            "threshold": current_threshold,
            "calibrated": self.is_calibrated
        }
    
    def reset(self):
        """Reset decision state."""
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.is_alerting = False
        self.recent_motion.clear()
        
    def reset_calibration(self):
        """Reset calibration to re-learn baseline."""
        self.baseline_motion = 0.0
        self.baseline_samples = 0
        self.is_calibrated = False