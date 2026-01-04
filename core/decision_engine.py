# core/decision_engine.py
import config

DISPLAY_FRAMES = 90


class DecisionEngine:
    def __init__(self):
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.is_alerting = False
        
        self.baseline_motion = 0.0
        self.baseline_samples = 0
        self.calibration_frames = config.FPS_ASSUMED * config.BASELINE_CALIBRATION_SECONDS
        self.is_calibrated = (config.BASELINE_CALIBRATION_SECONDS == 0)
        
        self.recent_motion = []
        self.motion_history_size = 30
        
    def _update_baseline(self, motion_confidence):
        if self.is_calibrated:
            return
            
        if self.baseline_samples < self.calibration_frames:
            self.baseline_motion = (
                (self.baseline_motion * self.baseline_samples + motion_confidence) / 
                (self.baseline_samples + 1)
            )
            self.baseline_samples += 1
            
            if self.baseline_samples >= self.calibration_frames:
                self.is_calibrated = True
                print(f"[DecisionEngine] Calibration complete. Baseline motion: {self.baseline_motion:.3f}")
    
    def _get_adaptive_threshold(self, fall_detected):
        base_threshold = config.MOTION_THRESHOLD
        
        if self.is_calibrated and self.baseline_motion > 0:
            adaptive_threshold = max(base_threshold, self.baseline_motion * 2.5)
            adaptive_threshold = min(adaptive_threshold, 0.85)
        else:
            adaptive_threshold = base_threshold
        
        if fall_detected:
            adaptive_threshold = min(adaptive_threshold, 0.35)
        
        return adaptive_threshold
    
    def _update_motion_history(self, motion_confidence):
        self.recent_motion.append(motion_confidence)
        if len(self.recent_motion) > self.motion_history_size:
            self.recent_motion.pop(0)

    def process(self, motion_confidence: float, fall_detected: bool) -> tuple:
        self._update_baseline(motion_confidence)
        self._update_motion_history(motion_confidence)
        
        if self.display_timer > 0:
            self.display_timer -= 1
            remaining_secs = self.display_timer // config.FPS_ASSUMED
            return False, {
                "status": f"🚨 DETECTED (reset in {remaining_secs}s)", 
                "risk": motion_confidence,
                "counter": 0,
                "calibrated": self.is_calibrated
            }
        
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False, {
                "status": "MONITORING", 
                "risk": motion_confidence,
                "counter": 0,
                "calibrated": self.is_calibrated
            }

        current_threshold = self._get_adaptive_threshold(fall_detected)
        
        detected_this_frame = False
        detection_reason = None
        
        if motion_confidence >= current_threshold:
            detected_this_frame = True
            detection_reason = "high_motion"
            
        if fall_detected and motion_confidence >= 0.50:
            detected_this_frame = True
            detection_reason = "fall_plus_motion"

        if detected_this_frame:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

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
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.is_alerting = False
        self.recent_motion.clear()
        
    def reset_calibration(self):
        self.baseline_motion = 0.0
        self.baseline_samples = 0
        self.is_calibrated = False