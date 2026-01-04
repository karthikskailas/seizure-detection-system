# core/decision_engine.py
import config

class DecisionEngine:
    def __init__(self):
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.is_alerting = False

    def process(self, motion_confidence, fall_detected):
        """
        Inputs:
        - motion_confidence (float): From MotionAnalyzer
        - fall_detected (bool): From PoseAnalyzer (Developer 2)
        
        Returns:
        - is_alert (bool): Trigger the alarm?
        - metadata (dict): Debug data for the UI
        """
        
        # 1. Cooldown Check
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False, {"status": "COOLDOWN", "risk": 0}

        # 2. Dynamic Thresholding
        # If they fell, we are more paranoid about shaking
        current_threshold = config.MOTION_THRESHOLD
        if fall_detected:
            current_threshold = 0.3 # Lower threshold
            
        # 3. Frame Evaluation
        detected_this_frame = False
        
        # Rule A: High Intensity Shaking
        if motion_confidence > current_threshold:
            detected_this_frame = True
            
        # Rule B: Fall + Moderate Shaking
        if fall_detected and motion_confidence > 0.2:
            detected_this_frame = True

        # 4. Temporal Consistency (Debouncing)
        if detected_this_frame:
            self.consecutive_frames += 1
        else:
            # Decay counter slowly
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

        # 5. Alert Trigger
        if self.consecutive_frames >= config.DURATION_THRESHOLD:
            self.is_alerting = True
            self.cooldown_timer = config.COOLDOWN_FRAMES
            self.consecutive_frames = 0
            return True, {"status": "SEIZURE DETECTED", "risk": motion_confidence}

        return False, {
            "status": "MONITORING", 
            "risk": motion_confidence, 
            "counter": self.consecutive_frames
        }