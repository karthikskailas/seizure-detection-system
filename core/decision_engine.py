# core/decision_engine.py
"""
Decision Engine (The Brain)
---------------------------
Combines motion analysis and fall detection signals to make final seizure decision.

Key Features:
- Temporal consistency: Requires sustained detection (not single-frame spikes)
- Display timer: Shows "DETECTED" for 5 seconds after detection
- Cooldown: Prevents alert spam after display period ends
- Combined signals: Motion + Fall = stronger evidence
"""

import config

# 5 seconds display timer at 30fps
DISPLAY_FRAMES = 150


class DecisionEngine:
    def __init__(self):
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0  # Shows "DETECTED" status
        self.is_alerting = False

    def process(self, motion_confidence: float, fall_detected: bool) -> tuple:
        """
        Main decision logic.
        
        Inputs:
        - motion_confidence (float): 0.0-1.0 from MotionAnalyzer
        - fall_detected (bool): From PoseAnalyzer
        
        Returns:
        - is_alert (bool): Should we trigger the alarm?
        - metadata (dict): Debug data for UI and logging
        """
        
        # 1. Display Timer (Show "DETECTED" for 5 seconds)
        # After a seizure is detected, keep showing the status
        if self.display_timer > 0:
            self.display_timer -= 1
            remaining_secs = self.display_timer // config.FPS_ASSUMED
            return False, {
                "status": f"🚨 DETECTED (reset in {remaining_secs}s)", 
                "risk": motion_confidence,
                "counter": 0
            }
        
        # 2. Cooldown Check (after display timer ends)
        # Prevents immediate re-trigger
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False, {
                "status": "MONITORING", 
                "risk": motion_confidence,
                "counter": 0
            }

        # 3. Dynamic Thresholding
        # If a fall was detected, we're more paranoid about subsequent shaking
        current_threshold = config.MOTION_THRESHOLD
        if fall_detected:
            current_threshold = 0.35  # Lower threshold after fall
            
        # 4. Frame Evaluation
        detected_this_frame = False
        
        # Rule A: High Intensity Rhythmic Shaking
        if motion_confidence >= current_threshold:
            detected_this_frame = True
            
        # Rule B: Fall + Moderate Shaking
        if fall_detected and motion_confidence >= 0.25:
            detected_this_frame = True

        # 5. Temporal Consistency (Debouncing)
        if detected_this_frame:
            self.consecutive_frames += 1
        else:
            # Decay counter quickly to prevent buildup from normal motion
            self.consecutive_frames = max(0, self.consecutive_frames - 2)

        # 6. Alert Trigger
        if self.consecutive_frames >= config.DURATION_THRESHOLD:
            self.is_alerting = True
            self.display_timer = DISPLAY_FRAMES  # 5 second display
            self.cooldown_timer = config.COOLDOWN_FRAMES  # 3 second cooldown after display
            self.consecutive_frames = 0
            return True, {
                "status": "🚨 DETECTED (reset in 5s)", 
                "risk": motion_confidence,
                "counter": config.DURATION_THRESHOLD
            }

        # 7. Status Reporting
        status = "MONITORING"
        if self.consecutive_frames > 5:
            status = "ANALYZING..."
        if fall_detected:
            status = "FALL WARNING"
            
        return False, {
            "status": status, 
            "risk": motion_confidence, 
            "counter": self.consecutive_frames
        }
    
    def reset(self):
        """Reset decision state."""
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.is_alerting = False