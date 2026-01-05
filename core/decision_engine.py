# core/decision_engine.py
import numpy as np
from collections import deque
import config


class DecisionEngine:
    STATE_IDLE = 0
    STATE_ANALYZING = 1
    STATE_ALERT = 2
    
    def __init__(self, fps=30):
        self.fps = fps
        self.state = self.STATE_IDLE
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.alert_count = 0
        self.motion_history = deque(maxlen=30)
        self.risk_history = deque(maxlen=20)
        self.high_risk_streak = 0
        self.last_trigger_reason = None
        
    def analyze_multimodal_frame(self, motion_data, pose_data, face_data):
        motion_score = motion_data.get('score', 0.0)
        tremor_score = pose_data.get('tremor', 0.0)
        is_rigid = pose_data.get('is_rigid', False)
        is_fallen = pose_data.get('is_fallen', False)
        head_shake = face_data.get('head_shake_score', 0.0)
        facial_distortion = face_data.get('facial_distortion', 0.0)
        mouth_open = face_data.get('mouth_open_wide', False)
        
        self.motion_history.append(motion_score)
        
        risk_score = 0.0
        detection_type = None
        reason = None
        
        # RULE 1: Violent head shaking + facial distortion (very specific)
        if head_shake > 0.7 and facial_distortion > 0.6 and motion_score > 0.3:
            risk_score = 0.95
            detection_type = "CLONIC"
            reason = "Head shake + face distortion"
            
        # RULE 2: Tonic - rigid + mouth open + low motion
        elif is_rigid and mouth_open and motion_score < 0.3:
            risk_score = 0.9
            detection_type = "TONIC"
            reason = "Rigid + mouth open"
            
        # RULE 3: Atonic - fall + sudden motion drop
        elif is_fallen and self._detect_motion_drop():
            risk_score = 0.95
            detection_type = "ATONIC"
            reason = "Fall + motion drop"
            
        # RULE 4: Clonic - high motion + high tremor (both required)
        elif motion_score > 0.6 and tremor_score > 0.5:
            risk_score = 0.8
            detection_type = "CLONIC"
            reason = "Motion + tremor"
            
        # RULE 5: Silent tonic
        elif motion_score < 0.2 and is_rigid and is_fallen:
            risk_score = 0.8
            detection_type = "TONIC"
            reason = "Low motion + rigid + fallen"
        
        # Everything else = low risk (no alert)
        else:
            risk_score = min(0.3, motion_score * 0.4)
        
        return risk_score, detection_type, reason
    
    def _detect_motion_drop(self):
        if len(self.motion_history) < 15:
            return False
        motion = np.array(self.motion_history)
        recent = motion[-5:]
        previous = motion[-15:-5]
        return np.mean(previous) > 0.4 and np.mean(recent) < 0.1
    
    def process(self, motion_data, pose_data, face_data):
        risk_score, detection_type, reason = self.analyze_multimodal_frame(
            motion_data, pose_data, face_data
        )
        
        self.risk_history.append(risk_score)
        
        # Display timer
        if self.display_timer > 0:
            self.display_timer -= 1
            return False, {
                "status": f"🚨 {self.last_trigger_reason}",
                "risk": risk_score, "counter": 0,
                "type": self.last_trigger_reason, "reason": reason
            }
        
        # Cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False, {
                "status": "MONITORING", "risk": risk_score,
                "counter": 0, "type": None, "reason": None
            }
        
        # Track high risk streak
        if risk_score >= 0.8:
            self.high_risk_streak += 1
        else:
            self.high_risk_streak = max(0, self.high_risk_streak - 2)
        
        # Counter logic
        if risk_score >= 0.8 and detection_type:
            self.consecutive_frames += 1
        elif risk_score >= 0.5:
            self.consecutive_frames += 0.3
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)
        
        # TRIGGER: Need sustained high risk
        # Require 15+ consecutive high-risk frames (~0.5 seconds at 30fps)
        trigger_threshold = 15
        
        # OR immediate trigger if 10+ frames of 80%+ risk in last 20 frames
        recent_high = sum(1 for r in self.risk_history if r >= 0.8)
        
        should_trigger = (
            (self.consecutive_frames >= trigger_threshold and detection_type) or
            (recent_high >= 12 and self.high_risk_streak >= 8)
        )
        
        if should_trigger:
            self.alert_count += 1
            self.last_trigger_reason = detection_type or "SEIZURE"
            self.display_timer = int(3 * self.fps)
            self.cooldown_timer = config.COOLDOWN_FRAMES
            self.consecutive_frames = 0
            self.high_risk_streak = 0
            
            return True, {
                "status": f"🚨 {self.last_trigger_reason} DETECTED",
                "risk": risk_score, "counter": 0,
                "type": self.last_trigger_reason, "reason": reason,
                "alert_num": self.alert_count
            }
        
        status = "ANALYZING" if self.consecutive_frames > 5 else "MONITORING"
        if risk_score > 0.5:
            status = f"ELEVATED ({detection_type or 'MOTION'})"
        
        return False, {
            "status": status, "risk": risk_score,
            "counter": int(self.consecutive_frames),
            "type": detection_type, "reason": reason
        }
    
    def reset(self):
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.motion_history.clear()
        self.risk_history.clear()
        self.high_risk_streak = 0