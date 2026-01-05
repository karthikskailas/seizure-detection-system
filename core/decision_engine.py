# core/decision_engine.py
import numpy as np
from collections import deque
import config


class DecisionEngine:
    """
    Multimodal Seizure Detection State Machine.
    
    Processes signals from:
    - Motion Analyzer (optical flow + FFT)
    - Pose Analyzer (body rigidity, fall detection)
    - Face Analyzer (head shake, facial distortion, mouth)
    
    Detects: Clonic, Tonic, Atonic seizures
    """
    
    STATE_IDLE = 0
    STATE_ANALYZING = 1
    STATE_ALERT = 2
    STATE_COOLDOWN = 3
    
    def __init__(self, fps=30):
        self.fps = fps
        self.state = self.STATE_IDLE
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.alert_count = 0
        
        # Motion history for drop detection
        self.motion_history = deque(maxlen=15)
        
        # Last detection info
        self.last_trigger_reason = None
        self.last_risk_score = 0.0
        
    def analyze_multimodal_frame(self, motion_data, pose_data, face_data):
        """
        Core multimodal analysis.
        
        Args:
            motion_data: dict with 'score', 'frequency'
            pose_data: dict with 'is_rigid', 'is_fallen', 'is_slumping', 'tremor'
            face_data: dict with 'head_shake_score', 'facial_distortion', 'mouth_open_wide'
        
        Returns: (risk_score, detection_type, reason)
        """
        risk_score = 0.0
        detection_type = None
        reason = None
        
        # Extract values with defaults
        motion_score = motion_data.get('score', 0.0)
        tremor_score = pose_data.get('tremor', 0.0)
        is_rigid = pose_data.get('is_rigid', False)
        is_fallen = pose_data.get('is_fallen', False)
        is_slumping = pose_data.get('is_slumping', is_fallen)
        
        head_shake = face_data.get('head_shake_score', 0.0)
        facial_distortion = face_data.get('facial_distortion', 0.0)
        mouth_open = face_data.get('mouth_open_wide', False)
        
        # Update motion history
        self.motion_history.append(motion_score)
        
        # === RULE 1: Violent Head Shaking (Clonic) ===
        # "Man in Car" scenario - head shaking even with restrained body
        if head_shake > 0.75 and facial_distortion > 0.6:
            risk_score = 0.95
            detection_type = "CLONIC"
            reason = "Violent head shaking + facial distortion"
            
        # === RULE 2: Gasping Tonic ===
        # Stiff body + open mouth = classic tonic onset
        elif is_rigid and mouth_open and motion_score < 0.3:
            risk_score = 0.9
            detection_type = "TONIC"
            reason = "Body rigidity + mouth open (gasping)"
            
        # === RULE 3: Silent Tonic ===
        # Low motion + rigid + slumping = loss of posture with muscle lock
        elif motion_score < 0.2 and is_rigid and is_slumping:
            risk_score = 0.85
            detection_type = "TONIC"
            reason = "Low motion + rigidity + posture loss"
            
        # === RULE 4: Drop Attack (Atonic) ===
        # Fall + sudden motion drop
        elif is_fallen and self._detect_motion_drop():
            risk_score = 0.95
            detection_type = "ATONIC"
            reason = "Fall + motion collapse"
            
        # === RULE 5: High Motion + Tremor (General Clonic) ===
        elif motion_score > 0.6 and tremor_score > 0.5:
            risk_score = 0.85
            detection_type = "CLONIC"
            reason = "High motion + rhythmic tremor"
            
        # === RULE 6: Head Shake Only (Moderate) ===
        elif head_shake > 0.65 and motion_score > 0.4:
            risk_score = 0.7
            detection_type = "POSSIBLE"
            reason = "Head shaking + elevated motion"
            
        # === RULE 7: Fall with Motion ===
        elif is_fallen and motion_score > 0.3:
            risk_score = 0.6
            detection_type = "FALL"
            reason = "Fall detected with motion"
            
        # === FALSE POSITIVE FILTER ===
        # High motion but no tremor, no head shake, not rigid = normal activity
        elif motion_score > 0.5 and tremor_score < 0.1 and head_shake < 0.2 and not is_rigid:
            risk_score = 0.1
            detection_type = None
            reason = "Normal movement (filtered)"
        
        return risk_score, detection_type, reason
    
    def _detect_motion_drop(self):
        """Detect sudden motion drop (atonic pattern)."""
        if len(self.motion_history) < 10:
            return False
        
        motion = np.array(self.motion_history)
        recent = motion[-5:]
        previous = motion[-10:-5]
        
        # Motion was elevated, then dropped suddenly
        if np.mean(previous) > 0.3 and np.mean(recent) < 0.1:
            return True
        
        return False
    
    def process(self, motion_data, pose_data, face_data):
        """
        Main processing loop - call once per frame.
        
        Returns: (is_alert, debug_data)
        """
        # Multimodal analysis
        risk_score, detection_type, reason = self.analyze_multimodal_frame(
            motion_data, pose_data, face_data
        )
        
        self.last_risk_score = risk_score
        
        # Handle display timer (post-alert)
        if self.display_timer > 0:
            self.display_timer -= 1
            return False, {
                "status": f"🚨 {self.last_trigger_reason} ({self.display_timer // self.fps}s)",
                "risk": risk_score,
                "counter": 0,
                "type": self.last_trigger_reason,
                "reason": reason
            }
        
        # Handle cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False, {
                "status": "MONITORING",
                "risk": risk_score,
                "counter": 0,
                "type": None,
                "reason": None
            }
        
        # Determine trigger threshold and duration based on detection type
        threshold = self._get_threshold(detection_type)
        duration_frames = self._get_duration_frames(detection_type)
        
        # Check if this frame is positive
        is_positive = risk_score >= threshold
        
        if is_positive:
            self.consecutive_frames += 1
            self.state = self.STATE_ANALYZING
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)
            if self.consecutive_frames == 0:
                self.state = self.STATE_IDLE
        
        # Check for alert trigger
        if self.consecutive_frames >= duration_frames:
            self.state = self.STATE_ALERT
            self.alert_count += 1
            self.last_trigger_reason = detection_type or "SEIZURE"
            self.display_timer = int(3 * self.fps)
            self.cooldown_timer = config.COOLDOWN_FRAMES
            self.consecutive_frames = 0
            
            return True, {
                "status": f"🚨 {self.last_trigger_reason} DETECTED",
                "risk": risk_score,
                "counter": duration_frames,
                "type": self.last_trigger_reason,
                "reason": reason,
                "alert_num": self.alert_count
            }
        
        # Status reporting
        status = self._get_status(risk_score, detection_type)
        
        return False, {
            "status": status,
            "risk": risk_score,
            "counter": self.consecutive_frames,
            "type": detection_type,
            "reason": reason
        }
    
    def _get_threshold(self, detection_type):
        """Get risk threshold based on detection type."""
        thresholds = {
            "CLONIC": 0.7,
            "TONIC": 0.6,
            "ATONIC": 0.5,
            "FALL": 0.5,
            "POSSIBLE": 0.65,
            None: 0.7
        }
        return thresholds.get(detection_type, 0.7)
    
    def _get_duration_frames(self, detection_type):
        """Get required consecutive frames for detection."""
        durations = {
            "ATONIC": max(5, int(0.25 * self.fps)),   # 0.25s
            "TONIC": max(10, int(0.4 * self.fps)),    # 0.4s
            "CLONIC": max(15, int(0.5 * self.fps)),   # 0.5s
            "FALL": max(8, int(0.3 * self.fps)),      # 0.3s
            "POSSIBLE": max(20, int(0.7 * self.fps)), # 0.7s - needs more confirmation
            None: max(15, int(0.5 * self.fps))
        }
        return durations.get(detection_type, 15)
    
    def _get_status(self, risk_score, detection_type):
        """Get display status string."""
        if self.consecutive_frames > 3:
            type_str = detection_type or "MOTION"
            return f"ANALYZING... ({type_str})"
        
        if risk_score > 0.4:
            return "ELEVATED"
        
        return "MONITORING"
    
    def reset(self):
        """Reset engine state."""
        self.state = self.STATE_IDLE
        self.consecutive_frames = 0
        self.cooldown_timer = 0
        self.display_timer = 0
        self.motion_history.clear()