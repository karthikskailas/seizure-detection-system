import numpy as np
from collections import deque
import config


class PoseVelocityAnalyzer:
    """Analyzes body part velocities for seizure pattern detection."""
    
    # MediaPipe landmark indices
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
    }
    
    # Body part groups for pattern analysis
    UPPER_BODY = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
    LOWER_BODY = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    LEFT_SIDE = ['left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee', 'left_ankle']
    RIGHT_SIDE = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']
    
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.prev_positions = {}
        self.velocity_history = {name: deque(maxlen=buffer_size) for name in self.LANDMARKS}
        self.magnitude_history = deque(maxlen=buffer_size)
        
    def update(self, pose_landmarks):
        """Update with new pose landmarks and calculate velocities."""
        if not pose_landmarks or len(pose_landmarks) == 0:
            return None
            
        landmarks = pose_landmarks[0]
        velocities = {}
        total_magnitude = 0
        valid_count = 0
        
        for name, idx in self.LANDMARKS.items():
            if idx >= len(landmarks):
                continue
                
            lm = landmarks[idx]
            if lm.visibility < 0.5:
                continue
            
            current_pos = np.array([lm.x, lm.y])
            
            if name in self.prev_positions:
                velocity = current_pos - self.prev_positions[name]
                magnitude = np.linalg.norm(velocity)
                velocities[name] = {
                    'velocity': velocity,
                    'magnitude': magnitude,
                    'direction': np.arctan2(velocity[1], velocity[0])
                }
                self.velocity_history[name].append(magnitude)
                total_magnitude += magnitude
                valid_count += 1
            
            self.prev_positions[name] = current_pos
        
        if valid_count > 0:
            self.magnitude_history.append(total_magnitude / valid_count)
        
        return velocities
    
    def get_body_pattern(self):
        """Analyze current movement pattern across body parts."""
        pattern = {
            'upper_body_score': self._calculate_group_activity(self.UPPER_BODY),
            'lower_body_score': self._calculate_group_activity(self.LOWER_BODY),
            'left_side_score': self._calculate_group_activity(self.LEFT_SIDE),
            'right_side_score': self._calculate_group_activity(self.RIGHT_SIDE),
            'symmetry': self._calculate_symmetry(),
            'overall_activity': self._get_overall_activity(),
        }
        return pattern
    
    def _calculate_group_activity(self, group):
        """Calculate average activity for a body part group."""
        activities = []
        for name in group:
            if len(self.velocity_history[name]) > 0:
                activities.append(np.mean(self.velocity_history[name]))
        return np.mean(activities) if activities else 0.0
    
    def _calculate_symmetry(self):
        """Calculate left-right symmetry (1.0 = symmetric, 0.0 = asymmetric)."""
        left_score = self._calculate_group_activity(self.LEFT_SIDE)
        right_score = self._calculate_group_activity(self.RIGHT_SIDE)
        
        if left_score == 0 and right_score == 0:
            return 1.0
        
        max_score = max(left_score, right_score)
        min_score = min(left_score, right_score)
        
        return min_score / max_score if max_score > 0 else 1.0
    
    def _get_overall_activity(self):
        """Get average overall body activity."""
        if len(self.magnitude_history) == 0:
            return 0.0
        return np.mean(self.magnitude_history)
    
    def detect_rigidity(self, threshold=0.01):
        """Detect body rigidity (very low movement - tonic phase)."""
        activity = self._get_overall_activity()
        return activity < threshold and len(self.magnitude_history) >= 10
    
    def detect_tremor(self, min_freq=2.0, max_freq=7.0):
        """Detect rhythmic tremor patterns."""
        if len(self.magnitude_history) < self.buffer_size:
            return 0.0
        
        data = np.array(self.magnitude_history)
        data = data - np.mean(data)
        
        if np.std(data) < 0.001:
            return 0.0
        
        # Simple frequency detection via zero crossings
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        if len(zero_crossings) < 4:
            return 0.0
        
        # Estimate frequency from zero crossings
        avg_period = np.mean(np.diff(zero_crossings)) * 2
        fps = config.FPS_ASSUMED
        estimated_freq = fps / avg_period if avg_period > 0 else 0
        
        # Check if in seizure frequency range
        if min_freq <= estimated_freq <= max_freq:
            return min(1.0, np.std(data) * 10)
        
        return 0.0
    
    def get_seizure_confidence(self):
        """Calculate overall seizure likelihood based on body patterns."""
        pattern = self.get_body_pattern()
        
        # Tremor detection
        tremor_score = self.detect_tremor()
        
        # High activity + rhythmic = likely seizure
        activity_score = min(1.0, pattern['overall_activity'] * 5)
        
        # Bilateral symmetry often seen in generalized seizures
        symmetry_bonus = pattern['symmetry'] * 0.2 if pattern['symmetry'] > 0.7 else 0
        
        # Combined score
        confidence = (tremor_score * 0.5 + activity_score * 0.4 + symmetry_bonus * 0.1)
        
        return min(1.0, confidence)
    
    def reset(self):
        """Reset all tracking state."""
        self.prev_positions.clear()
        for name in self.LANDMARKS:
            self.velocity_history[name].clear()
        self.magnitude_history.clear()
