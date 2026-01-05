# core/face_analyzer.py
import cv2
import numpy as np
import os
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque


class FaceAnalyzer:
    """
    Face analysis for seizure detection using MediaPipe Face Landmarker.
    
    Detects:
    - Head shake (rapid left-right/up-down movement)
    - Facial distortion (eye squeezing, grimacing)
    - Mouth open wide (tonic phase symptom)
    """
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
    
    # Key landmark indices (MediaPipe Face Mesh)
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_OUTER = 362
    RIGHT_EYE_INNER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    
    def __init__(self, buffer_size=10):
        self._ensure_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Movement history buffers
        self.nose_x_history = deque(maxlen=buffer_size)
        self.nose_y_history = deque(maxlen=buffer_size)
        self.ear_history = deque(maxlen=buffer_size)  # Eye Aspect Ratio
        self.mar_history = deque(maxlen=buffer_size)  # Mouth Aspect Ratio
        
        self.buffer_size = buffer_size
        
    def _ensure_model_exists(self):
        if os.path.exists(self.MODEL_PATH):
            return
            
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        print("Downloading face landmarker model...")
        response = requests.get(self.MODEL_URL)
        response.raise_for_status()
        with open(self.MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Face model downloaded!")
    
    def analyze(self, frame):
        """
        Analyze face in frame.
        
        Returns dict with:
        - head_shake_score (0.0-1.0)
        - facial_distortion (0.0-1.0)
        - mouth_open_wide (bool)
        """
        result = {
            'head_shake_score': 0.0,
            'facial_distortion': 0.0,
            'mouth_open_wide': False,
            'face_detected': False
        }
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection = self.landmarker.detect(mp_image)
            
            if not detection.face_landmarks or len(detection.face_landmarks) == 0:
                return result
            
            landmarks = detection.face_landmarks[0]
            result['face_detected'] = True
            
            # Extract key points
            nose = landmarks[self.NOSE_TIP]
            
            # Update nose position history
            self.nose_x_history.append(nose.x)
            self.nose_y_history.append(nose.y)
            
            # Calculate head shake score
            result['head_shake_score'] = self._calculate_head_shake()
            
            # Calculate facial metrics
            ear = self._calculate_eye_aspect_ratio(landmarks)
            mar = self._calculate_mouth_aspect_ratio(landmarks)
            
            self.ear_history.append(ear)
            self.mar_history.append(mar)
            
            # Facial distortion = variance in EAR + MAR
            result['facial_distortion'] = self._calculate_distortion()
            
            # Mouth open wide detection (stricter threshold)
            result['mouth_open_wide'] = mar > 0.6
            
        except Exception as e:
            pass
        
        return result
    
    def _calculate_head_shake(self):
        """Calculate head shake score from nose movement variance."""
        if len(self.nose_x_history) < 5:
            return 0.0
        
        x_data = np.array(self.nose_x_history)
        y_data = np.array(self.nose_y_history)
        
        # Calculate velocity (derivative)
        x_velocity = np.diff(x_data)
        y_velocity = np.diff(y_data)
        
        # Count direction changes (oscillations)
        x_sign_changes = np.sum(np.diff(np.signbit(x_velocity)) != 0)
        y_sign_changes = np.sum(np.diff(np.signbit(y_velocity)) != 0)
        
        # High variance + many direction changes = shaking
        x_var = np.std(x_data) * 100
        y_var = np.std(y_data) * 100
        
        oscillation_score = (x_sign_changes + y_sign_changes) / (2 * self.buffer_size)
        movement_score = min(1.0, (x_var + y_var) / 2)
        
        # Combined score - require stronger signal
        shake_score = oscillation_score * 0.5 + movement_score * 0.3
        
        return min(1.0, shake_score * 1.5)
    
    def _calculate_eye_aspect_ratio(self, landmarks):
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        # Left eye
        left_v = abs(landmarks[self.LEFT_EYE_TOP].y - landmarks[self.LEFT_EYE_BOTTOM].y)
        left_h = abs(landmarks[self.LEFT_EYE_OUTER].x - landmarks[self.LEFT_EYE_INNER].x)
        
        # Right eye
        right_v = abs(landmarks[self.RIGHT_EYE_TOP].y - landmarks[self.RIGHT_EYE_BOTTOM].y)
        right_h = abs(landmarks[self.RIGHT_EYE_OUTER].x - landmarks[self.RIGHT_EYE_INNER].x)
        
        left_ear = left_v / left_h if left_h > 0 else 0
        right_ear = right_v / right_h if right_h > 0 else 0
        
        return (left_ear + right_ear) / 2
    
    def _calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR)."""
        vertical = abs(landmarks[self.UPPER_LIP].y - landmarks[self.LOWER_LIP].y)
        horizontal = abs(landmarks[self.LEFT_MOUTH].x - landmarks[self.RIGHT_MOUTH].x)
        
        return vertical / horizontal if horizontal > 0 else 0
    
    def _calculate_distortion(self):
        """Calculate facial distortion from EAR/MAR variance."""
        if len(self.ear_history) < 5:
            return 0.0
        
        ear_var = np.std(self.ear_history) * 10
        mar_var = np.std(self.mar_history) * 5
        
        # High variance = distortion (grimacing, eye squeezing)
        distortion = min(1.0, ear_var + mar_var)
        
        return distortion
    
    def reset(self):
        """Reset all history buffers."""
        self.nose_x_history.clear()
        self.nose_y_history.clear()
        self.ear_history.clear()
        self.mar_history.clear()
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
