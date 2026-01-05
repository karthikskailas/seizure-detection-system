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
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
    
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
        
        self.nose_x_history = deque(maxlen=buffer_size)
        self.nose_y_history = deque(maxlen=buffer_size)
        self.ear_history = deque(maxlen=buffer_size)
        self.mar_history = deque(maxlen=buffer_size)
        
        self.buffer_size = buffer_size
        
    def _ensure_model_exists(self):
        if os.path.exists(self.MODEL_PATH):
            return
            
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        try:
            response = requests.get(self.MODEL_URL)
            response.raise_for_status()
            with open(self.MODEL_PATH, 'wb') as f:
                f.write(response.content)
        except:
            pass
    
    def analyze(self, frame):
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
            
            nose = landmarks[self.NOSE_TIP]
            self.nose_x_history.append(nose.x)
            self.nose_y_history.append(nose.y)
            
            result['head_shake_score'] = self._calculate_head_shake()
            
            ear = self._calculate_eye_aspect_ratio(landmarks)
            mar = self._calculate_mouth_aspect_ratio(landmarks)
            
            self.ear_history.append(ear)
            self.mar_history.append(mar)
            
            result['facial_distortion'] = self._calculate_distortion()
            result['mouth_open_wide'] = mar > 0.6
            
        except Exception:
            pass
        
        return result
    
    def _calculate_head_shake(self):
        if len(self.nose_x_history) < 5:
            return 0.0
        
        x_data = np.array(self.nose_x_history)
        y_data = np.array(self.nose_y_history)
        
        x_velocity = np.diff(x_data)
        y_velocity = np.diff(y_data)
        
        x_sign_changes = np.sum(np.diff(np.signbit(x_velocity)) != 0)
        y_sign_changes = np.sum(np.diff(np.signbit(y_velocity)) != 0)
        
        x_var = np.std(x_data) * 100
        y_var = np.std(y_data) * 100
        
        oscillation_score = (x_sign_changes + y_sign_changes) / (2 * self.buffer_size)
        movement_score = min(1.0, (x_var + y_var) / 2)
        
        shake_score = oscillation_score * 0.5 + movement_score * 0.3
        
        return min(1.0, shake_score * 1.5)
    
    def _calculate_eye_aspect_ratio(self, landmarks):
        left_v = abs(landmarks[self.LEFT_EYE_TOP].y - landmarks[self.LEFT_EYE_BOTTOM].y)
        left_h = abs(landmarks[self.LEFT_EYE_OUTER].x - landmarks[self.LEFT_EYE_INNER].x)
        
        right_v = abs(landmarks[self.RIGHT_EYE_TOP].y - landmarks[self.RIGHT_EYE_BOTTOM].y)
        right_h = abs(landmarks[self.RIGHT_EYE_OUTER].x - landmarks[self.RIGHT_EYE_INNER].x)
        
        left_ear = left_v / left_h if left_h > 0 else 0
        right_ear = right_v / right_h if right_h > 0 else 0
        
        return (left_ear + right_ear) / 2
    
    def _calculate_mouth_aspect_ratio(self, landmarks):
        vertical = abs(landmarks[self.UPPER_LIP].y - landmarks[self.LOWER_LIP].y)
        horizontal = abs(landmarks[self.LEFT_MOUTH].x - landmarks[self.RIGHT_MOUTH].x)
        
        return vertical / horizontal if horizontal > 0 else 0
    
    def _calculate_distortion(self):
        if len(self.ear_history) < 5:
            return 0.0
        
        ear_var = np.std(self.ear_history) * 10
        mar_var = np.std(self.mar_history) * 5
        
        distortion = min(1.0, ear_var + mar_var)
        
        return distortion
    
    def reset(self):
        self.nose_x_history.clear()
        self.nose_y_history.clear()
        self.ear_history.clear()
        self.mar_history.clear()
    
    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
