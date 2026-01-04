# core/pose_analyzer.py
import cv2
import numpy as np
import config
import os
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseAnalyzer:
    # Model download URL and local path
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker.task")
    
    # Pose landmark indices
    NOSE = 0
    
    def __init__(self):
        self._ensure_model_exists()
        
        # Create PoseLandmarker using the new Tasks API
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.prev_y = None
        
    def _ensure_model_exists(self):
        """Download the pose landmarker model if it doesn't exist."""
        if os.path.exists(self.MODEL_PATH):
            return
            
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        
        print("Downloading pose landmarker model...")
        response = requests.get(self.MODEL_URL)
        response.raise_for_status()
        
        with open(self.MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
        
    def detect_fall(self, frame):
        """
        Returns: (bool is_fall, PoseLandmarkerResult or None)
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect pose landmarks
        results = self.landmarker.detect(mp_image)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            self.prev_y = None
            return False, None

        # Get Nose Y-coordinate (normalized 0.0 to 1.0)
        landmarks = results.pose_landmarks[0]
        nose = landmarks[self.NOSE]
        current_y = nose.y
        
        is_fall = False
        
        # Check Velocity (Change in Y)
        if self.prev_y is not None:
            velocity = current_y - self.prev_y
            
            # If velocity is positive (moving down) and exceeds threshold
            # Tuning: 0.06 is a moderate speed drop per frame at 30fps
            if velocity > 0.06: 
                is_fall = True
        
        self.prev_y = current_y
        return is_fall, results
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()