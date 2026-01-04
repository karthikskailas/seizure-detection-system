# core/pose_analyzer.py
import cv2
import numpy as np
import config
import os
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque


class PoseAnalyzer:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker.task")
    
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    TRACKED_POINTS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    
    def __init__(self):
        self._ensure_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        self.prev_y_positions = {}
        self.velocity_buffer = deque(maxlen=config.VELOCITY_SMOOTH_FRAMES)
        self.fall_frame_counter = 0
        self.consecutive_fall_frames = 0
        
    def _ensure_model_exists(self):
        if os.path.exists(self.MODEL_PATH):
            return
            
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        
        print("Downloading pose landmarker model...")
        response = requests.get(self.MODEL_URL)
        response.raise_for_status()
        
        with open(self.MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    
    def _calculate_body_velocity(self, landmarks):
        velocities = []
        weights = []
        
        for point_idx in self.TRACKED_POINTS:
            landmark = landmarks[point_idx]
            
            if landmark.visibility < 0.5:
                continue
            
            current_y = landmark.y
            
            if point_idx in self.prev_y_positions:
                prev_y = self.prev_y_positions[point_idx]
                velocity = current_y - prev_y
                velocities.append(velocity)
                weights.append(landmark.visibility)
            
            self.prev_y_positions[point_idx] = current_y
        
        if not velocities:
            return 0.0, 0
        
        weights = np.array(weights)
        velocities = np.array(velocities)
        weighted_velocity = np.sum(velocities * weights) / np.sum(weights)
        
        return weighted_velocity, len(velocities)
    
    def _get_smoothed_velocity(self, velocity):
        self.velocity_buffer.append(velocity)
        
        if len(self.velocity_buffer) < 2:
            return velocity
        
        return np.median(self.velocity_buffer)
        
    def detect_fall(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            self.prev_y_positions.clear()
            self.velocity_buffer.clear()
            self.consecutive_fall_frames = 0
            return False, None

        landmarks = results.pose_landmarks[0]
        raw_velocity, num_points = self._calculate_body_velocity(landmarks)
        
        if num_points < 3:
            self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 1)
            return False, results
        
        smoothed_velocity = self._get_smoothed_velocity(raw_velocity)
        is_falling_this_frame = smoothed_velocity > config.FALL_VELOCITY_THRESHOLD
        
        if is_falling_this_frame:
            self.consecutive_fall_frames += 1
        else:
            self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 1)
        
        is_fall = self.consecutive_fall_frames >= config.FALL_CONFIRMATION_FRAMES
        
        return is_fall, results
    
    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()