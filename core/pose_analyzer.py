# core/pose_analyzer.py
"""
Pose Analyzer - Improved Fall Detection
-----------------------------------------
Uses MediaPipe Pose to detect falls with multi-point tracking.

Improvements:
- Tracks 5 body points (nose, shoulders, hips) instead of just nose
- Velocity smoothing over multiple frames
- Multi-frame confirmation to prevent false triggers
- Visibility filtering for robust landmark detection
"""
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
    # Model download URL and local path
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker.task")
    
    # Key pose landmark indices for fall detection
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # All tracked points for multi-point fall detection
    TRACKED_POINTS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    
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
        
        # Multi-point tracking state
        self.prev_y_positions = {}  # Track Y position for each point
        self.velocity_buffer = deque(maxlen=config.VELOCITY_SMOOTH_FRAMES)
        
        # Fall confirmation state
        self.fall_frame_counter = 0
        self.consecutive_fall_frames = 0
        
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
    
    def _calculate_body_velocity(self, landmarks):
        """
        Calculate average downward velocity across key body points.
        
        Why multi-point:
        - Single point (nose) is fragile - head tilts cause false positives
        - Multiple points provide consensus - a real fall shows ALL points moving down
        - Weighted by visibility ensures we only use reliable landmarks
        
        Returns: (velocity, num_valid_points)
        """
        velocities = []
        weights = []
        
        for point_idx in self.TRACKED_POINTS:
            landmark = landmarks[point_idx]
            
            # Only use landmarks with good visibility
            if landmark.visibility < 0.5:
                continue
            
            current_y = landmark.y
            
            if point_idx in self.prev_y_positions:
                prev_y = self.prev_y_positions[point_idx]
                velocity = current_y - prev_y  # Positive = moving down
                velocities.append(velocity)
                weights.append(landmark.visibility)
            
            # Update position history
            self.prev_y_positions[point_idx] = current_y
        
        if not velocities:
            return 0.0, 0
        
        # Weighted average velocity
        weights = np.array(weights)
        velocities = np.array(velocities)
        weighted_velocity = np.sum(velocities * weights) / np.sum(weights)
        
        return weighted_velocity, len(velocities)
    
    def _get_smoothed_velocity(self, velocity):
        """
        Apply temporal smoothing to velocity using a rolling buffer.
        
        This prevents single-frame spikes from triggering false falls.
        """
        self.velocity_buffer.append(velocity)
        
        if len(self.velocity_buffer) < 2:
            return velocity
        
        # Use median for robustness against outliers
        return np.median(self.velocity_buffer)
        
    def detect_fall(self, frame):
        """
        Detect if a fall is occurring using multi-point pose analysis.
        
        Returns: (bool is_fall, PoseLandmarkerResult or None)
        
        Improvements:
        1. Tracks 5 body points instead of just nose
        2. Smooths velocity over multiple frames
        3. Requires sustained downward motion (confirmation)
        4. Filters by landmark visibility
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect pose landmarks
        results = self.landmarker.detect(mp_image)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            # No pose detected - reset tracking
            self.prev_y_positions.clear()
            self.velocity_buffer.clear()
            self.consecutive_fall_frames = 0
            return False, None

        # Get landmarks for the first detected person
        landmarks = results.pose_landmarks[0]
        
        # Calculate multi-point velocity
        raw_velocity, num_points = self._calculate_body_velocity(landmarks)
        
        # Need at least 3 valid points for reliable detection
        if num_points < 3:
            self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 1)
            return False, results
        
        # Apply temporal smoothing
        smoothed_velocity = self._get_smoothed_velocity(raw_velocity)
        
        # Check if velocity indicates a fall (fast downward motion)
        is_falling_this_frame = smoothed_velocity > config.FALL_VELOCITY_THRESHOLD
        
        # Multi-frame confirmation
        if is_falling_this_frame:
            self.consecutive_fall_frames += 1
        else:
            # Decay counter but don't reset immediately
            self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 1)
        
        # Only trigger fall if sustained for multiple frames
        is_fall = self.consecutive_fall_frames >= config.FALL_CONFIRMATION_FRAMES
        
        return is_fall, results
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()