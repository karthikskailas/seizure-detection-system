# ui/overlay.py
import cv2
import numpy as np


# Pose connections for drawing skeleton (based on MediaPipe Pose model)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
]


class Overlay:
    def draw_hud(self, frame, motion_score, is_seizure, fall_detected, debug_data):
        h, w = frame.shape[:2]
        
        # 1. Status Bar Background
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # 2. Status Text
        status_color = (0, 255, 0)  # Green
        status_text = "MONITORING"
        
        if is_seizure:
            status_color = (0, 0, 255)  # Red
            status_text = "!! SEIZURE DETECTED !!"
        elif fall_detected:
            status_color = (0, 165, 255)  # Orange
            status_text = "WARNING: FALL DETECTED"

        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # 3. Metrics (Right Side)
        score_text = f"Motion Energy: {int(motion_score * 100)}%"
        cv2.putText(frame, score_text, (w - 300, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 4. Progress Bar for Motion
        bar_width = 200
        bar_fill = int(motion_score * bar_width)
        cv2.rectangle(frame, (w - 250, 50), (w - 50, 55), (100, 100, 100), -1)
        cv2.rectangle(frame, (w - 250, 50), (w - 250 + bar_fill, 55), status_color, -1)

        return frame

    def draw_skeleton(self, frame, pose_results):
        """
        Draw pose landmarks on the frame using the new MediaPipe Tasks API result.
        
        Args:
            frame: The BGR image frame
            pose_results: PoseLandmarkerResult from the new API
        """
        if pose_results is None or not pose_results.pose_landmarks:
            return frame
            
        h, w = frame.shape[:2]
        
        # Get the first detected pose
        landmarks_list = pose_results.pose_landmarks[0]
        
        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                start = landmarks_list[start_idx]
                end = landmarks_list[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
        
        # Draw landmarks
        for landmark in landmarks_list:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        return frame