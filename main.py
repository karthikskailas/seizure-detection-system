# main.py
import cv2
import time
import config
from core.motion_analyzer import MotionAnalyzer
from core.pose_analyzer import PoseAnalyzer
from core.decision_engine import DecisionEngine
from core.event_logger import EventLogger
from ui.overlay import Overlay
from utils.fps_controller import FPSController

def main():
    # 1. Initialize Modules
    cap = cv2.VideoCapture(0) # Use 0 for webcam, or 'path/to/video.mp4'
    
    motion_engine = MotionAnalyzer()
    pose_engine = PoseAnalyzer()
    brain = DecisionEngine()
    logger = EventLogger()
    ui = Overlay()
    fps_control = FPSController(target_fps=config.FPS_ASSUMED)

    print(f"System Started. Analysis Resolution: {config.RESIZE_WIDTH}px width")
    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Analyze (The "Sensors")
        # Symptom 1: Shaking (Spectral Analysis)
        motion_score = motion_engine.get_motion_score(frame)
        
        # Symptom 2: Falling (Pose Analysis)
        fall_detected, pose_landmarks = pose_engine.detect_fall(frame)

        # 3. Decide (The "Brain")
        is_seizure, debug_data = brain.process(motion_score, fall_detected)

        # 4. Act (Log & Alert)
        if is_seizure:
            logger.log_event(debug_data)
            # Optional: Trigger sound alarm here
            # print("\a") 

        # 5. Visualize (The UI)
        display_frame = ui.draw_hud(frame, motion_score, is_seizure, fall_detected, debug_data)
        
        # Draw skeleton if fall detected (optional debug)
        if pose_landmarks:
            ui.draw_skeleton(display_frame, pose_landmarks)

        cv2.imshow('Seizure Detection System - MVP', display_frame)

        # 6. Maintain FPS (Crucial for FFT math)
        fps_control.sync()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()