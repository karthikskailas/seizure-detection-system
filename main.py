# main.py
"""
Seizure Detection System - Main Entry Point
--------------------------------------------
Real-time video-based seizure detection using:
- Person isolation (MOG2 background subtraction)
- Motion physics engine (optical flow + FFT)
- Fall detection (MediaPipe pose)

This is an assistive alert system, not a medical diagnostic device.
"""

import cv2
import sys
import config
from core.motion_analyzer import MotionAnalyzer
from core.pose_analyzer import PoseAnalyzer
from core.decision_engine import DecisionEngine
from core.event_logger import EventLogger
from core.person_isolator import ForegroundIsolator
from core.alert_system import AlertManager
from ui.overlay import Overlay
from utils.fps_controller import FPSController


def main():
    # ========================================
    # 1. INITIALIZE MODULES
    # ========================================
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or 'path/to/video.mp4'
    
    # Handle camera open failure safely
    if not cap.isOpened():
        print("ERROR: Could not open camera. Check connection and permissions.")
        sys.exit(1)
    
    # Core analysis engines
    person_isolator = ForegroundIsolator()  # Locks onto foreground patient
    motion_engine = MotionAnalyzer()        # Detects rhythmic shaking via FFT
    pose_engine = PoseAnalyzer()            # Detects falls
    brain = DecisionEngine()                # Makes final seizure decision
    
    # Support modules
    logger = EventLogger()
    alert_manager = AlertManager(cooldown_seconds=config.ALERT_COOLDOWN_SECONDS)
    ui = Overlay()
    fps_control = FPSController(target_fps=config.FPS_ASSUMED)

    print("="* 50)
    print("  SEIZURE DETECTION SYSTEM - MVP")
    print("="* 50)
    print(f"  Resolution: {config.RESIZE_WIDTH}px | Buffer: {config.BUFFER_SECONDS}s")
    print(f"  Seizure Band: {config.FREQ_SEIZURE_LOW}-{config.FREQ_SEIZURE_HIGH} Hz")
    print(f"  Alert Cooldown: {config.ALERT_COOLDOWN_SECONDS}s")
    print("  Press 'q' or ESC to exit.")
    print("="* 50)

    # ========================================
    # 2. MAIN PROCESSING LOOP
    # ========================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame. Camera disconnected?")
            break

        # ----------------------------------------
        # STEP 1: Person Isolation
        # ----------------------------------------
        # Lock onto the closest person (foreground patient)
        fg_bbox, fg_roi = person_isolator.get_foreground_roi(frame)
        
        # Validate ROI size - if too small, use full frame
        # This prevents analyzing tiny noise regions instead of actual person
        frame_area = frame.shape[0] * frame.shape[1]
        use_full_frame = True
        
        if fg_roi is not None and fg_roi.size > 100:
            roi_area = fg_roi.shape[0] * fg_roi.shape[1]
            roi_ratio = roi_area / frame_area
            # Only use ROI if it's at least 5% of frame
            if roi_ratio >= config.MIN_ROI_RATIO:
                analysis_frame = fg_roi
                use_full_frame = False
        
        if use_full_frame:
            analysis_frame = frame
            # Don't draw tiny/invalid bounding boxes
            if fg_bbox is not None:
                bbox_area = fg_bbox[2] * fg_bbox[3]
                if bbox_area / frame_area < config.MIN_ROI_RATIO:
                    fg_bbox = None  # Clear invalid small box
        
        # ----------------------------------------
        # STEP 2: Motion Analysis (FFT Physics)
        # ----------------------------------------
        # Detects rhythmic shaking in the 2-7 Hz range
        # Returns 0.0 if person is idle (noise gate active)
        motion_score = motion_engine.get_motion_score(analysis_frame)
        
        # ----------------------------------------
        # STEP 3: Fall Detection (Pose Analysis)
        # ----------------------------------------
        # Uses MediaPipe to detect torso collapse
        fall_detected, pose_landmarks = pose_engine.detect_fall(frame)

        # ----------------------------------------
        # STEP 4: Decision (The Brain)
        # ----------------------------------------
        # Combines motion + fall signals with temporal logic
        is_seizure, debug_data = brain.process(motion_score, fall_detected)
        
        # Add foreground info to debug data
        debug_data['fg_bbox'] = fg_bbox

        # ----------------------------------------
        # STEP 5: Act (Log & Alert)
        # ----------------------------------------
        if is_seizure:
            logger.log_event(debug_data)
            # Comprehensive alert system (audio + WhatsApp/Email)
            alert_manager.trigger_alert(debug_data)

        # ----------------------------------------
        # STEP 6: Visualize (UI Overlay)
        # ----------------------------------------
        display_frame = ui.draw_hud(frame, motion_score, is_seizure, fall_detected, debug_data)
        
        # Draw foreground person bounding box
        person_isolator.draw_foreground_box(display_frame, fg_bbox, label="TARGET LOCKED")
        
        # Draw skeleton if available (optional debug)
        if pose_landmarks:
            ui.draw_skeleton(display_frame, pose_landmarks)

        cv2.imshow('Seizure Detection System - MVP', display_frame)

        # ----------------------------------------
        # STEP 7: Maintain FPS (Critical for FFT)
        # ----------------------------------------
        fps_control.sync()

        # Exit controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    # ========================================
    # 3. CLEANUP
    # ========================================
    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem shutdown complete.")


if __name__ == "__main__":
    main()