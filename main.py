# main.py
import cv2
import sys
import config
from core.motion_analyzer import MotionAnalyzer
from core.pose_analyzer import PoseAnalyzer
from core.pose_velocity import PoseVelocityAnalyzer
from core.decision_engine import DecisionEngine
from core.event_logger import EventLogger
from core.person_isolator import ForegroundIsolator
from core.alert_system import AlertManager
from ui.overlay import Overlay
from utils.fps_controller import FPSController


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)
    
    # Core analysis engines
    person_isolator = ForegroundIsolator()
    motion_engine = MotionAnalyzer()
    pose_engine = PoseAnalyzer()
    pose_velocity = PoseVelocityAnalyzer()
    brain = DecisionEngine()
    
    # Support modules
    logger = EventLogger()
    alert_manager = AlertManager(cooldown_seconds=config.ALERT_COOLDOWN_SECONDS)
    ui = Overlay()
    fps_control = FPSController(target_fps=config.FPS_ASSUMED)

    print("="* 50)
    print("  SEIZURE DETECTION SYSTEM")
    print("="* 50)
    print(f"  Resolution: {config.RESIZE_WIDTH}px | Buffer: {config.BUFFER_SECONDS}s")
    print(f"  Seizure Band: {config.FREQ_SEIZURE_LOW}-{config.FREQ_SEIZURE_HIGH} Hz")
    print(f"  Motion Threshold: {config.MOTION_THRESHOLD}")
    print("  Press 'q' or ESC to exit.")
    print("="* 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame.")
            break

        # Person Isolation
        fg_bbox, fg_roi = person_isolator.get_foreground_roi(frame)
        
        frame_area = frame.shape[0] * frame.shape[1]
        use_full_frame = True
        
        if fg_roi is not None and fg_roi.size > 100:
            roi_area = fg_roi.shape[0] * fg_roi.shape[1]
            roi_ratio = roi_area / frame_area
            if roi_ratio >= config.MIN_ROI_RATIO:
                analysis_frame = fg_roi
                use_full_frame = False
        
        if use_full_frame:
            analysis_frame = frame
            if fg_bbox is not None:
                bbox_area = fg_bbox[2] * fg_bbox[3]
                if bbox_area / frame_area < config.MIN_ROI_RATIO:
                    fg_bbox = None
        
        # Motion Analysis (Optical Flow + FFT)
        motion_score = motion_engine.get_motion_score(analysis_frame)
        
        # Pose Analysis
        fall_detected, pose_landmarks = pose_engine.detect_fall(frame)
        
        # Pose Velocity Analysis (Pattern Detection)
        pose_pattern_score = 0.0
        if pose_landmarks and pose_landmarks.pose_landmarks:
            pose_velocity.update(pose_landmarks.pose_landmarks)
            pose_pattern_score = pose_velocity.get_seizure_confidence()
        
        # Combined score: motion (60%) + pose pattern (40%)
        combined_score = motion_score * 0.6 + pose_pattern_score * 0.4

        # Decision
        is_seizure, debug_data = brain.process(combined_score, fall_detected)
        debug_data['fg_bbox'] = fg_bbox
        debug_data['pose_pattern'] = pose_pattern_score

        # Alert
        if is_seizure:
            logger.log_event(debug_data)
            alert_manager.trigger_alert(debug_data)

        # Visualize
        display_frame = ui.draw_hud(frame, combined_score, is_seizure, fall_detected, debug_data)
        person_isolator.draw_foreground_box(display_frame, fg_bbox, label="TARGET")
        
        if pose_landmarks:
            ui.draw_skeleton(display_frame, pose_landmarks)

        cv2.imshow('Seizure Detection System', display_frame)

        fps_control.sync()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem shutdown complete.")


if __name__ == "__main__":
    main()