"""Real-time video-based seizure detection system.

Integrates motion analysis, pose tracking, face monitoring, and multi-modal
decision logic to detect potential seizure events and trigger alerts.
"""

import cv2
import sys
import numpy as np
import config
from core.motion_analyzer import MotionAnalyzer
from core.pose_analyzer import PoseAnalyzer
from core.pose_velocity import PoseVelocityAnalyzer
from core.face_analyzer import FaceAnalyzer
from core.decision_engine import DecisionEngine
from core.event_logger import EventLogger
from core.person_isolator import ForegroundIsolator
from core.alert_system import AlertManager
from ui.overlay import Overlay
from ui.video_controls import VideoControlPanel
from utils.fps_controller import FPSController


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        sys.exit(1)
    
    person_isolator = ForegroundIsolator()
    motion_engine = MotionAnalyzer()
    pose_engine = PoseAnalyzer()
    pose_velocity = PoseVelocityAnalyzer()
    face_engine = FaceAnalyzer()
    brain = DecisionEngine(fps=config.FPS_ASSUMED)
    
    logger = EventLogger()
    alert_manager = AlertManager(
        cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
        audio_duration=config.ALERT_SOUND_DURATION
    )
    ui = Overlay()
    video_controls = VideoControlPanel(window_name='Seizure Detection')
    fps_control = FPSController(target_fps=config.FPS_ASSUMED)

    print("=" * 40)
    print("SEIZURE DETECTION SYSTEM")
    print("RUNNING")
    print("=" * 40)
    
    # Create window and setup video controls
    cv2.namedWindow('Seizure Detection', cv2.WINDOW_NORMAL)
    video_controls.setup_mouse_callback()


    while cap.isOpened():
        # Check if user clicked stop button
        if video_controls.is_stopped():
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # If paused, just display the frame with controls and skip processing
        if video_controls.is_paused():
            display_frame = frame.copy()
            display_frame = video_controls.draw_controls(display_frame)
            cv2.imshow('Seizure Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            continue

        # 1. Background Subtraction / Motion
        fg_bbox, fg_roi = person_isolator.get_foreground_roi(frame)
        
        frame_area = frame.shape[0] * frame.shape[1]
        analysis_frame = frame
        
        if fg_roi is not None and fg_roi.size > 100:
            roi_area = fg_roi.shape[0] * fg_roi.shape[1]
            if roi_area / frame_area >= config.MIN_ROI_RATIO:
                analysis_frame = fg_roi
        
        motion_score = motion_engine.get_motion_score(analysis_frame)
        motion_data = {
            'score': motion_score,
            'frequency': motion_engine.get_frequency()
        }
        
        # 2. Pose Analysis (Multi-Person + Seizure Scoring)
        is_fallen, pose_results = pose_engine.analyze(frame)
        
        tremor_score = 0.0
        is_rigid = False
        is_slumping = is_fallen
        
        # Analyze the TARGET person specifically if found
        if pose_results and pose_results.pose_landmarks:
            pose_velocity.update(pose_results.pose_landmarks) # Use the selected target
            tremor_score = pose_velocity.detect_tremor()
            is_rigid = pose_velocity.detect_rigidity()
        
        pose_data = {
            'is_rigid': is_rigid,
            'is_fallen': is_fallen,
            'is_slumping': is_slumping,
            'tremor': tremor_score
        }
        
        # 3. Face Analysis
        face_data = face_engine.analyze(frame)
        
        # 4. Decision Engine
        is_seizure, debug_data = brain.process(motion_data, pose_data, face_data)
        
        debug_data['fg_bbox'] = fg_bbox
        debug_data['head_shake'] = face_data.get('head_shake_score', 0.0)
        debug_data['face_detected'] = face_data.get('face_detected', False)

        if is_seizure:
            logger.log_event(debug_data)
            alert_manager.trigger_alert(debug_data)

        # 5. UI Drawing
        # Use the debug image from PoseAnalyzer which has the RED/GREEN boxes
        base_display = pose_results.debug_image if (pose_results and hasattr(pose_results, 'debug_image')) else frame
        
        display_frame = ui.draw_hud(base_display, debug_data['risk'], is_seizure, is_fallen, debug_data)
        person_isolator.draw_foreground_box(display_frame, fg_bbox, label="MOTION TARGET")
        
        # Draw video controls
        display_frame = video_controls.draw_controls(display_frame)
        
        if pose_results and pose_results.pose_landmarks:
            # We don't need to draw skeleton again if PoseAnalyzer did it, but Overlay.draw_skeleton is useful
            # Let's verify: PoseAnalyzer draws boxes, but maybe not skeleton?
            # My PoseAnalyzer implementation drew boxes and scores, not skeletons.
            # So we pass the MockResults to Overlay to draw skeleton of the Patient.
            ui.draw_skeleton(display_frame, pose_results)
        
        if face_data.get('face_detected'):
            cv2.putText(display_frame, f"Face: {face_data['head_shake_score']:.0%}", 
                       (10, display_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Seizure Detection', display_frame)
        
        # [NEW] Mask Filter Window
        if pose_results and hasattr(pose_results, 'mask_visual'):
            cv2.imshow('Mask Filter (Patient)', pose_results.mask_visual)
        else:
            # Show empty black frame if no mask
            cv2.imshow('Mask Filter (Patient)', np.zeros_like(frame))

        fps_control.sync()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    face_engine.close()
    pose_engine.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()