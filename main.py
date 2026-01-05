# main.py
import cv2
import sys
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
from utils.fps_controller import FPSController


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)
    
    # Core engines (run in parallel)
    person_isolator = ForegroundIsolator()
    motion_engine = MotionAnalyzer()
    pose_engine = PoseAnalyzer()
    pose_velocity = PoseVelocityAnalyzer()
    face_engine = FaceAnalyzer()
    brain = DecisionEngine(fps=config.FPS_ASSUMED)
    
    # Support modules
    logger = EventLogger()
    alert_manager = AlertManager(
        cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
        audio_duration=config.ALERT_SOUND_DURATION
    )
    ui = Overlay()
    fps_control = FPSController(target_fps=config.FPS_ASSUMED)

    print("=" * 50)
    print("  SEIZURE DETECTION SYSTEM")
    print("  Multimodal: Motion + Pose + Face")
    print("=" * 50)
    print("  Detection: CLONIC | TONIC | ATONIC")
    print("  Press 'q' or ESC to exit.")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # === 1. Person Isolation ===
        fg_bbox, fg_roi = person_isolator.get_foreground_roi(frame)
        
        frame_area = frame.shape[0] * frame.shape[1]
        analysis_frame = frame
        
        if fg_roi is not None and fg_roi.size > 100:
            roi_area = fg_roi.shape[0] * fg_roi.shape[1]
            if roi_area / frame_area >= config.MIN_ROI_RATIO:
                analysis_frame = fg_roi
        
        # === 2. Motion Analysis (Optical Flow + FFT) ===
        motion_score = motion_engine.get_motion_score(analysis_frame)
        motion_data = {
            'score': motion_score,
            'frequency': motion_engine.get_frequency()
        }
        
        # === 3. Pose Analysis (Body) ===
        is_fallen, pose_results = pose_engine.detect_fall(frame)
        
        # Pose velocity for tremor and rigidity
        tremor_score = 0.0
        is_rigid = False
        is_slumping = is_fallen
        
        if pose_results and pose_results.pose_landmarks:
            pose_velocity.update(pose_results.pose_landmarks)
            tremor_score = pose_velocity.detect_tremor()
            is_rigid = pose_velocity.detect_rigidity()
        
        pose_data = {
            'is_rigid': is_rigid,
            'is_fallen': is_fallen,
            'is_slumping': is_slumping,
            'tremor': tremor_score
        }
        
        # === 4. Face Analysis (Head + Face) ===
        face_data = face_engine.analyze(frame)
        
        # === 5. Multimodal Decision ===
        is_seizure, debug_data = brain.process(motion_data, pose_data, face_data)
        
        # Add extra debug info
        debug_data['fg_bbox'] = fg_bbox
        debug_data['head_shake'] = face_data.get('head_shake_score', 0.0)
        debug_data['face_detected'] = face_data.get('face_detected', False)

        # === 6. Alert ===
        if is_seizure:
            logger.log_event(debug_data)
            alert_manager.trigger_alert(debug_data)

        # === 7. Visualize ===
        display_frame = ui.draw_hud(frame, debug_data['risk'], is_seizure, is_fallen, debug_data)
        person_isolator.draw_foreground_box(display_frame, fg_bbox, label="TARGET")
        
        # Draw skeleton
        if pose_results:
            ui.draw_skeleton(display_frame, pose_results)
        
        # Show face status
        if face_data.get('face_detected'):
            cv2.putText(display_frame, f"Face: {face_data['head_shake_score']:.0%}", 
                       (10, display_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Seizure Detection System', display_frame)

        fps_control.sync()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup
    face_engine.close()
    pose_engine.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem shutdown complete.")


if __name__ == "__main__":
    main()