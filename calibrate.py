# calibrate.py
import cv2
import numpy as np
from scipy.fft import fft, fftfreq
from collections import deque
import os
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MultimodalCalibrator:
    def __init__(self):
        self.motion_samples = []
        self.head_shake_samples = []
        self.facial_distortion_samples = []
        self.mouth_ratio_samples = []
        self.tremor_samples = []
        self.skip_preview = False
        self._init_face_detector()
        
    def _init_face_detector(self):
        model_path = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(options)
        else:
            self.face_detector = None
            print("  Face model not found")
    
    def analyze_video(self, video_path):
        print(f"\n{os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {fps:.0f}fps, {frame_count} frames")
        
        prev_gray = None
        nose_x_history = deque(maxlen=10)
        nose_y_history = deque(maxlen=10)
        velocity_history = deque(maxlen=30)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            scale = 320 / w
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            motion_score = 0.0
            if prev_gray is not None and prev_gray.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 1, 15, 1, 5, 1.1, 0
                )
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = np.mean(magnitude)
                self.motion_samples.append(motion_score)
                velocity_history.append(motion_score)
            
            prev_gray = gray
            
            if len(velocity_history) >= 20:
                v_data = np.array(velocity_history)
                self.tremor_samples.append(np.std(v_data))
            
            head_shake = 0.0
            if self.face_detector:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = self.face_detector.detect(mp_img)
                    
                    if result.face_landmarks:
                        lm = result.face_landmarks[0]
                        nose = lm[1]
                        nose_x_history.append(nose.x)
                        nose_y_history.append(nose.y)
                        
                        if len(nose_x_history) >= 5:
                            x_var = np.std(nose_x_history) * 100
                            y_var = np.std(nose_y_history) * 100
                            head_shake = min(1.0, (x_var + y_var) / 2)
                            self.head_shake_samples.append(head_shake)
                        
                        mouth_v = abs(lm[13].y - lm[14].y)
                        mouth_h = abs(lm[61].x - lm[291].x)
                        mouth_ratio = mouth_v / mouth_h if mouth_h > 0 else 0
                        self.mouth_ratio_samples.append(mouth_ratio)
                        
                        left_ear = abs(lm[159].y - lm[145].y) / max(abs(lm[33].x - lm[133].x), 0.001)
                        self.facial_distortion_samples.append(left_ear)
                except:
                    pass
            
            frame_idx += 1
            
            if not self.skip_preview:
                display = frame.copy()
                cv2.putText(display, f"M:{motion_score:.2f} H:{head_shake:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Calibration", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.skip_preview = True
                    cv2.destroyAllWindows()
        
        cap.release()
        print(f"  {frame_idx} frames analyzed")
    
    def generate_thresholds(self):
        print("\n" + "="*40)
        print("RESULTS")
        print("="*40)
        
        results = {}
        
        if self.motion_samples:
            motion = np.array(self.motion_samples)
            results['motion_noise_floor'] = round(np.percentile(motion, 30), 3)
            results['motion_p50'] = round(np.percentile(motion, 50), 3)
            print(f"Motion: P30={results['motion_noise_floor']:.3f}")
        
        if self.head_shake_samples:
            shake = np.array(self.head_shake_samples)
            results['head_shake_p75'] = round(np.percentile(shake, 75), 3)
            print(f"Head Shake: P75={results['head_shake_p75']:.3f}")
        
        if self.mouth_ratio_samples:
            mouth = np.array(self.mouth_ratio_samples)
            results['mouth_p75'] = round(np.percentile(mouth, 75), 3)
            print(f"Mouth Ratio: P75={results['mouth_p75']:.3f}")
        
        if self.tremor_samples:
            tremor = np.array(self.tremor_samples)
            results['tremor_p50'] = round(np.percentile(tremor, 50), 3)
            print(f"Tremor: P50={results['tremor_p50']:.3f}")
        
        motion_threshold = results.get('motion_p50', 0.5) / max(results.get('motion_p50', 1.0) * 2, 0.1) + 0.3
        head_shake_threshold = min(0.9, results.get('head_shake_p75', 0.6) + 0.15)
        mouth_threshold = min(0.8, results.get('mouth_p75', 0.5) + 0.1)
        
        thresholds = {
            'MOTION_NOISE_FLOOR': results.get('motion_noise_floor', 0.1),
            'MOTION_THRESHOLD': round(min(0.7, motion_threshold), 2),
            'HEAD_SHAKE_THRESHOLD': round(head_shake_threshold, 2),
            'FACIAL_DISTORTION_THRESHOLD': 0.6,
            'MOUTH_OPEN_THRESHOLD': round(mouth_threshold, 2),
            'TREMOR_THRESHOLD': round(results.get('tremor_p50', 0.4), 2),
        }
        
        self._save_config(thresholds, results)
        return thresholds
    
    def _save_config(self, thresholds, results):
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        
        content = f'''# config.py
RESIZE_WIDTH = 320
FPS_ASSUMED = 30
BUFFER_SECONDS = 3.0
BUFFER_SIZE = int(FPS_ASSUMED * BUFFER_SECONDS)

MOTION_EMA_ALPHA = 0.2
VELOCITY_SMOOTH_FRAMES = 5
BASELINE_CALIBRATION_SECONDS = 0

MOTION_NOISE_FLOOR = {thresholds['MOTION_NOISE_FLOOR']}
ENERGY_NOISE_FLOOR = {results.get('motion_p50', 0.3) * 10}

FALL_VELOCITY_THRESHOLD = 0.06
FALL_CONFIRMATION_FRAMES = 4

FREQ_NORMAL_LOW = 0.1
FREQ_NORMAL_HIGH = 2.0
FREQ_SEIZURE_LOW = 2.0
FREQ_SEIZURE_HIGH = 7.0

MOTION_THRESHOLD = {thresholds['MOTION_THRESHOLD']}
FALL_TRIGGER_SENSITIVITY = 0.5
DURATION_THRESHOLD = 15
COOLDOWN_FRAMES = 90

HEAD_SHAKE_THRESHOLD = {thresholds['HEAD_SHAKE_THRESHOLD']}
FACIAL_DISTORTION_THRESHOLD = {thresholds['FACIAL_DISTORTION_THRESHOLD']}
MOUTH_OPEN_THRESHOLD = {thresholds['MOUTH_OPEN_THRESHOLD']}
TREMOR_THRESHOLD = {thresholds['TREMOR_THRESHOLD']}

FG_HISTORY = 300
FG_VAR_THRESHOLD = 20
FG_MIN_AREA_RATIO = 0.005
FG_STABILITY_THRESHOLD = 0.2
FG_LOST_FRAMES_MAX = 15

MIN_ROI_RATIO = 0.05

ALERT_COOLDOWN_SECONDS = 30
ALERT_SOUND_DURATION = 5
ALERT_CONFIG_FILE = "data/alert_config.json"
'''
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print(f"\nSaved: {config_path}")


def main():
    clips_dir = os.path.join(os.path.dirname(__file__), "data", "clips")
    
    print("="*40)
    print("CALIBRATOR")
    print("="*40)
    
    videos = set()
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        videos.update(glob.glob(os.path.join(clips_dir, ext)))
    videos = sorted(list(videos))
    
    if not videos:
        print(f"No videos in: {clips_dir}")
        return
    
    print(f"Found {len(videos)} video(s)")
    
    calibrator = MultimodalCalibrator()
    
    for video in videos:
        calibrator.analyze_video(video)
    
    cv2.destroyAllWindows()
    calibrator.generate_thresholds()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
