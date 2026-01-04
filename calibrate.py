# calibrate.py
import cv2
import numpy as np
from scipy.fft import fft, fftfreq
from collections import deque
import os
import glob
import config


class VideoCalibrator:
    def __init__(self):
        self.motion_samples = []
        self.freq_samples = []
        
    def analyze_video(self, video_path):
        print(f"\n📹 Analyzing: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ❌ Could not open video")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"  📊 FPS: {fps:.1f}, Frames: {frame_count}, Duration: {duration:.1f}s")
        
        prev_gray = None
        motion_buffer = deque(maxlen=int(fps * 3))
        all_motion = []
        
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
            
            if prev_gray is not None and prev_gray.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=1, winsize=15,
                    iterations=1, poly_n=5, poly_sigma=1.1, flags=0
                )
                
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_motion = np.mean(magnitude)
                
                ref_pixels = 320 * 240
                curr_pixels = gray.shape[0] * gray.shape[1]
                normalized = avg_motion * (ref_pixels / max(curr_pixels, 1))
                
                all_motion.append(normalized)
                motion_buffer.append(normalized)
                
                if len(motion_buffer) >= int(fps * 2) and frame_idx % int(fps) == 0:
                    self._analyze_fft(motion_buffer, fps)
            
            prev_gray = gray
            frame_idx += 1
        
        cap.release()
        
        if all_motion:
            stats = {
                'file': os.path.basename(video_path),
                'fps': fps,
                'duration': duration,
                'motion_mean': np.mean(all_motion),
                'motion_max': np.max(all_motion),
                'motion_std': np.std(all_motion),
                'motion_p90': np.percentile(all_motion, 90),
                'motion_p95': np.percentile(all_motion, 95),
            }
            
            print(f"  ✅ Motion Stats:")
            print(f"     Mean: {stats['motion_mean']:.3f}")
            print(f"     Max:  {stats['motion_max']:.3f}")
            print(f"     P90:  {stats['motion_p90']:.3f}")
            print(f"     P95:  {stats['motion_p95']:.3f}")
            
            self.motion_samples.extend(all_motion)
            return stats
        
        return None
    
    def _analyze_fft(self, motion_buffer, fps):
        data = np.array(motion_buffer)
        data = data - np.mean(data)
        
        if np.std(data) < 0.01:
            return
        
        N = len(data)
        windowed = data * np.hanning(N)
        yf = fft(windowed)
        xf = fftfreq(N, 1 / fps)
        
        power = np.abs(yf[:N//2])
        freqs = xf[:N//2]
        
        mask = (freqs >= 2.0) & (freqs <= 7.0)
        if np.any(mask):
            seizure_power = power[mask]
            seizure_freqs = freqs[mask]
            if len(seizure_power) > 0:
                peak_idx = np.argmax(seizure_power)
                peak_freq = seizure_freqs[peak_idx]
                peak_power = seizure_power[peak_idx]
                self.freq_samples.append((peak_freq, peak_power))
    
    def generate_config(self):
        if not self.motion_samples:
            print("\n❌ No video data to analyze!")
            return None
        
        motion = np.array(self.motion_samples)
        
        print("\n" + "="*60)
        print("📊 CALIBRATION RESULTS")
        print("="*60)
        
        noise_floor = np.percentile(motion, 30)
        motion_threshold = np.percentile(motion, 50)
        energy_threshold = np.percentile(motion, 40) * 15
        
        print(f"\n📈 Motion Analysis (from {len(motion)} samples):")
        print(f"   Min:    {np.min(motion):.3f}")
        print(f"   P30:    {np.percentile(motion, 30):.3f}")
        print(f"   P50:    {np.percentile(motion, 50):.3f}")
        print(f"   P70:    {np.percentile(motion, 70):.3f}")
        print(f"   P90:    {np.percentile(motion, 90):.3f}")
        print(f"   Max:    {np.max(motion):.3f}")
        
        if self.freq_samples:
            freqs = [f[0] for f in self.freq_samples]
            print(f"\n🔊 Dominant Frequencies:")
            print(f"   Mean: {np.mean(freqs):.2f} Hz")
            print(f"   Range: {np.min(freqs):.2f} - {np.max(freqs):.2f} Hz")
        
        recommended = {
            'MOTION_NOISE_FLOOR': round(noise_floor * 0.8, 2),
            'ENERGY_NOISE_FLOOR': round(energy_threshold, 1),
            'MOTION_THRESHOLD': round(min(0.85, motion_threshold / np.max(motion) + 0.3), 2),
            'MOTION_EMA_ALPHA': 0.2,
            'DURATION_THRESHOLD': 20,
        }
        
        if np.max(motion) < 3.0:
            print("\n⚠️ Detected LIGHT seizure pattern - using sensitive settings")
            recommended['MOTION_NOISE_FLOOR'] = round(noise_floor * 0.5, 2)
            recommended['MOTION_THRESHOLD'] = 0.55
            recommended['ENERGY_NOISE_FLOOR'] = 5.0
        elif np.max(motion) > 8.0:
            print("\n💪 Detected INTENSE seizure pattern - using robust settings")
            recommended['MOTION_NOISE_FLOOR'] = round(noise_floor, 2)
            recommended['MOTION_THRESHOLD'] = 0.70
        
        print("\n" + "="*60)
        print("📋 RECOMMENDED CONFIG VALUES")
        print("="*60)
        print("\n# Copy these to config.py:\n")
        for key, value in recommended.items():
            print(f"{key} = {value}")
        
        return recommended


def main():
    clips_dir = os.path.join(os.path.dirname(__file__), "data", "clips")
    
    print("="*60)
    print("  SEIZURE DETECTION CALIBRATION TOOL")
    print("="*60)
    print(f"\n📁 Looking for videos in: {clips_dir}")
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.MP4', '*.MOV']
    videos = []
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(clips_dir, ext)))
    
    if not videos:
        print("\n❌ No videos found in data/clips/")
        print("\n📝 Please add seizure videos to this folder:")
        print(f"   {clips_dir}")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .webm")
        return
    
    print(f"\n✅ Found {len(videos)} video(s)")
    
    calibrator = VideoCalibrator()
    
    for video in videos:
        calibrator.analyze_video(video)
    
    calibrator.generate_config()
    
    print("\n" + "="*60)
    print("✅ Calibration complete!")
    print("="*60)


if __name__ == "__main__":
    main()
