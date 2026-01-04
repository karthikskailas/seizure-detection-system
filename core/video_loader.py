# core/video_loader.py
import cv2
import time

class VideoLoader:
    def __init__(self, source=0):
        """
        source: 0 for Webcam, or string path "data/clips/test_video.mp4"
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.is_file = isinstance(source, str)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

    def get_frame(self):
        """
        Returns: (ret, frame)
        If video file ends, it automatically rewinds to the start (Looping).
        """
        ret, frame = self.cap.read()
        
        # Auto-loop logic for video files
        if not ret and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        return ret, frame

    def release(self):
        self.cap.release()