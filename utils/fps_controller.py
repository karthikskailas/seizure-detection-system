# utils/fps_controller.py
import time

class FPSController:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        self.prev_time = time.time()

    def sync(self):
        """
        Call this at the end of your main loop.
        It sleeps for the exact time needed to maintain steady FPS.
        """
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        wait_time = self.frame_duration - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
            
        self.prev_time = time.time()