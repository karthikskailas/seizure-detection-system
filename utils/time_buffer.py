# utils/time_buffer.py
import time

class TimeBuffer:
    def __init__(self, duration_seconds):
        self.duration = duration_seconds
        self.last_event_time = 0

    def trigger(self):
        self.last_event_time = time.time()

    def is_active(self):
        """Returns True if we are still within the time buffer window"""
        return (time.time() - self.last_event_time) < self.duration