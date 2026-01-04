# core/event_logger.py
import json
import time
import os
from datetime import datetime

class EventLogger:
    def __init__(self):
        self.log_dir = "data/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file = os.path.join(self.log_dir, f"session_{int(time.time())}.json")
        self.last_log_time = 0

    def log_event(self, data):
        # Don't spam the log file (limit to 1 log per second)
        if time.time() - self.last_log_time < 1.0:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "risk_score": float(data.get('risk', 0)),
            "status": "SEIZURE_DETECTED",
            "metadata": data
        }

        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
            
        print(f"[ALERT] Event Logged: {event['timestamp']}")
        self.last_log_time = time.time()