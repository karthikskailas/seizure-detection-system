import os
import json
import time
import threading
import requests
from datetime import datetime
from typing import Optional, Dict, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class AlertConfig:
    DEFAULT_CONFIG = {
        "emergency_contact_email":("",),
        "use_email": True,
        "alert_message": "ALERT: Possible seizure detected",
        "location_enabled": True,
        "smtp_host": "sandbox.smtp.mailtrap.io",
        "smtp_port": 2525,
        "smtp_username": "",
        "smtp_password": "",
        "email_from": "seizure-alert@detection.local"
    }
    
    def __init__(self, config_path: str = "data/alert_config.json"):
        self.config_path = config_path
        self._ensure_config_exists()
        self.config = self._load_config()
    
    def _ensure_config_exists(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2)
    
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r') as f:
                loaded = json.load(f)
                return {**self.DEFAULT_CONFIG, **loaded}
        except (json.JSONDecodeError, FileNotFoundError):
            return self.DEFAULT_CONFIG.copy()
    
    def save(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        self.save()


class LocationService:
    IPINFO_URL = "https://ipinfo.io/json"
    
    def __init__(self):
        self._cached_location: Optional[Dict] = None
        self._cache_time: float = 0
        self._cache_duration: float = 300
    
    def get_location(self) -> Dict[str, Any]:
        if self._cached_location and (time.time() - self._cache_time) < self._cache_duration:
            return self._cached_location
        
        try:
            response = requests.get(self.IPINFO_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            location = {
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
                "country": data.get("country", "Unknown"),
                "coordinates": data.get("loc", "Unknown"),
                "ip": data.get("ip", "Unknown"),
                "source": "IP-based",
                "timestamp": datetime.now().isoformat()
            }
            
            self._cached_location = location
            self._cache_time = time.time()
            return location
            
        except Exception:
            return {
                "city": "Unknown",
                "region": "Unknown",
                "country": "Unknown",
                "coordinates": "Unknown",
                "source": "Error",
                "timestamp": datetime.now().isoformat()
            }
    
    def format_for_message(self) -> str:
        loc = self.get_location()
        if loc["city"] != "Unknown":
            return f"Location: {loc['city']}, {loc['region']}, {loc['country']}"
        return "Location: Unable to determine"


class AudioAlert:
    ALARM_FREQUENCIES = [880, 1100, 880, 1100]
    TONE_DURATION_MS = 300
    MP3_PATH = "data/alert_sound.mp3"
    
    def __init__(self, duration_seconds: int = 5):
        self.duration_seconds = duration_seconds
        self._is_playing = False
        self._stop_flag = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._use_pygame = False
        
        try:
            import pygame
            pygame.mixer.init()
            if os.path.exists(self.MP3_PATH):
                pygame.mixer.music.load(self.MP3_PATH)
                self._use_pygame = True
        except ImportError:
            pass
        except Exception:
            pass
    
    def play(self):
        if self._use_pygame:
            self._play_mp3()
        elif HAS_WINSOUND:
            self._play_winsound()
        else:
            self._play_console_beeps()
            
    def _play_mp3(self):
        try:
            import pygame
            pygame.mixer.music.play(loops=-1)
            
            end_time = time.time() + self.duration_seconds
            while time.time() < end_time and not self._stop_flag.is_set():
                time.sleep(0.1)
                
            pygame.mixer.music.stop()
            
        except Exception:
            if HAS_WINSOUND:
                self._play_winsound()

    def _play_winsound(self):
        end_time = time.time() + self.duration_seconds
        freq_index = 0
        
        while time.time() < end_time and not self._stop_flag.is_set():
            freq = self.ALARM_FREQUENCIES[freq_index % len(self.ALARM_FREQUENCIES)]
            winsound.Beep(freq, self.TONE_DURATION_MS)
            freq_index += 1
    
    def _play_console_beeps(self):
        end_time = time.time() + self.duration_seconds
        while time.time() < end_time and not self._stop_flag.is_set():
            print("\a", end="", flush=True)
            time.sleep(0.5)
    
    def play_async(self):
        if self._is_playing:
            return
        
        self._stop_flag.clear()
        self._is_playing = True
        self._play_thread = threading.Thread(target=self._async_play_worker, daemon=True)
        self._play_thread.start()
    
    def _async_play_worker(self):
        try:
            self.play()
        finally:
            self._is_playing = False
    
    def stop(self):
        self._stop_flag.set()
        if self._use_pygame:
            try:
                import pygame
                pygame.mixer.music.stop()
            except:
                pass
                
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1)
        self._is_playing = False
    
    @property
    def is_playing(self) -> bool:
        return self._is_playing


class EmailAlert:
    def __init__(self, config: AlertConfig):
        self.config = config
        self._last_send_time = 0
        self._min_interval = 60
    
    def send_email(self, subject: str, body: str, to_email: str) -> bool:
        smtp_host = self.config.get("smtp_host")
        smtp_port = self.config.get("smtp_port")
        smtp_user = self.config.get("smtp_username")
        smtp_pass = self.config.get("smtp_password")
        from_email = self.config.get("email_from")
        
        if not all([smtp_host, smtp_user, smtp_pass, to_email]):
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            
            return True
            
        except Exception:
            return False
    
    def send_alert(self, detection_data: dict, location_info: str) -> bool:
        if time.time() - self._last_send_time < self._min_interval:
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_score = detection_data.get("risk", 0)
        
        message = f"""
{self.config.get("alert_message")}

Time: {timestamp}
Confidence: {int(risk_score * 100)}%
{location_info}
        """.strip()
        
        if self.config.get("use_email", True):
            email = self.config.get("emergency_contact_email")
            if email:
                success = self.send_email(
                    subject="SEIZURE ALERT - Immediate Attention Required",
                    body=message,
                    to_email=email
                )
                
                if success:
                    self._last_send_time = time.time()
                return success
        
        return False
    
    def send_async(self, detection_data: dict, location_info: str):
        thread = threading.Thread(
            target=self.send_alert,
            args=(detection_data, location_info),
            daemon=True
        )
        thread.start()


class AlertLogger:
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "alerts.json")
    
    def log_alert(self, alert_type: str, detection_data: dict, location: dict, success: bool):
        event = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "success": success,
            "confidence_score": detection_data.get("risk", 0),
            "location": location,
            "detection_status": detection_data.get("status", "UNKNOWN")
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass


class AlertManager:
    def __init__(self, config_path: str = "data/alert_config.json", cooldown_seconds: int = 30, audio_duration: int = 10):
        self.config = AlertConfig(config_path)
        self.audio = AudioAlert(duration_seconds=audio_duration)
        self.location = LocationService()
        self.email_alert = EmailAlert(self.config)
        self.logger = AlertLogger()
        
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time: float = 0
        self._alert_count: int = 0
    
    def can_trigger(self) -> bool:
        return (time.time() - self._last_alert_time) >= self.cooldown_seconds
    
    def trigger_alert(self, detection_data: dict) -> bool:
        if not self.can_trigger():
            return False
        
        self._last_alert_time = time.time()
        self._alert_count += 1
        
        print(f"\nALERT #{self._alert_count} TRIGGERED")
        
        location = self.location.get_location()
        location_str = self.location.format_for_message()
        
        self.audio.play_async()
        self.logger.log_alert("audio", detection_data, location, True)
        
        if self.config.get("emergency_contact_email"):
            self.email_alert.send_async(detection_data, location_str)
            self.logger.log_alert("email", detection_data, location, True)
        
        return True
