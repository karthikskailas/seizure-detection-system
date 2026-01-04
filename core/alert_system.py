# core/alert_system.py
"""
Alert System for Seizure Detection
-----------------------------------
Provides multi-channel alerting when seizures are detected:
1. Local Audio Alarm (offline, non-blocking)
2. WhatsApp Alert via Twilio (with location)
3. Email Fallback via SMTP/Mailtrap

This is an assistive alert tool, not a medical diagnostic system.
"""

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

# Windows-specific audio
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Cross-platform audio fallback
try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================

class AlertConfig:
    """
    Manages alert configuration stored in JSON file.
    Allows runtime editing of emergency contacts and preferences.
    """
    
    DEFAULT_CONFIG = {
        "emergency_contact_email": "",
        "use_email": True,
        "alert_message": "⚠️ ALERT: Possible seizure detected!",
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
        """Create default config file if it doesn't exist."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2)
    
    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**self.DEFAULT_CONFIG, **loaded}
        except (json.JSONDecodeError, FileNotFoundError):
            return self.DEFAULT_CONFIG.copy()
    
    def save(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value and persist."""
        self.config[key] = value
        self.save()


# =============================================================================
# LOCATION SERVICE
# =============================================================================

class LocationService:
    """
    Fetches location information for alerts.
    Uses IP-based geolocation (free API, no GPS required).
    """
    
    # Free IP geolocation API (50K requests/month)
    IPINFO_URL = "https://ipinfo.io/json"
    
    def __init__(self):
        self._cached_location: Optional[Dict] = None
        self._cache_time: float = 0
        self._cache_duration: float = 300  # Cache for 5 minutes
    
    def get_location(self) -> Dict[str, Any]:
        """
        Get current location (IP-based).
        Returns dict with city, region, country, coordinates.
        """
        # Return cached location if still valid
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
                "source": "IP-based (approximate)",
                "timestamp": datetime.now().isoformat()
            }
            
            self._cached_location = location
            self._cache_time = time.time()
            return location
            
        except Exception as e:
            return {
                "city": "Unknown",
                "region": "Unknown",
                "country": "Unknown",
                "coordinates": "Unknown",
                "source": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def format_for_message(self) -> str:
        """Format location for alert message."""
        loc = self.get_location()
        if loc["city"] != "Unknown":
            return f"📍 Location: {loc['city']}, {loc['region']}, {loc['country']} ({loc['source']})"
        return "📍 Location: Unable to determine"


# =============================================================================
# AUDIO ALERT
# =============================================================================

class AudioAlert:
    """
    Plays loud audible alarm when seizure is detected.
    Non-blocking implementation using threading.
    Works offline (no internet required).
    """
    
    # Frequencies for alarm sound (Hz)
    ALARM_FREQUENCIES = [880, 1100, 880, 1100]  # Alternating tones
    TONE_DURATION_MS = 300  # Duration per tone
    
    def __init__(self, duration_seconds: int = 5):
        self.duration_seconds = duration_seconds
        self._is_playing = False
        self._stop_flag = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
    
    def play(self):
        """Play alarm sound (blocking)."""
        if not HAS_WINSOUND:
            print("[AUDIO] winsound not available, using console beeps")
            self._play_console_beeps()
            return
        
        end_time = time.time() + self.duration_seconds
        freq_index = 0
        
        while time.time() < end_time and not self._stop_flag.is_set():
            freq = self.ALARM_FREQUENCIES[freq_index % len(self.ALARM_FREQUENCIES)]
            winsound.Beep(freq, self.TONE_DURATION_MS)
            freq_index += 1
    
    def _play_console_beeps(self):
        """Fallback: console beeps."""
        end_time = time.time() + self.duration_seconds
        while time.time() < end_time and not self._stop_flag.is_set():
            print("\a", end="", flush=True)
            time.sleep(0.5)
    
    def play_async(self):
        """Play alarm sound in background thread (non-blocking)."""
        if self._is_playing:
            return  # Already playing
        
        self._stop_flag.clear()
        self._is_playing = True
        self._play_thread = threading.Thread(target=self._async_play_worker, daemon=True)
        self._play_thread.start()
    
    def _async_play_worker(self):
        """Worker thread for async playback."""
        try:
            self.play()
        finally:
            self._is_playing = False
    
    def stop(self):
        """Stop the alarm."""
        self._stop_flag.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1)
        self._is_playing = False
    
    @property
    def is_playing(self) -> bool:
        return self._is_playing


# =============================================================================
# EMAIL ALERT (Mailtrap SMTP)
# =============================================================================

class EmailAlert:
    """
    Sends email notifications via SMTP (Mailtrap).
    Simple, reliable alerting without complex API integrations.
    """
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self._last_send_time = 0
        self._min_interval = 30  # Minimum seconds between alerts
    
    def send_email(self, subject: str, body: str, to_email: str) -> bool:
        """
        Send email via SMTP (Mailtrap or other provider).
        
        Returns True if successful, False otherwise.
        """
        smtp_host = self.config.get("smtp_host")
        smtp_port = self.config.get("smtp_port")
        smtp_user = self.config.get("smtp_username")
        smtp_pass = self.config.get("smtp_password")
        from_email = self.config.get("email_from")
        
        if not all([smtp_host, smtp_user, smtp_pass, to_email]):
            print("[EMAIL] Missing SMTP credentials or recipient email")
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
            
            print(f"[EMAIL] Message sent successfully to {to_email}")
            return True
            
        except Exception as e:
            print(f"[EMAIL] Error: {str(e)}")
            return False
    
    def send_alert(self, detection_data: dict, location_info: str) -> bool:
        """
        Send alert via WhatsApp (primary) with Email fallback.
        
        Returns True if any notification was sent successfully.
        """
        # Rate limiting
        if time.time() - self._last_send_time < self._min_interval:
            print("[EMAIL] Rate limited - skipping alert")
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_score = detection_data.get("risk", 0)
        
        # Compose email body
        message = f"""
{self.config.get("alert_message")}

🕐 Time: {timestamp}
📊 Confidence: {int(risk_score * 100)}%
{location_info}

This is an automated alert from the Seizure Detection System.
Please check on the patient immediately.
        """.strip()
        
        # Send email alert
        if self.config.get("use_email", True):
            email = self.config.get("emergency_contact_email")
            if email:
                success = self.send_email(
                    subject="🚨 SEIZURE ALERT - Immediate Attention Required",
                    body=message,
                    to_email=email
                )
                
                if success:
                    self._last_send_time = time.time()
                return success
        
        print("[EMAIL] No email configured or email alerts disabled")
        return False
    
    def send_async(self, detection_data: dict, location_info: str):
        """Send alert in background thread (non-blocking)."""
        thread = threading.Thread(
            target=self.send_alert,
            args=(detection_data, location_info),
            daemon=True
        )
        thread.start()


# =============================================================================
# ALERT LOGGER
# =============================================================================

class AlertLogger:
    """
    Logs all alert events for audit trail.
    Separate from EventLogger to track alert-specific data.
    """
    
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "alerts.json")
    
    def log_alert(self, alert_type: str, detection_data: dict, location: dict, success: bool):
        """Log an alert event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,  # "audio", "whatsapp", "email"
            "success": success,
            "confidence_score": detection_data.get("risk", 0),
            "location": location,
            "detection_status": detection_data.get("status", "UNKNOWN")
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[ALERT_LOG] Error writing log: {e}")


# =============================================================================
# ALERT MANAGER (Main Orchestrator)
# =============================================================================

class AlertManager:
    """
    Main alert system orchestrator.
    
    Manages all alert channels and ensures:
    - Alerts trigger only once per seizure event (cooldown)
    - Audio alerts are non-blocking
    - Email alerts via Mailtrap
    - All events are logged for audit
    """
    
    def __init__(self, config_path: str = "data/alert_config.json", cooldown_seconds: int = 30):
        self.config = AlertConfig(config_path)
        self.audio = AudioAlert(duration_seconds=5)
        self.location = LocationService()
        self.email_alert = EmailAlert(self.config)
        self.logger = AlertLogger()
        
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time: float = 0
        self._alert_count: int = 0
    
    def can_trigger(self) -> bool:
        """Check if enough time has passed since last alert."""
        return (time.time() - self._last_alert_time) >= self.cooldown_seconds
    
    def trigger_alert(self, detection_data: dict) -> bool:
        """
        Trigger all configured alerts.
        
        This is the main entry point called from main.py when seizure is detected.
        
        Args:
            detection_data: Dict containing 'risk', 'status', 'counter' from DecisionEngine
            
        Returns:
            True if alerts were triggered, False if cooldown prevented triggering
        """
        if not self.can_trigger():
            remaining = self.cooldown_seconds - (time.time() - self._last_alert_time)
            print(f"[ALERT] Cooldown active ({remaining:.0f}s remaining)")
            return False
        
        self._last_alert_time = time.time()
        self._alert_count += 1
        
        print(f"\n{'='*50}")
        print(f"  🚨 ALERT #{self._alert_count} TRIGGERED")
        print(f"{'='*50}")
        
        # Get location info
        location = self.location.get_location()
        location_str = self.location.format_for_message()
        
        # 1. Audio Alert (non-blocking)
        print("[ALERT] Starting audio alarm...")
        self.audio.play_async()
        self.logger.log_alert("audio", detection_data, location, True)
        
        # 2. Email Alert via Mailtrap (non-blocking)
        if self.config.get("emergency_contact_email"):
            print("[ALERT] Sending email notification...")
            self.email_alert.send_async(detection_data, location_str)
            self.logger.log_alert("email", detection_data, location, True)
        else:
            print("[ALERT] No email configured - skipping email alert")
        
        return True
    
    def stop_audio(self):
        """Manually stop the audio alarm."""
        self.audio.stop()
    
    def reset(self):
        """Reset alert state (useful for testing)."""
        self._last_alert_time = 0
        self._alert_count = 0
        self.audio.stop()
    
    def get_status(self) -> dict:
        """Get current alert system status."""
        return {
            "alerts_triggered": self._alert_count,
            "cooldown_active": not self.can_trigger(),
            "audio_playing": self.audio.is_playing,
            "last_alert_time": datetime.fromtimestamp(self._last_alert_time).isoformat() if self._last_alert_time else None,
            "email_configured": bool(self.config.get("emergency_contact_email")),
        }


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  ALERT SYSTEM TEST")
    print("=" * 50)
    
    # Initialize
    manager = AlertManager()
    
    # Show status
    print("\nSystem Status:")
    for key, value in manager.get_status().items():
        print(f"  {key}: {value}")
    
    # Test audio
    print("\n[TEST] Playing audio alarm for 3 seconds...")
    manager.audio.duration_seconds = 3
    manager.audio.play()
    
    # Test location
    print("\n[TEST] Fetching location...")
    print(manager.location.format_for_message())
    
    print("\n[TEST] Complete!")
