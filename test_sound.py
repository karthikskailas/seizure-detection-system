import os
import time

print("Testing Audio System...")

file_path = "data/alert_sound.mp3"
abs_path = os.path.abspath(file_path)

print(f"Looking for: {abs_path}")

if not os.path.exists(file_path):
    print("❌ File NOT found!")
    exit(1)

print("✅ File found.")

try:
    import pygame
    print(f"✅ Pygame version: {pygame.ver}")
    
    pygame.mixer.init()
    print("✅ Mixer initialized")
    
    pygame.mixer.music.load(abs_path)
    print("✅ Music loaded")
    
    print("▶ Playing...")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
        
    print("✅ Playback finished")
    
except ImportError:
    print("❌ Pygame NOT installed!")
except Exception as e:
    print(f"❌ Error: {e}")
