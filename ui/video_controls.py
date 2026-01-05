"""Video playback control panel for OpenCV windows.

Provides interactive play, pause, and stop buttons directly in the video window.
"""

import cv2
import numpy as np


class VideoControlPanel:
    """Interactive control panel for video playback with play/pause/stop buttons."""
    
    def __init__(self, window_name='Seizure Detection'):
        """Initialize the control panel.
        
        Args:
            window_name: Name of the OpenCV window to attach controls to
        """
        self.window_name = window_name
        self.state = 'playing'  # 'playing', 'paused', or 'stopped'
        self.buttons = {}
        self.hovered_button = None
        self.panel_height = 60
        
        # Button colors
        self.btn_color = (60, 60, 60)
        self.btn_hover_color = (80, 80, 80)
        self.btn_active_color = (100, 200, 100)
        self.text_color = (255, 255, 255)
        
    def setup_mouse_callback(self):
        """Set up mouse callback for the window."""
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events.
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_MOUSEMOVE:
            # Check if mouse is hovering over any button
            self.hovered_button = None
            for btn_name, btn_rect in self.buttons.items():
                if self._is_point_in_rect((x, y), btn_rect):
                    self.hovered_button = btn_name
                    break
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if any button was clicked
            for btn_name, btn_rect in self.buttons.items():
                if self._is_point_in_rect((x, y), btn_rect):
                    self._handle_button_click(btn_name)
                    break
    
    def _is_point_in_rect(self, point, rect):
        """Check if a point is inside a rectangle.
        
        Args:
            point: (x, y) tuple
            rect: (x, y, width, height) tuple
            
        Returns:
            True if point is inside rectangle
        """
        px, py = point
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh
    
    def _handle_button_click(self, button_name):
        """Handle button click events.
        
        Args:
            button_name: Name of the clicked button
        """
        if button_name == 'play':
            self.state = 'playing'
        elif button_name == 'pause':
            self.state = 'paused'
        elif button_name == 'stop':
            self.state = 'stopped'
    
    def draw_controls(self, frame):
        """Draw control panel on the frame.
        
        Args:
            frame: Video frame to draw controls on
            
        Returns:
            Frame with controls drawn
        """
        height, width = frame.shape[:2]
        
        # Create a copy to draw on
        display = frame.copy()
        
        # Draw semi-transparent panel background at the bottom
        panel_y = height - self.panel_height
        overlay = display.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Calculate button positions
        btn_width = 80
        btn_height = 40
        btn_spacing = 20
        total_width = (btn_width * 3) + (btn_spacing * 2)
        start_x = (width - total_width) // 2
        btn_y = panel_y + (self.panel_height - btn_height) // 2
        
        # Define buttons
        buttons_info = [
            ('play', 'PLAY', start_x),
            ('pause', 'PAUSE', start_x + btn_width + btn_spacing),
            ('stop', 'STOP', start_x + (btn_width + btn_spacing) * 2)
        ]
        
        # Draw each button
        for btn_name, btn_text, btn_x in buttons_info:
            # Store button rectangle for click detection
            self.buttons[btn_name] = (btn_x, btn_y, btn_width, btn_height)
            
            # Determine button color
            if self.state == btn_name:
                color = self.btn_active_color
            elif self.hovered_button == btn_name:
                color = self.btn_hover_color
            else:
                color = self.btn_color
            
            # Draw button background
            cv2.rectangle(display, (btn_x, btn_y), 
                         (btn_x + btn_width, btn_y + btn_height), 
                         color, -1)
            
            # Draw button border
            border_color = (150, 150, 150) if self.hovered_button == btn_name else (100, 100, 100)
            cv2.rectangle(display, (btn_x, btn_y), 
                         (btn_x + btn_width, btn_y + btn_height), 
                         border_color, 2)
            
            # Draw button text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(btn_text, font, font_scale, thickness)[0]
            text_x = btn_x + (btn_width - text_size[0]) // 2
            text_y = btn_y + (btn_height + text_size[1]) // 2
            
            cv2.putText(display, btn_text, (text_x, text_y), 
                       font, font_scale, self.text_color, thickness, cv2.LINE_AA)
        
        # Draw status indicator
        status_text = f"Status: {self.state.upper()}"
        cv2.putText(display, status_text, (20, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return display
    
    def is_playing(self):
        """Check if video is currently playing.
        
        Returns:
            True if video should be playing
        """
        return self.state == 'playing'
    
    def is_paused(self):
        """Check if video is currently paused.
        
        Returns:
            True if video is paused
        """
        return self.state == 'paused'
    
    def is_stopped(self):
        """Check if video should stop.
        
        Returns:
            True if video should stop
        """
        return self.state == 'stopped'
    
    def get_state(self):
        """Get current playback state.
        
        Returns:
            Current state string ('playing', 'paused', or 'stopped')
        """
        return self.state
