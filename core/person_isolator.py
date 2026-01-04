# core/foreground_isolator.py
"""
Foreground Person Isolation Pipeline
-------------------------------------
CPU-friendly, real-time OpenCV pipeline that isolates the foreground person
(closest to camera) from a fixed camera feed, ignoring background people/motion.

Uses MOG2 background subtraction + largest contour selection with temporal stability.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import config


class ForegroundIsolator:
    """
    Isolates the foreground person from a fixed camera feed.
    
    Key assumption: The closest person to the camera will have the largest
    contour due to perspective geometry. This is ideal for patient monitoring
    where the patient (in bed, closest to camera) should be tracked.
    """
    
    def __init__(self):
        # MOG2 Background Subtractor
        # - detectShadows=False: Shadows cause false positive contours
        # - history: How many frames to model background (higher = more stable)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.FG_HISTORY,
            varThreshold=config.FG_VAR_THRESHOLD,
            detectShadows=False
        )
        
        # Morphological kernels (pre-computed for efficiency)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Temporal tracking state
        self.prev_bbox = None          # Last known bounding box (x, y, w, h)
        self.lost_frames = 0           # Counter for frames without valid target
        self.stable_bbox = None        # Smoothed bounding box for output
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for background subtraction.
        
        Why grayscale: Reduces computation by 3x (1 channel vs 3).
        Why blur: Gaussian smoothing reduces sensor noise that would
        otherwise create tiny false-positive motion blobs.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def _apply_background_subtraction(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Extract foreground mask using MOG2.
        
        MOG2 models each pixel as a mixture of Gaussians. Pixels that deviate
        significantly from the model are marked as foreground (moving objects).
        """
        fg_mask = self.bg_subtractor.apply(preprocessed)
        return fg_mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove noise and connect fragmented regions.
        
        OPEN (erode then dilate): Removes small white noise blobs while
        preserving larger connected regions. Essential for removing
        background flicker and small movements.
        
        DILATE: Expands remaining regions to reconnect body parts that
        may have been separated (e.g., arm from torso).
        """
        # Remove small noise blobs
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        # Connect fragmented body parts
        cleaned = cv2.dilate(cleaned, self.kernel_dilate, iterations=2)
        return cleaned
    
    def _extract_contours(self, mask: np.ndarray, frame_area: int) -> list:
        """
        Find external contours and filter by minimum area.
        
        RETR_EXTERNAL: Only outer contours (ignores holes inside a person).
        Area filter: Rejects contours smaller than FG_MIN_AREA_RATIO of frame.
        This eliminates background noise like swaying curtains or small pets.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        min_area = frame_area * config.FG_MIN_AREA_RATIO
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return valid_contours
    
    def _select_foreground_person(self, contours: list) -> Optional[Tuple[int, int, int, int]]:
        """
        Select the largest contour as the foreground person.
        
        Why largest = closest: Due to perspective projection, objects closer
        to the camera have larger projected areas. In a patient monitoring
        setup, the patient (in bed, near camera) will always be larger than
        any background person walking at the room's far end.
        """
        if not contours:
            return None
        
        # Find contour with maximum area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        IoU is a robust metric for tracking stability - if the new detection
        significantly overlaps with the previous one, it's likely the same person.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0  # No intersection
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _stabilize_tracking(self, current_bbox: Optional[Tuple]) -> Optional[Tuple[int, int, int, int]]:
        """
        Apply temporal stability to prevent abrupt bounding box switching.
        
        Problem: Without this, if two people briefly have similar sizes,
        the bounding box could rapidly switch between them, causing
        downstream motion analysis to produce garbage data.
        
        Solution: Track IoU with previous frame. If new detection overlaps
        significantly with previous, accept it. Otherwise, use memory to
        resist switching for FG_LOST_FRAMES_MAX frames.
        """
        if current_bbox is None:
            # No detection this frame
            self.lost_frames += 1
            if self.lost_frames > config.FG_LOST_FRAMES_MAX:
                # Target truly lost - reset tracking
                self.prev_bbox = None
                self.stable_bbox = None
                return None
            # Return last known position during brief occlusions
            return self.stable_bbox
        
        if self.prev_bbox is None:
            # First detection - accept immediately
            self.prev_bbox = current_bbox
            self.stable_bbox = current_bbox
            self.lost_frames = 0
            return current_bbox
        
        # Check if new detection is consistent with previous
        iou = self._calculate_iou(current_bbox, self.prev_bbox)
        
        if iou >= config.FG_STABILITY_THRESHOLD:
            # Same person - update tracking
            self.prev_bbox = current_bbox
            self.stable_bbox = current_bbox
            self.lost_frames = 0
            return current_bbox
        else:
            # Possible different person or noise
            # Check area similarity as secondary validation
            prev_area = self.prev_bbox[2] * self.prev_bbox[3]
            curr_area = current_bbox[2] * current_bbox[3]
            area_ratio = min(prev_area, curr_area) / max(prev_area, curr_area)
            
            if area_ratio > 0.5:
                # Similar size - might be same person who moved quickly
                self.prev_bbox = current_bbox
                self.stable_bbox = current_bbox
                self.lost_frames = 0
                return current_bbox
            else:
                # Likely a different person or noise - resist switching
                self.lost_frames += 1
                if self.lost_frames > config.FG_LOST_FRAMES_MAX:
                    # Previous target truly gone - switch to new
                    self.prev_bbox = current_bbox
                    self.stable_bbox = current_bbox
                    self.lost_frames = 0
                    return current_bbox
                return self.stable_bbox
    
    def get_foreground_roi(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Main API: Extract the foreground person's bounding box and cropped frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            bbox: (x, y, w, h) bounding box or None if no person detected
            cropped: Cropped BGR image of the person or None
        """
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        
        # Pipeline execution
        preprocessed = self.preprocess_frame(frame)
        fg_mask = self._apply_background_subtraction(preprocessed)
        clean_mask = self._clean_mask(fg_mask)
        contours = self._extract_contours(clean_mask, frame_area)
        raw_bbox = self._select_foreground_person(contours)
        stable_bbox = self._stabilize_tracking(raw_bbox)
        
        if stable_bbox is None:
            return None, None
        
        x, y, w, h = stable_bbox
        cropped = frame[y:y+h, x:x+w]
        
        return stable_bbox, cropped
    
    def draw_foreground_box(self, frame: np.ndarray, bbox: Optional[Tuple], 
                            label: str = "Foreground Patient") -> np.ndarray:
        """
        Draw bounding box with label on frame for visualization.
        
        Args:
            frame: BGR image to draw on (will be modified in-place)
            bbox: (x, y, w, h) bounding box or None
            label: Text label to display
            
        Returns:
            frame: The same frame with box drawn
        """
        if bbox is None:
            return frame
        
        x, y, w, h = bbox
        
        # Green box for visibility
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Label with background for readability
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x + 5, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
