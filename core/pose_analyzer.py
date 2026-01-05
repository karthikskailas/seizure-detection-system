# core/pose_analyzer.py
import cv2
import numpy as np
import config
import os
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

class Person:
    def __init__(self, person_id):
        self.id = person_id
        self.landmarks = None
        self.world_landmarks = None
        self.segmentation_mask = None
        self.bbox = None
        self.centroid = None
        
        # Scoring History
        self.velocity_buffer = deque(maxlen=30)
        self.limb_positions = deque(maxlen=30) # For tremor detection
        
        # Scores
        self.susceptibility_score = 0.0
        self.tremor_score = 0.0
        self.is_horizontal = False
        self.is_rigid = False
        
        # State
        self.last_seen = time.time()

    def update(self, landmarks, world_landmarks, segmentation_mask, frame_shape):
        self.landmarks = landmarks
        self.world_landmarks = world_landmarks
        self.segmentation_mask = segmentation_mask
        self.last_seen = time.time()
        
        # Calculate BBox and Centroid
        h, w = frame_shape[:2]
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        self.bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        self.centroid = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
        
        self._analyze_motion()

    def _analyze_motion(self):
        # 1. Posture (Is Horizontal?)
        # Use simple logic: Shoulder-Hip line vs Vertical
        # 11: L_Shoulder, 12: R_Shoulder, 23: L_Hip, 24: R_Hip
        l_s = self.landmarks[11]
        r_s = self.landmarks[12]
        l_h = self.landmarks[23]
        r_h = self.landmarks[24]
        
        mid_shoulder_y = (l_s.y + r_s.y) / 2
        mid_hip_y = (l_h.y + r_h.y) / 2
        
        # If head/shoulders are roughly at same Y as hips (or lower?), considering y increases downwards
        # Lying down: Vertical difference is small compared to horizontal
        # Vertical alignment
        vertical_diff = abs(mid_shoulder_y - mid_hip_y)
        self.is_horizontal = vertical_diff < 0.2  # Threshold for "lying down" roughly
        
        # 2. Tremor (High Frequency in peripherals)
        # Track wrist motion
        # 15: L_Wrist, 16: R_Wrist
        l_wrist = self.landmarks[15]
        
        self.limb_positions.append(l_wrist.y)
        if len(self.limb_positions) > 10:
            # Simple variance/crossing count for "shaking"
            # High frequency changes
            pos = np.array(self.limb_positions)
            diffs = np.diff(pos)
            zero_crossings = np.where(np.diff(np.sign(diffs)))[0]
            
            # Tremor if many direction changes
            freq_score = len(zero_crossings) / len(pos) # 0.0 to 1.0 (approximated)
            self.tremor_score = freq_score * 5.0 # Scale to 0-5
            
    def calculate_susceptibility(self, all_people):
        score = 0.0
        
        # +3 Posture
        if self.is_horizontal:
            score += 3.0
            
        # +4 Tremor
        score += min(4.0, self.tremor_score)
        
        # +1 Vertical Position (Lowest in frame)
        # Check if I am the lowest (highest Y value of COG)
        sorted_by_y = sorted(all_people, key=lambda p: p.centroid[1], reverse=True)
        if sorted_by_y and sorted_by_y[0].id == self.id:
            score += 1.0
            
        self.susceptibility_score = score
        return score


class PoseAnalyzer:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker.task")
    
    def __init__(self):
        self._ensure_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        self.people = {} # id -> Person
        self.next_id = 1
        
    def _ensure_model_exists(self):
        if os.path.exists(self.MODEL_PATH):
            return
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        print("Downloading pose landmarker model...")
        response = requests.get(self.MODEL_URL)
        response.raise_for_status()
        with open(self.MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")

    def _match_persons(self, new_landmarks_list, frame_shape):
        # Existing People Centroids
        existing_ids = list(self.people.keys())
        active_ids = []
        
        h, w = frame_shape[:2]
        
        for idx, landmarks in enumerate(new_landmarks_list):
            # Calculate centroid for new detection
            # Just use Hip center for tracking
            l_hip = landmarks[23]
            r_hip = landmarks[24]
            cx = (l_hip.x + r_hip.x) / 2 * w
            cy = (l_hip.y + r_hip.y) / 2 * h
            
            matched_id = None
            min_dist = 100.0 # Pixel distance threshold
            
            # Simple Euclidean Match
            for pid in existing_ids:
                if pid in active_ids: continue # Already matched
                
                person = self.people[pid]
                if person.centroid:
                    dist = np.sqrt((person.centroid[0] - cx)**2 + (person.centroid[1] - cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = pid
            
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                self.people[matched_id] = Person(matched_id)
            
            active_ids.append(matched_id)
            
            # Return mapping index -> person_id
            yield idx, matched_id, landmarks

        # Clean up stale people
        current_time = time.time()
        stale_ids = [pid for pid, p in self.people.items() if current_time - p.last_seen > 2.0]
        for pid in stale_ids:
            del self.people[pid]

    def analyze(self, frame):
        # Alias for main compatibility (detect_fall was old name)
        # Returns: is_fallen (bool), results (object with landmarks)
        # But we need to return more rich data now.
        # We will return the SELECTED person's data to fit the system.
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)
        
        if not results.pose_landmarks:
            return False, None
            
        # Match detections to tracking IDs
        patient_candidate = None
        max_score = -1.0
        
        # Temporary mask holder
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        active_people = []
        
        for i, (idx, pid, landmarks) in enumerate(self._match_persons(results.pose_landmarks, frame.shape)):
            person = self.people[pid]
            
            # Extract mask if available
            seg_mask = None
            if results.segmentation_masks and len(results.segmentation_masks) > idx:
                # MP returns float mask, convert to binary
                mask_float = results.segmentation_masks[idx].numpy_view()
                seg_mask = (mask_float > 0.5).astype(np.uint8) * 255
            
            person.update(landmarks, results.pose_world_landmarks[idx], seg_mask, frame.shape)
            active_people.append(person)

        # Calculate Scores
        for person in active_people:
            score = person.calculate_susceptibility(active_people)
            if score > max_score:
                max_score = score
                patient_candidate = person
                
        # Filter: If score is too low, maybe nobody is the patient?
        # Requirement: "If highest score < 2.0... return None"
        is_seizure_suspect = False
        target_person = None
        
        if max_score >= 2.0:
            target_person = patient_candidate
            is_seizure_suspect = True
        elif active_people:
             # Fallback: Just track the one with highest score anyway for UI, but mark low confidence
             target_person = patient_candidate

        # Create Debug Image
        debug_image = frame.copy()
        patient_mask_visual = np.zeros_like(frame)

        for person in active_people:
            is_target = (target_person and person.id == target_person.id and is_seizure_suspect)
            
            color = (0, 0, 255) if is_target else (0, 255, 0) # Red for patient, Green for helper
            label = f"ID:{person.id} Sc:{person.susceptibility_score:.1f}"
            if is_target: label += " PATIENT"
            
            if person.bbox:
                x, y, w, h = person.bbox
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw scores details
                if is_target:
                    cv2.putText(debug_image, f"Tremor: {person.tremor_score:.1f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    cv2.putText(debug_image, f"Horiz: {person.is_horizontal}", (x, y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            # Add to mask output if target
            if is_target and person.segmentation_mask is not None:
                # Resize mask to frame size if needed? MP masks usually match aspect ratio but check docs
                # Usually MP masks are same size as input if Image mode?
                # Actually MP masks might be smaller. Resize to be safe.
                if person.segmentation_mask.shape != frame.shape[:2]:
                    m = cv2.resize(person.segmentation_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    patient_mask_visual[m > 128] = (255, 255, 255)
                else:
                    patient_mask_visual[person.segmentation_mask > 128] = (255, 255, 255)

        # Return format expected by main.py but augmented
        # We need to return an object that mimics the old `results` but only contains the TARGET
        # OR just return the data main needs.
        # Main calls: is_fallen, pose_results = pose_engine.detect_fall(frame)
        # It expects `pose_results.pose_landmarks` to exist (list of landmarks)
        
        class MockResults:
            def __init__(self, landmarks, mask_img, debug_img):
                self.pose_landmarks = [landmarks] if landmarks else [] # Wrapper list as Main expects results.pose_landmarks[0] usually or iterates
                self.mask_visual = mask_img
                self.debug_image = debug_img
        
        # If no target found, return empty results
        final_landmarks = target_person.landmarks if target_person else None
        
        # Determine "is_fallen" based on the target person
        is_fallen = target_person.is_horizontal if target_person else False
        
        return is_fallen, MockResults(final_landmarks, patient_mask_visual, debug_image)

    # Maintain backward compatibility alias
    def detect_fall(self, frame):
        return self.analyze(frame)

    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()