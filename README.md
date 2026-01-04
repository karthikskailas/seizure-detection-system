#  Real-Time Video-Based Seizure Detection System  
##  Problem Statement

Epileptic seizures and seizure-like convulsive events often occur suddenly and without warning, especially in unsupervised environments such as homes, hospital wards, hostels, and public areas monitored by CCTV cameras. Continuous manual monitoring of video feeds is impractical, and many existing solutions rely on wearable devices or computationally expensive AI models.

The objective of this project is to build a **real-time, video-based seizure detection system** that analyzes a **fixed camera feed** to identify **convulsive seizure-like motion patterns** using **motion signal analysis and spectral features**, while running efficiently on a **standard CPU-based laptop**.

This system is designed as an **assistive alert tool** to support caregivers and monitoring staff and is **not a medical diagnostic device**.


##  Objectives

- Detect seizure-like convulsive motion patterns from video
- Use spectral analysis to identify rhythmic movements
- Detect sudden falls or posture collapse
- Combine multiple symptoms to reduce false positives
- Generate real-time alerts
- Log structured event reports
- Provide a clean event review interface
- Ensure real-time performance on CPU-only systems



##  System Architecture Overview

The system follows a **multi-stage, symptom-based pipeline architecture**:
<img src="https://image2url.com/r2/default/images/1767535608416-1158a4be-908b-4da4-8d57-ec9ff5a92a5a.png" width="300" />

---

##  End-to-End Design Approach

###  Video Input & Preprocessing
- Accepts laptop webcam or pre-recorded video
- Frame resizing and FPS control for CPU efficiency
- Timestamping for event tracking

---

###  Symptom Detection Layer

#### A. Convulsive Motion Detection (Spectral Features)
- Dense Optical Flow is used to capture motion vectors
- Motion magnitude is computed per frame
- A sliding time window stores motion signals
- Fast Fourier Transform (FFT) is applied
- Dominant frequencies in the **3–15 Hz range** indicate rhythmic convulsions

**Output:**
- Motion intensity score
- Dominant frequency
- Convulsion confidence level

---

#### B. Fall / Collapse Detection (Pose-Based)
- Uses MediaPipe Pose (CPU-optimized)
- Tracks key landmarks (head, shoulders, hips)
- Detects:
  - Sudden vertical displacement
  - Change in body orientation
  - Aspect ratio changes (standing → lying)

**Output:**
- Fall detected (Yes / No)
- Fall velocity
- Posture state

---

###  Decision Engine (Symptom Fusion)
- Rule-based and fully explainable
- Combines multiple symptom scores
- Uses temporal smoothing to reduce false positives

**Decision Logic Example:**
