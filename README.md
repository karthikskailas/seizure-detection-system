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

## Key Features

* **Rhythmic Motion Detection:** Uses Fast Fourier Transform (FFT) to identify 3–15 Hz oscillatory movements.
* **Posture & Fall Analysis:** Real-time monitoring of centroid displacement and bounding box aspect ratios.
* **CPU-Optimized Pipeline:** Optimized OpenCV and NumPy implementations to ensure high FPS without a dedicated GPU.
* **Automated Logging:** Saves event timestamps and motion metadata for post-event clinical review.

---

## Technical Implementation Deep-Dive

### 1. Motion Signal Extraction (Optical Flow)
Instead of simple frame differencing, which is prone to noise from lighting changes, the system employs **Farnebäck Dense Optical Flow**. 
* **Process:** It computes a displacement vector for every pixel between consecutive frames $I(t)$ and $I(t+1)$.
* **Aggregation:** The magnitude of these vectors is averaged across the detected person's region of interest (ROI) to create a single scalar value per frame, resulting in a **Temporal Motion Signal**.



### 2. Spectral Analysis (FFT)
To distinguish between random movement and a seizure, the system analyzes the frequency of the motion signal using a **Fast Fourier Transform (FFT)**.
* **Windowing:** We use a sliding window (e.g., 2–4 seconds of video) to capture rhythmic patterns.
* **Frequency Range:** Research indicates that clonic seizures typically exhibit movements between **3 Hz and 15 Hz**. 
* **Detection Logic:** If the Power Spectral Density (PSD) shows a dominant peak within this range that exceeds the `ENERGY_THRESHOLD`, a convulsion flag is raised.



### 3. Pose & Velocity Tracking
While the FFT detects "shaking," the **Pose Analyzer** tracks the physical state of the subject using MediaPipe.
* **Centroid Velocity:** We track the $(x, y)$ coordinates of the body's center of mass. A sudden high-velocity spike followed by a prolonged period of zero movement is a primary indicator of a **Fall Event**.
* **Aspect Ratio Monitoring:** By calculating the bounding box of the pose landmarks, the system detects a "posture collapse" (e.g., a transition from a vertical ratio of 1:3 to a horizontal ratio of 3:1).

### 4. Multi-Symptom Decision Engine
To ensure high precision and low false-alarm rates, the system uses a **Weighted Confidence Scorer**. An alert is only triggered if a combination of conditions is met over a persistent time buffer ($T_{persistence}$):

| Symptom | Detection Method | Weight |
| :--- | :--- | :--- |
| **Rhythmic Shaking** | FFT (3-15 Hz range) | High |
| **Sudden Descent** | Vertical Velocity Spike | Medium |
| **Horizontal Posture** | Bounding Box Aspect Ratio | Medium |
| **Facial Distortion** | Face Landmarker Mesh Analysis | Low |

### 5. CPU Optimization Techniques
To maintain 30+ FPS on a standard laptop without a GPU, we implemented several optimization strategies:
* **Frame Skipping:** Background logic (like Pose Analysis) runs every 3rd frame, while Motion Extraction (FFT) runs on every frame to maintain signal integrity.
* **ROI Cropping:** Once a person is detected, the Optical Flow is calculated only within the person's bounding box rather than the full 1080p frame.
* **Vectorized NumPy Operations:** All signal processing is performed using vectorized arrays to avoid slow Python loops.
  

### Decision Engine 
To minimize false positives (e.g., someone simply brushing their teeth or waving), the system uses a **persistence buffer**. An alert is only triggered if the `ConvulsionConfidenceScore` remains above the threshold for a continuous period of $N$ frames.

---

## Getting Started 



### 1. Prerequisites
* Python 3.8+
*  Webcam 

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/karthikskailas/seizure-detection-system.git
cd seizure-detection

# Install dependencies
pip install -r requirements.txt
```


##  System Architecture Overview

The system follows a **multi-stage, symptom-based pipeline architecture**:
<img src="https://github.com/karthikskailas/demo-testing/blob/master/1767535608416-1158a4be-908b-4da4-8d57-ec9ff5a92a5a.png?raw=true" width="50%">

## Project Structure

The system is built with a modular, highly organized architecture to ensure production-ready quality, clear logic separation, and easy maintenance.

```text
seizure_detection_system/
├── core/                        # Primary logic and signal processing modules
│   ├── __init__.py              # Python package initializer
│   ├── alert_system.py          # Real-time notification and audio alert management
│   ├── decision_engine.py       # Multi-symptom logic and confidence scoring
│   ├── event_logger.py          # Structured logging for seizure events and metadata
│   ├── face_analyzer.py         # Face detection and facial movement analysis
│   ├── motion_analyzer.py       # Optical flow and motion signal extraction
│   ├── person_isolator.py       # ROI selection and human detection logic
│   ├── pose_analyzer.py         # Posture and fall detection analysis
│   ├── pose_velocity.py         # Velocity-based movement tracking
│   └── video_loader.py          # Frame acquisition and video stream handling
├── data/                        # Local storage for assets and output
│   ├── clips/                   # Saved video segments of detected events
│   └── logs/                    # System configurations and logging
│       ├── alert_config.json    # Thresholds and system parameter settings
│       └── alert_sound.mp3      # Audio file for real-time alerts
├── models/                      # Pre-trained AI models and task files
│   ├── face_landmarker.task     # MediaPipe model for facial landmarking
│   └── pose_landmarker.task     # MediaPipe model for body pose estimation
├── ui/                          # User interface and visualization
│   ├── overlay.py               # Real-time visual feedback and HUD on video feed
│   └── review_dashboard.py      # Interface for post-event clinical review
├── utils/                       # Performance tools and helper functions
│   ├── fps_controller.py        # Framerate synchronization for CPU-only systems
│   └── time_buffer.py           # Temporal data management for alert persistence
├── .gitignore                   # Files excluded from version control
├── calibrate.py                 # System calibration tool for specific environments
├── config.py                    # Global system constants and settings
├── main.py                      # Application entry point
├── README.md                    # Project documentation and roadmap
├── requirements.txt             # Project dependencies and libraries
└── test_sound.py                # Utility script to test alert audio output
```
---
### Signal Processing Outline
**Motion Signal Extraction**

- Consecutive video frames are processed using dense optical flow to measure pixel-level movement.

- The average motion magnitude per frame is computed, forming a time-series motion signal.
<img src="https://github.com/karthikskailas/demo-testing/blob/master/WhatsApp%20Image%202026-01-05%20at%201.38.37%20PM.jpeg?raw=true" width="50%">

**Frequency Band Evaluation**

- The system checks for dominant frequencies in the 3–15 Hz range, which corresponds to rhythmic convulsive movements typically seen in seizures.

**Convulsion Confidence Scoring**

- Based on motion energy and dominant frequency strength, a convulsion score is generated.

- Higher scores indicate sustained, rhythmic shaking.


**Decision Engine Input**

- The convulsion score is combined with other detected symptoms (e.g., fall detection) in the decision engine to trigger alerts only when risk persists over time.

Overall for the **signal processing**  we convert **video motion** into a **time-series signal** and use **FFT-based spectral analysis** to detect rhythmic convulsive movement patterns associated with seizure

---

## Conclusion

The **Real-Time Video-Based Seizure Detection System** represents a transition from a simple prototype to a robust, production-ready assistive tool. By leveraging **motion signal analysis and spectral features**, the system provides a non-invasive, cost-effective solution for seizure monitoring in unsupervised environments—such as homes and hospital wards—without the need for specialized wearable hardware.
<img src="https://github.com/karthikskailas/demo-testing/blob/master/WhatsApp%20Image%202026-01-05%20at%201.38.51%20PM.jpeg?raw=true" width="50%">

### Summary of Project Impact

* **Reliability through Multi-Symptom Analysis:** By combining spectral analysis of rhythmic movements with fall detection and posture collapse monitoring, the system significantly reduces false positives.
* **Performance Optimization:** The entire pipeline is engineered to run efficiently on **standard CPU-only systems**, ensuring high-performance monitoring is accessible even without expensive GPU hardware.
* **Maintainability and Scalability:** Adopting a **self-documenting code** approach—using descriptive names like `is_person_detected` and `motion_analyzer.py`—ensures that the system is easy to maintain and that knowledge transfer between team members is seamless.


### Final Word

As outlined in our documentation standards, a clear roadmap and structured logic separate a hobbyist effort from a production-ready solution. This system provides a **scalable, maintainable, and transparent framework** designed to enhance the safety of individuals prone to convulsive events while supporting the vital work of caregivers.

> **Disclaimer:** This system is an assistive alert tool designed to support caregivers and monitoring staff; it is not a medical diagnostic device.
