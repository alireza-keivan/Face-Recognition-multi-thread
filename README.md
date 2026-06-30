# Real-Time Face Recognition & Access Control System

> **An enterprise-grade AI-powered face recognition system for automated access control, real-time monitoring, and intelligent security management.**

![Hero Banner](https://github.com/alireza-keivan/Face-Recognition-multi-thread/blob/alireza-keivan/samples/Face-recognition-overview.png)

---

## Business Problem

Large organizations with high foot traffic require a reliable way to identify and verify individuals in real time while maintaining accurate logs and continuous security monitoring. Manual verification is slow, resource-intensive, and difficult to scale, especially across multiple entrances and surveillance points.

This project delivers an AI-powered face recognition and access control system that processes live video streams from IP cameras, detects and recognizes individuals from a distance, and publishes recognition events through MQTT. Authorized personnel can be automatically granted access, while every detection is logged for monitoring, auditing, and future analysis.

---

# Key Features

* Real-time face detection and recognition
* Multi-camera IP camera support
* Automatic access control
* Face tracking for improved performance
* MQTT-based communication
* Live dashboard visualization
* Recognition event logging
* Unknown person detection
* Easy configuration through JSON
* Modular and scalable architecture

---

# System Architecture

<p align="center">
<img src="https://github.com/alireza-keivan/Face-Recognition-multi-thread/blob/alireza-keivan/samples/face%20recognition%20architecture.png" width="900">
</p>

### Workflow

```
Milesight IP Cameras
        │
        ▼
Live Video Streams
        │
        ▼
Face Detection
        │
        ▼
Object Tracking
        │
        ▼
Face Recognition
        │
        ▼
Identity Verification
        │
        ▼
MQTT Publisher
        │
        ▼
HiveMQ Broker
      ┌──────────────┬──────────────┐
      ▼              ▼              ▼
Access Control   ThingsBoard   Recognition Logs
```

---

# Technology Stack

| Component        | Technology           |
| ---------------- | -------------------- |
| Language         | Python               |
| Computer Vision  | OpenCV               |
| Face Recognition | InsightFace          |
| Deep Learning    | ONNX Runtime         |
| Object Tracking  | KCF Tracker          |
| Messaging        | MQTT                 |
| MQTT Broker      | HiveMQ               |
| Dashboard        | ThingsBoard          |
| Cameras          | Milesight IP Cameras |
| Configuration    | JSON                 |

---

# System Pipeline

The system continuously processes live video streams using a multi-stage computer vision pipeline.

### 1. Video Acquisition

Multiple Milesight IP cameras provide real-time RTSP streams.

↓

### 2. Face Detection

Each incoming frame is analyzed to locate human faces.

↓

### 3. Face Tracking

Detected faces are tracked across frames using OpenCV KCF trackers, reducing unnecessary detections and improving processing efficiency.

↓

### 4. Face Recognition

Detected faces are converted into feature embeddings using InsightFace and compared against the enrolled database using cosine similarity.

↓

### 5. Identity Verification

The similarity score determines whether the detected individual matches an authorized identity.

↓

### 6. Event Publishing

Recognition events are published to the MQTT broker, allowing external systems to receive updates in real time.

↓

### 7. Monitoring & Access Control

Authorized users can trigger access control actions, while every event is stored for future auditing and monitoring through ThingsBoard.

---

# Results

## Live Recognition Dashboard

<p align="center">
<img src="https://github.com/alireza-keivan/Face-Recognition-multi-thread/blob/alireza-keivan/samples/Unknowns.png" width="900">
</p>

The dashboard displays every recognition event in real time, including the detected individual, confidence score, timestamp, captured image, and verification status.

---

## MQTT Communication

<p align="center">
<img src="https://github.com/alireza-keivan/Face-Recognition-multi-thread/blob/alireza-keivan/samples/d0b66a57-4179-4768-b59c-c49f8f94b849.png" width="850">
</p>

Recognition events are published through HiveMQ using MQTT, enabling seamless integration with external applications, IoT devices, and access control systems.

---

## Live Camera Processing

<p align="center">
<img src="https://github.com/alireza-keivan/Face-Recognition-multi-thread/blob/alireza-keivan/samples/cctv.png" width="900">
</p>

The system continuously detects, tracks, and recognizes faces from live video streams with minimal latency.

---

# Repository Structure

```
FaceRecognitionSystem/

├── config/
│   ├── config.json
│
├── database/
│   ├── enrolled_faces
│
├── logs/
│
├── models/
│
├── src/
│   ├── recognition
│   ├── tracking
│   ├── mqtt
│   ├── utilities
│
├── assets/
│
├── requirements.txt
│
└── main.py
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/alireza-keivan/FaceRecognitionSystem.git
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Configure your camera streams and MQTT settings inside the configuration file.

```bash
config/config.json
```

Run the application.

```bash
python recognizer.py
```

---

# Configuration

The application can be configured through a single JSON file.

Main configurable parameters include:

* Camera RTSP streams
* Recognition threshold
* MQTT broker settings
* Tracking parameters
* Logging options
* Face database location

No source code modification is required for deployment.

---

# Performance Highlights

* Real-time face recognition
* Multi-camera architecture
* Low-latency processing
* Efficient face tracking
* MQTT event streaming
* Enterprise-ready modular design

---

# Applications

This project is suitable for:

* Smart buildings
* Office access control
* Industrial facilities
* University campuses
* Residential complexes
* Secure laboratories
* Corporate headquarters
* Visitor management systems

---

# Future Improvements

* Docker deployment
* Web-based administration panel
* Face anti-spoofing
* Automatic face enrollment
* Distributed multi-server deployment
* GPU acceleration
* REST API integration
* Cloud synchronization

---

# Acknowledgements

This project was developed in collaboration with **Milesight** as part of an enterprise AI surveillance and access control solution.

Special thanks to the open-source communities behind:

* InsightFace
* OpenCV
* ONNX Runtime
* HiveMQ
* ThingsBoard

---

# License

This project is intended for educational, research, and commercial adaptation purposes under the terms of the repository license.
