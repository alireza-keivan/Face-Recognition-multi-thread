# Real-Time Face Recognition System

## Overview

This project is a professional-grade, real-time face recognition system designed for security and access control applications. It leverages the power of OpenCV and the lightweight, highly accurate InsightFace library to perform fast face detection and recognition from a live video stream.

The system is built to be modular and efficient, using multi-threading to handle video capture and face processing independently. It can be easily configured to work with various camera sources, including local webcams and RTSP streams. Recognized faces are published to an MQTT broker, enabling seamless integration with home automation or IoT security **platforms**.
Key Features
---
## Features
- **High-Performance Recognition: Utilizes the InsightFace model for robust and accurate facial embeddings.

- **Efficient Processing: Multi-threaded architecture separates video capture from computationally intensive face analysis to maximize FPS.

- **Scalable: Easily add new known faces by simply placing an image in a designated directory. The system automatically re-encodes and saves the new data.

- **Real-time Tracking: Employs a KCF tracker to follow detected faces between full detection scans, ensuring smooth and consistent tracking while reducing CPU load.

- **MQTT Integration: Publishes recognition events, including face images and metadata, to an MQTT broker for real-time notifications and actions (e.g., opening a door, sending an alert).  
