Real-Time Face Recognition System

This project is a professional-grade, real-time face recognition system designed for security and access control applications. It leverages the power of OpenCV and the lightweight, highly accurate InsightFace library to perform fast face detection and recognition from a live video stream.

The system is built to be modular and efficient, using multi-threading to handle video capture and face processing independently. It can be easily configured to work with various camera sources, including local webcams and RTSP streams. Recognized faces are published to an MQTT broker, enabling seamless integration with home automation or IoT security platforms.
Key Features

    High-Performance Recognition: Utilizes the InsightFace model for robust and accurate facial embeddings.

    Efficient Processing: Multi-threaded architecture separates video capture from computationally intensive face analysis to maximize FPS.

    Scalable: Easily add new known faces by simply placing an image in a designated directory. The system automatically re-encodes and saves the new data.

    Real-time Tracking: Employs a KCF tracker to follow detected faces between full detection scans, ensuring smooth and consistent tracking while reducing CPU load.

    MQTT Integration: Publishes recognition events, including face images and metadata, to an MQTT broker for real-time notifications and actions (e.g., opening a door, sending an alert).

Getting Started
1. Configuration

First, you'll need to update the config.json file with your specific settings. This is where you configure camera streams, MQTT credentials, and recognition parameters.

{
    "camera_settings": {
        "url": "rtsp://your_camera_url_here",
        "stream_id": "0", 
        "fps_limit": 5.0
    },
    "face_recognition_settings": {
        "COSINE_SIMILARITY_THRESHOLD": 0.35,
        "IMAGE_SEND_COOLDOWN": 5
    },
    "directories": {
        "KNOWN_FACES_DIR": "/path/to/your/known/faces",
        "ENCODINGS_FILE": "/path/to/your/encodings/known_faces_data.pkl"
    },
    "mqtt_settings": {
        "MQTT_BROKER_ADDRESS": "your_mqtt_broker_address",
        "MQTT_BROKER_PORT": 1883,
        "MQTT_CLIENT_ID": "your_client_id",
        "MQTT_USERNAME": "your_username",
        "MQTT_PASSWORD": "your_password",
        "MQTT_FACE_TOPIC": "v1/devices/me/telemetry"
    },
    "optimization_settings": {
        "detection_interval_seconds": 5.0,
        "tracker_type": "KCF"
    }
}

    camera_settings: Set url for an RTSP stream or stream_id (0 for the default webcam). Adjust fps_limit to control the processing frame rate.

    face_recognition_settings: Modify COSINE_SIMILARITY_THRESHOLD to adjust the confidence required for a positive match. IMAGE_SEND_COOLDOWN prevents the system from spamming the MQTT broker.

    directories: Specify the local paths for your known face images and the file where their encodings will be stored.

    mqtt_settings: Fill in the details for your MQTT broker.

2. Add Known Faces

Place high-quality images of the faces you want to recognize in the directory you specified under KNOWN_FACES_DIR. Name the images clearly, as the filename (without the extension) will be used as the person's name. For example, john_doe.jpg.
3. Run the System

Simply run the recognizer.py script to start the system. It will automatically load the configuration, encode faces, connect to the camera, and begin processing.

python recognizer.py

Code Structure

    recognizer.py: The main entry point for the application. It loads the configuration and starts the FaceRecognitionSystem class.

    thread.py: Contains the core logic for the system.

        WebcamStream: A multi-threaded class for non-blocking video capture.

        FaceRecognitionSystem: The central class that manages the entire pipeline, from loading faces and connecting to MQTT to running the main processing loop.

    dir.py: A utility script for managing known face encodings. It can save a pre-computed known_faces_data.pkl file to avoid re-encoding every time the application runs.

    config.json: The configuration file that stores all the system's settings in a single, easy-to-edit place.

Dependencies

This project requires several libraries, including:

    opencv-python

    insightface

    numpy

    paho-mqtt

    onnxruntime

    pickle

You can install them all at once using pip.
