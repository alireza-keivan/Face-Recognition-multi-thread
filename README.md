<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Face Recognition System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
            color: #333;
            margin: 0;
            padding: 2rem;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3, h4 {
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
        }
        h1 {
            font-size: 2.5rem;
            text-align: center;
            color: #1f2937;
        }
        h3 {
            font-size: 1.75rem;
            color: #1f2937;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
        h4 {
            font-size: 1.25rem;
            color: #4b5563;
        }
        p {
            margin-bottom: 1rem;
        }
        ul {
            list-style-type: disc;
            padding-left: 2rem;
            margin-bottom: 1rem;
        }
        li {
            margin-bottom: 0.5rem;
        }
        pre {
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        code {
            font-family: 'Fira Code', 'Courier New', Courier, monospace;
        }
        .divider {
            border-top: 1px solid #e5e7eb;
            margin: 2rem 0;
        }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container">
        <h1>Real-Time Face Recognition System</h1>
        <p>This project is a professional-grade, real-time face recognition system designed for security and access control applications. It leverages the power of OpenCV and the lightweight, highly accurate InsightFace library to perform fast face detection and recognition from a live video stream.</p>
        <p>The system is built to be modular and efficient, using multi-threading to handle video capture and face processing independently. It can be easily configured to work with various camera sources, including local webcams and RTSP streams. Recognized faces are published to an MQTT broker, enabling seamless integration with home automation or IoT security platforms.</p>

        <h3>Key Features</h3>
        <ul class="list-disc pl-6">
            <li><b>High-Performance Recognition</b>: Utilizes the InsightFace model for robust and accurate facial embeddings.</li>
            <li><b>Efficient Processing</b>: Multi-threaded architecture separates video capture from computationally intensive face analysis to maximize FPS.</li>
            <li><b>Scalable</b>: Easily add new known faces by simply placing an image in a designated directory. The system automatically re-encodes and saves the new data.</li>
            <li><b>Real-time Tracking</b>: Employs a KCF tracker to follow detected faces between full detection scans, ensuring smooth and consistent tracking while reducing CPU load.</li>
            <li><b>MQTT Integration</b>: Publishes recognition events, including face images and metadata, to an MQTT broker for real-time notifications and actions (e.g., opening a door, sending an alert).</li>
        </ul>
        
        <div class="divider"></div>

        <h3>Getting Started</h3>
        
        <h4>1. Configuration</h4>
        <p>First, you'll need to update the `config.json` file with your specific settings. This is where you configure camera streams, MQTT credentials, and recognition parameters.</p>
        <pre><code>{
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
</code></pre>
        <ul class="list-disc pl-6">
            <li>`camera_settings`: Set `url` for an RTSP stream or `stream_id` (0 for the default webcam). Adjust `fps_limit` to control the processing frame rate.</li>
            <li>`face_recognition_settings`: Modify `COSINE_SIMILARITY_THRESHOLD` to adjust the confidence required for a positive match. `IMAGE_SEND_COOLDOWN` prevents the system from spamming the MQTT broker.</li>
            <li>`directories`: Specify the local paths for your known face images and the file where their encodings will be stored.</li>
            <li>`mqtt_settings`: Fill in the details for your MQTT broker.</li>
        </ul>
        
        <div class="divider"></div>
        
        <h4>2. Add Known Faces</h4>
        <p>Place high-quality images of the faces you want to recognize in the directory you specified under `KNOWN_FACES_DIR`. Name the images clearly, as the filename (without the extension) will be used as the person's name. For example, `john_doe.jpg`.</p>
        
        <div class="divider"></div>

        <h4>3. Run the System</h4>
        <p>Simply run the `recognizer.py` script to start the system. It will automatically load the configuration, encode faces, connect to the camera, and begin processing.</p>
        <pre><code>python recognizer.py
</code></pre>
        
        <div class="divider"></div>

        <h3>Code Structure</h3>
        <ul class="list-disc pl-6">
            <li><b>`recognizer.py`</b>: The main entry point for the application. It loads the configuration and starts the `FaceRecognitionSystem` class.</li>
            <li><b>`thread.py`</b>: Contains the core logic for the system.
                <ul class="list-circle pl-6 mt-2">
                    <li>`WebcamStream`: A multi-threaded class for non-blocking video capture.</li>
                    <li>`FaceRecognitionSystem`: The central class that manages the entire pipeline, from loading faces and connecting to MQTT to running the main processing loop.</li>
                </ul>
            </li>
            <li><b>`dir.py`</b>: A utility script for managing known face encodings. It can save a pre-computed `known_faces_data.pkl` file to avoid re-encoding every time the application runs.</li>
            <li><b>`config.json`</b>: The configuration file that stores all the system's settings in a single, easy-to-edit place.</li>
        </ul>

        <div class="divider"></div>

        <h3>Dependencies</h3>
        <p>This project requires several libraries, including:</p>
        <ul class="list-disc pl-6">
            <li>`opencv-python`</li>
            <li>`insightface`</li>
            <li>`numpy`</li>
            <li>`paho-mqtt`</li>
            <li>`onnxruntime`</li>
            <li>`pickle`</li>
        </ul>
        <p>You can install them all at once using pip.</p>
    </div>
</body>
</html>
