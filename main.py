
from face_processor import FaceRecognitionProcessor # Import your new class
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
from threading import Thread
import paho.mqtt.client as mqtt
import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from directory_file import *
from threading import Thread
import time
from threads import *
import paho.mqtt.client as mqtt
from face_processor import *
# ... (your directory and known faces loading) ...

MQTT_BROKER_ADDRESS = '192.168.45.115'
MQTT_BROKER_PORT = 1883 
MQTT_CLIENT_ID = "h3s4if5snwex1s48248y"
MQTT_USERNAME = "whci9cx9ifv0xbayda45"
MQTT_PASSWORD = 'qmgfiay3oilhzxki868i'
MQTT_DOOR_OPEN_TOPIC = "v1/devices/me/telemetry" # This topic MUST match what your door device subscribes to.

# initializing and starting multi-threaded webcam capture input stream
webcam_stream = WebcamStream(stream_cap = 0) # Adjust stream_cap if using a different camera
webcam_stream.start()

# Initialize and start the face recognition processor in its own thread
face_processor = FaceRecognitionProcessor(
    webcam_stream_instance=webcam_stream,
    known_encodings=known_face_encodings, # These come from directory_file.py
    known_names=known_faces_names,       # These come from directory_file.py
    mqtt_broker_address=MQTT_BROKER_ADDRESS,
    mqtt_broker_port=MQTT_BROKER_PORT,
    mqtt_client_id=MQTT_CLIENT_ID,
    mqtt_username=MQTT_USERNAME,
    mqtt_password=MQTT_PASSWORD,
    mqtt_topic_open=MQTT_DOOR_OPEN_TOPIC,
    output_window_name="Face Recognition Stream", # Optional: can customize window
    window_width=800,
    window_height=600
)
face_processor.start() # This kicks off the new thread for face recognition and display

# The main thread (recognizer.py) now needs to keep running
# until you want to stop the application.
try:
    while True:
        # This loop keeps the main thread alive and allows it to
        # check if the other threads have stopped.
        if webcam_stream.stopped and face_processor.stopped:
            print("Both webcam stream and face processor have stopped. Exiting main.")
            break
        # Optional: If you want to stop everything when the face_processor stops (e.g., 'q' pressed)
        if face_processor.stopped:
            print("Face processor stopped. Stopping webcam stream and exiting.")
            webcam_stream.stop()
            break
        time.sleep(0.1) # Prevents busy-waiting
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Signaling threads to stop.")
finally:
    # Ensure proper cleanup
    if not webcam_stream.stopped:
        webcam_stream.stop()
    if not face_processor.stopped:
        face_processor.stop()
    print("Application cleanup complete. Exiting.")
