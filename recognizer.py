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

# MQTT Configuration
"""MQTT_BROKER_ADDRESS = '192.168.45.115'
MQTT_BROKER_PORT = 1883 
MQTT_CLIENT_ID = "h3s4if5snwex1s48248y"
MQTT_USERNAME = "whci9cx9ifv0xbayda45"
MQTT_PASSWORD = 'qmgfiay3oilhzxki868i'
MQTT_DOOR_OPEN_TOPIC = "v1/devices/me/telemetry" # This topic MUST match what your door device subscribes to.
"""

directory = KNOWN_FACES_DIR
images_folder(directory)


# initializing and starting multi-threaded webcam capture input stream 
webcam_stream = WebcamStream(stream_cap = 0) #  stream_id = 0 is for primary camera 
webcam_stream.start()

OUTPUT_WINDOW_NAME = "Face Recognition Stream" 
WINDOW_WIDTH = 800 
WINDOW_HEIGHT = 600 

cv2.namedWindow(OUTPUT_WINDOW_NAME, cv2.WINDOW_NORMAL) 
cv2.resizeWindow(OUTPUT_WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT) 

num_frames_processed = 0 
start = time.time()
mp_face_detection = mp.solutions.face_detection
face_detection_model = mp_face_detection.FaceDetection(min_detection_confidence=0.3)
# processing frames in input stream

while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read() 
        if frame is None:
            print("no frame received")
            break

    PROCESS_FRAME_SCALE = 0.3
    RECOGNITION_TOLERANCE = 0.6 # Adjust as needed for match strictness
    FRAME_SKIP_INTERVAL = 2 # Start with 2, you can experiment with 3, 4, etc.
    PERSISTENCE_THRESHOLD = 30 #important
    # Increment frame count

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=PROCESS_FRAME_SCALE, fy=PROCESS_FRAME_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # --- Debugging: Check detected faces in current frame (uncomment if you need this info) ---
    print(f"Faces detected in current frame: {len(face_locations)}")

    # --- Recognize faces and draw bounding boxes ---
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Scale back up face locations to match the original frame size for drawing
        top = int(top / PROCESS_FRAME_SCALE)
        right = int(right / PROCESS_FRAME_SCALE)
        bottom = int(bottom / PROCESS_FRAME_SCALE)
        left = int(left / PROCESS_FRAME_SCALE)

        face_encoding = face_encodings_in_frame[i]

        name = "unknown" # Default name if no match is found

        if known_face_encodings: # Only attempt comparison if there are known faces loaded
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # Given a list of face encodings, compare them to a known face encoding 
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0: # checking for best matches
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw a label with a name above the face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
      
    cv2.imshow(OUTPUT_WINDOW_NAME, frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    num_frames_processed+=1
end = time.time()
webcam_stream.stop() # stop the webcam stream 
face_detection_model.close() # Release MediaPipe resources

# printing time elapsed and fps 
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows 
cv2.destroyAllWindows()



print(f"Total known faces loaded: {len(known_face_encodings)}")
if len(known_face_encodings) == 0:
    print("IMPORTANT: No known faces were loaded. Face recognition will not work.")
    print("Please check the 'known_faces' directory and the image files within it.")
print("---------------------------------------")
