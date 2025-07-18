# recognizer.py
import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
import warnings # still needed
import paho.mqtt.client as mqtt 
from dir import KNOWN_FACES_DIR, images_folder, known_face_encodings, known_faces_names
from threads import WebcamStream

# --- Load Known Faces ---
known_faces_names, known_face_encodings = images_folder(KNOWN_FACES_DIR)

print(f"Total known faces loaded: {len(known_face_encodings)}")
if len(known_face_encodings) == 0:
    print("IMPORTANT: No known faces were loaded. Face recognition will not work.")
    print("Please check the 'known_faces' directory and the image files within it.")
print("---------------------------------------")


# --- MQTT Configuration ---
MQTT_BROKER_ADDRESS = ''
MQTT_BROKER_PORT = 
MQTT_CLIENT_ID = "" # Ensure unique client ID
MQTT_USERNAME = ""
MQTT_PASSWORD = ""
MQTT_DOOR_OPEN_TOPIC = ""

# --- MQTT Callbacks (for VERSION2 API) ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"MQTT: Connected successfully to broker {MQTT_BROKER_ADDRESS}")
    else:
        print(f"MQTT: Failed to connect, return code {rc}\n")

def on_disconnect(client, userdata, rc, *args, **kwargs):
    print(f"MQTT: Disconnected with result code {rc}\n")
    print(f"MQTT: Extra args received: {args}")
# --- MQTT Client Setup ---
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
if MQTT_USERNAME and MQTT_PASSWORD:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

try:
    mqtt_client.connect(MQTT_BROKER_ADDRESS, MQTT_BROKER_PORT)
    mqtt_client.loop_start() # Start MQTT background thread for handling messages
except Exception as e:
    print(f"MQTT: Could not connect to broker: {e}")
    # The program will continue running, but MQTT commands won't be sent.

last_open_time = 0 # To implement cooldown for sending "open" commands
COOLDOWN_PERIOD_SECONDS = 1 # Don't send "open" more often than this

def send_door_open_command(mqtt_client_obj, known_name):
    global last_open_time
    current_time = time.time()
    if (current_time - last_open_time) > COOLDOWN_PERIOD_SECONDS:
        payload = '{"IDENTITY_VERIFIED": "TRUE"}' # Example JSON payload
        try:
            # Check if client is actually connected before publishing
            if mqtt_client_obj.is_connected():
                result = mqtt_client_obj.publish(MQTT_DOOR_OPEN_TOPIC, payload)
                if result[0] == 0:
                    print(f"MQTT: Sent '{payload}' to topic '{MQTT_DOOR_OPEN_TOPIC}'")
                    last_open_time = current_time
                    #print()
                else:
                    print(f"MQTT: Failed to send message to topic {MQTT_DOOR_OPEN_TOPIC}. Result code: {result[0]}")
            else:
                print("MQTT: Client not connected, cannot publish message.")
        except Exception as e:
            print(f"MQTT: Error publishing message: {e}")
    else:
        print(f"MQTT: Cooldown active. Not sending 'open' command. ({COOLDOWN_PERIOD_SECONDS - (current_time - last_open_time):.1f}s remaining)")


# --- Initialize and Start Webcam Stream Thread ---
print("Initializing WebcamStream...")
webcam_stream = WebcamStream(stream_cap=0)
if webcam_stream.stopped: # Check if webcam stream failed to open in __init__
    print("Failed to start webcam stream. Exiting application.")
    exit(1)
webcam_stream.start()

# Give the webcam stream a moment to warm up and grab the first frame
print("Waiting for webcam stream to warm up...")
time.sleep(1)
if webcam_stream.read() is None:
    print("Webcam stream is not producing frames after startup. Please check camera connection/URL. Exiting.")
    webcam_stream.stop()
    exit(1)
print("WebcamStream ready.")

# --- OpenCV Window Setup (managed by recognizer.py directly) ---
OUTPUT_WINDOW_NAME = "Face Recognition Stream"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
cv2.namedWindow(OUTPUT_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(OUTPUT_WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)


# --- MediaPipe Face Detection Setup ---
mp_face_detection = mp.solutions.face_detection
face_detection_model = mp_face_detection.FaceDetection(min_detection_confidence=0.3)


# --- Face Recognition Loop ---
num_frames_processed = 0
start_time = time.time()
PROCESS_FRAME_SCALE = 0.3
RECOGNITION_TOLERANCE = 0.6 # Adjust as needed for match strictness

print("Starting face recognition loop. Press 'q' in the display window or Ctrl+C to exit.")
try:
    while True :
        if webcam_stream.stopped:
            print("Webcam stream stopped. Exiting recognition loop.")
            break

        frame = webcam_stream.read()
        if frame is None:
            print("Received NONE frame from webcam stream. Exiting recognition loop.")
            break
        elif frame.size == 0:
            print("Received EMPTY frame (size 0) from webcam stream. Exiting recognition loop.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=PROCESS_FRAME_SCALE, fy=PROCESS_FRAME_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)


        # --- Face Recognition and Drawing ---
        recognized_any_known_face = False
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Scale back up face locations to match the original frame size for drawing
            top = int(top / PROCESS_FRAME_SCALE)
            right = int(right / PROCESS_FRAME_SCALE)
            bottom = int(bottom / PROCESS_FRAME_SCALE)
            left = int(left / PROCESS_FRAME_SCALE)

            name = "Unknown" # Default name if no match is found

            if face_encodings_in_frame: # Check if there's an encoding for this face
                face_encoding = face_encodings_in_frame[i]
                if known_face_encodings: # Only attempt comparison if there are known faces loaded
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, RECOGNITION_TOLERANCE)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_faces_names[best_match_index]
                            recognized_any_known_face = True

            # Draw a box around the face and label it
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # --- MQTT Control ---
        if recognized_any_known_face:
            send_door_open_command(mqtt_client, known_name= "!")
        # No explicit 'else' to send a 'close' command; implement if needed by your door's logic.
        if not recognized_any_known_face:
            send_door_open_command(mt)
        # --- Display Frame ---
        cv2.imshow(OUTPUT_WINDOW_NAME, frame) # Use the consistent window name

        # --- Handle Key Press ---
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("'q' pressed. Exiting recognition loop.")
            break

        num_frames_processed += 1

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Signaling webcam stream to stop.")
finally:
    # --- Cleanup ---
    print("Performing cleanup...")
    if not webcam_stream.stopped:
        webcam_stream.stop()

    if mqtt_client.is_connected():
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("MQTT client disconnected.")

    if 'face_detection_model' in locals() and face_detection_model:
        face_detection_model.close() # Release MediaPipe resources

    cv2.destroyAllWindows()
    print("OpenCV windows closed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    if num_frames_processed > 0:
        fps = num_frames_processed / elapsed_time
        print(f"FPS: {fps:.2f}, Elapsed Time: {elapsed_time:.2f} seconds, Frames Processed: {num_frames_processed}")
    else:
        print("No frames processed.")

    print("Application exited.")
