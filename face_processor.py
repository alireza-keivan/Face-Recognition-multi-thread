import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
from threading import Thread
import paho.mqtt.client as mqtt


# You will need to ensure that known_face_encodings and known_faces_names
# are accessible here, either by passing them as arguments to the constructor
# or by importing them if directory_file.py defines them globally and they
# are populated before FaceRecognitionProcessor is instantiated.
# For simplicity and good practice, they are passed as arguments here.

class FaceRecognitionProcessor:
    def __init__(self, webcam_stream_instance, known_encodings, known_names,
                 mqtt_broker_address, mqtt_broker_port, mqtt_client_id,
                 mqtt_username=None, mqtt_password=None, mqtt_topic_open="door/command/open",
                 output_window_name="Face Recognition Stream", window_width=800, window_height=600):
        
        self.webcam_stream = webcam_stream_instance
        self.known_face_encodings = known_encodings
        self.known_faces_names = known_names
        self.stopped = True # Control flag for this thread
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # Allows thread to exit when main program exits

        # MediaPipe Face Detection Initialization (done once)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection_model = self.mp_face_detection.FaceDetection(min_detection_confidence=0.3)

        # OpenCV Window Initialization (done once)
        self.output_window_name = output_window_name
        self.window_width = window_width
        self.window_height = window_height
        cv2.namedWindow(self.output_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.output_window_name, self.window_width, self.window_height)

        # MQTT Configuration
        self.mqtt_broker_address = mqtt_broker_address
        self.mqtt_broker_port = mqtt_broker_port
        self.mqtt_client_id = mqtt_client_id
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password
        self.mqtt_topic_open = mqtt_topic_open # Topic to publish to for opening the door

        self.mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1, client_id=self.mqtt_client_id)
        if self.mqtt_username and self.mqtt_password:
            self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)

        # Set MQTT callbacks (optional but good practice for debugging)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect

        self.last_open_time = 0 # For debouncing MQTT messages
        self.cooldown_period = 5 # seconds, adjust as needed (e.g., 5 seconds before allowing another 'open' command)

        self.num_frames_processed = 0
        self.start_time = 0 # Will be set when update loop starts

    # --- MQTT Callback functions ---
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"MQTT: Connected successfully to broker {self.mqtt_broker_address}")
        else:
            print(f"MQTT: Failed to connect, return code {rc}\n")

    def _on_disconnect(self, client, userdata, rc):
        print(f"MQTT: Disconnected with result code {rc}\n")

    def connect_mqtt(self):
        try:
            self.mqtt_client.connect(self.mqtt_broker_address, self.mqtt_broker_port, 60) # 60 sec keepalive
            self.mqtt_client.loop_start() # Start MQTT network loop in a non-blocking way
            print("MQTT: Client loop started.")
        except Exception as e:
            print(f"MQTT: Error connecting or starting loop: {e}")

    def disconnect_mqtt(self):
        self.mqtt_client.loop_stop() # Stop MQTT network loop
        self.mqtt_client.disconnect()
        print("MQTT: Client loop stopped and disconnected.")

    # --- Thread Control Methods ---
    def start(self):
        self.stopped = False
        self.connect_mqtt() # Connect MQTT when starting the thread
        self.t.start()

    def update(self):
        self.start_time = time.time() # Start timing for FPS calculation
        while True :
            if self.stopped is True :
                break

            frame = self.webcam_stream.read() # Read from the other thread's output
            if frame is None:
                print("[FaceProcessor] No frame received, signaling stop.")
                self.stopped = True # Signal stop if stream no longer provides frames
                break

            PROCESS_FRAME_SCALE = 0.3
            RECOGNITION_TOLERANCE = 0.6 # Adjust as needed for match strictness

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=PROCESS_FRAME_SCALE, fy=PROCESS_FRAME_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # --- Debugging: Check detected faces in current frame (optional) ---
            # print(f"[FaceProcessor] Faces detected in current frame: {len(face_locations)}")

            # --- Recognize faces and draw bounding boxes ---
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Scale back up face locations to match the original frame size for drawing
                top = int(top / PROCESS_FRAME_SCALE)
                right = int(right / PROCESS_FRAME_SCALE)
                bottom = int(bottom / PROCESS_FRAME_SCALE)
                left = int(left / PROCESS_FRAME_SCALE)

                face_encoding = face_encodings_in_frame[i]

                name = "unknown" # Default name if no match is found

                if self.known_face_encodings: # Only attempt comparison if there are known faces loaded
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    if len(face_distances) > 0: # checking for best matches
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_faces_names[best_match_index]

                # --- MQTT Publishing Logic (Debounced) ---
                if name != "unknown": # If a known person is identified
                    current_time = time.time()
                    if (current_time - self.last_open_time) > self.cooldown_period:
                        print(f"Known person detected: {name}. Sending MQTT message to open door.")
                        message_payload = "OPEN" # Or whatever your door device expects (e.g., "true", "1")
                        try:
                            self.mqtt_client.publish(self.mqtt_topic_open, message_payload, qos=1) # QoS 1 for assured delivery
                            self.last_open_time = current_time # Update last open time for debouncing
                            print(f"MQTT: Message '{message_payload}' published to topic '{self.mqtt_topic_open}'")
                        except Exception as e:
                            print(f"MQTT: Error publishing message: {e}")

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with a name above the face
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Display the processed frame AFTER all drawing operations for the current frame
            cv2.imshow(self.output_window_name, frame)
            key = cv2.waitKey(1)
            if key == ord('q'): # Check for quit key
                print("[FaceProcessor] 'q' pressed, signaling stop.")
                self.stopped = True # Signal stop
                break

            self.num_frames_processed += 1

        # Calculate and print FPS once the loop breaks
        end_time = time.time()
        elapsed = end_time - self.start_time
        if elapsed > 0:
            fps = self.num_frames_processed / elapsed
            print(f"[FaceProcessor] FPS: {fps:.2f} , Elapsed Time: {elapsed:.2f} , Frames Processed: {self.num_frames_processed}")
        else:
            print("[FaceProcessor] No frames processed or elapsed time is zero.")

        self.stop() # Ensure stop method is called to clean up

    def stop(self):
        self.stopped = True
        self.disconnect_mqtt() # Disconnect MQTT when stopping the thread
        if self.face_detection_model:
            self.face_detection_model.close() # Release MediaPipe resources
        cv2.destroyAllWindows() # Close all OpenCV windows
        print("[FaceProcessor] FaceRecognitionProcessor stopped and resources released.")