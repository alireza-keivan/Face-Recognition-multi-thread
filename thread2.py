from threading import Thread
import time
import cv2
import json
import os
import numpy as np
import face_recognition
import mediapipe as mp
import paho.mqtt.client as mqtt
from datetime import datetime
import queue # Import the queue module

# Assuming 'dir.py' has a function like images_folder to load encodings
# You might need to adjust the import based on its actual content.
# For simplicity, I'll mock images_folder if not provided.
try:
    from dir import images_folder
except ImportError:
    print("Warning: 'dir.py' or 'images_folder' not found. Using a mock function.")
    def images_folder(directory_path):
        print(f"Mock: Loading known faces from {directory_path}")
        return [], [] # Return empty lists if not real images


class WebcamStream:
    """
    Helper class for implementing multi-threaded video stream processing.
    """
    def __init__(self, stream_cap=0, camera_url=None):
        # Use camera_url if provided, otherwise default to stream_cap (e.g., 0 for default webcam)
        self.camera_source = camera_url if camera_url else stream_cap
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            print(f"[ERROR]: Error accessing webcam stream at {self.camera_source}.")
            self.grabbed = False
            self.frame = None
            self.stopped = True
            return

        fps_input_stream = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"FPS of webcam hardware/input stream: {fps_input_stream}")

        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read from camera.')
            self.frame = None
            self.stopped = True
            return

        self.stopped = False
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # Daemon threads exit when the main program exits

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                break
            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed:
                print('[Exiting] No more frames to read in update loop.')
                self.stopped = True
                break
        self.cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class FaceRecognitionSystem:
    """
    Encapsulates the entire face recognition, MQTT communication,
    and display logic, using a separate thread for face processing.
    """
    def __init__(self, config):
        self.config = config

        # --- Extract Config Values ---
        camera_settings = self.config.get("camera_settings", {})
        face_recognition_settings = self.config.get("face_recognition_settings", {})
        directories = self.config.get("directories", {})
        mqtt_settings = self.config.get("mqtt_settings", {})

        # Camera Settings
        self.CAMERA_URL = camera_settings.get("url", "rtsp://admin:ms123456@192.168.48.67:554/main")
        self.STREAM_ID = int(camera_settings.get("stream_id", "0"))
        self.FPS_LIMIT = float(camera_settings.get("fps_limit", 30.0) or 30.0)
        self.OUTPUT_WINDOW_NAME = camera_settings.get("OUTPUT_WINDOW_NAME", "Face Recognition Stream")
        self.WINDOW_WIDTH = int(camera_settings.get("WINDOW_WIDTH", 1200))
        self.WINDOW_HEIGHT = int(camera_settings.get("WINDOW_HEIGHT", "1000"))

        # Face Recognition Settings
        self.PROCESS_FRAME_SCALE = float(face_recognition_settings.get("PROCESS_FRAME_SCALE", 0.3))
        self.RECOGNITION_TOLERANCE = float(face_recognition_settings.get("RECOGNITION_TOLERANCE", 0.5))
        self.DOOR_OPEN_COOLDOWN = float(face_recognition_settings.get("DOOR_OPEN_COOLDOWN", 5))
        self.RESET_DOOR_TIMER = float(face_recognition_settings.get("RESET_DOOR_TIMER", 5))

        # Directory Settings
        self.KNOWN_FACES_DIR = directories.get("KNOWN_FACES_DIR", "/home/alireza/Documents/face_recognition/known_faces/captured_known_faces")
        self.KNOWN_FACES_OUTPUT_DIR = directories.get("output_path", "/home/alireza/Documents/face_recognition/captured")
        self.UNKNOWN_FACES_OUTPUT_DIR = directories.get("unknown_output_path", "/home/alireza/Documents/face_recognition/unknown_captured")

        # MQTT Settings
        self.MQTT_BROKER_ADDRESS = mqtt_settings.get("MQTT_BROKER_ADDRESS", "localhost")
        self.MQTT_BROKER_PORT = int(mqtt_settings.get("MQTT_BROKER_PORT", 1883))
        self.MQTT_CLIENT_ID = mqtt_settings.get("MQTT_CLIENT_ID", "default_client")
        self.MQTT_USERNAME = mqtt_settings.get("MQTT_USERNAME")
        self.MQTT_PASSWORD = mqtt_settings.get("MQTT_PASSWORD")
        self.MQTT_DOOR_OPEN_TOPIC = mqtt_settings.get("MQTT_DOOR_OPEN_TOPIC", "v1/devices/me/telemetry")
        self.MQTT_FACE_STATUS_TOPIC = "face_detection/status"

        # --- Initialize Components ---
        self._load_known_faces()
        self._init_mediapipe()
        self._init_mqtt_client()
        self._init_webcam_stream()
        self._init_opencv_window()
        self._create_output_directories()

        # --- Queues for Thread Communication ---
        self.frame_queue = queue.Queue(maxsize=2) # Queue for raw frames to be processed
        self.results_queue = queue.Queue(maxsize=2) # Queue for processed frames/results

        # --- State Variables ---
        self.last_door_open_time = 0
        self.last_known_face_exit_time = 0
        self.num_frames_processed = 0
        self.start_time = time.time()
        self.processing_stopped = False # Flag to control the processing thread

    def _load_known_faces(self):
        """Loads known face encodings and names."""
        self.known_faces_names, self.known_face_encodings = images_folder(self.KNOWN_FACES_DIR)
        print(f"Total known faces loaded: {len(self.known_face_encodings)}")
        if len(self.known_face_encodings) == 0:
            print("Please check the 'known_faces' directory and the image files within it.")
        print("---------------------------------------")

    def _init_mediapipe(self):
        """Initializes MediaPipe Face Detection model."""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection_model = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    def _init_mqtt_client(self):
        """Initializes and connects MQTT client."""
        self.mqtt_client = mqtt.Client(client_id=self.MQTT_CLIENT_ID)
        if self.MQTT_USERNAME and self.MQTT_PASSWORD:
            self.mqtt_client.username_pw_set(username=self.MQTT_USERNAME, password=self.MQTT_PASSWORD)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print(f"Failed to connect to MQTT Broker, return code {rc}\n")

        self.mqtt_client.on_connect = on_connect
        try:
            self.mqtt_client.connect(self.MQTT_BROKER_ADDRESS, self.MQTT_BROKER_PORT, 60)
            self.mqtt_client.loop_start() # Start background thread for MQTT
        except Exception as e:
            print(f"Could not connect to MQTT broker: {e}. MQTT functionality will be disabled.")
            self.mqtt_client = None

    def _init_webcam_stream(self):
        """Initializes and starts the webcam stream."""
        print("Initializing WebcamStream...")
        self.webcam_stream = WebcamStream(stream_cap=self.STREAM_ID, camera_url=self.CAMERA_URL)
        if self.webcam_stream.stopped:
            print("Failed to start webcam stream. System cannot operate without stream.")
            exit(1)
        self.webcam_stream.start()
        time.sleep(1)
        if self.webcam_stream.read() is None:
            print("Webcam stream is not producing frames. Check camera connection/URL. Exiting.")
            self.webcam_stream.stop()
            exit(1)
        print("WebcamStream ready.")

    def _init_opencv_window(self):
        """Sets up the OpenCV display window."""
        cv2.namedWindow(self.OUTPUT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.OUTPUT_WINDOW_NAME, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

    def _create_output_directories(self):
        """Ensures output directories for captured faces exist."""
        os.makedirs(self.KNOWN_FACES_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.UNKNOWN_FACES_OUTPUT_DIR, exist_ok=True)
        print(f"Output directories ensured: {self.KNOWN_FACES_OUTPUT_DIR}, {self.UNKNOWN_FACES_OUTPUT_DIR}")

    def _send_door_open_command(self, recognized_person_name="!"):
        """Sends an MQTT command to open the door."""
        if self.mqtt_client and self.mqtt_client.is_connected():
            topic = self.MQTT_DOOR_OPEN_TOPIC
            message = json.dumps({"action": "open", "recognized_person": recognized_person_name})
            try:
                self.mqtt_client.publish(topic, message)
                print(f"MQTT: Sent door open command for {recognized_person_name} to topic {topic}")
            except Exception as e:
                print(f"Error publishing MQTT door open command: {e}")
        else:
            print("MQTT client not connected, cannot send door open command.")

    def _publish_face_status(self, face_name, location):
        """Publishes general face detection status to MQTT."""
        if self.mqtt_client and self.mqtt_client.is_connected():
            message = json.dumps({
                "person": face_name,
                "location": {"top": location[0], "right": location[1], "bottom": location[2], "left": location[3]},
                "timestamp": datetime.now().isoformat()
            })
            try:
                self.mqtt_client.publish(self.MQTT_FACE_STATUS_TOPIC, message)
            except Exception as e:
                print(f"Error publishing MQTT face status: {e}")

    def _process_frame_in_thread(self):
        """
        This method will run in a separate thread. It continuously
        pulls frames from frame_queue, processes them, and puts results
        into results_queue.
        """
        print("Face processing thread started.")
        while not self.processing_stopped:
            try:
                # Get a raw frame from the queue with a timeout
                frame = self.frame_queue.get(timeout=0.1)

                small_frame = cv2.resize(frame, (0, 0), fx=self.PROCESS_FRAME_SCALE, fy=self.PROCESS_FRAME_SCALE)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

                faces_info = []
                recognized_any_known_face_this_frame = False
                last_recognized_name_this_frame = "Unknown"

                for i, (top, right, bottom, left) in enumerate(face_locations):
                    # Scale back up face locations to original frame size
                    top = int(top / self.PROCESS_FRAME_SCALE)
                    right = int(right / self.PROCESS_FRAME_SCALE)
                    bottom = int(bottom / self.PROCESS_FRAME_SCALE)
                    left = int(left / self.PROCESS_FRAME_SCALE)

                    name = "Unknown"
                    if face_encodings_in_frame and i < len(face_encodings_in_frame):
                        face_encoding = face_encodings_in_frame[i]
                        if self.known_face_encodings:
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, self.RECOGNITION_TOLERANCE)
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = self.known_faces_names[best_match_index]
                                    recognized_any_known_face_this_frame = True
                                    last_recognized_name_this_frame = name

                    faces_info.append({"name": name, "location": (top, right, bottom, left)})

                    # Draw box and label (on the original frame, as it will be returned for display)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # Capture unknown faces
                    if name == "Unknown":
                        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"Unknown_{current_time_str}_{hash(tuple(face_encoding)) % 10000}.jpg"
                        filepath = os.path.join(self.UNKNOWN_FACES_OUTPUT_DIR, filename)
                        os.makedirs(self.UNKNOWN_FACES_OUTPUT_DIR, exist_ok=True)
                        cv2.imwrite(filepath, frame[top:bottom, left:right])

                # Put results into the results queue
                try:
                    self.results_queue.put_nowait((faces_info, recognized_any_known_face_this_frame, last_recognized_name_this_frame, frame))
                except queue.Full:
                    # If queue is full, skip this frame to prevent blocking
                    # print("Results queue full, skipping frame processing output.")
                    pass
            except queue.Empty:
                # No frame in queue, continue looping or sleep briefly
                time.sleep(0.001) # Small sleep to prevent busy-waiting
            except Exception as e:
                print(f"Error in face processing thread: {e}")
                self.processing_stopped = True # Stop thread on unhandled error

        print("Face processing thread stopped.")

    def run(self):
        """
        Main loop for video capture, putting frames into queue,
        displaying results from queue, and door control.
        """
        # Start the dedicated face processing thread
        self.processing_thread = Thread(target=self._process_frame_in_thread, args=())
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("Started face processing thread.")

        print("Starting main display loop. Press 'q' in the display window or Ctrl+C to exit.")
        try:
            while True:
                if self.webcam_stream.stopped:
                    print("Webcam stream stopped. Exiting main loop.")
                    break

                # --- Get raw frame and put into processing queue ---
                frame = self.webcam_stream.read()
                if frame is None or frame.size == 0:
                    print("Received empty/NONE frame from webcam. Exiting main loop.")
                    break

                try:
                    # Put raw frame into the queue for the processing thread
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # If processing queue is full, it means the processor is slow.
                    # We can skip this frame to keep the video stream real-time.
                    # print("Frame queue full, skipping frame to maintain real-time stream.")
                    pass

                # --- Get processed results from results queue ---
                try:
                    faces_info, recognized_any_known_face, last_recognized_name, processed_frame = self.results_queue.get_nowait()

                    current_time = time.time()

                    # --- Door Control Logic ---
                    if recognized_any_known_face:
                        self.last_known_face_exit_time = 0

                        if current_time - self.last_door_open_time > self.DOOR_OPEN_COOLDOWN:
                            self._send_door_open_command(recognized_person_name=last_recognized_name)
                            self.last_door_open_time = current_time
                    else: # No known face recognized in the current frame
                        if self.last_known_face_exit_time == 0:
                            self.last_known_face_exit_time = current_time
                        elif current_time - self.last_known_face_exit_time > self.RESET_DOOR_TIMER:
                            self.last_door_open_time = 0
                            print(f"No known face for {self.RESET_DOOR_TIMER} seconds. Door open cooldown reset.")

                    # --- MQTT Status Update for all faces ---
                    for face_data in faces_info:
                        self._publish_face_status(face_data["name"], face_data["location"])

                    # --- Display Processed Frame ---
                    cv2.imshow(self.OUTPUT_WINDOW_NAME, processed_frame)
                    self.num_frames_processed += 1 # Only count frames that were actually processed and displayed

                except queue.Empty:
                    # No processed frame available yet, just display the last known good frame
                    # or handle appropriately to keep display responsive.
                    # For simplicity, if no new frame, cv2.imshow will keep displaying previous.
                    pass
                except Exception as e:
                    print(f"Error in main loop while handling results: {e}")


                # --- Handle Key Press ---
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("'q' pressed. Exiting main loop.")
                    break

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Signaling system to stop.")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Performs cleanup operations."""
        print("Performing cleanup...")
        # Stop the face processing thread
        self.processing_stopped = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5) # Give it a chance to finish
            if self.processing_thread.is_alive():
                print("Warning: Face processing thread did not terminate gracefully.")

        if not self.webcam_stream.stopped:
            self.webcam_stream.stop()

        if self.mqtt_client:
            if self.mqtt_client.is_connected():
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                print("MQTT client disconnected.")
            self.mqtt_client = None

        if hasattr(self, 'face_detection_model') and self.face_detection_model:
            self.face_detection_model.close()

        cv2.destroyAllWindows()
        print("OpenCV windows closed.")

        elapsed_time = time.time() - self.start_time
        if self.num_frames_processed > 0:
            fps = self.num_frames_processed / elapsed_time
            print(f"FPS: {fps:.2f}, Elapsed Time: {elapsed_time:.2f} seconds, Frames Processed: {self.num_frames_processed}")
        else:
            print("No frames processed.")
        print("Application exited.")