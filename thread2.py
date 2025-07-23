# thread2.py
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
import base64 # Import base64 for image encoding

try:
    from dir import images_folder, known_face_encodings, known_faces_names # known_face_encodings, known_faces_names are global in dir.py
except ImportError:
    print("Warning: 'dir.py' or 'images_folder' not found. Using a mock function.")

    known_face_encodings = []
    known_faces_names = []
    def images_folder(directory_path, encodings_file_path): # Mock must accept new arg
        print(f"Mock: Loading known faces from {directory_path} (encodings_file: {encodings_file_path})")
        return [], [] # Return empty lists if not real images


class WebcamStream:
    """
        multi-threaded video stream processing.
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
    Face recognition, MQTT communication,
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
        self.CAMERA_URL = camera_settings.get("url")
        self.STREAM_ID = int(camera_settings.get("stream_id"))
        self.FPS_LIMIT = float(camera_settings.get("fps_limit"))
        self.OUTPUT_WINDOW_NAME = camera_settings.get("OUTPUT_WINDOW_NAME")
        self.WINDOW_WIDTH = int(camera_settings.get("WINDOW_WIDTH"))
        self.WINDOW_HEIGHT = int(camera_settings.get("WINDOW_HEIGHT"))

        # Face Recognition Settings
        self.PROCESS_FRAME_SCALE = float(face_recognition_settings.get("PROCESS_FRAME_SCALE"))
        self.RECOGNITION_TOLERANCE = float(face_recognition_settings.get("RECOGNITION_TOLERANCE"))
        self.DOOR_OPEN_COOLDOWN = float(face_recognition_settings.get("DOOR_OPEN_COOLDOWN"))
        self.RESET_DOOR_TIMER = float(face_recognition_settings.get("RESET_DOOR_TIMER"))
        self.IMAGE_SEND_COOLDOWN = float(face_recognition_settings.get("IMAGE_SEND_COOLDOWN")) # New cooldown for image sending

        # Directory Settings
        self.KNOWN_FACES_DIR = directories.get("KNOWN_FACES_DIR")
        self.KNOWN_FACES_OUTPUT_DIR = directories.get("output_path")
        self.UNKNOWN_FACES_OUTPUT_DIR = directories.get("unknown_output_path")
        self.ENCODINGS_FILE_PATH = directories.get("ENCODINGS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__))))


        # MQTT Settings
        self.MQTT_BROKER_ADDRESS = mqtt_settings.get("MQTT_BROKER_ADDRESS")
        self.MQTT_BROKER_PORT = int(mqtt_settings.get("MQTT_BROKER_PORT"))
        self.MQTT_CLIENT_ID = mqtt_settings.get("MQTT_CLIENT_ID")
        self.MQTT_USERNAME = mqtt_settings.get("MQTT_USERNAME")
        self.MQTT_PASSWORD = mqtt_settings.get("MQTT_PASSWORD")
        self.MQTT_DOOR_OPEN_TOPIC = mqtt_settings.get("MQTT_DOOR_OPEN_TOPIC")
        self.MQTT_FACE_STATUS_TOPIC = mqtt_settings.get("MQTT_FACE_STATUS_TOPIC")
        self.MQTT_IMAGE_TOPIC = mqtt_settings.get("MQTT_IMAGE_TOPIC")
        self.MQTT_UNKNOWN_FACE_TOPIC = mqtt_settings.get("MQTT_UNKNOWN_FACE_TOPIC")

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
        self.face_recognition_settings = self.config.get("face_recognition_settings", {})
        # --- Image Sending Cooldowns ---
        # Tracks last sent time for each known face by name
        self.last_sent_known_face_image_time = {}
        # Tracks last sent time for any unknown face (simple cooldown for all unknown)
        self.last_sent_unknown_face_image_time = 0
        self.last_image_send_time = 0 
        

        
    def send_face_image(self, face_image_np):
       
        current_time = time.time()
        # Apply cooldown to prevent sending too many images
        if (current_time - self.last_image_send_time) < self.IMAGE_SEND_COOLDOWN:
            # print(f"MQTT: Cooldown active ({self.IMAGE_SEND_COOLDOWN}s). Not sending unknown face image yet.")
            return

        if self.mqtt_client is None or not self.mqtt_client.is_connected():
            print("MQTT: Client not connected, cannot publish unknown face image.")
            return

        if face_image_np is None or face_image_np.size == 0:
            print("Error: face_image_np is empty or None. Cannot encode for MQTT.")
            return

        try:
            # Convert NumPy array (image) to JPEG bytes
            # .jpg is a good format for image compression
            _, buffer = cv2.imencode('.jpg', face_image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 70]) # Quality 70
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            # Create payload with timestamp and Base64 image
            payload = json.dumps({
                "image": encoded_image,
                "timestamp": datetime.now().isoformat()

            })

            # Publish the message
            result = self.mqtt_client.publish(self.MQTT_UNKNOWN_FACE_TOPIC, payload, qos=1) # QoS 1 for reliable delivery
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"MQTT: Successfully published unknown face image (len: {len(encoded_image)} bytes) to topic '{self.MQTT_UNKNOWN_FACE_TOPIC}'")
                self.last_image_send_time = current_time # Update time only on successful send
            else:
                print(f"MQTT: Failed to publish unknown face image. Return code: {result.rc}")

        except Exception as e:
            print(f"MQTT: Error during encoding or publishing unknown face image: {e}")



    def _load_known_faces(self):
        """Loads known face encodings and names."""
        # Pass the ENCODINGS_FILE_PATH to images_folder
        global known_face_encodings, known_faces_names # Ensure we are modifying the global vars from dir.py
        known_faces_names, known_face_encodings = images_folder(self.KNOWN_FACES_DIR, self.ENCODINGS_FILE_PATH)
        self.known_face_encodings = known_face_encodings 
        self.known_faces_names = known_faces_names 

        print(f"Total known faces loaded: {len(self.known_face_encodings)}")
        if len(self.known_face_encodings) == 0:
            print("Please check the 'known_faces' directory and the image files within it.")
        print("---------------------------------------")

    def _init_mediapipe(self):
        """Initializes MediaPipe Face Detection model."""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection_model = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

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
            message = json.dumps({"IDENTITY_VERIFIED": "TRUE", "recognized_person": recognized_person_name})
            try:
                self.mqtt_client.publish(topic, message)
                print(f"MQTT: Sent door open command for {recognized_person_name} to topic {topic}")
            except Exception as e:
                print(f"Error publishing MQTT door open command: {e}")
        else:
            print("MQTT client not connected, cannot send door open command.")

    def _publish_face_status(self, face_name, location):

        if self.mqtt_client and self.mqtt_client.is_connected():
            message = json.dumps({
                "person": face_name,
                #"location": {"top": location[0], "right": location[1], "bottom": location[2], "left": location[3]},
                "timestamp": datetime.now().isoformat()
            })
            try:
                self.mqtt_client.publish(self.MQTT_FACE_STATUS_TOPIC, message)
            except Exception as e:
                print(f"Error publishing MQTT face status: {e}")

    def _encode_image_to_base64(self, image_np_array):
        """Encodes an OpenCV image (numpy array) to a Base64 string."""

        if image_np_array is None or image_np_array.size == 0:
            return None
        # Encode as JPEG with quality 90 (can be adjusted)
        _, buffer = cv2.imencode('.jpg', image_np_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8'), "hello"

    def _publish_face_image(self, face_name, face_location_coords, base64_image_data, face_type): #ORIGINALLLLLLLLLLLLLLLLLLLLLLLLLLL
        """Publishes cropped face image data via MQTT."""
        if self.mqtt_client and self.mqtt_client.is_connected() and base64_image_data:
            topic = self.MQTT_DOOR_OPEN_TOPIC
            message_2 = json.dumps({
                "person": face_name,
                "type": face_type, # "known" or "unknown"
                "location": {"top": face_location_coords[0], "right": face_location_coords[1], "bottom": face_location_coords[2], "left": face_location_coords[3]},
                "timestamp": datetime.now().isoformat(),
                "image_data": base64_image_data
            })
            try:
                self.mqtt_client.publish(topic, message_2)
                print(f"MQTT: Sent {face_type} face image for {face_name} to topic {self.MQTT_IMAGE_TOPIC}")
            except Exception as e:
                print(f"Error publishing MQTT face image: {e}")
        elif not base64_image_data:
            print("Skipping MQTT image publish: No image data provided.")
        else:
            print("MQTT client not connected, cannot send face image.")


    def _process_frame_in_thread(self):
  
        print("Face processing thread started.")
        while not self.processing_stopped:
            try:


                frame = self.frame_queue.get(timeout=0.1)

                small_frame = cv2.resize(frame, (0, 0), fx=self.PROCESS_FRAME_SCALE, fy=self.PROCESS_FRAME_SCALE)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

                faces_info = []
                recognized_any_known_face_this_frame = False
                last_recognized_name_this_frame = "Unknown"

                current_processing_time = time.time() # Get current time for cooldown checks

                for i, (top, right, bottom, left) in enumerate(face_locations):
                    # Scale back up face locations to original frame size
                    top_orig = int(top / self.PROCESS_FRAME_SCALE)
                    right_orig = int(right / self.PROCESS_FRAME_SCALE)
                    bottom_orig = int(bottom / self.PROCESS_FRAME_SCALE)
                    left_orig = int(left / self.PROCESS_FRAME_SCALE)

                    name = "Unknown"
                    face_encoding = None # Initialize to None

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

                    faces_info.append({"name": name, "location": (top_orig, right_orig, bottom_orig, left_orig)})

                    # --- Prepare cropped face for MQTT image sending ---
                    # Ensure bounding box coordinates are within frame dimensions
                    h, w, _ = frame.shape
                    cropped_face = frame[max(0, top_orig):min(h, bottom_orig), max(0, left_orig):min(w, right_orig)]

                    # Send image via MQTT with cooldown
                    if name != "Unknown":
                        # For known faces, track cooldown per individual name
                        if (name not in self.last_sent_known_face_image_time or
                                current_processing_time - self.last_sent_known_face_image_time[name] > self.IMAGE_SEND_COOLDOWN):
                            base64_img = self._encode_image_to_base64(cropped_face)
                            self._publish_face_image(name, (top_orig, right_orig, bottom_orig, left_orig), base64_img, "known")
                            self.last_sent_known_face_image_time[name] = current_processing_time

                    else: # name is "Unknown"
                        # For unknown faces, send any unknown face image if the general cooldown is met
                        if current_processing_time - self.last_sent_unknown_face_image_time > self.IMAGE_SEND_COOLDOWN:
                            base64_img = self._encode_image_to_base64(cropped_face)
                            self._publish_face_image("Unknown", (top_orig, right_orig, bottom_orig, left_orig), base64_img, "unknown")
                            self.last_sent_unknown_face_image_time = current_processing_time

                            # Capture unknown faces to file system as before (now from cropped_face)
                            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # Use a hash of the encoding to make filename more unique to the face,
                            # or simply use a timestamp-based approach if unique filenames are desired for all captures.
                            # If face_encoding is None (no encoding generated), fallback to simple hash or ignore.
                            face_hash_part = hash(tuple(face_encoding.tobytes())) % 10000 if face_encoding is not None else np.random.randint(0, 10000)
                            filename = f"Unknown_{current_time_str}_{face_hash_part}.jpg"
                            filepath = os.path.join(self.UNKNOWN_FACES_OUTPUT_DIR, filename)
                            os.makedirs(self.UNKNOWN_FACES_OUTPUT_DIR, exist_ok=True)
                            cv2.imwrite(filepath, cropped_face)
                            #face_image = frame[top:bottom, left:right]
                            self.send_face_image(cropped_face)
                    # Call the function to send the unknown face image
                    
                    # Draw box and label (on the original frame, as it will be returned for display)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), color, 2)
                    cv2.putText(frame, name, (left_orig, top_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


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
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                self.processing_stopped = True # Stop thread on unhandled error

        print("Face processing thread stopped.")


    def run(self):
        """
        Main loop for video capture, putting frames into queue,
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
                    import traceback
                    traceback.print_exc() # Print full traceback for debugging


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