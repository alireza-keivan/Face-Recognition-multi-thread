# thread2.py
from threading import Thread
import time
import cv2
import json
import os
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
import queue
import base64

# InsightFace imports
import insightface
from insightface.app import FaceAnalysis

try:
    # images_folder will now accept the insightface_app instance
    from dir import images_folder, known_face_encodings, known_faces_names
except ImportError:
    print("Warning: 'dir.py' or 'images_folder' not found. Using a mock function.")
    known_face_encodings = []
    known_faces_names = []
    def images_folder(directory_path, encodings_file_path, insightface_app_mock): # Mock must accept new arg
        print(f"Mock: Loading known faces from {directory_path} (encodings_file: {encodings_file_path})")
        return [], []


class WebcamStream:
    """
    Multi-threaded video stream processing.
    """
    def __init__(self, stream_cap=0, camera_url=None):
        self.camera_source = camera_url if camera_url else stream_cap
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            print(f"[ERROR]: Error accessing webcam stream at {self.camera_source}. cap.isOpened() returned False.")
            self.grabbed = False
            self.frame = None
            self.stopped = True
            return
        
        fps_input_stream = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"FPS of webcam hardware/input stream: {fps_input_stream}")

        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read from camera during initial read.')
            self.frame = None
            self.stopped = True
            return



        self.stopped = False
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                break
            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed:
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
        self.RECOGNITION_TOLERANCE = float(face_recognition_settings.get("RECOGNITION_TOLERANCE")) # Kept for backward compatibility if needed, but COSINE_SIMILARITY_THRESHOLD is primary
        self.COSINE_SIMILARITY_THRESHOLD = float(face_recognition_settings.get("COSINE_SIMILARITY_THRESHOLD")) # NEW
        self.DOOR_OPEN_COOLDOWN = float(face_recognition_settings.get("DOOR_OPEN_COOLDOWN"))
        self.RESET_DOOR_TIMER = float(face_recognition_settings.get("RESET_DOOR_TIMER"))
        self.IMAGE_SEND_COOLDOWN = float(face_recognition_settings.get("IMAGE_SEND_COOLDOWN"))

        # Directory Settings
        self.KNOWN_FACES_DIR = directories.get("KNOWN_FACES_DIR")
        self.KNOWN_FACES_OUTPUT_DIR = directories.get("output_path")
        self.UNKNOWN_FACES_OUTPUT_DIR = directories.get("unknown_output_path")
        self.ENCODINGS_FILE_PATH = directories.get("ENCODINGS_FILE")


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
        self._init_insightface() # Initialize InsightFace before loading known faces
        self._load_known_faces()
        self._init_mqtt_client()
        self._init_webcam_stream()
        self._init_opencv_window()
        self._create_output_directories()

        # --- Queues for Thread Communication ---
        self.frame_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=2)

        # --- State Variables ---
        self.last_door_open_time = 0
        self.last_known_face_exit_time = 0
        self.num_frames_processed = 0
        self.start_time = time.time()
        self.processing_stopped = False
        
        # --- Image Sending Cooldowns ---
        self.last_sent_known_face_image_time = {}
        self.last_sent_unknown_face_image_time = 0
        self.last_image_send_time = 0


    def _init_insightface(self):
        """Initializes the InsightFace FaceAnalysis app."""
        print("Initializing InsightFace FaceAnalysis app...")
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # Prepare the app with detection and recognition models.
        # det_size: Input size for the detection model. Adjust for performance vs. accuracy.
        # ctx_id: 0 for GPU, -1 for CPU.
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace FaceAnalysis app ready.")

    def send_unknown_face_image(self, face_image_np):
        """
        Encodes a face image to Base64 and publishes it to the MQTT unknown face topic.
        """
        current_time = time.time()
        if (current_time - self.last_image_send_time) < self.IMAGE_SEND_COOLDOWN:
            return

        if self.mqtt_client is None or not self.mqtt_client.is_connected():
            print("MQTT: Client not connected, cannot publish unknown face image.")
            return

        if face_image_np is None or face_image_np.size == 0:
            print("Error: face_image_np is empty or None. Cannot encode for MQTT.")
            return

        try:
            _, buffer = cv2.imencode('.jpg', face_image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            payload = json.dumps({
                "image": encoded_image,
                "timestamp": datetime.now().isoformat()
            })

            result = self.mqtt_client.publish(self.MQTT_UNKNOWN_FACE_TOPIC, payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"MQTT: Successfully published unknown face image (len: {len(encoded_image)} bytes) to topic '{self.MQTT_UNKNOWN_FACE_TOPIC}'")
                self.last_image_send_time = current_time
            else:
                print(f"MQTT: Failed to publish unknown face image. Return code: {result.rc}")

        except Exception as e:
            print(f"MQTT: Error during encoding or publishing unknown face image: {e}")


    def _load_known_faces(self):
        """Loads known face encodings and names using InsightFace."""
        global known_face_encodings, known_faces_names
        # Pass the initialized InsightFace app to images_folder
        known_faces_names, known_face_encodings_list = images_folder(
            self.KNOWN_FACES_DIR,
            self.ENCODINGS_FILE_PATH,
            self.app # Pass the InsightFace app here
        )
        
        # Ensure known_face_encodings is always a 2D numpy array
        if not known_face_encodings_list:
            # If no encodings loaded, create an empty 2D array with expected embedding dimension (e.g., 512 for ArcFace)
            self.known_face_encodings = np.empty((0, 512), dtype=np.float32) # Assuming 512-dim embeddings
        else:
            self.known_face_encodings = np.array(known_face_encodings_list, dtype=np.float32)

        self.known_faces_names = known_faces_names

        print(f"Total known faces loaded: {len(self.known_face_encodings)}")
        if len(self.known_face_encodings) == 0:
            print("Please check the 'known_faces' directory and the image files within it.")
        print("---------------------------------------")

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


        def on_publish(client, userdata, mid):
            print(f"MQTT: Message with ID {mid} published.")

        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_publish = on_publish # Assign the on_publish callback

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
        """Publishers general face detection status to MQTT."""
        if self.mqtt_client and self.mqtt_client.is_connected():
            # Ensure all location coordinates are standard Python integers
            loc_top = int(location[0])
            loc_right = int(location[1])
            loc_bottom = int(location[2])
            loc_left = int(location[3])

            message = json.dumps({
                "person": face_name,
                "location": {"top": loc_top, "right": loc_right, "bottom": loc_bottom, "left": loc_left},
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
        _, buffer = cv2.imencode('.jpg', image_np_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    def _publish_face_image(self, face_name, face_location_coords, base64_image_data, face_type):
        """Publishes cropped face image data via MQTT."""
        if self.mqtt_client and self.mqtt_client.is_connected() and base64_image_data:
            # Ensure all face_location_coords are standard Python integers
            loc_top = int(face_location_coords[0])
            loc_right = int(face_location_coords[1])
            loc_bottom = int(face_location_coords[2])
            loc_left = int(face_location_coords[3])

            message = json.dumps({
                "person": face_name,
                "type": face_type, # "known" or "unknown"
                "location": {"top": loc_top, "right": loc_right, "bottom": loc_bottom, "left": loc_left},
                "timestamp": datetime.now().isoformat(),
                "image_data": base64_image_data
            })
            try:
                self.mqtt_client.publish(self.MQTT_IMAGE_TOPIC, message)
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

                # Convert to RGB (InsightFace expects RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- InsightFace: Get all faces (detection and embedding) ---
                faces = self.app.get(rgb_frame)

                faces_info = []
                recognized_any_known_face_this_frame = False
                last_recognized_name_this_frame = "Unknown"

                current_processing_time = time.time()

                for face in faces:
                    # Bounding box coordinates from InsightFace (x1, y1, x2, y2)
                    x1, y1, x2, y2 = face.bbox.astype(int)

                    # Ensure bounding box coordinates are within frame dimensions
                    h, w, _ = frame.shape
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2) # FIXED: Was min(h, h), now min(h, y2)

                    cropped_face = frame[y1:y2, x1:x2] # Crop face from original frame
            

                    name = "Unknown"
                    face_embedding = face.embedding # Get the ArcFace embedding

                    # Perform comparison only if there are known faces and a valid embedding
                    if self.known_face_encodings.size > 0 and face_embedding is not None:
      
                        # Calculate cosine similarities between current face and known faces
                        # Normalize embeddings for cosine similarity calculation
                        norm_face_embedding = face_embedding / np.linalg.norm(face_embedding)
                        
                        # Ensure known_face_encodings is normalized
                        # This normalization should ideally happen once when loaded, but doing it here for safety
                        # if the known_face_encodings might not be normalized from dir.py
                        norm_known_encodings = self.known_face_encodings / np.linalg.norm(self.known_face_encodings, axis=1, keepdims=True)

                        similarities = np.dot(norm_known_encodings, norm_face_embedding)

                        best_match_index = np.argmax(similarities) # Max similarity is the best match
                        
                        # Compare similarity to the threshold
                        if similarities[best_match_index] > self.COSINE_SIMILARITY_THRESHOLD:
                            name = self.known_faces_names[best_match_index]
                            recognized_any_known_face_this_frame = True
                            last_recognized_name_this_frame = name
                        print(f"DEBUG: Face comparison result: {name}, Similarity: {similarities[best_match_index]:.2f}")
                    

                    # Store info in (top, right, bottom, left) format for consistency with old code/drawing
                    faces_info.append({"name": name, "location": (y1, x2, y2, x1)})

                    # Send image via MQTT with cooldown
                    if name != "Unknown":
                        # For known faces, track cooldown per individual name
                        if (name not in self.last_sent_known_face_image_time or
                                current_processing_time - self.last_sent_known_face_image_time[name] > self.IMAGE_SEND_COOLDOWN):
                            base64_img = self._encode_image_to_base64(cropped_face)
                            self._publish_face_image(name, (y1, x2, y2, x1), base64_img, "known")
                            self.last_sent_known_face_image_time[name] = current_processing_time
                    else: # name is "Unknown"
                        # For unknown faces, send any unknown face image if the general cooldown is met
                        if current_processing_time - self.last_sent_unknown_face_image_time > self.IMAGE_SEND_COOLDOWN:
                            base64_img = self._encode_image_to_base64(cropped_face)
                            self._publish_face_image("Unknown", (y1, x2, y2, x1), base64_img, "unknown")
                            self.last_sent_unknown_face_image_time = current_processing_time

                            # Capture unknown faces to file system
                            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            face_hash_part = hash(face_embedding.tobytes()) % 10000 if face_embedding is not None else np.random.randint(0, 10000)
                            filename = f"Unknown_{current_time_str}_{face_hash_part}.jpg"
                            filepath = os.path.join(self.UNKNOWN_FACES_OUTPUT_DIR, filename)
                            os.makedirs(self.UNKNOWN_FACES_OUTPUT_DIR, exist_ok=True)
                            cv2.imwrite(filepath, cropped_face)
                            self.send_unknown_face_image(cropped_face)

                    # Draw box and label (on the original frame, as it will be returned for display)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


                try:
                    self.results_queue.put_nowait((faces_info, recognized_any_known_face_this_frame, last_recognized_name_this_frame, frame))
                except queue.Full:
                    pass
            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                print(f"ERROR: Unhandled exception in face processing thread: {e}")
                import traceback
                traceback.print_exc()
                self.processing_stopped = True

        print("Face processing thread stopped.")

    def run(self):
        """
        Main loop for video capture, putting frames into queue,
        """
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

                frame = self.webcam_stream.read()
                if frame is None or frame.size == 0:
                    print("Received empty/NONE frame from webcam. Exiting main loop.")
                    break

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

                try:
                    faces_info, recognized_any_known_face, last_recognized_name, processed_frame = self.results_queue.get_nowait()

                    current_time = time.time()

                    # --- Door Control Logic ---
                    if recognized_any_known_face:
                        self.last_known_face_exit_time = 0

                        if current_time - self.last_door_open_time > self.DOOR_OPEN_COOLDOWN:
                            self._send_door_open_command(recognized_person_name=last_recognized_name)
                            self.last_door_open_time = current_time
                    else:
                        if self.last_known_face_exit_time == 0:
                            self.last_known_face_exit_time = current_time
                        elif current_time - self.last_known_face_exit_time > self.RESET_DOOR_TIMER:
                            self.last_door_open_time = 0

                    # --- MQTT Status Update for all faces ---
                    for face_data in faces_info:
                        self._publish_face_status(face_data["name"], face_data["location"])

                    # --- Display Processed Frame ---
                    cv2.imshow(self.OUTPUT_WINDOW_NAME, processed_frame)
                    self.num_frames_processed += 1

                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"ERROR: Unhandled exception in main loop while handling results: {e}")
                    import traceback
                    traceback.print_exc()

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
        self.processing_stopped = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
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

        cv2.destroyAllWindows()
       

        elapsed_time = time.time() - self.start_time
        if self.num_frames_processed > 0:
            fps = self.num_frames_processed / elapsed_time
            print(f"FPS: {fps:.2f}, Elapsed Time: {elapsed_time:.2f} seconds, Frames Processed: {self.num_frames_processed}")
        else:
            print("No frames processed.")
        print("Application exited.")
