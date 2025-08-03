# thread2.py
from threading import Thread
import time
import cv2
import json
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
import queue
import base64
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
    Maybe using CONCURRENT
    """
    def __init__(self, stream_cap=0, camera_url=None):
        self.camera_source = camera_url if camera_url else stream_cap
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            print(f"{self.camera_source}. cap.isOpened() returned False.")
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
    Using a separate thread for face processing.
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
        self.COSINE_SIMILARITY_THRESHOLD = float(face_recognition_settings.get("COSINE_SIMILARITY_THRESHOLD"))
        self.DOOR_OPEN_COOLDOWN = float(face_recognition_settings.get("DOOR_OPEN_COOLDOWN"))
        self.RESET_DOOR_TIMER = float(face_recognition_settings.get("RESET_DOOR_TIMER"))
        self.IMAGE_SEND_COOLDOWN = float(face_recognition_settings.get("IMAGE_SEND_COOLDOWN"))

        # Directory Settings
        self.KNOWN_FACES_DIR = directories.get("KNOWN_FACES_DIR")
        self.ENCODINGS_FILE_PATH = directories.get("ENCODINGS_FILE")

        # MQTT Settings
        self.MQTT_BROKER_ADDRESS = mqtt_settings.get("MQTT_BROKER_ADDRESS")
        self.MQTT_BROKER_PORT = int(mqtt_settings.get("MQTT_BROKER_PORT"))
        self.MQTT_CLIENT_ID = mqtt_settings.get("MQTT_CLIENT_ID")
        self.MQTT_USERNAME = mqtt_settings.get("MQTT_USERNAME")
        self.MQTT_PASSWORD = mqtt_settings.get("MQTT_PASSWORD")
        self.MQTT_FACE_TOPIC = mqtt_settings.get("MQTT_FACE_TOPIC")

        # --- Initialize Components ---
        self._init_insightface() 
        self._load_known_faces()
        self._init_mqtt_client()
        self._init_webcam_stream()
        #self._init_opencv_window()
        #self._create_output_directories()

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

        self.app = FaceAnalysis(providers= ['CPUExecutionProvider']) # 'CUDAExecutionProvider'

        self.app.prepare(ctx_id=-1, det_size=(320, 320))


    def _load_known_faces(self):
        """Loads known face encodings and names using InsightFace."""
        global known_face_encodings, known_faces_names
        # Pass the initialized InsightFace app to images_folder
        known_faces_names, known_face_encodings_list = images_folder(
            self.KNOWN_FACES_DIR,
            self.ENCODINGS_FILE_PATH,
            self.app 
        )
        
        # Ensure known_face_encodings is always a 2D numpy array
        if not known_face_encodings_list:
            # If no encodings loaded, create an empty 2D array with expected embedding dimension (e.g., 512 for ArcFace)
            self.known_face_encodings = np.empty((0, 512), dtype=np.float32) # Assuming 512-dim embeddings
        else:
            self.known_face_encodings = np.array(known_face_encodings_list, dtype=np.float32)

        self.known_faces_names = known_faces_names

        print(f"Total known faces loaded: {len(self.known_face_encodings)}")


    def _init_mqtt_client(self):
        """Initializes and connects MQTT client."""
        self.mqtt_client = mqtt.Client(client_id=self.MQTT_CLIENT_ID)
        if self.MQTT_USERNAME and self.MQTT_PASSWORD:
            self.mqtt_client.username_pw_set(username=self.MQTT_USERNAME, password=self.MQTT_PASSWORD)


        def on_publish(client, userdata, mid):
            print(f"MQTT: Message with ID {mid} published.")

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



    def _encode_image_to_base64(self, image_np_array):
        """Encodes an OpenCV image (numpy array) to a Base64 string."""
        if image_np_array is None or image_np_array.size == 0:
            return None
        _, buffer = cv2.imencode('.jpg', image_np_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    def _publish_face_image(self, face_name, base64_image_data, face_type, score):
        """Publishes cropped face image data via MQTT."""
        if self.mqtt_client and self.mqtt_client.is_connected() and base64_image_data:
            # Ensure all face_location_coords are standard Python integers
            topic = self.MQTT_FACE_TOPIC
            if face_type != "unknown":
                
                message = json.dumps({

                    "IDENTITY_VERIFIED": "TRUE",
                    "timestamp": datetime.now().isoformat(),
                    "image_data": base64_image_data,
                    "recognized_person": face_name,
                    "cosine_similarity": str(score)
                })

            if face_type == "unknown":
                message = json.dumps({

                    "IDENTITY_VERIFIED": "FALSE",
                    "timestamp": datetime.now().isoformat(),
                    "image_data": base64_image_data,
                    "recognized_person": face_type,
                    "cosine_similarity": str(score)
                })
            try:
                self.mqtt_client.publish(topic, message, qos=1)
                print(f"MQTT: Sent {face_type} face image for {face_name} to topic {self.MQTT_FACE_TOPIC}")
            except Exception as e:
                print(f"Error publishing MQTT face image: {e}")

    def _process_frame_in_thread(self):
        print("Face processing thread started.")
        while not self.processing_stopped:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.app.get(rgb_frame)
                faces_info = []

                current_processing_time = time.time()

                can_send_unknown_faces = current_processing_time - self.last_sent_unknown_face_image_time > self.IMAGE_SEND_COOLDOWN #NEW
                unknown_face_sent_this_batch = False #NEW
                
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    h, w, _ = frame.shape
                    padding = 100
                    x1 = max(0, x1- padding)
                    y1 = max(0, y1- padding)
                    x2 = min(w, x2+ padding)
                    y2 = min(h, y2+ padding)
                    cropped_face = frame[y1:y2, x1:x2] 

                    name = "Unknown"
                    face_embedding = face.embedding 
                    best_similarity = 0.0 #NEW

                    # Perform comparison only if there are known faces and a valid embedding
                    if self.known_face_encodings.size > 0 and face_embedding is not None:
                        norm_face_embedding = face_embedding / np.linalg.norm(face_embedding)                        
                        norm_known_encodings = self.known_face_encodings / np.linalg.norm(self.known_face_encodings, axis=1, keepdims=True)
                        similarities = np.dot(norm_known_encodings, norm_face_embedding)
                        best_match_index = np.argmax(similarities) 
                        best_similarity = similarities[best_match_index]
                        # similarity = f"{similarities[best_match_index]:.2f}" #NEW REMOVE
                        
                        if best_similarity > self.COSINE_SIMILARITY_THRESHOLD:
                            name = self.known_faces_names[best_match_index]
                            # recognized_any_known_face_this_frame = True NEW
                            # last_recognized_name_this_frame = name NEW

                    sim_str = f"{best_similarity:.2f}"
                    print(f"Face comparison result: {name}, Similarity: {sim_str}")
                    faces_info.append({"name": name, "location": (y1, x2, y2, x1)})

                    # Send image via MQTT with cooldown
                    if name != "Unknown":
                        # For known faces, track cooldown per individual name
                        if (name not in self.last_sent_known_face_image_time or 
                                current_processing_time - self.last_sent_known_face_image_time.get(name, 0) > self.IMAGE_SEND_COOLDOWN):
                            base64_img = self._encode_image_to_base64(cropped_face)
                            self._publish_face_image(name, base64_img, "known", sim_str)
                            self.last_sent_known_face_image_time[name] = current_processing_time
                    else:
                        # For unknown faces, send any unknown face image if the general cooldown is met
                        if can_send_unknown_faces:
                        #if current_processing_time - self.last_sent_unknown_face_image_time > self.IMAGE_SEND_COOLDOWN: NEW REMOVE
                            base64_img = self._encode_image_to_base64(cropped_face)
                            self._publish_face_image("Unknown",base64_img, "unknown", sim_str)
                            unknown_face_sent_this_batch = True
                            # self.last_sent_unknown_face_image_time = current_processing_time NEW
                if unknown_face_sent_this_batch:
                    self.last_sent_unknown_face_image_time = current_processing_time
                    
                try:
                    self.results_queue.put_nowait((faces_info, frame)) # NEW REMOVED recognized_any_known_face_this_frame, last_recognized_name_this_frame,
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

        #
        try:
            while True:
                if self.webcam_stream.stopped:
                    print("Webcam stream stopped. Exiting main loop.")
                    break

                frame = self.webcam_stream.read()
                if frame is None or frame.size == 0:
                    print("Received empty/NONE frame from webcam. Exiting main loop.")
                    time.sleep(1)
                    continue

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass


                    self.num_frames_processed += 1

                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"ERROR: Unhandled exception in main loop while handling results: {e}")
                    import traceback
                    traceback.print_exc()

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

    
       

        elapsed_time = time.time() - self.start_time
        if self.num_frames_processed > 0:
            fps = self.num_frames_processed / elapsed_time
            print(f"FPS: {fps:.2f}, Elapsed Time: {elapsed_time:.2f} seconds, Frames Processed: {self.num_frames_processed}")
        else:
            print("No frames processed.")
        print("Application exited.")