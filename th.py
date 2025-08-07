# thread100.py
import time
import cv2
import json
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
import base64
from insightface.app import FaceAnalysis
import logging
from threading import Thread
import queue

try:
    from dir import images_folder, known_face_encodings, known_faces_names
except ImportError:
    print("Warning: 'dir.py' or 'images_folder' not found. Using a mock function.")
    known_face_encodings = []
    known_faces_names = []
    def images_folder(directory_path, encodings_file_path, insightface_app_mock):
        print(f"Mock: Loading known faces from {directory_path} (encodings_file: {encodings_file_path})")
        return [], []

class FaceRecognitionSystem:
    """
    Face recognition with MQTT communication, optimized for low CPU usage and minimal latency at 5 FPS.
    """
    def __init__(self, config):
        self.config = config

        # --- Extract Config Values ---
        camera_settings = self.config.get("camera_settings", {})
        face_recognition_settings = self.config.get("face_recognition_settings", {})
        directories = self.config.get("directories", {})
        mqtt_settings = self.config.get("mqtt_settings", {})
        operation_region_settings = self.config.get("operation_region", {})
        optimization_settings = self.config.get("optimization_settings", {})

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

        # Store the region of interest (ROI) coordinates
        self.roi_x1 = int(operation_region_settings.get("x1", 0))
        self.roi_y1 = int(operation_region_settings.get("y1", 0))
        self.roi_x2 = int(operation_region_settings.get("x2", 0))
        self.roi_y2 = int(operation_region_settings.get("y2", 0))

        self.detection_interval_seconds = float(optimization_settings.get("detection_interval_seconds", 10.0))
        self.tracker_type = optimization_settings.get("tracker_type", "KCF")
        
        # --- Initialize Components ---
        self.logger = self.setup_logger('my_app', 'face_recognition.log')
        self._init_insightface()
        self._load_known_faces()
        self._init_mqtt_client()
        self._init_camera()

        # --- State Variables ---
        self.last_door_open_time = 0
        self.last_known_face_exit_time = 0
        self.num_frames_processed = 0
        self.start_time = time.time()
        self.last_frame_time = 0
        self.last_detection_time = 0
        self.trackers = []
        self.mqtt_queue = queue.Queue(maxsize=10)  # Queue for async MQTT publishing
        self.mqtt_thread = None
        self.mqtt_stopped = False  # Initialize mqtt_stopped
        self.last_sent_known_face_image_time = {}
        self.last_sent_unknown_face_image_time = 0
        self.last_image_send_time = 0

    def _init_insightface(self):
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(160, 160))  # Reduced det_size for lower CPU usage

    def _load_known_faces(self):
        """Loads known face encodings and names using InsightFace."""
        global known_face_encodings, known_faces_names
        known_faces_names, known_face_encodings_list = images_folder(
            self.KNOWN_FACES_DIR,
            self.ENCODINGS_FILE_PATH,
            self.app 
        )
        
        if not known_face_encodings_list:
            self.known_face_encodings = np.empty((0, 512), dtype=np.float32)
        else:
            self.known_face_encodings = np.array(known_face_encodings_list, dtype=np.float32)
        
        self.known_faces_names = known_faces_names
        self.logger.info(f"Total known faces loaded: {len(self.known_face_encodings)}")

    def _init_mqtt_client(self):
        """Initializes and connects MQTT client."""
        self.mqtt_client = mqtt.Client(client_id=self.MQTT_CLIENT_ID)
        if self.MQTT_USERNAME and self.MQTT_PASSWORD:
            self.mqtt_client.username_pw_set(username=self.MQTT_USERNAME, password=self.MQTT_PASSWORD)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                self.logger.info("Connected to MQTT broker successfully")
            else:
                self.logger.error(f"Failed to connect to MQTT broker with code {rc}: {mqtt.error_string(rc)}")

        def on_disconnect(client, userdata, rc):
            self.logger.error(f"MQTT disconnected with reason code {rc}: {mqtt.error_string(rc)}")
            if rc != 0:
                self.logger.info("Attempting to reconnect to MQTT broker...")
                try:
                    client.reconnect()
                except Exception as e:
                    self.logger.error(f"Reconnection failed: {e}")
        
        def on_log(client, userdata, level, buf):
            self.logger.debug(f"MQTT Log: {buf}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_disconnect = on_disconnect
        self.mqtt_client.on_log = on_log
        self.mqtt_client.on_publish = lambda client, userdata, mid: self.logger.info(f"MQTT: Message {mid} published")
        
        try:
            self.mqtt_client.connect(self.MQTT_BROKER_ADDRESS, self.MQTT_BROKER_PORT, 120)
            self.mqtt_client.loop_start()
        except Exception as e:
            self.logger.error(f"Could not connect to MQTT broker: {e}. MQTT functionality will be disabled.")
            self.mqtt_client = None

        # Start MQTT publishing thread
        self.mqtt_thread = Thread(target=self._mqtt_publish_thread, args=())
        self.mqtt_thread.daemon = True
        self.mqtt_thread.start()

    def _init_camera(self):
        """Initializes the camera with low-latency settings."""
        self.cap = cv2.VideoCapture(self.CAMERA_URL if self.CAMERA_URL else self.STREAM_ID)
        if not self.cap.isOpened():
            self.logger.error("Failed to open camera stream. System cannot operate without stream.")
            exit(1)
        
        # Optimize for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.FPS_LIMIT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.logger.info(f"Requested FPS: {self.FPS_LIMIT}, Actual FPS: {actual_fps}, "
                         f"Resolution: {actual_width}x{actual_height}, Buffer Size: {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

        # Test initial frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.logger.error("Camera stream is not producing frames. Check camera connection/URL. Exiting.")
            self.cap.release()
            exit(1)
        self.logger.info("Camera initialized successfully.")

    def _encode_image_to_base64(self, image_np_array):
        """Encodes an OpenCV image (numpy array) to a Base64 string."""
        if image_np_array is None or image_np_array.size == 0:
            return None
        _, buffer = cv2.imencode('.jpg', image_np_array, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode('utf-8')

    def _publish_face_image(self, face_name, base64_image_data, face_type, score, face_bbox):
        """Queues face image data for async MQTT publishing."""
        if self.mqtt_client and self.mqtt_client.is_connected() and base64_image_data:
            message = {
                "IDENTITY_VERIFIED": "FALSE" if face_type == "unknown" else "TRUE", 
                "timestamp": datetime.now().isoformat(),
                "image_data": base64_image_data,
                "recognized_person": face_name,
                "cosine_similarity": str(score),
                "face_coorindate": {
                    "x1": int(face_bbox[0]),
                    "y1": int(face_bbox[1]),
                    "x2": int(face_bbox[2]),
                    "y2": int(face_bbox[3])  
                },
                "operation_region": {
                    "x1": self.roi_x1,
                    "y1": self.roi_y1,
                    "x2": self.roi_x2,
                    "y2": self.roi_y2
                }
            }
            try:
                self.mqtt_queue.put_nowait((self.MQTT_FACE_TOPIC, json.dumps(message)))
            except queue.Full:
                self.logger.warning("MQTT queue full, dropping message.")

    def _mqtt_publish_thread(self):
        """Thread for asynchronous MQTT publishing."""
        self.logger.info("MQTT publishing thread started.")
        while True:
            try:
                topic, message = self.mqtt_queue.get(timeout=0.1)
                self.mqtt_client.publish(topic, message, qos=1)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in MQTT publish thread: {e}")
        self.logger.info("MQTT publishing thread stopped.")

    def _create_tracker(self):
        """Creates a tracker based on the configured tracker type."""
        if self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        else:
            self.logger.warning(f"Tracker type {self.tracker_type} not supported. Falling back to KCF.")
            return cv2.TrackerKCF_create()

    def _process_frame(self, frame, use_detection=True):
        """Processes a single frame for face recognition or tracking."""
        current_processing_time = time.time()
        faces = []
        updated_trackers = []

        if use_detection and current_processing_time - self.last_detection_time >= self.detection_interval_seconds:
            # Downscale frame for face detection
            small_frame = cv2.resize(frame, (0, 0), fx=self.PROCESS_FRAME_SCALE, fy=self.PROCESS_FRAME_SCALE)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb_frame)
            self.last_detection_time = current_processing_time
            self.trackers = []

            for face in faces:
                x1, y1, x2, y2 = [int(coord / self.PROCESS_FRAME_SCALE) for coord in face.bbox]
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                name = "Unknown"
                face_embedding = face.embedding
                best_similarity = 0.0

                if self.known_face_encodings.size > 0 and face_embedding is not None:
                    norm_face_embedding = face_embedding / np.linalg.norm(face_embedding)
                    norm_known_encodings = self.known_face_encodings / np.linalg.norm(self.known_face_encodings, axis=1, keepdims=True)
                    similarities = np.dot(norm_known_encodings, norm_face_embedding)
                    best_match_index = np.argmax(similarities)
                    best_similarity = similarities[best_match_index]

                    if best_similarity > self.COSINE_SIMILARITY_THRESHOLD:
                        name = self.known_faces_names[best_match_index]

                self.logger.info(f"Face detected: {name}, Similarity: {best_similarity} at {datetime.now().isoformat()}")

                tracker = self._create_tracker()
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker.init(frame, bbox)
                self.trackers.append({
                    'tracker': tracker,
                    'bbox': (x1, y1, x2, y2),
                    'name': name,
                    'similarity': best_similarity
                })

        else:
            # Update trackers on a smaller frame for speed
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 50% scale for tracking
            for tracker_info in self.trackers:
                tracker = tracker_info['tracker']
                ok, bbox = tracker.update(small_frame)
                if ok:
                    x1, y1, w, h = [int(v / 0.5) for v in bbox]  # Scale back to original size
                    x2, y2 = x1 + w, y1 + h
                    tracker_info['bbox'] = (x1, y1, x2, y2)
                    updated_trackers.append(tracker_info)
                else:
                    self.logger.info(f"Tracker for {tracker_info['name']} lost.")
            self.trackers = updated_trackers

        # Process tracked or detected faces
        can_send_unknown_faces = current_processing_time - self.last_sent_unknown_face_image_time > self.IMAGE_SEND_COOLDOWN
        unknown_face_sent_this_batch = False

        for tracker_info in self.trackers:
            x1, y1, x2, y2 = tracker_info['bbox']
            name = tracker_info['name']
            best_similarity = tracker_info['similarity']
            h, w, _ = frame.shape
            padding = 50  # Reduced padding for faster cropping
            x1_cropped = max(0, x1 - padding)
            y1_cropped = max(0, y1 - padding)
            x2_cropped = min(w, x2 + padding)
            y2_cropped = min(h, y2 + padding)

            cropped_face = frame[y1_cropped:y2_cropped, x1_cropped:x2_cropped]
            face_bbox = (x1, y1, x2, y2)

            if name != "Unknown":
                if (name not in self.last_sent_known_face_image_time or 
                        current_processing_time - self.last_sent_known_face_image_time[name] > self.IMAGE_SEND_COOLDOWN):
                    base64_img = self._encode_image_to_base64(cropped_face)
                    self._publish_face_image(name, base64_img, "known", best_similarity, face_bbox)
                    self.last_sent_known_face_image_time[name] = current_processing_time
            else:
                if can_send_unknown_faces and not unknown_face_sent_this_batch:
                    base64_img = self._encode_image_to_base64(cropped_face)
                    self._publish_face_image("Unknown", base64_img, "unknown", best_similarity, face_bbox)
                    unknown_face_sent_this_batch = True

            if unknown_face_sent_this_batch:
                self.last_sent_unknown_face_image_time = current_processing_time

    def run(self):
        """
        Main loop for capturing and processing frames at exactly 5 FPS with minimal latency.
        """
        frame_interval = 1.0 / self.FPS_LIMIT  # 0.2 seconds for 5 FPS

        try:
            while True:
                current_time = time.time()
                if current_time - self.last_frame_time < frame_interval:
                    time.sleep(frame_interval - (current_time - self.last_frame_time))
                    continue

                # Flush any buffered frames
                for _ in range(5):  # Clear up to 5 frames to ensure latest
                    self.cap.grab()
                ret, frame = self.cap.retrieve()
                self.last_frame_time = current_time

                if not ret or frame is None or frame.size == 0:
                    self.logger.warning("Failed to capture frame. Attempting to reconnect camera...")
                    self.cap.release()
                    self._init_camera()
                    time.sleep(1)  # Reduced reconnection delay
                    continue

                # Process frame immediately
                use_detection = (current_time - self.last_detection_time >= self.detection_interval_seconds)
                self._process_frame(frame, use_detection=use_detection)
                self.num_frames_processed += 1

        except KeyboardInterrupt:
            self.logger.info("\nKeyboardInterrupt detected. Signaling system to stop.")
        finally:
            self._cleanup()

    def setup_logger(self, loggername, log_file="face_recognition.log", level=logging.INFO):
        logger = logging.getLogger(loggername)
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(process)d - %(message)s')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(file_handler)
        return logger

    def _cleanup(self):
        """Performs cleanup operations."""
        self.logger.info("Performing cleanup...")
        self.mqtt_stopped = True
        if self.mqtt_thread and self.mqtt_thread.is_alive():
            self.mqtt_thread.join(timeout=2)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera released.")
        if self.mqtt_client:
            if self.mqtt_client.is_connected():
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                self.logger.info("MQTT client disconnected.")
            self.mqtt_client = None

        elapsed_time = time.time() - self.start_time
        if self.num_frames_processed > 0:
            fps = self.num_frames_processed / elapsed_time
            print(f"FPS: {fps:.2f}, Elapsed Time: {elapsed_time:.2f} seconds, Frames Processed: {self.num_frames_processed}")
        else:
            print("No frames processed.")
        print("Application exited.")