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
import logging
import gc


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
    def __init__(self, stream_cap=0, camera_url=None): # NEW
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
        operation_region_settings = self.config.get("operation_region", {})
        optimization_settings = self.config.get("optimization_settings", {})

        # Camera Settings
        self.CAMERA_URL = camera_settings.get("url")
        self.STREAM_ID = int(camera_settings.get("stream_id"))
        self.FPS_LIMIT = float(camera_settings.get("fps_limit"))
        self.OUTPUT_WINDOW_NAME = camera_settings.get("OUTPUT_WINDOW_NAME")
        self.WINDOW_WIDTH = int(camera_settings.get("WINDOW_WIDTH"))
        self.WINDOW_HEIGHT = int(camera_settings.get("WINDOW_HEIGHT")),
        self.RESIZE_SCALE = float(camera_settings.get("resize_scale", 1.0))

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
        self._init_webcam_stream()

        # Initialize Tracker List
        self.trackers = []  # List of dicts: {'tracker': KCF, 'bbox': (x1, y1, x2, y2), 'name': str, 'similarity': float}
        self.last_detection_time = 0
        self.processing_stopped = False

        # --- Queues for Thread Communication ---
        self.frame_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=2)

        # --- State Variables ---
        self.last_door_open_time = 0
        self.last_known_face_exit_time = 0
        self.num_frames_processed = 0
        self.start_time = time.time()
        
        self.last_sent_known_face_image_time = {}
        self.last_sent_unknown_face_image_time = 0
        self.last_image_send_time = 0

        self.cv2_tracker_type = self.tracker_type
        if self.cv2_tracker_type == 'KCF':
            self.tracker_constructor = cv2.TrackerKCF_create
    def _init_insightface(self):

        self.app = FaceAnalysis(name= 'buffalo_l', providers= ['CPUExecutionProvider']) # 'CUDAExecutionProvider'
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
        
        if not known_face_encodings_list:
            # If no encodings loaded, create an empty 2D array with expected embedding dimension (e.g., 512 for ArcFace)
            self.known_face_encodings = np.empty((0, 512), dtype=np.float32) # Assuming 512-dim embeddings
        else:
            self.known_face_encodings = np.array(known_face_encodings_list, dtype=np.float32)

        if self.known_face_encodings.size > 0:
            self.known_face_encodings /= np.linalg.norm(self.known_face_encodings, axis=1, keepdims=True)
        
        self.known_faces_names = known_faces_names
        self.logger.info(f"Total known faces loaded: {len(self.known_face_encodings)}")

    def setup_logger(self, loggername,  log_file= "face_recognition.log", level = logging.INFO):
        logger = logging.getLogger(loggername)
        logger.setLevel(level)

        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(process)d - %(message)s')
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            return logger

    def _init_mqtt_client(self):
        """Initializes and connects MQTT client."""
        self.mqtt_client = mqtt.Client(client_id=self.MQTT_CLIENT_ID)
        if self.MQTT_USERNAME and self.MQTT_PASSWORD:
            self.mqtt_client.username_pw_set(username=self.MQTT_USERNAME, password=self.MQTT_PASSWORD)

        def on_publish(client, userdata, mid):
            logging.info(f"MQTT: Message with ID {mid} published.")

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info("Connected to MQTT broker successfully")
            else:
                logging.error(f"Failed to connect to MQTT broker with code {rc}: {mqtt.error_string(rc)}") ###################################

        def on_disconnect(client, userdata, rc):
            logging.error(f"MQTT disconnected with reason code{rc}: {mqtt.error_string(rc)}")
            if rc!=0:
                logging.info("attempting to reconnect to MQTT broker...")
                try:
                    client.reconnect()
                except Exception as e:
                    logging.error(f"Reconnection failed: {e}")
        
        def on_log(client, userdata, level, buf):
            logging.debug(f"MQTT Log: {buf}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_disconnect = on_disconnect
        self.mqtt_client.on_log = on_log
        self.mqtt_client.on_publish = lambda client, userdata, mid: logging.info(f"MQTT: Message {mid} published")
        

        try:
            self.mqtt_client.connect(self.MQTT_BROKER_ADDRESS, self.MQTT_BROKER_PORT, 120)
            self.mqtt_client.loop_start() # Start background thread for MQTT
        except Exception as e:
            self.logger.error(f"Could not connect to MQTT broker: {e}. MQTT functionality will be disabled.")
            self.mqtt_client = None

    def _init_webcam_stream(self):
        """Initializes and starts the webcam stream."""
        self.webcam_stream = WebcamStream(stream_cap=self.STREAM_ID, camera_url=self.CAMERA_URL)
        if self.webcam_stream.stopped:
            self.logger.error("Failed to start webcam stream. System cannot operate without stream.")
            exit(1)
        self.webcam_stream.start()
        time.sleep(1)
        if self.webcam_stream.read() is None:
            self.logger.error("Webcam stream is not producing frames. Check camera connection/URL. Exiting.")
            self.webcam_stream.stop()
            exit(1)
        self.logger.info("WebcamStream ready.")

    def _encode_image_to_base64(self, image_np_array):
        """Encodes an OpenCV image (numpy array) to a Base64 string."""
        if image_np_array is None or image_np_array.size == 0:
            return None
        _, buffer = cv2.imencode('.jpg', image_np_array, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return base64.b64encode(buffer).decode('utf-8')

    def _publish_face_image(self, face_name, base64_image_data, face_type, score, face_bbox):
        """Publishes cropped face image data via MQTT."""

        if self.mqtt_client and self.mqtt_client.is_connected() and base64_image_data:

            topic = self.MQTT_FACE_TOPIC
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
                    "operation_region":{
                        "x1": self.roi_x1,
                        "y1": self.roi_y1,
                        "x2": self.roi_x2,
                        "y2": self.roi_y2
                    }
                }
            try:
                message = json.dumps(message)
                self.mqtt_client.publish(topic, message, qos=1)
            except Exception as e:
                logging.error(f"Error publishing MQTT face image: {e}")


    def _process_frame_in_thread(self):
        self.logger.info("Face processing thread started.")
        while not self.processing_stopped:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                current_time = time.time()
                
                # Check if it's time to run a full face detection
                if (current_time - self.last_detection_time) > self.detection_interval_seconds:
                    #self.logger.info("Performing full face detection.")
                    self.last_detection_time = current_time
                    self.trackers = [] # Clear old trackers
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(rgb_frame)
                    
                    for face in faces:
                        x1, y1, x2, y2 = face.bbox.astype(int)
                        face_embedding = face.embedding
                        
                        tracker = self.tracker_constructor()
                        bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        tracker.init(frame, bbox)
                        
                        name = "Unknown"
                        best_similarity = 0.0
                        if self.known_face_encodings.size > 0 and face_embedding is not None:
                            norm_face_embedding = face_embedding / np.linalg.norm(face_embedding)
                            norm_known_encodings = self.known_face_encodings / np.linalg.norm(self.known_face_encodings, axis=1, keepdims=True)
                            similarities = np.dot(norm_known_encodings, norm_face_embedding)
                            best_match_index = np.argmax(similarities)
                            best_similarity = similarities[best_match_index]
                            
                            if best_similarity > self.COSINE_SIMILARITY_THRESHOLD:
                                name = self.known_faces_names[best_match_index]
                        
                        # Apply cooldown logic to both known and unknown faces
                        can_send_image = False
                        if name == "Unknown":
                            if (current_time - self.last_sent_unknown_face_image_time) > self.IMAGE_SEND_COOLDOWN:
                                can_send_image = True
                                self.last_sent_unknown_face_image_time = current_time
                        else: # Known face
                            if (name not in self.last_sent_known_face_image_time or 
                                (current_time - self.last_sent_known_face_image_time.get(name, 0)) > self.IMAGE_SEND_COOLDOWN):
                                can_send_image = True
                                self.last_sent_known_face_image_time[name] = current_time
                       
                        if can_send_image:
                            h, w, _ = frame.shape
                            padding = 100
                            x1_cropped = max(0, x1 - padding)
                            y1_cropped = max(0, y1 - padding)
                            x2_cropped = min(w, x2 + padding)
                            y2_cropped = min(h, y2 + padding)
                            cropped_face = frame[y1_cropped:y2_cropped, x1_cropped:x2_cropped]
                            
                            base64_img = self._encode_image_to_base64(cropped_face)
                            face_bbox = (x1, y1, x2, y2)
                            face_type = "known" if name != "Unknown" else "unknown"
                            self._publish_face_image(name, base64_img, face_type, best_similarity, face_bbox)
                        
                        self.trackers.append({
                            'tracker': tracker,
                            'name': name,
                            'similarity': best_similarity,
                            'last_sent_time': current_time # This now accurately reflects the initial sent time
                        })
                        
                else: # Update existing trackers
                    self.logger.debug("Updating trackers.")
                    trackers_to_remove = []
                    for tracker_info in self.trackers:
                        success, bbox = tracker_info['tracker'].update(frame)
                        
                        if success:
                            x1, y1, w, h = [int(v) for v in bbox]
                            x2, y2 = x1 + w, y1 + h
                            
                            face_bbox = (x1, y1, x2, y2)
                            name = tracker_info['name']
                            similarity = tracker_info['similarity']

                            # Only publish if it's a known face and cooldown is over
                            if name != "Unknown":
                                if (current_time - tracker_info['last_sent_time']) > self.IMAGE_SEND_COOLDOWN:
                                    cropped_face = frame[y1:y2, x1:x2]
                                    base64_img = self._encode_image_to_base64(cropped_face)
                                    self._publish_face_image(name, base64_img, "known", similarity, face_bbox)
                                    tracker_info['last_sent_time'] = current_time
                        else:
                            trackers_to_remove.append(tracker_info)
                    
                    for tracker_info in trackers_to_remove:
                        self.trackers.remove(tracker_info)
                    
               
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"ERROR: Unhandled exception in face processing thread: {e}")
                import traceback
                traceback.print_exc()
                self.processing_stopped = True
                
        self.logger.info("Face processing thread stopped.")

    def run(self):
        """
        Main loop for video capture, putting frames into queue at a limited FPS.
        """
        self.processing_thread = Thread(target=self._process_frame_in_thread, args=())
        self.processing_thread.daemon = True
        self.processing_thread.start()

        last_frame_time = time.time()
        frame_interval = 1.0 / self.FPS_LIMIT
      

        try:
            while True:
                if self.webcam_stream.stopped:
                    print("Webcam stream has stopped. Attempting to restart it...")
                    self.webcam_stream.stop()
                    self._init_webcam_stream()
                    time.sleep(5)
                    print("running for another time")
                    continue

                frame = self.webcam_stream.read()
              
                if frame is None or frame.size == 0:
                    print("Received an empty frame from webcam. Trying to restart.")
                    time.sleep(1)
                    continue
                    
                current_time = time.time()
                if (current_time - last_frame_time) > frame_interval:
                    try:
                        if self.RESIZE_SCALE != 1.0:
                            frame = cv2.resize(frame, (0, 0), fx=self.RESIZE_SCALE, fy=self.RESIZE_SCALE)
                        
                        self.frame_queue.put_nowait(frame) 
                        self.num_frames_processed += 1
                        last_frame_time = current_time
                       
                    except queue.Full:
                        pass
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Signaling system to stop.")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Performs cleanup operations."""
        logging.info("Performing cleanup...")
        self.processing_stopped = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
            if self.processing_thread.is_alive():
                self.logger.warning("Face processing thread did not terminate gracefully.")

        if not self.webcam_stream.stopped:
            self.webcam_stream.stop()

        if self.mqtt_client:
            if self.mqtt_client.is_connected():
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                logging.info("MQTT client disconnected.")
            self.mqtt_client = None
        if hasattr(self, 'mqtt_thread') and self.mqtt_thread.is_alive():
            self.mqtt_thread.join(timeout=5)
        elapsed_time = time.time() - self.start_time
        if self.num_frames_processed > 0:
            fps = self.num_frames_processed / elapsed_time
            self.logger.info(f"FPS: {fps:.2f}, Elapsed Time: {elapsed_time:.2f} seconds, Frames Processed: {self.num_frames_processed}")
        else:
            self.logger.info("No frames processed.")
        self.logger.info("Application exited.")