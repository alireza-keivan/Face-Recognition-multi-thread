# My **Headless Face Recognition** project with  multiple use cases!
Hey, so I built this face recognition system using InsightFace for spotting and identifying faces, OpenCV for handling video, and MQTT to send data to a server. It’s pretty dope for stuff like unlocking doors, keeping an eye on security, or even just playing around with some cool computer vision tech. The whole thing runs in real-time, processes video streams, and shoots results to an MQTT broker. I’ve made it super tweakable with a config.json file and kept it snappy with some threading tricks. This README’s gonna walk you through it like a tutorial, but it’s also got enough meat to impress the academic crowd.

---
### What’s This Thing Do?

Basically, it grabs video from a camera (like an RTSP stream or a webcam), spots faces with InsightFace’s deep learning magic, and figures out if they’re someone we know or not. Then it sends the results—like who’s in the frame, their face coordinates, and even a base64-encoded image—over MQTT. It’s got some smart optimizations, like scaling down frames and only doing full detection every few seconds, so it doesn’t hog your CPU. Plus, it’s tough—handles errors like a champ and logs everything for debugging.

---

## Where Can You Use It?

This thing’s got a bunch of cool uses:

* Access Control: Spot a known face in a specific area and send a signal to open a door. Super handy for secure buildings.

* Security Monitoring: Catch unknown faces and send alerts, perfect for keeping tabs on a place.

* Attendance Tracking: Automatically log who’s showing up at school or work. No more roll call!

* Retail Stuff: Figure out who’s coming back to a store or track customer vibes for marketing.

* Research Toy: If you’re into computer vision or IoT, this is a solid starting point for messing with face recognition or real-time systems.
---

## How I Built It

I went for a modular setup with some threading to keep things smooth. Here’s the breakdown:

#### Face Detection and Recognition

* Tool: [InsightFace’s](https://github.com/deepinsight/insightface) [buffalo_l model](https://github.com/deepinsight/insightface/tree/master/model_zoo) for detecting faces and making 512-dimensional embeddings (fancy face fingerprints, basically).
* How It Works: Frames get converted to RGB, scaled down, and fed to InsightFace. It spits out embeddings, which I compare to known faces using cosine similarity.
* Trick: I use a PROCESS_FRAME_SCALE to shrink frames and only do full detection every few seconds (detection_interval_seconds) to save juice.

#### Face Tracking

* Tool: OpenCV’s KCF tracker to keep tabs on faces between detections.

* How It Works: Once a face is spotted, I set up a tracker to follow it until the next full detection. Saves a ton of processing power.

#### Saving Data

* Encodings: Known face embeddings get saved in a known_faces_data.pkl file with pickle. No need to re-encode every time.

* Images: Known face pics live in KNOWN_FACES_DIR. If there’s no encoding file, I process these images at startup.

#### MQTT Stuff

* What It Does: Sends face data (name, similarity score, coords, and a base64 image) to an MQTT broker.
*Cooldown: I added timers (IMAGE_SEND_COOLDOWN, DOOR_OPEN_COOLDOWN) so it doesn’t spam the broker with messages.

#### Threading

* Webcam Thread: A WebcamStream class runs in its own thread to grab video frames without slowing things down.
* Processing Thread: Another thread handles face detection and recognition, pulling frames from a queue to keep it real-time.

#### Error Handling
* I made it tough with error checks for file issues, MQTT hiccups, and camera fails. Everything gets logged to face_recognition.log so you can see what’s up.

#### The Config File (config.json)
* The config.json is where you tweak everything. It’s super flexible, so you don’t have to dig into the code. Here’s what it looks like:
---
#### Config Layout

``` json
{
    "camera_settings": {
        "url": "rtsp://192.168...",
        "stream_id": "0",
        "fps_limit": 5.0,
        "OUTPUT_WINDOW_NAME": "Face Recognition Stream",
        "WINDOW_WIDTH": 1200,
        "WINDOW_HEIGHT": 1000,
        "resize_scale": 0.5
    },
    "face_recognition_settings": {
        "PROCESS_FRAME_SCALE": 0.2,
        "RECOGNITION_TOLERANCE": 0.5,
        "COSINE_SIMILARITY_THRESHOLD": 0.35,
        "DOOR_OPEN_COOLDOWN": 10,
        "RESET_DOOR_TIMER": 10,
        "IMAGE_SEND_COOLDOWN": 5,
        "detection_width": 320,
        "detection_height": 320
    },
    "directories": {
        "KNOWN_FACES_DIR": "/.../captured_known_faces",
        "ENCODINGS_FILE": "/.../known_faces_data.pkl"
    },
    "mqtt_settings": {
        "MQTT_BROKER_ADDRESS": "192.168...",
        "MQTT_BROKER_PORT": 1883,
        "MQTT_CLIENT_ID": "...",
        "MQTT_USERNAME": "...",
        "MQTT_PASSWORD": "...",
        "MQTT_FACE_TOPIC": "v1/.../telemetry"
    },
    "optimization_settings": {
        "detection_interval_seconds": 5.0,
        "tracker_type": "KCF"
    },
    "operation_region": {
        "x1": 600,
        "x2": 2000,
        "y1": 10,
        "y2": 2500,
        "door_num": 1
    }
}
```
### Key Settings
camera_settings:

* url: Your camera’s RTSP link or index (like 0 for a webcam).
* fps_limit: Caps how many frames we process per second (e.g., 5.0).
* resize_scale: Shrinks frames for display (e.g., 0.5 for half size).
* face_recognition_settings:
* PROCESS_FRAME_SCALE: Shrinks frames for detection (e.g., 0.2 for 20% size).
* COSINE_SIMILARITY_THRESHOLD: How close a face embedding needs to be to match a known face (e.g., 0.35).
* IMAGE_SEND_COOLDOWN: Seconds to wait before sending another face image via MQTT.
* directories:
* KNOWN_FACES_DIR: Where your known face images live.
* ENCODINGS_FILE: Where we save the face embeddings.
* mqtt_settings:
* Stuff like broker address, port, and credentials for MQTT.
* operation_region:
* ROI coords (x1, y1, x2, y2) and a door number for access control.
* optimization_settings:
* detection_interval_seconds: How often we do full face detection.
* tracker_type: Which tracker we’re using (KCF’s my go-to).
---
#### Twiking it
* Swap out the url for your camera.

* Point KNOWN_FACES_DIR and ENCODINGS_FILE to your folders.

* Update MQTT creds and ROI coords to fit your setup.
---
Project Files

* config.json: All the settings live here.

* dir.py: Loads and encodes known faces from images.

* recognizer.py: The main script to kick things off.

* thread.py: Does the heavy lifting—video streaming, face recognition, MQTT publishing.

* known_faces_data.pkl: Stores face embeddings (gets created automatically).

* captured_known_faces/: Folder for your known face images.

* face_recognition.log: Logs everything for debugging.
---
## What You Need to Run It
* Python 3.8+: Gotta have it.
#### Libraries:
* insightface: For face detection and recognition.
* numpy: For math and embeddings.
* paho-mqtt: For talking to the MQTT broker.
* pickle: For saving encodings.
## Hardware:
A camera (RTSP or webcam) and maybe an MQTT broker.
---
Install the libraries:
``` bash
pip install opencv-python insightface numpy paho-mqtt
```
#### For InsightFace, grab the buffalo_l model:
``` bash
pip install insightface
```
#### How to Get It Running

1. Grab the Code:
``` bash
git clone <https://github.com/alireza-keivan/Face-Recognition-multi-thread/blob/alireza-keivan/README.md>
cd face_recognition
```
2. Set Up Known Faces:
* Drop .png, .jpg, or .jpeg images of known people into KNOWN_FACES_DIR.
* Make sure each image has one face to avoid confusion.
3. Tweak config.json:
* Update the camera URL, MQTT settings, and ROI coords for your setup.
4. Fire It Up:
``` bash
  python recognizer.py
  ```
* It’ll load known faces, start the camera, connect to MQTT, and process video.
* Check face_recognition.log for what’s happening.

5. Watch the Output:
* Face data goes to your MQTT topic.
* Logs in face_recognition.log show errors or performance.




  
