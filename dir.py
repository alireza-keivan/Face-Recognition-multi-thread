# dir.py
import os
import numpy as np
import pickle
import cv2 # Required for cv2.imread and cv2.cvtColor

# No longer need face_recognition here

# Initialize lists for known faces (these will be populated by images_folder)
known_face_encodings = []
known_faces_names = []

# Modify the function signature to accept the InsightFace app
def images_folder(directory, encodings_file_path, insightface_app):
    global known_face_encodings, known_faces_names

    print(f"--- Loading Known Faces ---")
    print(f"Checking for existing encodings file: {encodings_file_path}...")

    loaded_successfully = False
    if os.path.exists(encodings_file_path):
        try:
            with open(encodings_file_path, 'rb') as f:
                loaded_data = pickle.load(f)
                known_face_encodings = loaded_data.get('encodings', [])
                known_faces_names = loaded_data.get('names', [])
            print(f"Successfully loaded {len(known_face_encodings)} pre-encoded faces from {encodings_file_path}.")
            loaded_successfully = True
        except Exception as e:
            print(f"Error loading encodings from {encodings_file_path}: {e}")
            print("Proceeding to re-encode faces from directory.")

    # Only re-encode if loading failed or file didn't exist, OR if no encodings were loaded
    if not loaded_successfully or not known_face_encodings:
        temp_encodings = []
        temp_names = []

        print(f"Encoding faces from directory: {directory}...")
        if not os.path.exists(directory):
            print(f"Error: Known faces directory '{directory}' not found. No faces will be encoded.")
            return [], []

        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory, filename)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not load image {filename}. Skipping.")
                        continue

                    # Convert to RGB (InsightFace expects RGB)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # --- InsightFace: Get face embeddings ---
                    # app.get() returns a list of Face objects.
                    # We expect one face per known image for encoding.
                    faces = insightface_app.get(image_rgb)

                    if len(faces) == 1:
                        face_embedding = faces[0].embedding # Get the ArcFace embedding
                        temp_encodings.append(face_embedding)
                        name = os.path.splitext(filename)[0]
                        temp_names.append(name)
                        print(f"Successfully encoded face from: {name}. Total encoded so far: {len(temp_encodings)}")
                    elif len(faces) > 1:
                        print(f"Warning: Multiple faces found in image '{filename}'. Encoding only the first one.")
                        face_embedding = faces[0].embedding
                        temp_encodings.append(face_embedding)
                        name = os.path.splitext(filename)[0]
                        temp_names.append(name)
                    else:
                        print(f"Warning: No face found in image '{filename}' in '{directory}'. Please ensure the image clearly shows a single face.")
                except Exception as img_e:
                    print(f"Error loading or processing image {filename} from '{directory}': {img_e}")
            else:
                print(f"Skipping non-image file in '{directory}': {filename}")

        known_face_encodings = temp_encodings
        known_faces_names = temp_names

        print(f"Finished encoding process. Total faces encoded for saving: {len(known_face_encodings)}")

        # Save the newly generated encodings
        if known_face_encodings:
            try:
                os.makedirs(os.path.dirname(encodings_file_path), exist_ok=True)
                with open(encodings_file_path, 'wb') as f:
                    pickle.dump({'encodings': known_face_encodings, 'names': known_faces_names}, f)
                print(f"Successfully saved {len(known_face_encodings)} encodings to {encodings_file_path}.")
            except Exception as e:
                print(f"Error saving encodings to {encodings_file_path}: {e}")
        else:
            print("No faces were encoded, so no data was saved.")

    return known_faces_names, known_face_encodings
