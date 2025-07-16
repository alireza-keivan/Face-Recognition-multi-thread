# directory_file.py
import os
import face_recognition
import numpy as np # Required for face_recognition encodings to be properly handled
import pickle      # Required for saving/loading data

# Initialize lists for known faces (these will be populated by images_folder)
known_face_encodings = []
known_faces_names = []

# Directory where known face images are stored
KNOWN_FACES_DIR = "known_faces"

# File to save/load pre-calculated encodings
ENCODINGS_FILE = "known_faces_data.pkl"

def images_folder(directory):
    global known_face_encodings, known_faces_names # Ensure we modify the global lists

    print(f"--- Loading Known Faces ---")
    print(f"Checking for existing encodings file: {ENCODINGS_FILE}...")

    # Attempt to load pre-encoded faces
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                loaded_data = pickle.load(f)
                known_face_encodings = loaded_data.get('encodings', [])
                known_faces_names = loaded_data.get('names', [])
            print(f"Successfully loaded {len(known_face_encodings)} pre-encoded faces from {ENCODINGS_FILE}.")
            return known_faces_names, known_face_encodings
        except Exception as e:
            print(f"Error loading encodings from {ENCODINGS_FILE}: {e}")
            print("Proceeding to re-encode faces from directory.")
            # Fall through to re-encoding if loading fails or file is corrupt

    # If file doesn't exist or loading failed, proceed with encoding from images
    print(f"No pre-encoded data found or loading failed. Encoding faces from directory: {KNOWN_FACES_DIR}...")
    temp_encodings = [] # Use temporary lists to collect new data
    temp_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory '{KNOWN_FACES_DIR}' does not exist. Please create it and add face images.")
        return [], []
    if not os.listdir(KNOWN_FACES_DIR):
        print(f"Warning: Directory '{KNOWN_FACES_DIR}' is empty. No images to encode.")
        return [], []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            print(f"Processing image: {path}")
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    temp_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    temp_names.append(name)
                    print(f"Successfully encoded face from: {name}. Total encoded so far: {len(temp_encodings)}")
                else:
                    print(f"Warning: No face found in image '{filename}'. Please ensure the image clearly shows a single face.")
            except Exception as img_e:
                print(f"Error loading or processing image {filename}: {img_e}")
        else:
            print(f"Skipping non-image file in '{KNOWN_FACES_DIR}': {filename}")

    # Update global lists after processing all images
    known_face_encodings = temp_encodings
    known_faces_names = temp_names

    print(f"Finished encoding process. Total faces encoded for saving: {len(known_face_encodings)}")

    # Save the newly generated encodings
    if known_face_encodings: # Only save if some faces were actually encoded
        try:
            with open(ENCODINGS_FILE, 'wb') as f:
                pickle.dump({'encodings': known_face_encodings, 'names': known_faces_names}, f)
            print(f"Successfully saved {len(known_face_encodings)} encodings to {ENCODINGS_FILE}.")
        except Exception as e:
            print(f"Error saving encodings to {ENCODINGS_FILE}: {e}")
    else:
        print("No faces were encoded, so no data was saved.")

    return known_faces_names, known_face_encodings