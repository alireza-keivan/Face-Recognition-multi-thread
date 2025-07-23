# dir.py
import os
import face_recognition
import numpy as np
import pickle

# Initialize lists for known faces (these will be populated by images_folder)
# These remain global to be accessible after the function returns them.
known_face_encodings = []
known_faces_names = []

# KNOWN_FACES_DIR will now be passed from FaceRecognitionSystem based on config.
# ENCODINGS_FILE will also be passed as an argument to images_folder.

def images_folder(directory, encodings_file_path):
    global known_face_encodings, known_faces_names # Ensure we modify the global lists

    print(f"--- Loading Known Faces ---")
    print(f"Checking for existing encodings file: {encodings_file_path}...")

    # Attempt to load pre-encoded faces
    loaded_successfully = False # Flag to track if loading was successful
    if os.path.exists(encodings_file_path):
        try:
            with open(encodings_file_path, 'rb') as f:
                loaded_data = pickle.load(f)
                known_face_encodings = loaded_data.get('encodings', [])
                known_faces_names = loaded_data.get('names', [])
            print(f"Successfully loaded {len(known_face_encodings)} pre-encoded faces from {encodings_file_path}.")
            loaded_successfully = True # Set flag to True on success
        except Exception as e:
            print(f"Error loading encodings from {encodings_file_path}: {e}")
            print("Proceeding to re-encode faces from directory.")
            # Fall through to re-encoding logic
    else:
        print(f"Encodings file '{encodings_file_path}' not found. Will encode faces from directory.")
        # This branch also falls through to re-encoding

    # Only re-encode if loading failed or file didn't exist, OR if no encodings were loaded
    # This ensures that even if a valid but empty PKL file exists, it will re-encode.
    if not loaded_successfully or not known_face_encodings:
        temp_encodings = []
        temp_names = []

        print(f"Encoding faces from directory: {directory}...")
        # Ensure the directory exists before listing
        if not os.path.exists(directory):
            print(f"Error: Known faces directory '{directory}' not found. No faces will be encoded.")
            return [], [] # Return empty if directory doesn't exist

        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        temp_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        temp_names.append(name)
                        print(f"Successfully encoded face from: {name}. Total encoded so far: {len(temp_encodings)}")
                    else:
                        print(f"Warning: No face found in image '{filename}' in '{directory}'. Please ensure the image clearly shows a single face.")
                except Exception as img_e:
                    print(f"Error loading or processing image {filename} from '{directory}': {img_e}")
            else:
                print(f"Skipping non-image file in '{directory}': {filename}")


        # Update global lists after processing all images
        known_face_encodings = temp_encodings
        known_faces_names = temp_names

        print(f"Finished encoding process. Total faces encoded for saving: {len(known_face_encodings)}")

        # Save the newly generated encodings
        if known_face_encodings: # Only save if some faces were actually encoded
            try:
                # Ensure the directory for the encodings file exists
                os.makedirs(os.path.dirname(encodings_file_path), exist_ok=True)
                with open(encodings_file_path, 'wb') as f:
                    pickle.dump({'encodings': known_face_encodings, 'names': known_faces_names}, f)
                print(f"Successfully saved {len(known_face_encodings)} encodings to {encodings_file_path}.")
            except Exception as e:
                print(f"Error saving encodings to {encodings_file_path}: {e}")
        else:
            print("No faces were encoded, so no data was saved.")

    return known_faces_names, known_face_encodings