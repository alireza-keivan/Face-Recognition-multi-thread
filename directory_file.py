import os
import face_recognition

# Initialize lists for known faces
known_face_encodings = []
known_faces_names = []

# Directory where known face images are stored
KNOWN_FACES_DIR = "known_faces"

# --- Load known faces and their encodings ---
print(f"--- Loading Known Faces from: {KNOWN_FACES_DIR} ---")
def images_folder(directory):
    for filename in os.listdir(KNOWN_FACES_DIR):
        # Process only common image file types
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            print(f"Attempting to load image: {path}")
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    # Name is derived from the filename (e.g., 'john_doe' from 'john_doe.jpg')
                    name = os.path.splitext(filename)[0]
                    known_faces_names.append(name)
                    print(f"Successfully loaded encoding for: {name}")
                else:
                    print(f"Warning: No face found in image '{filename}'. Please ensure the image clearly shows a single face.")
            except Exception as img_e:
                print(f"Error loading or processing image {filename}: {img_e}")
        else:
            print(f"Skipping non-image file in '{KNOWN_FACES_DIR}': {filename}")
    return known_faces_names, known_face_encodings, image, name