# recognizer.py (Main entry point)
import json
import sys
import os
from thread2 import FaceRecognitionSystem


def main():
    config_file_path = "config.json"
    F = {} # Initialize F to an empty dict

    try:
        with open(config_file_path, "r") as file:
            F = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        sys.exit(1) # Exit if config file is not found
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{config_file_path}'. Check file format. Error: {e}")
        sys.exit(1) # Exit if JSON is malformed

    # Initialize and run the FaceRecognitionSystem
    print("Starting Face Recognition System...")
    face_system = FaceRecognitionSystem(config=F)
    face_system.run() # This will start the main loop and handle cleanup

if __name__ == "__main__":
    main()