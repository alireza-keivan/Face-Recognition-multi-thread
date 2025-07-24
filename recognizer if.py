import json
import sys
import os
from thread import FaceRecognitionSystem


def main():
    config_file_path = "config.json"
    F = {} 

    try:
        with open(config_file_path, "r") as file:
            F = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        sys.exit(1) 
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{config_file_path}'. Check file format. Error: {e}")
        sys.exit(1) 


    print("Starting Face Recognition System...")
    face_system = FaceRecognitionSystem(config=F)
    face_system.run() 

if __name__ == "__main__":
    main()