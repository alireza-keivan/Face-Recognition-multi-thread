import json
import sys
from th import FaceRecognitionSystem
import logging
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
            logging.FileHandler("face_recognition.log"),
            logging.StreamHandler()
    ]
)


def main():
    config_file_path = "config.json"
    F = {} 

    try:
        with open(config_file_path, "r") as file:
            F = json.load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_file_path}' not found.")
        sys.exit(1) 
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{config_file_path}'. Check file format. Error: {e}")
        sys.exit(1)


    logging.info("starting Face Recognition System...")
    face_system = FaceRecognitionSystem(config=F)
    face_system.run()

if __name__ == "__main__":
    main()