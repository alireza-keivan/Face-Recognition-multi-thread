#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="recognizer.py"
LOG_FILE="face_recognition.log"
# The specific error message that triggers a restart
ERROR_STRING="MQTT disconnected with reason code7: The connection was lost."
# The interval in seconds to check the log file
CHECK_INTERVAL=10

# --- Functions ---

# Function to start the python script
start_script() {
    echo "Starting '$PYTHON_SCRIPT' at $(date)..."
    # Clear the log file before starting to avoid old errors
    > "$LOG_FILE"
    # Execute the script in the background and store its Process ID (PID)
    python3 "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
    PYTHON_PID=$!
    echo "'$PYTHON_SCRIPT' started with PID: $PYTHON_PID"
}

# Function to handle exiting the watchdog script (e.g., with Ctrl+C)
cleanup() {
    echo -e "\nSignal received. Shutting down..."
    # Check if the python process is still running
    if ps -p $PYTHON_PID > /dev/null; then
        echo "Stopping Python script (PID: $PYTHON_PID)..."
        kill $PYTHON_PID
    fi
    echo "Exiting monitor script."
    exit 0
}

# Trap signals (like Ctrl+C) to run the cleanup function
trap cleanup SIGINT SIGTERM

# --- Main Logic ---

# Initial start of your program
start_script

# Loop forever to monitor the log
while true; do
    # Check the log file for the specific error string
    if grep -q "$ERROR_STRING" "$LOG_FILE"; then
        echo "---------------------------------"
        echo "RESTARTING: Critical error detected in '$LOG_FILE'."

        echo "Stopping faulty process (PID: $PYTHON_PID)..."
        # Forcefully kill the old, broken process
        kill -9 $PYTHON_PID 2>/dev/null

        # Relaunch the script
        start_script
        echo "---------------------------------"
    fi

    # Wait for the defined interval before checking again
    sleep "$CHECK_INTERVAL"
done
