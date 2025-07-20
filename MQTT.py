import paho.mqtt.client as mqtt 
import time
import json

with open ("config.json", "r") as file:
    J = json.load(file)
    mqtt_settings = J.get("mqtt_settings", {})

MQTT_BROKER_ADDRESS = mqtt_settings.get("MQTT_BROKER_ADDRESS")
MQTT_BROKER_PORT = mqtt_settings.get("MQTT_BROKER_PORT")
MQTT_CLIENT_ID = mqtt_settings.get("MQTT_CLIENT_ID")
MQTT_USERNAME = mqtt_settings.get("MQTT_USERNAME")
MQTT_PASSWORD = mqtt_settings.get("MQTT_PASSWORD")
MQTT_DOOR_OPEN_TOPIC = mqtt_settings.get("MQTT_DOOR_OPEN_TOPIC")

# --- MQTT Configuration ---

# --- MQTT Callbacks (for VERSION2 API) ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"MQTT: Connected successfully to broker {mqtt_settings.get}")
    else:
        print(f"MQTT: Failed to connect, return code {rc}\n")

def on_disconnect(client, userdata, rc, *args, **kwargs):
    print(f"MQTT: Disconnected with result code {rc}\n")
    print(f"MQTT: Extra args received: {args}")
# --- MQTT Client Setup ---

mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
if MQTT_USERNAME and MQTT_PASSWORD:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

try:
    mqtt_client.connect(MQTT_BROKER_ADDRESS, MQTT_BROKER_PORT)
    mqtt_client.loop_start() # Start MQTT background thread for handling messages
except Exception as e:
    print(f"MQTT: Could not connect to broker: {e}")
    # The program will continue running, but MQTT commands won't be sent.

last_open_time = 0 # To implement cooldown for sending "open" commands
COOLDOWN_PERIOD_SECONDS = 1 # Don't send "open" more often than this

def send_door_open_command(mqtt_client_obj, known_name):
    global last_open_time
    current_time = time.time()
    if (current_time - last_open_time) > COOLDOWN_PERIOD_SECONDS:
        payload = '{"IDENTITY_VERIFIED": "TRUE"}' # Example JSON payload
        
        try:
            # Check if client is actually connected before publishing
            if mqtt_client_obj.is_connected():
                result = mqtt_client_obj.publish(MQTT_DOOR_OPEN_TOPIC, payload)
                if result[0] == 0:
                    print(f"MQTT: Sent '{payload}' to topic '{MQTT_DOOR_OPEN_TOPIC}'")
                    last_open_time = current_time
                    #print()
                else:
                    print(f"MQTT: Failed to send message to topic {MQTT_DOOR_OPEN_TOPIC}. Result code: {result[0]}")
            else:
                print("MQTT: Client not connected, cannot publish message.")
        except Exception as e:
            print(f"MQTT: Error publishing message: {e}")
    else:
        print(f"MQTT: Cooldown active. Not sending 'open' command. ({COOLDOWN_PERIOD_SECONDS - (current_time - last_open_time):.1f}s remaining)")
