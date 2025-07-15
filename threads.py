from threading import Thread
import time
import cv2


# defining a helper class for implementing multi-threaded processing 
class WebcamStream :
    def __init__(self, stream_cap):
        self.cap = cv2.VideoCapture("rtsp://admin:ms123456@192.168.48.67:554/main")
        # "rtsp://admin:ms123456@192.168.48.67:554/main"
        self.stream_id = stream_cap   # default is 0 for primary camera 
        
        # opening video capture stream 
        
        if self.cap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.cap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from cap stream for initializing 
        self.grabbed , self.frame = self.cap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.cap stream 
        self.stopped = True 

  

        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
    # method for starting the thread for grabbing next available frame in input stream 
    def start(self):
        self.stopped = False
        self.t.start() 

    # method for reading next frame 
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.cap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.cap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True 