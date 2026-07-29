import cv2
import threading
import time

class ThreadedWebcam:
    """
    Asynchronous threaded webcam reader that continuously reads frames in a background thread.
    This prevents OpenCV's internal video buffer from accumulating latency and ensures lag-free live streaming.
    """
    def __init__(self, src=0, name="ThreadedWebcam"):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            self.status = False
            self.frame = None
        else:
            self.status, self.frame = self.cap.read()
        
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, name=name, args=())
        self.thread.daemon = True

    def start(self):
        if self.cap.isOpened() and not self.thread.is_alive():
            self.stopped = False
            self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.005)
                continue
            
            with self.lock:
                self.status = grabbed
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.status, self.frame.copy()

    def is_opened(self):
        return self.cap.isOpened() and not self.stopped

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def release(self):
        self.stop()
