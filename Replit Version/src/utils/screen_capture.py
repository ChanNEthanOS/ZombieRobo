import cv2
import numpy as np
import mss
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

    def capture_frame(self):
        try:
            frame = np.array(self.sct.grab(self.monitor))
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None

    def display_capture(self):
        try:
            while True:
                frame = self.capture_frame()
                if frame is not None:
                    cv2.imshow("Screen Capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error in display loop: {e}")
        finally:
            self.sct.close()
