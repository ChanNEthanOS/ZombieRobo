"""
bot/screen_capture.py
Handles capturing the game screen.
"""

import numpy as np
import mss
import cv2
import time
import logging

logger = logging.getLogger("ScreenCapture")

class ScreenCapture:
    def __init__(self, game_region):
        self.sct = mss.mss()
        self.game_region = game_region
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0

        logger.info(f"Screen capture initialized with region: {self.game_region}")

    def capture(self):
        img = np.array(self.sct.grab(self.game_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.prev_time)
            self.frame_count = 0
            self.prev_time = current_time
            logger.debug(f"Screen capture FPS: {self.fps:.1f}")

        return img

    def get_fps(self):
        return self.fps
