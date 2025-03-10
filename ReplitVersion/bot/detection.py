"""
bot/detection.py
Detects zombies using YOLO or fallback color-based detection, plus HUD elements.
"""

import cv2
import numpy as np
import logging
import time

# If you have a YOLO model file, import it:
# from models.yolo_model import ZombieYOLOModel

logger = logging.getLogger("Detection")

class ZombieDetector:
    def __init__(self, config):
        self.config = config
        self.lower_zombie = np.array(config.get('lower_zombie_hsv', [0,120,70]))
        self.upper_zombie = np.array(config.get('upper_zombie_hsv', [10,255,255]))
        self.confidence_threshold = config.get('detection_confidence', 0.5)
        # If you have a YOLO model, load it here. Otherwise fallback.

    def detect(self, frame):
        # Example fallback color detection:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_zombie, self.upper_zombie)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        zombies = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                zombies.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'confidence': 0.7,  # placeholder
                    'center_x': x + w//2, 'center_y': y + h//2
                })
        logger.debug(f"Detected {len(zombies)} zombies (fallback).")
        return zombies

class HUDDetector:
    def __init__(self, config):
        self.config = config

    def detect_stats(self, frame):
        # Example: just return random or default
        # You can add OCR or color-based detection for health/ammo
        health = 100
        ammo = 30
        current_weapon = "rifle"
        current_round = 1
        return health, ammo, current_weapon, current_round
