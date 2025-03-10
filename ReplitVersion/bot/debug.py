"""
bot/debug.py
Displays debug overlays and logs info for debugging.
"""

import cv2
import numpy as np
import time
import logging
import sys

logger = logging.getLogger("Debug")

class DebugInterface:
    def __init__(self):
        self.show_visualization = True
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.current_fps = 0
        if self.show_visualization:
            cv2.namedWindow("Game View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Game View", 960, 540)

    def display(self, frame, zombies=None, health=None, ammo=None, game_state=None, action=None):
        self.frame_count += 1
        now = time.time()
        if now - self.last_frame_time >= 1.0:
            self.current_fps = self.frame_count / (now - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = now

        overlay = frame.copy()
        if zombies:
            for z in zombies:
                x,y,w,h = z['x'], z['y'], z['width'], z['height']
                cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,0,255), 2)

        # Display HUD info
        if health is not None:
            cv2.putText(overlay, f"Health: {health}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        if ammo is not None:
            cv2.putText(overlay, f"Ammo: {ammo}", (30,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(overlay, f"FPS: {self.current_fps:.1f}", (30,150), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        if action:
            cv2.putText(overlay, f"Action: {action['type']}", (30,200), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        if self.show_visualization:
            cv2.imshow("Game View", overlay)
            cv2.waitKey(1)

    def close(self):
        if self.show_visualization:
            cv2.destroyAllWindows()
        logger.info("Debug interface closed.")
