"""
Screen capture module for the COD WaW Zombies Bot.
This module handles capturing the game screen for processing.
"""

import numpy as np
import mss
import cv2
import time
import logging

logger = logging.getLogger("ScreenCapture")

class ScreenCapture:
    """Screen capture class for grabbing game frames"""
    
    def __init__(self, game_region=None):
        """
        Initialize the screen capture with specified game region
        
        Args:
            game_region (dict): Dictionary with top, left, width, height keys
                                for the game window region
        """
        self.sct = mss.mss()
        
        # Default game region (full screen)
        self.game_region = game_region or {
            'top': 0,
            'left': 0,
            'width': 1920,
            'height': 1080
        }
        
        # For FPS calculation
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        logger.info(f"Screen capture initialized with region: {self.game_region}")
    
    def capture(self):
        """
        Capture the current game screen
        
        Returns:
            numpy.ndarray: The captured screen as BGR image
        """
        # Capture the screen
        img = np.array(self.sct.grab(self.game_region))
        
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.prev_time
        
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.prev_time = current_time
            logger.debug(f"Screen capture FPS: {self.fps:.1f}")
        
        return img
    
    def get_region_center(self):
        """
        Get the center coordinates of the game region
        
        Returns:
            tuple: (x, y) coordinates of the center
        """
        center_x = self.game_region['left'] + self.game_region['width'] // 2
        center_y = self.game_region['top'] + self.game_region['height'] // 2
        return (center_x, center_y)
    
    def adjust_region(self, game_region):
        """
        Adjust the capture region
        
        Args:
            game_region (dict): Dictionary with top, left, width, height keys
        """
        self.game_region = game_region
        logger.info(f"Screen capture region adjusted to: {self.game_region}")
    
    def get_fps(self):
        """Get the current capture FPS"""
        return self.fps
