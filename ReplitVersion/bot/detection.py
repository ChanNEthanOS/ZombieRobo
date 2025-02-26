"""
Detection module for the COD WaW Zombies Bot.
This module handles detection of zombies, health, ammo and other game elements.
"""

import cv2
import numpy as np
import logging
import pytesseract
import os
import time
from models.yolo_model import ZombieYOLOModel

logger = logging.getLogger("Detection")

class ZombieDetector:
    """Class for detecting zombies in the game frames"""
    
    def __init__(self, config):
        """
        Initialize the zombie detector
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Initialize YOLO model for zombie detection
        model_path = config.get('yolo_model_path', 'models/yolo_weights.pt')
        
        # Confidence threshold for detection
        self.confidence_threshold = config.get('detection_confidence', 0.5)
        
        # Initialize yolo model
        try:
            self.model = ZombieYOLOModel(model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.warning("Falling back to color-based detection")
            self.model = None
        
        # Fallback detection params
        self.lower_zombie = np.array(config.get('lower_zombie_hsv', [0, 120, 70]))
        self.upper_zombie = np.array(config.get('upper_zombie_hsv', [10, 255, 255]))
        
        # Track detection time for performance monitoring
        self.detection_time = 0
        
    def detect(self, frame):
        """
        Detect zombies in the current frame
        
        Args:
            frame (numpy.ndarray): BGR image of the game screen
            
        Returns:
            list: List of dictionaries containing zombie info (x, y, width, height, confidence)
        """
        start_time = time.time()
        
        if self.model is not None:
            try:
                # Use YOLO model for detection
                zombies = self.model.detect(frame, self.confidence_threshold)
                
                if zombies:
                    logger.debug(f"Detected {len(zombies)} zombies using YOLO")
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
                zombies = self._fallback_detect(frame)
        else:
            # Use fallback color-based detection
            zombies = self._fallback_detect(frame)
        
        self.detection_time = time.time() - start_time
        return zombies
    
    def _fallback_detect(self, frame):
        """
        Fallback method using color-based detection
        
        Args:
            frame (numpy.ndarray): BGR image
            
        Returns:
            list: List of dictionaries with zombie info
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for zombie colors
        mask = cv2.inRange(hsv, self.lower_zombie, self.upper_zombie)
        
        # Add some morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zombies = []
        for cnt in contours:
            # Filter small contours
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                zombies.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': 0.7,  # Estimated confidence
                    'center_x': x + w // 2,
                    'center_y': y + h // 2
                })
        
        logger.debug(f"Detected {len(zombies)} zombies using color-based detection")
        return zombies
    
    def get_detection_time(self):
        """Get the time taken for the last detection"""
        return self.detection_time


class HUDDetector:
    """Class for detecting HUD elements like health, ammo, etc."""
    
    def __init__(self, config):
        """
        Initialize the HUD detector
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Initialize OCR if available
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            logger.info("Tesseract OCR available")
        except:
            self.ocr_available = False
            logger.warning("Tesseract OCR not available, using fallback methods")
        
        # HUD regions (to be adjusted based on game resolution and UI)
        self.health_region = config.get('health_region', {'top': 0.9, 'left': 0.1, 'width': 0.2, 'height': 0.05})
        self.ammo_region = config.get('ammo_region', {'top': 0.9, 'left': 0.8, 'width': 0.15, 'height': 0.05})
        self.weapon_region = config.get('weapon_region', {'top': 0.85, 'left': 0.7, 'width': 0.25, 'height': 0.1})
        self.round_region = config.get('round_region', {'top': 0.05, 'left': 0.45, 'width': 0.1, 'height': 0.1})
        
        # Color thresholds for health detection
        self.health_low_threshold = np.array(config.get('health_low_hsv', [0, 100, 100]))
        self.health_high_threshold = np.array(config.get('health_high_hsv', [10, 255, 255]))
        
    def detect_stats(self, frame):
        """
        Detect game stats from the HUD
        
        Args:
            frame (numpy.ndarray): BGR image of the game screen
            
        Returns:
            tuple: (health_percentage, ammo_count, current_weapon, current_round)
        """
        height, width = frame.shape[:2]
        
        # Extract regions of interest
        health_roi = self._extract_roi(frame, self.health_region)
        ammo_roi = self._extract_roi(frame, self.ammo_region)
        weapon_roi = self._extract_roi(frame, self.weapon_region)
        round_roi = self._extract_roi(frame, self.round_region)
        
        # Detect health
        health = self._detect_health(health_roi)
        
        # Detect ammo
        ammo = self._detect_ammo(ammo_roi)
        
        # Detect current weapon
        weapon = self._detect_weapon(weapon_roi)
        
        # Detect current round
        current_round = self._detect_round(round_roi)
        
        logger.debug(f"Detected stats - Health: {health}%, Ammo: {ammo}, Weapon: {weapon}, Round: {current_round}")
        return health, ammo, weapon, current_round
    
    def _extract_roi(self, frame, region_config):
        """Extract region of interest based on relative coordinates"""
        height, width = frame.shape[:2]
        
        top = int(region_config['top'] * height)
        left = int(region_config['left'] * width)
        roi_width = int(region_config['width'] * width)
        roi_height = int(region_config['height'] * height)
        
        return frame[top:top+roi_height, left:left+roi_width]
    
    def _detect_health(self, health_roi):
        """
        Detect player health
        
        Args:
            health_roi (numpy.ndarray): Region of the frame containing health indicator
            
        Returns:
            int: Health percentage (0-100)
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(health_roi, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the health bar (usually red/green)
        mask = cv2.inRange(hsv, self.health_low_threshold, self.health_high_threshold)
        
        # Calculate the percentage of health
        health_pixels = cv2.countNonZero(mask)
        total_pixels = health_roi.shape[0] * health_roi.shape[1]
        
        if total_pixels > 0:
            health_percentage = min(100, int((health_pixels / total_pixels) * 200))
        else:
            health_percentage = 100  # Default if detection fails
        
        return health_percentage
    
    def _detect_ammo(self, ammo_roi):
        """
        Detect ammo count
        
        Args:
            ammo_roi (numpy.ndarray): Region of the frame containing ammo indicator
            
        Returns:
            int: Ammo count
        """
        if self.ocr_available:
            try:
                # Preprocess the image for better OCR
                gray = cv2.cvtColor(ammo_roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                
                # Use OCR to extract text
                text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789/')
                
                # Parse the text to get ammo count
                if '/' in text:
                    current_ammo = text.split('/')[0].strip()
                    try:
                        return int(current_ammo)
                    except ValueError:
                        pass
            except Exception as e:
                logger.debug(f"OCR failed for ammo detection: {e}")
        
        # Fallback: estimate based on white pixels (ammo numbers are usually white)
        gray = cv2.cvtColor(ammo_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        
        # Map pixel count to approximate ammo value
        if white_pixels > 500:
            return 30  # Full ammo
        elif white_pixels > 300:
            return 15  # Half ammo
        else:
            return 5   # Low ammo
    
    def _detect_weapon(self, weapon_roi):
        """
        Detect current weapon
        
        Args:
            weapon_roi (numpy.ndarray): Region of the frame containing weapon indicator
            
        Returns:
            str: Weapon name or 'unknown'
        """
        # This is a placeholder - actual implementation would require training data
        # and possibly a classifier for weapon recognition
        
        # For now, just detect if a weapon is present based on non-black pixels
        gray = cv2.cvtColor(weapon_roi, cv2.COLOR_BGR2GRAY)
        non_black = cv2.countNonZero(gray)
        
        if non_black > 100:
            return "primary"  # Placeholder
        else:
            return "unknown"
    
    def _detect_round(self, round_roi):
        """
        Detect current round number
        
        Args:
            round_roi (numpy.ndarray): Region of the frame containing round indicator
            
        Returns:
            int: Current round number
        """
        if self.ocr_available:
            try:
                # Preprocess the image for better OCR
                gray = cv2.cvtColor(round_roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                
                # Use OCR to extract text
                text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')
                
                # Parse the text to get round number
                try:
                    return int(text.strip())
                except ValueError:
                    pass
            except Exception as e:
                logger.debug(f"OCR failed for round detection: {e}")
        
        # Fallback: return estimatd round based on game time
        # This is very approximate and should be improved
        return 1  # Default to round 1
