"""
YOLO model implementation for zombie detection.
This module provides the YOLO-based object detection for the bot.
"""

import logging
import numpy as np
import cv2
import os
import time
import torch

logger = logging.getLogger("YOLO")

class ZombieYOLOModel:
    """YOLO model for zombie detection"""
    
    def __init__(self, model_path):
        """
        Initialize the YOLO model
        
        Args:
            model_path (str): Path to the YOLO model weights
        """
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model if available
        self._load_model()
        
        # Keep track of inference time
        self.inference_time = 0
        self.frame_count = 0
        self.avg_inference_time = 0
        
        # Class names (zombie types in CoD)
        self.class_names = ["zombie", "crawler", "hellhound", "boss"]
        
        logger.info(f"YOLO model initialized on device: {self.device}")
    
    def _load_model(self):
        """Load the YOLO model from weights file"""
        try:
            if os.path.exists(self.model_path):
                # Load using torch hub
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                           path=self.model_path, force_reload=True)
                
                # Move to appropriate device
                self.model.to(self.device)
                
                # Set inference parameters
                self.model.conf = 0.25  # Confidence threshold
                self.model.iou = 0.45   # NMS IoU threshold
                self.model.classes = None  # All classes
                self.model.max_det = 100  # Maximum detections
                
                logger.info(f"Loaded YOLO model from {self.model_path}")
                return True
            else:
                # If model doesn't exist, create a dummy implementation
                logger.warning(f"Model file not found: {self.model_path}")
                logger.warning("Using fallback color-based detection instead")
                return False
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect(self, frame, confidence_threshold=0.5):
        """
        Detect zombies in a frame
        
        Args:
            frame (numpy.ndarray): BGR image
            confidence_threshold (float): Detection confidence threshold
            
        Returns:
            list: List of dictionaries containing zombie information
        """
        start_time = time.time()
        
        # If no model available, use fallback implementation
        if self.model is None:
            self.inference_time = time.time() - start_time
            return self._fallback_detect(frame)
        
        try:
            # Run inference
            results = self.model(frame)
            
            # Extract detections
            detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
            
            # Filter by confidence
            detections = detections[detections['confidence'] >= confidence_threshold]
            
            # Convert to list of dicts
            zombies = []
            for _, det in detections.iterrows():
                # Filter for zombie classes only
                class_name = det['name'] if 'name' in detections.columns else f"class_{int(det['class'])}"
                
                # Skip non-zombie classes if defined
                if class_name not in self.class_names and 'class' in det and int(det['class']) >= len(self.class_names):
                    continue
                
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                width = x2 - x1
                height = y2 - y1
                
                zombie = {
                    'x': x1,
                    'y': y1,
                    'width': width,
                    'height': height,
                    'confidence': float(det['confidence']),
                    'class': int(det['class']) if 'class' in det else 0,
                    'class_name': class_name,
                    'center_x': x1 + width // 2,
                    'center_y': y1 + height // 2
                }
                
                zombies.append(zombie)
            
            # Update inference time stats
            self.inference_time = time.time() - start_time
            self.frame_count += 1
            self.avg_inference_time = ((self.frame_count - 1) * self.avg_inference_time + self.inference_time) / self.frame_count
            
            if self.frame_count % 100 == 0:
                logger.debug(f"Average inference time: {self.avg_inference_time:.4f}s ({1/self.avg_inference_time:.1f} FPS)")
            
            return zombies
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            self.inference_time = time.time() - start_time
            return self._fallback_detect(frame)
    
    def _fallback_detect(self, frame):
        """
        Fallback color-based detection when YOLO model is unavailable
        
        Args:
            frame (numpy.ndarray): BGR image
            
        Returns:
            list: List of dictionaries containing zombie information
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for zombie-like colors
        # This is a very rough estimation and will need game-specific tuning
        lower_zombie = np.array([0, 50, 50])
        upper_zombie = np.array([30, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_zombie, upper_zombie)
        
        # Add another color range (greenish for some zombies)
        lower_zombie2 = np.array([40, 50, 50])
        upper_zombie2 = np.array([80, 255, 255])
        
        mask2 = cv2.inRange(hsv, lower_zombie2, upper_zombie2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zombies = []
        for cnt in contours:
            # Filter small contours
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter out unreasonably sized boxes
                if w > 20 and h > 30 and w < frame.shape[1]//2 and h < frame.shape[0]//2:
                    confidence = min(1.0, cv2.contourArea(cnt) / 10000)
                    
                    zombie = {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': confidence,
                        'class': 0,
                        'class_name': 'zombie',
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    }
                    
                    zombies.append(zombie)
        
        logger.debug(f"Fallback detection found {len(zombies)} potential zombies")
        return zombies
    
    def get_inference_stats(self):
        """
        Get model inference statistics
        
        Returns:
            dict: Inference statistics
        """
        return {
            'device': self.device,
            'model_path': self.model_path,
            'model_loaded': self.model is not None,
            'inference_time': f"{self.inference_time:.4f}s",
            'fps': f"{1/self.inference_time:.1f}" if self.inference_time > 0 else "N/A",
            'avg_inference_time': f"{self.avg_inference_time:.4f}s",
            'avg_fps': f"{1/self.avg_inference_time:.1f}" if self.avg_inference_time > 0 else "N/A",
            'frames_processed': self.frame_count
        }
