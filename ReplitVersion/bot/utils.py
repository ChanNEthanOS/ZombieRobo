"""
Utility functions for the COD WaW Zombies Bot.
This module provides helper functions used across the bot.
"""

import time
import logging
import numpy as np
import cv2
import os
import random
import string

logger = logging.getLogger("Utils")

def timeit(func):
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

def create_directories(path):
    """
    Create directories if they don't exist
    
    Args:
        path (str): Directory path
        
    Returns:
        bool: True if created or already exists
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def save_frame(frame, directory="captured_frames", prefix="frame"):
    """
    Save a frame to disk with timestamp
    
    Args:
        frame (numpy.ndarray): Image to save
        directory (str): Directory to save to
        prefix (str): Filename prefix
        
    Returns:
        str: Path to saved file or None if failed
    """
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        filename = f"{prefix}_{timestamp}_{random_suffix}.jpg"
        filepath = os.path.join(directory, filename)
        
        # Save the image
        cv2.imwrite(filepath, frame)
        logger.debug(f"Saved frame to {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Failed to save frame: {e}")
        return None

def calculate_center(bbox):
    """
    Calculate center coordinates of a bounding box
    
    Args:
        bbox (dict/tuple): Bounding box with x, y, width, height
        
    Returns:
        tuple: (center_x, center_y)
    """
    if isinstance(bbox, dict):
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    else:
        x, y, w, h = bbox
    
    return (x + w // 2, y + h // 2)

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1 (tuple): First point (x, y)
        point2 (tuple): Second point (x, y)
        
    Returns:
        float: Distance between points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def crop_image(image, region):
    """
    Crop an image based on region
    
    Args:
        image (numpy.ndarray): Image to crop
        region (dict): Region with top, left, width, height
        
    Returns:
        numpy.ndarray: Cropped image
    """
    h, w = image.shape[:2]
    
    # Handle relative coordinates (0-1 range)
    if isinstance(region['top'], float) and 0 <= region['top'] <= 1:
        top = int(region['top'] * h)
        left = int(region['left'] * w)
        width = int(region['width'] * w)
        height = int(region['height'] * h)
    else:
        top = region['top']
        left = region['left']
        width = region['width']
        height = region['height']
    
    # Ensure within bounds
    top = max(0, min(h - 1, top))
    left = max(0, min(w - 1, left))
    width = max(1, min(w - left, width))
    height = max(1, min(h - top, height))
    
    return image[top:top+height, left:left+width]

def enhance_image(image, brightness=None, contrast=None, sharpen=False):
    """
    Enhance image for better detection
    
    Args:
        image (numpy.ndarray): Image to enhance
        brightness (float, optional): Brightness factor (0.5-1.5)
        contrast (float, optional): Contrast factor (0.5-2.0)
        sharpen (bool): Whether to apply sharpening
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    enhanced = image.copy()
    
    # Apply brightness adjustment
    if brightness is not None:
        brightness = max(0.5, min(1.5, brightness))
        enhanced = cv2.convertScaleAbs(enhanced, alpha=brightness, beta=0)
    
    # Apply contrast adjustment
    if contrast is not None:
        contrast = max(0.5, min(2.0, contrast))
        mean = np.mean(enhanced)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=(1.0 - contrast) * mean)
    
    # Apply sharpening
    if sharpen:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def get_hsv_range(image, rect=None):
    """
    Get HSV color range from a region of an image
    
    Args:
        image (numpy.ndarray): Image to analyze
        rect (tuple, optional): Rectangle (x, y, w, h) to analyze
        
    Returns:
        tuple: (lower_hsv, upper_hsv) arrays
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract region if specified
    if rect:
        x, y, w, h = rect
        hsv_roi = hsv[y:y+h, x:x+w]
    else:
        hsv_roi = hsv
    
    # Flatten the array to get all pixels
    pixels = hsv_roi.reshape((-1, 3))
    
    # Calculate min/max values with some margin
    h_min, s_min, v_min = np.percentile(pixels, 5, axis=0)
    h_max, s_max, v_max = np.percentile(pixels, 95, axis=0)
    
    # Add some margin
    h_margin = 10
    s_margin = 30
    v_margin = 30
    
    lower_hsv = np.array([max(0, h_min - h_margin), 
                          max(0, s_min - s_margin), 
                          max(0, v_min - v_margin)])
    
    upper_hsv = np.array([min(179, h_max + h_margin), 
                          min(255, s_max + s_margin), 
                          min(255, v_max + v_margin)])
    
    return (lower_hsv, upper_hsv)
