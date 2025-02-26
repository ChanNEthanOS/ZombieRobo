#!/usr/bin/env python3
"""
COD World at War Zombies Bot - Test Module
This is a simplified test version for local testing with an actual game.
"""

import time
import logging
import os
import sys
import cv2
import numpy as np
import json
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TestBot")

# Default game region (can be adjusted via command line)
DEFAULT_REGION = {
    'top': 0,
    'left': 0,
    'width': 1920,
    'height': 1080
}

# RGB ranges for zombie detection (can be adjusted)
ZOMBIE_HSV_LOWER = np.array([0, 120, 70])   # Red/orange-ish color of zombies
ZOMBIE_HSV_UPPER = np.array([10, 255, 255])

# Health and ammo detection regions (relative to screen)
HEALTH_REGION = {'top': 0.9, 'left': 0.1, 'width': 0.2, 'height': 0.05}
AMMO_REGION = {'top': 0.9, 'left': 0.8, 'width': 0.15, 'height': 0.05}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='COD WaW Zombies Bot Test Module')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--region', type=str, help='Game region as "top,left,width,height"')
    parser.add_argument('--test-detection', action='store_true', help='Run detection test only')
    parser.add_argument('--collect-data', action='store_true', help='Collect and save frames for training')
    parser.add_argument('--delay', type=int, default=3, help='Startup delay')
    return parser.parse_args()

def calculate_absolute_region(relative_region, screen_region):
    """Convert a relative region to absolute pixel coordinates"""
    return {
        'top': int(screen_region['top'] + relative_region['top'] * screen_region['height']),
        'left': int(screen_region['left'] + relative_region['left'] * screen_region['width']),
        'width': int(relative_region['width'] * screen_region['width']),
        'height': int(relative_region['height'] * screen_region['height'])
    }

def capture_screen(region):
    """Capture the screen within the specified region"""
    try:
        import mss
        with mss.mss() as sct:
            img = np.array(sct.grab(region))
            # Convert BGRA to BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        # Return a black frame as fallback
        return np.zeros((region['height'], region['width'], 3), dtype=np.uint8)

def detect_zombies(frame):
    """Detect zombies in the frame using color thresholding"""
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for zombie colors
        mask = cv2.inRange(hsv, ZOMBIE_HSV_LOWER, ZOMBIE_HSV_UPPER)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zombies = []
        for contour in contours:
            # Filter out small contours (noise)
            if cv2.contourArea(contour) > 100:  
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center point and confidence (based on size)
                center_x = x + w // 2
                center_y = y + h // 2
                confidence = min(1.0, cv2.contourArea(contour) / 5000)
                
                zombies.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'center': (center_x, center_y),
                    'confidence': confidence,
                    'class_id': 0  # 0 = standard zombie
                })
        
        return zombies
    except Exception as e:
        logger.error(f"Error detecting zombies: {e}")
        return []

def detect_health_and_ammo(frame, game_region):
    """
    Detect health and ammo from the HUD
    Returns: (health_percentage, ammo_count)
    """
    try:
        # Extract health region
        health_region = calculate_absolute_region(HEALTH_REGION, game_region)
        health_roi = frame[
            health_region['top'] - game_region['top']:health_region['top'] - game_region['top'] + health_region['height'],
            health_region['left'] - game_region['left']:health_region['left'] - game_region['left'] + health_region['width']
        ]
        
        # Extract ammo region
        ammo_region = calculate_absolute_region(AMMO_REGION, game_region)
        ammo_roi = frame[
            ammo_region['top'] - game_region['top']:ammo_region['top'] - game_region['top'] + ammo_region['height'],
            ammo_region['left'] - game_region['left']:ammo_region['left'] - game_region['left'] + ammo_region['width']
        ]
        
        # Basic red detection for health (red bar)
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(health_roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for red health bar (typical color)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Calculate health percentage based on red pixels
        health_percentage = int(np.count_nonzero(mask) / (health_roi.shape[0] * health_roi.shape[1]) * 100)
        health_percentage = min(100, max(0, health_percentage))
        
        # For ammo, we would normally use OCR to read the number
        # This is a placeholder - actual implementation would use pytesseract
        ammo_count = 30  # placeholder
        
        return health_percentage, ammo_count
    except Exception as e:
        logger.error(f"Error detecting health/ammo: {e}")
        return 100, 30  # Default values

def draw_detection_overlay(frame, zombies, health, ammo):
    """Draw detection overlays on the frame for visualization"""
    # Copy the frame to avoid modifying the original
    overlay = frame.copy()
    
    # Draw zombie detection boxes
    for zombie in zombies:
        x, y = zombie['x'], zombie['y']
        w, h = zombie['width'], zombie['height']
        confidence = zombie.get('confidence', 0.0)
        
        # Color based on confidence (green to red)
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        
        # Draw rectangle and confidence
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, f"{confidence:.2f}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw health and ammo indicators
    cv2.putText(overlay, f"Health: {health}%", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(overlay, f"Ammo: {ammo}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw zombie count
    cv2.putText(overlay, f"Zombies: {len(zombies)}", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return overlay

def save_frame_data(frame, zombies, health, ammo, save_dir="collected_data"):
    """Save frame and detection data for training purposes"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save the frame
        frame_path = os.path.join(save_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Save the detection data
        data = {
            "timestamp": timestamp,
            "zombies": zombies,
            "health": health,
            "ammo": ammo
        }
        
        json_path = os.path.join(save_dir, f"data_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return frame_path, json_path
    except Exception as e:
        logger.error(f"Error saving frame data: {e}")
        return None, None

def auto_detect_game_window():
    """
    Attempt to automatically detect the game window
    Returns a region dict or None if not found
    """
    try:
        # This would be replaced with a more robust detection
        # based on your specific system
        return DEFAULT_REGION
    except Exception as e:
        logger.error(f"Failed to detect game window: {e}")
        return None

def run_detection_test(args):
    """Run detection test mode"""
    logger.info("Running detection test mode...")
    
    # Parse region from args or use default
    if args.region:
        try:
            top, left, width, height = map(int, args.region.split(','))
            game_region = {'top': top, 'left': left, 'width': width, 'height': height}
        except:
            logger.warning("Invalid region format, using default")
            game_region = DEFAULT_REGION
    else:
        # Try to detect game window or use default
        detected_region = auto_detect_game_window()
        game_region = detected_region if detected_region else DEFAULT_REGION
    
    logger.info(f"Using game region: {game_region}")
    
    # Wait for startup delay
    print(f"Starting detection in {args.delay} seconds...")
    time.sleep(args.delay)
    
    # Create window for display
    cv2.namedWindow('Zombie Detection Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Zombie Detection Test', 800, 600)
    
    try:
        # Main detection loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Capture screen
            frame = capture_screen(game_region)
            
            # Detect zombies
            zombies = detect_zombies(frame)
            
            # Detect health and ammo
            health, ammo = detect_health_and_ammo(frame, game_region)
            
            # Draw detection overlay
            display_frame = draw_detection_overlay(frame, zombies, health, ammo)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                logger.info(f"FPS: {fps:.1f}, Zombies: {len(zombies)}, Health: {health}%, Ammo: {ammo}")
                frame_count = 0
                start_time = time.time()
            
            # Display the frame
            cv2.imshow('Zombie Detection Test', display_frame)
            
            # Save frame data if in collection mode
            if args.collect_data and frame_count % 10 == 0:  # Save every 10th frame
                frame_path, _ = save_frame_data(frame, zombies, health, ammo)
                if frame_path:
                    logger.debug(f"Saved frame to {frame_path}")
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
                
    except KeyboardInterrupt:
        logger.info("Detection test stopped by user")
    except Exception as e:
        logger.exception(f"Error in detection test: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("Detection test completed")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Run detection test mode
    if args.test_detection:
        run_detection_test(args)
    else:
        # Default to running detection test
        run_detection_test(args)

if __name__ == "__main__":
    main()