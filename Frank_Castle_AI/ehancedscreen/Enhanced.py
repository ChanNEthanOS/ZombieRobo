"""
Enhanced Screen Capture Module for CoD WaW Zombies Bot
This module provides multiple fallback methods for screen capture to ensure reliable operation
in different environments and fullscreen exclusive mode.
"""

import numpy as np
import cv2
import time
import logging
import os
import platform
import random
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_capture.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedCapture")

# Default game region (full screen)
DEFAULT_REGION = {
    'top': 0,
    'left': 0,
    'width': 1920,
    'height': 1080
}

# Try to import screen capture libraries
CAPTURE_METHODS = []

# MSS Screen Capture - cross-platform but doesn't always work with fullscreen exclusive
try:
    import mss
    CAPTURE_METHODS.append("mss")
    logger.info("MSS screen capture available")
except ImportError:
    logger.warning("MSS not available for screen capture")

# Pillow/PIL for screenshot
try:
    from PIL import ImageGrab
    CAPTURE_METHODS.append("pil")
    logger.info("PIL/Pillow screen capture available")
except ImportError:
    logger.warning("PIL/Pillow not available for screen capture")

# X11 for Linux systems
if platform.system() == "Linux":
    try:
        from Xlib import display, X
        import Xlib.error
        CAPTURE_METHODS.append("x11")
        logger.info("X11 screen capture available")
    except ImportError:
        logger.warning("X11 libraries not available for screen capture on Linux")

# Windows specific methods
if platform.system() == "Windows":
    try:
        import win32gui
        import win32ui
        import win32con
        CAPTURE_METHODS.append("win32")
        logger.info("Win32 screen capture available")
    except ImportError:
        logger.warning("Win32 libraries not available for screen capture on Windows")
    
    # Try D3D capture for DirectX games on Windows
    try:
        import d3dshot
        CAPTURE_METHODS.append("d3d")
        logger.info("D3D screen capture available")
    except ImportError:
        logger.warning("D3D libraries not available for screen capture")

# Display available capture methods
if CAPTURE_METHODS:
    logger.info(f"Available screen capture methods: {', '.join(CAPTURE_METHODS)}")
else:
    logger.warning("No screen capture methods available! Will use simulated frames.")

class EnhancedScreenCapture:
    """Enhanced screen capture with multiple fallback methods"""
    
    def __init__(self, game_region=None, preferred_method=None):
        """
        Initialize the screen capture with specified game region and method
        
        Args:
            game_region (dict): Dictionary with top, left, width, height keys
                               for the game window region
            preferred_method (str): Preferred capture method (mss, pil, x11, win32, d3d)
        """
        self.game_region = game_region or DEFAULT_REGION
        self.preferred_method = preferred_method
        
        # For thread safety during capture
        self.capture_lock = Lock()
        
        # Capture utilities
        self.mss_sct = None
        self.d3d_capture = None
        self.display_obj = None
        self.root = None
        
        # Try to initialize preferred method first
        if preferred_method:
            self._init_method(preferred_method)
        
        # If no preferred method or it failed, try all available methods
        if not self.preferred_method:
            for method in CAPTURE_METHODS:
                success = self._init_method(method)
                if success:
                    self.preferred_method = method
                    break
        
        # For FPS calculation
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Failed capture counter
        self.consecutive_failures = 0
        self.last_successful_method = None
        
        # Record when methods fail
        self.failed_methods = set()
        
        logger.info(f"Screen capture initialized with region: {self.game_region}")
        logger.info(f"Using preferred capture method: {self.preferred_method}")
    
    def _init_method(self, method):
        """Initialize a specific capture method"""
        try:
            if method == "mss" and "mss" in CAPTURE_METHODS:
                self.mss_sct = mss.mss()
                return True
                
            elif method == "d3d" and "d3d" in CAPTURE_METHODS:
                self.d3d_capture = d3dshot.create(capture_output="numpy")
                return True
                
            elif method == "x11" and "x11" in CAPTURE_METHODS:
                self.display_obj = display.Display()
                self.root = self.display_obj.screen().root
                return True
                
            return False
        except Exception as e:
            logger.error(f"Failed to initialize capture method {method}: {e}")
            return False
    
    def capture(self):
        """
        Capture the current game screen using the best available method
        
        Returns:
            numpy.ndarray: The captured screen as BGR image
        """
        with self.capture_lock:
            # Try the preferred method first
            if self.preferred_method and self.preferred_method not in self.failed_methods:
                try:
                    frame = self._capture_with_method(self.preferred_method)
                    if frame is not None:
                        self._update_fps()
                        self.consecutive_failures = 0
                        self.last_successful_method = self.preferred_method
                        return frame
                except Exception as e:
                    logger.error(f"Error with preferred capture method {self.preferred_method}: {e}")
                    self.failed_methods.add(self.preferred_method)
            
            # Try other methods if preferred method failed
            for method in CAPTURE_METHODS:
                if method != self.preferred_method and method not in self.failed_methods:
                    try:
                        frame = self._capture_with_method(method)
                        if frame is not None:
                            logger.info(f"Successfully captured using fallback method: {method}")
                            self._update_fps()
                            self.consecutive_failures = 0
                            self.last_successful_method = method
                            self.preferred_method = method  # Update preferred method
                            return frame
                    except Exception as e:
                        logger.error(f"Error with fallback capture method {method}: {e}")
                        self.failed_methods.add(method)
            
            # If all methods failed, use test frame
            self.consecutive_failures += 1
            
            # Reset failed methods list after several attempts to try again
            if self.consecutive_failures % 10 == 0:
                logger.warning(f"Resetting failed methods list after {self.consecutive_failures} failures")
                self.failed_methods = set()
            
            # Generate a test frame
            logger.warning(f"All capture methods failed, using test frame (failure #{self.consecutive_failures})")
            return self._generate_test_frame()
    
    def _capture_with_method(self, method):
        """Capture using a specific method"""
        if method == "mss":
            return self._capture_mss()
        elif method == "pil":
            return self._capture_pil()
        elif method == "x11":
            return self._capture_x11()
        elif method == "win32":
            return self._capture_win32()
        elif method == "d3d":
            return self._capture_d3d()
        else:
            return None
    
    def _capture_mss(self):
        """Capture using MSS"""
        img = np.array(self.mss_sct.grab(self.game_region))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def _capture_pil(self):
        """Capture using PIL/Pillow"""
        left = self.game_region['left']
        top = self.game_region['top']
        right = left + self.game_region['width']
        bottom = top + self.game_region['height']
        
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def _capture_x11(self):
        """Capture using X11 on Linux"""
        x = self.game_region['left']
        y = self.game_region['top']
        width = self.game_region['width']
        height = self.game_region['height']
        
        raw = self.root.get_image(
            x, y, width, height,
            X.ZPixmap, 0xffffffff
        )
        
        # Create frame from image data
        # Note: Format depends on X11 settings, we'll try RGB depth=24 first
        try:
            img = np.frombuffer(raw.data, dtype=np.uint8)
            img = img.reshape((height, width, 4))  # Assuming BGRA format
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            # Try alternative format if first attempt fails
            try:
                img = np.frombuffer(raw.data, dtype=np.uint8)
                img = img.reshape((height, width, 3))  # Assuming BGR format
                return img.copy()  # Return a copy to ensure memory safety
            except Exception as e2:
                logger.error(f"Failed to process X11 image: {e}, {e2}")
                raise
    
    def _capture_win32(self):
        """Capture using Win32 API on Windows"""
        left = self.game_region['left']
        top = self.game_region['top']
        width = self.game_region['width']
        height = self.game_region['height']
        
        # Get a handle to the desktop window
        hdesktop = win32gui.GetDesktopWindow()
        
        # Create device context for entire screen
        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        mem_dc = img_dc.CreateCompatibleDC()
        
        # Create bitmap object
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, width, height)
        mem_dc.SelectObject(screenshot)
        
        # Copy screen to bitmap
        mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)
        
        # Convert bitmap to numpy array
        signedIntsArray = screenshot.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)
        
        # Free resources
        mem_dc.DeleteDC()
        win32gui.DeleteObject(screenshot.GetHandle())
        
        # Convert to BGR format
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def _capture_d3d(self):
        """Capture using D3DShot (DirectX) on Windows"""
        # Grab the frame
        frame = self.d3d_capture.screenshot(
            region=(
                self.game_region['left'],
                self.game_region['top'],
                self.game_region['left'] + self.game_region['width'],
                self.game_region['top'] + self.game_region['height']
            )
        )
        
        # D3DShot returns in RGB format, convert to BGR for OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def _generate_test_frame(self):
        """Generate a test frame for simulation"""
        # Create a dark frame with some visual elements
        width = self.game_region['width']
        height = self.game_region['height']
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some environment elements
        cv2.rectangle(frame, (0, height//2), (width, height), (50, 50, 50), -1)  # Floor
        
        # Generate simulated game elements
        current_time = int(time.time())
        
        # Add some red elements (potential zombies)
        for i in range(3 + (current_time % 5)):
            # Position varies with time
            x = (current_time * 50 + i * 100) % width
            y = height//2 + (i * 50) % (height//2 - 100)
            size = 30 + (i % 3) * 10
            
            # Make it reddish (to trigger zombie detection)
            color = (0, 0, 150 + (i * 30) % 100)
            cv2.circle(frame, (x, y), size, color, -1)
        
        # Add HUD elements
        # Health bar (red)
        health = max(10, 100 - (current_time % 100))
        cv2.rectangle(frame, (10, height - 30), (10 + health*2, height - 20), (0, 0, 255), -1)
        
        # Ammo counter
        ammo = 30 - (current_time % 31)
        cv2.putText(frame, f"Ammo: {ammo}", (width-150, height-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Round indicator
        round_num = 1 + (current_time % 10)
        cv2.putText(frame, f"Round: {round_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add "TEST MODE" indicator
        cv2.putText(frame, "TEST MODE", (width//2 - 80, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate FPS
        self._update_fps()
        
        return frame
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.prev_time
        
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.prev_time = current_time
            logger.debug(f"Screen capture FPS: {self.fps:.1f}")
    
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

    def get_status(self):
        """
        Get the current status of the screen capture
        
        Returns:
            dict: Status information
        """
        return {
            'fps': self.fps,
            'preferred_method': self.preferred_method,
            'available_methods': CAPTURE_METHODS,
            'last_successful_method': self.last_successful_method,
            'consecutive_failures': self.consecutive_failures,
            'failed_methods': list(self.failed_methods),
            'region': self.game_region
        }
    
    def __del__(self):
        """Clean up resources"""
        if self.mss_sct:
            self.mss_sct.close()
        
        if self.d3d_capture:
            self.d3d_capture.stop()


# Function to test the screen capture
def test_capture(duration=10):
    """Test the screen capture for a specified duration"""
    capture = EnhancedScreenCapture()
    
    print(f"Testing screen capture for {duration} seconds...")
    print(f"Available methods: {CAPTURE_METHODS}")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        # Capture frame
        frame = capture.capture()
        frame_count += 1
        
        # Display captured frame
        cv2.imshow('Captured Frame', frame)
        
        # Add status information
        status = capture.get_status()
        info_frame = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(info_frame, f"FPS: {status['fps']:.1f}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(info_frame, f"Method: {status['preferred_method']}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(info_frame, f"Last method: {status['last_successful_method']}", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(info_frame, f"Failures: {status['consecutive_failures']}", (10, 120), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(info_frame, f"Available: {', '.join(CAPTURE_METHODS)}", (10, 150), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(info_frame, f"Failed: {', '.join(status['failed_methods'])}", (10, 180), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Status', info_frame)
        
        # Wait for a key press (1ms delay) - allows window to refresh
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cv2.destroyAllWindows()
    
    # Print results
    elapsed = time.time() - start_time
    print(f"Captured {frame_count} frames in {elapsed:.1f} seconds")
    print(f"Average FPS: {frame_count / elapsed:.1f}")
    print(f"Final status: {capture.get_status()}")


if __name__ == "__main__":
    # Test the screen capture if run directly
    test_capture()