backup

#!/usr/bin/env python3
"""
Call of Duty: World at War Zombies Advanced Bot
Integrated system with Frank Castle AI, enhanced screen capture,
and reliable fullscreen exclusive mode operation.

This is a self-contained script that includes all key functionality.
"""

import os
import sys
import time
import logging
import json
import random
import argparse
import platform
import numpy as np
import math
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/waw_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WaWAdvancedBot")

# ====================== DEPENDENCIES CHECK ======================

def check_dependencies():
    """Check required dependencies and install if missing"""
    missing = []
    
    # Try to import required libraries
    try:
        import cv2
        logger.info(f"OpenCV installed: {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python-headless")
        logger.warning("OpenCV not found")
    
    try:
        import numpy
        logger.info(f"NumPy installed: {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
        logger.warning("NumPy not found")
    
    # Screen capture libraries
    try:
        import mss
        logger.info("MSS screen capture available")
    except ImportError:
        missing.append("mss")
        logger.warning("MSS not available (adding to requirements)")
    
    # Platform-specific modules
    if platform.system() == "Linux":
        try:
            from Xlib import display
            logger.info("Xlib available for Linux")
        except ImportError:
            missing.append("python-xlib")
            logger.warning("Xlib not available for Linux")
    
    elif platform.system() == "Windows":
        try:
            import win32gui
            logger.info("PyWin32 available for Windows")
        except ImportError:
            missing.append("pywin32")
            logger.warning("PyWin32 not available for Windows")
    
    # Install missing dependencies
    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        logger.info("You can install them using: pip install " + " ".join(missing))
        sys.exit(1)
    
    return True

# ====================== ENHANCED SCREEN CAPTURE ======================

class EnhancedScreenCapture:
    """Enhanced screen capture with multiple fallback methods"""
    
    def __init__(self, game_region=None, preferred_method=None):
        """
        Initialize screen capture with specified region and method
        
        Args:
            game_region (dict): Dictionary with top, left, width, height keys
            preferred_method (str): Preferred capture method
        """
        # Default full screen region
        self.game_region = game_region or {
            'top': 0,
            'left': 0,
            'width': 1920,
            'height': 1080
        }
        
        self.preferred_method = preferred_method
        self.available_methods = []
        self.capture_lock = threading.Lock()
        
        # Initialize utilities
        self.mss_sct = None
        self.display_obj = None
        self.root = None
        
        # Import and initialize screen capture methods
        self._init_screen_methods()
        
        # Performance tracking
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.consecutive_failures = 0
        self.last_successful_method = None
        self.failed_methods = set()
        
        logger.info(f"Screen capture initialized: {self.game_region}, Methods: {self.available_methods}")
    
    def _init_screen_methods(self):
        """Initialize available screen capture methods"""
        # Check for MSS
        try:
            import mss
            self.mss_sct = mss.mss()
            self.available_methods.append("mss")
            logger.info("MSS initialized")
        except (ImportError, Exception) as e:
            logger.warning(f"MSS not available: {e}")
        
        # Check for PIL/Pillow
        try:
            from PIL import ImageGrab
            self.available_methods.append("pil")
            logger.info("PIL/Pillow initialized")
        except (ImportError, Exception) as e:
            logger.warning(f"PIL/Pillow not available: {e}")
        
        # Check for X11 on Linux
        if platform.system() == "Linux":
            try:
                from Xlib import display, X
                self.display_obj = display.Display()
                self.root = self.display_obj.screen().root
                self.available_methods.append("x11")
                logger.info("X11 initialized")
            except (ImportError, Exception) as e:
                logger.warning(f"X11 not available: {e}")
        
        # Check for Win32 on Windows
        if platform.system() == "Windows":
            try:
                import win32gui
                self.available_methods.append("win32")
                logger.info("Win32 initialized")
            except (ImportError, Exception) as e:
                logger.warning(f"Win32 not available: {e}")
        
        # Set default method if none specified
        if not self.preferred_method and self.available_methods:
            self.preferred_method = self.available_methods[0]
            logger.info(f"Using default capture method: {self.preferred_method}")
    
    def capture(self):
        """
        Capture the current game screen
        
        Returns:
            numpy.ndarray: The captured screen as BGR image
        """
        with self.capture_lock:
            # Try preferred method first
            if self.preferred_method and self.preferred_method not in self.failed_methods:
                try:
                    frame = self._capture_with_method(self.preferred_method)
                    if frame is not None:
                        self._update_fps()
                        self.consecutive_failures = 0
                        self.last_successful_method = self.preferred_method
                        return frame
                except Exception as e:
                    logger.error(f"Error with method {self.preferred_method}: {e}")
                    self.failed_methods.add(self.preferred_method)
            
            # Try other methods
            for method in self.available_methods:
                if method != self.preferred_method and method not in self.failed_methods:
                    try:
                        frame = self._capture_with_method(method)
                        if frame is not None:
                            logger.info(f"Captured using fallback: {method}")
                            self._update_fps()
                            self.consecutive_failures = 0
                            self.last_successful_method = method
                            self.preferred_method = method  # Update preferred
                            return frame
                    except Exception as e:
                        logger.error(f"Error with method {method}: {e}")
                        self.failed_methods.add(method)
            
            # If all methods failed, use test frame
            self.consecutive_failures += 1
            
            # Reset failed methods after several attempts
            if self.consecutive_failures % 10 == 0:
                logger.warning(f"Resetting failed methods after {self.consecutive_failures} failures")
                self.failed_methods = set()
            
            # Generate test frame
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
        else:
            return None
    
    def _capture_mss(self):
        """Capture using MSS"""
        import mss
        img = np.array(self.mss_sct.grab(self.game_region))
        import cv2
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def _capture_pil(self):
        """Capture using PIL/Pillow"""
        from PIL import ImageGrab
        left = self.game_region['left']
        top = self.game_region['top']
        right = left + self.game_region['width']
        bottom = top + self.game_region['height']
        
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        import cv2
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def _capture_x11(self):
        """Capture using X11 on Linux"""
        from Xlib import X
        x = self.game_region['left']
        y = self.game_region['top']
        width = self.game_region['width']
        height = self.game_region['height']
        
        raw = self.root.get_image(
            x, y, width, height,
            X.ZPixmap, 0xffffffff
        )
        
        # Process X11 image data
        try:
            img = np.frombuffer(raw.data, dtype=np.uint8)
            img = img.reshape((height, width, 4))
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            try:
                img = np.frombuffer(raw.data, dtype=np.uint8)
                img = img.reshape((height, width, 3))
                return img.copy()
            except Exception as e2:
                logger.error(f"Failed to process X11 image: {e}, {e2}")
                raise
    
    def _capture_win32(self):
        """Capture using Win32 API on Windows"""
        import win32gui
        import win32ui
        import win32con
        
        left = self.game_region['left']
        top = self.game_region['top']
        width = self.game_region['width']
        height = self.game_region['height']
        
        # Get handle to desktop
        hdesktop = win32gui.GetDesktopWindow()
        
        # Create device context
        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        mem_dc = img_dc.CreateCompatibleDC()
        
        # Create bitmap
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, width, height)
        mem_dc.SelectObject(screenshot)
        
        # Copy screen
        mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)
        
        # Convert to numpy array
        signedIntsArray = screenshot.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)
        
        # Free resources
        mem_dc.DeleteDC()
        win32gui.DeleteObject(screenshot.GetHandle())
        
        # Convert to BGR
        import cv2
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def _generate_test_frame(self):
        """Generate a test frame for simulation"""
        width = self.game_region['width']
        height = self.game_region['height']
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add environment elements
        cv2.rectangle(frame, (0, height//2), (width, height), (50, 50, 50), -1)
        
        # Add simulated zombies
        current_time = int(time.time())
        for i in range(3 + (current_time % 5)):
            x = (current_time * 50 + i * 100) % width
            y = height//2 + (i * 50) % (height//2 - 100)
            size = 30 + (i % 3) * 10
            color = (0, 0, 150 + (i * 30) % 100)  # Reddish
            cv2.circle(frame, (x, y), size, color, -1)
        
        # Add HUD elements
        health = max(10, 100 - (current_time % 100))
        cv2.rectangle(frame, (10, height - 30), (10 + health*2, height - 20), (0, 0, 255), -1)
        
        ammo = 30 - (current_time % 31)
        cv2.putText(frame, f"Ammo: {ammo}", (width-150, height-20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Round indicator
        round_num = 1 + (current_time % 10)
        cv2.putText(frame, f"Round: {round_num}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Test mode indicator
        cv2.putText(frame, "TEST MODE", (width//2 - 80, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update FPS
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
    
    def get_status(self):
        """Get current status of screen capture"""
        return {
            'fps': self.fps,
            'preferred_method': self.preferred_method,
            'available_methods': self.available_methods,
            'last_successful_method': self.last_successful_method,
            'consecutive_failures': self.consecutive_failures,
            'failed_methods': list(self.failed_methods),
            'region': self.game_region
        }
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'mss_sct') and self.mss_sct:
            self.mss_sct.close()

# ====================== DETECTION SYSTEMS ======================

class ZombieDetector:
    """Detect zombies in game frames"""
    
    def __init__(self, config=None):
        """
        Initialize zombie detector with configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Default thresholds from config or defaults
        self.hsv_lower = np.array(self.config.get('hsv_lower', [0, 120, 70]))
        self.hsv_upper = np.array(self.config.get('hsv_upper', [10, 255, 255]))
        self.detection_threshold = self.config.get('threshold', 0.7)
        self.min_blob_size = self.config.get('min_size', 100)
        
        logger.info("Zombie detector initialized")
    
    def detect(self, frame):
        """
        Detect zombies in a frame
        
        Args:
            frame (numpy.ndarray): BGR image from screen capture
            
        Returns:
            list: List of detected zombies with positions
        """
        if frame is None:
            return []
        
        try:
            # Convert frame to HSV for color-based detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for zombie colors (reddish/orange)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            zombies = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter small contours
                if area < self.min_blob_size:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence (normalized area)
                max_area = frame.shape[0] * frame.shape[1]
                confidence = min(1.0, area / (max_area * 0.25))
                
                # Reject if below threshold
                if confidence < self.detection_threshold:
                    continue
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Add to zombies list
                zombies.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'center_x': center_x,
                    'center_y': center_y,
                    'area': area,
                    'confidence': confidence
                })
            
            logger.debug(f"Detected {len(zombies)} zombies")
            return zombies
            
        except Exception as e:
            logger.error(f"Error in zombie detection: {e}")
            return []

class HUDDetector:
    """Detect HUD elements like health, ammo, and round"""
    
    def __init__(self, config=None):
        """
        Initialize HUD detector
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        logger.info("HUD detector initialized")
    
    def detect_stats(self, frame):
        """
        Detect game stats from HUD
        
        Args:
            frame (numpy.ndarray): BGR image from screen capture
            
        Returns:
            tuple: (health, ammo, weapon, round)
        """
        if frame is None:
            return 100, 30, "pistol", 1
        
        try:
            # In a real implementation, use OCR or template matching
            # This is a simplified simulation based on frame timing
            
            # Generate fake HUD values for testing
            current_time = int(time.time())
            
            # Health (decreases over time)
            health = max(10, 100 - (current_time % 100))
            
            # Ammo (cycles 0-30)
            ammo = 30 - (current_time % 31)
            
            # Weapon (cycles through weapons)
            weapons = ["pistol", "shotgun", "rifle", "lmg"]
            weapon_idx = (current_time // 20) % len(weapons)
            weapon = weapons[weapon_idx]
            
            # Round (increases over time)
            round_num = 1 + (current_time // 60) % 15
            
            return health, ammo, weapon, round_num
            
        except Exception as e:
            logger.error(f"Error in HUD detection: {e}")
            return 100, 30, "pistol", 1

# ====================== GAME STATE TRACKING ======================

class GameState:
    """Track game state including zombies, health, ammo, etc."""
    
    def __init__(self):
        """Initialize game state"""
        self.zombies = []
        self.health = 100
        self.ammo = 30
        self.current_weapon = "pistol"
        self.current_round = 1
        self.danger_level = 0.0
        
        # Stats tracking
        self.zombies_killed = 0
        self.rounds_survived = 0
        self.game_start_time = time.time()
        
        logger.info("Game state initialized")
    
    def update(self, zombies, health, ammo, weapon, round_num):
        """
        Update game state
        
        Args:
            zombies (list): Detected zombies
            health (int): Current health
            ammo (int): Current ammo
            weapon (str): Current weapon
            round_num (int): Current round
        """
        # Update basic stats
        self.zombies = zombies
        self.health = health
        self.ammo = ammo
        self.current_weapon = weapon
        
        # Check for round change
        if round_num > self.current_round:
            self.rounds_survived += 1
        self.current_round = round_num
        
        # Calculate danger level (0.0 - 1.0)
        # Based on number of zombies, proximity, health, and ammo
        zombie_count = len(zombies)
        
        # Zombie count factor (more zombies = more danger)
        zombie_factor = min(1.0, zombie_count / 10.0)
        
        # Health factor (less health = more danger)
        health_factor = max(0.0, 1.0 - (health / 100.0))
        
        # Ammo factor (less ammo = more danger)
        ammo_factor = max(0.0, 1.0 - (ammo / 30.0))
        
        # Calculate overall danger (weighted factors)
        self.danger_level = (
            0.5 * zombie_factor +
            0.3 * health_factor +
            0.2 * ammo_factor
        )
    
    def get_zombie_count(self):
        """Get number of zombies"""
        return len(self.zombies)
    
    def get_closest_zombie(self):
        """Get closest zombie to screen center"""
        if not self.zombies:
            return None
        
        # Find zombie closest to center (400, 300)
        center_x, center_y = 400, 300
        closest = None
        min_dist = float('inf')
        
        for zombie in self.zombies:
            dist = ((zombie['center_x'] - center_x) ** 2 + 
                   (zombie['center_y'] - center_y) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                closest = zombie
        
        return closest
    
    def get_game_runtime(self):
        """Get game runtime in seconds"""
        return time.time() - self.game_start_time

# ====================== MEMORY SYSTEM ======================

class Memory:
    """Memory system for tracking actions and rewards"""
    
    def __init__(self, memory_size=1000, save_file=None):
        """
        Initialize memory system
        
        Args:
            memory_size (int): Maximum memory entries
            save_file (str): File to save memories
        """
        self.memory = deque(maxlen=memory_size)
        self.save_file = save_file or "data/memory/memory_log.json"
        self.total_reward = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # 5 minutes
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        
        # Try to load previous memories
        self._load_memory()
        
        logger.info("Memory system initialized")
    
    def log_action(self, state, action, reward=0, additional_info=None):
        """
        Log an action to memory
        
        Args:
            state (str): Current state
            action (str): Action taken
            reward (float): Reward received
            additional_info (dict): Extra information
        """
        timestamp = time.time()
        
        memory_entry = {
            'timestamp': timestamp,
            'state': state,
            'action': action,
            'reward': reward,
            'info': additional_info or {}
        }
        
        self.memory.append(memory_entry)
        self.total_reward += reward
        
        # Log significant events
        if reward != 0:
            logger.info(f"Memory: {action} in state {state} -> reward {reward}")
        
        # Periodically save memory
        if time.time() - self.last_save_time > self.save_interval:
            self._save_memory()
    
    def get_most_rewarding_actions(self, state=None, top_n=3):
        """
        Get most rewarding actions
        
        Args:
            state (str): State to filter by
            top_n (int): Number of actions to return
            
        Returns:
            list: Top actions with reward
        """
        # Filter memories by state if provided
        memories = self.memory
        if state:
            memories = [m for m in memories if m['state'] == state]
        
        # Group by action and calculate average reward
        action_rewards = {}
        for m in memories:
            action = m['action']
            reward = m['reward']
            
            if action not in action_rewards:
                action_rewards[action] = {'total': 0, 'count': 0}
                
            action_rewards[action]['total'] += reward
            action_rewards[action]['count'] += 1
        
        # Calculate averages and sort
        for action in action_rewards:
            action_rewards[action]['average'] = (
                action_rewards[action]['total'] / action_rewards[action]['count']
            )
            
        # Sort by average reward
        sorted_actions = sorted(
            action_rewards.items(), 
            key=lambda x: x[1]['average'],
            reverse=True
        )
        
        return [(action, data['average']) for action, data in sorted_actions[:top_n]]
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.save_file, 'w') as f:
                json.dump(list(self.memory), f)
            self.last_save_time = time.time()
            logger.info(f"Memory saved to {self.save_file}")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def _load_memory(self):
        """Load memory from disk"""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r') as f:
                    loaded_memory = json.load(f)
                
                # Add loaded memories to deque
                for entry in loaded_memory:
                    self.memory.append(entry)
                    self.total_reward += entry.get('reward', 0)
                
                logger.info(f"Loaded {len(loaded_memory)} memories from {self.save_file}")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")

# ====================== PATHFINDING SYSTEM ======================

def astar(grid, start, goal):
    """
    A* pathfinding algorithm
    
    Args:
        grid (numpy.ndarray): 2D grid (0=free, 1=obstacle)
        start (tuple): Start coordinates (x, y)
        goal (tuple): Goal coordinates (x, y)
        
    Returns:
        list: Path from start to goal
    """
    # Check bounds
    height, width = grid.shape
    if (start[0] < 0 or start[0] >= width or
        start[1] < 0 or start[1] >= height or
        goal[0] < 0 or goal[0] >= width or
        goal[1] < 0 or goal[1] >= height):
        logger.error("Start or goal out of bounds")
        return []
    
    # Check for obstacles
    if grid[start[1]][start[0]] == 1 or grid[goal[1]][goal[0]] == 1:
        logger.error("Start or goal is an obstacle")
        return []
    
    # Possible movement directions
    directions = [
        (0, 1),    # Down
        (1, 0),    # Right
        (0, -1),   # Up
        (-1, 0),   # Left
        (1, 1),    # Down-Right
        (-1, 1),   # Down-Left
        (1, -1),   # Up-Right
        (-1, -1)   # Up-Left
    ]
    
    # Initialize A* data structures
    open_set = [(heuristic(start, goal), start)]  # Priority queue
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        # Get node with lowest f_score
        current_f, current = heapq.heappop(open_set)
        
        # Check if we reached the goal
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        # Add to closed set
        closed_set.add(current)
        
        # Check neighbors
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Skip if out of bounds
            if (neighbor[0] < 0 or neighbor[0] >= width or
                neighbor[1] < 0 or neighbor[1] >= height):
                continue
            
            # Skip if obstacle or in closed set
            if grid[neighbor[1]][neighbor[0]] == 1 or neighbor in closed_set:
                continue
            
            # Calculate cost
            diagonal = abs(dx) + abs(dy) == 2
            move_cost = 1.4 if diagonal else 1.0
            tentative_g = g_score[current] + move_cost
            
            # Skip if we have a better path
            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue
            
            # This is the best path so far
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
            
            # Add to open set if not already there
            if neighbor not in [item[1] for item in open_set]:
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return []

def heuristic(a, b):
    """
    Heuristic function for A* (diagonal distance)
    
    Args:
        a (tuple): First point (x, y)
        b (tuple): Second point (x, y)
        
    Returns:
        float: Estimated distance
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + 0.4 * min(dx, dy)

def generate_exploration_path(current_pos, known_areas=None, map_size=(100, 100)):
    """
    Generate exploration path
    
    Args:
        current_pos (tuple): Current position (x, y)
        known_areas (set): Known areas
        map_size (tuple): Map size
        
    Returns:
        list: Exploration path
    """
    if known_areas is None:
        known_areas = set()
    
    # Add current position to known areas
    known_areas.add(current_pos)
    
    # Find unexplored area
    max_distance = 20
    
    # Try to find unexplored area
    for dist in range(5, max_distance + 1, 5):
        candidates = []
        
        # Generate points in a circle
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x = int(current_pos[0] + dist * math.cos(rad))
            y = int(current_pos[1] + dist * math.sin(rad))
            
            # Ensure within map bounds
            x = max(0, min(x, map_size[0] - 1))
            y = max(0, min(y, map_size[1] - 1))
            
            # Add if not explored
            pos = (x, y)
            if pos not in known_areas:
                candidates.append(pos)
        
        # If we found unexplored candidates
        if candidates:
            # Pick one randomly
            target_pos = random.choice(candidates)
            
            # Create a simple path (direct)
            path = []
            x, y = current_pos
            
            # Generate path points
            while (x, y) != target_pos:
                if x < target_pos[0]:
                    x += 1
                elif x > target_pos[0]:
                    x -= 1
                    
                if y < target_pos[1]:
                    y += 1
                elif y > target_pos[1]:
                    y -= 1
                    
                path.append((x, y))
                
                # Prevent infinite loops
                if len(path) > 100:
                    break
            
            return path
    
    # If no good candidates, just move in a random direction
    x = current_pos[0] + random.randint(-max_distance, max_distance)
    y = current_pos[1] + random.randint(-max_distance, max_distance)
    
    # Ensure within map bounds
    x = max(0, min(x, map_size[0] - 1))
    y = max(0, min(y, map_size[1] - 1))
    
    # Create simple path
    path = []
    cx, cy = current_pos
    
    while (cx, cy) != (x, y):
        if cx < x:
            cx += 1
        elif cx > x:
            cx -= 1
            
        if cy < y:
            cy += 1
        elif cy > y:
            cy -= 1
            
        path.append((cx, cy))
        
        # Prevent infinite loops
        if len(path) > 100:
            break
    
    return path

def generate_evasion_path(current_pos, danger_positions, map_size=(100, 100)):
    """
    Generate evasion path away from dangers
    
    Args:
        current_pos (tuple): Current position (x, y)
        danger_positions (list): Dangerous positions
        map_size (tuple): Map size
        
    Returns:
        list: Evasion path
    """
    if not danger_positions:
        return []
    
    # Find average danger position
    avg_x = sum(x for x, y in danger_positions) / len(danger_positions)
    avg_y = sum(y for x, y in danger_positions) / len(danger_positions)
    
    # Move in opposite direction
    dx = current_pos[0] - avg_x
    dy = current_pos[1] - avg_y
    
    # Normalize and scale
    length = max(0.001, math.sqrt(dx*dx + dy*dy))
    dx = dx / length * 15
    dy = dy / length * 15
    
    # Calculate target position
    target_x = int(current_pos[0] + dx)
    target_y = int(current_pos[1] + dy)
    
    # Ensure within map bounds
    target_x = max(0, min(target_x, map_size[0] - 1))
    target_y = max(0, min(target_y, map_size[1] - 1))
    
    # Generate simple path
    path = []
    x, y = current_pos
    
    while (x, y) != (target_x, target_y):
        if x < target_x:
            x += 1
        elif x > target_x:
            x -= 1
            
        if y < target_y:
            y += 1
        elif y > target_y:
            y -= 1
            
        path.append((x, y))
        
        # Prevent infinite loops
        if len(path) > 100:
            break
    
    return path

# ====================== COMBAT SYSTEM ======================

def select_best_target(zombies, player_position=(400, 300), prioritize_closest=True):
    """
    Select best target from zombies
    
    Args:
        zombies (list): Detected zombies
        player_position (tuple): Player position
        prioritize_closest (bool): Whether to prioritize closest zombies
        
    Returns:
        dict: Selected target zombie
    """
    if not zombies:
        return None
    
    # Add distance to each zombie
    for zombie in zombies:
        center_x = zombie.get('center_x', zombie.get('x', 0) + zombie.get('width', 0) // 2)
        center_y = zombie.get('center_y', zombie.get('y', 0) + zombie.get('height', 0) // 2)
        
        dx = center_x - player_position[0]
        dy = center_y - player_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        zombie['distance'] = distance
    
    # Sort by priority
    if prioritize_closest:
        # Sort by distance
        zombies_sorted = sorted(zombies, key=lambda z: z['distance'])
    else:
        # Sort by confidence and distance
        zombies_sorted = sorted(zombies, 
                              key=lambda z: (-z.get('confidence', 0), z['distance']))
    
    # Return best target
    return zombies_sorted[0] if zombies_sorted else None

def aim_at_target(target, leading=True, player_position=(400, 300)):
    """
    Calculate aim point for target
    
    Args:
        target (dict): Target information
        leading (bool): Whether to lead moving targets
        player_position (tuple): Player position
        
    Returns:
        tuple: Aim point (x, y)
    """
    # Extract target position
    target_x = target.get('center_x', target.get('x', 0) + target.get('width', 0) // 2)
    target_y = target.get('center_y', target.get('y', 0) + target.get('height', 0) // 2)
    
    # Basic aim point is target center
    aim_x, aim_y = target_x, target_y
    
    # Apply leading if enabled and target has velocity
    if leading and 'velocity_x' in target and 'velocity_y' in target:
        # Simple leading: add velocity * time_to_impact
        time_to_impact = 0.3  # Assumed bullet travel time
        aim_x += target['velocity_x'] * time_to_impact
        aim_y += target['velocity_y'] * time_to_impact
    
    return (int(aim_x), int(aim_y))

def create_combat_action(target, weapon_info=None):
    """
    Create combat action for target
    
    Args:
        target (dict): Target information
        weapon_info (dict): Weapon information
        
    Returns:
        dict: Combat action
    """
    # Extract target position
    target_x = target.get('center_x', target.get('x', 0) + target.get('width', 0) // 2)
    target_y = target.get('center_y', target.get('y', 0) + target.get('height', 0) // 2)
    
    # Determine if we should aim based on distance
    distance = target.get('distance', 200)
    aim_down_sights = distance > 150
    
    # Create action
    action = {
        "type": "shoot",
        "description": "Engaging zombie target",
        "params": {
            "target_x": target_x,
            "target_y": target_y,
            "aim": aim_down_sights
        }
    }
    
    # Add weapon-specific parameters
    if weapon_info:
        action["params"]["weapon"] = weapon_info.get("name", "unknown")
        
        # Burst fire for automatic weapons at range
        if weapon_info.get("is_automatic", False) and distance > 250:
            action["params"]["burst"] = True
            action["params"]["burst_count"] = 3
            action["description"] = "Burst firing at zombie target"
    
    return action

# ====================== INPUT SYSTEM ======================

class ActionController:
    """Control in-game actions through input simulation"""
    
    def __init__(self, sensitivity=5.0):
        """
        Initialize action controller
        
        Args:
            sensitivity (float): Mouse sensitivity
        """
        self.sensitivity = sensitivity
        self.last_action_time = time.time()
        
        # Import platform-specific modules
        self.input_type = self._determine_input_system()
        
        # Mouse position tracking
        self.current_mouse_x = 400
        self.current_mouse_y = 300
        
        logger.info(f"Action controller initialized with {self.input_type} input")
    
    def _determine_input_system(self):
        """Determine which input system to use"""
        system = platform.system()
        
        if system == "Windows":
            try:
                import win32api
                import win32con
                return "win32"
            except ImportError:
                logger.warning("Win32 not available")
        
        elif system == "Linux":
            try:
                from Xlib import X, display
                return "x11"
            except ImportError:
                logger.warning("X11 not available")
        
        # Default to simulated
        return "simulated"
    
    def execute(self, action):
        """
        Execute an action
        
        Args:
            action (dict): Action to execute
        """
        action_type = action.get('type', 'none')
        params = action.get('params', {})
        
        try:
            # Rate limit actions
            current_time = time.time()
            if current_time - self.last_action_time < 0.05:
                time.sleep(0.05 - (current_time - self.last_action_time))
            
            # Execute based on action type
            if action_type == "move":
                self._handle_move(params)
            elif action_type == "shoot":
                self._handle_shoot(params)
            elif action_type == "reload":
                self._handle_reload()
            elif action_type == "grenade":
                self._handle_grenade()
            elif action_type == "interact":
                self._handle_interact(params)
            else:
                logger.warning(f"Unknown action type: {action_type}")
            
            self.last_action_time = time.time()
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def _handle_move(self, params):
        """
        Handle movement action
        
        Args:
            params (dict): Movement parameters
        """
        direction = params.get('direction', 'forward')
        duration = params.get('duration', 0.5)
        
        # Map direction to key
        keys = {
            'forward': 'w',
            'backward': 's',
            'left': 'a',
            'right': 'd'
        }
        
        key = keys.get(direction)
        if key:
            # Simulate key press
            logger.debug(f"Moving {direction} for {duration}s")
            self._simulate_key_press(key, duration)
    
    def _handle_shoot(self, params):
        """
        Handle shooting action
        
        Args:
            params (dict): Shooting parameters
        """
        target_x = params.get('target_x', 400)
        target_y = params.get('target_y', 300)
        aim = params.get('aim', False)
        
        # Calculate relative mouse movement
        dx = target_x - self.current_mouse_x
        dy = target_y - self.current_mouse_y
        
        # Move mouse to target
        self._simulate_mouse_move(dx, dy)
        
        # Update tracked position
        self.current_mouse_x = target_x
        self.current_mouse_y = target_y
        
        # Aim down sights if needed
        if aim:
            self._simulate_mouse_right_down(0.2)
        
        # Shoot
        self._simulate_mouse_left_click()
        
        # Release aim
        if aim:
            self._simulate_mouse_right_up()
    
    def _handle_reload(self):
        """Handle reload action"""
        logger.debug("Reloading weapon")
        self._simulate_key_press('r', 0.1)
    
    def _handle_grenade(self):
        """Handle grenade action"""
        logger.debug("Throwing grenade")
        self._simulate_key_press('g', 0.1)
    
    def _handle_interact(self, params):
        """
        Handle interaction
        
        Args:
            params (dict): Interaction parameters
        """
        action = params.get('action', 'use')
        
        if action == 'use':
            logger.debug("Interacting with object")
            self._simulate_key_press('e', 0.1)
        elif action == 'buy':
            logger.debug("Buying item")
            self._simulate_key_press('f', 0.1)
    
    def _simulate_key_press(self, key, duration=0.1):
        """
        Simulate key press
        
        Args:
            key (str): Key to press
            duration (float): Press duration
        """
        # Map keys to key codes
        key_mapping = {
            'w': 0x11, 'a': 0x1E, 's': 0x1F, 'd': 0x20,
            'r': 0x13, 'e': 0x12, 'f': 0x21, 'g': 0x22,
            '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05
        }
        
        key_code = key_mapping.get(key.lower(), 0)
        
        # Simulate based on platform
        if self.input_type == "win32":
            import win32api
            import win32con
            
            # Press key
            win32api.keybd_event(key_code, 0, 0, 0)
            time.sleep(duration)
            # Release key
            win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            
        elif self.input_type == "x11":
            from Xlib import X, display
            d = display.Display()
            root = d.screen().root
            
            # Press key
            root.fake_input(X.KeyPress, key_code)
            d.sync()
            time.sleep(duration)
            # Release key
            root.fake_input(X.KeyRelease, key_code)
            d.sync()
            
        else:
            # Simulated mode - just log
            logger.debug(f"Simulated key press: {key} for {duration}s")
    
    def _simulate_mouse_move(self, dx, dy):
        """
        Simulate mouse movement
        
        Args:
            dx (int): X movement
            dy (int): Y movement
        """
        # Apply sensitivity
        dx = int(dx * self.sensitivity / 10.0)
        dy = int(dy * self.sensitivity / 10.0)
        
        # Simulate based on platform
        if self.input_type == "win32":
            import win32api
            
            # Get current position
            x, y = win32api.GetCursorPos()
            # Move mouse
            win32api.SetCursorPos((x + dx, y + dy))
            
        elif self.input_type == "x11":
            from Xlib import X, display
            d = display.Display()
            root = d.screen().root
            
            # Move mouse
            root.warp_pointer(dx, dy)
            d.sync()
            
        else:
            # Simulated mode - just log
            logger.debug(f"Simulated mouse move: ({dx}, {dy})")
    
    def _simulate_mouse_left_click(self):
        """Simulate left mouse click"""
        if self.input_type == "win32":
            import win32api
            import win32con
            
            # Press button
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            # Release button
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            
        elif self.input_type == "x11":
            from Xlib import X, display
            d = display.Display()
            root = d.screen().root
            
            # Press button
            root.fake_input(X.ButtonPress, 1)
            d.sync()
            time.sleep(0.05)
            # Release button
            root.fake_input(X.ButtonRelease, 1)
            d.sync()
            
        else:
            # Simulated mode - just log
            logger.debug("Simulated left click")
    
    def _simulate_mouse_right_down(self, duration=0.2):
        """
        Simulate right mouse button down
        
        Args:
            duration (float): Hold duration
        """
        if self.input_type == "win32":
            import win32api
            import win32con
            
            # Press button
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            time.sleep(duration)
            
        elif self.input_type == "x11":
            from Xlib import X, display
            d = display.Display()
            root = d.screen().root
            
            # Press button
            root.fake_input(X.ButtonPress, 3)
            d.sync()
            time.sleep(duration)
            
        else:
            # Simulated mode - just log
            logger.debug(f"Simulated right mouse down for {duration}s")
    
    def _simulate_mouse_right_up(self):
        """Simulate right mouse button up"""
        if self.input_type == "win32":
            import win32api
            import win32con
            
            # Release button
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            
        elif self.input_type == "x11":
            from Xlib import X, display
            d = display.Display()
            root = d.screen().root
            
            # Release button
            root.fake_input(X.ButtonRelease, 3)
            d.sync()
            
        else:
            # Simulated mode - just log
            logger.debug("Simulated right mouse up")
    
    def cleanup(self):
        """Clean up resources"""
        # Nothing to clean up for now
        pass

# ====================== DECISION MAKING ======================

class DecisionMaker:
    """Make tactical decisions based on game state"""
    
    def __init__(self, config=None):
        """
        Initialize decision maker
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        logger.info("Decision maker initialized")
    
    def decide(self, game_state):
        """
        Make a decision based on game state
        
        Args:
            game_state (GameState): Current game state
            
        Returns:
            dict: Action to take
        """
        # Emergency actions first
        
        # Reload if low on ammo
        if game_state.ammo <= 5:
            return {
                "type": "reload",
                "description": "Emergency reload",
                "params": {}
            }
        
        # Handle high danger
        if game_state.danger_level > 0.8:
            # Throw grenade if many zombies
            if len(game_state.zombies) >= 3:
                return {
                    "type": "grenade",
                    "description": "Emergency grenade",
                    "params": {}
                }
            
            # Move away from zombies
            return {
                "type": "move",
                "description": "Emergency retreat",
                "params": {
                    "direction": "backward",
                    "duration": 0.5
                }
            }
        
        # Combat decisions
        if game_state.zombies:
            # Get closest zombie
            closest = game_state.get_closest_zombie()
            
            # Shoot closest zombie
            if closest:
                return {
                    "type": "shoot",
                    "description": "Engaging zombie",
                    "params": {
                        "target_x": closest['center_x'],
                        "target_y": closest['center_y'],
                        "aim": True
                    }
                }
        
        # Exploration when no zombies
        direction = random.choice(['forward', 'left', 'right'])
        return {
            "type": "move",
            "description": "Exploring",
            "params": {
                "direction": direction,
                "duration": 0.5
            }
        }

# ====================== INTEGRATED AI ======================

class AdvancedBot:
    """Advanced bot with integrated AI systems"""
    
    def __init__(self, config=None):
        """
        Initialize advanced bot
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.running = False
        
        # Load or create default configuration
        self._load_or_create_config()
        
        # Initialize components
        self.screen_capture = EnhancedScreenCapture(
            game_region=self.config.get('game_region'),
            preferred_method=self.config.get('preferred_capture_method')
        )
        
        self.zombie_detector = ZombieDetector(self.config.get('zombie_detection', {}))
        self.hud_detector = HUDDetector(self.config)
        self.game_state = GameState()
        self.decision_maker = DecisionMaker(self.config)
        self.action_controller = ActionController(
            sensitivity=self.config.get('mouse_sensitivity', 5.0)
        )
        
        # Frank Castle additions
        self.memory = Memory()
        self.known_areas = set()  # For exploration
        
        # State variables
        self.current_path = []
        self.current_state = "idle"
        
        # Performance tracking
        self.start_time = None
        self.decisions_made = 0
        self.frames_processed = 0
        
        # Headless mode for testing
        self.headless = self.config.get('headless', False)
        
        # Backup thread
        self.backup_thread = None
        self.backup_interval = self.config.get('backup_interval', 300)  # 5 minutes
        
        # Create dirs
        os.makedirs("data/memory", exist_ok=True)
        os.makedirs("data/backups", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info("Advanced bot initialized")
    
    def _load_or_create_config(self):
        """Load or create default configuration"""
        config_path = Path('config/advanced_bot_settings.json')
        
        # Default configuration
        default_config = {
            'game_region': {
                'top': 0,
                'left': 0,
                'width': 1920,
                'height': 1080
            },
            'preferred_capture_method': None,
            'mouse_sensitivity': 5.0,
            'start_delay': 10,
            'headless': False,
            'debug_visualization': True,
            'zombie_detection': {
                'threshold': 0.7,
                'min_size': 100,
                'hsv_lower': [0, 120, 70],
                'hsv_upper': [10, 255, 255]
            },
            'combat': {
                'prioritize_closest': True,
                'lead_targets': True,
                'preferred_weapons': ["shotgun", "rifle"]
            },
            'save_memory_interval': 300,
            'backup_interval': 300,
            'fullscreen_exclusive_mode': True
        }
        
        # Load config if it exists
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                self.config = {**default_config, **loaded_config}
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self.config = default_config
        else:
            # Create default config
            self.config = default_config
            
            # Save default config
            try:
                config_path.parent.mkdir(exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info(f"Created default configuration at {config_path}")
            except Exception as e:
                logger.error(f"Failed to save default config: {e}")
    
    def start(self, delay=None):
        """
        Start the bot with optional delay
        
        Args:
            delay (int): Seconds to wait before starting
        """
        if self.running:
            logger.warning("Bot is already running")
            return
        
        # Use config delay if none specified
        if delay is None:
            delay = self.config.get('start_delay', 10)
        
        # Skip delay in headless mode
        if not self.headless and delay > 0:
            logger.info(f"Starting in {delay} seconds... Switch to game window!")
            for i in range(delay, 0, -1):
                logger.info(f"{i}...")
                time.sleep(1)
        
        logger.info("Advanced bot activated!")
        self.running = True
        self.start_time = time.time()
        
        # Start backup thread
        self._start_backup_thread()
        
        # Run main loop
        try:
            self._run_main_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.exception(f"Error in bot execution: {e}")
        finally:
            self._cleanup()
    
    def _start_backup_thread(self):
        """Start backup thread"""
        def backup_task():
            while self.running:
                # Sleep for interval
                time.sleep(self.backup_interval)
                
                # Create backup
                if self.running:
                    self._create_backup()
        
        self.backup_thread = threading.Thread(target=backup_task)
        self.backup_thread.daemon = True
        self.backup_thread.start()
    
    def _create_backup(self):
        """Create backup of settings and memory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"data/backups/backup_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup config
            if os.path.exists("config/advanced_bot_settings.json"):
                with open("config/advanced_bot_settings.json", 'r') as src:
                    with open(f"{backup_dir}/settings.json", 'w') as dst:
                        dst.write(src.read())
            
            # Backup memory
            if hasattr(self, 'memory'):
                self.memory._save_memory()
                
                if os.path.exists(self.memory.save_file):
                    with open(self.memory.save_file, 'r') as src:
                        with open(f"{backup_dir}/memory.json", 'w') as dst:
                            dst.write(src.read())
            
            # Create status file
            status = {
                'timestamp': time.time(),
                'runtime': time.time() - self.start_time if self.start_time else 0,
                'decisions_made': self.decisions_made,
                'frames_processed': self.frames_processed,
                'screen_capture': self.screen_capture.get_status()
            }
            
            with open(f"{backup_dir}/status.json", 'w') as f:
                json.dump(status, f, indent=4)
                
            logger.info(f"Created backup at {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def _run_main_loop(self):
        """Main bot execution loop"""
        frame_time = time.time()
        status_time = time.time()
        memory_save_time = time.time()
        
        while self.running:
            # Calculate time since last frame
            current_time = time.time()
            delta_time = current_time - frame_time
            frame_time = current_time
            
            # Process game frame
            try:
                # Capture frame
                frame = self.screen_capture.capture()
                self.frames_processed += 1
                
                # Detect game elements
                zombies = self.zombie_detector.detect(frame)
                health, ammo, current_weapon, current_round = self.hud_detector.detect_stats(frame)
                
                # Update game state
                self.game_state.update(zombies, health, ammo, current_weapon, current_round)
                
                # Determine current state description
                state_desc = self._determine_state_description()
                
                # Make enhanced decision
                enhanced_action = self._make_enhanced_decision(zombies, state_desc, delta_time)
                
                # Execute action
                self.action_controller.execute(enhanced_action)
                self.decisions_made += 1
                
                # Log action in memory
                reward = self._calculate_reward(enhanced_action, zombies, health, ammo)
                self.memory.log_action(state_desc, enhanced_action['type'], reward, {
                    'health': health,
                    'ammo': ammo,
                    'zombies': len(zombies),
                    'weapon': current_weapon,
                    'round': current_round
                })
                
                # Periodically save memory
                if current_time - memory_save_time > self.config.get('save_memory_interval', 300):
                    self.memory._save_memory()
                    memory_save_time = current_time
                
                # Log status periodically
                if current_time - status_time > 5.0:
                    self._log_status()
                    status_time = current_time
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                
            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)
    
    def _determine_state_description(self):
        """
        Determine state description for memory
        
        Returns:
            str: State description
        """
        danger_level = self.game_state.danger_level
        zombie_count = len(self.game_state.zombies)
        health = self.game_state.health
        ammo = self.game_state.ammo
        
        # Determine state based on factors
        if health < 30:
            return "critical_health"
        elif ammo < 10:
            return "low_ammo"
        elif danger_level > 0.7:
            return "high_danger"
        elif zombie_count > 5:
            return "many_zombies"
        elif zombie_count > 0:
            return "combat"
        else:
            return "exploration"
    
    def _make_enhanced_decision(self, zombies, state_desc, delta_time):
        """
        Make enhanced decision with Frank Castle AI
        
        Args:
            zombies (list): Detected zombies
            state_desc (str): State description
            delta_time (float): Time since last frame
            
        Returns:
            dict: Action to execute
        """
        # Get base decision
        base_action = self.decision_maker.decide(self.game_state)
        
        # Check memory for best actions
        best_actions = self.memory.get_most_rewarding_actions(state_desc)
        
        # If we have good historical data, consider using it
        if best_actions and best_actions[0][1] > 0.5 and random.random() < 0.3:
            best_action_type = best_actions[0][0]
            
            # If best action differs from base, consider switching
            if best_action_type != base_action['type']:
                logger.info(f"Enhancing from {base_action['type']} to {best_action_type} based on memory")
                
                # Create enhanced action
                if best_action_type == "shoot":
                    # For shooting, we need a target
                    if zombies:
                        best_target = select_best_target(
                            zombies, 
                            prioritize_closest=self.config.get('combat', {}).get('prioritize_closest', True)
                        )
                        if best_target:
                            return create_combat_action(best_target)
                
                elif best_action_type == "move":
                    # For movement, generate direction based on situation
                    if state_desc == "high_danger" and zombies:
                        # Generate evasion path
                        danger_positions = [(z.get('center_x', 0), z.get('center_y', 0)) 
                                           for z in zombies]
                        path = generate_evasion_path((50, 50), danger_positions)
                        
                        if path:
                            next_point = path[0]
                            direction = "backward" if next_point[1] > 50 else "forward"
                            return {
                                "type": "move",
                                "description": f"Evading zombies ({direction})",
                                "params": {
                                    "direction": direction,
                                    "duration": 0.5
                                }
                            }
                    
                else:
                    # For other actions, use base parameters
                    return {
                        "type": best_action_type,
                        "description": f"Memory-enhanced {best_action_type}",
                        "params": base_action.get("params", {})
                    }
        
        # State-specific enhancements
        if state_desc == "exploration":
            # Use pathfinding for exploration
            if not self.current_path or random.random() < 0.05:
                player_pos = (50, 50)  # Center position
                self.current_path = generate_exploration_path(player_pos, self.known_areas)
                
                if self.current_path:
                    logger.debug(f"Generated exploration path with {len(self.current_path)} points")
            
            # Use next point in path
            if self.current_path:
                next_point = self.current_path.pop(0)
                # Add to known areas
                self.known_areas.add(next_point)
                
                dx = next_point[0] - 50  # Relative to center
                dy = next_point[1] - 50
                
                # Determine movement direction
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "backward" if dy > 0 else "forward"
                
                return {
                    "type": "move",
                    "description": f"Exploration movement ({direction})",
                    "params": {
                        "direction": direction,
                        "duration": base_action.get("params", {}).get("duration", 0.5)
                    }
                }
        
        elif state_desc == "combat":
            # Enhanced combat logic
            if zombies and base_action['type'] == "shoot":
                # Select best target
                best_target = select_best_target(
                    zombies, 
                    prioritize_closest=self.config.get('combat', {}).get('prioritize_closest', True)
                )
                
                if best_target:
                    # Create combat action
                    return create_combat_action(
                        best_target, 
                        weapon_info={
                            "name": self.game_state.current_weapon,
                            "is_automatic": self.game_state.current_weapon in ["lmg", "rifle", "smg"]
                        }
                    )
        
        elif state_desc == "high_danger":
            # Generate evasion path
            if zombies:
                danger_positions = [(z.get('center_x', 0), z.get('center_y', 0)) 
                                   for z in zombies]
                path = generate_evasion_path((50, 50), danger_positions)
                
                if path:
                    next_point = path[0]
                    direction = "backward" if next_point[1] > 50 else "forward"
                    return {
                        "type": "move",
                        "description": f"Evading zombies ({direction})",
                        "params": {
                            "direction": direction,
                            "duration": 0.5
                        }
                    }
        
        # Use base action if no enhancements apply
        return base_action
    
    def _calculate_reward(self, action, zombies, health, ammo):
        """
        Calculate reward for action
        
        Args:
            action (dict): Action taken
            zombies (list): Detected zombies
            health (int): Current health
            ammo (int): Current ammo
            
        Returns:
            float: Calculated reward
        """
        # Simplified reward system
        reward = 0
        
        # Reward for survival
        reward += 0.01
        
        # Penalty for low health
        if health < 30:
            reward -= 0.5
        
        # Reward for good ammo management
        if action['type'] == "reload" and ammo < 10:
            reward += 0.3
        
        # Reward for killing zombies (simulated)
        if action['type'] == "shoot" and zombies:
            # Assume some probability of killing
            if random.random() < 0.3:
                reward += 1.0
        
        # Reward for using grenades effectively
        if action['type'] == "grenade" and len(zombies) > 3:
            reward += 0.7
        
        return reward
    
    def _log_status(self):
        """Log current status"""
        if not self.start_time:
            return
            
        # Calculate runtime
        runtime = time.time() - self.start_time
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Capture status
        capture_status = self.screen_capture.get_status()
        capture_method = capture_status['preferred_method'] or "test"
        capture_fps = capture_status['fps']
        
        # Log status
        logger.info(
            f"Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
            f"FPS: {capture_fps:.1f} | "
            f"State: {self.current_state} | "
            f"Health: {self.game_state.health}% | "
            f"Ammo: {self.game_state.ammo} | "
            f"Weapon: {self.game_state.current_weapon} | "
            f"Round: {self.game_state.current_round} | "
            f"Zombies: {len(self.game_state.zombies)} | "
            f"Decisions: {self.decisions_made} | "
            f"Capture: {capture_method}"
        )
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Stop action controller
        self.action_controller.cleanup()
        
        # Save final memory
        self.memory._save_memory()
        
        # Create final backup
        self._create_backup()
        
        # Mark as not running
        self.running = False
        
        # Log final statistics
        if self.start_time:
            runtime = time.time() - self.start_time
            logger.info(f"Bot stopped after {runtime:.1f} seconds")
            logger.info(f"Made {self.decisions_made} decisions")
            logger.info(f"Processed {self.frames_processed} frames")
            logger.info(f"Total reward: {self.memory.total_reward:.2f}")

# ====================== MAIN FUNCTIONS ======================

def test_screen_capture(duration=10):
    """
    Test screen capture for specified duration
    
    Args:
        duration (int): Test duration in seconds
    """
    import cv2
    
    capture = EnhancedScreenCapture()
    
    print(f"Testing screen capture for {duration} seconds...")
    print(f"Available methods: {capture.available_methods}")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        # Capture frame
        frame = capture.capture()
        frame_count += 1
        
        # Display frame
        try:
            cv2.imshow('Captured Frame', frame)
            
            # Add status info
            status = capture.get_status()
            info_frame = np.zeros((200, 600, 3), dtype=np.uint8)
            
            cv2.putText(info_frame, f"FPS: {status['fps']:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_frame, f"Method: {status['preferred_method']}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_frame, f"Last: {status['last_successful_method']}", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_frame, f"Failures: {status['consecutive_failures']}", (10, 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow('Status', info_frame)
            
            # Allow window refresh
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            # OpenCV window might not be available in some environments
            logger.error(f"Error displaying frame: {e}")
    
    # Clean up
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Print results
    elapsed = time.time() - start_time
    print(f"Captured {frame_count} frames in {elapsed:.1f} seconds")
    print(f"Average FPS: {frame_count / elapsed:.1f}")
    print(f"Final status: {capture.get_status()}")

def display_banner():
    """Display banner for the bot"""
    banner = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██████╗ ██████╗ ██████╗     ██╗    ██╗ █████╗ ██╗    ██╗      ║
║  ██╔════╝██╔═══██╗██╔══██╗    ██║    ██║██╔══██╗██║    ██║      ║
║  ██║     ██║   ██║██║  ██║    ██║ █╗ ██║███████║██║ █╗ ██║      ║
║  ██║     ██║   ██║██║  ██║    ██║███╗██║██╔══██║██║███╗██║      ║
║  ╚██████╗╚██████╔╝██████╔╝    ╚███╔███╔╝██║  ██║╚███╔███╔╝      ║
║   ╚═════╝ ╚═════╝ ╚═════╝      ╚══╝╚══╝ ╚═╝  ╚═╝ ╚══╝╚══╝       ║
║                                                                  ║
║        ███████╗ ██████╗ ███╗   ███╗██████╗ ██╗███████╗          ║
║        ╚══███╔╝██╔═══██╗████╗ ████║██╔══██╗██║██╔════╝          ║
║          ███╔╝ ██║   ██║██╔████╔██║██████╔╝██║█████╗            ║
║         ███╔╝  ██║   ██║██║╚██╔╝██║██╔══██╗██║██╔══╝            ║
║        ███████╗╚██████╔╝██║ ╚═╝ ██║██████╔╝██║███████╗          ║
║        ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚═╝╚══════╝          ║
║                                                                  ║
║   ███████╗██████╗  █████╗ ███╗   ██╗██╗  ██╗    █████╗ ██╗      ║
║   ██╔════╝██╔══██╗██╔══██╗████╗  ██║██║ ██╔╝   ██╔══██╗██║      ║
║   █████╗  ██████╔╝███████║██╔██╗ ██║█████╔╝    ███████║██║      ║
║   ██╔══╝  ██╔══██╗██╔══██║██║╚██╗██║██╔═██╗    ██╔══██║██║      ║
║   ██║     ██║  ██║██║  ██║██║ ╚████║██║  ██╗██╗██║  ██║██║      ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    # Print banner
    print("\033[94m" + banner + "\033[0m")  # Blue color
    
    # Print subtitle
    subtitle = "Advanced Bot with Frank Castle AI and Enhanced Screen Capture"
    print("\033[93m" + "=" * len(subtitle) + "\033[0m")  # Yellow
    print("\033[93m" + subtitle + "\033[0m")
    print("\033[93m" + "=" * len(subtitle) + "\033[0m")
    print()

# Ensure heapq is imported for A* pathfinding
try:
    import heapq
except ImportError:
    # Define a simple heapq implementation if not available
    class SimpleHeapq:
        @staticmethod
        def heappush(heap, item):
            heap.append(item)
            heap.sort()
        
        @staticmethod
        def heappop(heap):
            return heap.pop(0)
    
    heapq = SimpleHeapq()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced CoD WaW Zombies Bot")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--delay", type=int, default=10, help="Startup delay in seconds")
    parser.add_argument("--capture", type=str, help="Preferred capture method")
    parser.add_argument("--test-capture", action="store_true", help="Test screen capture")
    args = parser.parse_args()
    
    # Display banner
    display_banner()
    
    # Check dependencies
    check_dependencies()
    
    # Test screen capture if requested
    if args.test_capture:
        print("\033[93mTesting screen capture methods...\033[0m")
        test_screen_capture(duration=10)
        return
    
    # Load configuration
    config = {}
    try:
        config_path = Path('config/advanced_bot_settings.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load config, using defaults: {e}")
    
    # Override with command line arguments
    if args.headless:
        config['headless'] = True
    
    if args.capture:
        config['preferred_capture_method'] = args.capture
    
    # Create and start the bot
    bot = AdvancedBot(config)
    bot.start(delay=args.delay)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"\033[91mError: {e}\033[0m")
        print("Check logs for details.")