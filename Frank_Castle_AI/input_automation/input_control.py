#!/usr/bin/env python3
"""
Input Controller for COD WaW Zombies Bot
This module provides precise control over keyboard and mouse inputs
with various methods to avoid detection and ensure reliable operation.
"""

import time
import random
import logging
import platform
import threading
from typing import Tuple, List, Dict, Union, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InputController")

# Determine which input modules to use based on platform
SYSTEM = platform.system().lower()

try:
    if SYSTEM == 'windows':
        import win32api
        import win32con
        import ctypes
        import pydirectinput  # More reliable than pyautogui on Windows
        USE_WIN32 = True
    else:
        import pyautogui
        USE_WIN32 = False
except ImportError as e:
    logger.error(f"Failed to import input modules: {e}")
    logger.warning("Will use limited fallback mode")
    USE_WIN32 = False

# Default key mappings (can be overridden)
DEFAULT_KEYBINDINGS = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "reload": "r",
    "weapon1": "1",
    "weapon2": "2",
    "melee": "v",
    "sprint": "shift",
    "jump": "space",
    "crouch": "ctrl",
    "use": "f",
    "grenade": "g",
    "shoot": "mouseLeft",
    "aim": "mouseRight"
}

class InputController:
    """
    Controls keyboard and mouse inputs for the game, providing
    human-like inputs with variable timing and movements.
    """
    
    def __init__(self, keybindings=None, mouse_sensitivity=1.0, movement_interval=0.05):
        """
        Initialize the input controller
        
        Args:
            keybindings (dict): Custom key mappings (optional)
            mouse_sensitivity (float): Sensitivity multiplier (0.1-2.0)
            movement_interval (float): Time between movement updates
        """
        self.keybindings = keybindings or DEFAULT_KEYBINDINGS
        self.mouse_sensitivity = max(0.1, min(2.0, mouse_sensitivity))
        self.movement_interval = movement_interval
        
        # State tracking
        self.pressed_keys = set()
        self.mouse_position = (0, 0)
        
        # For continuous actions
        self.continuous_actions = {}
        self.action_threads = {}
        self.running = True
        
        # Get screen dimensions
        if not USE_WIN32:
            try:
                self.screen_width, self.screen_height = pyautogui.size()
            except:
                self.screen_width, self.screen_height = 1920, 1080
        else:
            try:
                self.screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
                self.screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            except:
                self.screen_width, self.screen_height = 1920, 1080
        
        logger.info(f"Input controller initialized with {SYSTEM} driver")
        logger.info(f"Screen dimensions: {self.screen_width}x{self.screen_height}")
    
    def press_key(self, key: str, duration: Optional[float] = None, human_like: bool = True) -> bool:
        """
        Press a key and optionally hold it for a duration
        
        Args:
            key (str): Key to press
            duration (float, optional): Duration to hold the key in seconds
            human_like (bool): Add human-like randomness to timing
            
        Returns:
            bool: True if successful
        """
        try:
            # Get the actual key based on binding
            mapped_key = self.keybindings.get(key, key)
            
            # Handle special keys
            if mapped_key == 'shift':
                mapped_key = 'shiftleft' if not USE_WIN32 else 'shift'
            elif mapped_key == 'ctrl':
                mapped_key = 'ctrlleft' if not USE_WIN32 else 'ctrl'
            elif mapped_key == 'space':
                mapped_key = 'space'
                
            # Track pressed key
            self.pressed_keys.add(mapped_key)
                
            # Press the key
            if USE_WIN32:
                pydirectinput.keyDown(mapped_key)
            else:
                pyautogui.keyDown(mapped_key)
                
            logger.debug(f"Pressed key: {mapped_key}")
            
            # Hold for duration if specified
            if duration is not None:
                # Add human-like randomness to timing
                if human_like:
                    hold_time = duration * random.uniform(0.9, 1.1)
                else:
                    hold_time = duration
                    
                time.sleep(hold_time)
                self.release_key(key)
            
            return True
        except Exception as e:
            logger.error(f"Error pressing key {key}: {e}")
            return False
    
    def release_key(self, key: str) -> bool:
        """
        Release a pressed key
        
        Args:
            key (str): Key to release
            
        Returns:
            bool: True if successful
        """
        try:
            # Get the actual key based on binding
            mapped_key = self.keybindings.get(key, key)
            
            # Handle special keys
            if mapped_key == 'shift':
                mapped_key = 'shiftleft' if not USE_WIN32 else 'shift'
            elif mapped_key == 'ctrl':
                mapped_key = 'ctrlleft' if not USE_WIN32 else 'ctrl'
            elif mapped_key == 'space':
                mapped_key = 'space'
            
            # Release the key
            if USE_WIN32:
                pydirectinput.keyUp(mapped_key)
            else:
                pyautogui.keyUp(mapped_key)
                
            # Remove from pressed keys
            self.pressed_keys.discard(mapped_key)
                
            logger.debug(f"Released key: {mapped_key}")
            return True
        except Exception as e:
            logger.error(f"Error releasing key {key}: {e}")
            return False
    
    def click(self, button: str = 'left', duration: Optional[float] = None) -> bool:
        """
        Perform mouse click
        
        Args:
            button (str): Mouse button ('left', 'right', 'middle')
            duration (float, optional): How long to hold the click
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert to pyautogui/pydirectinput format
            button_map = {
                'left': 'left', 
                'right': 'right', 
                'middle': 'middle',
                'mouseLeft': 'left',
                'mouseRight': 'right'
            }
            btn = button_map.get(button, button)
            
            if duration is None:
                # Regular click
                if USE_WIN32:
                    pydirectinput.click(button=btn)
                else:
                    pyautogui.click(button=btn)
                logger.debug(f"Clicked {btn} mouse button")
            else:
                # Hold and release
                if USE_WIN32:
                    pydirectinput.mouseDown(button=btn)
                    time.sleep(duration)
                    pydirectinput.mouseUp(button=btn)
                else:
                    pyautogui.mouseDown(button=btn)
                    time.sleep(duration)
                    pyautogui.mouseUp(button=btn)
                logger.debug(f"Held {btn} mouse button for {duration}s")
            
            return True
        except Exception as e:
            logger.error(f"Error clicking mouse button {button}: {e}")
            return False
    
    def move_mouse(self, x: int, y: int, relative: bool = False, 
                   speed: Optional[float] = None, smooth: bool = True) -> bool:
        """
        Move mouse to position
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            relative (bool): If True, treats coordinates as relative movement
            speed (float, optional): Movement speed (0.1-10, lower is faster)
            smooth (bool): Whether to use smooth movement
            
        Returns:
            bool: True if successful
        """
        try:
            # Get current position
            if USE_WIN32:
                curr_x, curr_y = win32api.GetCursorPos()
            else:
                curr_x, curr_y = pyautogui.position()
            
            # Calculate target position
            if relative:
                target_x = curr_x + x
                target_y = curr_y + y
            else:
                target_x = x
                target_y = y
            
            # Ensure within screen bounds
            target_x = max(0, min(self.screen_width - 1, target_x))
            target_y = max(0, min(self.screen_height - 1, target_y))
            
            # Store new position
            self.mouse_position = (target_x, target_y)
            
            # Move the mouse
            if smooth and not USE_WIN32:
                # Use pyautogui smooth move
                duration = speed if speed is not None else 0.1
                pyautogui.moveTo(target_x, target_y, duration=duration)
                logger.debug(f"Smooth moved mouse to ({target_x}, {target_y})")
            elif smooth and USE_WIN32:
                # Manual smooth move for Windows
                steps = 20
                sleep_time = (speed if speed is not None else 0.1) / steps
                
                for i in range(1, steps + 1):
                    step_x = curr_x + (target_x - curr_x) * i / steps
                    step_y = curr_y + (target_y - curr_y) * i / steps
                    
                    # Use direct Windows API for smoother movement
                    ctypes.windll.user32.SetCursorPos(int(step_x), int(step_y))
                    time.sleep(sleep_time)
                
                logger.debug(f"Smooth moved mouse to ({target_x}, {target_y})")
            else:
                # Instant move
                if USE_WIN32:
                    ctypes.windll.user32.SetCursorPos(int(target_x), int(target_y))
                else:
                    pyautogui.moveTo(target_x, target_y)
                logger.debug(f"Moved mouse to ({target_x}, {target_y})")
            
            return True
        except Exception as e:
            logger.error(f"Error moving mouse: {e}")
            return False
    
    def scroll(self, clicks: int, direction: str = 'down') -> bool:
        """
        Scroll the mouse wheel
        
        Args:
            clicks (int): Number of scroll clicks
            direction (str): 'up' or 'down'
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert direction to value
            amount = clicks if direction == 'down' else -clicks
            
            if USE_WIN32:
                for _ in range(abs(clicks)):
                    pydirectinput.scroll(-1 if direction == 'down' else 1)
            else:
                pyautogui.scroll(amount)
                
            logger.debug(f"Scrolled {clicks} clicks {direction}")
            return True
        except Exception as e:
            logger.error(f"Error scrolling mouse: {e}")
            return False
    
    def move_to_target(self, target_x: int, target_y: int, max_speed: Optional[float] = None,
                      accuracy: float = 0.9, overshoot: bool = True) -> bool:
        """
        Move mouse to a target with human-like motion
        
        Args:
            target_x (int): Target X coordinate
            target_y (int): Target Y coordinate
            max_speed (float, optional): Maximum movement speed
            accuracy (float): Accuracy factor (0.5-1.0)
            overshoot (bool): Whether to simulate human overshoot
            
        Returns:
            bool: True if successful
        """
        try:
            if USE_WIN32:
                curr_x, curr_y = win32api.GetCursorPos()
            else:
                curr_x, curr_y = pyautogui.position()
            
            # Calculate distance
            distance = ((target_x - curr_x) ** 2 + (target_y - curr_y) ** 2) ** 0.5
            
            # If very close, just move directly
            if distance < 10:
                return self.move_mouse(target_x, target_y)
            
            # Adjust for sensitivity
            sens_factor = self.mouse_sensitivity
            
            # Calculate speed based on distance
            if max_speed is None:
                # Faster for longer distances, but with a cap
                max_speed = min(0.5, 0.1 + distance / 2000)
            
            # Add some human-like randomness to the target
            accuracy = max(0.5, min(1.0, accuracy))
            if accuracy < 1.0:
                rand_offset = distance * (1 - accuracy) * 0.1
                target_x += random.uniform(-rand_offset, rand_offset)
                target_y += random.uniform(-rand_offset, rand_offset)
            
            # Decide whether to overshoot
            if overshoot and random.random() < 0.3:
                overshoot_x = target_x + (target_x - curr_x) * random.uniform(0.05, 0.2)
                overshoot_y = target_y + (target_y - curr_y) * random.uniform(0.05, 0.2)
                
                # Move with overshoot
                self.move_mouse(int(overshoot_x), int(overshoot_y), 
                               smooth=True, speed=max_speed * 0.8)
                time.sleep(random.uniform(0.01, 0.05))
                
                # Then correct to actual target
                return self.move_mouse(target_x, target_y, smooth=True, 
                                      speed=max_speed * 1.5)
            else:
                # Direct move with human-like smoothing
                return self.move_mouse(target_x, target_y, smooth=True, 
                                      speed=max_speed)
        except Exception as e:
            logger.error(f"Error moving to target: {e}")
            return False
    
    def aim_at_target(self, target_x: int, target_y: int, shoot: bool = False, 
                     shoot_duration: Optional[float] = None, pause_before_shoot: float = 0.1) -> bool:
        """
        Aim at a target and optionally shoot
        
        Args:
            target_x (int): Target X coordinate
            target_y (int): Target Y coordinate
            shoot (bool): Whether to shoot after aiming
            shoot_duration (float, optional): How long to shoot
            pause_before_shoot (float): Pause time between aim and shoot
            
        Returns:
            bool: True if successful
        """
        try:
            # Aim at target using human-like movement
            self.move_to_target(target_x, target_y)
            
            # Optional pause before shooting (human reaction time)
            if shoot and pause_before_shoot > 0:
                time.sleep(pause_before_shoot * random.uniform(0.8, 1.2))
            
            # Shoot if requested
            if shoot:
                if shoot_duration is None:
                    # Quick click
                    self.click('left')
                else:
                    # Hold for duration
                    self.click('left', duration=shoot_duration)
            
            return True
        except Exception as e:
            logger.error(f"Error aiming at target: {e}")
            return False
    
    def press_key_sequence(self, keys: List[str], delays: Optional[List[float]] = None) -> bool:
        """
        Press a sequence of keys with specified delays
        
        Args:
            keys (list): List of keys to press
            delays (list, optional): Delays between key presses
            
        Returns:
            bool: True if successful
        """
        try:
            if delays is None:
                # Default to fixed delay if none provided
                delays = [0.1] * (len(keys) - 1)
            
            # Ensure delays list is correct length
            if len(delays) < len(keys) - 1:
                delays.extend([0.1] * (len(keys) - 1 - len(delays)))
            
            # Press the keys with delays
            for i, key in enumerate(keys):
                self.press_key(key)
                
                # Release immediately if not the last key
                if i < len(keys) - 1:
                    self.release_key(key)
                    time.sleep(delays[i])
            
            # Release the last key
            self.release_key(keys[-1])
            
            return True
        except Exception as e:
            logger.error(f"Error pressing key sequence: {e}")
            return False
    
    def start_continuous_action(self, action_type: str, **kwargs) -> str:
        """
        Start a continuous action in a separate thread
        
        Args:
            action_type (str): Type of action ('move', 'press', etc.)
            **kwargs: Action-specific parameters
            
        Returns:
            str: Action ID
        """
        try:
            # Generate a unique ID for this action
            action_id = f"{action_type}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            # Create action config
            action_config = {
                'type': action_type,
                'params': kwargs,
                'active': True
            }
            
            # Store it
            self.continuous_actions[action_id] = action_config
            
            # Start thread
            thread = threading.Thread(
                target=self._continuous_action_worker,
                args=(action_id,),
                daemon=True
            )
            self.action_threads[action_id] = thread
            thread.start()
            
            logger.debug(f"Started continuous action: {action_type} (ID: {action_id})")
            return action_id
        except Exception as e:
            logger.error(f"Error starting continuous action: {e}")
            return ""
    
    def stop_continuous_action(self, action_id: str) -> bool:
        """
        Stop a continuous action
        
        Args:
            action_id (str): ID of the action to stop
            
        Returns:
            bool: True if successfully stopped
        """
        if action_id in self.continuous_actions:
            # Mark as inactive
            self.continuous_actions[action_id]['active'] = False
            
            # Wait for thread to finish
            if action_id in self.action_threads:
                self.action_threads[action_id].join(timeout=1.0)
                del self.action_threads[action_id]
            
            # Clean up
            action_type = self.continuous_actions[action_id]['type']
            params = self.continuous_actions[action_id]['params']
            
            # Release keys if it was a key press action
            if action_type == 'press' and 'key' in params:
                self.release_key(params['key'])
            
            del self.continuous_actions[action_id]
            
            logger.debug(f"Stopped continuous action: {action_id}")
            return True
        else:
            logger.warning(f"Action ID not found: {action_id}")
            return False
    
    def stop_all_actions(self) -> None:
        """Stop all continuous actions and release all keys"""
        # Stop all continuous actions
        for action_id in list(self.continuous_actions.keys()):
            self.stop_continuous_action(action_id)
        
        # Release all pressed keys
        for key in list(self.pressed_keys):
            if USE_WIN32:
                pydirectinput.keyUp(key)
            else:
                pyautogui.keyUp(key)
        
        self.pressed_keys.clear()
        logger.info("Stopped all actions and released all keys")
    
    def _continuous_action_worker(self, action_id: str) -> None:
        """
        Worker thread for continuous actions
        
        Args:
            action_id (str): ID of the action to perform
        """
        try:
            if action_id not in self.continuous_actions:
                return
                
            action = self.continuous_actions[action_id]
            action_type = action['type']
            params = action['params']
            
            if action_type == 'press':
                # Continuous key press
                key = params.get('key', '')
                if key:
                    self.press_key(key)
                    
                    # Keep thread alive until action is stopped
                    while (action_id in self.continuous_actions and 
                           self.continuous_actions[action_id]['active']):
                        time.sleep(0.1)
                    
                    # Release key when done
                    self.release_key(key)
            
            elif action_type == 'move':
                # Continuous mouse movement
                while (action_id in self.continuous_actions and 
                       self.continuous_actions[action_id]['active']):
                    # Extract movement parameters
                    dx = params.get('dx', 0)
                    dy = params.get('dy', 0)
                    
                    # Apply movement
                    if dx != 0 or dy != 0:
                        self.move_mouse(dx, dy, relative=True)
                    
                    # Wait for next update
                    time.sleep(self.movement_interval)
            
            elif action_type == 'walk_forward':
                # Walking forward with mouse adjustments
                # Start forward movement
                forward_key = self.keybindings.get('forward', 'w')
                self.press_key(forward_key)
                
                while (action_id in self.continuous_actions and 
                       self.continuous_actions[action_id]['active']):
                    # Extract walking parameters
                    turn_amount = params.get('turn_amount', 0)
                    
                    # Apply turning if needed
                    if turn_amount != 0:
                        self.move_mouse(turn_amount, 0, relative=True)
                    
                    # Wait for next update
                    time.sleep(self.movement_interval)
                
                # Stop walking
                self.release_key(forward_key)
        
        except Exception as e:
            logger.error(f"Error in continuous action worker: {e}")
            
            # Ensure cleanup
            if action_id in self.continuous_actions:
                action_type = self.continuous_actions[action_id]['type']
                params = self.continuous_actions[action_id]['params']
                
                # Release keys if it was a key press action
                if action_type == 'press' and 'key' in params:
                    self.release_key(params['key'])
        
        finally:
            # Remove action if it still exists
            if action_id in self.continuous_actions:
                del self.continuous_actions[action_id]
    
    def type_text(self, text: str, interval: Optional[float] = None) -> bool:
        """
        Type text with human-like timing
        
        Args:
            text (str): Text to type
            interval (float, optional): Time between keystrokes
            
        Returns:
            bool: True if successful
        """
        try:
            # Default to human-like variable intervals
            if interval is None:
                # Use pyautogui's write which has built-in randomization
                if USE_WIN32:
                    # Manual implementation for pydirectinput
                    for char in text:
                        pydirectinput.press(char)
                        time.sleep(random.uniform(0.05, 0.15))
                else:
                    pyautogui.write(text, interval=random.uniform(0.05, 0.15))
            else:
                # Use fixed interval
                if USE_WIN32:
                    for char in text:
                        pydirectinput.press(char)
                        time.sleep(interval)
                else:
                    pyautogui.write(text, interval=interval)
            
            logger.debug(f"Typed text: {text}")
            return True
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return False
    
    def perform_action(self, action: Dict) -> bool:
        """
        Perform a game action based on action dictionary
        
        Args:
            action (dict): Action specification
            
        Returns:
            bool: True if successful
        """
        try:
            action_type = action.get('type', '').lower()
            
            # Movement actions
            if action_type == 'move':
                direction = action.get('direction', '')
                duration = action.get('duration', 0.5)
                
                key = None
                if direction == 'forward':
                    key = 'forward'
                elif direction == 'backward':
                    key = 'backward'
                elif direction == 'left':
                    key = 'left'
                elif direction == 'right':
                    key = 'right'
                
                if key:
                    return self.press_key(key, duration)
            
            # Shooting action
            elif action_type == 'shoot':
                duration = action.get('duration', 0.2)
                target = action.get('target', None)
                
                if target:
                    # Shoot at specific target
                    x, y = target.get('x', 0), target.get('y', 0)
                    return self.aim_at_target(x, y, shoot=True, shoot_duration=duration)
                else:
                    # Just shoot where we're pointing
                    return self.click('left', duration)
            
            # Reload action
            elif action_type == 'reload':
                return self.press_key('reload')
            
            # Weapon switch
            elif action_type == 'weapon_switch':
                weapon = action.get('weapon', 1)
                key = f'weapon{weapon}'
                return self.press_key(key)
            
            # Look action
            elif action_type == 'look':
                x = action.get('x', 0)
                y = action.get('y', 0)
                relative = action.get('relative', True)
                
                if relative:
                    return self.move_mouse(x, y, relative=True)
                else:
                    return self.move_mouse(x, y)
            
            # Jump action
            elif action_type == 'jump':
                return self.press_key('jump')
            
            # Use/interact action
            elif action_type == 'use':
                return self.press_key('use')
            
            # Sequence of actions
            elif action_type == 'sequence':
                actions = action.get('actions', [])
                for sub_action in actions:
                    success = self.perform_action(sub_action)
                    if not success:
                        return False
                    
                    # Add delay between actions
                    time.sleep(action.get('delay', 0.1))
                
                return True
            
            # Continuous action
            elif action_type == 'continuous':
                sub_type = action.get('subtype', '')
                params = action.get('params', {})
                
                if action.get('start', True):
                    # Start continuous action
                    action_id = self.start_continuous_action(sub_type, **params)
                    action['id'] = action_id
                    return bool(action_id)
                else:
                    # Stop continuous action
                    action_id = action.get('id', '')
                    if action_id:
                        return self.stop_continuous_action(action_id)
                    else:
                        logger.error("No action ID provided for stopping continuous action")
                        return False
            
            # Unknown action
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error performing action: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources and stop all actions"""
        self.running = False
        self.stop_all_actions()
        logger.info("Input controller cleaned up")


def test_input_controller():
    """Run a simple test of the input controller"""
    print("Testing input controller...")
    controller = InputController()
    
    # Wait for user to switch windows
    print("You have 3 seconds to switch to your target window...")
    time.sleep(3)
    
    print("Testing keyboard input...")
    # Test basic movement keys
    controller.press_key('forward', 0.5)
    time.sleep(0.2)
    controller.press_key('backward', 0.5)
    time.sleep(0.2)
    controller.press_key('left', 0.5)
    time.sleep(0.2)
    controller.press_key('right', 0.5)
    time.sleep(0.5)
    
    print("Testing mouse movement...")
    # Test mouse movement
    current_pos = controller.mouse_position
    controller.move_mouse(current_pos[0] + 100, current_pos[1], relative=True, smooth=True)
    time.sleep(0.5)
    controller.move_mouse(current_pos[0], current_pos[1] + 100, relative=True, smooth=True)
    time.sleep(0.5)
    controller.move_mouse(current_pos[0], current_pos[1], smooth=True)
    
    print("Testing human-like aiming...")
    # Test aiming
    screen_center_x = controller.screen_width // 2
    screen_center_y = controller.screen_height // 2
    controller.aim_at_target(screen_center_x + 200, screen_center_y - 150)
    time.sleep(0.5)
    controller.aim_at_target(screen_center_x - 200, screen_center_y + 100)
    time.sleep(0.5)
    
    print("Testing continuous actions...")
    # Test continuous action
    walk_id = controller.start_continuous_action('press', key='forward')
    time.sleep(2)
    controller.stop_continuous_action(walk_id)
    
    print("Testing combined actions...")
    # Test complex action
    action = {
        'type': 'sequence',
        'actions': [
            {'type': 'move', 'direction': 'forward', 'duration': 1.0},
            {'type': 'jump'},
            {'type': 'shoot', 'duration': 0.3}
        ],
        'delay': 0.2
    }
    controller.perform_action(action)
    
    print("Tests completed!")
    controller.cleanup()

if __name__ == "__main__":
    test_input_controller()