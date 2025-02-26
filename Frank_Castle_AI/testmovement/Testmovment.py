#!/usr/bin/env python3
"""
Test script for movement controls in COD WaW Zombies Bot.
This script allows testing basic movement and input handling.
"""

import time
import argparse
import logging
import cv2
import numpy as np
import os
import random
from input_controller import InputController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MovementTest")

# Movement patterns
MOVEMENT_PATTERNS = {
    "circle": [
        {'type': 'move', 'direction': 'forward', 'duration': 1.0},
        {'type': 'look', 'x': 100, 'y': 0, 'relative': True},
        {'type': 'move', 'direction': 'forward', 'duration': 1.0},
        {'type': 'look', 'x': 100, 'y': 0, 'relative': True},
        {'type': 'move', 'direction': 'forward', 'duration': 1.0},
        {'type': 'look', 'x': 100, 'y': 0, 'relative': True},
        {'type': 'move', 'direction': 'forward', 'duration': 1.0},
    ],
    "zigzag": [
        {'type': 'move', 'direction': 'forward', 'duration': 1.0},
        {'type': 'look', 'x': 45, 'y': 0, 'relative': True},
        {'type': 'move', 'direction': 'forward', 'duration': 1.0},
        {'type': 'look', 'x': -90, 'y': 0, 'relative': True},
        {'type': 'move', 'direction': 'forward', 'duration': 1.0},
        {'type': 'look', 'x': 45, 'y': 0, 'relative': True},
    ],
    "sprint": [
        {'type': 'sequence', 'actions': [
            {'type': 'press', 'key': 'sprint'},
            {'type': 'press', 'key': 'forward', 'duration': 2.0}
        ]},
        {'type': 'look', 'x': 0, 'y': -45, 'relative': True},
        {'type': 'jump'},
        {'type': 'sequence', 'actions': [
            {'type': 'press', 'key': 'sprint'},
            {'type': 'press', 'key': 'forward', 'duration': 2.0}
        ]},
    ],
    "train": [  # Movement pattern for "training" zombies
        {'type': 'continuous', 'subtype': 'walk_forward', 'params': {'turn_amount': 5}, 'start': True},
        # This will be stopped later in the code
    ],
    "defense": [
        {'type': 'shoot', 'duration': 0.5},
        {'type': 'move', 'direction': 'backward', 'duration': 0.5},
        {'type': 'reload'},
        {'type': 'sequence', 'actions': [
            {'type': 'shoot', 'duration': 0.3},
            {'type': 'look', 'x': 30, 'y': 0, 'relative': True},
            {'type': 'shoot', 'duration': 0.3},
            {'type': 'look', 'x': -60, 'y': 0, 'relative': True},
            {'type': 'shoot', 'duration': 0.3},
            {'type': 'look', 'x': 30, 'y': 0, 'relative': True},
        ]},
    ]
}

def capture_screen(region=None):
    """Capture the screen for testing"""
    try:
        import mss
        with mss.mss() as sct:
            # Default to full screen if no region specified
            if region is None:
                region = {
                    'top': 0,
                    'left': 0,
                    'width': 1920,
                    'height': 1080
                }
            
            img = np.array(sct.grab(region))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except ImportError:
        logger.error("MSS not installed. Using blank image.")
        return np.zeros((600, 800, 3), dtype=np.uint8)
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        return np.zeros((600, 800, 3), dtype=np.uint8)

def test_basic_movement(controller, duration=10):
    """Test basic WASD movement"""
    logger.info("Testing basic WASD movement...")
    
    moves = ['forward', 'left', 'backward', 'right']
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Choose random direction
        direction = random.choice(moves)
        move_time = random.uniform(0.3, 1.0)
        
        logger.info(f"Moving {direction} for {move_time:.1f} seconds")
        controller.press_key(direction, move_time)
        
        # Add some mouse movement
        x_delta = random.randint(-100, 100)
        controller.move_mouse(x_delta, 0, relative=True, smooth=True)
        
        # Occasional jump
        if random.random() < 0.3:
            logger.info("Jumping")
            controller.press_key('jump')
        
        time.sleep(0.2)
    
    logger.info("Basic movement test completed")

def test_pattern_movement(controller, pattern_name, duration=15):
    """Test specific movement pattern"""
    if pattern_name not in MOVEMENT_PATTERNS:
        logger.error(f"Unknown pattern: {pattern_name}")
        return
    
    logger.info(f"Testing {pattern_name} movement pattern...")
    pattern = MOVEMENT_PATTERNS[pattern_name]
    
    # For continuous patterns
    continuous_action_id = None
    
    start_time = time.time()
    pattern_index = 0
    
    try:
        while time.time() - start_time < duration:
            if pattern_index >= len(pattern):
                pattern_index = 0
            
            # Get next action in pattern
            action = pattern[pattern_index]
            
            # Handle continuous actions specially
            if action['type'] == 'continuous' and action.get('start', True):
                if continuous_action_id:
                    controller.stop_continuous_action(continuous_action_id)
                
                # Start new continuous action
                continuous_action_id = controller.start_continuous_action(
                    action['subtype'], **action['params'])
                
                # For continuous actions, just cycle through the whole duration
                time.sleep(2.0)
            else:
                # Execute normal action
                logger.info(f"Executing: {action['type']}")
                controller.perform_action(action)
                
                # Wait briefly between actions
                time.sleep(0.3)
            
            pattern_index += 1
    
    finally:
        # Make sure to stop any continuous actions
        if continuous_action_id:
            controller.stop_continuous_action(continuous_action_id)
    
    logger.info(f"{pattern_name} pattern test completed")

def test_human_like_combat(controller, duration=20):
    """Test human-like combat behavior"""
    logger.info("Testing human-like combat behavior...")
    
    # Simulated targets (as if detected by the bot)
    screen_w, screen_h = controller.screen_width, controller.screen_height
    
    def generate_random_target():
        """Generate a random target position"""
        center_x, center_y = screen_w // 2, screen_h // 2
        target_x = center_x + random.randint(-300, 300)
        target_y = center_y + random.randint(-200, 200)
        return target_x, target_y
    
    # Main test loop
    start_time = time.time()
    while time.time() - start_time < duration:
        # Aim at random target
        target_x, target_y = generate_random_target()
        logger.info(f"Aiming at target: ({target_x}, {target_y})")
        
        # First, move close to target (simulates initial sighting)
        controller.move_to_target(
            target_x + random.randint(-50, 50),
            target_y + random.randint(-30, 30),
            max_speed=random.uniform(0.3, 0.6)
        )
        
        # Then fine-tune aim with higher accuracy
        time.sleep(random.uniform(0.05, 0.15))  # Human reaction time
        controller.aim_at_target(
            target_x, target_y, 
            shoot=True,
            shoot_duration=random.uniform(0.1, 0.3),
            pause_before_shoot=random.uniform(0.05, 0.15)
        )
        
        # Move briefly after shooting (tactical reposition)
        strafe_dir = random.choice(['left', 'right'])
        controller.press_key(strafe_dir, random.uniform(0.2, 0.5))
        
        # Occasionally reload
        if random.random() < 0.3:
            logger.info("Reloading...")
            controller.press_key('reload')
            time.sleep(random.uniform(0.5, 1.5))  # Simulate reload time
        
        # Wait briefly before next target
        time.sleep(random.uniform(0.3, 0.8))
    
    logger.info("Human-like combat test completed")

def test_custom_sequence(controller, sequence):
    """Test a custom sequence of actions"""
    logger.info("Testing custom action sequence...")
    
    for i, action_str in enumerate(sequence):
        logger.info(f"Executing action {i+1}/{len(sequence)}: {action_str}")
        
        # Parse action string
        parts = action_str.split(':')
        action_type = parts[0].strip()
        
        if action_type == 'move':
            # Format: move:direction:duration
            if len(parts) >= 3:
                direction = parts[1].strip()
                try:
                    duration = float(parts[2].strip())
                except ValueError:
                    duration = 1.0
                
                action = {'type': 'move', 'direction': direction, 'duration': duration}
                controller.perform_action(action)
            else:
                logger.error(f"Invalid move format: {action_str}")
        
        elif action_type == 'look':
            # Format: look:x:y
            if len(parts) >= 3:
                try:
                    x = int(parts[1].strip())
                    y = int(parts[2].strip())
                except ValueError:
                    logger.error(f"Invalid look coordinates: {action_str}")
                    continue
                
                action = {'type': 'look', 'x': x, 'y': y, 'relative': True}
                controller.perform_action(action)
            else:
                logger.error(f"Invalid look format: {action_str}")
        
        elif action_type == 'click' or action_type == 'shoot':
            # Format: click:button:duration or shoot:duration
            if len(parts) >= 2:
                if action_type == 'click':
                    button = parts[1].strip()
                    duration = float(parts[2].strip()) if len(parts) >= 3 else None
                    controller.click(button, duration)
                else:
                    try:
                        duration = float(parts[1].strip())
                    except ValueError:
                        duration = 0.2
                    
                    action = {'type': 'shoot', 'duration': duration}
                    controller.perform_action(action)
            else:
                controller.click('left')
        
        elif action_type == 'key' or action_type == 'press':
            # Format: key:keyname:duration
            if len(parts) >= 2:
                key = parts[1].strip()
                duration = float(parts[2].strip()) if len(parts) >= 3 else None
                controller.press_key(key, duration)
            else:
                logger.error(f"Invalid key format: {action_str}")
        
        elif action_type == 'wait' or action_type == 'sleep':
            # Format: wait:seconds
            if len(parts) >= 2:
                try:
                    seconds = float(parts[1].strip())
                    time.sleep(seconds)
                except ValueError:
                    logger.error(f"Invalid wait time: {action_str}")
            else:
                time.sleep(1.0)
        
        elif action_type == 'sequence':
            # Format: sequence:name
            if len(parts) >= 2:
                pattern_name = parts[1].strip()
                if pattern_name in MOVEMENT_PATTERNS:
                    for seq_action in MOVEMENT_PATTERNS[pattern_name]:
                        controller.perform_action(seq_action)
                        time.sleep(0.3)
                else:
                    logger.error(f"Unknown sequence: {pattern_name}")
            else:
                logger.error(f"Invalid sequence format: {action_str}")
        
        else:
            logger.error(f"Unknown action type: {action_type}")
        
        # Wait briefly between actions
        time.sleep(0.2)
    
    logger.info("Custom sequence test completed")

def main():
    parser = argparse.ArgumentParser(description="Test movement controls for COD WaW Zombies Bot")
    parser.add_argument('--test', choices=['basic', 'pattern', 'combat', 'custom'], 
                        default='basic', help='Test type to run')
    parser.add_argument('--pattern', choices=list(MOVEMENT_PATTERNS.keys()),
                        default='circle', help='Movement pattern to test')
    parser.add_argument('--duration', type=int, default=15,
                        help='Test duration in seconds')
    parser.add_argument('--delay', type=int, default=3,
                        help='Startup delay in seconds')
    parser.add_argument('--custom', type=str, nargs='+',
                        help='Custom action sequence (e.g., "move:forward:1.0", "look:100:0")')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Mouse sensitivity multiplier')
    args = parser.parse_args()
    
    # Create the input controller
    controller = InputController(mouse_sensitivity=args.sensitivity)
    
    # Allow time for the user to switch to the game window
    print(f"Starting in {args.delay} seconds... Switch to your game window!")
    time.sleep(args.delay)
    
    try:
        # Run the specified test
        if args.test == 'basic':
            test_basic_movement(controller, args.duration)
        elif args.test == 'pattern':
            test_pattern_movement(controller, args.pattern, args.duration)
        elif args.test == 'combat':
            test_human_like_combat(controller, args.duration)
        elif args.test == 'custom' and args.custom:
            test_custom_sequence(controller, args.custom)
        else:
            logger.error("Please specify a valid test or custom sequence")
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.exception(f"Error during test: {e}")
    finally:
        # Clean up
        controller.cleanup()
        logger.info("Test completed, all actions stopped")

if __name__ == "__main__":
    main()