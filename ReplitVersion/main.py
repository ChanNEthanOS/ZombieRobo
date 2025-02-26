#!/usr/bin/env python3
"""
COD World at War Zombies Bot - Main Entry Point
This program creates an AI bot that can play Call of Duty: World at War Zombies.
It uses computer vision, object detection, and decision algorithms to automate gameplay.
"""

import argparse
import time
import logging
import os
import sys
from bot.screen_capture import ScreenCapture
from bot.detection import ZombieDetector, HUDDetector
from bot.navigation import Navigator
from bot.actions import ActionController
from bot.game_state import GameState
from bot.decision import DecisionMaker
from bot.debug import DebugInterface
from bot.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='COD WaW Zombies Bot')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--map', type=str, default='default', help='Map name to use')
    parser.add_argument('--config', type=str, default='config/game_settings.json', help='Path to config file')
    parser.add_argument('--delay', type=int, default=3, help='Startup delay in seconds')
    parser.add_argument('--test-detection', action='store_true', help='Test detection only')
    parser.add_argument('--collect-data', action='store_true', help='Collect data for training')
    return parser.parse_args()

def main():
    """Main function that initializes and runs the bot"""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config, args.map)
    
    logger.info("Initializing COD WaW Zombies Bot...")
    logger.info(f"Using map: {args.map}")
    
    # Give user time to switch to game window
    print(f"Starting bot in {args.delay} seconds... Switch to the game window.")
    time.sleep(args.delay)
    
    # Initialize components
    screen_capture = ScreenCapture(config['game_region'])
    zombie_detector = ZombieDetector(config)
    hud_detector = HUDDetector(config)
    game_state = GameState()
    navigator = Navigator(config)
    action_controller = ActionController(config)
    decision_maker = DecisionMaker(config)
    
    # Initialize debug interface if needed
    debug_interface = None
    if args.debug:
        debug_interface = DebugInterface()
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Test detection only if requested
    if args.test_detection:
        logger.info("Running in detection test mode")
        while True:
            frame = screen_capture.capture()
            zombies = zombie_detector.detect(frame)
            health, ammo = hud_detector.detect_stats(frame)
            logger.info(f"Detected {len(zombies)} zombies, Health: {health}, Ammo: {ammo}")
            if debug_interface:
                debug_interface.display(frame, zombies, health, ammo)
            time.sleep(0.1)
    
    # Data collection mode
    if args.collect_data:
        logger.info("Running in data collection mode")
        # This will be implemented in a future update
        pass
    
    # Main bot loop
    try:
        logger.info("Bot started, running main loop")
        while True:
            # Capture the screen
            frame = screen_capture.capture()
            
            # Detect game elements
            zombies = zombie_detector.detect(frame)
            health, ammo, current_weapon, current_round = hud_detector.detect_stats(frame)
            
            # Update game state
            game_state.update(zombies, health, ammo, current_weapon, current_round)
            
            # Make decisions
            action = decision_maker.decide(game_state)
            
            # Execute actions
            action_controller.execute(action)
            
            # Debug visualization
            if debug_interface:
                debug_interface.display(frame, zombies, health, ammo, game_state, action)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception("An error occurred during bot execution")
    finally:
        logger.info("Shutting down bot...")
        if debug_interface:
            debug_interface.close()

if __name__ == "__main__":
    main()