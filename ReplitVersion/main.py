"""
main.py
Entry point for the COD WaW Zombies bot. Initializes components and runs the main loop.
"""

import time
import logging
import sys
import os

# If you need cross-platform input handling, do conditional imports:
# For Windows-only: import pydirectinput
# For cross-platform: import pyautogui
# We'll assume cross-platform for now:
import pyautogui

# Adjust Python's path if needed to ensure 'bot' package is found
# sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))

from bot.config import load_config
from bot.screen_capture import ScreenCapture
from bot.detection import ZombieDetector, HUDDetector
from bot.navigation import Navigator
from bot.decision import DecisionMaker
from bot.debug import DebugInterface
from bot.game_state import GameState
from bot.action_controller import ActionController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

def main():
    # 1) Load Configuration
    config_path = os.path.join("config", "game_settings.json")
    config = load_config(config_path, map_name=None)

    # 2) Initialize Bot Components
    screen_capture = ScreenCapture(config['game_region'])
    zombie_detector = ZombieDetector(config)
    hud_detector = HUDDetector(config)
    navigator = Navigator(config)
    decision_maker = DecisionMaker(config)
    game_state = GameState()
    debug_interface = DebugInterface()
    action_controller = ActionController(config)

    # 3) Let user switch to game window
    logger.info("Starting bot in 3 seconds... Switch to your game window now.")
    time.sleep(3)

    logger.info("Bot started. Press Ctrl+C to stop.")

    try:
        while True:
            # a) Capture Screen
            frame = screen_capture.capture()

            # b) Detect Zombies and HUD
            zombies = zombie_detector.detect(frame)
            health, ammo, current_weapon, current_round = hud_detector.detect_stats(frame)

            # c) Update Game State
            game_state.update(zombies, health, ammo, current_weapon, current_round)

            # d) Decide on Action
            action = decision_maker.decide(game_state)

            # e) Execute Action
            action_controller.execute(action)

            # f) Debug Visualization
            debug_interface.display(
                frame,
                zombies=zombies,
                health=health,
                ammo=ammo,
                game_state=game_state,
                action=action
            )

            # Add small delay to avoid overloading CPU
            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    finally:
        debug_interface.close()

if __name__ == "__main__":
    main()
