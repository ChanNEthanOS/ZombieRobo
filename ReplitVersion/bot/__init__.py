"""
COD WaW Zombies Bot - Bot Package
This package contains the core bot functionality for playing CoD WaW Zombies.
"""

__version__ = '1.0.0'
__author__ = 'ZombiesBot Development Team'

from bot.screen_capture import ScreenCapture
from bot.detection import ZombieDetector, HUDDetector
from bot.navigation import Navigator
from bot.actions import ActionController
from bot.game_state import GameState
from bot.decision import DecisionMaker
from bot.debug import DebugInterface
from bot.config import load_config

# Export main classes
__all__ = [
    'ScreenCapture',
    'ZombieDetector',
    'HUDDetector',
    'Navigator',
    'ActionController',
    'GameState',
    'DecisionMaker',
    'DebugInterface',
    'load_config'
]
