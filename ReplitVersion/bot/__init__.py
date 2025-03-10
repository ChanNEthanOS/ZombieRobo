"""
bot/__init__.py
Makes 'bot' a Python package and exposes core modules.
"""

__version__ = '1.0.0'
__author__ = 'ZombiesBot Dev Team'

# If you want to import submodules globally, you can do so here:
from .screen_capture import ScreenCapture
from .detection import ZombieDetector, HUDDetector
from .navigation import Navigator
from .decision import DecisionMaker
from .debug import DebugInterface 
from .game_state import GameState
from .action_controller import ActionController
