"""
bot/decision.py
Decides actions (move, shoot, reload, etc.) based on game state.
"""

import logging
import random
import numpy as np
import time

logger = logging.getLogger("Decision")

class DecisionMaker:
    def __init__(self, config):
        self.config = config
        self.min_shoot_distance = config.get('min_shoot_distance', 20)
        self.max_shoot_distance = config.get('max_shoot_distance', 500)
        self.preferred_distance = config.get('preferred_distance', 150)
        self.aggression_level = config.get('aggression_level', 0.7)
        self.camping_tendency = config.get('camping_tendency', 0.3)
        self.reload_threshold = config.get('reload_threshold', 10)

    def decide(self, game_state):
        # Basic example:
        if game_state.is_reload_needed() and not game_state.is_high_danger():
            return {'type': 'reload'}

        if game_state.closest_zombie:
            dist = game_state.closest_zombie.get('distance', 9999)
            if dist < self.min_shoot_distance:
                return {'type': 'move', 'direction': 'backward', 'duration': 0.5}
            elif dist < self.max_shoot_distance:
                return {'type': 'shoot'}
            else:
                return {'type': 'move', 'direction': 'forward', 'duration': 0.5}
        else:
            # No zombies? Just random idle action
            return {'type': 'move', 'direction': random.choice(['left','right']), 'duration':0.5}
