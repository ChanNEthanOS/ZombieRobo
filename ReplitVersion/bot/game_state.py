"""
bot/game_state.py
Maintains and updates the current game state.
"""

import logging
import time
import numpy as np

logger = logging.getLogger("GameState")

class GameState:
    def __init__(self):
        self.health = 100
        self.ammo = 30
        self.current_weapon = "rifle"
        self.current_round = 1
        self.zombies = []
        self.closest_zombie = None
        self.danger_level = 0

        self.start_time = time.time()

    def update(self, zombies, health, ammo, weapon, round_num):
        self.zombies = zombies
        self.health = health
        self.ammo = ammo
        self.current_weapon = weapon
        self.current_round = round_num

        if zombies:
            center_x = 960  # Approx screen center
            center_y = 540
            # find closest
            min_dist = float('inf')
            cz = None
            for z in zombies:
                dx = z['center_x'] - center_x
                dy = z['center_y'] - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
                    cz = z
            if cz:
                cz['distance'] = min_dist
            self.closest_zombie = cz
        else:
            self.closest_zombie = None

        self._assess_danger()

    def _assess_danger(self):
        # Example simplistic approach
        self.danger_level = 10 - (self.health / 10)
        if self.zombies and len(self.zombies) > 5:
            self.danger_level += 2

    def is_reload_needed(self):
        return self.ammo < 5

    def is_high_danger(self):
        return self.danger_level > 7
