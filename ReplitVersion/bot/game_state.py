"""
Game state module for the COD WaW Zombies Bot.
This module maintains and updates the current game state.
"""

import time
import logging
import numpy as np

logger = logging.getLogger("GameState")

class GameState:
    """Class for tracking and analyzing the game state"""
    
    def __init__(self):
        """Initialize the game state"""
        # Player state
        self.health = 100
        self.ammo = 30
        self.current_weapon = "primary"
        self.current_round = 1
        self.score = 0
        
        # Zombie tracking
        self.zombies = []
        self.closest_zombie = None
        self.zombie_count = 0
        self.zombies_killed = 0
        
        # Game environment
        self.current_position = None
        self.nearest_power_up = None
        self.nearest_wall_weapon = None
        self.doors_opened = set()
        
        # Performance metrics
        self.survival_time = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.update_count = 0
        
        # Game status
        self.is_game_active = True
        self.is_menu_screen = False
        self.is_game_over = False
        self.round_start_time = time.time()
        
        # Danger assessment
        self.danger_level = 0  # 0 (safe) to 10 (extreme danger)
        self.is_surrounded = False
        
        logger.info("Game state initialized")
    
    def update(self, zombies, health, ammo, current_weapon, current_round):
        """
        Update the game state with new information
        
        Args:
            zombies (list): List of detected zombies
            health (int): Current player health
            ammo (int): Current ammo count
            current_weapon (str): Current weapon
            current_round (int): Current game round
        """
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        
        # Update player state
        self.health = health
        self.ammo = ammo
        self.current_weapon = current_weapon
        
        # Check for round change
        if current_round != self.current_round:
            logger.info(f"Round changed: {self.current_round} -> {current_round}")
            self.round_start_time = current_time
            self.current_round = current_round
        
        # Update zombie information
        self.zombies = zombies
        self.zombie_count = len(zombies)
        
        # Find closest zombie
        if zombies:
            # Find center of screen (approximate player position)
            center_x = 1920 // 2  # Assuming 1920x1080 resolution
            center_y = 1080 // 2
            
            # Calculate distances from center
            distances = []
            for zombie in zombies:
                dx = zombie['center_x'] - center_x
                dy = zombie['center_y'] - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                distances.append((distance, zombie))
            
            # Sort by distance
            distances.sort(key=lambda x: x[0])
            
            # Get closest zombie
            self.closest_zombie = distances[0][1]
            self.closest_zombie['distance'] = distances[0][0]
        else:
            self.closest_zombie = None
        
        # Assess danger level
        self._assess_danger()
        
        # Update metrics
        self.survival_time = current_time - self.start_time
        self.last_update_time = current_time
        self.update_count += 1
        
        if self.update_count % 100 == 0:
            logger.debug(f"Game state updated {self.update_count} times, "
                        f"Survival time: {self.survival_time:.1f}s, "
                        f"Round: {self.current_round}, "
                        f"Zombies: {self.zombie_count}, "
                        f"Health: {self.health}%, "
                        f"Ammo: {self.ammo}")
    
    def _assess_danger(self):
        """Assess the current danger level based on zombies and health"""
        # Start with base danger level based on health
        self.danger_level = max(0, 10 - (self.health / 10))
        
        # Adjust based on zombie count and proximity
        if self.zombie_count > 0:
            # More zombies = more danger
            self.danger_level += min(5, self.zombie_count / 5)
            
            # Closer zombies = more danger
            if self.closest_zombie:
                proximity_danger = max(0, 5 - (self.closest_zombie['distance'] / 100))
                self.danger_level += proximity_danger
        
        # Adjust based on ammo
        if self.ammo < 10:
            self.danger_level += (10 - self.ammo) / 2
        
        # Check if surrounded
        surrounded_count = 0
        if self.zombies:
            # Count zombies in different quadrants of the screen
            quadrants = [0, 0, 0, 0]  # top-left, top-right, bottom-left, bottom-right
            
            center_x = 1920 // 2
            center_y = 1080 // 2
            
            for zombie in self.zombies:
                dx = zombie['center_x'] - center_x
                dy = zombie['center_y'] - center_y
                
                # Determine quadrant
                quadrant = 0
                if dx >= 0 and dy < 0:  # top-right
                    quadrant = 1
                elif dx < 0 and dy >= 0:  # bottom-left
                    quadrant = 2
                elif dx >= 0 and dy >= 0:  # bottom-right
                    quadrant = 3
                
                quadrants[quadrant] += 1
            
            # Count quadrants with zombies
            surrounded_count = sum(1 for q in quadrants if q > 0)
        
        self.is_surrounded = surrounded_count >= 3
        
        # Cap danger level at 10
        self.danger_level = min(10, self.danger_level)
    
    def is_low_health(self):
        """Check if health is low"""
        return self.health < 50
    
    def is_critical_health(self):
        """Check if health is critically low"""
        return self.health < 25
    
    def is_low_ammo(self):
        """Check if ammo is low"""
        return self.ammo < 10
    
    def is_reload_needed(self):
        """Check if reload is needed"""
        return self.ammo < 5
    
    def is_high_danger(self):
        """Check if danger level is high"""
        return self.danger_level > 7
    
    def get_nearest_zombie(self):
        """Get the nearest zombie if any"""
        return self.closest_zombie
    
    def get_zombie_count(self):
        """Get the number of detected zombies"""
        return self.zombie_count
    
    def get_survival_time(self):
        """Get the total survival time in seconds"""
        return self.survival_time
    
    def get_round_time(self):
        """Get the time spent in current round"""
        return time.time() - self.round_start_time
    
    def get_state_summary(self):
        """
        Get a summary of the current game state
        
        Returns:
            dict: Summary of current game state
        """
        return {
            'health': self.health,
            'ammo': self.ammo,
            'weapon': self.current_weapon,
            'round': self.current_round,
            'zombies': self.zombie_count,
            'danger_level': f"{self.danger_level:.1f}/10",
            'surrounded': self.is_surrounded,
            'survival_time': f"{self.survival_time:.1f}s",
            'round_time': f"{self.get_round_time():.1f}s",
            'closest_zombie_distance': self.closest_zombie['distance'] if self.closest_zombie else None
        }
