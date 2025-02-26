"""
Decision module for the COD WaW Zombies Bot.
This module decides what actions to take based on game state.
"""

import logging
import numpy as np
import time
import random

logger = logging.getLogger("Decision")

class DecisionMaker:
    """Class for making gameplay decisions based on game state"""
    
    def __init__(self, config):
        """
        Initialize the decision maker
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Decision parameters
        self.min_shoot_distance = config.get('min_shoot_distance', 20)
        self.max_shoot_distance = config.get('max_shoot_distance', 500)
        self.reload_threshold = config.get('reload_threshold', 10)
        self.health_critical_threshold = config.get('health_critical_threshold', 30)
        self.ammo_critical_threshold = config.get('ammo_critical_threshold', 5)
        
        # Strategy params
        self.preferred_distance = config.get('preferred_distance', 150)
        self.aggression_level = config.get('aggression_level', 0.7)  # 0-1
        self.camping_tendency = config.get('camping_tendency', 0.3)  # 0-1
        
        # Decision state
        self.last_decision_time = time.time()
        self.last_action = None
        self.action_history = []
        self.strategy_mode = "normal"  # normal, aggressive, defensive, camping
        self.current_target = None
        
        # Cooldowns for strategies
        self.strategy_change_time = time.time()
        self.strategy_change_cooldown = 30.0  # seconds
        
        logger.info("Decision maker initialized")
    
    def decide(self, game_state):
        """
        Decide what action to take based on the current game state
        
        Args:
            game_state (GameState): Current game state
            
        Returns:
            dict: Action to execute
        """
        current_time = time.time()
        time_since_last_decision = current_time - self.last_decision_time
        
        # Update strategy periodically or based on game state
        self._update_strategy(game_state)
        
        # Check for critical situations first
        if game_state.is_critical_health() and game_state.is_high_danger():
            # Critical situation - try to escape and survive
            action = self._get_survival_action(game_state)
        elif game_state.is_reload_needed() and not game_state.is_high_danger():
            # Need to reload and not in immediate danger
            action = {'type': 'reload'}
        elif game_state.is_low_health() and game_state.danger_level > 5:
            # Low health and moderate danger - be defensive
            action = self._get_defensive_action(game_state)
        else:
            # Normal gameplay based on current strategy
            if self.strategy_mode == "aggressive":
                action = self._get_aggressive_action(game_state)
            elif self.strategy_mode == "defensive":
                action = self._get_defensive_action(game_state)
            elif self.strategy_mode == "camping":
                action = self._get_camping_action(game_state)
            else:  # normal
                action = self._get_normal_action(game_state)
        
        # Track action history
        self.last_action = action
        self.action_history.append((current_time, action['type']))
        self.last_decision_time = current_time
        
        # Keep action history from growing too large
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        logger.debug(f"Decision: {action['type']} (strategy: {self.strategy_mode})")
        return action
    
    def _update_strategy(self, game_state):
        """Update the current strategy based on game state"""
        current_time = time.time()
        
        # Only change strategy after cooldown period
        if current_time - self.strategy_change_time < self.strategy_change_cooldown:
            return
        
        # Factors that influence strategy
        round_factor = min(1.0, game_state.current_round / 15)
        health_factor = max(0, min(1.0, game_state.health / 100))
        ammo_factor = max(0, min(1.0, game_state.ammo / 30))
        zombies_factor = min(1.0, game_state.zombie_count / 10)
        
        # Calculate strategy scores
        aggressive_score = (
            self.aggression_level * 2 + 
            round_factor * 0.5 + 
            health_factor * 0.7 + 
            ammo_factor - 
            zombies_factor * 0.5
        )
        
        defensive_score = (
            (1 - health_factor) * 2 + 
            zombies_factor - 
            ammo_factor * 0.5
        )
        
        camping_score = (
            self.camping_tendency * 2 + 
            (1 - ammo_factor) * 0.5 + 
            round_factor - 
            health_factor * 0.3
        )
        
        normal_score = 1.0  # baseline
        
        # Determine highest scoring strategy
        scores = {
            "aggressive": aggressive_score,
            "defensive": defensive_score,
            "camping": camping_score,
            "normal": normal_score
        }
        
        # Add some randomness to prevent getting stuck in one strategy
        for strategy in scores:
            scores[strategy] += random.uniform(0, 0.3)
        
        new_strategy = max(scores, key=scores.get)
        
        # Only log if strategy changed
        if new_strategy != self.strategy_mode:
            logger.info(f"Strategy changed: {self.strategy_mode} -> {new_strategy}")
            self.strategy_mode = new_strategy
            self.strategy_change_time = current_time
    
    def _get_normal_action(self, game_state):
        """Get a normal balanced action"""
        # Check for zombies
        if game_state.closest_zombie:
            zombie = game_state.closest_zombie
            distance = zombie['distance']
            
            # If zombie is within shooting range
            if self.min_shoot_distance <= distance <= self.max_shoot_distance:
                # Aim and shoot
                return {
                    'type': 'shoot',
                    'target': {
                        'x': zombie['center_x'],
                        'y': zombie['center_y']
                    },
                    'burst': distance < 100  # Use burst fire for close zombies
                }
            
            # If zombie is too close, back up
            elif distance < self.min_shoot_distance:
                return {
                    'type': 'move',
                    'direction': 'backward',
                    'duration': 0.5
                }
            
            # If zombie is far, move closer if we're aggressive
            elif distance > self.preferred_distance and random.random() < self.aggression_level:
                return {
                    'type': 'move',
                    'direction': 'forward',
                    'duration': 0.5
                }
        
        # No immediate threats, move around
        move_actions = [
            {'type': 'move', 'direction': 'forward', 'duration': 0.7},
            {'type': 'move', 'direction': 'left', 'duration': 0.3},
            {'type': 'move', 'direction': 'right', 'duration': 0.3},
            {'type': 'look', 'angle': random.uniform(0, 360), 'distance': 50}
        ]
        
        return random.choice(move_actions)
    
    def _get_aggressive_action(self, game_state):
        """Get an aggressive action focused on attacking zombies"""
        # Check for zombies
        if game_state.closest_zombie:
            zombie = game_state.closest_zombie
            distance = zombie['distance']
            
            # Always shoot if in range
            if distance <= self.max_shoot_distance:
                # Aim and shoot
                return {
                    'type': 'shoot',
                    'target': {
                        'x': zombie['center_x'],
                        'y': zombie['center_y']
                    },
                    'burst': True  # Always use burst fire when aggressive
                }
            
            # Move towards zombies
            else:
                angle_to_zombie = np.arctan2(
                    zombie['center_y'] - 1080//2,
                    zombie['center_x'] - 1920//2
                ) * 180 / np.pi
                
                return {
                    'type': 'sequence',
                    'sequence': [
                        {'type': 'look', 'angle': angle_to_zombie, 'distance': 50},
                        {'type': 'move', 'direction': 'forward', 'duration': 1.0, 'sprint': True}
                    ]
                }
        
        # No zombies visible, search aggressively
        search_actions = [
            {'type': 'move', 'direction': 'forward', 'duration': 1.0, 'sprint': True},
            {'type': 'look', 'angle': random.uniform(0, 360), 'distance': 100}
        ]
        
        return random.choice(search_actions)
    
    def _get_defensive_action(self, game_state):
        """Get a defensive action focused on survival"""
        # Check for zombies
        if game_state.closest_zombie:
            zombie = game_state.closest_zombie
            distance = zombie['distance']
            
            # If zombie is too close, back up and shoot
            if distance < self.preferred_distance:
                return {
                    'type': 'sequence',
                    'sequence': [
                        {'type': 'move', 'direction': 'backward', 'duration': 0.7},
                        {'type': 'shoot', 'target': {
                            'x': zombie['center_x'],
                            'y': zombie['center_y']
                        }}
                    ]
                }
            
            # If zombie is at good distance, just shoot
            elif distance <= self.max_shoot_distance:
                return {
                    'type': 'shoot',
                    'target': {
                        'x': zombie['center_x'],
                        'y': zombie['center_y']
                    }
                }
        
        # No immediate threats, move to safer position
        if game_state.is_surrounded:
            # Try to break out if surrounded
            return {'type': 'sequence',
                   'sequence': [
                       {'type': 'look', 'angle': random.uniform(0, 360), 'distance': 50},
                       {'type': 'move', 'direction': 'forward', 'duration': 1.0, 'sprint': True},
                       {'type': 'jump'}
                   ]}
        else:
            # Back up to a safer position
            return {'type': 'move', 'direction': 'backward', 'duration': 0.5}
    
    def _get_camping_action(self, game_state):
        """Get a camping action focused on holding position"""
        # Check for zombies
        if game_state.closest_zombie:
            zombie = game_state.closest_zombie
            distance = zombie['distance']
            
            # If zombie is in range, shoot
            if distance <= self.max_shoot_distance:
                return {
                    'type': 'shoot',
                    'target': {
                        'x': zombie['center_x'],
                        'y': zombie['center_y']
                    }
                }
        
        # Look around for zombies
        look_directions = [0, 45, 90, 135, 180, 225, 270, 315]
        return {'type': 'look', 'angle': random.choice(look_directions), 'distance': 80}
    
    def _get_survival_action(self, game_state):
        """Get an action focused on pure survival in critical situations"""
        # This is for when health is critical and danger is high
        
        # If surrounded, try to create an escape path
        if game_state.is_surrounded:
            # Complex escape sequence
            return {'type': 'sequence',
                   'sequence': [
                       # Look in random direction
                       {'type': 'look', 'angle': random.uniform(0, 360), 'distance': 100},
                       # Sprint forward to break through
                       {'type': 'move', 'direction': 'forward', 'duration': 1.5, 'sprint': True},
                       # Jump to avoid zombies
                       {'type': 'jump'},
                       # Sharp turn
                       {'type': 'look', 'angle': random.uniform(0, 360), 'distance': 100},
                       # Continue sprinting
                       {'type': 'move', 'direction': 'forward', 'duration': 1.0, 'sprint': True}
                   ]}
        
        # If not surrounded but in danger, back away and shoot closest zombie
        elif game_state.closest_zombie:
            return {'type': 'sequence',
                   'sequence': [
                       # Back up first
                       {'type': 'move', 'direction': 'backward', 'duration': 0.7},
                       # Shoot closest zombie
                       {'type': 'shoot', 'target': {
                           'x': game_state.closest_zombie['center_x'],
                           'y': game_state.closest_zombie['center_y']
                       }},
                       # Strafe to avoid getting hit
                       {'type': 'move', 'direction': random.choice(['left', 'right']), 'duration': 0.5}
                   ]}
        
        # No clear threat but health is critical, just run
        return {'type': 'move', 'direction': 'forward', 'duration': 2.0, 'sprint': True}
    
    def get_decision_stats(self):
        """
        Get statistics about recent decisions
        
        Returns:
            dict: Decision statistics
        """
        if not self.action_history:
            return {'no_actions': True}
        
        # Calculate action frequencies
        action_types = {}
        for _, action_type in self.action_history:
            if action_type in action_types:
                action_types[action_type] += 1
            else:
                action_types[action_type] = 1
        
        # Convert to percentages
        total = len(self.action_history)
        action_percentages = {k: f"{v/total*100:.1f}%" for k, v in action_types.items()}
        
        return {
            'current_strategy': self.strategy_mode,
            'action_frequencies': action_percentages,
            'last_action': self.last_action['type'] if self.last_action else None,
            'decisions_made': len(self.action_history),
            'time_since_last': time.time() - self.last_decision_time
        }
