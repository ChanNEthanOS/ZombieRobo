"""
Integrated AI System for CoD WaW Zombies Bot
This module connects Frank Castle AI with the original bot, using enhanced screen capture capabilities.
"""

import os
import sys
import time
import logging
import random
import json
import numpy as np
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integrated_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntegratedAI")

# Use enhanced screen capture
from enhanced_screen_capture import EnhancedScreenCapture

# Import existing bot components
from bot.actions import ActionController
from bot.detection import ZombieDetector, HUDDetector
from bot.decision import DecisionMaker
from bot.game_state import GameState

# Import new Frank Castle capabilities
from frank_castle_integration import Memory
from pathfinding import (
    astar, 
    calculate_path_to_target, 
    generate_exploration_path,
    generate_evasion_path
)
from combat_engagement import (
    select_best_target,
    aim_at_target,
    create_combat_action,
    calculate_zombie_threat_level
)

class IntegratedAI:
    """Integrated Frank Castle AI and WaW Zombies Bot"""
    
    def __init__(self, config=None):
        """
        Initialize the integrated AI
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.running = False
        
        # Load or create default configuration
        self._load_or_create_config()
        
        # Initialize enhanced screen capture
        self.screen_capture = EnhancedScreenCapture(
            game_region=self.config.get('game_region'),
            preferred_method=self.config.get('preferred_capture_method')
        )
        
        # Initialize Frank Castle components
        self.memory = Memory()
        self.known_areas = set()  # For exploration
        self.navigation_grid = np.zeros((100, 100), dtype=int)  # For pathfinding
        
        # Initialize original bot components
        self.game_state = GameState()
        self.zombie_detector = ZombieDetector(self.config)
        self.hud_detector = HUDDetector(self.config)
        self.decision_maker = DecisionMaker(self.config)
        self.action_controller = ActionController(
            sensitivity=self.config.get('mouse_sensitivity', 5.0)
        )
        
        # State variables
        self.current_path = []
        self.current_goal = None
        self.current_state = "idle"
        
        # Performance tracking
        self.start_time = None
        self.decisions_made = 0
        self.frames_processed = 0
        self.zombies_killed = 0
        self.rounds_survived = 0
        
        # Headless mode for testing
        self.headless = self.config.get('headless', False)
        
        logger.info("Integrated AI initialized")
        logger.info(f"Frank Castle integration active with state awareness")
        logger.info(f"Enhanced screen capture using method: {self.screen_capture.preferred_method}")
        
        if self.headless:
            logger.info("Running in headless mode (for development/testing)")
    
    def _load_or_create_config(self):
        """Load or create default configuration"""
        config_path = Path('config/integrated_ai_settings.json')
        
        # Default configuration
        default_config = {
            'game_region': {
                'top': 0,
                'left': 0,
                'width': 1920,
                'height': 1080
            },
            'preferred_capture_method': None,  # Auto-select best method
            'mouse_sensitivity': 5.0,
            'start_delay': 10,
            'headless': False,
            'debug_visualization': True,
            'zombie_detection': {
                'threshold': 0.7,
                'min_size': 100,
                'hsv_lower': [0, 120, 70],
                'hsv_upper': [10, 255, 255]
            },
            'combat': {
                'prioritize_closest': True,
                'lead_targets': True,
                'preferred_weapons': ['shotgun', 'rifle']
            },
            'save_memory_interval': 300  # seconds
        }
        
        # Load config if it exists
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                self.config = {**default_config, **loaded_config}
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self.config = default_config
        else:
            # Create default config
            self.config = default_config
            
            # Save default config
            try:
                config_path.parent.mkdir(exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info(f"Created default configuration at {config_path}")
            except Exception as e:
                logger.error(f"Failed to save default config: {e}")
    
    def start(self, delay=None):
        """
        Start the AI with optional delay
        
        Args:
            delay (int): Seconds to wait before starting (to switch to game)
        """
        if self.running:
            logger.warning("AI is already running")
            return
        
        # Use config delay if none specified
        if delay is None:
            delay = self.config.get('start_delay', 10)
        
        # Skip delay in headless mode
        if not self.headless and delay > 0:
            logger.info(f"Starting in {delay} seconds... Switch to game window!")
            for i in range(delay, 0, -1):
                logger.info(f"{i}...")
                time.sleep(1)
        
        logger.info("Integrated AI activated!")
        self.running = True
        self.start_time = time.time()
        
        # Run main loop
        try:
            self._run_main_loop()
        except KeyboardInterrupt:
            logger.info("AI stopped by user")
        except Exception as e:
            logger.exception(f"Error in AI execution: {e}")
        finally:
            self._cleanup()
    
    def _run_main_loop(self):
        """Main AI execution loop"""
        frame_time = time.time()
        status_time = time.time()
        memory_save_time = time.time()
        
        while self.running:
            # Calculate time since last frame
            current_time = time.time()
            delta_time = current_time - frame_time
            frame_time = current_time
            
            # Process game frame
            try:
                # Capture frame
                frame = self.screen_capture.capture()
                self.frames_processed += 1
                
                # Detect game elements
                zombies = self.zombie_detector.detect(frame)
                health, ammo, current_weapon, current_round = self.hud_detector.detect_stats(frame)
                
                # Update game state
                self.game_state.update(zombies, health, ammo, current_weapon, current_round)
                
                # Determine current state description
                state_desc = self._determine_state_description()
                
                # Enhanced decision making with Frank Castle AI
                enhanced_action = self._make_enhanced_decision(zombies, state_desc, delta_time)
                
                # Execute the action
                self.action_controller.execute(enhanced_action)
                self.decisions_made += 1
                
                # Log action in memory
                reward = self._calculate_reward(enhanced_action, zombies, health, ammo)
                self.memory.log_action(state_desc, enhanced_action['type'], reward, {
                    'health': health,
                    'ammo': ammo,
                    'zombies': len(zombies),
                    'weapon': current_weapon,
                    'round': current_round
                })
                
                # Periodically save memory
                if current_time - memory_save_time > self.config.get('save_memory_interval', 300):
                    self.memory._save_memory()
                    memory_save_time = current_time
                
                # Log status periodically
                if current_time - status_time > 5.0:
                    self._log_status()
                    status_time = current_time
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                
            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)
    
    def _determine_state_description(self):
        """
        Determine a descriptive state for memory logging
        
        Returns:
            str: Current state description
        """
        danger_level = self.game_state.danger_level
        zombie_count = len(self.game_state.zombies)
        health = self.game_state.health
        ammo = self.game_state.ammo
        
        # Determine state based on various factors
        if health < 30:
            return "critical_health"
        elif ammo < 10:
            return "low_ammo"
        elif danger_level > 0.7:
            return "high_danger"
        elif zombie_count > 5:
            return "many_zombies"
        elif zombie_count > 0:
            return "combat"
        else:
            return "exploration"
    
    def _make_enhanced_decision(self, zombies, state_desc, delta_time):
        """
        Make enhanced decisions using Frank Castle AI capabilities
        
        Args:
            zombies (list): Detected zombies
            state_desc (str): Current state description
            delta_time (float): Time since last frame
            
        Returns:
            dict: Action to execute
        """
        # Get base decision from original bot
        base_action = self.decision_maker.decide(self.game_state)
        
        # Check memory for best historical actions in this state
        best_actions = self.memory.get_most_rewarding_actions(state_desc)
        
        # If we have good historical data and some randomness, consider using it
        if best_actions and best_actions[0][1] > 0.5 and random.random() < 0.3:
            best_action_type = best_actions[0][0]
            
            # If best historical action differs from base, consider switching
            if best_action_type != base_action['type']:
                logger.info(f"Enhancing decision from {base_action['type']} to {best_action_type} based on memory")
                
                # Create enhanced action
                if best_action_type == "shoot":
                    # For shooting, we need a target
                    if zombies:
                        best_target = select_best_target(
                            zombies, 
                            prioritize_closest=self.config.get('combat', {}).get('prioritize_closest', True)
                        )
                        return create_combat_action(best_target)
                elif best_action_type == "move":
                    # For movement, generate direction based on the current situation
                    if state_desc == "high_danger" and zombies:
                        # Generate evasion path
                        danger_positions = [(z.get('center_x', 0), z.get('center_y', 0)) for z in zombies]
                        path = generate_evasion_path((50, 50), danger_positions)
                        if path:
                            next_point = path[0]
                            direction = "backward" if next_point[1] > 50 else "forward"
                            return {
                                "type": "move",
                                "description": f"Evading zombies ({direction})",
                                "params": {
                                    "direction": direction,
                                    "duration": 0.5
                                }
                            }
                    else:
                        # For regular movement, use base action
                        return base_action
                else:
                    # For other actions, use base parameters
                    return {
                        "type": best_action_type,
                        "description": f"Memory-enhanced {best_action_type}",
                        "params": base_action.get("params", {})
                    }
        
        # State-specific enhancements
        if state_desc == "exploration":
            # Use pathfinding for exploration
            if not self.current_path or random.random() < 0.05:  # 5% chance to change path
                player_pos = (50, 50)  # Center position
                self.current_path = generate_exploration_path(player_pos, self.known_areas)
                if self.current_path:
                    logger.debug(f"Generated new exploration path with {len(self.current_path)} points")
            
            # Use next point in path for movement direction
            if self.current_path:
                next_point = self.current_path.pop(0)
                # Add to known areas
                self.known_areas.add(next_point)
                
                dx = next_point[0] - 50  # Relative to center
                dy = next_point[1] - 50
                
                # Determine movement direction
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "backward" if dy > 0 else "forward"
                
                return {
                    "type": "move",
                    "description": f"Exploration movement ({direction})",
                    "params": {
                        "direction": direction,
                        "duration": base_action.get("params", {}).get("duration", 0.5)
                    }
                }
        
        elif state_desc == "combat":
            # Enhanced combat logic
            if zombies and base_action['type'] == "shoot":
                # Select best target
                best_target = select_best_target(
                    zombies, 
                    prioritize_closest=self.config.get('combat', {}).get('prioritize_closest', True)
                )
                
                # Create combat action with target leading
                return create_combat_action(
                    best_target, 
                    weapon_info={
                        "name": self.game_state.current_weapon,
                        "is_automatic": self.game_state.current_weapon in ["lmg", "rifle", "smg"]
                    }
                )
        
        elif state_desc == "high_danger":
            # Generate evasion path
            if zombies:
                danger_positions = [(z.get('center_x', 0), z.get('center_y', 0)) for z in zombies]
                path = generate_evasion_path((50, 50), danger_positions)
                if path:
                    next_point = path[0]
                    direction = "backward" if next_point[1] > 50 else "forward"
                    return {
                        "type": "move",
                        "description": f"Evading zombies ({direction})",
                        "params": {
                            "direction": direction,
                            "duration": 0.5
                        }
                    }
        
        # Use base action if no enhancements apply
        return base_action
    
    def _calculate_reward(self, action, zombies, health, ammo):
        """
        Calculate a reward for the current action
        
        Args:
            action (dict): The action taken
            zombies (list): Detected zombies
            health (int): Current health
            ammo (int): Current ammo
            
        Returns:
            float: Calculated reward
        """
        # This is a simplified reward system
        reward = 0
        
        # Reward for survival
        reward += 0.01
        
        # Penalty for low health
        if health < 30:
            reward -= 0.5
        
        # Reward for good ammo management
        if action['type'] == "reload" and ammo < 10:
            reward += 0.3
        
        # Reward for killing zombies (simulated)
        if action['type'] == "shoot" and zombies:
            # Assume some probability of killing zombie
            if random.random() < 0.3:
                reward += 1.0
                self.zombies_killed += 1
        
        # Reward for using grenades effectively
        if action['type'] == "grenade" and len(zombies) > 3:
            reward += 0.7
        
        return reward
    
    def _log_status(self):
        """Log current AI status"""
        if not self.start_time:
            return
            
        # Calculate runtime
        runtime = time.time() - self.start_time
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Capture method status
        capture_status = self.screen_capture.get_status()
        capture_method = capture_status['preferred_method'] or "test"
        capture_fps = capture_status['fps']
        
        # Log status
        logger.info(
            f"Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
            f"FPS: {capture_fps:.1f} | "
            f"State: {self.current_state} | "
            f"Health: {self.game_state.health}% | "
            f"Ammo: {self.game_state.ammo} | "
            f"Weapon: {self.game_state.current_weapon} | "
            f"Round: {self.game_state.current_round} | "
            f"Zombies: {len(self.game_state.zombies)} | "
            f"Decisions: {self.decisions_made} | "
            f"Capture: {capture_method}"
        )
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Stop action controller
        self.action_controller.cleanup()
        
        # Save memory
        self.memory._save_memory()
        
        # Mark as not running
        self.running = False
        
        # Log final statistics
        if self.start_time:
            runtime = time.time() - self.start_time
            logger.info(f"AI stopped after {runtime:.1f} seconds")
            logger.info(f"Made {self.decisions_made} decisions")
            logger.info(f"Processed {self.frames_processed} frames")
            logger.info(f"Killed {self.zombies_killed} zombies")
            logger.info(f"Total reward: {self.memory.total_reward:.2f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Integrated CoD WaW Zombies Bot with Frank Castle AI")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (for development)")
    parser.add_argument("--delay", type=int, default=10, help="Startup delay in seconds")
    parser.add_argument("--capture", type=str, help="Preferred capture method (mss, pil, x11, win32, d3d)")
    args = parser.parse_args()
    
    # Load default config
    config = {}
    try:
        config_path = Path('config/integrated_ai_settings.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
    
    # Override with command line arguments
    if args.headless:
        config['headless'] = True
    
    if args.capture:
        config['preferred_capture_method'] = args.capture
    
    # Create and start the AI
    ai = IntegratedAI(config)
    ai.start(delay=args.delay)


if __name__ == "__main__":
    main()