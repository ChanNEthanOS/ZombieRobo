"""
Configuration module for the COD WaW Zombies Bot.
This module handles loading and validating configuration settings.
"""

import json
import os
import logging
import copy

logger = logging.getLogger("Config")

# Default configuration settings
DEFAULT_CONFIG = {
    'game_region': {
        'top': 0,
        'left': 0,
        'width': 1920,
        'height': 1080
    },
    'detection_confidence': 0.5,
    'lower_zombie_hsv': [0, 120, 70],
    'upper_zombie_hsv': [10, 255, 255],
    'health_region': {'top': 0.9, 'left': 0.1, 'width': 0.2, 'height': 0.05},
    'ammo_region': {'top': 0.9, 'left': 0.8, 'width': 0.15, 'height': 0.05},
    'weapon_region': {'top': 0.85, 'left': 0.7, 'width': 0.25, 'height': 0.1},
    'round_region': {'top': 0.05, 'left': 0.45, 'width': 0.1, 'height': 0.1},
    'yolo_model_path': 'models/yolo_weights.pt',
    'min_shoot_distance': 20,
    'max_shoot_distance': 500,
    'preferred_distance': 150,
    'aggression_level': 0.7,
    'camping_tendency': 0.3,
    'reload_threshold': 10,
    'health_critical_threshold': 30,
    'ammo_critical_threshold': 5,
    'key_mappings': {
        'forward': 'w',
        'backward': 's',
        'left': 'a',
        'right': 'd',
        'reload': 'r',
        'weapon1': '1',
        'weapon2': '2',
        'melee': 'v',
        'sprint': 'shift',
        'jump': 'space',
        'crouch': 'ctrl',
        'use': 'f',
        'grenade': 'g'
    },
    'map_name': 'default',
    'rotation_speed': 10,
    'movement_smoothing': 0.5,
    'reload_cooldown': 3.0,
    'weapon_switch_cooldown': 1.0,
    'melee_cooldown': 0.5,
    'shoot_cooldown': 0.1,
    'movement_cooldown': 0.05,
    'mouse_sensitivity': 1.0,
    'movement_duration': 0.2
}

def load_config(config_path, map_name=None):
    """
    Load configuration from a JSON file
    
    Args:
        config_path (str): Path to the config file
        map_name (str, optional): Map name to use
        
    Returns:
        dict: Configuration dictionary
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    # Load config from file if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge config with defaults
            _merge_configs(config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
    
    # Override map name if provided
    if map_name:
        config['map_name'] = map_name
    
    # Validate the config
    _validate_config(config)
    
    return config

def _merge_configs(base_config, new_config):
    """
    Recursively merge two configuration dictionaries
    
    Args:
        base_config (dict): Base configuration
        new_config (dict): New configuration to merge in
    """
    for key, value in new_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value

def _validate_config(config):
    """
    Validate configuration values and set bounds if needed
    
    Args:
        config (dict): Configuration dictionary
    """
    # Validate game region
    if 'game_region' in config:
        region = config['game_region']
        if region['width'] <= 0 or region['height'] <= 0:
            logger.warning("Invalid game region dimensions, resetting to defaults")
            config['game_region']['width'] = DEFAULT_CONFIG['game_region']['width']
            config['game_region']['height'] = DEFAULT_CONFIG['game_region']['height']
    
    # Validate HSV ranges
    for key in ['lower_zombie_hsv', 'upper_zombie_hsv']:
        if key in config:
            for i in range(len(config[key])):
                if key == 'lower_zombie_hsv':
                    config[key][i] = max(0, min(255, config[key][i]))
                else:
                    config[key][i] = max(config['lower_zombie_hsv'][i], min(255, config[key][i]))
    
    # Validate numeric ranges
    ranges = {
        'detection_confidence': (0.1, 1.0),
        'min_shoot_distance': (5, 100),
        'max_shoot_distance': (100, 1000),
        'preferred_distance': (50, 300),
        'aggression_level': (0.0, 1.0),
        'camping_tendency': (0.0, 1.0),
        'mouse_sensitivity': (0.1, 3.0)
    }
    
    for key, (min_val, max_val) in ranges.items():
        if key in config:
            config[key] = max(min_val, min(max_val, config[key]))
    
    logger.debug("Configuration validated")
    return config

def save_config(config, config_path):
    """
    Save configuration to a JSON file
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save the config file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        return False
