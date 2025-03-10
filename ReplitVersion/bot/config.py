"""
bot/config.py
Handles loading and merging config JSON files.
"""

import json
import os
import logging
import copy

logger = logging.getLogger("Config")

DEFAULT_CONFIG = {
    "game_region": {
        "top": 0,
        "left": 0,
        "width": 1920,
        "height": 1080
    },
    "detection_confidence": 0.5,
    "lower_zombie_hsv": [0, 120, 70],
    "upper_zombie_hsv": [10, 255, 255],
    "health_region": {"top": 0.9, "left": 0.1, "width": 0.2, "height": 0.05},
    "ammo_region": {"top": 0.9, "left": 0.8, "width": 0.15, "height": 0.05},
    "weapon_region": {"top": 0.85, "left": 0.7, "width": 0.25, "height": 0.1},
    "round_region": {"top": 0.05, "left": 0.45, "width": 0.1, "height": 0.1},
    "yolo_model_path": "models/yolo_weights.pt",
    "min_shoot_distance": 20,
    "max_shoot_distance": 500,
    "preferred_distance": 150,
    "aggression_level": 0.7,
    "camping_tendency": 0.3,
    "reload_threshold": 10,
    "health_critical_threshold": 30,
    "ammo_critical_threshold": 5,
    "key_mappings": {
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "reload": "r",
        "weapon1": "1",
        "weapon2": "2",
        "melee": "v",
        "sprint": "shift",
        "jump": "space",
        "crouch": "ctrl",
        "use": "f",
        "grenade": "g"
    },
    "map_name": "default",
    "rotation_speed": 10,
    "movement_smoothing": 0.5,
    "reload_cooldown": 3.0,
    "weapon_switch_cooldown": 1.0,
    "melee_cooldown": 0.5,
    "shoot_cooldown": 0.1,
    "movement_cooldown": 0.05,
    "mouse_sensitivity": 1.0,
    "movement_duration": 0.2
}

def load_config(config_path, map_name=None):
    config = copy.deepcopy(DEFAULT_CONFIG)

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            _merge_configs(config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")

    if map_name:
        config['map_name'] = map_name

    _validate_config(config)
    return config

def _merge_configs(base_config, new_config):
    for key, value in new_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value

def _validate_config(config):
    # Example: ensure numeric ranges are valid
    config['detection_confidence'] = max(0.1, min(1.0, config['detection_confidence']))
    # ... etc. Expand as needed
    pass
