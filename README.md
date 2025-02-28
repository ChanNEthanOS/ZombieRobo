# COD WaW Zombies Bot
DOESNT WORK YET BUT IT WILL EVENTUALLY
An advanced Python bot for playing Call of Duty: World at War Zombies offline, featuring YOLO object detection, pathfinding, health/ammo monitoring, and situational weapon switching.

## Project Overview

This bot uses computer vision, machine learning, and decision algorithms to:

1. Detect zombies, weapons, and other game elements
2. Monitor player health, ammo, and round number
3. Make intelligent decisions about movement, shooting, and resource management
4. Navigate the map using pathfinding algorithms
5. Adapt to different situations and play styles

## Project Structure

```
├── bot/                  # Core bot logic
│   ├── actions.py        # Game actions (shooting, movement)
│   ├── config.py         # Configuration loading/saving
│   ├── debug.py          # Debug visualization
│   ├── decision.py       # Decision-making algorithms
│   ├── detection.py      # Object detection
│   ├── game_state.py     # Game state tracking
│   ├── navigation.py     # Pathfinding and navigation
│   ├── screen_capture.py # Screen capture utilities
│   └── utils.py          # Helper functions
├── config/               # Configuration files
│   ├── game_settings.json # Main settings
│   └── maps/             # Map-specific configurations
├── data/                 # Data management for ML training
│   ├── annotations.py    # Annotation utilities
│   └── dataset.py        # Dataset management
├── models/               # ML models
│   ├── train.py          # Training scripts
│   └── yolo_model.py     # YOLO implementation
└── main.py               # Main entry point
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install opencv-python numpy pyautogui mss torch torchvision pytesseract scikit-learn matplotlib networkx tqdm pyyaml
   ```
3. Make sure you have Call of Duty: World at War installed and running in windowed mode

## Usage

### Basic Usage

```bash
python main.py
```

### Testing Detection

```bash
python test_bot.py --test-detection
```

### Collecting Training Data

```bash
python test_bot.py --collect-data
```

### Debug Mode

```bash
python main.py --debug
```

## Map Support

The bot currently supports the original World at War Zombies maps:

- Nacht der Untoten
- Verrückt 
- Shi No Numa
- Der Riese

Custom map configurations can be added to the `config/maps/` directory.

## Features

### Zombie Detection

- YOLO-based object detection for zombies, crawlers, hellhounds
- Fallback color-based detection when YOLO is unavailable
- Distance and threat assessment

### HUD Detection

- Health monitoring
- Ammo counting
- Weapon identification
- Round number tracking

### Decision Making

- Threat assessment based on zombie proximity and player health
- Dynamic strategy switching between aggressive, defensive, and survival modes
- Weapon selection based on distance, ammo, and zombie count

### Navigation

- Graph-based map representation
- A* pathfinding for efficient navigation
- Obstacle detection and avoidance
- Training route planning for "zombie training"

### Action Execution

- Precise aiming with smooth mouse movement
- Strategic shooting, reloading, and weapon switching
- Movement actions with collision avoidance
- Use of perks, mystery box, and other game elements

## Customization

Edit the `config/game_settings.json` file to customize:

- Game region and detection parameters
- Weapon preferences and strategies
- Key bindings and sensitivity
- Debug visualization options

## Contributing

Contributions are welcome! Feel free to submit pull requests for new features, bug fixes, or improved detection models.

## Disclaimer

This bot is for educational purposes and intended for offline play only. Using bots in online games may violate terms of service.
