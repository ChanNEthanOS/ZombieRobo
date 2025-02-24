import time
from src.game_state import GameState
from src.control import FrankControl
from src.config import DIALOGUE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def display_conversation():
    try:
        print("\n=== Character Dialogue ===", flush=True)
        for line in DIALOGUE:
            print(line, flush=True)
            time.sleep(2)
        print("\n=== Dialogue Complete ===\n", flush=True)
    except Exception as e:
        logger.error(f"Error in conversation display: {e}")

def main():
    try:
        # Display initial dialogue
        display_conversation()

        state = GameState()
        frank_control = FrankControl(state)

        print("\nGame Controls:", flush=True)
        print("W - Move Up", flush=True)
        print("S - Move Down", flush=True)
        print("A - Move Left", flush=True)
        print("D - Move Right", flush=True)
        print("Space - Shoot", flush=True)
        print("Q - Quit\n", flush=True)

        # Main game loop
        while True:
            try:
                user_input = input("Enter command (W/A/S/D/Space, Q to quit): ")
                if user_input.lower() == 'q':
                    logger.info("Shutting down...")
                    break

                state = frank_control.control(user_input)
                time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break

    except Exception as e:
        logger.error(f"Critical error in main loop: {e}")

if __name__ == "__main__":
    main()