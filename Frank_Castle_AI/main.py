import cv2
from vision_system.enemy_detection import detect_enemies
from memory_system.memory import Memory
from movement_system.pathfinding import astar, plan_in_game_action
from combat_system.combat_engagement import engage_target
from models.mistral_model import generate_decision
from gui_overlay.overlay import Overlay
from voice_interaction.whisper_integration import transcribe_audio

def main():
    # Initialize systems
    memory = Memory()
    overlay = Overlay()
    
    # Example: Initialize video capture (this might be game capture or webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open video capture. Exiting...")
        return

    # Example grid for pathfinding demo (update with actual game map data later)
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (2, 3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Vision: Detect enemies
        enemy_coords = detect_enemies(frame)
        if enemy_coords:
            for coord in enemy_coords:
                engage_target(coord)
                memory.log_action(state="enemy_detected", action=f"engage at {coord}", reward=1)
        
        # Pathfinding example
        path = astar(grid, start, goal)
        print("Calculated path:", path)
        
        # Example in-game decision based on player state
        player_state = {'ammo': 3, 'funds': 120}  # update with real data
        action = plan_in_game_action(player_state)
        print("Planned in-game action:", action)
        
        # Decision making via ML model (example prompt)
        decision = generate_decision("Frank, should I engage, retreat, or reposition?")
        print("ML Decision:", decision)
        
        # Update GUI overlay with the decision text
        overlay.update_text(decision)
        
        # Display frame for debugging (press 'q' to exit)
        cv2.imshow("Game Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
