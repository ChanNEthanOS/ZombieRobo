import time
from memory_system.memory import Memory
from movement_system.pathfinding import astar, plan_in_game_action
from combat_system.combat_engagement import engage_target
from models.mistral_model import generate_decision
from gui_overlay.overlay import Overlay
from input_automation.input_control import perform_action

def main():
    # Initialize systems
    memory = Memory()
    overlay = Overlay()
    
    # Define a grid for simulation (0 = free, 1 = obstacle)
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (2, 3)
    path = astar(grid, start, goal)
    print("Calculated path:", path)
    
    # Initialize player state
    player_state = {
        'ammo': 10,
        'funds': 150,
        'position': start,
    }
    
    # Simulate game loop along the calculated path
    for pos in path:
        player_state['position'] = pos
        print(f"\nPlayer moved to {pos}")
        
        # Simulate an enemy encounter at a designated position (for demo, pos (0,2))
        if pos == (0, 2):
            print("Enemy encountered!")
            engage_target(pos)  # Simulate shooting at the enemy
            player_state['ammo'] -= 1  # Reduce ammo
            memory.log_action(state="enemy_encounter", action="engaged enemy", reward=1)
            # Execute enemy engagement action (simulate shooting via input automation)
            perform_action("engage_enemy")
        
        # Get an ML decision for flavor (this can be expanded later)
        decision = generate_decision("Frank, decide next action based on current state.")
        print("ML Decision:", decision)
        
        # Determine in-game action based on player state (buy door, reload, or advance)
        action = plan_in_game_action(player_state)
        print("Planned in-game action:", action)
        
        # Execute the planned action using input automation
        perform_action(action)
        
        # Update overlay with current state info
        overlay.update_text(f"Pos: {pos}, Action: {action}")
        print("Player state:", player_state)
        
        time.sleep(1)  # simulate a delay per game tick
        
    print("\nReached goal!")
    overlay.update_text("Goal reached!")
    time.sleep(3)

if __name__ == "__main__":
    main()
