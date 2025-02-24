class GameState:
    def __init__(self):
        self.player_position = [0, 0]
        self.enemy_positions = [[5, 5], [10, 10]]
        self.action_space = ['move_up', 'move_down', 'move_left', 'move_right', 'shoot']

    def get_possible_actions(self):
        return self.action_space

    def apply_action(self, action):
        new_state = GameState()
        new_state.player_position = self.player_position.copy()
        new_state.enemy_positions = [pos.copy() for pos in self.enemy_positions]

        if action == 'move_up':
            new_state.player_position[1] += 1
        elif action == 'move_down':
            new_state.player_position[1] -= 1
        elif action == 'move_left':
            new_state.player_position[0] -= 1
        elif action == 'move_right':
            new_state.player_position[0] += 1
        elif action == 'shoot':
            print("\n*** BANG! ***\n", flush=True)

        return new_state

    def __str__(self):
        return (
            "\n------------------------"
            f"\nPlayer Position: ({self.player_position[0]}, {self.player_position[1]})"
            f"\nEnemies Nearby: {len(self.enemy_positions)}"
            "\n------------------------\n"
        )