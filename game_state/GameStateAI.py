# ai_game_project/game_state/GameStateAI.py

import numpy as np

class GameStateAI:
    def __init__(self):
        self.player_position = np.array([0, 0], dtype=np.float32)
        self.enemy_positions = [np.array([5, 5], dtype=np.float32), np.array([10, 10], dtype=np.float32)]
        self.action_space = ['move_up', 'move_down', 'move_left', 'move_right', 'shoot']

    def get_possible_actions(self):
        return self.action_space

    def apply_action(self, action):
        if action == 'move_up':
            self.player_position[1] += 1
        elif action == 'move_down':
            self.player_position[1] -= 1
        elif action == 'move_left':
            self.player_position[0] -= 1
        elif action == 'move_right':
            self.player_position[0] += 1
        elif action == 'shoot':
            print("AI Frank shoots!")
        return self
