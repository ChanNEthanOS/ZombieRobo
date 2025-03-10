import sys
import time
import pyautogui

class ActionController:
    def __init__(self, config):
        self.keys = config['key_mappings']

    def execute(self, action):
        method_name = f"_do_{action['type']}"
        method = getattr(self, method_name, self._unknown_action)
        method(action)

    def _do_move(self, action):
        key = self.keys.get(action['direction'], 'w')
        duration = action.get('duration', 0.5)
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def _do_shoot(self, action):
        # Consider mouse aiming logic
        pyautogui.click()

    def _unknown_action(self, action):
        print(f"Unknown action: {action['type']}")
