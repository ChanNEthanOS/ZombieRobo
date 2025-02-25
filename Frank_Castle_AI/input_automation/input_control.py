import pyautogui
import time

def press_key(key, duration=0.1):
    """Presses a key for a specified duration."""
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)

def move(direction, duration=0.5):
    """Simulates movement by holding down a directional key."""
    print(f"Moving {direction} for {duration} seconds")
    press_key(direction, duration)

def click_mouse(button='left'):
    """Simulates a mouse click."""
    print(f"Clicking {button} mouse button")
    pyautogui.click(button=button)

def perform_action(action):
    """
    Maps game actions to real keystrokes.
    - "advance": moves forward (W)
    - "buy_door": presses 'e' to buy a door
    - "reload": presses 'r' to reload
    - "engage_enemy": simulates shooting via a left-click
    """
    if action == "advance":
        move('w', duration=0.5)
    elif action == "buy_door":
        print("Performing buy door action: pressing 'f'")
        press_key('e', duration=0.2)
    elif action == "reload":
        print("Performing reload action: pressing 'r'")
        press_key('r', duration=0.2)
    elif action == "engage_enemy":
        print("Performing engage enemy action: left mouse click")
        click_mouse('left')
    else:
        print("No mapped action for:", action)
