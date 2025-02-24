# ai_game_project/screen_capture/capture_screen.py

import mss
import numpy as np
import keyboard

def capture_screen_ai():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Captures the first monitor
        while True:
            frame = np.array(sct.grab(monitor))
            if keyboard.is_pressed('q'):  # Press 'q' to quit
                break
