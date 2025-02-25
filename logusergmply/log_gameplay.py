import logging
import time
import threading
import tempfile

import tkinter as tk
from tkinter import simpledialog

import sounddevice as sd
import soundfile as sf
import whisper

from pynput import keyboard, mouse
from inputs import get_gamepad, UnpluggedError
import os
os.environ["PATH"] += r";C:\ffmpeg-2025-02-24-git-6232f416b1-essentials_build\bin"

import logging
import time
import threading
import tempfile
# ... rest of your imports ...

# Configure logging to output to a file with timestamps
logging.basicConfig(
    filename='gameplay_log.txt',
    level=logging.INFO,
    format='%(asctime)s: %(message)s'
)

# Create a hidden Tkinter root for text annotation dialogs
root = tk.Tk()
root.withdraw()  # Hide the main window

# Load the Whisper model once (this may take a moment)
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

def annotate_text():
    """
    Pops up a dialog to let you type in the reason behind your last action.
    """
    reason = simpledialog.askstring("Annotation", "Why are you doing this action?")
    if reason:
        logging.info("Text Annotation: " + reason)
    else:
        logging.info("Text Annotation: (No input provided)")

def annotate_voice(duration=122):
    """
    Records a short audio clip from the microphone, transcribes it using Whisper,
    and logs the transcription as an annotation.
    """
    samplerate = 16000  # Whisper works well with 16kHz audio
    print(f"Recording voice annotation for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()  # Wait until recording is finished
    # Save recording to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name
        sf.write(filename, recording, samplerate)
    # Transcribe using Whisper
    result = whisper_model.transcribe(filename)
    annotation = result['text']
    print("Voice annotation transcribed:", annotation)
    logging.info("Voice Annotation: " + annotation)

def on_press(key):
    # F9 for text annotation
    if key == keyboard.Key.f9:
        annotate_text()
        return
    # F10 for voice annotation
    elif key == keyboard.Key.f10:
        annotate_voice()
        return
    try:
        logging.info('Key pressed: {}'.format(key.char))
    except AttributeError:
        logging.info('Special key pressed: {}'.format(key))

def on_release(key):
    try:
        logging.info('Key released: {}'.format(key.char))
    except AttributeError:
        logging.info('Special key released: {}'.format(key))
    # Stop listener if ESC is pressed
    if key == keyboard.Key.esc:
        return False

def on_click(x, y, button, pressed):
    if pressed:
        logging.info('Mouse clicked at ({}, {}) with {}'.format(x, y, button))
    else:
        logging.info('Mouse released at ({}, {}) with {}'.format(x, y, button))

def on_scroll(x, y, dx, dy):
    logging.info('Mouse scrolled at ({}, {}), delta: ({}, {})'.format(x, y, dx, dy))

def log_controller_events():
    """
    Continuously listens for controller events and logs them.
    If no gamepad is found, we log a warning and keep trying every 5 seconds.
    """
    while True:
        try:
            events = get_gamepad()
            for event in events:
                logging.info(f"Controller event: {event.ev_type} - {event.code} (State: {event.state})")
        except UnpluggedError:
            logging.warning("No gamepad found. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    print("Starting extended gameplay logging.")
    print("Press F9 for text annotation, F10 for voice annotation, and ESC to stop.")
    
    # 4-second delay to give you time to open the game
    print("Waiting 4 seconds before logging starts. Go open your game!")
    time.sleep(4)

    # Start controller logging in a separate thread
    controller_thread = threading.Thread(target=log_controller_events, daemon=True)
    controller_thread.start()
    
    # Set up keyboard and mouse listeners
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    
    # Start the listeners
    keyboard_listener.start()
    mouse_listener.start()
    
    # Keep the script running until ESC is pressed
    keyboard_listener.join()
    mouse_listener.join()
