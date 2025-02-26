"""
Direct input controller for COD WaW Zombies Bot.
This module provides direct Windows API input for full screen games.
"""

import ctypes
import time
import random
import logging

# Configure logging
logger = logging.getLogger("DirectInput")

# Define required constants for Windows API
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# Mouse event constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000

# Key codes (scan codes are more reliable than virtual key codes for games)
KEY_W = 0x11      # Forward
KEY_S = 0x1F      # Backward
KEY_A = 0x1E      # Left
KEY_D = 0x20      # Right
KEY_SPACE = 0x39  # Jump
KEY_LSHIFT = 0x2A # Sprint
KEY_CTRL = 0x1D   # Crouch
KEY_R = 0x13      # Reload
KEY_F = 0x21      # Use
KEY_G = 0x22      # Grenade
KEY_1 = 0x02      # Weapon 1
KEY_2 = 0x03      # Weapon 2
KEY_ESC = 0x01    # Menu

# Set up the SendInput function
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class InputI(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", InputI)
    ]

class DirectInputController:
    """Direct input controller for Call of Duty: World at War"""
    
    def __init__(self, sensitivity=5.0):
        """Initialize controller with specified sensitivity"""
        self.sensitivity = sensitivity
        self.pressed_keys = set()
        logger.info("Direct Input Controller initialized")
    
    def press_key_scan(self, scan_code):
        """Press a key using scan code (more reliable for games)"""
        if scan_code in self.pressed_keys:
            return  # Already pressed
            
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, scan_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self.pressed_keys.add(scan_code)
    
    def release_key_scan(self, scan_code):
        """Release a key using scan code"""
        if scan_code not in self.pressed_keys:
            return  # Not pressed
            
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, scan_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 
                           ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self.pressed_keys.discard(scan_code)
    
    def press_key_for_duration(self, scan_code, duration=0.5):
        """Press and hold a key for specified duration"""
        try:
            self.press_key_scan(scan_code)
            time.sleep(duration)
        finally:
            self.release_key_scan(scan_code)
    
    def mouse_move(self, dx, dy):
        """Move mouse by dx, dy pixels"""
        # Apply sensitivity
        dx = int(dx * self.sensitivity)
        dy = int(dy * self.sensitivity)
        
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.mi = MouseInput(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(0), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def mouse_left_down(self):
        """Press left mouse button"""
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(0), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def mouse_left_up(self):
        """Release left mouse button"""
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(0), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def mouse_right_down(self):
        """Press right mouse button"""
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTDOWN, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(0), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def mouse_right_up(self):
        """Release right mouse button"""
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(0), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def mouse_click(self, button='left', duration=0.1):
        """Click and hold a mouse button"""
        try:
            if button == 'left':
                self.mouse_left_down()
            elif button == 'right':
                self.mouse_right_down()
            
            time.sleep(duration)
            
        finally:
            if button == 'left':
                self.mouse_left_up()
            elif button == 'right':
                self.mouse_right_up()
    
    def cleanup(self):
        """Release all pressed keys"""
        logger.info("Cleaning up - releasing all keys")
        for key in list(self.pressed_keys):
            self.release_key_scan(key)
        self.pressed_keys.clear()