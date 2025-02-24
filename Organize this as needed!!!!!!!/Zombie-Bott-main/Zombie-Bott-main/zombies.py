#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import math
import random
import logging
from collections import deque
import cv2
import numpy as np
import keyboard
import mouse
from mss import mss
import torch
import torch.nn as nn
import torch.optim as optim


logging.basicConfig(filename="zombie_ai.log", level=logging.INFO, format="%(asctime)s - %(message)s")

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        return self.fc(x)

class AIReasoner:
    def __init__(self):
        self.knowledge_base = []
    
    def analyze_logs(self):
        try:
            with open("zombie_ai.log", "r") as f:
                logs = f.readlines()
                self.knowledge_base = logs[-20:]  # Store last 20 logs for analysis
        except FileNotFoundError:
            pass
    
    def make_decision(self):
        self.analyze_logs()
        if "low ammo" in str(self.knowledge_base):
            return "reload"
        elif "threat detected" in str(self.knowledge_base):
            return "shoot"
        return "move"

class ZombieAI:
    def __init__(self):
        self.sct = mss()
        self.model = NeuralNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=50000)
        self.reasoner = AIReasoner()
        self.config = {
            "screen_region": {"top": 0, "left": 0, "width": 1920, "height": 1080},
            "reaction_time": 0.006,
            "movement_speed": 0.14,
            "ballistics": {
                "mouse_sensitivity": 4.2,
                "smooth_steps": 90,
                "burst_delay": 0.012,
            },
            "safety": {
                "health_threshold": 35,
                "min_ammo": 4,
                "reload_time": 0.35
            }
        }
        self.last_reload = time.time()
        self.current_target = None

    def capture_screen(self):
        img = np.array(self.sct.grab(self.config["screen_region"]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def detect_zombies(self, frame):
        _, thresh = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 400]

    def select_target(self, threats):
        if not threats:
            return None
        screen_center = (self.config["screen_region"]["width"]//2, self.config["screen_region"]["height"]//2)
        return min(threats, key=lambda t: math.dist(t[:2], screen_center))

    def aim_and_shoot(self, target):
        if target:
            self._aim(target)
            if self.crosshair_is_red():
                self._shoot()

    def _aim(self, target):
        target_x, target_y = target[:2]
        screen_x, screen_y = mouse.get_position()
        dx, dy = (target_x - screen_x) / self.config["ballistics"]["smooth_steps"], (target_y - screen_y) / self.config["ballistics"]["smooth_steps"]
        for _ in range(self.config["ballistics"]["smooth_steps"]):
            mouse.move(screen_x + dx, screen_y + dy)
            time.sleep(0.0003)

    def _shoot(self):
        mouse.press()
        time.sleep(self.config["ballistics"]["burst_delay"])
        mouse.release()

    def crosshair_is_red(self):
        return random.choice([True, False])  # Placeholder logic for detecting red crosshair

    def move_tactically(self):
        actions = ["w", "a", "s", "d", "ctrl", "shift"]  # Forward, left, back, right, crouch, sprint
        action = random.choice(actions)
        keyboard.press(action)
        time.sleep(self.config["movement_speed"])
        keyboard.release(action)

    def reload_weapon(self):
        keyboard.press("r")
        time.sleep(self.config["safety"]["reload_time"])
        keyboard.release("r")

    def update_ai(self):
        self.config["reaction_time"] = max(0.002, self.config["reaction_time"] * 0.93)
        self.config["ballistics"]["smooth_steps"] = min(110, self.config["ballistics"]["smooth_steps"] + 1)
        logging.info(f"Updated AI Config: {self.config}")

    def run(self):
        print("Elite AI Combat System Activated - Press Ctrl+C to Exit")
        time.sleep(3)
        try:
            while True:
                decision = self.reasoner.make_decision()
                if decision == "reload":
                    self.reload_weapon()
                elif decision == "shoot" and self.current_target:
                    self.aim_and_shoot(self.current_target)
                else:
                    self.move_tactically()
                
                frame = self.capture_screen()
                threats = self.detect_zombies(frame)
                self.current_target = self.select_target(threats)
                self.update_ai()
                time.sleep(self.config["reaction_time"])
        except Exception as e:
            logging.error(f"Error encountered: {e}")
        except KeyboardInterrupt:
            print("\nAI Combat System Deactivated")

if __name__ == "__main__":
    bot = ZombieAI()
    bot.run()