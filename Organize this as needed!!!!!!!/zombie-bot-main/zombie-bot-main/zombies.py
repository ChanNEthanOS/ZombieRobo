import sys
import time
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# Offline Mistral 7B Configuration
MODEL_PATH = "C:/Users/Ethan/Documents/Models/mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

def query_mistral(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------------------------------------------
# Smart Point Spending System
class PointsManager:
    def __init__(self, initial_points=1000):
        self.points = initial_points
    
    def can_spend(self, cost):
        return self.points - cost >= 500  # Ensure 500 points remain
    
    def spend(self, cost):
        if self.can_spend(cost):
            self.points -= cost
            logging.info(f"Spent {cost} points. Remaining: {self.points}")
            return True
        logging.info(f"Insufficient points to spend {cost}. Remaining: {self.points}")
        return False

# ------------------------------------------------------------------
# AI Zombie Combat System with Mistral 7B as Unsupervised Controller
class ZombieAI:
    def __init__(self):
        self.sct = mss()
        self.points_manager = PointsManager()
        self.memory = deque(maxlen=10000)
        self.last_update_time = time.time()
        self.health = 100
    
    def capture_screen(self):
        img = np.array(self.sct.grab({"top": 0, "left": 0, "width": 1920, "height": 1080}))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def detect_zombies(self, frame):
        _, thresh = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 400]
        return detected
    
    def mistral_decision(self, state):
        prompt = f"Given the state {state}, what is the optimal action (move, shoot, reload)?"
        return query_mistral(prompt).strip().lower()
    
    def execute_action(self, action):
        if action == "move":
            direction = random.choice(["w", "a", "s", "d"])
            keyboard.press(direction)
            time.sleep(0.2)
            keyboard.release(direction)
        elif action == "shoot" and self.points_manager.spend(50):
            mouse.press()
            time.sleep(0.1)
            mouse.release()
        elif action == "reload" and self.points_manager.spend(30):
            keyboard.press("r")
            time.sleep(1)
            keyboard.release("r")
    
    def run(self):
        print("AI Combat System Activated - Press Ctrl+C to Exit")
        try:
            while True:
                frame = self.capture_screen()
                zombies = self.detect_zombies(frame)
                state = [len(zombies) / 10.0, self.health / 100.0] + [0] * 8
                action = self.mistral_decision(state)
                self.execute_action(action)
                print(f"Action: {action} | Health: {self.health} | Points: {self.points_manager.points}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nAI Combat System Deactivated")

if __name__ == "__main__":
    bot = ZombieAI()
    bot.run()

