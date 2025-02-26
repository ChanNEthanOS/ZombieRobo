"""
Debug visualization and logging module for the COD WaW Zombies Bot.
This module handles visualization and console output for debugging.
"""

import cv2
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import io
import threading
import queue
import sys

logger = logging.getLogger("Debug")

class DebugInterface:
    """Class for debug visualization and logging"""
    
    def __init__(self):
        """Initialize the debug interface"""
        # Display settings
        self.display_scale = 0.7  # Scale factor for display
        self.show_visualization = True
        self.show_console = True
        
        # Performance tracking
        self.fps_history = []
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.current_fps = 0
        
        # Create windows for visualization
        if self.show_visualization:
            cv2.namedWindow('Game View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Game View', int(1920 * self.display_scale), int(1080 * self.display_scale))
        
        # Console output queue and thread
        self.console_queue = queue.Queue()
        if self.show_console:
            self.console_thread = threading.Thread(target=self._console_worker)
            self.console_thread.daemon = True
            self.console_thread.start()
        
        logger.info("Debug interface initialized")
    
    def display(self, frame, zombies=None, health=None, ammo=None, game_state=None, action=None):
        """
        Display the current game frame with debug overlays
        
        Args:
            frame (numpy.ndarray): Current game frame
            zombies (list, optional): Detected zombies
            health (int, optional): Current health
            ammo (int, optional): Current ammo
            game_state (GameState, optional): Current game state
            action (dict, optional): Current action being executed
        """
        # Calculate FPS
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_frame_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_frame_time)
            self.fps_history.append(self.current_fps)
            self.frame_count = 0
            self.last_frame_time = current_time
            
            # Keep history from growing too large
            if len(self.fps_history) > 100:
                self.fps_history = self.fps_history[-100:]
        
        # Create a copy of the frame for visualization
        display_frame = frame.copy()
        
        # Add zombie bounding boxes
        if zombies:
            for zombie in zombies:
                x, y, w, h = zombie['x'], zombie['y'], zombie['width'], zombie['height']
                confidence = zombie.get('confidence', 0)
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Add confidence label
                label = f"{confidence:.2f}"
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add HUD info
        if health is not None:
            cv2.putText(display_frame, f"Health: {health}%", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if ammo is not None:
            cv2.putText(display_frame, f"Ammo: {ammo}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add FPS counter
        cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add game state info
        if game_state:
            # Add round info
            cv2.putText(display_frame, f"Round: {game_state.current_round}", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add danger level
            color = (0, 255, 0)  # Green for low danger
            if game_state.danger_level > 7:
                color = (0, 0, 255)  # Red for high danger
            elif game_state.danger_level > 3:
                color = (0, 255, 255)  # Yellow for medium danger
                
            cv2.putText(display_frame, f"Danger: {game_state.danger_level:.1f}/10", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Indicate if surrounded
            if game_state.is_surrounded:
                cv2.putText(display_frame, "SURROUNDED!", (display_frame.shape[1]//2 - 150, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Add current action
        if action:
            cv2.putText(display_frame, f"Action: {action['type']}", (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show visualization
        if self.show_visualization:
            cv2.imshow('Game View', display_frame)
            cv2.waitKey(1)
        
        # Update console
        if self.show_console and game_state:
            state_summary = game_state.get_state_summary()
            action_str = action['type'] if action else "none"
            
            console_text = [
                f"Health: {state_summary['health']}% | Ammo: {state_summary['ammo']} | Round: {state_summary['round']}",
                f"Zombies: {state_summary['zombies']} | Danger: {state_summary['danger_level']}",
                f"Action: {action_str} | FPS: {self.current_fps:.1f}"
            ]
            
            self.console_queue.put('\n'.join(console_text))
    
    def _console_worker(self):
        """Worker thread for updating console output"""
        while True:
            try:
                text = self.console_queue.get(timeout=1)
                
                # Clear the line and write new content
                sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear line
                sys.stdout.write(text)
                sys.stdout.flush()
                
                self.console_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Console worker error: {e}")
                break
    
    def plot_fps_history(self):
        """Plot FPS history using matplotlib"""
        if not self.fps_history:
            return
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.fps_history)
        plt.title('FPS History')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        plt.grid(True)
        
        # Save to buffer instead of file
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to OpenCV image
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # Display image
        cv2.imshow('FPS History', img)
        cv2.waitKey(1)
        
        plt.close()
    
    def close(self):
        """Clean up resources"""
        if self.show_visualization:
            cv2.destroyAllWindows()
        
        # Add final newline to console
        if self.show_console:
            sys.stdout.write('\n')
            sys.stdout.flush()
        
        logger.info("Debug interface closed")
