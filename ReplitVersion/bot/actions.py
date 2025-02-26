"""
Navigation module for the COD WaW Zombies Bot.
This module handles pathfinding and movement in the game world.
"""

import numpy as np
import networkx as nx
import cv2
import logging
import json
import os
import time

logger = logging.getLogger("Navigation")

class Navigator:
    """Class for handling navigation and pathfinding in the game world"""
    
    def __init__(self, config):
        """
        Initialize the navigator
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Load map data
        self.map_graph = None
        self.map_name = config.get('map_name', 'default')
        self.load_map_data()
        
        # Current position estimation
        self.current_position = None
        self.last_known_position = None
        self.position_confidence = 0
        
        # Pathfinding settings
        self.waypoints = []
        self.current_waypoint_index = 0
        self.path_completion = 0
        
        # Movement parameters
        self.rotation_speed = config.get('rotation_speed', 10)
        self.movement_smoothing = config.get('movement_smoothing', 0.5)
        
        # Navigation state
        self.is_navigating = False
        self.navigation_start_time = 0
        self.obstacle_detected = False
        self.last_position_update = time.time()
        
        logger.info(f"Navigator initialized for map: {self.map_name}")
    
    def load_map_data(self):
        """Load map graph and waypoints from config files"""
        try:
            map_file = f"config/maps/{self.map_name}.json"
            if os.path.exists(map_file):
                with open(map_file, 'r') as f:
                    map_data = json.load(f)
                
                # Create graph from nodes and edges
                G = nx.Graph()
                
                # Add nodes
                for node_id, node_data in map_data.get('nodes', {}).items():
                    G.add_node(node_id, pos=(node_data['x'], node_data['y']), 
                              type=node_data.get('type', 'normal'))
                
                # Add edges
                for edge in map_data.get('edges', []):
                    G.add_edge(edge['from'], edge['to'], 
                              weight=edge.get('weight', 1),
                              type=edge.get('type', 'normal'))
                
                self.map_graph = G
                logger.info(f"Loaded map data with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            else:
                logger.warning(f"Map file not found: {map_file}")
                self.create_default_map()
        except Exception as e:
            logger.error(f"Failed to load map data: {e}")
            self.create_default_map()
    
    def create_default_map(self):
        """Create a simple default map if no map data is available"""
        G = nx.Graph()
        
        # Create a simple grid layout
        size = 5
        for i in range(size):
            for j in range(size):
                node_id = f"{i}_{j}"
                G.add_node(node_id, pos=(i*100, j*100), type='normal')
                
                # Connect to adjacent nodes
                if i > 0:
                    G.add_edge(f"{i}_{j}", f"{i-1}_{j}", weight=1, type='normal')
                if j > 0:
                    G.add_edge(f"{i}_{j}", f"{i}_{j-1}", weight=1, type='normal')
        
        self.map_graph = G
        logger.warning("Created default map with grid layout")
    
    def estimate_position(self, frame, zombies):
        """
        Estimate the player's position in the map
        
        Args:
            frame (numpy.ndarray): Current game frame
            zombies (list): Detected zombies in the frame
            
        Returns:
            tuple: (x, y) coordinates of estimated position
        """
        # This is a placeholder for actual position estimation
        # In a real implementation, this would use visual landmarks, checkpoints, etc.
        
        # For now, we'll just use a simple time-based position estimation
        # and pretend we're moving around the map
        
        current_time = time.time()
        if self.current_position is None:
            # Initialize to a starting position
            self.current_position = (100, 100)
            self.position_confidence = 0.7
        else:
            # Update position with some random movement to simulate navigation
            # Only update every few seconds to avoid constant changes
            if current_time - self.last_position_update > 5:
                dx = np.random.randint(-50, 50)
                dy = np.random.randint(-50, 50)
                self.current_position = (
                    max(0, min(500, self.current_position[0] + dx)),
                    max(0, min(500, self.current_position[1] + dy))
                )
                self.last_position_update = current_time
                
                # Fluctuate confidence
                self.position_confidence = min(0.9, max(0.3, self.position_confidence + np.random.uniform(-0.1, 0.1)))
        
        self.last_known_position = self.current_position
        return self.current_position
    
    def find_path(self, start_node, end_node):
        """
        Find a path between two nodes in the map
        
        Args:
            start_node (str): Starting node ID
            end_node (str): Target node ID
            
        Returns:
            list: List of node IDs representing the path
        """
        if self.map_graph is None:
            logger.error("No map graph available for pathfinding")
            return []
        
        try:
            if start_node in self.map_graph and end_node in self.map_graph:
                path = nx.shortest_path(self.map_graph, start_node, end_node, weight='weight')
                logger.debug(f"Found path from {start_node} to {end_node}: {path}")
                return path
            else:
                logger.warning(f"Start or end node not in graph: {start_node}, {end_node}")
                return []
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {start_node} and {end_node}")
            return []
        except Exception as e:
            logger.error(f"Pathfinding error: {e}")
            return []
    
    def find_nearest_node(self, position):
        """
        Find the graph node nearest to a position
        
        Args:
            position (tuple): (x, y) coordinates
            
        Returns:
            str: Node ID of the nearest node
        """
        if self.map_graph is None:
            return None
        
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.map_graph.nodes():
            node_pos = self.map_graph.nodes[node]['pos']
            distance = np.sqrt((node_pos[0] - position[0])**2 + (node_pos[1] - position[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def navigate_to(self, target_position):
        """
        Start navigation to a target position
        
        Args:
            target_position (tuple): (x, y) coordinates of the target
            
        Returns:
            bool: True if navigation started, False otherwise
        """
        if self.map_graph is None:
            logger.error("No map graph available for navigation")
            return False
        
        # Find nearest nodes to current and target positions
        current_node = self.find_nearest_node(self.current_position or (100, 100))
        target_node = self.find_nearest_node(target_position)
        
        if current_node is None or target_node is None:
            logger.error("Failed to find valid nodes for navigation")
            return False
        
        # Find path between nodes
        path_nodes = self.find_path(current_node, target_node)
        
        if not path_nodes:
            logger.warning("No path found to target")
            return False
        
        # Convert node path to waypoints
        self.waypoints = [self.map_graph.nodes[node]['pos'] for node in path_nodes]
        self.current_waypoint_index = 0
        self.is_navigating = True
        self.navigation_start_time = time.time()
        self.path_completion = 0
        
        logger.info(f"Started navigation with {len(self.waypoints)} waypoints")
        return True
    
    def navigate_to_safe_area(self, zombies):
        """
        Find and navigate to a safe area away from zombies
        
        Args:
            zombies (list): List of detected zombies
            
        Returns:
            bool: True if navigation started, False otherwise
        """
        if self.map_graph is None or not zombies:
            return False
        
        # Create a heat map of zombie density
        heat_map = np.zeros((500, 500))
        
        for zombie in zombies:
            # Assuming zombie positions are normalized to the map
            x = int(zombie['center_x'] / 5)  # Scale to our map size
            y = int(zombie['center_y'] / 5)
            
            # Add a Gaussian blob around each zombie position
            for i in range(max(0, x-20), min(500, x+20)):
                for j in range(max(0, y-20), min(500, y+20)):
                    distance = np.sqrt((i-x)**2 + (j-y)**2)
                    if distance < 20:
                        heat_map[j, i] += max(0, 20 - distance) / 20
        
        # Find the point with minimum zombie density
        min_val = float('inf')
        safe_pos = None
        
        # Only consider points near known nodes
        for node in self.map_graph.nodes():
            node_pos = self.map_graph.nodes[node]['pos']
            x, y = int(node_pos[0]), int(node_pos[1])
            
            if 0 <= x < 500 and 0 <= y < 500:
                if heat_map[y, x] < min_val:
                    min_val = heat_map[y, x]
                    safe_pos = node_pos
        
        if safe_pos:
            logger.info(f"Found safe position at {safe_pos} with density {min_val}")
            return self.navigate_to(safe_pos)
        else:
            logger.warning("Could not find a safe position")
            return False
    
    def get_next_movement(self):
        """
        Get the next movement action based on the current navigation state
        
        Returns:
            dict: Movement action with keys 'type', 'direction', 'duration'
        """
        if not self.is_navigating or not self.waypoints:
            return {'type': 'none'}
        
        if self.current_waypoint_index >= len(self.waypoints):
            # Navigation complete
            self.is_navigating = False
            return {'type': 'none'}
        
        # Get current waypoint
        target = self.waypoints[self.current_waypoint_index]
        current = self.current_position or (100, 100)
        
        # Calculate direction to waypoint
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Check if we've reached the waypoint
        if distance < 20:
            self.current_waypoint_index += 1
            self.path_completion = min(1.0, self.current_waypoint_index / len(self.waypoints))
            
            if self.current_waypoint_index >= len(self.waypoints):
                logger.info("Navigation completed")
                self.is_navigating = False
                return {'type': 'none'}
            else:
                # Move to next waypoint
                return self.get_next_movement()
        
        # Calculate movement action
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Normalize to 0-360 degrees
        if angle < 0:
            angle += 360
        
        # Determine movement type
        if self.obstacle_detected:
            # If obstacle detected, try to move around it
            return {
                'type': 'strafe',
                'direction': 'right' if np.random.random() > 0.5 else 'left',
                'duration': 0.5
            }
        else:
            # Normal movement towards waypoint
            return {
                'type': 'move',
                'direction': 'forward',
                'angle': angle,
                'duration': min(1.0, distance / 100)  # Limit duration
            }
    
    def detect_obstacles(self, frame):
        """
        Detect obstacles in the game frame
        
        Args:
            frame (numpy.ndarray): Current game frame
            
        Returns:
            bool: True if obstacle detected, False otherwise
        """
        # This is a placeholder for actual obstacle detection
        # In a real implementation, this would use computer vision to detect walls, barriers, etc.
        
        # For now, just randomly set obstacle detection
        self.obstacle_detected = np.random.random() < 0.1
        return self.obstacle_detected
    
    def get_navigation_status(self):
        """
        Get current navigation status
        
        Returns:
            dict: Navigation status information
        """
        return {
            'is_navigating': self.is_navigating,
            'current_position': self.current_position,
            'position_confidence': self.position_confidence,
            'waypoints': len(self.waypoints),
            'current_waypoint': self.current_waypoint_index,
            'progress': f"{self.path_completion:.0%}",
            'obstacle_detected': self.obstacle_detected,
            'time_elapsed': time.time() - self.navigation_start_time if self.is_navigating else 0
        }
