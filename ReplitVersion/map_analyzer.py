#!/usr/bin/env python3
"""
Map Analyzer for COD WaW Zombies Bot
This script helps analyze map layouts and create navigation data for the bot.
It takes screenshots of the game map and allows marking key locations, windows, etc.
"""

import cv2
import numpy as np
import json
import os
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("map_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MapAnalyzer")

class MapAnalyzer:
    """Class for analyzing zombie map layouts and creating navigation data"""
    
    def __init__(self, map_name, image_path=None):
        """Initialize the map analyzer"""
        self.map_name = map_name
        self.nodes = {}
        self.edges = []
        self.training_routes = []
        self.camping_spots = []
        self.weapon_locations = []
        self.perk_locations = []
        self.window_spawns = []
        self.door_costs = []
        
        # Node types and their colors for visualization
        self.node_types = {
            'spawn': (255, 0, 0),       # Red
            'corner': (0, 255, 0),      # Green
            'window': (0, 0, 255),      # Blue
            'barrier': (255, 255, 0),   # Yellow
            'power': (255, 0, 255),     # Magenta
            'weapon': (0, 255, 255),    # Cyan
            'perk': (128, 0, 128),      # Purple
            'mystery_box': (255, 165, 0),# Orange
            'camping_spot': (0, 128, 0), # Dark Green
            'door': (192, 192, 192),    # Silver
            'default': (255, 255, 255)  # White
        }
        
        # Path for map configuration
        self.config_dir = os.path.join('config', 'maps')
        self.config_path = os.path.join(self.config_dir, f"{map_name}.json")
        
        # Load existing config if it exists
        if os.path.exists(self.config_path):
            self.load_config()
        
        # Load map image if provided
        if image_path and os.path.exists(image_path):
            self.map_image = cv2.imread(image_path)
            if self.map_image is None:
                logger.error(f"Failed to load map image: {image_path}")
                self.map_image = np.zeros((600, 800, 3), dtype=np.uint8)
        else:
            # Create a blank image
            self.map_image = np.zeros((600, 800, 3), dtype=np.uint8)
            
        # For tracking GUI state
        self.current_node_type = 'default'
        self.selected_nodes = []
        self.is_drawing_edge = False
        self.is_drawing_route = False
        self.current_route = []
        self.drawing_state = 'none'  # 'node', 'edge', 'route', etc.
        
    def load_config(self):
        """Load existing map configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            # Load basic properties
            self.map_name = config.get('name', self.map_name)
            self.nodes = config.get('nodes', {})
            self.edges = config.get('edges', [])
            self.training_routes = config.get('training_routes', [])
            self.camping_spots = config.get('camping_spots', [])
            self.weapon_locations = config.get('weapon_locations', [])
            self.perk_locations = config.get('perk_locations', [])
            self.window_spawns = config.get('window_spawns', [])
            self.door_costs = config.get('door_costs', [])
            
            logger.info(f"Loaded map config with {len(self.nodes)} nodes and {len(self.edges)} edges")
        except Exception as e:
            logger.error(f"Error loading map config: {e}")
    
    def save_config(self):
        """Save map configuration to JSON file"""
        try:
            # Ensure config directory exists
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Create config dictionary
            config = {
                'name': self.map_name,
                'description': f"{self.map_name} map layout for CoD WaW Zombies",
                'nodes': self.nodes,
                'edges': self.edges,
                'training_routes': self.training_routes,
                'camping_spots': self.camping_spots,
                'weapon_locations': self.weapon_locations,
                'perk_locations': self.perk_locations,
                'window_spawns': self.window_spawns,
                'door_costs': self.door_costs
            }
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved map config to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving map config: {e}")
            return False
    
    def add_node(self, x, y, node_type='default', properties=None):
        """
        Add a node to the map
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            node_type (str): Type of node (spawn, corner, window, etc.)
            properties (dict, optional): Additional properties
        
        Returns:
            str: Node ID
        """
        # Generate node ID based on type and count
        node_count = sum(1 for node_id in self.nodes if node_id.startswith(node_type))
        node_id = f"{node_type}_{node_count + 1}"
        
        # Create node data
        node_data = {
            'x': x,
            'y': y,
            'type': node_type
        }
        
        # Add additional properties if provided
        if properties:
            node_data.update(properties)
            
        # Add to nodes dictionary
        self.nodes[node_id] = node_data
        
        logger.info(f"Added node {node_id} at ({x}, {y})")
        return node_id
    
    def add_edge(self, from_node, to_node, weight=1.0, edge_type='path'):
        """
        Add an edge between two nodes
        
        Args:
            from_node (str): ID of starting node
            to_node (str): ID of ending node
            weight (float): Edge weight (distance/cost)
            edge_type (str): Type of edge (path, door, etc.)
            
        Returns:
            bool: True if added successfully
        """
        # Check if nodes exist
        if from_node not in self.nodes or to_node not in self.nodes:
            logger.error(f"Cannot add edge: node not found ({from_node} or {to_node})")
            return False
            
        # Check if edge already exists
        for edge in self.edges:
            if edge['from'] == from_node and edge['to'] == to_node:
                logger.warning(f"Edge already exists between {from_node} and {to_node}")
                return False
                
        # Add edge
        edge = {
            'from': from_node,
            'to': to_node,
            'weight': weight,
            'type': edge_type
        }
        self.edges.append(edge)
        
        logger.info(f"Added edge from {from_node} to {to_node} with weight {weight}")
        return True
    
    def add_training_route(self, node_ids):
        """
        Add a training route (sequence of nodes for zombie training)
        
        Args:
            node_ids (list): List of node IDs in the route
            
        Returns:
            bool: True if added successfully
        """
        # Validate nodes
        for node_id in node_ids:
            if node_id not in self.nodes:
                logger.error(f"Cannot add training route: node {node_id} not found")
                return False
                
        # Add route
        self.training_routes.append(node_ids)
        
        logger.info(f"Added training route with {len(node_ids)} nodes")
        return True
    
    def add_camping_spot(self, node_id, rating=3, escape_route=None):
        """
        Add a camping spot (good defensive position)
        
        Args:
            node_id (str): ID of node
            rating (int): Rating from 1-5 (5 best)
            escape_route (list, optional): List of node IDs for escape route
            
        Returns:
            bool: True if added successfully
        """
        # Validate node
        if node_id not in self.nodes:
            logger.error(f"Cannot add camping spot: node {node_id} not found")
            return False
            
        # Validate escape route if provided
        if escape_route:
            for node_id in escape_route:
                if node_id not in self.nodes:
                    logger.error(f"Cannot add camping spot: escape route node {node_id} not found")
                    return False
                    
        # Add camping spot
        spot = {
            'node': node_id,
            'rating': rating,
            'escape_route': escape_route or []
        }
        self.camping_spots.append(spot)
        
        logger.info(f"Added camping spot at {node_id} with rating {rating}")
        return True
    
    def add_weapon_location(self, node_id, weapon, cost):
        """
        Add a wall weapon location
        
        Args:
            node_id (str): ID of node
            weapon (str): Weapon name
            cost (int): Cost in points
            
        Returns:
            bool: True if added successfully
        """
        # Validate node
        if node_id not in self.nodes:
            logger.error(f"Cannot add weapon location: node {node_id} not found")
            return False
            
        # Add weapon location
        weapon_loc = {
            'node': node_id,
            'weapon': weapon,
            'cost': cost
        }
        self.weapon_locations.append(weapon_loc)
        
        logger.info(f"Added {weapon} at {node_id} for {cost} points")
        return True
    
    def add_perk_location(self, node_id, perk, cost):
        """
        Add a perk machine location
        
        Args:
            node_id (str): ID of node
            perk (str): Perk name
            cost (int): Cost in points
            
        Returns:
            bool: True if added successfully
        """
        # Validate node
        if node_id not in self.nodes:
            logger.error(f"Cannot add perk location: node {node_id} not found")
            return False
            
        # Add perk location
        perk_loc = {
            'node': node_id,
            'perk': perk,
            'cost': cost
        }
        self.perk_locations.append(perk_loc)
        
        logger.info(f"Added {perk} at {node_id} for {cost} points")
        return True
    
    def add_window_spawn(self, node_id, difficulty=1):
        """
        Add a window spawn location
        
        Args:
            node_id (str): ID of node
            difficulty (int): Difficulty level (1-5)
            
        Returns:
            bool: True if added successfully
        """
        # Validate node
        if node_id not in self.nodes:
            logger.error(f"Cannot add window spawn: node {node_id} not found")
            return False
            
        # Add window spawn
        window = {
            'node': node_id,
            'difficulty': difficulty
        }
        self.window_spawns.append(window)
        
        logger.info(f"Added window spawn at {node_id} with difficulty {difficulty}")
        return True
    
    def add_door_cost(self, node_id, cost):
        """
        Add a door cost
        
        Args:
            node_id (str): ID of node
            cost (int): Cost in points
            
        Returns:
            bool: True if added successfully
        """
        # Validate node
        if node_id not in self.nodes:
            logger.error(f"Cannot add door cost: node {node_id} not found")
            return False
            
        # Add door cost
        door = {
            'node': node_id,
            'cost': cost
        }
        self.door_costs.append(door)
        
        logger.info(f"Added door at {node_id} with cost {cost} points")
        return True
    
    def visualize_map(self):
        """
        Visualize the map with nodes and edges
        
        Returns:
            numpy.ndarray: Image with map visualization
        """
        # Create a copy of the map image
        vis_img = self.map_image.copy()
        
        # Draw edges
        for edge in self.edges:
            from_node = self.nodes[edge['from']]
            to_node = self.nodes[edge['to']]
            
            start_point = (from_node['x'], from_node['y'])
            end_point = (to_node['x'], to_node['y'])
            
            # Draw line
            cv2.line(vis_img, start_point, end_point, (200, 200, 200), 2)
            
            # Draw weight
            mid_point = ((start_point[0] + end_point[0]) // 2, 
                         (start_point[1] + end_point[1]) // 2)
            cv2.putText(vis_img, f"{edge['weight']}", mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw nodes
        for node_id, node in self.nodes.items():
            # Get node color based on type
            node_type = node['type']
            color = self.node_types.get(node_type, self.node_types['default'])
            
            # Draw circle for node
            cv2.circle(vis_img, (node['x'], node['y']), 5, color, -1)
            
            # Draw node ID
            cv2.putText(vis_img, node_id, (node['x'] + 5, node['y'] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw training routes
        for i, route in enumerate(self.training_routes):
            for j in range(len(route) - 1):
                from_node = self.nodes[route[j]]
                to_node = self.nodes[route[j + 1]]
                
                start_point = (from_node['x'], from_node['y'])
                end_point = (to_node['x'], to_node['y'])
                
                # Draw arrow for route
                cv2.arrowedLine(vis_img, start_point, end_point, 
                               (0, 255, 0), 1, tipLength=0.03)
        
        return vis_img
    
    def create_graph(self):
        """
        Create a networkx graph from nodes and edges
        
        Returns:
            networkx.Graph: Graph representation of the map
        """
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, pos=(node['x'], node['y']), type=node['type'])
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge['from'], edge['to'], weight=edge['weight'], type=edge['type'])
        
        return G
    
    def generate_path_image(self, path):
        """
        Generate an image showing a path through the map
        
        Args:
            path (list): List of node IDs
            
        Returns:
            numpy.ndarray: Image with path visualization
        """
        # Create base visualization
        vis_img = self.visualize_map()
        
        # Draw path
        for i in range(len(path) - 1):
            from_node = self.nodes[path[i]]
            to_node = self.nodes[path[i + 1]]
            
            start_point = (from_node['x'], from_node['y'])
            end_point = (to_node['x'], to_node['y'])
            
            # Draw thick line for path
            cv2.line(vis_img, start_point, end_point, (0, 255, 255), 3)
            
            # Draw direction arrow
            mid_point = ((start_point[0] + end_point[0]) // 2, 
                         (start_point[1] + end_point[1]) // 2)
            cv2.arrowedLine(vis_img, start_point, end_point, 
                           (255, 128, 0), 2, tipLength=0.03)
            
        # Highlight start and end nodes
        start_node = self.nodes[path[0]]
        end_node = self.nodes[path[-1]]
        
        cv2.circle(vis_img, (start_node['x'], start_node['y']), 8, (0, 255, 0), -1)
        cv2.circle(vis_img, (end_node['x'], end_node['y']), 8, (0, 0, 255), -1)
        
        return vis_img
    
    def run_interactive_editor(self):
        """Run the interactive map editor"""
        # Create window
        cv2.namedWindow('Map Editor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Map Editor', 800, 600)
        
        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self._handle_left_click(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._handle_right_click(x, y)
                
        # Set mouse callback
        cv2.setMouseCallback('Map Editor', mouse_callback)
        
        # Main loop
        while True:
            # Generate visualization
            vis_img = self.visualize_map()
            
            # Add UI elements
            self._add_ui_elements(vis_img)
            
            # Show image
            cv2.imshow('Map Editor', vis_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on ESC
            if key == 27:
                break
                
            # Save on 's'
            elif key == ord('s'):
                self.save_config()
                
            # Handle other keys
            self._handle_keyboard(key)
        
        # Clean up
        cv2.destroyAllWindows()
    
    def _handle_left_click(self, x, y):
        """Handle left mouse button click"""
        if self.drawing_state == 'node':
            # Add a node at click position
            self.add_node(x, y, self.current_node_type)
        elif self.drawing_state == 'edge':
            # Select node for edge
            node_id = self._find_node_at_position(x, y)
            if node_id:
                if not self.selected_nodes:
                    # First node
                    self.selected_nodes.append(node_id)
                else:
                    # Second node, create edge
                    self.add_edge(self.selected_nodes[0], node_id)
                    self.selected_nodes = []
        elif self.drawing_state == 'route':
            # Add node to training route
            node_id = self._find_node_at_position(x, y)
            if node_id:
                self.current_route.append(node_id)
    
    def _handle_right_click(self, x, y):
        """Handle right mouse button click"""
        # Cancel current operation
        self.selected_nodes = []
        
        if self.drawing_state == 'route' and self.current_route:
            # Finish route
            if len(self.current_route) > 1:
                self.add_training_route(self.current_route)
            self.current_route = []
    
    def _handle_keyboard(self, key):
        """Handle keyboard input"""
        # Node type selection
        if key == ord('1'):
            self.current_node_type = 'spawn'
            self.drawing_state = 'node'
        elif key == ord('2'):
            self.current_node_type = 'corner'
            self.drawing_state = 'node'
        elif key == ord('3'):
            self.current_node_type = 'window'
            self.drawing_state = 'node'
        elif key == ord('4'):
            self.current_node_type = 'barrier'
            self.drawing_state = 'node'
        elif key == ord('5'):
            self.current_node_type = 'power'
            self.drawing_state = 'node'
        elif key == ord('6'):
            self.current_node_type = 'weapon'
            self.drawing_state = 'node'
        elif key == ord('7'):
            self.current_node_type = 'perk'
            self.drawing_state = 'node'
        elif key == ord('8'):
            self.current_node_type = 'mystery_box'
            self.drawing_state = 'node'
        elif key == ord('9'):
            self.current_node_type = 'camping_spot'
            self.drawing_state = 'node'
        elif key == ord('0'):
            self.current_node_type = 'door'
            self.drawing_state = 'node'
            
        # Drawing mode selection
        elif key == ord('n'):
            self.drawing_state = 'node'
            self.selected_nodes = []
        elif key == ord('e'):
            self.drawing_state = 'edge'
            self.selected_nodes = []
        elif key == ord('r'):
            self.drawing_state = 'route'
            self.current_route = []
        elif key == ord('c'):
            self.drawing_state = 'none'
            self.selected_nodes = []
            self.current_route = []
    
    def _add_ui_elements(self, img):
        """Add UI elements to the visualization"""
        # Add mode indicator
        cv2.putText(img, f"Mode: {self.drawing_state.capitalize()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        # Add node type indicator if in node mode
        if self.drawing_state == 'node':
            cv2.putText(img, f"Node Type: {self.current_node_type}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
        # Add selected nodes indicator if in edge mode
        if self.drawing_state == 'edge' and self.selected_nodes:
            cv2.putText(img, f"Selected: {self.selected_nodes[0]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
        # Add route indicator if in route mode
        if self.drawing_state == 'route' and self.current_route:
            route_str = '->'.join(self.current_route)
            cv2.putText(img, f"Route: {route_str}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                       
        # Add help text
        cv2.putText(img, "Keys: 1-0=Node Types, n=Node Mode, e=Edge Mode, r=Route Mode, s=Save", 
                   (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _find_node_at_position(self, x, y, max_distance=10):
        """
        Find a node at or near the given position
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            max_distance (int): Maximum distance to consider
            
        Returns:
            str: Node ID or None if not found
        """
        closest_node = None
        closest_distance = float('inf')
        
        for node_id, node in self.nodes.items():
            distance = np.sqrt((node['x'] - x)**2 + (node['y'] - y)**2)
            
            if distance < max_distance and distance < closest_distance:
                closest_node = node_id
                closest_distance = distance
                
        return closest_node
    
    def find_path(self, start_node, end_node):
        """
        Find the shortest path between two nodes
        
        Args:
            start_node (str): Starting node ID
            end_node (str): Ending node ID
            
        Returns:
            list: List of node IDs representing the path
        """
        # Create graph
        G = self.create_graph()
        
        try:
            # Find shortest path
            path = nx.shortest_path(G, start_node, end_node, weight='weight')
            logger.info(f"Found path from {start_node} to {end_node}: {path}")
            return path
        except nx.NetworkXNoPath:
            logger.error(f"No path found from {start_node} to {end_node}")
            return []
        except nx.NodeNotFound:
            logger.error(f"Node not found in graph")
            return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Map Analyzer for COD WaW Zombies Bot')
    parser.add_argument('map_name', type=str, help='Map name (e.g., nacht_der_untoten)')
    parser.add_argument('--image', type=str, help='Path to map image (optional)')
    parser.add_argument('--test', action='store_true', help='Run test instead of editor')
    args = parser.parse_args()
    
    analyzer = MapAnalyzer(args.map_name, args.image)
    
    if args.test:
        # Run basic test
        if not analyzer.nodes:
            # Add some test nodes
            analyzer.add_node(100, 100, 'spawn')
            analyzer.add_node(200, 200, 'corner')
            analyzer.add_node(300, 150, 'window')
            analyzer.add_node(400, 400, 'corner')
            
            # Add some test edges
            analyzer.add_edge('spawn_1', 'corner_1', 1.0)
            analyzer.add_edge('corner_1', 'window_1', 1.5)
            analyzer.add_edge('window_1', 'corner_2', 2.0)
            analyzer.add_edge('corner_1', 'corner_2', 2.5)
            
            # Add a test training route
            analyzer.add_training_route(['spawn_1', 'corner_1', 'window_1', 'corner_2', 'corner_1'])
            
            # Save test config
            analyzer.save_config()
        
        # Find a test path
        if len(analyzer.nodes) >= 2:
            node_ids = list(analyzer.nodes.keys())
            path = analyzer.find_path(node_ids[0], node_ids[-1])
            
            # Visualize the path
            if path:
                path_img = analyzer.generate_path_image(path)
                
                # Show the path
                cv2.namedWindow('Test Path', cv2.WINDOW_NORMAL)
                cv2.imshow('Test Path', path_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        # Run interactive editor
        analyzer.run_interactive_editor()

if __name__ == "__main__":
    main()