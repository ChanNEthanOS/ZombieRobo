"""
bot/navigation.py
Handles pathfinding and movement logic using a graph-based approach.
"""

import networkx as nx
import numpy as np
import logging
import os
import json
import time

logger = logging.getLogger("Navigation")

class Navigator:
    def __init__(self, config):
        self.config = config
        self.map_name = config.get('map_name', 'default')
        self.map_graph = None
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False
        self.load_map_data()

    def load_map_data(self):
        map_file = os.path.join("config", "maps", f"{self.map_name}.json")
        if not os.path.exists(map_file):
            logger.warning(f"Map file not found: {map_file}")
            self.create_default_map()
            return

        try:
            with open(map_file, 'r') as f:
                map_data = json.load(f)
            G = nx.Graph()
            for node_id, node_data in map_data.get('nodes', {}).items():
                G.add_node(node_id, pos=(node_data['x'], node_data['y']))
            for edge in map_data.get('edges', []):
                G.add_edge(edge['from'], edge['to'], weight=edge.get('weight', 1))
            self.map_graph = G
            logger.info(f"Loaded map with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Failed to load map: {e}")
            self.create_default_map()

    def create_default_map(self):
        # Basic 5x5 grid if no map is found
        G = nx.Graph()
        size = 5
        for i in range(size):
            for j in range(size):
                node_id = f"{i}_{j}"
                G.add_node(node_id, pos=(i*100, j*100))
                if i > 0:
                    G.add_edge(f"{i}_{j}", f"{i-1}_{j}", weight=1)
                if j > 0:
                    G.add_edge(f"{i}_{j}", f"{i}_{j-1}", weight=1)
        self.map_graph = G
        logger.warning("Created default 5x5 map layout.")

    def navigate_to(self, target_pos):
        # Example: find path from (0,0) node to nearest node of target
        # Implementation depends on how you track the player's node
        pass
