"""
Data management package for the COD WaW Zombies Bot.
This package handles dataset creation and management for model training.
"""

__all__ = ['ZombieDataset', 'create_dataset', 'export_annotations']

from data.dataset import ZombieDataset, create_dataset
from data.annotations import export_annotations
