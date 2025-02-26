"""
Dataset management module for the COD WaW Zombies Bot.
This module handles the creation and management of datasets for model training.
"""

import os
import cv2
import numpy as np
import logging
import random
import shutil
import glob
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logger = logging.getLogger("Dataset")

class ZombieDataset:
    """Class for managing the zombie detection dataset"""
    
    def __init__(self, data_dir='data/zombies'):
        """
        Initialize the dataset manager
        
        Args:
            data_dir (str): Base directory for the dataset
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        self.classes = ['zombie', 'crawler', 'hellhound', 'boss']
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Dataset statistics
        self.image_count = 0
        self.annotation_count = 0
        self.class_distribution = {cls: 0 for cls in self.classes}
        
        # Load dataset stats
        self._load_dataset_stats()
        
        logger.info(f"ZombieDataset initialized with {self.image_count} images and {self.annotation_count} annotations")
    
    def _create_directories(self):
        """Create necessary directories for the dataset"""
        for directory in [self.data_dir, self.images_dir, self.labels_dir, 
                         self.train_dir / 'images', self.train_dir / 'labels',
                         self.val_dir / 'images', self.val_dir / 'labels']:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_dataset_stats(self):
        """Load and calculate dataset statistics"""
        # Count images
        self.image_count = len(list(self.images_dir.glob('*.jpg'))) + len(list(self.images_dir.glob('*.png')))
        
        # Count annotations
        self.annotation_count = len(list(self.labels_dir.glob('*.txt')))
        
        # Calculate class distribution
        for label_file in self.labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            if 0 <= class_id < len(self.classes):
                                self.class_distribution[self.classes[class_id]] += 1
            except Exception as e:
                logger.warning(f"Error reading label file {label_file}: {e}")
    
    def add_image(self, image, annotations=None, image_name=None):
        """
        Add an image and its annotations to the dataset
        
        Args:
            image (numpy.ndarray): Image data
            annotations (list): List of annotation dictionaries with class_id, x, y, width, height
            image_name (str, optional): Name for the image file
            
        Returns:
            tuple: (image_path, label_path) of saved files
        """
        # Generate a filename if not provided
        if image_name is None:
            image_name = f"zombie_{self.image_count + 1:06d}"
        
        # Ensure filename has no extension
        image_name = os.path.splitext(image_name)[0]
        
        # Save the image
        image_path = self.images_dir / f"{image_name}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save annotations if provided
        label_path = None
        if annotations:
            label_path = self.labels_dir / f"{image_name}.txt"
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # Convert to YOLO format (class_id, x_center, y_center, width, height)
                    # All values normalized to 0-1
                    img_height, img_width = image.shape[:2]
                    
                    class_id = ann['class_id']
                    x_center = (ann['x'] + ann['width'] / 2) / img_width
                    y_center = (ann['y'] + ann['height'] / 2) / img_height
                    width = ann['width'] / img_width
                    height = ann['height'] / img_height
                    
                    # Write in YOLO format
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    
                    # Update class distribution
                    if 0 <= class_id < len(self.classes):
                        self.class_distribution[self.classes[class_id]] += 1
        
        # Update stats
        self.image_count += 1
        if label_path:
            self.annotation_count += 1
        
        logger.debug(f"Added image {image_name} to dataset")
        return str(image_path), str(label_path) if label_path else None
    
    def split_dataset(self, val_ratio=0.2, shuffle=True):
        """
        Split the dataset into training and validation sets
        
        Args:
            val_ratio (float): Ratio of validation set (0-1)
            shuffle (bool): Whether to shuffle the data before splitting
            
        Returns:
            dict: Split statistics
        """
        # Get all image files
        image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        
        # Get corresponding label files
        valid_images = []
        for img_file in image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_images.append(img_file)
        
        # Split the dataset
        train_images, val_images = train_test_split(
            valid_images, test_size=val_ratio, random_state=42 if shuffle else None
        )
        
        # Clear existing train/val directories
        for dir_path in [self.train_dir / 'images', self.train_dir / 'labels',
                        self.val_dir / 'images', self.val_dir / 'labels']:
            for file in dir_path.glob('*'):
                file.unlink()
        
        # Copy to train directory
        for img_file in tqdm(train_images, desc="Copying training data"):
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            
            # Copy image
            shutil.copy(str(img_file), str(self.train_dir / 'images' / img_file.name))
            
            # Copy label
            if label_file.exists():
                shutil.copy(str(label_file), str(self.train_dir / 'labels' / label_file.name))
        
        # Copy to validation directory
        for img_file in tqdm(val_images, desc="Copying validation data"):
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            
            # Copy image
            shutil.copy(str(img_file), str(self.val_dir / 'images' / img_file.name))
            
            # Copy label
            if label_file.exists():
                shutil.copy(str(label_file), str(self.val_dir / 'labels' / label_file.name))
        
        # Create dataset YAML file
        yaml_content = f"""
# COD WaW Zombies YOLO Dataset
path: {self.data_dir.absolute()}
train: {(self.train_dir).absolute()}
val: {(self.val_dir).absolute()}

# Classes
nc: {len(self.classes)}  # number of classes
names: {self.classes}  # class names
"""
        
        with open(self.data_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Dataset split complete. Train: {len(train_images)}, Val: {len(val_images)}")
        
        return {
            'train_count': len(train_images),
            'val_count': len(val_images),
            'yaml_path': str(self.data_dir / 'dataset.yaml')
        }
    
    def augment_data(self, augmentation_factor=2):
        """
        Augment the dataset with transformed versions of existing images
        
        Args:
            augmentation_factor (int): Number of augmented samples per original image
            
        Returns:
            int: Number of augmented images created
        """
        augmented_count = 0
        
        # Get all valid image files with labels
        image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        valid_images = []
        
        for img_file in image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_images.append((img_file, label_file))
        
        logger.info(f"Augmenting {len(valid_images)} images with factor {augmentation_factor}")
        
        for img_file, label_file in tqdm(valid_images, desc="Augmenting data"):
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    logger.warning(f"Failed to load image: {img_file}")
                    continue
                
                # Load annotations
                with open(label_file, 'r') as f:
                    annotations = [line.strip() for line in f if line.strip()]
                
                # Create augmented versions
                for i in range(augmentation_factor):
                    # Apply random transformations
                    aug_image, aug_annotations = self._apply_augmentation(image, annotations)
                    
                    # Save augmented image and annotations
                    aug_name = f"{img_file.stem}_aug_{i+1}"
                    aug_img_path = self.images_dir / f"{aug_name}.jpg"
                    aug_label_path = self.labels_dir / f"{aug_name}.txt"
                    
                    # Save image
                    cv2.imwrite(str(aug_img_path), aug_image)
                    
                    # Save annotations
                    with open(aug_label_path, 'w') as f:
                        for ann in aug_annotations:
                            f.write(f"{ann}\n")
                    
                    augmented_count += 1
                
            except Exception as e:
                logger.error(f"Error augmenting image {img_file}: {e}")
        
        # Update dataset stats
        self._load_dataset_stats()
        
        logger.info(f"Data augmentation complete. Created {augmented_count} new images.")
        return augmented_count
    
    def _apply_augmentation(self, image, annotations):
        """
        Apply random augmentation to an image and update annotations
        
        Args:
            image (numpy.ndarray): Original image
            annotations (list): List of annotation strings in YOLO format
            
        Returns:
            tuple: (augmented_image, augmented_annotations)
        """
        height, width = image.shape[:2]
        
        # Choose random augmentations
        # 1. Horizontal flip (50% chance)
        do_hflip = random.random() > 0.5
        
        # 2. Brightness/contrast adjustment
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        
        # 3. Small rotation (±15 degrees)
        angle = random.uniform(-15, 15)
        
        # 4. Small scale jitter
        scale = random.uniform(0.8, 1.2)
        
        # Apply transformations
        augmented_image = image.copy()
        
        # Brightness/contrast
        augmented_image = cv2.convertScaleAbs(augmented_image, alpha=contrast, beta=brightness * 10)
        
        # Horizontal flip
        if do_hflip:
            augmented_image = cv2.flip(augmented_image, 1)
        
        # Rotation and scaling
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        augmented_image = cv2.warpAffine(augmented_image, M, (width, height))
        
        # Update annotations
        augmented_annotations = []
        for ann in annotations:
            parts = ann.split()
            class_id = parts[0]
            coords = [float(x) for x in parts[1:]]
            
            # Original coordinates in YOLO format (centerX, centerY, width, height)
            x_center, y_center, w, h = coords
            
            # Process horizontal flip
            if do_hflip:
                x_center = 1.0 - x_center
            
            # Process rotation and scaling (more complex)
            # For simplicity, we'll just adjust slightly based on the transformation
            # A more accurate approach would transform each bounding box corner
            if abs(angle) > 5 or abs(scale - 1.0) > 0.1:
                # Adjust box size slightly to account for rotation
                w *= scale * (1.0 - abs(angle/90))
                h *= scale * (1.0 - abs(angle/90))
                
                # Keep the center within bounds
                x_center = min(max(x_center, 0.05), 0.95)
                y_center = min(max(y_center, 0.05), 0.95)
            
            # Create new annotation
            new_ann = f"{class_id} {x_center} {y_center} {w} {h}"
            augmented_annotations.append(new_ann)
        
        return augmented_image, augmented_annotations
    
    def get_dataset_stats(self):
        """
        Get statistics about the dataset
        
        Returns:
            dict: Dataset statistics
        """
        return {
            'image_count': self.image_count,
            'annotation_count': self.annotation_count,
            'class_distribution': self.class_distribution,
            'train_images': len(list((self.train_dir / 'images').glob('*.jpg'))),
            'val_images': len(list((self.val_dir / 'images').glob('*.jpg'))),
            'data_directory': str(self.data_dir)
        }


def create_dataset(source_dir, output_dir='data/zombies', annotation_format='yolo'):
    """
    Create a dataset from source images and annotations
    
    Args:
        source_dir (str): Directory containing source images and annotations
        output_dir (str): Output directory for the dataset
        annotation_format (str): Format of annotations ('yolo', 'voc', 'coco')
        
    Returns:
        ZombieDataset: The created dataset
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return None
    
    # Initialize the dataset
    dataset = ZombieDataset(output_dir)
    
    # Find all images
    image_files = list(source_path.glob('**/*.jpg')) + list(source_path.glob('**/*.png'))
    
    if not image_files:
        logger.warning(f"No images found in {source_dir}")
        return dataset
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Failed to load image: {img_file}")
                continue
            
            # Look for corresponding annotation file
            if annotation_format == 'yolo':
                ann_file = img_file.with_suffix('.txt')
            elif annotation_format == 'voc':
                ann_file = img_file.with_suffix('.xml')
            else:
                # For COCO, expect a single JSON file
                ann_file = source_path / 'annotations.json'
                if not ann_file.exists():
                    logger.warning(f"Annotation file not found for {img_file}")
                    continue
            
            # Parse annotations based on format
            annotations = []
            
            if annotation_format == 'yolo' and ann_file.exists():
                with open(ann_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to pixel coordinates
                            img_height, img_width = image.shape[:2]
                            x = int((x_center - width/2) * img_width)
                            y = int((y_center - height/2) * img_height)
                            w = int(width * img_width)
                            h = int(height * img_height)
                            
                            annotations.append({
                                'class_id': class_id,
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h
                            })
            
            # Add to dataset
            dataset.add_image(image, annotations, img_file.stem)
            
        except Exception as e:
            logger.error(f"Error processing image {img_file}: {e}")
    
    logger.info(f"Dataset creation complete. Added {dataset.image_count} images.")
    return dataset
