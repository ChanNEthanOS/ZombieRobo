#!/usr/bin/env python3
"""
Zombie Training Dataset Generator for COD WaW Zombies Bot
This script helps collect and prepare training data for the zombie detection model.
"""

import os
import cv2
import numpy as np
import json
import time
import argparse
import logging
import random
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zombie_dataset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ZombieDataset")

class ZombieDatasetGenerator:
    """Class for generating training data for zombie detection"""
    
    def __init__(self, output_dir="data/zombies"):
        """Initialize the dataset generator"""
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        self.label_dir = os.path.join(output_dir, "labels")
        self.class_names = ["zombie", "crawler", "hellhound", "boss"]
        
        # Create directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        
        # Dataset statistics
        self.stats = {
            "total_images": 0,
            "total_annotations": 0,
            "class_distribution": {cls: 0 for cls in self.class_names},
            "image_sizes": []
        }
        
        # Load existing stats if available
        stats_file = os.path.join(output_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing stats: {e}")
    
    def capture_frames(self, duration=60, fps=5, region=None):
        """
        Capture frames from the screen for the dataset
        
        Args:
            duration (int): Duration in seconds
            fps (int): Frames per second to capture
            region (dict): Screen region to capture
            
        Returns:
            int: Number of frames captured
        """
        try:
            import mss
            import time
            
            # Default to full screen if no region specified
            if region is None:
                region = {
                    'top': 0,
                    'left': 0,
                    'width': 1920,
                    'height': 1080
                }
            
            with mss.mss() as sct:
                logger.info(f"Starting screen capture for {duration} seconds at {fps} FPS...")
                
                # Calculate total frames and interval
                total_frames = duration * fps
                interval = 1.0 / fps
                
                # Create progress bar
                pbar = tqdm(total=total_frames, desc="Capturing frames")
                
                frames_captured = 0
                start_time = time.time()
                
                while frames_captured < total_frames:
                    # Capture frame
                    img = np.array(sct.grab(region))
                    
                    # Convert BGRA to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"frame_{timestamp}.jpg"
                    filepath = os.path.join(self.image_dir, filename)
                    
                    cv2.imwrite(filepath, img)
                    frames_captured += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Sleep for interval
                    elapsed = time.time() - start_time
                    expected = frames_captured * interval
                    sleep_time = max(0, expected - elapsed)
                    time.sleep(sleep_time)
                
                pbar.close()
                
                # Update stats
                self.stats["total_images"] += frames_captured
                
                logger.info(f"Captured {frames_captured} frames")
                return frames_captured
                
        except ImportError:
            logger.error("MSS not installed. Cannot capture screen.")
            return 0
        except Exception as e:
            logger.error(f"Error capturing frames: {e}")
            return 0
    
    def annotate_images(self, annotation_tool="auto"):
        """
        Launch an annotation tool for labeling zombies in images
        
        Args:
            annotation_tool (str): Tool to use ('auto', 'opencv', 'labelimg')
            
        Returns:
            int: Number of images annotated
        """
        # Check for un-annotated images
        image_files = [f for f in os.listdir(self.image_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Check for existing annotations
        annotation_files = [f for f in os.listdir(self.label_dir) 
                          if f.endswith('.txt')]
        
        annotation_basenames = [os.path.splitext(f)[0] for f in annotation_files]
        image_basenames = [os.path.splitext(f)[0] for f in image_files]
        
        # Find images without annotations
        to_annotate = [f for f in image_files 
                      if os.path.splitext(f)[0] not in annotation_basenames]
        
        if not to_annotate:
            logger.info("All images have annotations.")
            return 0
            
        logger.info(f"Found {len(to_annotate)} images to annotate.")
        
        if annotation_tool == "opencv" or annotation_tool == "auto":
            return self._opencv_annotator(to_annotate)
        elif annotation_tool == "labelimg":
            return self._labelimg_annotator()
        else:
            logger.error(f"Unknown annotation tool: {annotation_tool}")
            return 0
    
    def _opencv_annotator(self, image_files):
        """
        Simple OpenCV-based annotation tool
        
        Args:
            image_files (list): List of image filenames to annotate
            
        Returns:
            int: Number of images annotated
        """
        annotated_count = 0
        current_image_idx = 0
        current_class_idx = 0
        drawing = False
        rect_start = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, rect_start, current_class_idx, annotations
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                rect_start = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if drawing and rect_start:
                    drawing = False
                    x_min = min(rect_start[0], x)
                    y_min = min(rect_start[1], y)
                    width = abs(x - rect_start[0])
                    height = abs(y - rect_start[1])
                    
                    # Add annotation if it has reasonable size
                    if width > 5 and height > 5:
                        annotations.append({
                            'class_id': current_class_idx,
                            'x': x_min,
                            'y': y_min,
                            'width': width,
                            'height': height
                        })
                        logger.debug(f"Added annotation: class={current_class_idx}, box=({x_min},{y_min},{width},{height})")
        
        # Create window and set mouse callback
        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', mouse_callback)
        
        # Process each image
        while current_image_idx < len(image_files):
            image_file = image_files[current_image_idx]
            image_path = os.path.join(self.image_dir, image_file)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_file}")
                current_image_idx += 1
                continue
            
            # Initialize annotations
            annotations = []
            
            while True:
                # Create display image
                display_img = image.copy()
                
                # Draw existing annotations
                for ann in annotations:
                    x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
                    cls = ann['class_id']
                    
                    # Different color for each class
                    colors = [
                        (0, 0, 255),    # Red for zombies
                        (0, 255, 0),    # Green for crawlers
                        (255, 0, 0),    # Blue for hellhounds
                        (255, 0, 255)   # Magenta for bosses
                    ]
                    color = colors[cls % len(colors)]
                    
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
                    
                    # Add class label
                    cls_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                    cv2.putText(display_img, cls_name, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw current rectangle if drawing
                if drawing and rect_start:
                    cv2.rectangle(
                        display_img, 
                        rect_start, 
                        (cv2.getMousePos()[0], cv2.getMousePos()[1]), 
                        (255, 255, 0), 
                        2
                    )
                
                # Add UI information
                cv2.putText(display_img, f"Image: {current_image_idx+1}/{len(image_files)} - {image_file}", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_img, f"Class: {current_class_idx} ({self.class_names[current_class_idx] if current_class_idx < len(self.class_names) else 'unknown'})", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_img, "Controls: [n]ext, [p]rev, [s]ave, [c]lass+, [d]elete last, [q]uit", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Show the image
                cv2.imshow('Annotator', display_img)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    cv2.destroyAllWindows()
                    return annotated_count
                
                elif key == ord('s'):
                    # Save annotations
                    self._save_annotations(image_file, annotations, image.shape)
                    annotated_count += 1
                    logger.info(f"Saved annotations for {image_file}")
                    break
                    
                elif key == ord('n'):
                    # Next image
                    if annotations:
                        self._save_annotations(image_file, annotations, image.shape)
                        annotated_count += 1
                        logger.info(f"Saved annotations for {image_file}")
                    current_image_idx += 1
                    break
                    
                elif key == ord('p'):
                    # Previous image
                    if annotations:
                        self._save_annotations(image_file, annotations, image.shape)
                        annotated_count += 1
                        logger.info(f"Saved annotations for {image_file}")
                    current_image_idx = max(0, current_image_idx - 1)
                    break
                    
                elif key == ord('c'):
                    # Cycle through classes
                    current_class_idx = (current_class_idx + 1) % len(self.class_names)
                    logger.debug(f"Current class set to {current_class_idx} ({self.class_names[current_class_idx]})")
                    
                elif key == ord('d'):
                    # Delete last annotation
                    if annotations:
                        annotations.pop()
                        logger.debug("Removed last annotation")
        
        cv2.destroyAllWindows()
        return annotated_count
    
    def _labelimg_annotator(self):
        """
        Launch LabelImg for annotation (requires LabelImg installed)
        
        Returns:
            int: Number of images annotated (always 0 since we can't track)
        """
        try:
            # Try to import labelImg
            import subprocess
            
            # Set up command line arguments for labelImg
            subprocess.run(['labelImg', self.image_dir, os.path.join(self.output_dir, 'classes.txt')])
            
            return 0  # We can't track how many were annotated
            
        except ImportError:
            logger.error("LabelImg not installed. Cannot launch annotation tool.")
            return 0
    
    def _save_annotations(self, image_file, annotations, image_shape):
        """
        Save annotations in YOLO format
        
        Args:
            image_file (str): Image filename
            annotations (list): List of annotation dictionaries
            image_shape (tuple): Image shape (height, width, channels)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            height, width = image_shape[:2]
            
            # Create YOLO format annotations
            yolo_annotations = []
            
            for ann in annotations:
                class_id = ann['class_id']
                
                # Convert to YOLO format (normalized)
                x_center = (ann['x'] + ann['width'] / 2) / width
                y_center = (ann['y'] + ann['height'] / 2) / height
                norm_width = ann['width'] / width
                norm_height = ann['height'] / height
                
                # Ensure values are within bounds
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")
            
            # Save to file
            base_name = os.path.splitext(image_file)[0]
            label_file = os.path.join(self.label_dir, f"{base_name}.txt")
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            # Update stats
            self.stats["total_annotations"] += len(annotations)
            
            for ann in annotations:
                class_id = ann['class_id']
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    self.stats["class_distribution"][class_name] = self.stats["class_distribution"].get(class_name, 0) + 1
            
            if (width, height) not in self.stats["image_sizes"]:
                self.stats["image_sizes"].append((width, height))
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving annotations: {e}")
            return False
    
    def prepare_yolo_dataset(self):
        """
        Prepare dataset for YOLO training
        
        Returns:
            bool: True if prepared successfully
        """
        try:
            # Create data YAML file
            data_yaml = {
                "path": os.path.abspath(self.output_dir),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(self.class_names),
                "names": self.class_names
            }
            
            # Create directories
            os.makedirs(os.path.join(self.output_dir, "images", "train"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "images", "val"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "images", "test"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", "train"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", "val"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", "test"), exist_ok=True)
            
            # Write data YAML
            with open(os.path.join(self.output_dir, "data.yaml"), 'w') as f:
                f.write(f"path: {data_yaml['path']}\n")
                f.write(f"train: {data_yaml['train']}\n")
                f.write(f"val: {data_yaml['val']}\n")
                f.write(f"test: {data_yaml['test']}\n")
                f.write(f"nc: {data_yaml['nc']}\n")
                f.write("names: [" + ", ".join([f'"{name}"' for name in data_yaml['names']]) + "]\n")
            
            # Write classes.txt for LabelImg
            with open(os.path.join(self.output_dir, "classes.txt"), 'w') as f:
                f.write('\n'.join(self.class_names))
            
            # Split dataset
            self._split_dataset()
            
            # Save stats
            with open(os.path.join(self.output_dir, "dataset_stats.json"), 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            logger.info(f"Dataset prepared for YOLO training")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing YOLO dataset: {e}")
            return False
    
    def _split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split dataset into train/val/test sets
        
        Args:
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
            test_ratio (float): Ratio for test set
            
        Returns:
            dict: Split statistics
        """
        try:
            # Get all image files that have corresponding label files
            images = []
            for img_file in os.listdir(self.image_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    base_name = os.path.splitext(img_file)[0]
                    label_file = os.path.join(self.label_dir, f"{base_name}.txt")
                    
                    if os.path.exists(label_file):
                        images.append(img_file)
            
            # Shuffle images
            random.shuffle(images)
            
            # Calculate split
            train_size = int(len(images) * train_ratio)
            val_size = int(len(images) * val_ratio)
            
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]
            
            # Move files to split directories
            self._move_files_to_split(train_images, "train")
            self._move_files_to_split(val_images, "val")
            self._move_files_to_split(test_images, "test")
            
            # Update stats
            split_stats = {
                "train": len(train_images),
                "val": len(val_images),
                "test": len(test_images),
                "total": len(images)
            }
            
            logger.info(f"Dataset split: {split_stats}")
            return split_stats
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return {"train": 0, "val": 0, "test": 0, "total": 0}
    
    def _move_files_to_split(self, image_files, split_name):
        """
        Move files to train/val/test directories
        
        Args:
            image_files (list): List of image filenames
            split_name (str): Split name ('train', 'val', 'test')
            
        Returns:
            int: Number of files moved
        """
        for img_file in image_files:
            # Source files
            src_img = os.path.join(self.image_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            src_label = os.path.join(self.label_dir, f"{base_name}.txt")
            
            # Destination files
            dst_img_dir = os.path.join(self.output_dir, "images", split_name)
            dst_label_dir = os.path.join(self.output_dir, "labels", split_name)
            
            dst_img = os.path.join(dst_img_dir, img_file)
            dst_label = os.path.join(dst_label_dir, f"{base_name}.txt")
            
            # Create hard links to save disk space
            try:
                if not os.path.exists(dst_img):
                    os.link(src_img, dst_img)
                if not os.path.exists(dst_label):
                    os.link(src_label, dst_label)
            except OSError:
                # Fall back to copy if link fails
                import shutil
                if not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)
                if not os.path.exists(dst_label):
                    shutil.copy2(src_label, dst_label)
        
        return len(image_files)
    
    def augment_data(self, augmentation_factor=2):
        """
        Augment the dataset with transformed versions of existing images
        
        Args:
            augmentation_factor (int): Number of augmented samples per original image
            
        Returns:
            int: Number of augmented images created
        """
        try:
            # Only augment training images
            train_img_dir = os.path.join(self.output_dir, "images", "train")
            train_label_dir = os.path.join(self.output_dir, "labels", "train")
            
            if not os.path.exists(train_img_dir) or not os.path.exists(train_label_dir):
                logger.error("Training split not found. Run prepare_yolo_dataset first.")
                return 0
            
            # Get training images
            train_images = [f for f in os.listdir(train_img_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            augmented_count = 0
            
            for img_file in tqdm(train_images, desc="Augmenting images"):
                # Load image
                img_path = os.path.join(train_img_dir, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    logger.warning(f"Failed to load image: {img_file}")
                    continue
                
                # Load annotations
                base_name = os.path.splitext(img_file)[0]
                label_path = os.path.join(train_label_dir, f"{base_name}.txt")
                
                if not os.path.exists(label_path):
                    logger.warning(f"No annotations found for {img_file}")
                    continue
                
                with open(label_path, 'r') as f:
                    annotations = f.read().splitlines()
                
                # Create augmented versions
                for i in range(augmentation_factor):
                    aug_image, aug_annotations = self._apply_augmentation(image, annotations)
                    
                    # Save augmented image
                    aug_filename = f"{base_name}_aug{i+1}.jpg"
                    aug_img_path = os.path.join(train_img_dir, aug_filename)
                    cv2.imwrite(aug_img_path, aug_image)
                    
                    # Save augmented annotations
                    aug_label_path = os.path.join(train_label_dir, f"{base_name}_aug{i+1}.txt")
                    with open(aug_label_path, 'w') as f:
                        f.write('\n'.join(aug_annotations))
                    
                    augmented_count += 1
            
            # Update stats
            self.stats["total_images"] += augmented_count
            
            logger.info(f"Created {augmented_count} augmented images")
            return augmented_count
            
        except Exception as e:
            logger.error(f"Error augmenting data: {e}")
            return 0
    
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
        
        # Random augmentation choices
        flip_horizontal = random.choice([True, False])
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        hue_shift = random.uniform(-0.1, 0.1)
        saturation = random.uniform(0.8, 1.2)
        
        # Apply augmentations to image
        augmented = image.copy()
        
        # Flip horizontally
        if flip_horizontal:
            augmented = cv2.flip(augmented, 1)
        
        # Convert to HSV for color adjustments
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust hue
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
        
        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        # Adjust brightness/value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        
        # Convert back to BGR
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Adjust contrast
        augmented = np.clip((augmented.astype(np.float32) - 128) * contrast + 128, 0, 255).astype(np.uint8)
        
        # Update annotations
        augmented_annotations = []
        
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) == 5:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
                
                # Update for horizontal flip
                if flip_horizontal:
                    x_center = 1.0 - x_center
                
                # Keep within bounds
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                box_width = max(0, min(1, box_width))
                box_height = max(0, min(1, box_height))
                
                augmented_annotations.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")
        
        return augmented, augmented_annotations
    
    def generate_statistics(self):
        """
        Generate and print comprehensive dataset statistics
        
        Returns:
            dict: Dataset statistics
        """
        # Update total counts
        img_counts = {
            "train": len(os.listdir(os.path.join(self.output_dir, "images", "train"))) if os.path.exists(os.path.join(self.output_dir, "images", "train")) else 0,
            "val": len(os.listdir(os.path.join(self.output_dir, "images", "val"))) if os.path.exists(os.path.join(self.output_dir, "images", "val")) else 0,
            "test": len(os.listdir(os.path.join(self.output_dir, "images", "test"))) if os.path.exists(os.path.join(self.output_dir, "images", "test")) else 0,
            "raw": len(os.listdir(self.image_dir))
        }
        
        # Count annotations
        ann_counts = {
            "train": self._count_annotations(os.path.join(self.output_dir, "labels", "train")),
            "val": self._count_annotations(os.path.join(self.output_dir, "labels", "val")),
            "test": self._count_annotations(os.path.join(self.output_dir, "labels", "test")),
            "raw": self._count_annotations(self.label_dir)
        }
        
        # Update stats
        stats = {
            "image_counts": img_counts,
            "annotation_counts": ann_counts,
            "class_distribution": self.stats["class_distribution"],
            "image_sizes": self.stats["image_sizes"],
            "total_images": sum(img_counts.values()),
            "total_annotations": sum(ann_counts.values()),
            "annotations_per_image": sum(ann_counts.values()) / max(1, sum(img_counts.values()))
        }
        
        # Print statistics
        logger.info("=== Dataset Statistics ===")
        logger.info(f"Total Images: {stats['total_images']}")
        logger.info(f"Total Annotations: {stats['total_annotations']}")
        logger.info(f"Annotations per Image: {stats['annotations_per_image']:.2f}")
        logger.info("Split:")
        for split, count in img_counts.items():
            logger.info(f"  - {split}: {count} images, {ann_counts[split]} annotations")
        logger.info("Class Distribution:")
        for cls, count in stats['class_distribution'].items():
            logger.info(f"  - {cls}: {count} annotations")
        
        # Save stats
        with open(os.path.join(self.output_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        return stats
    
    def _count_annotations(self, label_dir):
        """
        Count annotations in a directory
        
        Args:
            label_dir (str): Directory containing label files
            
        Returns:
            int: Number of annotations
        """
        if not os.path.exists(label_dir):
            return 0
            
        count = 0
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    count += len(f.read().splitlines())
        
        return count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Zombie Training Dataset Generator')
    parser.add_argument('--output-dir', type=str, default='data/zombies', help='Output directory for dataset')
    parser.add_argument('--capture', action='store_true', help='Capture frames from screen')
    parser.add_argument('--duration', type=int, default=60, help='Duration to capture in seconds')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to capture')
    parser.add_argument('--annotate', action='store_true', help='Annotate images')
    parser.add_argument('--prepare', action='store_true', help='Prepare YOLO dataset')
    parser.add_argument('--augment', action='store_true', help='Augment training data')
    parser.add_argument('--factor', type=int, default=2, help='Augmentation factor')
    parser.add_argument('--stats', action='store_true', help='Generate dataset statistics')
    args = parser.parse_args()
    
    # Create dataset generator
    generator = ZombieDatasetGenerator(args.output_dir)
    
    # Capture frames if requested
    if args.capture:
        generator.capture_frames(args.duration, args.fps)
    
    # Annotate images if requested
    if args.annotate:
        generator.annotate_images()
    
    # Prepare YOLO dataset if requested
    if args.prepare:
        generator.prepare_yolo_dataset()
    
    # Augment data if requested
    if args.augment:
        generator.augment_data(args.factor)
    
    # Generate statistics if requested
    if args.stats or (not args.capture and not args.annotate and not args.prepare and not args.augment):
        generator.generate_statistics()

if __name__ == "__main__":
    main()