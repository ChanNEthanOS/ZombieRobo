"""
Annotation utilities for the COD WaW Zombies Bot.
This module handles creating and exporting annotations for training data.
"""

import os
import cv2
import json
import logging
import time
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger("Annotations")

def export_annotations(annotations, image_path, format='yolo', output_dir=None):
    """
    Export annotations to various formats
    
    Args:
        annotations (list): List of annotation dictionaries with class_id, x, y, width, height
        image_path (str): Path to the image
        format (str): Output format ('yolo', 'voc', 'coco')
        output_dir (str, optional): Directory to save annotations
        
    Returns:
        str: Path to the exported annotation file
    """
    image_path = Path(image_path)
    
    # If no output directory specified, use the same directory as the image
    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    img_height, img_width = image.shape[:2]
    
    # Export based on format
    if format == 'yolo':
        return _export_yolo(annotations, image_path, img_width, img_height, output_dir)
    elif format == 'voc':
        return _export_voc(annotations, image_path, img_width, img_height, output_dir)
    elif format == 'coco':
        return _export_coco(annotations, image_path, img_width, img_height, output_dir)
    else:
        logger.error(f"Unsupported annotation format: {format}")
        return None

def _export_yolo(annotations, image_path, img_width, img_height, output_dir):
    """Export annotations to YOLO format"""
    # YOLO format: class_id center_x center_y width height
    # All values are normalized to 0-1
    
    output_file = output_dir / f"{image_path.stem}.txt"
    
    with open(output_file, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            
            # Convert to YOLO format (normalized)
            x_center = (ann['x'] + ann['width'] / 2) / img_width
            y_center = (ann['y'] + ann['height'] / 2) / img_height
            width = ann['width'] / img_width
            height = ann['height'] / img_height
            
            # Ensure values are within bounds
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    logger.debug(f"Exported YOLO annotations to {output_file}")
    return str(output_file)

def _export_voc(annotations, image_path, img_width, img_height, output_dir):
    """Export annotations to Pascal VOC format"""
    # VOC format: XML file with object tags
    
    output_file = output_dir / f"{image_path.stem}.xml"
    
    # Create XML structure
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = image_path.parent.name
    ET.SubElement(root, 'filename').text = image_path.name
    ET.SubElement(root, 'path').text = str(image_path)
    
    # Source info
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    
    # Size info
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)
    ET.SubElement(size, 'depth').text = '3'
    
    # Object annotations
    class_names = ['zombie', 'crawler', 'hellhound', 'boss']
    
    for ann in annotations:
        obj = ET.SubElement(root, 'object')
        
        # Class name from ID
        class_id = ann['class_id']
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        
        # Bounding box
        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(max(0, ann['x']))
        ET.SubElement(bbox, 'ymin').text = str(max(0, ann['y']))
        ET.SubElement(bbox, 'xmax').text = str(min(img_width, ann['x'] + ann['width']))
        ET.SubElement(bbox, 'ymax').text = str(min(img_height, ann['y'] + ann['height']))
    
    # Write to file
    tree = ET.ElementTree(root)
    tree.write(output_file)
    
    logger.debug(f"Exported VOC annotations to {output_file}")
    return str(output_file)

def _export_coco(annotations, image_path, img_width, img_height, output_dir):
    """Export annotations to COCO format"""
    # COCO format: JSON with images and annotations
    
    output_file = output_dir / 'annotations.json'
    
    # Load existing COCO file if it exists
    coco_data = {
        "info": {
            "description": "COD WaW Zombies Dataset",
            "version": "1.0",
            "year": time.strftime("%Y"),
            "contributor": "ZombiesBot",
            "date_created": time.strftime("%Y-%m-%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [
            {"id": 0, "name": "zombie", "supercategory": "enemy"},
            {"id": 1, "name": "crawler", "supercategory": "enemy"},
            {"id": 2, "name": "hellhound", "supercategory": "enemy"},
            {"id": 3, "name": "boss", "supercategory": "enemy"}
        ],
        "images": [],
        "annotations": []
    }
    
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing COCO file: {e}")
    
    # Generate image ID from path
    img_id = abs(hash(str(image_path))) % 10000000
    
    # Check if image already exists in dataset
    image_exists = False
    for img in coco_data["images"]:
        if img["id"] == img_id:
            image_exists = True
            break
    
    # Add image if it doesn't exist
    if not image_exists:
        coco_data["images"].append({
            "id": img_id,
            "file_name": image_path.name,
            "width": img_width,
            "height": img_height,
            "license": 1,
            "date_captured": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Remove existing annotations for this image
    coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann["image_id"] != img_id]
    
    # Add annotations
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
        
        # Ensure within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        
        # Create COCO annotation
        coco_ann = {
            "id": abs(hash(f"{img_id}_{i}")) % 10000000,
            "image_id": img_id,
            "category_id": ann['class_id'],
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation": [],
            "iscrowd": 0
        }
        
        coco_data["annotations"].append(coco_ann)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.debug(f"Exported COCO annotations to {output_file}")
    return str(output_file)

def create_annotation_tool(image_dir, output_dir=None, class_names=None):
    """
    Create a simple OpenCV-based annotation tool
    
    Args:
        image_dir (str): Directory containing images to annotate
        output_dir (str, optional): Directory to save annotations
        class_names (list, optional): List of class names
        
    Returns:
        bool: True if annotator was run, False on error
    """
    if class_names is None:
        class_names = ['zombie', 'crawler', 'hellhound', 'boss']
    
    # Setup directories
    image_dir = Path(image_dir)
    
    if output_dir is None:
        output_dir = image_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return False
    
    logger.info(f"Starting annotation tool with {len(image_files)} images")
    
    # Variables for annotation
    class_id = 0
    drawing = False
    rect_start = None
    annotations = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, rect_start, current_image, annotations
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            pass  # Used for drawing preview, handled in main loop
        
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
                        'class_id': class_id,
                        'x': x_min,
                        'y': y_min,
                        'width': width,
                        'height': height
                    })
                    logger.debug(f"Added annotation: class={class_id}, box=({x_min},{y_min},{width},{height})")
    
    # Create window and set mouse callback
    cv2.namedWindow('Annotator')
    cv2.setMouseCallback('Annotator', mouse_callback)
    
    # Process each image
    current_idx = 0
    while current_idx < len(image_files):
        image_file = image_files[current_idx]
        
        # Load image
        current_image = cv2.imread(str(image_file))
        if current_image is None:
            logger.warning(f"Failed to load image: {image_file}")
            current_idx += 1
            continue
        
        # Load existing annotations if available
        ann_file = output_dir / f"{image_file.stem}.txt"
        annotations = []
        
        if ann_file.exists():
            try:
                with open(ann_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to pixel coordinates
                            img_height, img_width = current_image.shape[:2]
                            x = int((x_center - width/2) * img_width)
                            y = int((y_center - height/2) * img_height)
                            w = int(width * img_width)
                            h = int(height * img_height)
                            
                            annotations.append({
                                'class_id': cls_id,
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h
                            })
            except Exception as e:
                logger.warning(f"Error loading annotations for {image_file}: {e}")
        
        # Main annotation loop for current image
        while True:
            # Create display image
            display_img = current_image.copy()
            
            # Draw existing annotations
            for ann in annotations:
                x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
                cls = ann['class_id']
                
                # Different color for each class
                color = [
                    (0, 0, 255),    # Red for zombies
                    (0, 255, 0),    # Green for crawlers
                    (255, 0, 0),    # Blue for hellhounds
                    (255, 0, 255)   # Magenta for bosses
                ][cls % 4]
                
                cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
                
                # Add class label
                cls_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
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
            cv2.putText(display_img, f"Image: {current_idx+1}/{len(image_files)} - {image_file.name}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(display_img, f"Class: {class_id} ({class_names[class_id] if class_id < len(class_names) else 'unknown'})", 
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
                return True
            
            elif key == ord('s'):
                # Save annotations
                export_annotations(annotations, image_file, 'yolo', output_dir)
                logger.info(f"Saved annotations for {image_file.name}")
                break
                
            elif key == ord('n'):
                # Next image
                export_annotations(annotations, image_file, 'yolo', output_dir)
                current_idx += 1
                break
                
            elif key == ord('p'):
                # Previous image
                export_annotations(annotations, image_file, 'yolo', output_dir)
                current_idx = max(0, current_idx - 1)
                break
                
            elif key == ord('c'):
                # Cycle through classes
                class_id = (class_id + 1) % len(class_names)
                logger.debug(f"Current class set to {class_id} ({class_names[class_id]})")
                
            elif key == ord('d'):
                # Delete last annotation
                if annotations:
                    annotations.pop()
                    logger.debug("Removed last annotation")
    
    cv2.destroyAllWindows()
    return True

def create_annotation_from_detection(image, detections, confidence_threshold=0.5):
    """
    Create annotations from detection results
    
    Args:
        image (numpy.ndarray): Image
        detections (list): List of detection dictionaries
        confidence_threshold (float): Confidence threshold
        
    Returns:
        list: Annotation dictionaries ready for export
    """
    annotations = []
    
    for det in detections:
        if det['confidence'] >= confidence_threshold:
            annotation = {
                'class_id': det.get('class', 0),
                'x': det['x'],
                'y': det['y'],
                'width': det['width'],
                'height': det['height']
            }
            annotations.append(annotation)
    
    return annotations
