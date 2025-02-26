"""
Training script for the YOLO zombie detection model.
This module handles the training and fine-tuning of the YOLO model.
"""

import os
import logging
import argparse
import yaml
import torch
import shutil
import time
from pathlib import Path

logger = logging.getLogger("Training")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLO model for zombie detection')
    parser.add_argument('--data', type=str, default='data/zombies.yaml', help='Path to data config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--name', default='zombie_detector', help='Name for the trained model')
    return parser.parse_args()

def setup_training_environment(data_config_path):
    """
    Setup the training environment and verify data
    
    Args:
        data_config_path (str): Path to data configuration
        
    Returns:
        dict: Training configuration
    """
    # Check if data config exists
    if not os.path.exists(data_config_path):
        logger.error(f"Data config not found: {data_config_path}")
        raise FileNotFoundError(f"Data config not found: {data_config_path}")
    
    # Load data config
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Verify data paths
    train_path = data_config.get('train', '')
    val_path = data_config.get('val', '')
    
    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    if not os.path.exists(val_path):
        logger.error(f"Validation data not found: {val_path}")
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    # Count dataset size
    train_images = len(list(Path(train_path).glob('images/*.jpg')))
    train_labels = len(list(Path(train_path).glob('labels/*.txt')))
    
    val_images = len(list(Path(val_path).glob('images/*.jpg')))
    val_labels = len(list(Path(val_path).glob('labels/*.txt')))
    
    logger.info(f"Training dataset: {train_images} images, {train_labels} labels")
    logger.info(f"Validation dataset: {val_images} images, {val_labels} labels")
    
    # Check if dataset is sufficient
    if train_images < 100 or train_labels < 100:
        logger.warning("Training dataset may be too small. Consider augmenting data.")
    
    # Prepare output directories
    output_dir = Path('runs/train') / data_config.get('name', 'zombie_detector')
    if output_dir.exists():
        logger.warning(f"Output directory already exists: {output_dir}")
        backup_dir = output_dir.with_name(f"{output_dir.name}_{int(time.time())}")
        shutil.move(str(output_dir), str(backup_dir))
        logger.info(f"Moved existing directory to: {backup_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'data_config': data_config,
        'output_dir': output_dir,
        'train_images': train_images,
        'val_images': val_images
    }

def train_model(args, training_config):
    """
    Train the YOLO model
    
    Args:
        args (Namespace): Command line arguments
        training_config (dict): Training configuration
        
    Returns:
        str: Path to the trained model weights
    """
    logger.info("Starting model training...")
    
    try:
        # Use torch hub to load YOLOv5 train procedure
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)
        
        # Setup training parameters
        train_params = {
            'data': args.data,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'img_size': args.img_size,
            'device': args.device,
            'project': str(training_config['output_dir'].parent),
            'name': training_config['output_dir'].name,
            'exist_ok': True
        }
        
        logger.info(f"Training with parameters: {train_params}")
        
        # Start training
        results = model.train(**train_params)
        
        # Get best model path
        best_model_path = str(Path(training_config['output_dir']) / 'weights' / 'best.pt')
        
        logger.info(f"Training completed. Best model saved at: {best_model_path}")
        return best_model_path
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Model training failed: {e}")

def evaluate_model(model_path, data_config_path):
    """
    Evaluate the trained model
    
    Args:
        model_path (str): Path to the trained model
        data_config_path (str): Path to data configuration
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating trained model...")
    
    try:
        # Load the model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        
        # Load data config to get validation path
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        val_path = data_config.get('val', '')
        
        if not os.path.exists(val_path):
            logger.error(f"Validation data not found: {val_path}")
            return None
        
        # Setup validation parameters
        val_params = {
            'data': data_config_path,
            'batch_size': 16,
            'imgsz': 640,
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'device': '',
            'save_json': True,
            'save_hybrid': False,
            'save_conf': True,
            'verbose': False
        }
        
        # Run validation
        results = model.val(**val_params)
        
        # Extract metrics from results
        metrics = {
            'precision': results.results_dict.get('metrics/precision', 0),
            'recall': results.results_dict.get('metrics/recall', 0),
            'mAP_0.5': results.results_dict.get('metrics/mAP_0.5', 0),
            'mAP_0.5:0.95': results.results_dict.get('metrics/mAP_0.5:0.95', 0),
            'fitness': results.fitness
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None

def main():
    """
    Main function for model training
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    args = parse_arguments()
    
    try:
        # Setup training environment
        logger.info("Setting up training environment...")
        training_config = setup_training_environment(args.data)
        
        # Train model
        model_path = train_model(args, training_config)
        
        # Evaluate model
        metrics = evaluate_model(model_path, args.data)
        
        # Copy best model to project models directory
        final_model_path = os.path.join('models', f"{args.name}.pt")
        os.makedirs('models', exist_ok=True)
        shutil.copy(model_path, final_model_path)
        
        logger.info(f"Model training and evaluation complete. Final model saved at: {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training process failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
