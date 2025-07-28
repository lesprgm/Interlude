#!/usr/bin/env python3
"""
ASL Training Setup and Helper Script

This script helps set up the environment and provides utilities for ASL model training.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'tensorflow', 'sklearn', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed!")
    return True

def create_data_directory(data_dir="asl_data"):
    """Create the data directory structure."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Create subdirectories for each gesture
    gestures = ['HELLO', 'YES', 'NO', 'THANK_YOU', 'GOODBYE']
    for gesture in gestures:
        gesture_dir = data_path / gesture.lower()
        gesture_dir.mkdir(exist_ok=True)
        logger.info(f"Created directory: {gesture_dir}")
    
    logger.info(f"Data directory structure created at: {data_path.absolute()}")
    return data_path

def quick_train_with_sample_data():
    """Train a model with minimal sample data for testing."""
    import numpy as np
    import json
    from datetime import datetime
    
    logger.info("Creating sample training data for testing...")
    
    # Create sample data directory
    data_dir = create_data_directory("sample_asl_data")
    
    # Generate synthetic training data for each gesture
    gestures = ['HELLO', 'YES', 'NO']
    sequence_length = 30
    num_features = 258
    
    for gesture in gestures:
        gesture_dir = data_dir / gesture.lower()
        
        # Create 5 sample recordings per gesture
        for i in range(5):
            # Generate random keypoint data (this would normally come from MediaPipe)
            frames = []
            for frame_idx in range(sequence_length):
                frame_data = {
                    'pose': [{'x': np.random.random(), 'y': np.random.random(), 
                             'z': np.random.random(), 'visibility': np.random.random()} 
                            for _ in range(33)],
                    'leftHand': [{'x': np.random.random(), 'y': np.random.random(), 
                                 'z': np.random.random()} for _ in range(21)],
                    'rightHand': [{'x': np.random.random(), 'y': np.random.random(), 
                                  'z': np.random.random()} for _ in range(21)],
                    'timestamp': datetime.now().isoformat()
                }
                frames.append(frame_data)
            
            # Save as JSON file
            recording_data = {
                'gesture': gesture,
                'frames': frames,
                'frameCount': len(frames),
                'timestamp': datetime.now().isoformat()
            }
            
            file_path = gesture_dir / f"{gesture.lower()}_{i+1:02d}.json"
            with open(file_path, 'w') as f:
                json.dump(recording_data, f)
            
            logger.info(f"Created sample data: {file_path}")
    
    logger.info("Sample data created successfully!")
    return data_dir

def train_model(data_dir, epochs=20, use_sample_data=False):
    """Run the training script with specified parameters."""
    
    if use_sample_data:
        data_dir = quick_train_with_sample_data()
    
    # Check if training script exists
    train_script = Path("train_asl_model.py")
    if not train_script.exists():
        logger.error("train_asl_model.py not found!")
        return False
    
    # Build training command
    cmd = [
        sys.executable, "train_asl_model.py",
        "--data_dir", str(data_dir),
        "--epochs", str(epochs),
        "--batch_size", "16",  # Smaller batch size for limited data
        "--output_model", "asl_recognition_model"
    ]
    
    if use_sample_data:
        cmd.append("--augment")  # Use augmentation for small datasets
    
    logger.info(f"Running training command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Training completed successfully!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ASL Training Setup Helper')
    parser.add_argument('--check-deps', action='store_true', 
                       help='Check if dependencies are installed')
    parser.add_argument('--create-dirs', action='store_true',
                       help='Create data directory structure')
    parser.add_argument('--quick-train', action='store_true',
                       help='Train with sample synthetic data for testing')
    parser.add_argument('--train', type=str, metavar='DATA_DIR',
                       help='Train model with data from specified directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
    
    if args.create_dirs:
        create_data_directory()
    
    if args.quick_train:
        logger.info("Starting quick training with synthetic data...")
        if check_dependencies():
            train_model("", epochs=args.epochs, use_sample_data=True)
        else:
            logger.error("Please install dependencies first!")
    
    if args.train:
        logger.info(f"Starting training with data from: {args.train}")
        if check_dependencies():
            if Path(args.train).exists():
                train_model(args.train, epochs=args.epochs)
            else:
                logger.error(f"Data directory {args.train} does not exist!")
        else:
            logger.error("Please install dependencies first!")
    
    if not any([args.check_deps, args.create_dirs, args.quick_train, args.train]):
        print("ASL Training Setup Helper")
        print("\nUsage examples:")
        print("  python setup_asl_training.py --check-deps          # Check dependencies")
        print("  python setup_asl_training.py --create-dirs         # Create directory structure")
        print("  python setup_asl_training.py --quick-train         # Train with synthetic data")
        print("  python setup_asl_training.py --train asl_data      # Train with real data")
        print("\nWorkflow:")
        print("1. Check dependencies")
        print("2. Create data directories")
        print("3. Collect real data using the frontend interface")
        print("4. Train the model with collected data")
        print("5. Restart the backend to load the trained model")

if __name__ == "__main__":
    main() 