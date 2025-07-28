#!/usr/bin/env python3
"""
ASL Recognition Model Training Script

This script processes collected keypoint data from the frontend and trains
an LSTM model for American Sign Language gesture recognition.

Usage:
    python train_asl_model.py --data_dir asl_data --output_model asl_recognition_model
"""

import os
import json
import numpy as np
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASLDataProcessor:
    """Handles loading and preprocessing of ASL keypoint data."""
    
    def __init__(self, sequence_length=30, num_features=258):
        self.sequence_length = sequence_length
        self.num_features = num_features  # 132 (pose) + 63 (left hand) + 63 (right hand)
        self.label_encoder = LabelEncoder()
        
    def preprocess_keypoints(self, keypoint_data):
        """
        Flattens and normalizes keypoint data from a single frame into a 1D numpy array.
        Matches the preprocessing logic from the frontend ASLRecognizer.
        """
        features = []

        # Process Pose landmarks (33 * 4 = 132 features)
        if keypoint_data and keypoint_data.get('pose'):
            for lm in keypoint_data['pose']:
                features.extend([lm['x'], lm['y'], lm['z'], lm['visibility']])
        else:
            features.extend([0.0] * (33 * 4))

        # Process Left Hand landmarks (21 * 3 = 63 features)
        if keypoint_data and keypoint_data.get('leftHand'):
            for lm in keypoint_data['leftHand']:
                features.extend([lm['x'], lm['y'], lm['z']])
        else:
            features.extend([0.0] * (21 * 3))

        # Process Right Hand landmarks (21 * 3 = 63 features)
        if keypoint_data and keypoint_data.get('rightHand'):
            for lm in keypoint_data['rightHand']:
                features.extend([lm['x'], lm['y'], lm['z']])
        else:
            features.extend([0.0] * (21 * 3))
        
        # Ensure the feature vector has the expected length
        if len(features) != self.num_features:
            logger.warning(f"Feature vector length mismatch: Expected {self.num_features}, got {len(features)}. Adjusting.")
            if len(features) > self.num_features:
                features = features[:self.num_features]
            else:
                features.extend([0.0] * (self.num_features - len(features)))

        return np.array(features, dtype=np.float32)
    
    def load_data_from_json_files(self, data_dir):
        """Load training data from JSON files in the specified directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
        
        sequences = []
        labels = []
        
        # Process individual JSON files
        json_files = list(data_path.glob("*.json"))
        if json_files:
            logger.info(f"Found {len(json_files)} JSON files in {data_dir}")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Handle different JSON structures
                    if 'recordings' in data:
                        # Downloaded training data format
                        for recording in data['recordings']:
                            gesture = recording['gesture']
                            frames = recording['frames']
                            sequence = self.process_sequence(frames)
                            if sequence is not None:
                                sequences.append(sequence)
                                labels.append(gesture)
                    else:
                        # Single recording format
                        gesture = data.get('gesture', 'UNKNOWN')
                        frames = data.get('frames', [])
                        sequence = self.process_sequence(frames)
                        if sequence is not None:
                            sequences.append(sequence)
                            labels.append(gesture)
                            
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {e}")
                    continue
        
        # Also check for organized directory structure
        for gesture_dir in data_path.iterdir():
            if gesture_dir.is_dir():
                gesture_name = gesture_dir.name.upper()
                json_files = list(gesture_dir.glob("*.json"))
                
                logger.info(f"Processing {len(json_files)} files for gesture: {gesture_name}")
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract frames from the JSON structure
                        frames = data.get('frames', data)  # Handle different structures
                        sequence = self.process_sequence(frames)
                        if sequence is not None:
                            sequences.append(sequence)
                            labels.append(gesture_name)
                            
                    except Exception as e:
                        logger.error(f"Error processing {json_file}: {e}")
                        continue
        
        if not sequences:
            raise ValueError("No valid training data found")
        
        logger.info(f"Loaded {len(sequences)} sequences for {len(set(labels))} unique gestures")
        logger.info(f"Gesture distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return np.array(sequences), np.array(labels)
    
    def process_sequence(self, frames):
        """Process a sequence of keypoint frames into model input format."""
        if not frames or len(frames) < 10:  # Minimum viable sequence
            return None
        
        processed_frames = []
        for frame_data in frames:
            processed_frame = self.preprocess_keypoints(frame_data)
            processed_frames.append(processed_frame)
        
        # Handle sequence length normalization
        if len(processed_frames) < self.sequence_length:
            # Pad with the last frame
            last_frame = processed_frames[-1]
            while len(processed_frames) < self.sequence_length:
                processed_frames.append(last_frame.copy())
        elif len(processed_frames) > self.sequence_length:
            # Take the most recent frames
            processed_frames = processed_frames[-self.sequence_length:]
        
        return np.array(processed_frames)
    
    def augment_data(self, sequences, labels, augment_factor=2):
        """Apply data augmentation to increase dataset size."""
        augmented_sequences = []
        augmented_labels = []
        
        for seq, label in zip(sequences, labels):
            # Original sequence
            augmented_sequences.append(seq)
            augmented_labels.append(label)
            
            # Apply augmentations
            for _ in range(augment_factor):
                # Add slight noise
                noise = np.random.normal(0, 0.01, seq.shape)
                noisy_seq = seq + noise
                
                # Scale slightly
                scale_factor = np.random.uniform(0.95, 1.05)
                scaled_seq = seq * scale_factor
                
                augmented_sequences.extend([noisy_seq, scaled_seq])
                augmented_labels.extend([label, label])
        
        logger.info(f"Augmented dataset from {len(sequences)} to {len(augmented_sequences)} sequences")
        return np.array(augmented_sequences), np.array(augmented_labels)
    
    def prepare_data(self, sequences, labels, test_size=0.2, validation_size=0.1):
        """Prepare data for training with proper train/validation/test splits."""
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(num_classes)))}")
        
        # One-hot encode
        categorical_labels = to_categorical(encoded_labels, num_classes)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, categorical_labels, test_size=test_size, 
            stratify=encoded_labels, random_state=42
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=np.argmax(y_temp, axis=1), random_state=42
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes

class ASLModelTrainer:
    """Handles LSTM model creation and training for ASL recognition."""
    
    def __init__(self, sequence_length=30, num_features=258):
        self.sequence_length = sequence_length
        self.num_features = num_features
        
    def build_model(self, num_classes):
        """Build and compile the LSTM model for ASL recognition."""
        
        model = Sequential([
            InputLayer(input_shape=(self.sequence_length, self.num_features)),
            
            # First LSTM layer with dropout
            LSTM(128, return_sequences=True, activation='tanh'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(256, return_sequences=True, activation='tanh'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(128, activation='tanh'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_callbacks(self, model_save_path):
        """Create training callbacks for better training control."""
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f"{model_save_path}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model, train_data, val_data, epochs=100, batch_size=32, model_save_path="asl_model"):
        """Train the ASL recognition model."""
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        callbacks = self.create_callbacks(model_save_path)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, test_data, label_encoder):
        """Evaluate the trained model on test data."""
        
        X_test, y_test = test_data
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Evaluation metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        class_names = label_encoder.classes_
        report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
        logger.info(f"Classification Report:\n{report}")
        
        return test_accuracy, test_loss

def main():
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    parser.add_argument('--data_dir', type=str, default='asl_data', 
                       help='Directory containing training data')
    parser.add_argument('--output_model', type=str, default='asl_recognition_model',
                       help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Length of input sequences')
    
    args = parser.parse_args()
    
    logger.info("Starting ASL Model Training")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output model: {args.output_model}")
    
    # Initialize processor and trainer
    processor = ASLDataProcessor(sequence_length=args.sequence_length)
    trainer = ASLModelTrainer(sequence_length=args.sequence_length)
    
    try:
        # Load data
        logger.info("Loading training data...")
        sequences, labels = processor.load_data_from_json_files(args.data_dir)
        
        # Apply augmentation if requested
        if args.augment:
            logger.info("Applying data augmentation...")
            sequences, labels = processor.augment_data(sequences, labels)
        
        # Prepare data
        logger.info("Preparing data for training...")
        train_data, val_data, test_data, num_classes = processor.prepare_data(sequences, labels)
        
        # Build model
        logger.info(f"Building model for {num_classes} classes...")
        model = trainer.build_model(num_classes)
        model.summary()
        
        # Train model
        logger.info("Training model...")
        history = trainer.train_model(
            model, train_data, val_data, 
            epochs=args.epochs, batch_size=args.batch_size,
            model_save_path=args.output_model
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        trainer.evaluate_model(model, test_data, processor.label_encoder)
        
        # Save final model
        model_path = f"{args.output_model}.keras"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save label encoder
        import pickle
        with open(f"{args.output_model}_label_encoder.pkl", 'wb') as f:
            pickle.dump(processor.label_encoder, f)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 