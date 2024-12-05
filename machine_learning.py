import os
import logging
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from tqdm import tqdm
import time

# Metadata file
metadata_file = "keys/DF/CM/trial_metadata.txt"

#Avspoof flac files are in AVSspoof2021_DF_eval/flac

# Set up logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Update the metadata processing section
def process_avspoof_metadata(metadata_file: str, target_part: str = "part00") -> List[dict]:
    """Process AVSpoof metadata file and filter for specific part."""
    metadata = []
    
    # Define range for part00
    PART00_START = 2000000
    PART00_END = 2749694
    
    with open(metadata_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 6:
                file_name = parts[1]  # DF_E_XXXXXXX
                try:
                    # Extract the number from the filename
                    file_number = int(file_name.split('_')[-1])
                    # Check if the number is in part00 range
                    if PART00_START <= file_number <= PART00_END:
                        metadata.append({
                            "file_name": file_name,
                            "label": 0 if parts[5] == "bonafide" else 1,
                            "original_speaker": parts[0],
                            "codec": parts[2],
                            "source": parts[3],
                            "attack_type": parts[8] if len(parts) > 8 else None
                        })
                except ValueError:
                    logger.warning(f"Could not parse file number from: {file_name}")
                    continue
    
    # Log dataset statistics
    total = len(metadata)
    real_count = sum(1 for item in metadata if item['label'] == 0)
    fake_count = sum(1 for item in metadata if item['label'] == 1)
    
    logger.info(f"\nMetadata Statistics for {target_part}:")
    logger.info(f"Total samples: {total}")
    logger.info(f"Bonafide samples: {real_count}")
    logger.info(f"Spoof samples: {fake_count}")
    
    return metadata

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, duration: int = 5):  # Updated sample rate for AVSpoof
        self.sr = sample_rate
        self.fixed_length = duration * sample_rate
        
    def create_spectrogram(self, audio_path: str) -> Optional[np.ndarray]:
        """Create mel spectrogram from audio file."""
        try:
            # Load and pad/trim audio to fixed length
            y, _ = librosa.load(audio_path, sr=self.sr)
            y = librosa.util.fix_length(y, size=self.fixed_length)
            
            # Data augmentation
            if np.random.rand() < 0.5:  # Apply augmentation with 50% probability
                y = self.apply_augmentation(y)
            
            # Use fixed parameters for consistent spectrogram size
            mel_spect = librosa.feature.melspectrogram(
                y=y, 
                sr=self.sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                win_length=2048,  # Added fixed window length
                center=True  # Ensure consistent padding
            )
            
            # Get fixed number of time steps
            n_time_steps = 157  # Use the larger of your observed sizes
            if mel_spect.shape[1] > n_time_steps:
                mel_spect = mel_spect[:, :n_time_steps]
            else:
                # Pad with zeros if too short
                pad_width = n_time_steps - mel_spect.shape[1]
                mel_spect = np.pad(mel_spect, ((0, 0), (0, pad_width)), mode='constant')
            
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            
            # Normalize spectrogram
            mel_spect_db -= mel_spect_db.min()
            mel_spect_db /= mel_spect_db.max()
            
            return mel_spect_db
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None

    def apply_augmentation(self, y: np.ndarray) -> np.ndarray:
        """Apply simple data augmentation techniques."""
        try:
            if np.random.rand() < 0.5:
                y = librosa.effects.time_stretch(y=y, rate=1.1)
            if np.random.rand() < 0.5:
                y = librosa.effects.pitch_shift(y=y, sr=self.sr, n_steps=2)
            # Ensure consistent length after augmentation
            y = librosa.util.fix_length(y, size=self.fixed_length)
            return y
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}. Returning original audio.")
            return y

    def prepare_avspoof_dataset(self, metadata_file: str, audio_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset from AVSpoof metadata and audio files."""
        specs: List[np.ndarray] = []
        labels: List[int] = []
        
        metadata = process_avspoof_metadata(metadata_file)
        
        files_processed = 0
        files_not_found = 0
        files_error = 0
        
        start_time = time.time()
        for entry in tqdm(metadata, desc=f"Processing {audio_dir}"):
            audio_path = os.path.join(audio_dir, f"{entry['file_name']}.flac")
            if os.path.exists(audio_path):
                try:
                    spec = self.create_spectrogram(audio_path)
                    if spec is not None:
                        # Make sure all specs have the same shape
                        if len(specs) > 0 and spec.shape != specs[0].shape:
                            logger.warning(f"Inconsistent spectrogram shape: {spec.shape} vs {specs[0].shape}")
                            continue
                        specs.append(spec)
                        labels.append(entry['label'])
                        files_processed += 1
                    else:
                        files_error += 1
                except Exception as e:
                    logger.error(f"Error processing {audio_path}: {str(e)}")
                    files_error += 1
            else:
                files_not_found += 1
        
        processing_time = time.time() - start_time
        
        # Log summary statistics
        logger.info(f"\nProcessing Summary for {audio_dir}:")
        logger.info(f"Files successfully processed: {files_processed}")
        logger.info(f"Files not found: {files_not_found}")
        logger.info(f"Files with errors: {files_error}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        # Convert to numpy arrays, preserving the 2D structure of spectrograms
        X = np.stack(specs)  # This will maintain the correct shape
        y = np.array(labels)
        
        logger.info(f"Final data shapes - X: {X.shape}, y: {y.shape}")
        
        return X, y

class AudioDetectorModel:
    def __init__(self, input_shape: Tuple[int, ...]):
        self.model = self._create_model(input_shape)
        
    def _create_model(self, input_shape: Tuple[int, ...]) -> models.Model:
        """Create CNN model architecture."""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu',  # Reduced from 256
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        # Optimizer with learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              class_weights: dict,
              epochs: int = 30,  # Reduced from 50
              batch_size: int = 128) -> tf.keras.callbacks.History:  # Increased from 32
        """Train the model with progress tracking."""
        logger.info("\n=== Training Configuration ===")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Input shape: {X_train.shape[1:]}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Class weights: {class_weights}")
        
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.info(f"\nEpoch {epoch + 1}/{epochs}")
                logger.info(f"Training loss: {logs['loss']:.4f}")
                logger.info(f"Training accuracy: {logs['accuracy']:.4f}")
                logger.info(f"Validation loss: {logs['val_loss']:.4f}")
                logger.info(f"Validation accuracy: {logs['val_accuracy']:.4f}")
        
        callbacks = [
            CustomCallback(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Add learning rate monitoring
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info("\n=== Training Complete ===")
        logger.info(f"Total training time: {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
        logger.info(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Evaluate model performance."""
        logger.info("\n=== Model Evaluation ===")
        logger.info(f"Testing on {len(X_test)} samples")
        
        start_time = time.time()
        test_loss, test_accuracy, test_auc = self.model.evaluate(X_test, y_test, verbose=0)
        evaluation_time = time.time() - start_time
        
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")
        
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

def main():
    metadata_file = "keys/DF/CM/trial_metadata.txt"
    # Only use part00
    audio_dirs = [
        "ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac"
    ]
    
    logger.info("=== Starting Audio Detection Model Training with ASVspoof Dataset (part00 only) ===")
    logger.info(f"Using metadata from: {metadata_file}")
    logger.info("Using audio files from:")
    for dir in audio_dirs:
        logger.info(f"- {dir}")

    preprocessor = AudioPreprocessor()
    
    specs_all = []
    labels_all = []
    
    for audio_dir in audio_dirs:
        logger.info(f"\nProcessing files from: {audio_dir}")
        # Verify directory exists and show some debug info
        if os.path.exists(audio_dir):
            files = os.listdir(audio_dir)
            logger.info(f"Found {len(files)} files in directory")
            if files:
                logger.info(f"Sample files: {files[:3]}")
        else:
            logger.warning(f"Directory not found: {audio_dir}")
            continue
            
        X, y = preprocessor.prepare_avspoof_dataset(metadata_file, audio_dir)
        if len(X) > 0:
            specs_all.append(X)
            labels_all.append(y)
        else:
            logger.warning(f"No files were processed in {audio_dir}")
    
    if not specs_all:
        raise ValueError("No audio files were successfully processed! Check the paths above.")
    
    # Combine all the processed data
    X = np.concatenate(specs_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    
    # Add channel dimension for CNN
    X = X[..., np.newaxis]
    
    logger.info("\n=== Combined Dataset Statistics ===")
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Memory usage: {X.nbytes / (1024 * 1024):.2f} MB")
    logger.info(f"Real samples: {np.sum(y == 0)}")
    logger.info(f"Fake samples: {np.sum(y == 1)}")
    
    # Split data and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"\nTrain set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    model = AudioDetectorModel(X_train.shape[1:])
    history = model.train(X_train, y_train, X_test, y_test, class_weights)
    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
