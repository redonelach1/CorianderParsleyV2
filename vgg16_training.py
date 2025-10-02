import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class CorianderParsleyClassifier:
    def __init__(self, data_dir="augmented_dataset", img_size=(224, 224), batch_size=32):
        """
        Initialize the VGG16-based classifier
        
        Args:
            data_dir (str): Directory containing the augmented dataset
            img_size (tuple): Input image size (width, height)
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def load_datasets(self, validation_split=0.2):
        """Load and prepare training and validation datasets"""
        
        print("Loading datasets...")
        
        # Load training dataset
        self.train_dataset = image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="training",
            seed=42,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="binary"  # Binary classification (0: coriander, 1: parsely)
        )
        
        # Load validation dataset
        self.val_dataset = image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=42,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="binary"
        )
        
        # Get class names
        self.class_names = self.train_dataset.class_names
        print(f"Classes found: {self.class_names}")
        
        # Calculate dataset sizes
        train_size = len(list(self.train_dataset.unbatch()))
        val_size = len(list(self.val_dataset.unbatch()))
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Normalize pixel values to [0,1]
        self.train_dataset = self.train_dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
        )
        self.val_dataset = self.val_dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
        )
        
        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_dataset = self.train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_dataset = self.val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        
        return self.train_dataset, self.val_dataset
    
    def build_model(self, fine_tune_layers=0):
        """
        Build VGG16-based model
        
        Args:
            fine_tune_layers (int): Number of top VGG16 layers to unfreeze for fine-tuning
        """
        
        print("Building VGG16 model...")
        
        # Load pre-trained VGG16 model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Unfreeze top layers for fine-tuning if specified
        if fine_tune_layers > 0:
            base_model.trainable = True
            # Freeze all layers except the top fine_tune_layers
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
            print(f"Fine-tuning enabled: last {fine_tune_layers} layers unfrozen")
        
        # Build the complete model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            
            # First dense layer
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            
            # Second dense layer
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        print("Model architecture built")
        self.model.summary()
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model with optimizer, loss, and metrics"""
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"Model compiled with learning rate: {learning_rate}")
    
    def setup_callbacks(self, model_name="coriander_vs_parsely_vgg16"):
        """Setup training callbacks"""
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                f"{model_name}.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, epochs=5, model_name="coriander_vs_parsely_vgg16"):
        """Train the model"""
        
        print(f"Starting training for {epochs} epochs...")
        print("="*60)
        
        callbacks = self.setup_callbacks(model_name)
        
        # Train the model
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        return self.history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        
        print("\nEvaluating model...")
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(
            self.val_dataset, verbose=0
        )
        
        # Calculate F1 score
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        
        print(f"Validation Results:")
        print(f"   Loss: {val_loss:.4f}")
        print(f"   Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"   Precision: {val_precision:.4f}")
        print(f"   Recall: {val_recall:.4f}")
        print(f"   F1 Score: {f1_score:.4f}")
        
        return {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': f1_score
        }
    
    def plot_training_history(self, save_plots=True):
        """Plot and save training history"""
        
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VGG16 Training History - Coriander vs Parsely', fontsize=16)
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', color='blue')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision', color='green')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', color='orange')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall plot
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall', color='purple')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', color='brown')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"Training plots saved as training_history_{timestamp}.png")
        
        plt.show()
    
    def fine_tune_model(self, fine_tune_layers=4, fine_tune_epochs=20, fine_tune_lr=1e-5):
        """
        Fine-tune the model by unfreezing top layers
        
        Args:
            fine_tune_layers (int): Number of top layers to unfreeze
            fine_tune_epochs (int): Number of epochs for fine-tuning
            fine_tune_lr (float): Learning rate for fine-tuning (should be lower)
        """
        
        print(f"\nStarting fine-tuning with {fine_tune_layers} unfrozen layers...")
        
        # Unfreeze top layers of the base model
        base_model = self.model.layers[0]  # VGG16 base model
        base_model.trainable = True
        
        # Freeze all layers except the top fine_tune_layers
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=fine_tune_lr)
        
        print(f"Trainable layers: {len([l for l in self.model.layers if l.trainable])}")
        
        # Continue training
        fine_tune_history = self.model.fit(
            self.train_dataset,
            epochs=fine_tune_epochs,
            validation_data=self.val_dataset,
            callbacks=self.setup_callbacks("coriander_vs_parsely_vgg16_finetuned"),
            verbose=1
        )
        
        # Update history
        if self.history is None:
            self.history = fine_tune_history
        else:
            # Combine histories
            for key in self.history.history.keys():
                self.history.history[key].extend(fine_tune_history.history[key])
        
        print("Fine-tuning completed!")
        
        return fine_tune_history


def main():
    """Main training pipeline"""
    
    print("Coriander vs Parsely Classification with VGG16")
    print("="*60)
    
    # Check if augmented dataset exists
    if not os.path.exists("augmented_dataset"):
        print("Augmented dataset not found!")
        print("Please run the data augmentation script first.")
        return
    
    # Initialize classifier
    classifier = CorianderParsleyClassifier(
        data_dir="augmented_dataset",
        img_size=(224, 224),
        batch_size=32
    )
    
    # Load datasets
    train_ds, val_ds = classifier.load_datasets(validation_split=0.2)
    
    # Build and compile model
    model = classifier.build_model(fine_tune_layers=0)  # Start with frozen base
    classifier.compile_model(learning_rate=0.001)  # Higher LR for initial training
    
    # Train model (initial phase)
    print("\nPhase 1: Training with frozen VGG16 base...")
    history1 = classifier.train_model(epochs=5, model_name="coriander_vs_parsely_phase1")
    
    # Evaluate initial training
    print("\nPhase 1 Results:")
    results1 = classifier.evaluate_model()
    
    # Fine-tune model (optional but recommended)
    print("\nPhase 2: Fine-tuning with unfrozen top layers...")
    history2 = classifier.fine_tune_model(
        fine_tune_layers=4, 
        fine_tune_epochs=5, 
        fine_tune_lr=1e-5
    )
    
    # Final evaluation
    print("\nFinal Results:")
    final_results = classifier.evaluate_model()
    
    # Plot training history
    classifier.plot_training_history(save_plots=True)
    
    print("\nTraining pipeline completed!")
    print(f"Model saved as: coriander_vs_parsely_vgg16_finetuned.h5")
    print(f"Final accuracy: {final_results['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()