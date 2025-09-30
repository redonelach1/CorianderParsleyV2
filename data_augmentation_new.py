import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from pathlib import Path
import matplotlib.pyplot as plt

def create_augmented_dataset(input_dir, output_dir, augmentations_per_image=5, target_size=(150, 150)):
    """
    Create augmented dataset from original images
    
    Args:
        input_dir (str): Path to directory containing class folders (coriander, parsely)
        output_dir (str): Path to output directory for augmented images
        augmentations_per_image (int): Number of augmented versions per original image
        target_size (tuple): Target size for images (width, height)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define data augmentation parameters optimized for VGG16
    datagen = ImageDataGenerator(
        rotation_range=20,           # Moderate rotation (reduced from 40 for VGG16)
        width_shift_range=0.15,      # Slight horizontal shift (reduced for stability)
        height_shift_range=0.15,     # Slight vertical shift (reduced for stability)
        shear_range=0.1,            # Light shear transformation (reduced)
        zoom_range=0.15,            # Moderate zoom (reduced from 0.2)
        horizontal_flip=True,        # Horizontal flip (good for leaf images)
        vertical_flip=False,         # Disabled - leaves don't naturally appear upside down
        brightness_range=[0.85, 1.15], # Subtle brightness adjustment (reduced range)
        channel_shift_range=0.1,     # Add channel shift for color variation
        fill_mode='nearest'          # Fill mode for transformed pixels
        # Note: rescale removed to prevent double normalization when saving images
    )
    
    # Get class folders (coriander, parsely)
    class_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    print(f"Found classes: {class_folders}")
    
    total_augmented = 0
    
    for class_name in class_folders:
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        
        # Create output class directory
        os.makedirs(class_output_path, exist_ok=True)
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        class_augmented = 0
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(class_input_path, image_file)
            
            try:
                # Load and preprocess image
                img = load_img(image_path, target_size=target_size)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Generate augmented images
                aug_iter = datagen.flow(img_array, batch_size=1)
                
                # Save original image first
                original_name = f"original_{i:04d}_{os.path.splitext(image_file)[0]}.jpg"
                original_path = os.path.join(class_output_path, original_name)
                img.save(original_path)
                
                # Generate and save augmented versions
                for j in range(augmentations_per_image):
                    augmented_img = next(aug_iter)[0]
                    
                    # Ensure pixel values are in correct range [0, 255]
                    augmented_img = np.clip(augmented_img, 0, 255).astype('uint8')
                    aug_img_pil = array_to_img(augmented_img)
                    
                    aug_name = f"aug_{i:04d}_{j:02d}_{os.path.splitext(image_file)[0]}.jpg"
                    aug_path = os.path.join(class_output_path, aug_name)
                    aug_img_pil.save(aug_path)
                    
                    class_augmented += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                print(f"  Error processing {image_file}: {str(e)}")
        
        print(f"  Generated {class_augmented} augmented images for {class_name}")
        total_augmented += class_augmented
    
    print(f"\n✅ Data augmentation complete!")
    print(f"📊 Total augmented images generated: {total_augmented}")
    print(f"📁 Augmented dataset saved in: {output_dir}")


def visualize_augmentations(input_dir, class_name, image_index=0, num_augmentations=6):
    """
    Visualize original image and its augmented versions
    
    Args:
        input_dir (str): Path to directory containing class folders
        class_name (str): Name of the class to visualize
        image_index (int): Index of the image to use for visualization
        num_augmentations (int): Number of augmented versions to show
    """
    
    # Define augmentation (same as main function for consistency)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,         # Leaves don't naturally appear upside down
        brightness_range=[0.85, 1.15],
        channel_shift_range=0.1,
        fill_mode='nearest'
        # Note: rescale removed to prevent visualization issues
    )
    
    # Get image path
    class_path = os.path.join(input_dir, class_name)
    image_files = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if image_index >= len(image_files):
        print(f"Image index {image_index} not available. Max index: {len(image_files)-1}")
        return
    
    image_path = os.path.join(class_path, image_files[image_index])
    
    # Load image with VGG16 standard size
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle(f'Original and Augmented Images - {class_name}', fontsize=16)
    
    # Show original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Generate and show augmented images
    aug_iter = datagen.flow(img_array, batch_size=1)
    
    positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    
    for i in range(min(num_augmentations + 1, 7)):  # +1 for original, max 7 more
        if i < len(positions):
            row, col = positions[i]
            if i == 0:
                continue  # Skip first position as it's used for original
            
            augmented_img = next(aug_iter)[0]
            # Ensure pixel values are in correct range for display
            augmented_img = np.clip(augmented_img, 0, 255).astype('uint8')
            axes[row, col].imshow(augmented_img)
            axes[row, col].set_title(f'Augmented {i}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "train"  # Directory containing coriander and parsely folders
    OUTPUT_DIR = "augmented_dataset"  # Output directory for augmented images
    AUGMENTATIONS_PER_IMAGE = 5  # Number of augmented versions per original image
    
    # VGG16 optimal settings (original images are 640x640):
    TARGET_SIZE = (224, 224)  # VGG16 standard input size - optimal for transfer learning
    # Alternative sizes:
    # TARGET_SIZE = (640, 640)  # Original size - maximum quality, high memory usage
    # TARGET_SIZE = (150, 150)  # Smaller size for faster training
    # TARGET_SIZE = (128, 128)  # Fastest training, lower memory
    
    print("🚀 Starting VGG16-Optimized Data Augmentation Process")
    print("="*55)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found!")
        print("Please make sure you're running this script from the correct directory.")
        exit(1)
    
    # Display current configuration
    print(f"📁 Input directory: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🔄 Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    print(f"📐 Target image size: {TARGET_SIZE} (VGG16 standard)")
    print(f"🧠 Optimized for: VGG16 Transfer Learning")
    print(f"🔢 Normalization: Applied during model training")
    print("="*55)
    
    # Perform data augmentation
    create_augmented_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        augmentations_per_image=AUGMENTATIONS_PER_IMAGE,
        target_size=TARGET_SIZE
    )
    
    # Optional: Visualize some augmentations
    print("\n🖼️  Would you like to see a visualization of augmentations?")
    print("Uncomment the lines below to visualize:")
    visualize_augmentations('train', 'coriander', image_index=0)
    visualize_augmentations('train', 'parsely', image_index=0)