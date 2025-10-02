import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt

def test_single_augmentation():
    """Test augmentation on a single image to verify the fix"""
    
    # Define the same augmentation as in the main script (without rescale)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.85, 1.15],
        channel_shift_range=0.1,
        fill_mode='nearest'
    )
    
    # Test with first coriander image
    test_dir = "train/coriander"
    if not os.path.exists(test_dir):
        print("Test directory not found!")
        return
    
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print("No images found in test directory!")
        return
    
    # Load first image
    image_path = os.path.join(test_dir, image_files[0])
    print(f"Testing with image: {image_path}")
    
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"Original image array range: {img_array.min():.2f} - {img_array.max():.2f}")
    
    # Generate one augmented image
    aug_iter = datagen.flow(img_array, batch_size=1)
    augmented_img = next(aug_iter)[0]
    
    print(f"Augmented image array range: {augmented_img.min():.2f} - {augmented_img.max():.2f}")
    
    # Apply the fix: clip values and convert to uint8
    augmented_img_fixed = np.clip(augmented_img, 0, 255).astype('uint8')
    
    print(f"Fixed image array range: {augmented_img_fixed.min()} - {augmented_img_fixed.max()}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Augmented (before fix)
    axes[1].imshow(augmented_img.astype('uint8'))
    axes[1].set_title('Augmented (Before Fix)')
    axes[1].axis('off')
    
    # Augmented (after fix)
    axes[2].imshow(augmented_img_fixed)
    axes[2].set_title('Augmented (After Fix)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Test completed! Check 'augmentation_test.png' for results.")
    
    # Test saving functionality
    test_output_dir = "test_aug_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Save original
    img.save(os.path.join(test_output_dir, "original.jpg"))
    
    # Save augmented (fixed)
    aug_img_pil = array_to_img(augmented_img_fixed)
    aug_img_pil.save(os.path.join(test_output_dir, "augmented_fixed.jpg"))
    
    print(f"Test images saved in '{test_output_dir}/' directory")

if __name__ == "__main__":
    test_single_augmentation()