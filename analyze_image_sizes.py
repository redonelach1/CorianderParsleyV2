import os
from PIL import Image
import numpy as np
from collections import Counter

def analyze_image_sizes(directory):
    """
    Analyze image sizes in a directory and its subdirectories
    
    Args:
        directory (str): Path to the directory containing images
    """
    
    sizes = []
    file_count = 0
    error_count = 0
    
    print(f"Analyzing images in: {directory}")
    print("="*50)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        class_name = os.path.basename(root)
        if class_name == os.path.basename(directory):
            continue  # Skip the root directory itself
            
        print(f"\nClass: {class_name}")
        print("-" * 30)
        
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print("No image files found.")
            continue
        
        class_sizes = []
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(root, image_file)
            
            try:
                with Image.open(image_path) as img:
                    size = img.size  # (width, height)
                    sizes.append(size)
                    class_sizes.append(size)
                    file_count += 1
                    
                    # Show progress for every 50 images
                    if (i + 1) % 50 == 0:
                        print(f"  Processed {i + 1}/{len(image_files)} images...")
                        
            except Exception as e:
                error_count += 1
                print(f"  Error reading {image_file}: {str(e)}")
        
        if class_sizes:
            # Analyze sizes for this class
            unique_sizes = list(set(class_sizes))
            size_counts = Counter(class_sizes)
            
            print(f"  Total images: {len(class_sizes)}")
            print(f"  Unique sizes: {len(unique_sizes)}")
            
            if len(unique_sizes) <= 5:
                print("  All sizes found:")
                for size, count in size_counts.most_common():
                    print(f"    {size[0]}x{size[1]}: {count} images")
            else:
                print("  Most common sizes:")
                for size, count in size_counts.most_common(5):
                    print(f"    {size[0]}x{size[1]}: {count} images")
            
            # Calculate statistics
            widths = [s[0] for s in class_sizes]
            heights = [s[1] for s in class_sizes]
            
            print(f"  Width  - Min: {min(widths):4d}, Max: {max(widths):4d}, Avg: {np.mean(widths):.1f}")
            print(f"  Height - Min: {min(heights):4d}, Max: {max(heights):4d}, Avg: {np.mean(heights):.1f}")
    
    # Overall statistics
    print("\n" + "="*50)
    print("OVERALL STATISTICS")
    print("="*50)
    print(f"Total images processed: {file_count}")
    print(f"Images with errors: {error_count}")
    
    if sizes:
        # Most common sizes across all classes
        size_counts = Counter(sizes)
        print(f"Total unique sizes: {len(set(sizes))}")
        
        print("\nMost common sizes across all classes:")
        for size, count in size_counts.most_common(10):
            print(f"  {size[0]}x{size[1]}: {count} images ({count/len(sizes)*100:.1f}%)")
        
        # Overall statistics
        all_widths = [s[0] for s in sizes]
        all_heights = [s[1] for s in sizes]
        
        print(f"\nOverall dimensions:")
        print(f"  Width  - Min: {min(all_widths):4d}, Max: {max(all_widths):4d}, Avg: {np.mean(all_widths):.1f}")
        print(f"  Height - Min: {min(all_heights):4d}, Max: {max(all_heights):4d}, Avg: {np.mean(all_heights):.1f}")
        
        # Recommendations
        print(f"\n📊 RECOMMENDATIONS:")
        most_common_size = size_counts.most_common(1)[0]
        print(f"🎯 Most common size: {most_common_size[0][0]}x{most_common_size[0][1]} ({most_common_size[1]} images)")
        
        # Calculate median size
        median_width = int(np.median(all_widths))
        median_height = int(np.median(all_heights))
        print(f"📐 Median size: {median_width}x{median_height}")
        
        # Suggest good target sizes
        print(f"\n💡 Suggested target sizes for training:")
        if most_common_size[1] / len(sizes) > 0.5:  # If more than 50% have same size
            print(f"   - Use original size: {most_common_size[0][0]}x{most_common_size[0][1]}")
        else:
            print(f"   - Conservative: {min(median_width, median_height)}x{min(median_width, median_height)} (square)")
            print(f"   - Balanced: {median_width}x{median_height}")
            print(f"   - Standard ML sizes: 224x224, 150x150, or 128x128")


if __name__ == "__main__":
    train_directory = "train"
    
    if not os.path.exists(train_directory):
        print(f"❌ Directory '{train_directory}' not found!")
        print("Please make sure you're in the correct directory.")
    else:
        analyze_image_sizes(train_directory)