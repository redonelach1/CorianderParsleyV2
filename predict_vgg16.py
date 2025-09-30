import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt

class CorianderParsleyPredictor:
    def __init__(self, model_path="coriander_vs_parsely_vgg16_finetuned.h5", img_size=(224, 224)):
        """
        Initialize the predictor with trained model
        
        Args:
            model_path (str): Path to the trained model file
            img_size (tuple): Input image size that model expects
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = ["coriander", "parsely"]  # 0: coriander, 1: parsely
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = load_model(self.model_path)
            print(f"✅ Model loaded from: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure you have trained the model first!")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.array: Preprocessed image array
        """
        try:
            # Load and resize image
            img = load_img(image_path, target_size=self.img_size)
            
            # Convert to array and normalize
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, img
        
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            return None, None
    
    def predict_single_image(self, image_path, show_confidence=True):
        """
        Predict class for a single image
        
        Args:
            image_path (str): Path to the image file
            show_confidence (bool): Whether to show confidence score
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            return None
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # Determine class and confidence
        if prediction >= 0.5:
            predicted_class = "parsely"
            confidence = prediction
        else:
            predicted_class = "coriander"
            confidence = 1 - prediction
        
        results = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'raw_prediction': prediction,
            'original_image': original_img
        }
        
        if show_confidence:
            print(f"📷 {os.path.basename(image_path)}")
            print(f"🔮 Prediction: {predicted_class.upper()}")
            print(f"📊 Confidence: {confidence*100:.2f}%")
            print("-" * 40)
        
        return results
    
    def predict_batch(self, image_folder, output_folder="predictions_output"):
        """
        Predict classes for all images in a folder
        
        Args:
            image_folder (str): Path to folder containing images
            output_folder (str): Path to save annotated images
        """
        if not os.path.exists(image_folder):
            print(f"❌ Image folder not found: {image_folder}")
            return
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print("❌ No image files found in the folder!")
            return
        
        print(f"🔍 Found {len(image_files)} images to process...")
        print("="*50)
        
        results = []
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            
            # Make prediction
            result = self.predict_single_image(image_path, show_confidence=True)
            
            if result:
                results.append(result)
                
                # Save annotated image
                self.save_annotated_image(result, output_folder)
        
        # Summary
        self.print_batch_summary(results)
        
        return results
    
    def save_annotated_image(self, result, output_folder):
        """Save image with prediction annotation"""
        try:
            # Load original image
            original_img = result['original_image']
            
            # Create a copy for annotation
            img_with_text = original_img.copy()
            draw = ImageDraw.Draw(img_with_text)
            
            # Prepare text
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            text = f"{predicted_class.upper()}\n{confidence*100:.1f}%"
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position (bottom of image)
            img_width, img_height = img_with_text.size
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text at bottom center
            x = (img_width - text_width) // 2
            y = img_height - text_height - 20
            
            # Draw background rectangle
            rectangle_coords = [x-10, y-5, x+text_width+10, y+text_height+5]
            draw.rectangle(rectangle_coords, fill=(0, 0, 0, 128))
            
            # Draw text
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            # Save annotated image
            output_filename = f"pred_{os.path.basename(result['image_path'])}"
            output_path = os.path.join(output_folder, output_filename)
            img_with_text.save(output_path)
            
        except Exception as e:
            print(f"⚠️ Could not save annotated image: {e}")
    
    def print_batch_summary(self, results):
        """Print summary of batch predictions"""
        if not results:
            return
        
        print("\n" + "="*50)
        print("📊 PREDICTION SUMMARY")
        print("="*50)
        
        # Count predictions by class
        coriander_count = sum(1 for r in results if r['predicted_class'] == 'coriander')
        parsely_count = sum(1 for r in results if r['predicted_class'] == 'parsely')
        
        print(f"🌿 Coriander: {coriander_count} images")
        print(f"🍃 Parsely: {parsely_count} images")
        print(f"📊 Total: {len(results)} images")
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"📈 Average confidence: {avg_confidence*100:.2f}%")
        
        # High confidence predictions (>90%)
        high_conf = sum(1 for r in results if r['confidence'] > 0.9)
        print(f"🎯 High confidence (>90%): {high_conf}/{len(results)} ({high_conf/len(results)*100:.1f}%)")
    
    def visualize_predictions(self, results, max_images=8):
        """Visualize prediction results in a grid"""
        if not results:
            print("No results to visualize")
            return
        
        # Limit number of images to display
        display_results = results[:max_images]
        
        # Calculate grid size
        n_images = len(display_results)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(display_results):
            if i >= len(axes):
                break
                
            # Load and display image
            img = result['original_image']
            axes[i].imshow(img)
            
            # Set title with prediction
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            title = f"{predicted_class.upper()}\n{confidence*100:.1f}%"
            
            # Color based on confidence
            color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'red'
            axes[i].set_title(title, fontsize=12, color=color, weight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(display_results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main prediction pipeline"""
    
    print("🔮 Coriander vs Parsely Predictor")
    print("="*40)
    
    # Initialize predictor
    predictor = CorianderParsleyPredictor()
    
    # Check if model exists
    if predictor.model is None:
        print("\n❌ No trained model found!")
        print("Please train the model first using vgg16_training.py")
        return
    
    # Example usage
    test_folder = "test"  # Change this to your test images folder
    
    if os.path.exists(test_folder):
        print(f"\n🔍 Processing images in: {test_folder}")
        results = predictor.predict_batch(test_folder)
        
        # Visualize some results
        if results:
            print(f"\n🖼️ Visualizing first {min(8, len(results))} predictions...")
            predictor.visualize_predictions(results)
    
    else:
        print(f"\n⚠️ Test folder '{test_folder}' not found.")
        print("Please create a test folder with images or modify the test_folder path.")
        
        # Example single prediction
        print("\n💡 Example usage for single image:")
        print("predictor.predict_single_image('path/to/your/image.jpg')")


if __name__ == "__main__":
    main()