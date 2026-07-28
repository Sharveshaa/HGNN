import os
import shutil
from PIL import Image
from tqdm import tqdm

def process_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    corrupt_count = 0
    processed_count = 0
    
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Processing {len(files)} images...")
    
    for filename in tqdm(files, desc="Resizing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(input_path) as img:
                # Convert to RGB if necessary (e.g., RGBA or Grayscale)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                img_resized.save(output_path, format='JPEG', quality=95)
                processed_count += 1
        except Exception as e:
            # print(f"Corrupt or invalid image: {filename} - {e}")
            corrupt_count += 1
            
    print(f"Successfully processed and resized {processed_count} images.")
    print(f"Skipped {corrupt_count} corrupt or invalid images.")
