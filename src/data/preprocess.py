import os
from text_processing import process_text_data
from url_processing import process_url_data
from image_processing import process_images

def main():
    raw_data_dir = os.path.join("data", "raw")
    processed_data_dir = os.path.join("data", "processed")
    
    # Text Data
    print("--- Text Preprocessing ---")
    posts_in = os.path.join(raw_data_dir, "posts.csv")
    posts_out = os.path.join(processed_data_dir, "posts.csv")
    if os.path.exists(posts_in):
        process_text_data(posts_in, posts_out)
    else:
        print(f"File not found: {posts_in}")
        
    # URL Data
    print("\n--- URL Preprocessing ---")
    urls_in = os.path.join(raw_data_dir, "urls.csv")
    urls_out = os.path.join(processed_data_dir, "urls.csv")
    if os.path.exists(urls_in):
        process_url_data(urls_in, urls_out)
    else:
        print(f"File not found: {urls_in}")
        
    # Image Data
    print("\n--- Image Preprocessing ---")
    images_in = os.path.join(raw_data_dir, "images")
    images_out = os.path.join(processed_data_dir, "images")
    if os.path.exists(images_in) and os.path.isdir(images_in):
        process_images(images_in, images_out)
    else:
        print(f"Directory not found: {images_in}")

if __name__ == "__main__":
    main()
