import numpy as np
from PIL import Image, ImageChops
import io
import base64

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_grayscale_images(num_images=16):
    """Generate random 4x4 grayscale images and return as base64"""
    images = []
    for i in range(num_images):
        img_array = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_base64 = image_to_base64(img)
        images.append({
            'id': i,
            'data': img_base64
        })
    return images

def calculate_rmse(img1, img2):
    """Calculate RMSE between two images"""
    diff = ImageChops.difference(img1, img2)
    diff_array = np.array(diff)
    mse = np.mean(diff_array ** 2)
    rmse = np.sqrt(mse)
    return rmse

def compare_image_pairs(images):
    """Compare random pairs of images using PIL"""
    # Convert base64 images back to PIL Images
    pil_images = []
    for img in images:
        img_data = base64.b64decode(img['data'])
        img_io = io.BytesIO(img_data)
        pil_img = Image.open(img_io)
        pil_images.append((img['id'], pil_img))
    
    # Randomly pair images
    np.random.shuffle(pil_images)
    pairs = [(pil_images[i], pil_images[i+1]) for i in range(0, len(pil_images), 2)]
    
    results = []
    for (id1, img1), (id2, img2) in pairs:
        # Calculate RMSE
        rmse = calculate_rmse(img1, img2)
        
        # Generate difference image with graduated red tint
        diff = ImageChops.difference(img1, img2)
        diff_array = np.array(diff, dtype=float)
        
        # Normalize difference to 0-1 range
        diff_normalized = diff_array / 255.0
        
        # Create RGB image with graduated red tint
        rgb_diff = np.zeros((diff_array.shape[0], diff_array.shape[1], 3), dtype=np.uint8)
        
        # Red channel: lighter (255) for small diffs, darker (128) for large diffs
        rgb_diff[:, :, 0] = np.uint8(255 - (diff_normalized * 127))  # Red ranges from 255 to 128
        
        # Green/Blue channels: white (255) for no diff, transparent (0) for full diff
        rgb_diff[:, :, 1] = np.uint8(255 * (1 - diff_normalized))  # Green fades to 0
        rgb_diff[:, :, 2] = np.uint8(255 * (1 - diff_normalized))  # Blue fades to 0
        
        diff_image = Image.fromarray(rgb_diff)
        diff_base64 = image_to_base64(diff_image)
        
        results.append({
            'image1_id': id1,
            'image2_id': id2,
            'image1': next(img['data'] for img in images if img['id'] == id1),
            'image2': next(img['data'] for img in images if img['id'] == id2),
            'diff_image': diff_base64,
            'rmse': float(rmse)
        })
    
    return results

if __name__ == "__main__":
    # Generate images
    image_files = generate_grayscale_images()
    
    # Compare random pairs
    results = compare_image_pairs(image_files)
    
    # Save results to JSON file
    with open('static/comparison_results.json', 'w') as f:
        json.dump(results, f)
